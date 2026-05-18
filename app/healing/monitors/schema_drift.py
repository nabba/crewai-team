"""schema_drift.py — Postgres schema vs ``migrations/*.sql`` consistency probe.

Closes the visibility gap behind the deliberate "no auto-apply for
migrations" safety policy: ``change_requests/validator.py:248`` forbids
auto-applying anything under ``migrations/`` — schema changes need
human review. Without this monitor, an unapplied migration only
surfaces when production code raises ``column X does not exist`` at
request time (the audit's 2026-05-18 ``pch_layer`` finding).

What it detects:

* Tables declared in ``CREATE TABLE`` statements that are missing
  from ``information_schema.tables``.
* Columns declared in ``ALTER TABLE … ADD COLUMN`` statements that
  are missing from ``information_schema.columns``.

What it does NOT do:

* Apply migrations (forbidden by policy).
* Detect drift in the other direction (DB has columns the migrations
  don't declare — these are usually older code paths or operator
  hand-edits, both legitimate).
* Track DROP / RENAME / complex CHECK constraints.

On drift:

1. Writes a markdown report to ``docs/proposed_fixes/
   schema_drift_<migration_basename>.md`` describing the gap and the
   manual apply command.
2. Fires a Signal alert per migration (deduped via state file, refires
   after 30 days if drift persists).
3. Records to ``workspace/healing/schema_drift_state.json``.

On resolution (operator applied the migration):

* Probe finds no drift, state entry + report file are cleared.

Master switch: ``schema_drift_monitor_enabled`` (default ON, read from
``runtime_settings`` at call time). Cadence: daily probe; internal
weekly cadence — schema drift doesn't appear faster than migrations
are committed.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Permissive regex covering the SQL flavours used in migrations/*.sql.
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.]+)\s*\(",
    re.IGNORECASE,
)
_ALTER_TABLE_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?([\w.]+)\b",
    re.IGNORECASE,
)
_ADD_COLUMN_RE = re.compile(
    r"ADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\b",
    re.IGNORECASE,
)

_INTERNAL_CADENCE_S = 7 * 24 * 3600    # weekly
_ALERT_REFIRE_S = 30 * 24 * 3600       # 30 days
_MAX_ALERTS_PER_RUN = 3                 # belt against alert storms


def _gate() -> bool:
    """Master switch. Fail-OPEN to True if runtime_settings is unavailable."""
    try:
        from app.runtime_settings import get_schema_drift_monitor_enabled
        return get_schema_drift_monitor_enabled()
    except Exception:
        return True


def _migrations_dir() -> Path:
    """Locate migrations/ relative to the source tree (works in container + dev)."""
    for parent in Path(__file__).resolve().parents:
        c = parent / "migrations"
        if c.is_dir() and any(c.glob("*.sql")):
            return c
    return Path("/app/migrations")


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT  # type: ignore
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _proposed_fixes_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs").is_dir():
            d = parent / "docs" / "proposed_fixes"
            d.mkdir(parents=True, exist_ok=True)
            return d
    d = Path("/app/docs/proposed_fixes")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path() -> Path:
    p = _workspace_root() / "healing" / "schema_drift_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state() -> dict:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    try:
        _state_path().write_text(json.dumps(state, indent=2))
    except Exception:
        logger.debug("schema_drift: state save failed", exc_info=True)


def _columns_from_create_body(body: str) -> list[str]:
    """Split a ``CREATE TABLE`` body on top-level commas; first word per part is the column name."""
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if re.match(r"^(CONSTRAINT|PRIMARY|UNIQUE|FOREIGN|CHECK|EXCLUDE)\b", s, re.IGNORECASE):
            continue
        m = re.match(r"^(\w+)\s+\w", s)
        if m:
            out.append(m.group(1).lower())
    return out


def _parse_migration(path: Path) -> dict[str, set[str]]:
    """Return ``{table_name -> set(column_names)}`` declared in this migration."""
    declared: dict[str, set[str]] = {}
    try:
        sql = path.read_text()
    except Exception:
        return {}

    # Strip comments to simplify regex.
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

    # CREATE TABLE — extract balanced parens.
    for m in _CREATE_TABLE_RE.finditer(sql):
        table = m.group(1).lower()
        start = m.end() - 1  # the '('
        depth = 0
        i = start
        body = ""
        while i < len(sql):
            if sql[i] == "(":
                depth += 1
            elif sql[i] == ")":
                depth -= 1
                if depth == 0:
                    body = sql[start + 1:i]
                    break
            i += 1
        if body:
            declared.setdefault(table, set()).update(_columns_from_create_body(body))

    # ALTER TABLE ... ADD COLUMN — single ALTER may add several columns.
    for m in _ALTER_TABLE_RE.finditer(sql):
        table = m.group(1).lower()
        end = sql.find(";", m.end())
        if end < 0:
            end = len(sql)
        body = sql[m.end():end]
        for cm in _ADD_COLUMN_RE.finditer(body):
            declared.setdefault(table, set()).add(cm.group(1).lower())

    return declared


def _live_schema(pg_url: str) -> dict[str, set[str]]:
    """Snapshot of ``information_schema``: ``{schema.table -> set(columns)}``."""
    import psycopg2
    out: dict[str, set[str]] = {}
    conn = psycopg2.connect(pg_url, connect_timeout=10)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_schema || '.' || table_name, column_name
                FROM information_schema.columns
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                """
            )
            for qname, col in cur.fetchall():
                out.setdefault(qname.lower(), set()).add(col.lower())
    finally:
        conn.close()
    return out


def _detect(
    declared: dict[str, set[str]], live: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """One drift entry per (table, kind)."""
    out: list[dict[str, Any]] = []
    for table, cols in declared.items():
        candidates = [table]
        if "." not in table:
            candidates.append(f"public.{table}")
        live_cols = next((live[c] for c in candidates if c in live), None)
        if live_cols is None:
            out.append({
                "table": table,
                "missing_table": True,
                "expected_cols": sorted(cols),
            })
        else:
            missing = sorted(cols - live_cols)
            if missing:
                out.append({
                    "table": table,
                    "missing_table": False,
                    "missing_columns": missing,
                })
    return out


def _write_report(mig_path: Path, drift: list[dict[str, Any]]) -> None:
    report = _proposed_fixes_dir() / f"schema_drift_{mig_path.stem}.md"
    lines = [
        f"# Schema drift: `{mig_path.name}`",
        "",
        f"_Detected: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## What's missing",
        "",
    ]
    for d in drift:
        if d.get("missing_table"):
            lines.append(
                f"- **Missing table** `{d['table']}` — expected columns: "
                f"`{', '.join(d['expected_cols'])}`"
            )
        else:
            lines.append(
                f"- **Missing columns** in `{d['table']}`: "
                f"`{', '.join(d['missing_columns'])}`"
            )
    lines += [
        "",
        "## Apply",
        "",
        "Manual operator action (the change_request validator forbids "
        "auto-applying anything under `migrations/`):",
        "",
        "```sh",
        f"psql -d $DATABASE_URL -f {mig_path}",
        "```",
        "",
        "After applying, the next probe (within 7 days) detects no drift "
        "and this report + the dedup entry are cleared automatically.",
    ]
    report.write_text("\n".join(lines))


def _clear_report(mig_path: Path) -> None:
    p = _proposed_fixes_dir() / f"schema_drift_{mig_path.stem}.md"
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


def run() -> None:
    if not _gate():
        return

    state = _load_state()
    now = time.time()
    if now - state.get("last_run_at", 0) < _INTERNAL_CADENCE_S:
        return
    state["last_run_at"] = now

    try:
        from app.config import get_settings
        pg_url = get_settings().mem0_postgres_url
    except Exception:
        logger.debug("schema_drift: settings unavailable", exc_info=True)
        _save_state(state)
        return
    if not pg_url:
        _save_state(state)
        return

    migrations_dir = _migrations_dir()
    if not migrations_dir.exists():
        _save_state(state)
        return

    try:
        live = _live_schema(pg_url)
    except Exception:
        logger.exception("schema_drift: live schema query failed")
        _save_state(state)
        return

    per_migration: dict = state.setdefault("drifts", {})
    alerts_fired = 0
    drifted_count = 0
    checked_count = 0

    for mig_path in sorted(migrations_dir.glob("*.sql")):
        checked_count += 1
        declared = _parse_migration(mig_path)
        if not declared:
            continue
        drift = _detect(declared, live)
        key = mig_path.name
        if drift:
            drifted_count += 1
            last_alert = per_migration.get(key, {}).get("last_alert_at", 0)
            should_alert = (
                (now - last_alert) > _ALERT_REFIRE_S
                and alerts_fired < _MAX_ALERTS_PER_RUN
            )
            _write_report(mig_path, drift)
            per_migration[key] = {
                "last_detected_at": now,
                "drift_count": len(drift),
                "last_alert_at": now if should_alert else last_alert,
            }
            if should_alert:
                try:
                    from app.notify import notify
                    desc_parts = []
                    for d in drift[:3]:
                        if d.get("missing_table"):
                            desc_parts.append(f"missing table `{d['table']}`")
                        else:
                            desc_parts.append(
                                f"`{d['table']}` missing "
                                f"{len(d['missing_columns'])} col(s)"
                            )
                    notify(
                        title=f"📜 Schema drift: {mig_path.name}",
                        body=(
                            f"{'; '.join(desc_parts)}. "
                            f"See docs/proposed_fixes/schema_drift_"
                            f"{mig_path.stem}.md and apply with psql."
                        ),
                        topic=f"schema_drift:{mig_path.name}",
                    )
                    alerts_fired += 1
                except Exception:
                    logger.debug("schema_drift: notify failed", exc_info=True)
        elif key in per_migration:
            del per_migration[key]
            _clear_report(mig_path)

    _save_state(state)
    logger.info(
        "schema_drift: %d migrations checked, %d with drift, %d alerts fired",
        checked_count, drifted_count, alerts_fired,
    )
