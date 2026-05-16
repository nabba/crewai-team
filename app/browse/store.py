"""JSONL store for canonical browse events + per-browser cursor state.

Layout under ``workspace/browse/``::

    events/YYYY-MM-DD.jsonl     append-only, one BrowseEvent per line
    state.json                  {browser:profile -> last_cursor_us}
    blocklist.txt               operator-managed (see app.browse.blocklist)

The per-day file shape is deliberate — it makes the LLM-batch step
(Phase B) trivially "yesterday's events.jsonl" and lets the operator
forget a specific day with a single unlink. Old days are never
auto-deleted; an operator command (``forget_all`` or future retention
monitor) is the only path to deletion.

All file I/O is failure-isolated. ``BROWSE_INGESTION_ENABLED`` defaults
to ``false`` and short-circuits :func:`append_events` on disabled.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from app.browse.models import BrowseEvent
from app.paths import WORKSPACE_ROOT

logger = logging.getLogger(__name__)


_DEFAULT_BASE = WORKSPACE_ROOT / "browse"
_path_override: Path | None = None


def _emit_policy_event(summary: str, detail: dict[str, Any]) -> None:
    """Best-effort identity-continuity ledger emission for a browse
    policy change. Failure-isolated — never blocks the caller."""
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="browse_ingestion_policy",
            actor="operator",
            summary=summary,
            detail=detail,
        )
    except Exception:
        logger.debug("browse.store: ledger emit failed", exc_info=True)


def enabled() -> bool:
    """Master switch. Default OFF — operator must explicitly flip the
    env var to opt in."""
    return os.getenv("BROWSE_INGESTION_ENABLED", "false").lower() in (
        "true", "1", "yes", "on",
    )


def resolve_base() -> Path:
    """Resolved base directory. Honours :func:`_reset_for_tests` first,
    then ``BROWSE_BASE_DIR`` env var, then default."""
    if _path_override:
        return _path_override
    raw = os.getenv("BROWSE_BASE_DIR")
    if raw:
        return Path(raw)
    return _DEFAULT_BASE


def _events_dir(base: Path | None = None) -> Path:
    root = base if base else resolve_base()
    return root / "events"


def _events_path_for(day: date, base: Path | None = None) -> Path:
    return _events_dir(base=base) / f"{day.isoformat()}.jsonl"


def _state_path(base: Path | None = None) -> Path:
    root = base if base else resolve_base()
    return root / "state.json"


# ── Append ────────────────────────────────────────────────────────────


def append_events(
    events: Iterable[BrowseEvent],
    *,
    base: Path | None = None,
) -> int:
    """Append events to per-day JSONL. Returns number written.

    Disabled short-circuit: when ``BROWSE_INGESTION_ENABLED`` is off,
    returns 0 without touching disk. This is the structural guarantee
    that flipping the master switch off stops persistence.
    """
    if not enabled():
        return 0
    written = 0
    # Group by day so each day's file is opened once per append batch.
    by_day: dict[date, list[BrowseEvent]] = {}
    for e in events:
        try:
            d = datetime.fromisoformat(e.visit_ts).date()
        except ValueError:
            logger.debug("browse.store: skipping bad ts %r", e.visit_ts)
            continue
        by_day.setdefault(d, []).append(e)
    for day, day_events in by_day.items():
        p = _events_path_for(day, base=base)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with p.open("a", encoding="utf-8") as f:
                for ev in day_events:
                    f.write(json.dumps(ev.to_dict(), sort_keys=True) + "\n")
                    written += 1
        except OSError as exc:
            logger.warning("browse.store: append to %s failed: %s", p, exc)
    return written


# ── Read ──────────────────────────────────────────────────────────────


def list_events_for_day(
    day: date,
    *,
    base: Path | None = None,
) -> list[BrowseEvent]:
    """Return all events stored for one day. Failure-isolated."""
    p = _events_path_for(day, base=base)
    if not p.exists():
        return []
    out: list[BrowseEvent] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    out.append(BrowseEvent.from_dict(raw))
                except (KeyError, TypeError):
                    continue
    except OSError:
        return []
    return out


def list_events_window(
    *,
    days: int,
    now: datetime | None = None,
    base: Path | None = None,
) -> list[BrowseEvent]:
    """All events in the last ``days`` (inclusive of today)."""
    cur = (now or datetime.now(timezone.utc)).date()
    out: list[BrowseEvent] = []
    for i in range(days):
        d = cur - timedelta(days=i)
        out.extend(list_events_for_day(d, base=base))
    return out


# ── Cursor state ──────────────────────────────────────────────────────


def _state_key(browser: str, profile: str | None) -> str:
    return f"{browser}:{profile}" if profile else browser


def load_cursors(*, base: Path | None = None) -> dict[str, int]:
    """Read the per-browser cursor map.

    Returns an empty dict on missing/corrupt file (caller restarts
    cursors from zero — first pass will look like a wider read, but
    dedup at the event-id layer keeps the JSONL clean)."""
    p = _state_path(base=base)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
    return out


def save_cursor(
    browser: str,
    profile: str | None,
    cursor: int,
    *,
    base: Path | None = None,
) -> None:
    """Update the per-browser cursor. Failure-isolated."""
    cursors = load_cursors(base=base)
    cursors[_state_key(browser, profile)] = int(cursor)
    p = _state_path(base=base)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(cursors, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except OSError as exc:
        logger.warning("browse.store: cursor write failed: %s", exc)


def get_cursor(
    browser: str,
    profile: str | None,
    *,
    base: Path | None = None,
) -> int:
    """Return last cursor for one source, or 0 if unset."""
    return load_cursors(base=base).get(_state_key(browser, profile), 0)


# ── Forget paths ──────────────────────────────────────────────────────


def forget_all(*, base: Path | None = None) -> int:
    """Delete every events file + cursor + day rollup.

    The blocklist.txt is preserved (operator's mute history is durable
    independent of event history).

    Returns the number of files removed."""
    root = base if base else resolve_base()
    removed = 0
    ev_dir = _events_dir(base=base)
    if ev_dir.exists():
        for f in ev_dir.glob("*.jsonl"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                continue
    state = _state_path(base=base)
    if state.exists():
        try:
            state.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        _emit_policy_event(
            "browse history forgotten (all)",
            {"action": "forget_all", "files_removed": removed},
        )
    return removed


def forget_day(day: date, *, base: Path | None = None) -> bool:
    """Delete one day's events file. Returns ``True`` when something
    was removed."""
    p = _events_path_for(day, base=base)
    if not p.exists():
        return False
    try:
        p.unlink()
        _emit_policy_event(
            f"browse history forgotten for {day.isoformat()}",
            {"action": "forget_day", "day": day.isoformat()},
        )
        return True
    except OSError:
        return False


def forget_domain(domain: str, *, base: Path | None = None) -> int:
    """Remove every event for ``domain`` from the live store. Returns
    rows removed. Walks each day's file and rewrites without the
    matching entries.

    Pairs with :func:`app.browse.blocklist.mute_domain` — the typical
    operator flow is ``mute_domain(d)`` (stops future writes) plus
    ``forget_domain(d)`` (clears history)."""
    if not domain.strip():
        return 0
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    ev_dir = _events_dir(base=base)
    if not ev_dir.exists():
        return 0
    removed = 0
    for path in sorted(ev_dir.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        kept: list[str] = []
        for line in lines:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            d = str(raw.get("domain", "")).lower()
            if d == domain or d.endswith("." + domain):
                removed += 1
                continue
            kept.append(line)
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8",
            )
            tmp.replace(path)
        except OSError:
            continue
    if removed:
        _emit_policy_event(
            f"browse history forgotten for domain {domain}",
            {"action": "forget_domain", "domain": domain, "rows_removed": removed},
        )
    return removed


# ── Stats (operator visibility) ──────────────────────────────────────


def event_counts(*, days: int = 7, base: Path | None = None) -> dict[str, Any]:
    """Lightweight stats for the React settings card.

    Returns ``{total, by_browser, by_domain_top_n}``."""
    events = list_events_window(days=days, base=base)
    by_browser: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for e in events:
        by_browser[e.browser] = by_browser.get(e.browser, 0) + 1
        by_domain[e.domain] = by_domain.get(e.domain, 0) + 1
    top_n = sorted(by_domain.items(), key=lambda kv: kv[1], reverse=True)[:25]
    return {
        "total": len(events),
        "window_days": days,
        "by_browser": by_browser,
        "by_domain_top": [{"domain": d, "count": c} for d, c in top_n],
    }


# ── Test seam ─────────────────────────────────────────────────────────


def _reset_for_tests(path: Path | None = None) -> None:
    """Test-only seam: pin the base directory."""
    global _path_override
    _path_override = path
    from app.browse import blocklist
    blocklist.reset_cache()
