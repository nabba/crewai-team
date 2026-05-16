"""wiki_staleness — knowledge-decay surface for the wiki tree.

PROGRAM §51 — Q16 Theme 5 (decade-resilience hardening, knowledge
management at decade-scale). The wiki accumulates pages over the
years. Many of them say things like "current tech stack" or
"Andrus's role at PLG" that decay over time. The wiki-index
reconciler keeps the INDEX honest about structure; this monitor
keeps the CONTENT honest about freshness.

What this monitor does:

  1. Walks ``wiki/`` looking for markdown files.
  2. For each, computes ``age_days`` = (now - mtime) / 86400.
  3. Stale = age > ``_STALE_THRESHOLD_DAYS`` (default 365).
  4. Persists a per-file ``last_surfaced_at`` so we don't re-alert
     the same files every week.
  5. Once per week, surfaces the N oldest stale files in a digest
     Signal alert (operator decides what to do — refresh, archive,
     or mark "intentionally historical").
  6. Failure-isolated: missing wiki dir → silent skip.

What this monitor **deliberately doesn't** do:

  * Auto-edit any wiki page. The wiki is the operator's
    narrative; refresh decisions are operator-only.
  * File CRs. The wiki-index reconciler already does that for
    structural reconciliation; content freshness is a softer
    signal and shouldn't be CR-volume.
  * Use mtime as gospel. Operators sometimes touch files in
    workspace syncs; mtime is a freshness PROXY. The operator
    is the source of truth.

Cadence: daily probe; internal weekly cadence for emission.
Master switch: ``wiki_staleness_monitor_enabled`` (default ON).
Alert dedup: per-file 90 days (each page surfaces at most quarterly).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


NAME = "wiki_staleness"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "wiki_staleness_monitor_enabled"

_INTERNAL_CADENCE_S = 7 * 24 * 3600
_DEDUP_WINDOW_S = 90 * 86400  # per-file
_STATE_FILE_NAME = "wiki_staleness_state.json"

_STALE_THRESHOLD_DAYS = 365
_DIGEST_MAX_FILES = 10  # surface at most 10 stalest files per digest
_MAX_TRACKED_FILES = 5000

# Categorical exclusions — these subdirectories are auto-generated or
# historical-only by intent and should NEVER alert.
_EXCLUDE_DIR_PREFIXES = (
    "wiki/self/legacy/",          # annual legacy essays, archive-by-design
    "wiki/self/value_reflections/",  # annual reflections, archive-by-design
    "wiki/self/quarterly_reviews/",  # quarterly reviews, archive-by-design
    "wiki/governance/",           # rare-edit governance docs
    "wiki/archive/",              # explicit archive dir
)


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_wiki_staleness_monitor_enabled
        return get_wiki_staleness_monitor_enabled()
    except Exception:
        return os.getenv(
            "WIKI_STALENESS_MONITOR_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _repo_root() -> Path:
    """The repo root (where ``wiki/`` lives, alongside ``app/``)."""
    # ``app/healing/monitors/wiki_staleness.py`` → parents[3] is repo root.
    return Path(__file__).resolve().parents[3]


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _wiki_root() -> Path:
    return _repo_root() / "wiki"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_surfaced_at": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_surfaced_at": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug(
            "wiki_staleness: state write failed", exc_info=True,
        )


def _is_excluded(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    for prefix in _EXCLUDE_DIR_PREFIXES:
        if norm.startswith(prefix):
            return True
    return False


def _scan_wiki(now: float) -> list[dict[str, Any]]:
    """Walk wiki/ and return list of ``{rel_path, age_days, mtime}``
    rows for markdown files past the staleness threshold."""
    root = _wiki_root()
    if not root.exists() or not root.is_dir():
        return []
    cutoff_age_s = _STALE_THRESHOLD_DAYS * 86400
    stale: list[dict[str, Any]] = []
    n_seen = 0
    try:
        for path in root.rglob("*.md"):
            n_seen += 1
            if n_seen > _MAX_TRACKED_FILES:
                break
            try:
                rel = str(path.relative_to(_repo_root()))
            except ValueError:
                continue
            if _is_excluded(rel):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            age_s = max(0.0, now - mtime)
            if age_s < cutoff_age_s:
                continue
            stale.append({
                "rel_path": rel,
                "age_days": round(age_s / 86400, 1),
                "mtime": mtime,
            })
    except Exception:
        logger.debug("wiki_staleness: scan failed", exc_info=True)
    # Stalest first.
    stale.sort(key=lambda r: r["age_days"], reverse=True)
    return stale


def _digest_due(
    stale_rows: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    now: float,
) -> list[dict[str, Any]]:
    """Pick the rows that haven't been surfaced in the dedup window.
    Returns at most ``_DIGEST_MAX_FILES`` rows, stalest first."""
    surfaced = state.setdefault("last_surfaced_at", {})
    if not isinstance(surfaced, dict):
        surfaced = {}
        state["last_surfaced_at"] = surfaced
    due: list[dict[str, Any]] = []
    for row in stale_rows:
        rel = row["rel_path"]
        last = float(surfaced.get(rel, 0))
        if now - last < _DEDUP_WINDOW_S:
            continue
        due.append(row)
        if len(due) >= _DIGEST_MAX_FILES:
            break
    return due


def _send_digest(rows: list[dict[str, Any]]) -> bool:
    """Signal digest naming the N stalest pages. Topic-keyed for arbiter
    dedup. Returns True on send."""
    try:
        from app.notify import notify
        lines = [
            f"📚 Wiki staleness — {len(rows)} page(s) past "
            f"{_STALE_THRESHOLD_DAYS}-day freshness threshold:",
            "",
        ]
        for r in rows:
            lines.append(
                f"  • `{r['rel_path']}` — {r['age_days']:.0f} days old"
            )
        lines.extend([
            "",
            "Each page surfaces at most once per 90 days. If a page is",
            "intentionally historical, no action needed (it'll keep",
            "aging silently). For everything else, consider one of:",
            "  • refresh: `touch` after a review pass",
            "  • archive: move under `wiki/archive/` (excluded from probe)",
            "  • prune: remove if no longer relevant",
        ])
        notify(
            title="📚 Wiki freshness digest",
            body="\n".join(lines),
            url="/cp/wiki",
            topic="wiki_staleness:digest",
            critical=False,
            arbitrate=True,
        )
        return True
    except Exception:
        logger.debug(
            "wiki_staleness: notify failed", exc_info=True,
        )
        return False


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """One probe pass. Daily wake-up gates on weekly internal
    cadence. Returns a summary dict."""
    summary: dict[str, Any] = {
        "ran": False,
        "wiki_present": False,
        "n_stale": 0,
        "n_digest_due": 0,
        "alert_sent": False,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last = float(state.get("last_run_at", 0))
    if last > 0 and cur - last < _INTERNAL_CADENCE_S:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    if not _wiki_root().exists():
        _write_state(state)
        return summary
    summary["wiki_present"] = True

    stale = _scan_wiki(cur)
    summary["n_stale"] = len(stale)
    if not stale:
        _write_state(state)
        return summary

    due = _digest_due(stale, state, now=cur)
    summary["n_digest_due"] = len(due)
    if due:
        sent = _send_digest(due)
        summary["alert_sent"] = sent
        # Record surfacing.
        surfaced = state.setdefault("last_surfaced_at", {})
        for r in due:
            surfaced[r["rel_path"]] = cur

    _write_state(state)
    return summary
