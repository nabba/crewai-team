"""Annual privacy review — yearly enumeration of data sources.

PROGRAM §51 — Q16 Theme 7. As the system collects more operator
data (person correlation, browse, health, inbox, calendar, email,
voice), the cumulative privacy envelope grows quietly. This module
composes a yearly markdown audit that:

  * Lists every data source the system actively reads.
  * Records its current ON/OFF state (from runtime_settings).
  * Names the retention surface for each (workspace path, cap).
  * Cross-references the continuity ledger for any
    ``*_policy`` events in the year (browse policy edits, person
    correlation typed-phrase flips, etc.).
  * Surfaces any new sources added since the last audit (delta).

The output lands at ``wiki/privacy/audit_<year>.md`` for operator
review. The composer is **observational** and **non-LLM** — pure
walks over runtime_settings + ledger + filesystem.

What this module deliberately doesn't do
========================================

  * No LLM rewriting. Structural enumeration only.
  * No auto-disable of any data source. The audit informs;
    decisions are operator-only.
  * No removal of historical audits — `wiki/privacy/audit_*.md`
    accumulates forever (decade-scale visibility).

Master switch: ``annual_privacy_review_enabled`` (default ON).
Cadence: annual (≥330d since last composition for the same year).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_STATE_FILE = "annual_privacy_review_state.json"
_MIN_DAYS_BETWEEN_REVIEWS = 330


# Catalogue of data sources the system reads. Order matters for the
# output. Each entry describes WHAT, FROM WHERE, RETENTION, and the
# RUNTIME-SETTINGS KEY that gates it (if any).
_DATA_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "name": "Signal messages",
        "purpose": "primary operator command surface",
        "retention": "workspace/audit.log (request_received + response_sent rows; ts + sender + length only, NEVER content)",
        "setting_key": None,
        "category": "messaging",
    },
    {
        "name": "Conversation history",
        "purpose": "context retention for follow-up questions",
        "retention": "workspace/conversation_store.db (SQLite); content stored; retention policy operator-managed",
        "setting_key": None,
        "category": "messaging",
    },
    {
        "name": "Person correlation L1 (presence)",
        "purpose": "track how often people appear in operator's inputs",
        "retention": "workspace/companion/person_model/* — counts only, no message bodies",
        "setting_key": "person_correlation_enabled",
        "category": "person",
    },
    {
        "name": "Person correlation L2-L4 (centrality, social graph)",
        "purpose": "social-graph features + suggestions",
        "retention": "workspace/companion/person_*.json — opt-in cascade; typed-phrase gates",
        "setting_key": "person_correlation_social_graph_enabled",
        "category": "person",
    },
    {
        "name": "Browser-history ingestion",
        "purpose": "interest-signal modality",
        "retention": "workspace/browse/events/<day>.jsonl (canon URLs, no queries/fragments); blocklist applied at read",
        "setting_key": "browse_ingestion_enabled",
        "category": "browse",
    },
    {
        "name": "Browse LLM topic clustering",
        "purpose": "daily theme extraction from titles",
        "retention": "workspace/browse/topics/<day>.json — clustered themes only; titles never sent to LLM without redaction",
        "setting_key": "browse_llm_topics_enabled",
        "category": "browse",
    },
    {
        "name": "Apple Health ingestion",
        "purpose": "personal health data (HR, sleep, steps, body mass)",
        "retention": "workspace/health/<kind>.jsonl — per-kind typed records; NEVER leaves the host (no ChromaDB, no LLM over raw records)",
        "setting_key": "health_ingestion_enabled",
        "category": "health",
    },
    {
        "name": "Inbox multi-modal ingestion",
        "purpose": "file-drop watcher (PDFs, images, audio, spreadsheets, YouTube links)",
        "retention": "workspace/inbox/ + per-handler outputs (notes/, finance/, etc.)",
        "setting_key": "inbox_ingestion_enabled",
        "category": "inbox",
    },
    {
        "name": "Google Workspace (Gmail/Calendar/Docs/Sheets/Slides/Drive)",
        "purpose": "calendar, email, document access",
        "retention": "OAuth refresh token at workspace/google_token.json; content fetched on-demand, not cached",
        "setting_key": None,
        "category": "google",
    },
    {
        "name": "Voice transcripts (Signal audio attachments)",
        "purpose": "voice-mode input",
        "retention": "ephemeral — STT result becomes text on the audit surface; raw audio not persisted past the request",
        "setting_key": "voice_mode",
        "category": "voice",
    },
    {
        "name": "Travel (TripIt iCal)",
        "purpose": "flight/ferry/hotel awareness for briefing",
        "retention": "workspace/life_companion/travel_state.json — segment summaries only",
        "setting_key": None,
        "category": "travel",
    },
    {
        "name": "Affect trace",
        "purpose": "internal welfare signal",
        "retention": "workspace/affect/trace.jsonl (rolled monthly) — emotional state INFERRED ABOUT THE SYSTEM, not about the operator",
        "setting_key": None,
        "category": "internal",
    },
)


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_annual_privacy_review_enabled
        return get_annual_privacy_review_enabled()
    except Exception:
        return os.getenv(
            "ANNUAL_PRIVACY_REVIEW_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _state_path() -> Path:
    return _workspace() / "privacy" / _STATE_FILE


def _wiki_target(year: int) -> Path:
    return _repo_root() / "wiki" / "privacy" / f"audit_{year}.md"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("annual_privacy_review: state write failed", exc_info=True)


def _setting_state(key: Optional[str]) -> str:
    """Read the runtime-settings value for a key. Returns a short
    human-readable string."""
    if not key:
        return "always-on"
    try:
        from app import runtime_settings
        getter = getattr(runtime_settings, f"get_{key}", None)
        if getter is not None:
            value = getter()
            return "ENABLED" if bool(value) else "DISABLED"
        # Fall through to raw read.
        raw = runtime_settings._ensure_initialized().get(key)
        if isinstance(raw, bool):
            return "ENABLED" if raw else "DISABLED"
        if isinstance(raw, str):
            return f"{raw!r}"
        return "?"
    except Exception:
        return "?"


def _ledger_policy_events(year: int) -> list[dict[str, Any]]:
    """Pull every ``*_policy`` continuity-ledger event in the year."""
    out: list[dict[str, Any]] = []
    ledger = _workspace() / "identity" / "continuity_ledger.jsonl"
    if not ledger.exists():
        return out
    year_start = datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()
    year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp()
    try:
        with open(ledger, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                kind = row.get("kind", "")
                if not isinstance(kind, str) or not kind.endswith("_policy"):
                    continue
                ts_str = row.get("ts") or ""
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ts = dt.timestamp()
                except Exception:
                    continue
                if year_start <= ts < year_end:
                    out.append(row)
    except OSError:
        pass
    return out


def _previous_audit_sources(year: int) -> set[str]:
    """Pull the previous year's audit (if any) to compute deltas."""
    prev = _wiki_target(year - 1)
    if not prev.exists():
        return set()
    out: set[str] = set()
    try:
        for line in prev.read_text(encoding="utf-8").splitlines():
            if line.startswith("### "):
                out.add(line[len("### "):].strip())
    except OSError:
        pass
    return out


def compose_review(year: Optional[int] = None) -> Path:
    """Compose this-year's audit and write it to
    ``wiki/privacy/audit_<year>.md``. Returns the written path."""
    yr = int(year) if year is not None else datetime.now(timezone.utc).year
    target = _wiki_target(yr)
    target.parent.mkdir(parents=True, exist_ok=True)
    prev_sources = _previous_audit_sources(yr)
    cur_sources = {s["name"] for s in _DATA_SOURCES}
    new_sources = sorted(cur_sources - prev_sources)
    removed_sources = sorted(prev_sources - cur_sources)
    policy_events = _ledger_policy_events(yr)
    lines: list[str] = [
        f"# Annual privacy audit — {yr}",
        "",
        f"_Composed at {datetime.now(timezone.utc).isoformat()} by "
        f"`app.privacy.annual_review.compose_review`._",
        "",
        "This audit enumerates every data source the system actively",
        "reads, its current on/off state, and where the data lives.",
        "It is observational only — no source is disabled by this",
        "process. The operator reviews and decides.",
        "",
        "## Sources by category",
        "",
    ]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for src in _DATA_SOURCES:
        by_category.setdefault(src["category"], []).append(src)
    for category in sorted(by_category):
        lines.append(f"### {category.capitalize()}")
        lines.append("")
        for src in by_category[category]:
            lines.append(f"### {src['name']}")
            lines.append("")
            lines.append(f"- **Purpose**: {src['purpose']}")
            lines.append(f"- **Retention**: {src['retention']}")
            lines.append(
                f"- **State**: `{_setting_state(src.get('setting_key'))}`"
                + (f" (gated by `{src['setting_key']}`)" if src.get("setting_key") else "")
            )
            lines.append("")
    lines.append("## Year delta")
    lines.append("")
    if new_sources:
        lines.append("**New sources since previous audit:**")
        for s in new_sources:
            lines.append(f"  - `{s}`")
        lines.append("")
    else:
        lines.append("No new sources since the previous audit.")
        lines.append("")
    if removed_sources:
        lines.append("**Sources removed since previous audit:**")
        for s in removed_sources:
            lines.append(f"  - `{s}`")
        lines.append("")
    lines.append("## Policy events this year")
    lines.append("")
    if policy_events:
        for evt in policy_events:
            ts = (evt.get("ts") or "")[:19]
            kind = evt.get("kind", "(unknown)")
            summary = (evt.get("summary") or "")[:200]
            lines.append(f"- `{ts}` `{kind}` — {summary}")
        lines.append("")
    else:
        lines.append(
            "No continuity-ledger ``*_policy`` events were emitted "
            "this year. (This usually means no master switches were "
            "flipped — review the per-source states above for the "
            "current envelope.)"
        )
        lines.append("")
    lines.append("## Operator next steps")
    lines.append("")
    lines.append(
        "  1. Confirm every ENABLED source above is still serving a "
        "purpose."
    )
    lines.append(
        "  2. For any source you no longer need, flip its master "
        "switch and (if applicable) run the source's `forget` path."
    )
    lines.append(
        "  3. Save this file to git so the year-over-year delta on the "
        "next audit is meaningful."
    )
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "annual_privacy_review: wrote %s (%d sources, %d new, %d policy events)",
        target, len(_DATA_SOURCES), len(new_sources), len(policy_events),
    )
    return target


def run_once(*, now: Optional[float] = None) -> dict[str, Any]:
    """Idle-job entry. Composes at most one review per year per
    process. Failure-isolated."""
    summary: dict[str, Any] = {
        "ran": False,
        "wrote": None,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary
    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last_run_at = float(state.get("last_run_at", 0))
    if last_run_at > 0 and (cur - last_run_at) < _MIN_DAYS_BETWEEN_REVIEWS * 86400:
        return summary
    summary["ran"] = True
    try:
        path = compose_review()
        summary["wrote"] = str(path)
        state["last_run_at"] = cur
        _write_state(state)
        try:
            from app.notify import notify
            notify(
                title="📋 Annual privacy audit composed",
                body=(
                    f"The annual privacy audit landed at `{path}`. "
                    f"Review each ENABLED data source and confirm the "
                    f"envelope is what you want."
                ),
                url="/cp/files",
                topic="annual_privacy_review",
                critical=False,
                arbitrate=True,
            )
        except Exception:
            logger.debug(
                "annual_privacy_review: notify failed", exc_info=True,
            )
    except Exception:
        logger.debug(
            "annual_privacy_review: compose failed", exc_info=True,
        )
    return summary
