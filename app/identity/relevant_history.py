"""Path-keyed history aggregator for proposal evaluation.

When a Tier-3 amendment or change-request is being decided, the
operator wants to see "this file has been amended 3× in the last
60 days" inline with the proposal. The data exists in two append-
only logs:

  * ``app/identity/continuity_ledger.py`` — six identity-shaping
    event kinds (tier3_amendment, governance_ratchet, soul_edit,
    integrity_regen, scorecard_change, self_quarantine_change)
  * ``workspace/change_requests/audit.jsonl`` — every CR state
    transition (created, approved, applied, rejected, …)

This module joins both by target path within a rolling window and
returns a structured summary for inclusion in operator-facing
review surfaces (``CR.reason``, Tier-3 ``proposal.evidence``,
governance_notifier Signal alerts).

Read-only: never writes to either source. The continuity ledger
remains a narrative artefact emitted by identity-shaping events,
not augmented by retrospective queries — preserving annual_reflection's
drift-summary correctness.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────


DEFAULT_WINDOW_DAYS = 90

# Hard cap on entries returned per source so CR.reason stays bounded
# (the 1 MB CR-content cap is far away, but readability matters).
_MAX_PER_SOURCE = 5
_MAX_DESCRIPTION_CHARS = 80


# ── Public API ────────────────────────────────────────────────────────


def relevant_history(path: str, *, window_days: int = DEFAULT_WINDOW_DAYS) -> dict:
    """Aggregate path-relevant history from continuity ledger + CR audit.

    Returns:
        {
          "window_days": int,
          "continuity_events": [
              {"ts": "...", "kind": "...", "actor": "...",
               "summary": "...", "detail": {...}},
              ...
          ],
          "change_request_events": [
              {"ts": "...", "event": "...", "cr_id": "...",
               "status": "...", "requestor": "...", "decided_by": "..."},
              ...
          ],
          "counts": {"ledger": int, "cr_audit": int},
          "summary_line": "3 CRs in 60d, 1 amendment in 90d, 0 governance ratchets",
        }

    Empty results: returns a dict with empty lists and counts both 0.
    Never raises — failures degrade silently (operator sees the
    proposal without history rather than a crashed CR creation).
    """
    if not (path or "").strip():
        return _empty_result(window_days)

    cutoff_iso = (
        datetime.now(timezone.utc) - timedelta(days=window_days)
    ).isoformat()

    ledger_events = _read_continuity_ledger(path, cutoff_iso=cutoff_iso)
    cr_events = _read_cr_audit_log(path, cutoff_iso=cutoff_iso)

    counts = {"ledger": len(ledger_events), "cr_audit": len(cr_events)}
    summary_line = _build_summary_line(
        ledger_events=ledger_events,
        cr_events=cr_events,
        window_days=window_days,
    )
    return {
        "window_days": window_days,
        "continuity_events": ledger_events[-_MAX_PER_SOURCE:],
        "change_request_events": cr_events[-_MAX_PER_SOURCE:],
        "counts": counts,
        "summary_line": summary_line,
    }


def format_for_operator(history: dict) -> str:
    """Render the history dict as a human-readable markdown block for
    appending to ``CR.reason`` or a Signal alert. Returns an empty
    string when history is empty (no need to clutter the surface)."""
    if not history:
        return ""
    counts = history.get("counts") or {}
    if not (counts.get("ledger", 0) or counts.get("cr_audit", 0)):
        return ""

    lines = [
        f"📜 Recent activity on this path (last {history['window_days']}d):",
        f"   {history['summary_line']}",
    ]
    ledger = history.get("continuity_events") or []
    cr_events = history.get("change_request_events") or []
    if ledger:
        lines.append("")
        lines.append("   Continuity-ledger events:")
        for e in ledger:
            ts_short = (e.get("ts") or "")[:10]  # YYYY-MM-DD
            kind = e.get("kind") or "?"
            actor = e.get("actor") or "?"
            summary = (e.get("summary") or "")[:_MAX_DESCRIPTION_CHARS]
            lines.append(f"     • {ts_short} [{kind}] {actor}: {summary}")
    if cr_events:
        lines.append("")
        lines.append("   Change-request audit events:")
        for e in cr_events:
            ts_short = (e.get("ts") or "")[:10]
            event = e.get("event") or "?"
            cr_id = e.get("cr_id") or "?"
            status = e.get("status") or "?"
            decided_by = e.get("decided_by") or "—"
            lines.append(
                f"     • {ts_short} [{event}] cr={cr_id} status={status} by={decided_by}"
            )
    return "\n".join(lines)


# ── Continuity-ledger read ────────────────────────────────────────────


def _read_continuity_ledger(path: str, *, cutoff_iso: str) -> list[dict]:
    """Walk the continuity ledger, return events whose ``detail``
    references the target path, sorted oldest→newest."""
    try:
        from app.identity.continuity_ledger import list_events
    except Exception:
        logger.debug("relevant_history: continuity_ledger import failed",
                     exc_info=True)
        return []
    try:
        events = list_events(since_iso=cutoff_iso)
    except Exception:
        logger.debug("relevant_history: list_events failed", exc_info=True)
        return []

    out: list[dict] = []
    for event in events:
        if not _ledger_event_matches_path(event, path):
            continue
        out.append({
            "ts": event.ts,
            "kind": event.kind,
            "actor": event.actor,
            "summary": event.summary,
            "detail": dict(event.detail or {}),
        })
    return out


def _ledger_event_matches_path(event: Any, path: str) -> bool:
    """Match an IdentityEvent's detail dict against the target path.

    Different emitters use different keys (``path`` /
    ``target_path`` / ``request_id`` etc). We check the full set of
    known path-keys; this is a forward-compatible filter — adding a
    new emitter that uses one of these keys works without changes
    here. Adding a new emitter with a NEW key requires extending
    this list, which is the right amount of friction.
    """
    detail = event.detail or {}
    for key in ("path", "target_path", "file_path"):
        value = detail.get(key)
        if isinstance(value, str) and value == path:
            return True
    return False


# ── Change-request audit read ─────────────────────────────────────────


def _cr_audit_path() -> Path:
    """Return the change-request audit log path. Honours the
    ``CHANGE_REQUESTS_DIR`` env var the same way auto_revert does."""
    base = Path(os.environ.get("CHANGE_REQUESTS_DIR")
                or "/app/workspace/change_requests")
    return base / "audit.jsonl"


def _read_cr_audit_log(path: str, *, cutoff_iso: str) -> list[dict]:
    """Walk the CR audit JSONL, return entries whose payload
    references the target path within the window. Sorted oldest→newest.
    """
    log_path = _cr_audit_path()
    if not log_path.exists():
        return []

    out: list[dict] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = entry.get("ts") or ""
                if ts < cutoff_iso:
                    continue
                payload = entry.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                if payload.get("path") != path:
                    continue
                out.append({
                    "ts": ts,
                    "event": payload.get("event") or "?",
                    "cr_id": payload.get("request_id") or "",
                    "status": payload.get("status") or "?",
                    "requestor": payload.get("requestor") or "?",
                    "decided_by": payload.get("decided_by") or "",
                })
    except OSError:
        return []
    return out


# ── Summary line ──────────────────────────────────────────────────────


def _build_summary_line(
    *, ledger_events: list[dict], cr_events: list[dict], window_days: int,
) -> str:
    """One-line summary for headers and Signal alerts."""
    if not ledger_events and not cr_events:
        return f"no prior activity in {window_days}d"

    ledger_by_kind: dict[str, int] = {}
    for e in ledger_events:
        ledger_by_kind[e["kind"]] = ledger_by_kind.get(e["kind"], 0) + 1

    cr_by_event: dict[str, int] = {}
    for e in cr_events:
        cr_by_event[e["event"]] = cr_by_event.get(e["event"], 0) + 1

    parts: list[str] = []
    # CRs: only the actionable transitions (created / applied / rejected)
    # are operator-relevant for context.
    cr_created = cr_by_event.get("created", 0)
    cr_applied = cr_by_event.get("applied", 0)
    cr_rejected = cr_by_event.get("rejected", 0) + cr_by_event.get("validation_failed", 0)
    if cr_created:
        parts.append(f"{cr_created} CR{'s' if cr_created != 1 else ''}")
    if cr_applied:
        parts.append(f"{cr_applied} applied")
    if cr_rejected:
        parts.append(f"{cr_rejected} rejected")

    # Ledger: enumerate top kinds.
    for kind, n in sorted(ledger_by_kind.items(), key=lambda x: -x[1]):
        if kind == "tier3_amendment":
            parts.append(f"{n} amendment{'s' if n != 1 else ''}")
        elif kind == "governance_ratchet":
            parts.append(f"{n} governance-ratchet event{'s' if n != 1 else ''}")
        elif kind == "soul_edit":
            parts.append(f"{n} soul edit{'s' if n != 1 else ''}")
        else:
            parts.append(f"{n} {kind.replace('_', '-')}")

    return ", ".join(parts) + f" in {window_days}d"


# ── Empty-result helper ───────────────────────────────────────────────


def _empty_result(window_days: int) -> dict:
    return {
        "window_days": window_days,
        "continuity_events": [],
        "change_request_events": [],
        "counts": {"ledger": 0, "cr_audit": 0},
        "summary_line": f"no prior activity in {window_days}d",
    }
