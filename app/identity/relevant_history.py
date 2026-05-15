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


# ── Q5.1: Ledger-as-governor — file-kind aggregator ───────────────────
#
# `relevant_history(path)` answers "what happened to THIS file lately?".
# `relevant_history_by_kind(path)` answers "what's the empirical track
# record for files of the SAME KIND as this one?". For Tier-3 amendment
# proposals especially, the question "have I been amending governance
# files a lot and rolling them back?" is the relevant institutional-
# memory signal — not just whether THIS file was touched.
#
# Goodhart guard: this is informational only. The proposer sees the
# history in the proposal evidence; the operator sees it in the Signal
# alert; neither gating logic nor the protocol consumes it. Surfacing
# bad history slows operators down; pretending it doesn't exist would
# silence institutional memory. We choose visibility.


# Default window for by-kind queries — longer than the per-path
# default because file-kind track records take longer to accumulate.
DEFAULT_BY_KIND_WINDOW_DAYS = 365


# Path → kind classifier. Explicit, hand-curated, no auto-learning.
# Order matters — we walk top to bottom and pick the first match.
# Tuples are (prefix_pattern, kind_label). The pattern is a simple
# startswith-or-equals check; tighter than regex but predictable.
_PATH_KIND_RULES: list[tuple[str, str]] = [
    # Governance + safety core
    ("app/souls/", "soul_edit"),
    ("wiki/governance/constitution.md", "governance_constitution"),
    ("wiki/governance/", "governance_doc"),
    ("app/governance_amendment/", "amendment_protocol"),
    ("app/governance_ratchet/", "governance_ratchet"),
    ("app/affect/welfare.py", "welfare_envelope"),
    ("app/goodhart_guard.py", "goodhart_gate"),
    ("app/safety_guardian.py", "safety_core"),
    ("app/eval_sandbox", "eval_sandbox"),
    ("app/alignment_audit", "alignment_audit"),
    # SubIA kernel — broad bucket
    ("app/subia/integrity.py", "integrity_manifest"),
    ("app/subia/", "kernel"),
    # Agent + tool surfaces
    ("app/agents/", "agent_definition"),
    ("app/tools/", "tool_implementation"),
    ("app/tool_registry/", "tool_registry"),
    ("app/tool_runtime/", "tool_runtime"),
    # Memory + storage
    ("app/memory/", "memory_store"),
    ("app/control_plane/", "control_plane"),
    # Identity + reflection
    ("app/identity/", "identity_layer"),
    ("app/affect/", "affect_layer"),
    # Companion + life
    ("app/companion/", "companion"),
    ("app/life_companion/", "life_companion"),
    # Healing
    ("app/healing/", "healing"),
    # Wiki + docs
    ("wiki/self/", "wiki_self"),
    ("wiki/", "wiki"),
    ("docs/", "docs"),
    # Change-request infrastructure
    ("app/change_requests/", "change_requests_infra"),
    # Tests
    ("tests/", "tests"),
    ("crewai-team/tests/", "tests"),
]


def classify_path(path: str) -> str:
    """Classify a repo-relative path into a coarse file-kind label.

    Returns ``"other"`` when no rule matches. The taxonomy is
    intentionally coarse — too-fine buckets fragment the track record
    and defeat the purpose of looking at *kinds*. New rules should be
    added explicitly when an operator notices a coherent bucket missing
    from the empirical track record.

    Failure-isolated: malformed input returns ``"other"``.
    """
    p = (path or "").strip()
    if not p:
        return "other"
    # Normalize Windows-style separators just in case.
    p = p.replace("\\", "/")
    # Strip a leading "./" or "/".
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        p = p[1:]
    for prefix, kind in _PATH_KIND_RULES:
        if p == prefix or p.startswith(prefix):
            return kind
    return "other"


def relevant_history_by_kind(
    path: str,
    *,
    window_days: int = DEFAULT_BY_KIND_WINDOW_DAYS,
) -> dict:
    """Aggregate ledger + CR-audit events for the FILE KIND of ``path``
    across the rolling window.

    Returns:
        {
          "file_kind": str,
          "window_days": int,
          "counts_by_outcome": {
            "applied": int, "rolled_back": int, "rejected": int,
            "in_flight": int, "amended": int, "ratcheted": int,
          },
          "recent_events": [
            {"ts": "...", "path": "...", "kind": "...",
             "outcome": "...", "summary": "..."}, ...
          ],
          "last_outcome_at": "..." | None,
          "success_rate": float,    # applied / (applied + rolled_back); 0.0 when N/A
          "summary_line": str,      # e.g. "3 amendments, 2 applied / 1 rolled back over 365d"
        }

    Empty-result discipline matches ``relevant_history``: never raises,
    returns a structured empty dict when the kind has no recorded
    events in the window."""
    kind = classify_path(path)
    if not (path or "").strip():
        return _empty_by_kind(kind, window_days)
    # Master-switch gate. When OFF we return the empty shell so callers
    # always see the same dict shape (defensive against ad-hoc reads).
    try:
        from app.runtime_settings import get_ledger_governor_enabled
        if not get_ledger_governor_enabled():
            return _empty_by_kind(kind, window_days)
    except Exception:
        # Fall through to the default-on behavior.
        pass

    cutoff_iso = (
        datetime.now(timezone.utc) - timedelta(days=window_days)
    ).isoformat()

    # Walk the ledger + CR audit collecting events for any path
    # classifying to the same kind.
    ledger_hits = _read_continuity_ledger_by_kind(kind, cutoff_iso=cutoff_iso)
    cr_hits = _read_cr_audit_log_by_kind(kind, cutoff_iso=cutoff_iso)

    counts = _aggregate_counts(ledger_hits, cr_hits)
    # Combined + chronologically sorted view (newest-first for operator
    # surfaces; readers wanting oldest-first can reverse).
    combined: list[dict] = []
    for h in ledger_hits:
        combined.append({
            "ts": h["ts"], "path": h.get("path", ""),
            "kind": "ledger:" + h["kind"], "outcome": h.get("outcome"),
            "summary": h.get("summary", ""),
        })
    for h in cr_hits:
        combined.append({
            "ts": h["ts"], "path": h.get("path", ""),
            "kind": "cr:" + h.get("event", "?"), "outcome": h.get("outcome"),
            "summary": h.get("summary", ""),
        })
    combined.sort(key=lambda e: e["ts"], reverse=True)
    recent = combined[: _MAX_PER_SOURCE * 2]
    last_outcome_at = combined[0]["ts"] if combined else None

    applied = counts.get("applied", 0)
    rolled_back = counts.get("rolled_back", 0)
    denom = applied + rolled_back
    success_rate = round(applied / denom, 3) if denom else 0.0
    # Q5.5 — distinguish "no history" (the prior is uniform) from
    # "proven 0% success" (the prior is bad). Both collapse to
    # success_rate=0.0 above; downstream consumers (RPT-1 producers
    # at Tier-3 + CR creation) need to know which case they're in.
    has_resolved_history = denom > 0

    summary_line = _build_by_kind_summary(kind, counts, window_days)

    return {
        "file_kind": kind,
        "window_days": window_days,
        "counts_by_outcome": counts,
        "recent_events": recent,
        "last_outcome_at": last_outcome_at,
        "success_rate": success_rate,
        "has_resolved_history": has_resolved_history,
        "summary_line": summary_line,
    }


def format_by_kind_for_operator(history: dict) -> str:
    """Render a by-kind history dict as a compact markdown block. Empty
    string when nothing actionable to show."""
    if not history:
        return ""
    counts = history.get("counts_by_outcome") or {}
    if not any(counts.values()):
        return ""
    kind = history.get("file_kind", "?")
    summary = history.get("summary_line") or ""
    success = history.get("success_rate") or 0.0
    lines = [
        f"📊 Track record for kind={kind!r} ({history.get('window_days', 0)}d):",
        f"   {summary}",
    ]
    applied = counts.get("applied", 0)
    rolled_back = counts.get("rolled_back", 0)
    if applied + rolled_back > 0:
        lines.append(f"   success rate: {success:.0%} ({applied} applied / {rolled_back} rolled back)")
    return "\n".join(lines)


# ── Internal: ledger + CR scans by-kind ───────────────────────────────


def _ledger_event_extract_paths(event: Any) -> list[str]:
    """Return all path-like strings from an event's ``detail`` dict.
    Mirrors the keys ``_ledger_event_matches_path`` consults."""
    detail = event.detail or {}
    out: list[str] = []
    for key in ("path", "target_path", "file_path"):
        value = detail.get(key)
        if isinstance(value, str) and value:
            out.append(value)
    return out


def _outcome_from_ledger_kind(kind: str) -> str | None:
    """Map a ledger event kind to a high-level outcome bucket. None when
    the event isn't an outcome-class event (e.g. integrity_regen)."""
    if kind == "tier3_amendment":
        return "applied"
    if kind == "governance_ratchet":
        return "ratcheted"
    if kind == "soul_edit":
        return "amended"
    return None


def _read_continuity_ledger_by_kind(target_kind: str, *, cutoff_iso: str) -> list[dict]:
    """Walk the ledger and pull events whose detail paths classify to
    the same kind."""
    try:
        from app.identity.continuity_ledger import list_events
    except Exception:
        return []
    try:
        events = list_events(since_iso=cutoff_iso)
    except Exception:
        return []
    out: list[dict] = []
    for ev in events:
        # An event may carry multiple paths (rare); match on any of them.
        paths = _ledger_event_extract_paths(ev)
        match_path: str | None = None
        for p in paths:
            if classify_path(p) == target_kind:
                match_path = p
                break
        if match_path is None:
            continue
        out.append({
            "ts": ev.ts,
            "kind": ev.kind,
            "actor": ev.actor,
            "summary": ev.summary,
            "path": match_path,
            "outcome": _outcome_from_ledger_kind(ev.kind),
        })
    return out


def _outcome_from_cr_event(event: str, status: str) -> str | None:
    """Map a CR audit event/status to an outcome bucket."""
    e = (event or "").lower()
    s = (status or "").lower()
    if e == "applied" or s == "applied":
        return "applied"
    if e in ("rolled_back", "reverted") or s in ("rolled_back", "reverted"):
        return "rolled_back"
    if e in ("rejected", "validation_failed") or s in ("rejected", "validation_failed"):
        return "rejected"
    if e == "created" or s in ("pending", "approved"):
        return "in_flight"
    return None


def _read_cr_audit_log_by_kind(target_kind: str, *, cutoff_iso: str) -> list[dict]:
    """Walk the CR audit JSONL, pull entries whose path classifies."""
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
                path = payload.get("path") or ""
                if not path or classify_path(path) != target_kind:
                    continue
                event = payload.get("event") or "?"
                status = payload.get("status") or ""
                out.append({
                    "ts": ts,
                    "event": event,
                    "cr_id": payload.get("request_id", ""),
                    "status": status,
                    "requestor": payload.get("requestor", ""),
                    "path": path,
                    "outcome": _outcome_from_cr_event(event, status),
                    "summary": f"cr={payload.get('request_id', '?')} {event}",
                })
    except OSError:
        return []
    return out


def _aggregate_counts(
    ledger_hits: list[dict], cr_hits: list[dict],
) -> dict[str, int]:
    """Combine ledger + CR hits into outcome buckets."""
    buckets = {
        "applied": 0,
        "rolled_back": 0,
        "rejected": 0,
        "in_flight": 0,
        "amended": 0,
        "ratcheted": 0,
    }
    for h in ledger_hits + cr_hits:
        outcome = h.get("outcome")
        if outcome and outcome in buckets:
            buckets[outcome] += 1
    return buckets


def _build_by_kind_summary(
    kind: str, counts: dict[str, int], window_days: int,
) -> str:
    total = sum(counts.values())
    if total == 0:
        return f"no prior activity for kind={kind!r} in {window_days}d"
    parts: list[str] = []
    if counts.get("applied"):
        parts.append(f"{counts['applied']} applied")
    if counts.get("rolled_back"):
        parts.append(f"{counts['rolled_back']} rolled back")
    if counts.get("rejected"):
        parts.append(f"{counts['rejected']} rejected")
    if counts.get("in_flight"):
        parts.append(f"{counts['in_flight']} in flight")
    if counts.get("amended"):
        parts.append(f"{counts['amended']} soul/doc edits")
    if counts.get("ratcheted"):
        parts.append(f"{counts['ratcheted']} ratchet events")
    return f"kind={kind}: " + ", ".join(parts) + f" over {window_days}d"


def _empty_by_kind(kind: str, window_days: int) -> dict:
    return {
        "file_kind": kind,
        "window_days": window_days,
        "counts_by_outcome": {
            "applied": 0, "rolled_back": 0, "rejected": 0,
            "in_flight": 0, "amended": 0, "ratcheted": 0,
        },
        "recent_events": [],
        "last_outcome_at": None,
        "success_rate": 0.0,
        "has_resolved_history": False,  # Q5.5 — distinguishes from proven 0%
        "summary_line": f"no prior activity for kind={kind!r} in {window_days}d",
    }
