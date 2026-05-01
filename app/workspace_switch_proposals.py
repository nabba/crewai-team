"""User-confirm dialog for auto-detected workspace switches.

When the keyword auto-detector finds a likely workspace match that
differs from the current active workspace, we don't switch silently —
we send the user a Signal asking them to confirm with 👍 / 👎. The
👍 reaction causes a deterministic switch with ``source="user"`` (so
it's sticky); 👎 records a decline that suppresses re-asking the same
detection for 24 h.

Added 2026-05-02 after the user reported their explicit
``switch workspace to eesti mets`` kept getting silently overridden by
keyword auto-detection (every message containing "estonia" / "event"
re-routed to PLG, and tickets ended up filed there). Two layered
fixes:

  1. ``ProjectManager.switch`` tracks `_active_project_source` and
     refuses to overwrite a "user" pick from "auto" calls (see
     control_plane/projects.py).

  2. This module turns auto-detect into a HINT: instead of attempting
     a silent switch (which the sticky-user guard would just block),
     it asks the user via Signal. If the user confirms, the resulting
     switch carries ``source="user"`` so it's also sticky.

Storage: JSON queue at ``/app/workspace/workspace_switch_proposals.json``
(same pattern as human_gate / proposals — no new DB migration needed).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_QUEUE_PATH = Path("/app/workspace/workspace_switch_proposals.json")

# How long a pending ask sits in the queue before we'll re-ask the
# same (sender, detected_name) pair. 30 min is short enough to retry
# after an inattentive user but long enough to avoid spam.
PENDING_TTL_S = 1800

# How long a declined detection is suppressed. 24 h is conservative —
# the user can always switch via the explicit Signal command.
DECLINE_TTL_S = 86400

# Cap queue file at 200 entries so it doesn't grow unbounded.
_MAX_ENTRIES = 200


# ── Persistence ──────────────────────────────────────────────────────

def _load() -> list[dict]:
    if not _QUEUE_PATH.exists():
        return []
    try:
        data = json.loads(_QUEUE_PATH.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(entries: list[dict]) -> None:
    try:
        _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        bounded = entries[-_MAX_ENTRIES:]
        tmp = _QUEUE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(bounded, indent=2, default=str))
        tmp.replace(_QUEUE_PATH)
    except OSError as e:
        logger.warning(f"workspace_switch_proposals: save failed: {e}")


# ── Public API ───────────────────────────────────────────────────────

def has_recent_decision(detected_name: str, sender: str, *, now: float | None = None) -> bool:
    """True iff we recently asked OR were declined on this same
    (sender, detected_name) pair. Used to suppress repeated nags."""
    ts = now if now is not None else time.time()
    pending_cutoff = ts - PENDING_TTL_S
    decline_cutoff = ts - DECLINE_TTL_S
    needle = (detected_name or "").lower()
    for e in _load():
        if e.get("sender") != sender:
            continue
        if (e.get("detected_name") or "").lower() != needle:
            continue
        decision = e.get("decision")
        created = e.get("created_at", 0) or 0
        if decision == "declined" and created > decline_cutoff:
            return True
        if decision == "pending" and created > pending_cutoff:
            return True
    return False


def propose(
    detected_name: str,
    current_name: str,
    sender: str,
    *,
    now: float | None = None,
    notifier: callable = None,  # noqa: ANN001 — injectable for tests
) -> str | None:
    """Send a Signal asking the user whether to switch.

    Returns the proposal_id on success (entry persisted), None on
    failure (Signal send failed). The ``notifier`` arg is injected for
    tests; production callers leave it None and we use signal_client.

    Idempotency: callers should check ``has_recent_decision`` first to
    avoid asking the same question twice in quick succession.
    """
    proposal_id = uuid.uuid4().hex[:12]
    msg = (
        f"💡 This message looks like it might belong to *{detected_name}* "
        f"(currently on *{current_name}*).\n\n"
        f"React 👍 to switch to *{detected_name}*, or 👎 to stay on *{current_name}*."
    )

    signal_ts: int | None = None
    if notifier is None:
        try:
            from app.signal_client import send_message_blocking
            signal_ts = send_message_blocking(sender, msg)
        except Exception as e:
            logger.warning(
                f"workspace_switch_proposals: notify failed for "
                f"{detected_name!r} → {sender!r}: {e}"
            )
            return None
    else:
        try:
            signal_ts = notifier(sender, msg)
        except Exception as e:
            logger.warning(f"workspace_switch_proposals: injected notifier failed: {e}")
            return None

    entries = _load()
    entries.append({
        "proposal_id": proposal_id,
        "sender": sender,
        "detected_name": detected_name,
        "current_name": current_name,
        "signal_timestamp": signal_ts,
        "created_at": now if now is not None else time.time(),
        "decision": "pending",
    })
    _save(entries)
    logger.info(
        f"workspace_switch_proposals: asked {sender} about "
        f"{current_name} → {detected_name} (proposal_id={proposal_id})"
    )
    return proposal_id


def find_by_signal_ts(ts: int) -> str | None:
    """Look up a pending proposal by its Signal notification timestamp.

    Used by the gateway reaction handler to route 👍 / 👎 reactions.
    Mirrors human_gate.find_request_by_signal_timestamp.
    """
    if not ts:
        return None
    for e in _load():
        if e.get("decision") != "pending":
            continue
        if e.get("signal_timestamp") == ts:
            return e.get("proposal_id")
    return None


def accept(proposal_id: str) -> str:
    """Apply the proposed switch (source='user' so it's sticky).

    Returns a human-readable status string for the reaction-ack
    Signal message.
    """
    entries = _load()
    for e in entries:
        if e.get("proposal_id") != proposal_id:
            continue
        if e.get("decision") != "pending":
            return f"Proposal already {e.get('decision', 'closed')}"
        try:
            from app.control_plane.projects import get_projects
            result = get_projects().switch(e["detected_name"], source="user")
            if result:
                e["decision"] = "accepted"
                e["decided_at"] = time.time()
                _save(entries)
                return f"Switched to {result.get('name')}"
            e["decision"] = "failed"
            _save(entries)
            return f"Project '{e['detected_name']}' not found"
        except Exception as exc:
            logger.warning(f"workspace_switch_proposals: accept failed: {exc}")
            e["decision"] = "error"
            _save(entries)
            return f"Error: {str(exc)[:200]}"
    return "Proposal not found"


def decline(proposal_id: str) -> str:
    """Mark proposal declined. The (sender, detected_name) pair is
    suppressed for DECLINE_TTL_S afterwards so we don't keep asking."""
    entries = _load()
    for e in entries:
        if e.get("proposal_id") != proposal_id:
            continue
        if e.get("decision") != "pending":
            return f"Proposal already {e.get('decision', 'closed')}"
        e["decision"] = "declined"
        e["decided_at"] = time.time()
        _save(entries)
        return f"Staying on {e.get('current_name', 'current workspace')}"
    return "Proposal not found"


def expire_stale(*, now: float | None = None) -> int:
    """Auto-expire pending proposals older than PENDING_TTL_S so the
    queue file doesn't fill with abandoned asks. Returns count expired."""
    ts = now if now is not None else time.time()
    cutoff = ts - PENDING_TTL_S
    entries = _load()
    expired = 0
    for e in entries:
        if e.get("decision") != "pending":
            continue
        if (e.get("created_at", 0) or 0) < cutoff:
            e["decision"] = "expired"
            e["decided_at"] = ts
            expired += 1
    if expired > 0:
        _save(entries)
    return expired
