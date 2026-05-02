"""Surfacing — decide whether to surface, format the card, send via Signal.

A converged idea is eligible for surfacing when:
  - novelty ≥ config.novelty_threshold
  - quality ≥ config.surface_threshold
  - panel_score ≥ config.panel_threshold (Phase 7 critic panel)
  - the workspace hasn't surfaced anything in the last SURFACE_COOLDOWN_HOURS
    (prevents flooding the user with too many cards)

On surface, a SURFACED event is appended to the workspace event log, the
formatted card is sent to Signal via ``_send_signal`` (Phase 4 stub —
logs the would-send text; Phase 4.5 wires ``conversation_store.enqueue_outbound``
once per-workspace recipient mapping is configured).

Failures of the Signal send are recorded in the SURFACED event payload
but do not block the state transition — the surface ATTEMPT is durable
even if delivery fails (operator can retry).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.companion import events as _events
from app.companion.config import CompanionConfig
from app.companion.idea_store import IdeaRecord

logger = logging.getLogger(__name__)

# Min hours between two surfaces for the same workspace.
SURFACE_COOLDOWN_HOURS = 4
SURFACE_COOLDOWN_S = SURFACE_COOLDOWN_HOURS * 3600


@dataclass
class SurfaceDecision:
    """Outcome of a surfacing check."""
    eligible: bool
    # "ok" | "below_novelty" | "below_quality" | "below_panel" |
    # "cooldown" | "no_text"
    reason: str


def should_surface(idea: IdeaRecord, config: CompanionConfig, *,
                   now: float | None = None) -> SurfaceDecision:
    """Threshold + panel + cooldown check. Pure — does not write any state."""
    from app.observability import valve_audit

    if not (idea.text or "").strip():
        valve_audit.log_rejection(
            filter_id="F8", callsite="app/companion/surfacing.py:50",
            input_text=idea.text or "", reason="no_text",
            extra={"workspace_id": idea.workspace_id},
        )
        return SurfaceDecision(False, "no_text")
    if idea.novelty < config.novelty_threshold:
        valve_audit.log_rejection(
            filter_id="F8", callsite="app/companion/surfacing.py:52",
            input_text=idea.text, reason="below_novelty",
            score=float(idea.novelty), threshold=float(config.novelty_threshold),
            extra={"workspace_id": idea.workspace_id, "quality": idea.quality,
                   "panel_score": idea.panel_score},
        )
        return SurfaceDecision(False, "below_novelty")
    if idea.quality < config.surface_threshold:
        valve_audit.log_rejection(
            filter_id="F8", callsite="app/companion/surfacing.py:54",
            input_text=idea.text, reason="below_quality",
            score=float(idea.quality), threshold=float(config.surface_threshold),
            extra={"workspace_id": idea.workspace_id, "novelty": idea.novelty,
                   "panel_score": idea.panel_score},
        )
        return SurfaceDecision(False, "below_quality")
    if idea.panel_score < config.panel_threshold:
        valve_audit.log_rejection(
            filter_id="F8", callsite="app/companion/surfacing.py:56",
            input_text=idea.text, reason="below_panel",
            score=float(idea.panel_score), threshold=float(config.panel_threshold),
            extra={"workspace_id": idea.workspace_id, "novelty": idea.novelty,
                   "quality": idea.quality},
        )
        return SurfaceDecision(False, "below_panel")
    if _recently_surfaced(idea.workspace_id, now=now):
        # Cooldown is intentionally narrow (rate-limit, not quality) — log but
        # treat as separate from substantive rejections at replay time.
        valve_audit.log_rejection(
            filter_id="F8", callsite="app/companion/surfacing.py:58",
            input_text=idea.text, reason="cooldown",
            extra={"workspace_id": idea.workspace_id},
        )
        return SurfaceDecision(False, "cooldown")
    return SurfaceDecision(True, "ok")


def surface(idea: IdeaRecord, config: CompanionConfig) -> bool:
    """Send the card and append a SURFACED event. Returns True on send.

    The SURFACED event is appended even when the send returns False, so
    cooldown counts the attempt and operators can see what was queued.
    """
    text = compose_card(idea, config)
    sent = False
    error: str | None = None
    try:
        sent = _send_signal(text, idea.workspace_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("companion.surfacing: send raised: %s", error)

    payload = {
        "card_text": text[:500],
        "novelty": idea.novelty,
        "quality": idea.quality,
        "transferability": idea.transferability,
        "panel_score": idea.panel_score,
        "signal_sent": sent,
    }
    if error:
        payload["send_error"] = error

    _events.append(_events.Event(
        workspace_id=idea.workspace_id,
        idea_id=idea.idea_id,
        type=_events.EventType.SURFACED,
        payload=payload,
    ))
    return sent


def compose_card(idea: IdeaRecord, config: CompanionConfig) -> str:
    """Format the user-facing Signal/React idea card."""
    seed = (config.seed_prompt or "").strip() or "(no seed)"
    head = (idea.text or "").strip()
    if len(head) > 1500:
        head = head[:1497] + "..."
    return (
        f"[Companion · {seed[:60]}]\n"
        f"New idea (novelty {idea.novelty:.2f}, quality {idea.quality:.2f}, "
        f"panel {idea.panel_score:.2f}):\n\n"
        f"{head}\n\n"
        f"Reply Y to keep, N to drop, or 'more <comment>' to refine."
    )


def _recently_surfaced(workspace_id: str, *, now: float | None = None) -> bool:
    """True if the workspace surfaced anything within the cooldown window."""
    cutoff = (now if now is not None else time.time()) - SURFACE_COOLDOWN_S
    for ev in _events.read_all(workspace_id):
        if ev.type == _events.EventType.SURFACED and ev.ts >= cutoff:
            return True
    return False


def _send_signal(text: str, workspace_id: str) -> bool:
    """Outbound to Signal via ``conversation_store.enqueue_outbound``.

    Recipient resolution priority:
      1. ``CompanionConfig.signal_recipient`` for the workspace
      2. ``COMPANION_SIGNAL_RECIPIENT`` env var (operator default)
      3. None — falls back to logging the would-send text so operators
         can verify formatting before configuring outbound.

    Returns True iff the message was queued. Failures (recipient
    unset, queue insert error, conversation_store unavailable) are
    logged + swallowed so the SURFACED event still records the attempt
    (cooldown counts; operator can retry once delivery is configured).
    """
    recipient = _resolve_recipient(workspace_id)
    if not recipient:
        logger.info(
            "companion.surfacing: would send to Signal (workspace=%s; "
            "set COMPANION_SIGNAL_RECIPIENT or workspace.signal_recipient "
            "to enable):\n%s",
            workspace_id, text[:500],
        )
        return False
    try:
        return _enqueue_outbound(recipient, text)
    except Exception as exc:
        logger.warning(
            "companion.surfacing: enqueue_outbound raised for %s: %s",
            workspace_id, exc,
        )
        return False


def _resolve_recipient(workspace_id: str) -> str | None:
    """Per-workspace override → env-var operator default → None."""
    try:
        from app.companion.config import load
        cfg = load(workspace_id)
        if cfg is not None:
            cfg_recipient = getattr(cfg, "signal_recipient", None)
            if cfg_recipient:
                return str(cfg_recipient)
    except Exception:
        pass
    import os
    env_recipient = os.environ.get("COMPANION_SIGNAL_RECIPIENT")
    return env_recipient or None


def _enqueue_outbound(recipient: str, text: str) -> bool:
    """Indirection over ``conversation_store.enqueue_outbound`` for testability.

    Returns True iff the outbound queue accepted the row.
    """
    from app.conversation_store import enqueue_outbound
    queue_id = enqueue_outbound(recipient, message=text)
    return queue_id is not None
