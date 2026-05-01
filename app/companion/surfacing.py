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
    if not (idea.text or "").strip():
        return SurfaceDecision(False, "no_text")
    if idea.novelty < config.novelty_threshold:
        return SurfaceDecision(False, "below_novelty")
    if idea.quality < config.surface_threshold:
        return SurfaceDecision(False, "below_quality")
    if idea.panel_score < config.panel_threshold:
        return SurfaceDecision(False, "below_panel")
    if _recently_surfaced(idea.workspace_id, now=now):
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
    """Stub for Signal outbound send. Phase 4.5 wires real enqueue.

    Returns True if the message was queued for delivery, False otherwise.
    Phase 4 just logs at INFO so operators can verify formatting; the
    real wire-up uses ``conversation_store.enqueue_outbound`` once we
    know the per-workspace recipient (Phase 4.5).
    """
    logger.info(
        "companion.surfacing: would send to Signal (workspace=%s):\n%s",
        workspace_id, text,
    )
    return True
