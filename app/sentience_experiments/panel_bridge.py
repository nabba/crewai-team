"""Bridge: philosophy decision panel → Q4.1 tensions store.

PROGRAM §43.1 — Q5.1. When ``philosophy.dialectics.consult_panel``
returns unresolved tensions (perspectives present but synthesis
absent), those are exactly the open questions the operator should
see in their daily briefing. The Q4.1 tensions store is the right
home for them — it already has OPEN/DORMANT/RESOLVED lifecycle,
freshness decay, and a daily-briefing surface.

This bridge:

  * takes a ``PanelResult``
  * for each unresolved tension, files an OPEN tension via
    ``app/companion/tensions.py:create_tension``
  * tags the source so the operator can trace it back to the
    originating amendment / identity claim / calibration decision

Design notes
------------

  * One tension per ``unresolved_tensions`` entry — the wording the
    panel produced is already the question shape.
  * The OPEN cap (30) in the tensions store is intact — if it's
    full, the bridge returns silently (panel result still persists
    in the proposal evidence).
  * Bridge is **observational from the proposer's perspective**:
    the bridge fails open (never raises). The proposer's path is
    unaffected by tension-store failures.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Cap on how many tensions a single panel consultation can file. The
# tensions store has its own 30-OPEN cap, but we don't want one
# Tier-3 amendment proposal to consume the operator's whole tension
# surface even when the cap isn't hit.
_MAX_TENSIONS_PER_CONSULT = 3


def file_unresolved_tensions(
    panel: Any,                # PanelResult — duck-typed to avoid import cycles
    *,
    source_kind: str,          # "tier3_amendment" | "identity_claim" | "calibration"
    source_ref: str,           # plan_id | claim_id | calibration_run_id
    workspace_id: str | None = None,
) -> list[str]:
    """File each unresolved panel tension as an OPEN tension.

    Returns the list of tension IDs filed. Empty list on no-op or
    failure. Never raises.

    The panel is duck-typed (any object with ``unresolved_tensions`` +
    ``question`` attributes) so callers can pass either a
    ``PanelResult`` dataclass or a dict deserialized from JSON.
    """
    if panel is None:
        return []
    # Duck-type: support both PanelResult dataclasses and dict payloads.
    # Try dict access first (most common in re-hydrated proposal evidence);
    # fall back to attribute access (live PanelResult objects).
    if isinstance(panel, dict):
        unresolved = list(panel.get("unresolved_tensions") or [])
        question = panel.get("question") or ""
    else:
        try:
            unresolved = list(getattr(panel, "unresolved_tensions", None) or [])
            question = getattr(panel, "question", "") or ""
        except Exception:
            return []

    if not unresolved or not question.strip():
        return []

    try:
        from app.companion.tensions import (
            TensionSource,
            create_tension,
            list_tensions,
            STATUS_OPEN,
        )
    except Exception:
        logger.debug("panel_bridge: tensions store import failed",
                     exc_info=True)
        return []

    # Skip filing if the same question already has an open tension
    # from a recent panel consultation — avoids duplicating the same
    # philosophical question across multiple amendment proposals.
    try:
        existing_questions = {
            (t.question or "").strip().lower()
            for t in list_tensions(status=STATUS_OPEN)
        }
    except Exception:
        existing_questions = set()

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    snippet = f"Philosophy panel ({source_kind}={source_ref[:60]}): {question[:120]}"

    filed_ids: list[str] = []
    for utext in unresolved[:_MAX_TENSIONS_PER_CONSULT]:
        # Compose a tension question from the panel's unresolved
        # description. The panel produces strings like
        # "Stoicism: claim present but no synthesis with counter".
        # Make that into an actual question the operator can act on.
        tension_q = _phrase_as_question(utext, question)
        if not tension_q:
            continue
        if tension_q.lower() in existing_questions:
            logger.debug(
                "panel_bridge: skipping duplicate-question tension %r",
                tension_q[:60],
            )
            continue
        try:
            t = create_tension(
                question=tension_q,
                sources=[TensionSource(
                    kind="philosophy_panel",
                    ts=now,
                    snippet=snippet,
                )],
                workspace_id=workspace_id,
                detection_source=f"panel:{source_kind}",
            )
        except Exception:
            logger.debug(
                "panel_bridge: create_tension raised for %r",
                tension_q[:60], exc_info=True,
            )
            continue
        if t is not None:
            filed_ids.append(t.id)
    if filed_ids:
        logger.info(
            "panel_bridge: filed %d unresolved tensions from %s=%s",
            len(filed_ids), source_kind, source_ref[:40],
        )
    return filed_ids


def _phrase_as_question(unresolved_text: str, original_question: str) -> str:
    """Convert a panel's unresolved-tension description into an
    operator-facing question. Keeps the original question's framing
    when possible; falls back to wrapping the unresolved text."""
    ut = (unresolved_text or "").strip()
    if not ut:
        return ""
    oq = (original_question or "").strip().rstrip("?.!")
    # If the unresolved text is already shaped as a question, keep it.
    if ut.endswith("?"):
        return ut[:300]
    # Heuristic: panel emits "Tradition: <detail>" — pull the tradition
    # name and make a question of the form "On <oq>, what does
    # <tradition> contribute that no synthesis answers yet?"
    if ":" in ut and oq:
        trad, detail = ut.split(":", 1)
        trad = trad.strip()
        detail = detail.strip()
        if trad and detail:
            q = (
                f"On {oq}: what does {trad} contribute that no "
                f"synthesis resolves yet? ({detail[:120]})"
            )
            return q[:300]
    # Fallback — wrap as a question.
    if oq:
        return f"Unresolved philosophical tension on {oq}: {ut[:160]}"[:300]
    return f"Unresolved philosophical tension: {ut[:200]}"[:300]
