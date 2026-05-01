"""FastAPI router for the Epistemic Integrity dashboard.

Exposes read-only endpoints under ``/epistemic/*`` for the React pane.
Mirrors the structure of :mod:`app.affect.api`.

Mounted in ``app/main.py`` via::

    from app.epistemic.api import router as epistemic_router
    app.include_router(epistemic_router)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Query

from app.epistemic.biases import BIAS_LIBRARY
from app.epistemic.span_writer import (
    list_bias_matches_for_task,
    list_recent_bias_matches,
    list_recent_incidents,
    list_recent_overrides,
    list_recent_peer_reviews,
    list_recent_pushback_events,
    list_tuning_proposals,
    load_incident,
    load_ledger_for_task,
    lookup_claim,
    lookup_tuning_proposal,
    override_aggregates,
    peer_review_aggregates,
    pushback_aggregates,
    update_tuning_proposal_status,
)
from app.epistemic.verification import VERIFIER_REGISTRY

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/epistemic", tags=["epistemic"])


@router.get("/now")
async def epistemic_now(task_id: str | None = None) -> dict:
    """Current ledger snapshot + affect grounding tile.

    If ``task_id`` is given, returns the full ledger for that task plus
    aggregate counts. If absent, returns aggregate counts and a
    sentinel ``ledger=null``. The React pane uses the latter when no
    task is selected.

    The ``calibration`` block carries the live ``factual_grounding``
    signal (from :mod:`app.epistemic.affect_bridge`) when the affect
    layer is wired; otherwise it carries ``null`` and the React tile
    renders "no grounding signal".
    """
    calibration = _calibration_snapshot()

    if task_id is None:
        return {
            "task_id": None,
            "ledger": None,
            "load_bearing_count": 0,
            "unverified_load_bearing_count": 0,
            "bias_match_count": 0,
            "calibration": calibration,
        }

    ledger = load_ledger_for_task(task_id)
    matches = list_bias_matches_for_task(task_id)
    return {
        "task_id": task_id,
        "ledger": [c.as_jsonable() for c in ledger.all()],
        "load_bearing_count": len(ledger.load_bearing()),
        "unverified_load_bearing_count": len(ledger.unverified_load_bearing()),
        "bias_match_count": len(matches),
        "calibration": calibration,
    }


def _calibration_snapshot() -> dict:
    """Read the live affect snapshot for the React calibration tile.

    Returns ``factual_grounding=None`` if the affect layer isn't
    wired — the tile then renders "no grounding signal" rather than
    showing a misleading zero.
    """
    try:
        from app.affect.core import latest_affect
        from app.epistemic.affect_bridge import compute_factual_grounding
    except ImportError:
        return {
            "factual_grounding": None,
            "valence": None,
            "arousal": None,
            "attractor": None,
        }
    try:
        state = latest_affect()
    except Exception:
        state = None
    if state is None:
        return {
            "factual_grounding": None,
            "valence": None,
            "arousal": None,
            "attractor": None,
        }
    return {
        "factual_grounding": round(compute_factual_grounding(state), 3),
        "valence": round(float(state.valence), 3),
        "arousal": round(float(state.arousal), 3),
        "attractor": state.attractor,
    }


@router.get("/feed")
async def epistemic_feed(
    window_min: int = Query(60, ge=1, le=1440),
    limit: int = Query(200, ge=1, le=1000),
) -> dict:
    """Recent bias matches across all tasks, newest first."""
    matches = list_recent_bias_matches(
        window_minutes=window_min, limit=limit,
    )
    return {
        "window_minutes": window_min,
        "count": len(matches),
        "matches": matches,
    }


@router.get("/claim/{claim_id}")
async def epistemic_claim(claim_id: str) -> dict:
    """Single-claim drill-in (evidence, verifier, lineage)."""
    claim = lookup_claim(claim_id)
    if claim is None:
        raise HTTPException(404, f"claim {claim_id!r} not found")
    return claim.as_jsonable()


@router.get("/biases")
async def epistemic_biases() -> dict:
    """The cognitive bias library — Phase 1 ships one entry."""
    return {
        "biases": [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "severity": d.severity.value,
                "phase": d.phase.value,
                "corrective_action": d.corrective_action,
                "blocking": d.blocking,
            }
            for d in BIAS_LIBRARY.all()
        ],
    }


@router.get("/pushback/stats")
async def epistemic_pushback_stats(
    window_min: int = Query(1440, ge=1, le=10080),
) -> dict:
    """Aggregate pushback counts (reverified / falsified / unverifiable)
    plus mean time-to-recheck across the window."""
    return pushback_aggregates(window_minutes=window_min)


@router.get("/pushback/recent")
async def epistemic_pushback_recent(
    window_min: int = Query(1440, ge=1, le=10080),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    """Recent pushback events, newest first."""
    events = list_recent_pushback_events(
        window_minutes=window_min, limit=limit,
    )
    return {
        "window_minutes": window_min,
        "count": len(events),
        "events": events,
    }


@router.get("/overrides/stats")
async def epistemic_override_stats(
    window_min: int = Query(1440, ge=1, le=10080),
) -> dict:
    """Override action counts (force_proceed/use_revision/abandon).

    Operators read this to decide when to flip blocking-mode on. A
    high force_proceed rate means the calibration gate has too many
    false positives and the bias library needs tuning down.
    """
    return override_aggregates(window_minutes=window_min)


@router.get("/overrides/recent")
async def epistemic_overrides_recent(
    window_min: int = Query(1440, ge=1, le=10080),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    """Recent override events with the user's stated reasoning."""
    overrides = list_recent_overrides(
        window_minutes=window_min, limit=limit,
    )
    return {
        "window_minutes": window_min,
        "count": len(overrides),
        "overrides": overrides,
    }


@router.post("/overrides")
async def epistemic_record_override(
    payload: dict = Body(...),
) -> dict:
    """Record a user override of an epistemic-gate verdict.

    Body shape:
      {
        "task_id": str,
        "blocked_action": "block" | "revise",
        "user_action": "force_proceed" | "use_revision" | "abandon",
        "user_reasoning": str,
        "peer_review_id": int | null,    # optional
        "flush_to_self_improver": bool   # optional, default true
      }

    The override is persisted regardless and (by default) flushed to
    the Self-Improver as a USER_CORRECTION LearningGap.
    """
    from app.epistemic.override import OverrideAction, record_override

    try:
        user_action = OverrideAction(payload["user_action"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, f"invalid user_action: {exc}") from exc

    blocked_action = payload.get("blocked_action")
    if blocked_action not in ("block", "revise"):
        raise HTTPException(
            400,
            f"blocked_action must be 'block' or 'revise', got {blocked_action!r}",
        )

    task_id = payload.get("task_id")
    if not task_id:
        raise HTTPException(400, "task_id is required")

    event = record_override(
        task_id=task_id,
        blocked_action=blocked_action,
        user_action=user_action,
        user_reasoning=str(payload.get("user_reasoning", "")),
        peer_review_id=payload.get("peer_review_id"),
        flush_to_self_improver=bool(payload.get("flush_to_self_improver", True)),
    )
    return event.as_jsonable()


@router.get("/tuning/proposals")
async def epistemic_tuning_proposals(
    status: str | None = Query("proposed"),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Open tuning proposals from the autotuner.

    By default returns ``status=proposed`` (the operator's queue).
    Pass ``status=null`` (URL-encoded as empty string) for the full
    history including accepted / rejected / superseded.
    """
    proposals = list_tuning_proposals(status=status, limit=limit)
    return {
        "status_filter": status,
        "count": len(proposals),
        "proposals": proposals,
    }


@router.get("/tuning/proposals/{proposal_id}")
async def epistemic_tuning_proposal(proposal_id: str) -> dict:
    detail = lookup_tuning_proposal(proposal_id)
    if detail is None:
        raise HTTPException(404, f"proposal {proposal_id!r} not found")
    return detail


@router.post("/tuning/proposals/{proposal_id}/accept")
async def epistemic_tuning_accept(
    proposal_id: str,
    payload: dict = Body(default={}),
) -> dict:
    """Mark a proposal accepted. Does NOT auto-apply the YAML — the
    operator opens a CODEOWNERS PR using the patch text in the body."""
    note = str(payload.get("operator_note", ""))
    ok = update_tuning_proposal_status(
        proposal_id=proposal_id, status="accepted", operator_note=note,
    )
    if not ok:
        raise HTTPException(500, "failed to update proposal status")
    return {"proposal_id": proposal_id, "status": "accepted"}


@router.post("/tuning/proposals/{proposal_id}/reject")
async def epistemic_tuning_reject(
    proposal_id: str,
    payload: dict = Body(default={}),
) -> dict:
    """Mark a proposal rejected with the operator's reasoning."""
    note = str(payload.get("operator_note", ""))
    ok = update_tuning_proposal_status(
        proposal_id=proposal_id, status="rejected", operator_note=note,
    )
    if not ok:
        raise HTTPException(500, "failed to update proposal status")
    return {"proposal_id": proposal_id, "status": "rejected"}


@router.post("/tuning/run")
async def epistemic_tuning_run(
    payload: dict = Body(default={}),
) -> dict:
    """Trigger a fresh autotune analysis. Persists new proposals.

    Idempotent: re-running over the same evidence refreshes existing
    proposals (UPSERT on content_hash) rather than duplicating them.
    """
    from app.epistemic.autotune import run_full_analysis
    window_days = int(payload.get("window_days", 7))
    proposals = run_full_analysis(window_days=window_days, persist=True)
    return {
        "window_days": window_days,
        "proposal_count": len(proposals),
        "proposals": [p.as_jsonable() for p in proposals],
    }


@router.get("/peer-reviews/stats")
async def epistemic_peer_review_stats(
    window_min: int = Query(1440, ge=1, le=10080),
) -> dict:
    """Allow / revise / veto aggregates + mean duration."""
    return peer_review_aggregates(window_minutes=window_min)


@router.get("/peer-reviews/recent")
async def epistemic_peer_reviews_recent(
    window_min: int = Query(1440, ge=1, le=10080),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    """Recent peer-review events, newest first."""
    reviews = list_recent_peer_reviews(
        window_minutes=window_min, limit=limit,
    )
    return {
        "window_minutes": window_min,
        "count": len(reviews),
        "reviews": reviews,
    }


@router.get("/incidents")
async def epistemic_incidents(
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    """Recent incidents (top-level fields only). Drill in via
    :func:`epistemic_incident` for the full timeline."""
    incidents = list_recent_incidents(limit=limit)
    return {"count": len(incidents), "incidents": incidents}


@router.get("/incidents/{incident_id}")
async def epistemic_incident(incident_id: str) -> dict:
    """Full IncidentReport: timeline, root cause, enabling factors,
    behavioral changes, missed signals."""
    detail = load_incident(incident_id)
    if detail is None:
        raise HTTPException(404, f"incident {incident_id!r} not found")
    return detail


@router.get("/verifiers")
async def epistemic_verifiers() -> dict:
    """The verifier registry — claim shapes the system can settle cheaply."""
    return {
        "verifiers": [
            {
                "id": s.id,
                "tool": s.tool,
                "expected_signal": s.expected_signal,
                "estimated_seconds": s.estimated_seconds,
                "tags_any": list(s.tags_any),
            }
            for s in VERIFIER_REGISTRY()
        ],
    }
