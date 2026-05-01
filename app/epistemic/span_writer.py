"""PostgreSQL bridge for the Claim Ledger.

Persists claims into ``control_plane.epistemic_claims`` (migration 026).
Read path reconstructs a :class:`~app.epistemic.ledger.Ledger` from the
table, which is how the React ``/epistemic`` pane (Phase 1+) and the
post-mortem pipeline (Phase 4) consume historical claims.

Write semantics
---------------

* :func:`persist_claim` is an UPSERT keyed on ``claim_id``. New claims
  INSERT; supersession updates ``status`` + ``superseded_by``. Idempotent.
* All writes are fire-and-forget — DB failures are logged at DEBUG and
  never propagate. This matches ``app.control_plane.crew_task_spans``:
  telemetry must never break the agent's user-facing path.
* The gate is ``app.epistemic.is_enabled()``. If the layer is disabled,
  writes are skipped entirely (no DB connection acquired).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.control_plane.db import execute, execute_one
from app.epistemic import is_enabled
from typing import TYPE_CHECKING

from app.epistemic.biases import BiasMatch
from app.epistemic.ledger import Claim, Ledger
from app.epistemic.pushback import ContradictionSignal, FoundationCheckResult

if TYPE_CHECKING:
    # Lazy imports to avoid cycles:
    # postmortem.synthesize_report calls span_writer functions; both
    # peer_review and override import span_writer at call time;
    # autotune.run_full_analysis calls back through here.
    from app.epistemic.autotune import TuningProposal
    from app.epistemic.override import OverrideEvent
    from app.epistemic.peer_review import PeerReviewVerdict
    from app.epistemic.postmortem import IncidentReport

logger = logging.getLogger(__name__)


# ── Write path ───────────────────────────────────────────────────────

def persist_claim(claim: Claim) -> None:
    """UPSERT a claim into ``control_plane.epistemic_claims``.

    No-op if the layer is disabled. Errors are swallowed at DEBUG —
    callers (the Ledger) must not be coupled to telemetry.
    """
    if not is_enabled():
        return
    try:
        execute(
            """
            INSERT INTO control_plane.epistemic_claims
                   (claim_id, task_id, span_id, agent_role, statement,
                    status, register, evidence, verifying_action,
                    load_bearing, tags, superseded_by, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                    %s, %s::jsonb, %s, %s)
            ON CONFLICT (claim_id) DO UPDATE SET
                status        = EXCLUDED.status,
                superseded_by = EXCLUDED.superseded_by,
                evidence      = EXCLUDED.evidence,
                tags          = EXCLUDED.tags,
                load_bearing  = EXCLUDED.load_bearing
            """,
            (
                claim.claim_id,
                claim.task_id,
                claim.span_id,
                claim.agent_role,
                claim.statement,
                claim.status.value,
                claim.register.value,
                json.dumps(_evidence_jsonable(claim)),
                json.dumps(_verifier_jsonable(claim)) if claim.verifying_action else None,
                claim.load_bearing,
                json.dumps(list(claim.tags)),
                claim.superseded_by,
                claim.created_at,
            ),
        )
    except Exception as exc:
        logger.debug("epistemic span_writer.persist_claim failed: %s", exc)


# ── Read path ────────────────────────────────────────────────────────

def load_ledger_for_task(task_id: str) -> Ledger:
    """Reconstruct a Ledger from the database.

    Returns an empty Ledger if the layer is disabled, the task has no
    claims, or the read fails. The Ledger is consistent regardless —
    callers can always iterate ``ledger.all()``.
    """
    if not is_enabled():
        return Ledger(task_id=task_id)
    try:
        rows = execute(
            """
            SELECT claim_id, task_id, span_id, agent_role, statement,
                   status, register, evidence, verifying_action,
                   load_bearing, tags, superseded_by, created_at
              FROM control_plane.epistemic_claims
             WHERE task_id = %s
          ORDER BY created_at ASC, claim_id ASC
            """,
            (task_id,),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug("epistemic span_writer.load_ledger_for_task failed: %s", exc)
        return Ledger(task_id=task_id)

    claims = [Claim.from_jsonable(_row_to_jsonable(r)) for r in rows]
    return Ledger.from_claims(task_id=task_id, claims=claims)


def persist_bias_matches(
    *,
    claim_id: str,
    task_id: str,
    matches: list[BiasMatch],
) -> None:
    """Persist a batch of bias matches into ``epistemic_bias_matches``.

    Bulk insert via ``execute_values`` would be cleaner; for the typical
    case of 0–2 matches per claim we just loop. No-op if disabled or if
    the matches list is empty.
    """
    if not is_enabled() or not matches:
        return
    for match in matches:
        try:
            execute(
                """
                INSERT INTO control_plane.epistemic_bias_matches
                       (task_id, claim_id, bias_id, severity,
                        matched_claim_ids, detail)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
                """,
                (
                    task_id,
                    claim_id,
                    match.bias_id,
                    match.severity.value,
                    json.dumps(list(match.matched_claim_ids)),
                    json.dumps(dict(match.detail)),
                ),
            )
        except Exception as exc:
            logger.debug(
                "epistemic span_writer.persist_bias_matches failed: %s", exc,
            )


def list_recent_bias_matches(
    *,
    window_minutes: int = 60,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return recent bias matches across all tasks, newest first.

    Used by the React BiasFeed view. Returns rows directly (not
    BiasMatch dataclasses) since the consumer is JSON-serializing them.
    """
    if not is_enabled():
        return []
    try:
        rows = execute(
            """
            SELECT id, task_id, claim_id, bias_id, severity,
                   matched_claim_ids, detail, detected_at
              FROM control_plane.epistemic_bias_matches
             WHERE detected_at >= NOW() - (%s || ' minutes')::interval
          ORDER BY detected_at DESC, id DESC
             LIMIT %s
            """,
            (str(int(window_minutes)), int(limit)),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.list_recent_bias_matches failed: %s", exc,
        )
        return []
    return [
        {
            "id": r["id"],
            "task_id": r["task_id"],
            "claim_id": r["claim_id"],
            "bias_id": r["bias_id"],
            "severity": r["severity"],
            "matched_claim_ids": r["matched_claim_ids"] or [],
            "detail": r["detail"] or {},
            "detected_at": r["detected_at"].isoformat()
                           if hasattr(r["detected_at"], "isoformat")
                           else r["detected_at"],
        }
        for r in rows
    ]


def list_bias_matches_for_task(task_id: str) -> list[dict[str, Any]]:
    """Return every bias match for a given task, oldest first.

    Used by the calibration check at end-of-task and by post-mortem
    (Phase 4)."""
    if not is_enabled():
        return []
    try:
        rows = execute(
            """
            SELECT id, task_id, claim_id, bias_id, severity,
                   matched_claim_ids, detail, detected_at
              FROM control_plane.epistemic_bias_matches
             WHERE task_id = %s
          ORDER BY detected_at ASC, id ASC
            """,
            (task_id,),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.list_bias_matches_for_task failed: %s", exc,
        )
        return []
    return [
        {
            "id": r["id"],
            "task_id": r["task_id"],
            "claim_id": r["claim_id"],
            "bias_id": r["bias_id"],
            "severity": r["severity"],
            "matched_claim_ids": r["matched_claim_ids"] or [],
            "detail": r["detail"] or {},
            "detected_at": r["detected_at"].isoformat()
                           if hasattr(r["detected_at"], "isoformat")
                           else r["detected_at"],
        }
        for r in rows
    ]


def persist_pushback_event(
    *,
    task_id: str,
    signal: ContradictionSignal,
    check: FoundationCheckResult,
) -> None:
    """Persist one row to ``epistemic_pushback_events``.

    Fire-and-forget. The pushback handler's user-facing path must not
    couple to telemetry.
    """
    if not is_enabled():
        return
    try:
        execute(
            """
            INSERT INTO control_plane.epistemic_pushback_events
                   (task_id, contradicted_claim_id, user_evidence,
                    confidence, detector, outcome, new_evidence_excerpt,
                    invalidated_claim_ids, duration_seconds, detected_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            """,
            (
                task_id,
                signal.contradicted_claim_id,
                signal.user_evidence,
                signal.confidence,
                signal.detector,
                check.outcome.value,
                check.new_evidence_excerpt,
                json.dumps(list(check.invalidated_claim_ids)),
                check.duration_seconds,
                signal.detected_at,
            ),
        )
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.persist_pushback_event failed: %s", exc,
        )


def list_recent_pushback_events(
    *,
    window_minutes: int = 1440,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return recent pushback events across all tasks, newest first."""
    if not is_enabled():
        return []
    try:
        rows = execute(
            """
            SELECT id, task_id, contradicted_claim_id, user_evidence,
                   confidence, detector, outcome, new_evidence_excerpt,
                   invalidated_claim_ids, duration_seconds, detected_at
              FROM control_plane.epistemic_pushback_events
             WHERE detected_at >= NOW() - (%s || ' minutes')::interval
          ORDER BY detected_at DESC, id DESC
             LIMIT %s
            """,
            (str(int(window_minutes)), int(limit)),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.list_recent_pushback_events failed: %s", exc,
        )
        return []
    return [_pushback_row_to_jsonable(r) for r in rows]


def pushback_aggregates(
    *,
    window_minutes: int = 1440,
) -> dict[str, Any]:
    """Aggregate counts + mean duration for the React Pushback panel."""
    if not is_enabled():
        return _empty_aggregates()
    try:
        rows = execute(
            """
            SELECT outcome, COUNT(*) AS n,
                   AVG(duration_seconds) AS mean_seconds
              FROM control_plane.epistemic_pushback_events
             WHERE detected_at >= NOW() - (%s || ' minutes')::interval
          GROUP BY outcome
            """,
            (str(int(window_minutes)),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.pushback_aggregates failed: %s", exc,
        )
        return _empty_aggregates()

    by_outcome = {r["outcome"]: r for r in rows}
    total = sum(int(r["n"]) for r in rows)
    weighted_mean = (
        sum(int(r["n"]) * float(r["mean_seconds"] or 0.0) for r in rows) / total
        if total > 0
        else 0.0
    )
    return {
        "window_minutes": window_minutes,
        "total": total,
        "reverified": int(by_outcome.get("reverified", {}).get("n", 0)),
        "falsified": int(by_outcome.get("falsified", {}).get("n", 0)),
        "unverifiable": int(by_outcome.get("unverifiable", {}).get("n", 0)),
        "mean_seconds_to_recheck": round(weighted_mean, 3),
    }


def _empty_aggregates() -> dict[str, Any]:
    return {
        "window_minutes": 0,
        "total": 0,
        "reverified": 0,
        "falsified": 0,
        "unverifiable": 0,
        "mean_seconds_to_recheck": 0.0,
    }


def _pushback_row_to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
    detected = row["detected_at"]
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "contradicted_claim_id": row["contradicted_claim_id"],
        "user_evidence": row["user_evidence"],
        "confidence": float(row["confidence"]),
        "detector": row["detector"],
        "outcome": row["outcome"],
        "new_evidence_excerpt": row["new_evidence_excerpt"],
        "invalidated_claim_ids": row["invalidated_claim_ids"] or [],
        "duration_seconds": float(row["duration_seconds"]),
        "detected_at": detected.isoformat() if hasattr(detected, "isoformat") else detected,
    }


def list_pushback_events_for_task(task_id: str) -> list[dict[str, Any]]:
    """All pushback events for a task, oldest first.

    Used by the post-mortem pipeline. Mirrors
    :func:`list_recent_pushback_events` but scoped to one task and
    returned in chronological order (timeline construction).
    """
    if not is_enabled():
        return []
    try:
        rows = execute(
            """
            SELECT id, task_id, contradicted_claim_id, user_evidence,
                   confidence, detector, outcome, new_evidence_excerpt,
                   invalidated_claim_ids, duration_seconds, detected_at
              FROM control_plane.epistemic_pushback_events
             WHERE task_id = %s
          ORDER BY detected_at ASC, id ASC
            """,
            (task_id,),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.list_pushback_events_for_task failed: %s",
            exc,
        )
        return []
    return [_pushback_row_to_jsonable(r) for r in rows]


def persist_incident(report: "IncidentReport") -> bool:
    """Persist an IncidentReport into ``epistemic_incidents``.

    Returns ``True`` if the row was inserted, ``False`` otherwise.
    Fire-and-forget on the post-mortem path — failures don't propagate.
    """
    if not is_enabled():
        return False
    try:
        execute(
            """
            INSERT INTO control_plane.epistemic_incidents
                   (incident_id, task_id, root_cause_bias_id, severity,
                    report, self_improver_emitted, created_at)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (incident_id) DO NOTHING
            """,
            (
                report.incident_id,
                report.task_id,
                report.root_cause.bias_id,
                report.severity.value,
                json.dumps(report.as_jsonable()),
                False,
                report.created_at,
            ),
        )
        return True
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.persist_incident failed: %s", exc,
        )
        return False


def mark_incident_emitted(incident_id: str) -> None:
    """Flip ``self_improver_emitted=TRUE`` after a successful flush.

    Called by the post-mortem pipeline once :func:`emit_to_self_improver`
    returns True. Idempotent — repeated calls are no-ops.
    """
    if not is_enabled():
        return
    try:
        execute(
            """
            UPDATE control_plane.epistemic_incidents
               SET self_improver_emitted = TRUE
             WHERE incident_id = %s
            """,
            (incident_id,),
        )
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.mark_incident_emitted failed: %s", exc,
        )


def list_recent_incidents(
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Recent incidents (top-level fields only). Newest first."""
    if not is_enabled():
        return []
    try:
        rows = execute(
            """
            SELECT incident_id, task_id, root_cause_bias_id, severity,
                   self_improver_emitted, created_at
              FROM control_plane.epistemic_incidents
          ORDER BY created_at DESC
             LIMIT %s
            """,
            (int(limit),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.list_recent_incidents failed: %s", exc,
        )
        return []
    return [
        {
            "incident_id": r["incident_id"],
            "task_id": r["task_id"],
            "root_cause_bias_id": r["root_cause_bias_id"],
            "severity": r["severity"],
            "self_improver_emitted": bool(r["self_improver_emitted"]),
            "created_at": r["created_at"].isoformat()
                          if hasattr(r["created_at"], "isoformat")
                          else r["created_at"],
        }
        for r in rows
    ]


def load_incident(incident_id: str) -> dict[str, Any] | None:
    """Full IncidentReport JSON by id, or ``None`` if not found."""
    if not is_enabled():
        return None
    try:
        row = execute_one(
            """
            SELECT incident_id, task_id, root_cause_bias_id, severity,
                   report, self_improver_emitted, created_at
              FROM control_plane.epistemic_incidents
             WHERE incident_id = %s
            """,
            (incident_id,),
        )
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.load_incident failed: %s", exc,
        )
        return None
    if row is None:
        return None
    report = row["report"] or {}
    # Surface the operational flag at the top level so the React panel
    # can render "queued for Self-Improver" without digging into the JSON.
    report["self_improver_emitted"] = bool(row["self_improver_emitted"])
    return report


def persist_peer_review(
    *,
    task_id: str,
    triggering_claim_id: str | None,
    proposal_text: str,
    verdict: "PeerReviewVerdict",
) -> None:
    """Persist one row to ``epistemic_peer_reviews``.

    Fire-and-forget. The peer-review path must not couple to telemetry.
    Long proposal_text is truncated to a 500-char excerpt — the full
    proposal lives in the LLM call's audit trail (Phase 7+).
    """
    if not is_enabled():
        return
    try:
        excerpt = (proposal_text or "")[:500]
        execute(
            """
            INSERT INTO control_plane.epistemic_peer_reviews
                   (task_id, triggering_claim_id, proposal_excerpt,
                    decision, rationale, suggested_revision, reviewers,
                    duration_seconds, requested_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            """,
            (
                task_id,
                triggering_claim_id,
                excerpt,
                verdict.decision.value,
                verdict.rationale,
                verdict.suggested_revision,
                json.dumps(list(verdict.reviewers)),
                verdict.duration_seconds,
                datetime.now(timezone.utc),
            ),
        )
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.persist_peer_review failed: %s", exc,
        )


def list_recent_peer_reviews(
    *,
    window_minutes: int = 1440,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent peer reviews across all tasks, newest first."""
    if not is_enabled():
        return []
    try:
        rows = execute(
            """
            SELECT id, task_id, triggering_claim_id, proposal_excerpt,
                   decision, rationale, suggested_revision, reviewers,
                   duration_seconds, requested_at
              FROM control_plane.epistemic_peer_reviews
             WHERE requested_at >= NOW() - (%s || ' minutes')::interval
          ORDER BY requested_at DESC, id DESC
             LIMIT %s
            """,
            (str(int(window_minutes)), int(limit)),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.list_recent_peer_reviews failed: %s",
            exc,
        )
        return []
    return [
        {
            "id": r["id"],
            "task_id": r["task_id"],
            "triggering_claim_id": r["triggering_claim_id"],
            "proposal_excerpt": r["proposal_excerpt"],
            "decision": r["decision"],
            "rationale": r["rationale"],
            "suggested_revision": r["suggested_revision"],
            "reviewers": r["reviewers"] or [],
            "duration_seconds": float(r["duration_seconds"]),
            "requested_at": r["requested_at"].isoformat()
                            if hasattr(r["requested_at"], "isoformat")
                            else r["requested_at"],
        }
        for r in rows
    ]


def peer_review_aggregates(
    *,
    window_minutes: int = 1440,
) -> dict[str, Any]:
    """Aggregate counts (allow/revise/veto) + mean duration."""
    if not is_enabled():
        return _empty_pr_aggregates()
    try:
        rows = execute(
            """
            SELECT decision, COUNT(*) AS n,
                   AVG(duration_seconds) AS mean_seconds
              FROM control_plane.epistemic_peer_reviews
             WHERE requested_at >= NOW() - (%s || ' minutes')::interval
          GROUP BY decision
            """,
            (str(int(window_minutes)),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.peer_review_aggregates failed: %s", exc,
        )
        return _empty_pr_aggregates()
    by_decision = {r["decision"]: r for r in rows}
    total = sum(int(r["n"]) for r in rows)
    weighted = (
        sum(int(r["n"]) * float(r["mean_seconds"] or 0.0) for r in rows) / total
        if total > 0 else 0.0
    )
    return {
        "window_minutes": window_minutes,
        "total": total,
        "allow": int(by_decision.get("allow", {}).get("n", 0)),
        "revise": int(by_decision.get("revise", {}).get("n", 0)),
        "veto": int(by_decision.get("veto", {}).get("n", 0)),
        "mean_seconds": round(weighted, 3),
    }


def _empty_pr_aggregates() -> dict[str, Any]:
    return {
        "window_minutes": 0,
        "total": 0,
        "allow": 0,
        "revise": 0,
        "veto": 0,
        "mean_seconds": 0.0,
    }


def persist_override(event: "OverrideEvent") -> None:
    """Persist one row to ``epistemic_overrides``.

    Fire-and-forget. Override capture must not break the
    user-facing forced-proceed path.
    """
    if not is_enabled():
        return
    try:
        execute(
            """
            INSERT INTO control_plane.epistemic_overrides
                   (override_id, task_id, peer_review_id, blocked_action,
                    user_action, user_reasoning, overridden_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (override_id) DO NOTHING
            """,
            (
                event.override_id,
                event.task_id,
                event.peer_review_id,
                event.blocked_action,
                event.user_action.value,
                event.user_reasoning,
                event.overridden_at,
            ),
        )
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.persist_override failed: %s", exc,
        )


def list_recent_overrides(
    *,
    window_minutes: int = 1440,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent overrides across all tasks, newest first."""
    if not is_enabled():
        return []
    try:
        rows = execute(
            """
            SELECT override_id, task_id, peer_review_id, blocked_action,
                   user_action, user_reasoning, overridden_at
              FROM control_plane.epistemic_overrides
             WHERE overridden_at >= NOW() - (%s || ' minutes')::interval
          ORDER BY overridden_at DESC
             LIMIT %s
            """,
            (str(int(window_minutes)), int(limit)),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.list_recent_overrides failed: %s", exc,
        )
        return []
    return [
        {
            "override_id": r["override_id"],
            "task_id": r["task_id"],
            "peer_review_id": r["peer_review_id"],
            "blocked_action": r["blocked_action"],
            "user_action": r["user_action"],
            "user_reasoning": r["user_reasoning"],
            "overridden_at": r["overridden_at"].isoformat()
                             if hasattr(r["overridden_at"], "isoformat")
                             else r["overridden_at"],
        }
        for r in rows
    ]


def override_aggregates(
    *,
    window_minutes: int = 1440,
) -> dict[str, Any]:
    """Counts by user_action — informs the false-positive-rate metric
    that the operator uses to decide whether to flip blocking-mode."""
    if not is_enabled():
        return {
            "window_minutes": 0, "total": 0,
            "force_proceed": 0, "use_revision": 0, "abandon": 0,
        }
    try:
        rows = execute(
            """
            SELECT user_action, COUNT(*) AS n
              FROM control_plane.epistemic_overrides
             WHERE overridden_at >= NOW() - (%s || ' minutes')::interval
          GROUP BY user_action
            """,
            (str(int(window_minutes)),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.override_aggregates failed: %s", exc,
        )
        return {
            "window_minutes": 0, "total": 0,
            "force_proceed": 0, "use_revision": 0, "abandon": 0,
        }
    by_action = {r["user_action"]: int(r["n"]) for r in rows}
    return {
        "window_minutes": window_minutes,
        "total": sum(by_action.values()),
        "force_proceed": by_action.get("force_proceed", 0),
        "use_revision": by_action.get("use_revision", 0),
        "abandon": by_action.get("abandon", 0),
    }


def bias_match_counts(*, window_days: int = 7) -> dict[str, int]:
    """Per-bias fire counts over the window. Used by the autotuner."""
    if not is_enabled():
        return {}
    try:
        rows = execute(
            """
            SELECT bias_id, COUNT(*) AS n
              FROM control_plane.epistemic_bias_matches
             WHERE detected_at >= NOW() - (%s || ' days')::interval
          GROUP BY bias_id
            """,
            (str(int(window_days)),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.bias_match_counts failed: %s", exc,
        )
        return {}
    return {r["bias_id"]: int(r["n"]) for r in rows}


def override_counts_by_bias(
    *, window_days: int = 7,
) -> dict[str, dict[str, int]]:
    """Per-bias override counts.

    Joins overrides to bias matches by ``task_id`` (the override
    happened in the same task that fired the bias). Returns
    ``{bias_id: {total, force_proceed, use_revision, abandon}}``.
    """
    if not is_enabled():
        return {}
    try:
        rows = execute(
            """
            SELECT bm.bias_id, ov.user_action, COUNT(*) AS n
              FROM control_plane.epistemic_overrides ov
              JOIN control_plane.epistemic_bias_matches bm
                ON bm.task_id = ov.task_id
             WHERE ov.overridden_at >= NOW() - (%s || ' days')::interval
          GROUP BY bm.bias_id, ov.user_action
            """,
            (str(int(window_days)),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.override_counts_by_bias failed: %s",
            exc,
        )
        return {}
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        bid = r["bias_id"]
        if bid not in out:
            out[bid] = {"total": 0, "force_proceed": 0,
                        "use_revision": 0, "abandon": 0}
        n = int(r["n"])
        out[bid]["total"] += n
        action = r["user_action"]
        if action in out[bid]:
            out[bid][action] += n
    return out


def peer_review_counts_by_bias(
    *, window_days: int = 7,
) -> dict[str, dict[str, int]]:
    """Per-bias peer-review decision counts.

    Joins peer reviews to the bias matches that triggered them via
    ``triggering_claim_id`` → ``epistemic_bias_matches.claim_id``.
    Returns ``{bias_id: {total, allow, revise, veto}}``.
    """
    if not is_enabled():
        return {}
    try:
        rows = execute(
            """
            SELECT bm.bias_id, pr.decision, COUNT(*) AS n
              FROM control_plane.epistemic_peer_reviews pr
              JOIN control_plane.epistemic_bias_matches bm
                ON bm.claim_id = pr.triggering_claim_id
             WHERE pr.requested_at >= NOW() - (%s || ' days')::interval
          GROUP BY bm.bias_id, pr.decision
            """,
            (str(int(window_days)),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.peer_review_counts_by_bias failed: %s",
            exc,
        )
        return {}
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        bid = r["bias_id"]
        if bid not in out:
            out[bid] = {"total": 0, "allow": 0, "revise": 0, "veto": 0}
        n = int(r["n"])
        out[bid]["total"] += n
        d = r["decision"]
        if d in out[bid]:
            out[bid][d] += n
    return out


def incident_counts_by_root_cause(
    *, window_days: int = 7,
) -> dict[str, int]:
    """Per-bias incident counts where the bias was the root cause."""
    if not is_enabled():
        return {}
    try:
        rows = execute(
            """
            SELECT root_cause_bias_id, COUNT(*) AS n
              FROM control_plane.epistemic_incidents
             WHERE created_at >= NOW() - (%s || ' days')::interval
          GROUP BY root_cause_bias_id
            """,
            (str(int(window_days)),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.incident_counts_by_root_cause failed: %s",
            exc,
        )
        return {}
    return {r["root_cause_bias_id"]: int(r["n"]) for r in rows}


def verifier_match_counts(*, window_days: int = 7) -> dict[str, int]:
    """Per-verifier-shape match counts.

    Approximate: counts how many claims in the window had a non-null
    ``verifying_action`` whose ``tool`` matches each shape's tool head.
    Tool heads are the first whitespace token of ``verifying_action.tool``.
    """
    if not is_enabled():
        return {}
    try:
        rows = execute(
            """
            SELECT split_part(verifying_action->>'tool', ' ', 1) AS tool_head,
                   COUNT(*) AS n
              FROM control_plane.epistemic_claims
             WHERE created_at >= NOW() - (%s || ' days')::interval
               AND verifying_action IS NOT NULL
          GROUP BY tool_head
            """,
            (str(int(window_days)),),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.verifier_match_counts failed: %s",
            exc,
        )
        return {}
    # Map tool_head → shape_id by walking the registry. Multiple
    # shapes may share a tool head; we attribute matches conservatively
    # by giving each match to ALL shapes with that head (the caller
    # treats this as an upper bound for retirement decisions).
    from app.epistemic.verification import VERIFIER_REGISTRY
    out: dict[str, int] = {}
    head_to_n = {r["tool_head"]: int(r["n"]) for r in rows if r["tool_head"]}
    for shape in VERIFIER_REGISTRY():
        head = shape.tool.split()[0] if shape.tool else ""
        out[shape.id] = head_to_n.get(head, 0)
    return out


def persist_tuning_proposal(proposal: "TuningProposal") -> None:
    """UPSERT a tuning proposal keyed on ``content_hash``.

    Re-running the analyzer over the same evidence produces the same
    hash; the row is refreshed in place rather than duplicated. This
    keeps the proposals table from growing unboundedly across re-runs.
    """
    if not is_enabled():
        return
    try:
        execute(
            """
            INSERT INTO control_plane.epistemic_tuning_proposals
                   (proposal_id, content_hash, target_kind, target_id,
                    kind, rationale, metric_evidence, yaml_patch,
                    confidence, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s,
                    'proposed', %s, %s)
            ON CONFLICT (content_hash) DO UPDATE SET
                metric_evidence = EXCLUDED.metric_evidence,
                confidence      = EXCLUDED.confidence,
                rationale       = EXCLUDED.rationale,
                updated_at      = EXCLUDED.updated_at
            """,
            (
                proposal.proposal_id,
                proposal.content_hash,
                proposal.target_kind,
                proposal.target_id,
                proposal.kind.value,
                proposal.rationale,
                json.dumps(dict(proposal.metric_evidence)),
                proposal.yaml_patch,
                proposal.confidence,
                proposal.created_at,
                datetime.now(timezone.utc),
            ),
        )
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.persist_tuning_proposal failed: %s",
            exc,
        )


def list_tuning_proposals(
    *,
    status: str | None = "proposed",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return tuning proposals filtered by status.

    Default is ``status="proposed"`` — the open queue the operator
    reviews. Pass ``status=None`` for everything; pass an explicit
    string for ``accepted``/``rejected``/``superseded``.
    """
    if not is_enabled():
        return []
    try:
        if status is None:
            rows = execute(
                """
                SELECT proposal_id, content_hash, target_kind, target_id,
                       kind, rationale, metric_evidence, yaml_patch,
                       confidence, status, operator_note,
                       created_at, updated_at
                  FROM control_plane.epistemic_tuning_proposals
              ORDER BY created_at DESC
                 LIMIT %s
                """,
                (int(limit),),
                fetch=True,
            ) or []
        else:
            rows = execute(
                """
                SELECT proposal_id, content_hash, target_kind, target_id,
                       kind, rationale, metric_evidence, yaml_patch,
                       confidence, status, operator_note,
                       created_at, updated_at
                  FROM control_plane.epistemic_tuning_proposals
                 WHERE status = %s
              ORDER BY created_at DESC
                 LIMIT %s
                """,
                (status, int(limit)),
                fetch=True,
            ) or []
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.list_tuning_proposals failed: %s", exc,
        )
        return []
    return [_proposal_row_to_jsonable(r) for r in rows]


def lookup_tuning_proposal(proposal_id: str) -> dict[str, Any] | None:
    if not is_enabled():
        return None
    try:
        row = execute_one(
            """
            SELECT proposal_id, content_hash, target_kind, target_id,
                   kind, rationale, metric_evidence, yaml_patch,
                   confidence, status, operator_note,
                   created_at, updated_at
              FROM control_plane.epistemic_tuning_proposals
             WHERE proposal_id = %s
            """,
            (proposal_id,),
        )
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.lookup_tuning_proposal failed: %s",
            exc,
        )
        return None
    if row is None:
        return None
    return _proposal_row_to_jsonable(row)


def update_tuning_proposal_status(
    *,
    proposal_id: str,
    status: str,
    operator_note: str = "",
) -> bool:
    """Flip a proposal's status. Returns True on success."""
    if not is_enabled():
        return False
    if status not in ("proposed", "accepted", "rejected", "superseded"):
        raise ValueError(f"invalid status {status!r}")
    try:
        execute(
            """
            UPDATE control_plane.epistemic_tuning_proposals
               SET status        = %s,
                   operator_note = %s,
                   updated_at    = %s
             WHERE proposal_id   = %s
            """,
            (status, operator_note, datetime.now(timezone.utc), proposal_id),
        )
        return True
    except Exception as exc:
        logger.debug(
            "epistemic span_writer.update_tuning_proposal_status failed: %s",
            exc,
        )
        return False


def _proposal_row_to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": row["proposal_id"],
        "content_hash": row["content_hash"],
        "target_kind": row["target_kind"],
        "target_id": row["target_id"],
        "kind": row["kind"],
        "rationale": row["rationale"],
        "metric_evidence": row["metric_evidence"] or {},
        "yaml_patch": row["yaml_patch"],
        "confidence": float(row["confidence"]),
        "status": row["status"],
        "operator_note": row["operator_note"],
        "created_at": row["created_at"].isoformat()
                      if hasattr(row["created_at"], "isoformat")
                      else row["created_at"],
        "updated_at": row["updated_at"].isoformat()
                      if hasattr(row["updated_at"], "isoformat")
                      else row["updated_at"],
    }


def lookup_claim(claim_id: str) -> Claim | None:
    """Fetch a single claim by id. Returns None if absent or on error."""
    if not is_enabled():
        return None
    try:
        row = execute_one(
            """
            SELECT claim_id, task_id, span_id, agent_role, statement,
                   status, register, evidence, verifying_action,
                   load_bearing, tags, superseded_by, created_at
              FROM control_plane.epistemic_claims
             WHERE claim_id = %s
            """,
            (claim_id,),
        )
    except Exception as exc:
        logger.debug("epistemic span_writer.lookup_claim failed: %s", exc)
        return None
    if row is None:
        return None
    return Claim.from_jsonable(_row_to_jsonable(row))


# ── Internal helpers ─────────────────────────────────────────────────

def _evidence_jsonable(claim: Claim) -> list[dict[str, Any]]:
    return [
        {
            "kind": e.kind,
            "source_ref": e.source_ref,
            "excerpt": e.excerpt,
            "confidence": e.confidence,
        }
        for e in claim.evidence
    ]


def _verifier_jsonable(claim: Claim) -> dict[str, Any] | None:
    va = claim.verifying_action
    if va is None:
        return None
    return {
        "tool": va.tool,
        "args": dict(va.args),
        "expected_signal": va.expected_signal,
        "estimated_seconds": va.estimated_seconds,
        "safety": va.safety,
    }


def _row_to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
    """psycopg2 returns JSONB columns as already-parsed Python objects
    (lists and dicts), and TIMESTAMPTZ as datetime. Normalize to the
    shape ``Claim.from_jsonable`` expects."""
    return {
        "claim_id": row["claim_id"],
        "task_id": row["task_id"],
        "span_id": row["span_id"],
        "agent_role": row["agent_role"],
        "statement": row["statement"],
        "status": row["status"],
        "register": row["register"],
        "evidence": row["evidence"] or [],
        "verifying_action": row["verifying_action"],
        "load_bearing": row["load_bearing"],
        "tags": row["tags"] or [],
        "superseded_by": row["superseded_by"],
        "created_at": row["created_at"],
    }
