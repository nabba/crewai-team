"""Idle-job entry points for the yearly identity-reflection passes.

Both passes (``annual_reflection`` and ``legacy_essay``) cadence-check
internally via ``_is_due`` against their target file's mtime, so a daily
fire is mostly a no-op. The actual LLM work runs once per year per essay.

The two passes are LIGHT-weight idle jobs — they cost a few hundred
milliseconds for the cadence check on the days they don't fire, and one
LLM call (a few seconds) on the day they do.

Failure-isolated: any exception during a pass is caught locally; the
idle scheduler sees a successful tick. The passes themselves never raise
(they return ``ReflectionResult`` / ``LegacyResult`` with status fields).

Master switches: ``ANNUAL_REFLECTION_ENABLED`` and ``LEGACY_ESSAY_ENABLED``
(both default ``true``). With either off, that pass's idle job becomes a
no-op short-circuit.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def _resolve_llm_call() -> Callable[[str, str], str] | None:
    """Wire the production LLM call. Failure-isolated: returns ``None``
    on any wiring failure so the caller defers the run."""
    try:
        from app.llm_factory import create_specialist_llm
    except Exception as exc:  # noqa: BLE001
        logger.debug("identity scheduler: llm_factory import failed: %s", exc)
        return None
    try:
        llm = create_specialist_llm(role="research", max_tokens=4096)
    except Exception as exc:  # noqa: BLE001
        logger.debug("identity scheduler: LLM construction failed: %s", exc)
        return None

    def call(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = llm.call(messages=messages)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM call failed: {exc}") from exc
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and "content" in response:
            return str(response["content"])
        return str(response)

    return call


def run_annual_reflection() -> None:
    """Idle-job tick for the §8.2 annual value-reflection pass.

    Cadence-checks via :func:`app.identity.annual_reflection._is_due`
    (350-day floor by default). LLM is resolved per-call so a transient
    factory failure simply defers to the next tick.
    """
    try:
        from app.identity.annual_reflection import run_one_pass
    except Exception:
        logger.debug("identity scheduler: annual_reflection import failed", exc_info=True)
        return
    llm_call = _resolve_llm_call()
    if llm_call is None:
        logger.debug("identity scheduler: LLM unavailable; deferring annual reflection")
        return
    try:
        result = run_one_pass(llm_call=llm_call)
        if result.status not in ("skipped_recent", "skipped_disabled"):
            logger.info(
                "identity scheduler: annual_reflection status=%s year=%s attempts=%d",
                result.status, result.year, result.attempts,
            )
    except Exception:
        logger.debug("identity scheduler: annual_reflection raised", exc_info=True)


def run_legacy_essay() -> None:
    """Idle-job tick for the §8.5 legacy essay pass.

    Same cadence + LLM-resolution discipline as the annual reflection.
    """
    try:
        from app.identity.legacy_essay import run_one_pass
    except Exception:
        logger.debug("identity scheduler: legacy_essay import failed", exc_info=True)
        return
    llm_call = _resolve_llm_call()
    if llm_call is None:
        logger.debug("identity scheduler: LLM unavailable; deferring legacy essay")
        return
    try:
        result = run_one_pass(llm_call=llm_call)
        if result.status not in ("skipped_recent", "skipped_disabled"):
            logger.info(
                "identity scheduler: legacy_essay status=%s year=%s attempts=%d",
                result.status, result.year, result.attempts,
            )
    except Exception:
        logger.debug("identity scheduler: legacy_essay raised", exc_info=True)


def run_long_term_goal_review() -> None:
    """Idle-job tick for the Q9.6 (PROGRAM §46.9) quarterly long-term
    goal review. Cadence-checks internally; LLM call is failure-
    isolated; phenomenal-language linter retry applies."""
    try:
        from app.identity.long_term_goal_review import run as _run
    except Exception:
        logger.debug(
            "identity scheduler: long_term_goal_review import failed",
            exc_info=True,
        )
        return
    try:
        result = _run()
        status = result.get("status") if isinstance(result, dict) else ""
        if status not in ("skipped_recent", "skipped_disabled"):
            logger.info(
                "identity scheduler: long_term_goal_review status=%s "
                "quarter=%s attempts=%s",
                status,
                (result.get("quarter_label") if isinstance(result, dict) else ""),
                (result.get("attempts") if isinstance(result, dict) else 0),
            )
    except Exception:
        logger.debug(
            "identity scheduler: long_term_goal_review raised",
            exc_info=True,
        )


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """Idle-scheduler job tuples — appended in :func:`app.companion.loop.get_idle_jobs`.

    All jobs are LIGHT (sub-second on the typical no-op tick).
    """
    from app.idle_scheduler import JobWeight
    return [
        ("identity-annual-reflection", run_annual_reflection, JobWeight.LIGHT),
        ("identity-legacy-essay", run_legacy_essay, JobWeight.LIGHT),
        # Q9.6 (PROGRAM §46.9) — quarterly long-term goal review.
        ("identity-long-term-goal-review",
         run_long_term_goal_review, JobWeight.LIGHT),
    ]
