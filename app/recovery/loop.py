"""
loop.py — Recovery orchestrator.

Single entry point ``maybe_recover()`` called from the orchestrator's
post-vetting/post-critic block. Walks the librarian's ranked
alternatives within a budget and returns the first successful
recovery (or, if nothing recovers, a structured diagnostic).

Hybrid sync/async per design decision 1:
  * sync strategies (re_route, escalate_tier) run inside the user's
    request — adds 30-90s of latency on the recovery path.
  * async strategies (forge_queue, future skill_chain, future
    sandbox_execute) get fire-and-forget'd; the diagnostic text is
    delivered immediately with a "I've queued X" note.

Recursion guard: ``_in_recovery`` ContextVar prevents recovery loops
from triggering recovery on their own intermediate refusals.

Off by default — toggle with RECOVERY_LOOP_ENABLED=true.
"""
from __future__ import annotations

import contextvars
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from app.recovery.refusal_detector import RefusalSignal, detect_refusal
from app.recovery.librarian import Alternative, find_alternatives

logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """Final outcome of the recovery loop."""
    triggered: bool                   # True if refusal was detected at all
    success: bool                     # True if a strategy produced a real answer
    text: str | None = None           # the recovered (or diagnostic) text
    note: str | None = None           # short note for the user (route_changed)
    route_changed: bool = False       # True → annotate the answer with `note`
    strategies_tried: list[str] = field(default_factory=list)
    refusal_signal: RefusalSignal | None = None
    elapsed_s: float = 0.0


_in_recovery: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "recovery_in_progress", default=False,
)


def is_enabled() -> bool:
    """Off by default. Flip via the React /cp/settings UI (preferred —
    no restart) or set ``RECOVERY_LOOP_ENABLED=true`` (legacy fallback).

    Runtime-settings wins on a live system. Env var is the test /
    degraded-boot fallback. The runtime_settings file is seeded from
    the env value at first boot so an existing ``.env`` setup keeps
    its current behaviour.
    """
    try:
        from app.runtime_settings import get_recovery_loop_enabled
        return bool(get_recovery_loop_enabled())
    except Exception:
        val = os.getenv("RECOVERY_LOOP_ENABLED", "").strip().lower()
        return val in ("1", "true", "yes", "on")


def _max_attempts() -> int:
    try:
        return max(1, min(5, int(os.getenv("RECOVERY_MAX_ATTEMPTS", "2"))))
    except ValueError:
        return 2


def _budget_seconds() -> float:
    try:
        return float(os.getenv("RECOVERY_BUDGET_SECONDS", "90"))
    except ValueError:
        return 90.0


def _audit(action: str, **fields: Any) -> None:
    """Best-effort audit-log entry. Failures are silent."""
    try:
        from app.control_plane.audit import get_audit
        get_audit().log(actor="recovery_loop", action=action, detail=fields)
    except Exception:
        pass


def _execute_strategy(alt: Alternative, task: str, ctx: dict):
    """Dispatch to the right strategy module. Returns StrategyResult."""
    if alt.strategy == "direct_tool":
        from app.recovery.strategies import direct_tool
        return direct_tool.execute(task, alt, ctx)
    if alt.strategy == "sandbox_execute":
        from app.recovery.strategies import sandbox_execute
        return sandbox_execute.execute(task, alt, ctx)
    if alt.strategy == "re_route":
        from app.recovery.strategies import re_route
        return re_route.execute(task, alt, ctx)
    if alt.strategy == "skill_chain":
        from app.recovery.strategies import skill_chain
        return skill_chain.execute(task, alt, ctx)
    if alt.strategy == "escalate_tier":
        from app.recovery.strategies import escalate_tier
        return escalate_tier.execute(task, alt, ctx)
    if alt.strategy == "forge_queue":
        from app.recovery.strategies import forge_queue
        return forge_queue.execute(task, alt, ctx)
    from app.recovery.strategies import StrategyResult
    return StrategyResult(success=False, error=f"unknown strategy: {alt.strategy}")


def maybe_recover(
    response_text: str,
    user_input: str,
    crew_used: str,
    *,
    commander: Any = None,
    difficulty: int = 5,
    used_tier: str | None = None,
    conversation_history: str = "",
    force: bool = False,
) -> RecoveryResult:
    """Detect + optionally recover a refusal-shaped response.

    Returns ``RecoveryResult`` with ``triggered=False`` when no
    refusal was detected (caller should keep the original response
    unchanged). When ``triggered=True``, ``success`` says whether
    we produced a recovered answer (``text``, possibly with ``note``).

    Always safe to call. Disabled? returns triggered=False instantly.
    Already inside recovery? returns triggered=False to prevent recursion.
    Errors? logs + returns triggered=False so caller doesn't lose the
    original response.
    """
    if not is_enabled():
        return RecoveryResult(triggered=False, success=False)

    if _in_recovery.get():
        # Recursion guard — a strategy's own LLM call shouldn't trigger
        # recovery on its intermediate refusal output.
        return RecoveryResult(triggered=False, success=False)

    if not response_text or not isinstance(response_text, str):
        return RecoveryResult(triggered=False, success=False)

    t_start = time.monotonic()
    try:
        signal = detect_refusal(response_text, force=force)
    except Exception as exc:
        logger.debug("recovery: detect_refusal raised: %s", exc, exc_info=True)
        return RecoveryResult(triggered=False, success=False)

    if signal is None:
        return RecoveryResult(triggered=False, success=False)

    logger.warning(
        "recovery: refusal detected — category=%s confidence=%.2f phrase=%r",
        signal.category, signal.confidence, signal.matched_phrase,
    )
    _audit(
        "refusal.detected",
        category=signal.category,
        confidence=signal.confidence,
        phrase=signal.matched_phrase,
        crew=crew_used,
        task=user_input[:300],
    )

    try:
        alts = find_alternatives(
            user_input, signal.category, crew_used,
            used_tier=used_tier,
            response_text=response_text,
        )
    except Exception as exc:
        logger.debug("recovery: find_alternatives raised: %s", exc, exc_info=True)
        return RecoveryResult(
            triggered=True, success=False,
            refusal_signal=signal,
            elapsed_s=time.monotonic() - t_start,
        )

    # Q8.1 (PROGRAM §46.1) — surface open-thread unblock-hints to
    # strategies. Failure-isolated; produces [] when threads module is
    # unavailable, so existing strategies are unaffected when there
    # are no open threads.
    try:
        from app.recovery.thread_consultation import collect_open_thread_hints
        thread_hints = collect_open_thread_hints()
    except Exception:
        thread_hints = []

    ctx = {
        "commander": commander,
        "user_input": user_input,
        "crew_used": crew_used,
        "conversation_history": conversation_history,
        "difficulty": difficulty,
        "refusal_category": signal.category,
        "original_response": response_text,
        "thread_hints": thread_hints,
    }

    max_attempts = _max_attempts()
    budget_s = _budget_seconds()
    tried: list[str] = []
    recovered_result = None

    token = _in_recovery.set(True)
    try:
        for alt in alts:
            if len(tried) >= max_attempts:
                break
            elapsed = time.monotonic() - t_start
            if elapsed >= budget_s:
                logger.info(
                    "recovery: budget exhausted (%.1fs) before trying %r",
                    elapsed, alt.strategy,
                )
                break

            tried.append(alt.strategy)
            logger.info(
                "recovery: trying strategy=%s rationale=%r",
                alt.strategy, alt.rationale,
            )

            try:
                result = _execute_strategy(alt, user_input, ctx)
            except Exception as exc:
                logger.warning(
                    "recovery: strategy %r raised: %s",
                    alt.strategy, exc, exc_info=True,
                )
                continue

            _audit(
                "strategy.executed",
                strategy=alt.strategy,
                success=result.success,
                error=(result.error or "")[:200],
                crew_used=crew_used,
                target_crew=alt.crew,
                target_tier=alt.tier,
            )

            if result.success and result.text:
                recovered_result = RecoveryResult(
                    triggered=True,
                    success=True,
                    text=result.text,
                    note=result.note,
                    route_changed=result.route_changed,
                    strategies_tried=list(tried),
                    refusal_signal=signal,
                    elapsed_s=time.monotonic() - t_start,
                )
                break
    finally:
        _in_recovery.reset(token)

    if recovered_result is not None:
        return recovered_result

    # Every alternative strategy was tried (or budget exhausted) and none
    # produced a successful completion.  This is the trigger for the
    # BotArmyLLMCascadeAllFailing alert — exposed as a Prometheus counter
    # so the alerting rule fires on increments over a 10m window.
    try:
        from app.observability.metrics import LLM_CASCADE_ALL_TIERS_FAILED_TOTAL
        LLM_CASCADE_ALL_TIERS_FAILED_TOTAL.inc()
    except Exception:
        # Never let a metric emission break the recovery contract.
        logger.debug("recovery: metric emission failed", exc_info=True)

    return RecoveryResult(
        triggered=True,
        success=False,
        strategies_tried=list(tried),
        refusal_signal=signal,
        elapsed_s=time.monotonic() - t_start,
    )
