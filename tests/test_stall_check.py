"""Tests for the request-path stall watchdog (``_evaluate_stall``).

The watchdog has four tiers — output-stall, crew-zero-progress,
zero-output, llm-stall — drained from progress / tool-activity / LLM
heartbeats kept by ``app.observability.task_progress``,
``app.tools_timeout``, and ``app.rate_throttle``.

These tests drive ``_evaluate_stall`` directly: the function is a pure
read over those globals, so the test simulates them rather than
spinning up an actual ``handle_task`` request.
"""
from __future__ import annotations

import time

import pytest

from app import rate_throttle, tools_timeout
from app.main import (
    _CREW_TOOL_QUIET_SECS,
    _CREW_ZERO_PROGRESS_KILL_SECS,
    _OUTPUT_STALL_THRESHOLD_SECS,
    _PROGRESS_CHECK_EVERY,
    _STALL_THRESHOLD_SECS,
    _ZERO_OUTPUT_KILL_SECS,
    _evaluate_stall,
)
from app.observability.task_progress import record_output_progress, reset_task


@pytest.fixture(autouse=True)
def _reset_global_heartbeats(monkeypatch):
    """Each test starts with no tool / LLM heartbeat history.

    The two timestamps live as module-globals (deliberate — they are
    process-wide signals shared across the entire request path). Tests
    must reset them so order doesn't matter.
    """
    monkeypatch.setattr(tools_timeout, "_last_tool_activity_ts", 0.0)
    monkeypatch.setattr(rate_throttle, "_last_llm_activity_ts", 0.0)
    yield


def _set_tool_idle_secs(idle_secs: float | None) -> None:
    """Drive ``seconds_since_last_tool_activity()`` to a chosen value.

    ``None`` → "no tool has ever run" (timestamp = 0.0).
    ``N``    → "last tool activity was N seconds ago".
    """
    if idle_secs is None:
        tools_timeout._last_tool_activity_ts = 0.0
    else:
        tools_timeout._last_tool_activity_ts = time.monotonic() - idle_secs


def _set_llm_idle_secs(idle_secs: float | None) -> None:
    """Drive ``seconds_since_last_llm_activity()`` to a chosen value."""
    if idle_secs is None:
        rate_throttle._last_llm_activity_ts = 0.0
    else:
        rate_throttle._last_llm_activity_ts = time.monotonic() - idle_secs


# ── Tier 2: crew-zero-progress (the new watchdog) ────────────────────


def test_crew_zero_progress_fires_when_tools_never_ran():
    """No tools ever + zero output + > 10 min elapsed → kill."""
    tid = "test-czp-never-ran"
    reset_task(tid)
    _set_tool_idle_secs(None)
    # An LLM may still be "active" (e.g. retry-loop). Tier 2 ignores it.
    _set_llm_idle_secs(2.0)

    elapsed = _CREW_ZERO_PROGRESS_KILL_SECS + 5  # 605s
    result = _evaluate_stall(tid, elapsed_secs=elapsed)

    assert result is not None
    kind, secs = result
    assert kind == "crew-zero-progress"
    assert secs == pytest.approx(elapsed, abs=0.5)


def test_crew_zero_progress_fires_when_tools_quiet_for_5_min():
    """Tools ran once but went quiet for > 4 min → kill."""
    tid = "test-czp-quiet"
    reset_task(tid)
    _set_tool_idle_secs(_CREW_TOOL_QUIET_SECS + 60)  # 5 min idle
    _set_llm_idle_secs(2.0)  # LLM still cycling

    elapsed = _CREW_ZERO_PROGRESS_KILL_SECS + 50
    result = _evaluate_stall(tid, elapsed_secs=elapsed)

    assert result is not None
    kind, _ = result
    assert kind == "crew-zero-progress"


def test_crew_zero_progress_inert_when_tools_active():
    """Tools cycling within the quiet window → DO NOT kill at 10 min.

    This is the "legitimate long workload" guarantee: deep-research at
    d=9 ticks tool-activity every minute or two, so even at 11+ min
    the new tier must not fire. The 1200s zero-output backstop is the
    correct gate for that pattern.
    """
    tid = "test-czp-active"
    reset_task(tid)
    _set_tool_idle_secs(60.0)  # tool ran 1 min ago — still fresh
    _set_llm_idle_secs(2.0)

    # Past the new tier's threshold but below the zero-output backstop.
    elapsed = _CREW_ZERO_PROGRESS_KILL_SECS + 100  # 700s
    result = _evaluate_stall(tid, elapsed_secs=elapsed)

    # Tier 2 must not fire. Tier 3 (zero-output) needs > 1200s, not met.
    # Tier 4 (llm-stall) needs > 240s LLM-idle, not met.
    assert result is None


def test_crew_zero_progress_inert_below_elapsed_threshold():
    """Below 10 min, even with no tools ever, do nothing."""
    tid = "test-czp-early"
    reset_task(tid)
    _set_tool_idle_secs(None)
    _set_llm_idle_secs(2.0)

    elapsed = _CREW_ZERO_PROGRESS_KILL_SECS - 30  # 570s
    result = _evaluate_stall(tid, elapsed_secs=elapsed)
    assert result is None


def test_crew_zero_progress_inert_when_partial_emitted():
    """Any output progress → output-stall takes over (or is silent).

    Tier 1 (output-stall) returns immediately on the
    output_progress branch — so tier 2 never gets a chance to fire,
    even if tools are quiet.
    """
    tid = "test-czp-with-partial"
    reset_task(tid)
    record_output_progress(tid, note="row:1")  # one partial recorded
    _set_tool_idle_secs(None)  # tools quiet — but tier 1 supersedes

    elapsed = _CREW_ZERO_PROGRESS_KILL_SECS + 100
    result = _evaluate_stall(tid, elapsed_secs=elapsed)

    # Either no kill (partial fresh, < 5 min stale) or output-stall —
    # but never crew-zero-progress.
    if result is not None:
        kind, _ = result
        assert kind != "crew-zero-progress"
        assert kind == "output-stall" or kind == "llm-stall"


# ── Tier 1: output-stall preempts the new tier ───────────────────────


def test_output_stall_preempts_crew_zero_progress():
    """A partial that went stale 6+ min ago → output-stall fires."""
    tid = "test-output-stall-preempts"
    reset_task(tid)
    # Record a partial, then backdate it to look 6 min old.
    record_output_progress(tid, note="row:1")
    from app.observability import task_progress
    task_progress._last_progress[tid] = (
        time.monotonic() - (_OUTPUT_STALL_THRESHOLD_SECS + 60)
    )
    _set_tool_idle_secs(None)  # tools quiet too — but tier 1 wins

    elapsed = _CREW_ZERO_PROGRESS_KILL_SECS + 100
    result = _evaluate_stall(tid, elapsed_secs=elapsed)

    assert result is not None
    kind, _ = result
    assert kind == "output-stall"


# ── Tier 3: zero-output backstop still works ─────────────────────────


def test_zero_output_backstop_when_tools_active_and_no_partials():
    """Tools cycling for 21 min, no partials → zero-output fires.

    This guarantees the backstop still catches the "slow but cycling"
    case: tools are firing every minute (so crew-zero-progress stays
    inert), but the system has produced no deliverable for 20+ min.
    """
    tid = "test-zero-output-backstop"
    reset_task(tid)
    _set_tool_idle_secs(30.0)  # tool ran 30s ago — fresh
    _set_llm_idle_secs(2.0)

    elapsed = _ZERO_OUTPUT_KILL_SECS + 60  # 21 min
    result = _evaluate_stall(tid, elapsed_secs=elapsed)

    assert result is not None
    kind, secs = result
    assert kind == "zero-output"
    assert secs == pytest.approx(elapsed, abs=0.5)


# ── Tier 4: llm-stall still works ────────────────────────────────────


def test_llm_stall_fires_when_no_other_tier_does():
    """LLM has been silent > 4 min, partials exist (so no zero-output)."""
    tid = "test-llm-stall"
    reset_task(tid)
    # No output progress, but tools are fresh (tier 2 inert) and we're
    # below the 1200s zero-output backstop.
    _set_tool_idle_secs(30.0)
    _set_llm_idle_secs(_STALL_THRESHOLD_SECS + 30)  # 270s LLM-idle

    elapsed = 500  # below crew-zero-progress threshold
    result = _evaluate_stall(tid, elapsed_secs=elapsed)

    assert result is not None
    kind, _ = result
    assert kind == "llm-stall"


# ── Acceptance: kill happens within 2× the threshold ─────────────────


def test_crew_zero_progress_kills_within_2x_threshold():
    """Simulate the polling loop; first kill must arrive ≤ 2× threshold.

    ``handle_task`` calls ``_evaluate_stall`` every
    ``_PROGRESS_CHECK_EVERY`` seconds during the extension window. The
    spec requires that a stuck-no-tool-no-output crew dies within
    ``2 * _CREW_ZERO_PROGRESS_KILL_SECS = 1200s`` worst-case. With the
    threshold at 600s and a 30s poll cadence, the actual worst case is
    ≤ 630s — well inside the budget.
    """
    tid = "test-czp-window"
    reset_task(tid)
    _set_tool_idle_secs(None)  # never any tools
    _set_llm_idle_secs(2.0)    # LLM "active" (retry-loop simulant)

    first_kill_at: float | None = None
    for elapsed in range(0, 2 * _CREW_ZERO_PROGRESS_KILL_SECS + 1,
                         _PROGRESS_CHECK_EVERY):
        if _evaluate_stall(tid, elapsed_secs=float(elapsed)) is not None:
            first_kill_at = float(elapsed)
            break

    assert first_kill_at is not None, "watchdog never fired"
    assert first_kill_at >= _CREW_ZERO_PROGRESS_KILL_SECS, (
        f"watchdog fired too early at {first_kill_at}s"
    )
    assert first_kill_at <= 2 * _CREW_ZERO_PROGRESS_KILL_SECS, (
        f"watchdog fired too late at {first_kill_at}s "
        f"(budget: {2 * _CREW_ZERO_PROGRESS_KILL_SECS}s)"
    )
    # Tighter bound documented for future regressions: 30s poll cadence
    # means worst-case kill is within one check-window of the threshold.
    assert first_kill_at <= _CREW_ZERO_PROGRESS_KILL_SECS + _PROGRESS_CHECK_EVERY
