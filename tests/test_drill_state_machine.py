"""Tests for the Q18 drill state machine (app/resilience_drills/state.py)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_state_dir(monkeypatch, tmp_path):
    """Point state.py at a temp dir so tests don't pollute workspace."""
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    yield


def _now():
    return datetime.now(timezone.utc)


def test_load_or_initialize_creates_warming_up_record():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("test_drill", warmup_days=7)
    assert rec.drill_name == "test_drill"
    assert rec.state == st.DrillState.WARMING_UP
    assert rec.warming_up_until is not None
    # Persisted: reloading returns the same record
    rec2 = st.load("test_drill")
    assert rec2 is not None
    assert rec2.warming_up_until == rec.warming_up_until


def test_pass_during_warmup_keeps_warming_when_within_window():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("test_drill", warmup_days=7)
    st.record_pass(rec, cadence_days=90)
    assert rec.state == st.DrillState.WARMING_UP


def test_pass_after_warmup_window_promotes_to_healthy():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("test_drill", warmup_days=7)
    # Force warmup to have ended
    rec.warming_up_until = (_now() - timedelta(days=1)).isoformat()
    st.record_pass(rec, cadence_days=90)
    assert rec.state == st.DrillState.HEALTHY
    assert rec.warming_up_until is None


def test_first_failure_after_warmup_goes_to_watch():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("test_drill", warmup_days=0)  # immediate active
    # Make sure warmup is over
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    st.record_failure(rec, cadence_days=7, failure_class="structural_fail",
                       summary="threshold not met")
    assert rec.state == st.DrillState.WATCH
    assert rec.consecutive_failures == 1
    assert rec.next_attempt_after is not None


def test_second_failure_demotes_to_degraded():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("test_drill", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    st.record_failure(rec, cadence_days=7, failure_class="structural_fail")
    st.record_failure(rec, cadence_days=7, failure_class="structural_fail")
    assert rec.state == st.DrillState.DEGRADED
    assert rec.consecutive_failures == 2


def test_pass_after_failure_returns_to_healthy_and_resets_counters():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("test_drill", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    st.record_failure(rec, cadence_days=7, failure_class="structural_fail")
    st.record_failure(rec, cadence_days=7, failure_class="structural_fail")
    assert rec.consecutive_failures == 2
    st.record_pass(rec, cadence_days=7)
    assert rec.state == st.DrillState.HEALTHY
    assert rec.consecutive_failures == 0
    assert rec.last_failure_summary == ""


def test_three_consecutive_code_errors_quarantine():
    """The core regression test — the embedding_migration drill case."""
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("embedding_migration", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    for _ in range(3):
        st.record_failure(rec, cadence_days=90, failure_class="code_error",
                           summary="AttributeError: 'DryRunStep' object has no attribute 'get'",
                           traceback_text="Traceback (most recent call last):\n  File ...")
    assert rec.state == st.DrillState.QUARANTINED
    assert rec.quarantined_at is not None
    assert "consecutive code errors" in rec.quarantined_reason


def test_quarantined_drill_is_not_runnable():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.state = st.DrillState.QUARANTINED
    rec.quarantined_at = _now().isoformat()
    ok, reason = st.is_runnable_now(rec)
    assert ok is False
    assert "quarantine" in reason.lower()


def test_muted_drill_is_not_runnable_until_expiry():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.state = st.DrillState.MUTED
    rec.muted_at = _now().isoformat()
    # No expiry → indefinite
    ok, reason = st.is_runnable_now(rec)
    assert ok is False
    assert "muted" in reason.lower()


def test_muted_drill_auto_unmutes_at_expiry():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.state = st.DrillState.MUTED
    rec.muted_at = (_now() - timedelta(days=1)).isoformat()
    rec.muted_until = (_now() - timedelta(hours=1)).isoformat()  # past
    ok, _ = st.is_runnable_now(rec)
    assert ok is True


def test_backoff_active_blocks_running():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    st.record_failure(rec, cadence_days=7, failure_class="structural_fail")
    # next_attempt_after is now ~15 min in the future
    ok, reason = st.is_runnable_now(rec)
    assert ok is False
    assert "backoff" in reason.lower()


def test_backoff_expires_drill_becomes_runnable():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.WATCH
    # Backdate next_attempt_after to the past
    rec.next_attempt_after = (_now() - timedelta(minutes=1)).isoformat()
    ok, _ = st.is_runnable_now(rec)
    assert ok is True


def test_backoff_grows_exponentially():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    waits = []
    for _ in range(6):
        st.record_failure(rec, cadence_days=90, failure_class="structural_fail")
        naa = datetime.fromisoformat(rec.next_attempt_after.replace("Z", "+00:00"))
        last_run = datetime.fromisoformat(rec.last_run_at.replace("Z", "+00:00"))
        waits.append((naa - last_run).total_seconds())
    # Strictly increasing (or at the cap)
    for i in range(1, len(waits)):
        assert waits[i] >= waits[i - 1]


def test_backoff_capped_at_cadence_days():
    """Backoff must never exceed the drill's cadence — the scheduler
    will run the drill at least that often regardless of past
    failures."""
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    for _ in range(20):  # way past cap
        st.record_failure(rec, cadence_days=7, failure_class="structural_fail")
    naa = datetime.fromisoformat(rec.next_attempt_after.replace("Z", "+00:00"))
    last_run = datetime.fromisoformat(rec.last_run_at.replace("Z", "+00:00"))
    wait_s = (naa - last_run).total_seconds()
    assert wait_s <= 7 * 86400  # capped at cadence


def test_unquarantine_resets_to_watch():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.QUARANTINED
    rec.consecutive_code_errors = 5
    st.save(rec)
    updated = st.unquarantine("d", operator="andrus", reason="fixed the bug")
    assert updated is not None
    assert updated.state == st.DrillState.WATCH
    assert updated.consecutive_code_errors == 0
    assert updated.quarantined_at is None
    # Next attempt is immediate after unquarantine.
    assert updated.next_attempt_after is None


def test_unquarantine_noop_when_not_quarantined():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.state = st.DrillState.HEALTHY
    st.save(rec)
    updated = st.unquarantine("d", operator="andrus")
    assert updated.state == st.DrillState.HEALTHY  # unchanged


def test_mute_then_unmute():
    from app.resilience_drills import state as st
    st.mute("d", operator="andrus", reason="too noisy")
    rec = st.load("d")
    assert rec.state == st.DrillState.MUTED
    st.unmute("d", operator="andrus")
    rec = st.load("d")
    assert rec.state == st.DrillState.HEALTHY


def test_warming_up_failure_does_not_escalate_state():
    """Warmup grace: failures during warmup are observational only —
    state stays WARMING_UP, no DEGRADED escalation, no alert in the
    scheduler."""
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("new_drill", warmup_days=7)
    for _ in range(5):
        st.record_failure(rec, cadence_days=90, failure_class="structural_fail")
    assert rec.state == st.DrillState.WARMING_UP
    # But counters DO increment so the operator sees them when reviewing.
    assert rec.consecutive_failures == 5


def test_code_error_counter_resets_on_non_code_error():
    """An accumulating CODE_ERROR streak is broken by a structural_fail —
    the drill is intermittently sick, not consistently broken-by-code."""
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    st.record_failure(rec, cadence_days=90, failure_class="code_error")
    st.record_failure(rec, cadence_days=90, failure_class="code_error")
    assert rec.consecutive_code_errors == 2
    st.record_failure(rec, cadence_days=90, failure_class="structural_fail")
    assert rec.consecutive_code_errors == 0
    # And so we don't reach quarantine threshold
    st.record_failure(rec, cadence_days=90, failure_class="code_error")
    assert rec.state != st.DrillState.QUARANTINED


def test_transitions_recorded_with_bounded_history():
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("d", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    # Flap many times to test history capping
    for i in range(30):
        if i % 2 == 0:
            st.record_failure(rec, cadence_days=90, failure_class="structural_fail")
        else:
            st.record_pass(rec, cadence_days=90)
    st.save(rec)
    assert len(rec.transitions) <= st.MAX_TRANSITION_HISTORY


def test_list_all_state_records_returns_persisted():
    from app.resilience_drills import state as st
    st.load_or_initialize("a", warmup_days=7)
    st.load_or_initialize("b", warmup_days=7)
    out = st.list_all_state_records()
    names = sorted(r.drill_name for r in out)
    assert names == ["a", "b"]


def test_no_hot_loop_regression_2026_05_16():
    """Pin: the §44 incident pattern cannot recur.

    A drill that keeps failing (structural_fail every time) must stop
    running on every scheduler pass. After N failures it has a
    non-trivial backoff — at least 1 hour by the second failure.
    """
    from app.resilience_drills import state as st
    rec = st.load_or_initialize("local_only", warmup_days=0)
    rec.warming_up_until = None
    rec.state = st.DrillState.HEALTHY
    st.record_failure(rec, cadence_days=90, failure_class="structural_fail")
    st.record_failure(rec, cadence_days=90, failure_class="structural_fail")
    # After 2 failures: at least 1h backoff
    naa = datetime.fromisoformat(rec.next_attempt_after.replace("Z", "+00:00"))
    last_run = datetime.fromisoformat(rec.last_run_at.replace("Z", "+00:00"))
    wait_s = (naa - last_run).total_seconds()
    assert wait_s >= 3600
