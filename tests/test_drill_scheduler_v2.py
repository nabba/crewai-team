"""Tests for the Q18 scheduler + orchestrator (PROGRAM §57).

The core regression test: a drill that fails on every invocation
must NOT enter a hot loop. The §44 scheduler ran a failing drill
every idle pass (~30s); the v2 scheduler respects backoff so a
failing drill is invoked at most once per backoff window.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(monkeypatch, tmp_path):
    """Point both state, baseline, and audit paths at temp dirs."""
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    yield


@pytest.fixture
def fresh_registry():
    """Drop any previously-registered drills before each test.

    Eagerly imports the drills package FIRST. Otherwise the package
    import triggered later by ``run_once()`` (which imports
    ``kill_the_gateway`` for its external-report ingest hook) would
    cascade-load every drill module, each re-running its module-
    level ``register(SPEC, run)`` call AFTER our clear — leaking
    production drills into the test and inflating
    ``summary["ran"]``.

    The eager import is wrapped in try/except so the host-only
    pytest pass (where gateway-deps like pydantic / crewai aren't
    installed) still gets the registry cleared. In production the
    import succeeds and the eager-clear catches the in-test drills
    correctly.
    """
    try:
        import app.resilience_drills.drills  # noqa: F401 — register side-effect
    except Exception:
        # Host-only: pydantic / crewai missing. The drills package
        # may not fully import; that's fine for the unit tests below
        # which register their own test drills.
        pass
    from app.resilience_drills.protocol import get_registry
    reg = get_registry()
    reg.clear_for_tests()
    yield reg
    reg.clear_for_tests()


def _register(reg, *, name, cadence_days=90, warmup_days=0, risk=None,
              runner=None):
    from app.resilience_drills.protocol import DrillRisk, DrillSpec, register
    spec = DrillSpec(
        name=name, cadence_days=cadence_days, warmup_days=warmup_days,
        risk=risk or DrillRisk.LOW,
    )
    register(spec, runner)
    return spec


def _make_runner(status, *, failure_class=None, observation=None, errors=None):
    """Return a runner function that yields a result with the given
    status. Used to simulate passing / failing / erroring drills."""
    from app.resilience_drills.protocol import (
        DrillResult, DrillStatus, FailureClass,
    )
    def _run(*, dry_run=True):
        return DrillResult(
            drill_name="(set by caller)",
            status=DrillStatus(status),
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0.01,
            dry_run=dry_run,
            errors=list(errors or []),
            failure_class=FailureClass(failure_class) if failure_class else None,
            observation=observation,
        )
    return _run


def _make_named_runner(name, status, **kwargs):
    """The DrillResult.drill_name must match the spec name. Build a
    runner that injects the right name."""
    from app.resilience_drills.protocol import (
        DrillResult, DrillStatus, FailureClass,
    )
    def _run(*, dry_run=True):
        return DrillResult(
            drill_name=name,
            status=DrillStatus(status),
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0.01,
            dry_run=dry_run,
            errors=list(kwargs.get("errors") or []),
            failure_class=(
                FailureClass(kwargs["failure_class"])
                if kwargs.get("failure_class") else None
            ),
            observation=kwargs.get("observation"),
        )
    return _run


def test_scheduler_skips_failing_drill_on_second_pass(fresh_registry, monkeypatch):
    """The §44 incident regression test. A drill that fails should be
    invoked exactly once over many scheduler passes — backoff blocks
    subsequent invocations until the window expires."""
    from app.resilience_drills.scheduler import run_once
    invocation_count = [0]
    def _failing(*, dry_run=True):
        from app.resilience_drills.protocol import (
            DrillResult, DrillStatus, FailureClass,
        )
        invocation_count[0] += 1
        return DrillResult(
            drill_name="flaky",
            status=DrillStatus.FAIL,
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0.01,
            dry_run=dry_run,
            errors=["always fails"],
            failure_class=FailureClass.STRUCTURAL_FAIL,
        )
    _register(fresh_registry, name="flaky", runner=_failing)

    # Disable notifications to avoid side effects during tests.
    monkeypatch.setattr(
        "app.resilience_drills.scheduler._notify_drill_failed",
        lambda *a, **k: None,
    )

    # First scheduler pass — drill should run.
    summary = run_once()
    assert summary["ran"] == 1
    assert summary["failed"] == 1
    assert invocation_count[0] == 1

    # Second pass immediately after — drill should be skipped due
    # to backoff (15 min for first failure).
    summary = run_once()
    assert summary["skipped"] >= 1
    assert invocation_count[0] == 1  # NOT re-invoked

    # Third pass — still skipped.
    summary = run_once()
    assert invocation_count[0] == 1


def test_quarantined_drill_never_runs_until_unquarantined(fresh_registry, monkeypatch):
    """3 consecutive code errors quarantine the drill; further passes
    don't invoke it. After operator unquarantine, the drill runs again."""
    from app.resilience_drills import state as st
    from app.resilience_drills.scheduler import run_once
    invocation_count = [0]
    def _crashing(*, dry_run=True):
        from app.resilience_drills.protocol import (
            DrillResult, DrillStatus, FailureClass,
        )
        invocation_count[0] += 1
        return DrillResult(
            drill_name="buggy",
            status=DrillStatus.ERROR,
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0.01,
            dry_run=dry_run,
            errors=["AttributeError: bug"],
            failure_class=FailureClass.CODE_ERROR,
        )
    _register(fresh_registry, name="buggy", runner=_crashing)

    monkeypatch.setattr(
        "app.resilience_drills.scheduler._notify_drill_failed",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "app.resilience_drills.scheduler._notify_drill_quarantined",
        lambda *a, **k: None,
    )

    # Trigger 3 code errors. After each, manually clear backoff so the
    # next scheduler pass actually invokes (we're testing quarantine
    # logic, not backoff).
    for i in range(3):
        run_once()
        rec = st.load("buggy")
        rec.next_attempt_after = None  # force runnable
        st.save(rec)

    # Drill should now be QUARANTINED.
    rec = st.load("buggy")
    assert rec.state == st.DrillState.QUARANTINED
    assert invocation_count[0] == 3

    # Further scheduler passes do NOT invoke.
    for _ in range(5):
        run_once()
    assert invocation_count[0] == 3  # unchanged

    # Operator unquarantines.
    st.unquarantine("buggy", operator="test", reason="fixed it")
    rec = st.load("buggy")
    assert rec.state == st.DrillState.WATCH

    # Next scheduler pass invokes again.
    run_once()
    assert invocation_count[0] == 4


def test_scheduler_skips_high_risk_drills_for_auto_run(fresh_registry, monkeypatch):
    """HIGH-risk drills never auto-invoke — operator must run the
    external script."""
    from app.resilience_drills.protocol import DrillRisk
    from app.resilience_drills.scheduler import run_once
    invoked = [False]
    def _runner(*, dry_run=True):
        invoked[0] = True
        from app.resilience_drills.protocol import DrillResult, DrillStatus
        return DrillResult(drill_name="kill", status=DrillStatus.PASS,
                            started_at="x", completed_at="x", duration_s=0,
                            dry_run=dry_run)
    _register(fresh_registry, name="kill", runner=_runner, risk=DrillRisk.HIGH)

    monkeypatch.setattr(
        "app.resilience_drills.scheduler._maybe_notify_high_risk_due",
        lambda spec: None,
    )

    run_once()
    assert invoked[0] is False


def test_warmup_failure_does_not_alert(fresh_registry, monkeypatch):
    """During WARMING_UP, failures don't fire the scheduler's notify
    helper (observational period)."""
    from app.resilience_drills.scheduler import run_once
    alerts = []
    monkeypatch.setattr(
        "app.resilience_drills.scheduler._notify_drill_failed",
        lambda *a, **k: alerts.append(a),
    )
    _register(
        fresh_registry, name="newbie", warmup_days=7,
        runner=_make_named_runner(
            "newbie", "fail", failure_class="structural_fail",
            errors=["initial finding"],
        ),
    )
    run_once()
    assert alerts == []  # warmup period suppresses alerts


def test_operator_invocation_bypasses_backoff(fresh_registry):
    """invoke_drill_by_name with triggered_by='operator' (default)
    should bypass backoff — operator explicitly asked."""
    from app.resilience_drills.runner import invoke_drill_by_name
    from app.resilience_drills.protocol import DrillStatus
    _register(
        fresh_registry, name="ondemand",
        runner=_make_named_runner(
            "ondemand", "fail", failure_class="structural_fail",
        ),
    )
    # First invocation puts the drill into WATCH with 15-min backoff.
    r1 = invoke_drill_by_name("ondemand")
    assert r1.status == DrillStatus.FAIL
    # Second operator invocation succeeds even though backoff is active.
    r2 = invoke_drill_by_name("ondemand")
    assert r2.status == DrillStatus.FAIL  # ran again, didn't skip


def test_scheduler_invocation_respects_backoff(fresh_registry):
    """invoke with triggered_by='scheduler' returns SKIPPED during
    backoff."""
    from app.resilience_drills.protocol import get_registry, DrillStatus
    from app.resilience_drills.runner import invoke_drill
    _register(
        fresh_registry, name="bf",
        runner=_make_named_runner(
            "bf", "fail", failure_class="structural_fail",
        ),
    )
    reg = get_registry()
    spec = reg.get("bf")
    runner = reg.runner_for("bf")
    r1 = invoke_drill(spec, runner, triggered_by="scheduler")
    assert r1.status == DrillStatus.FAIL
    # Backoff active; second scheduler invocation skips.
    r2 = invoke_drill(spec, runner, triggered_by="scheduler")
    assert r2.status == DrillStatus.SKIPPED
    assert "backoff" in r2.detail.get("skip_reason", "").lower()


def test_baseline_match_promotes_fail_to_pass(fresh_registry):
    """Ratified baseline overrides drill-internal pass/fail opinion.

    A drill reports FAIL because its built-in threshold isn't met,
    but the operator's ratified baseline matches the observation.
    The orchestrator promotes the result to PASS."""
    from app.resilience_drills import baseline as bl
    from app.resilience_drills.baseline import Observation
    from app.resilience_drills.protocol import (
        DrillResult, DrillStatus, FailureClass,
    )
    from app.resilience_drills.runner import invoke_drill_by_name

    def _picky(*, dry_run=True):
        # Drill returns FAIL but the observation is "n_fallbacks=1".
        # If operator ratifies that as min=1, this should become PASS.
        return DrillResult(
            drill_name="picky",
            status=DrillStatus.FAIL,
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0.01,
            dry_run=dry_run,
            errors=["wanted n_fallbacks >= 2"],
            failure_class=FailureClass.STRUCTURAL_FAIL,
            observation={"n_fallbacks": 1, "providers": ["groq"]},
        )
    _register(fresh_registry, name="picky", runner=_picky)

    # Operator ratifies: n_fallbacks >= 1 is acceptable
    bl.ratify_from_observation(
        Observation(drill_name="picky", observed_at="2026-05-18T00:00:00+00:00",
                    measurements={"n_fallbacks": 1, "providers": ["groq"]}),
        operator="test",
        tolerances={
            "n_fallbacks": {"rule": "min", "value": 1},
            "providers": {"rule": "superset_of", "value": ["groq"]},
        },
    )

    result = invoke_drill_by_name("picky")
    # FAIL got promoted to PASS — operator policy wins.
    assert result.status == DrillStatus.PASS
    assert result.failure_class is None


def test_baseline_regression_overrides_pass(fresh_registry):
    """A drill that internally returns PASS but whose observation
    diverges from the ratified baseline becomes a FAIL with
    failure_class=BASELINE_REGRESSION."""
    from app.resilience_drills import baseline as bl
    from app.resilience_drills.baseline import Observation
    from app.resilience_drills.protocol import (
        DrillResult, DrillStatus, FailureClass,
    )
    from app.resilience_drills.runner import invoke_drill_by_name

    obs_value = ["groq", "anthropic"]

    def _drill(*, dry_run=True):
        return DrillResult(
            drill_name="probe",
            status=DrillStatus.PASS,    # drill itself sees no problem
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0.01,
            dry_run=dry_run,
            observation={"providers_ready": obs_value},
        )
    _register(fresh_registry, name="probe", runner=_drill)

    # Operator ratifies baseline = ["groq", "anthropic", "openrouter"]
    # — requires all three
    bl.ratify_from_observation(
        Observation(drill_name="probe", observed_at="x",
                    measurements={"providers_ready": ["groq", "anthropic", "openrouter"]}),
        operator="test",
        tolerances={
            "providers_ready": {"rule": "superset_of",
                                 "value": ["groq", "anthropic", "openrouter"]},
        },
    )

    result = invoke_drill_by_name("probe")
    # Drill PASS but baseline says regression → FAIL with BASELINE_REGRESSION
    assert result.status == DrillStatus.FAIL
    assert result.failure_class == FailureClass.BASELINE_REGRESSION


def test_runner_exception_becomes_code_error(fresh_registry):
    """A runner that raises uncaught becomes ERROR + CODE_ERROR."""
    from app.resilience_drills.protocol import DrillStatus, FailureClass
    from app.resilience_drills.runner import invoke_drill_by_name

    def _bad(*, dry_run=True):
        raise RuntimeError("simulated drill bug")
    _register(fresh_registry, name="bad", runner=_bad)

    result = invoke_drill_by_name("bad")
    assert result.status == DrillStatus.ERROR
    assert result.failure_class == FailureClass.CODE_ERROR
    assert any("RuntimeError" in e for e in result.errors)


def test_muted_drill_returns_skipped_from_scheduler(fresh_registry, monkeypatch):
    from app.resilience_drills import state as st
    from app.resilience_drills.scheduler import run_once
    invoked = [False]
    def _r(*, dry_run=True):
        invoked[0] = True
        from app.resilience_drills.protocol import DrillResult, DrillStatus
        return DrillResult(drill_name="m", status=DrillStatus.PASS,
                            started_at="x", completed_at="x", duration_s=0,
                            dry_run=True)
    _register(fresh_registry, name="m", runner=_r)
    st.mute("m", operator="test", reason="too noisy")

    run_once()
    assert invoked[0] is False


def test_summary_counts_match(fresh_registry, monkeypatch):
    """Scheduler summary must accurately count passes/fails/skips."""
    from app.resilience_drills.scheduler import run_once
    monkeypatch.setattr(
        "app.resilience_drills.scheduler._notify_drill_failed",
        lambda *a, **k: None,
    )
    _register(fresh_registry, name="p",
              runner=_make_named_runner("p", "pass"))
    _register(fresh_registry, name="f",
              runner=_make_named_runner("f", "fail",
                                          failure_class="structural_fail"))
    summary = run_once()
    assert summary["checked"] == 2
    assert summary["ran"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
