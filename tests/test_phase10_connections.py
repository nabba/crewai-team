"""
Phase 10 regression tests — the six SIA inter-system connections
plus external-service circuit breaker.

Connections under test (SubIA Part II §18 numbering):

  #1 Wiki ↔ PDS bidirectional          (connections/pds_bridge.py)
  #2 Phronesis ↔ Homeostasis           (connections/phronesis_bridge.py)
  #4 Prediction-errors → Self-training (connections/training_signal.py)
  #6 Firecrawl → Predictor closed loop (connections/firecrawl_predictor.py)
  #7 DGM ↔ Homeostasis felt constraint (connections/dgm_felt_constraint.py)

Plus: external-service circuit-breaker registry
      (connections/service_health.py).

Phase 10 exit criteria: all seven SIA connections fire; no single
external outage cascades unrecoverably.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

import pytest

from app.subia.connections.dgm_felt_constraint import (
    apply_dgm_felt_constraint,
)
from app.subia.connections.firecrawl_predictor import (
    build_channel_key,
    record_firecrawl_outcome,
)
from app.subia.connections.pds_bridge import PDSBridge
from app.subia.connections.phronesis_bridge import (
    apply_phronesis_event,
    event_policy,
    registered_events,
)
from app.subia.connections.service_health import (
    ServiceHealthRegistry,
    State,
    apply_service_health_signal,
    get_registry,
    reset_singleton as reset_service_singleton,
)
from app.subia.connections.training_signal import (
    TrainingSignalEmitter,
    get_emitter,
    reset_singleton as reset_training_singleton,
)
from app.subia.kernel import HomeostaticState, Prediction, SubjectivityKernel


# ── SIA #1: PDS bridge ────────────────────────────────────────────

class FakePDS:
    def __init__(self, params=None):
        self.params = dict(params or {"love_of_learning": 0.5})

    def dimension_known(self, name):
        return name in self.params

    def get_parameter(self, name):
        return self.params[name]

    def set_parameter(self, name, value):
        self.params[name] = value


class TestPDSBridge:
    def test_nudge_within_caps(self):
        pds = FakePDS()
        bridge = PDSBridge(pds)
        result = bridge.apply_nudge("love_of_learning", 0.01,
                                     reason="high-quality CI")
        assert not result.clamped
        assert result.applied_delta == 0.01
        assert pds.params["love_of_learning"] == 0.51

    def test_nudge_exceeds_per_loop_cap(self):
        pds = FakePDS()
        bridge = PDSBridge(pds)
        result = bridge.apply_nudge("love_of_learning", 0.10)
        # Per-loop cap is ±0.02
        assert result.clamped
        assert abs(result.applied_delta) == 0.02

    def test_unknown_dimension_rejected(self):
        bridge = PDSBridge(FakePDS())
        result = bridge.apply_nudge("not_a_dim", 0.01)
        assert result.clamped
        assert result.applied_delta == 0.0
        assert "unknown" in result.reason.lower()

    def test_empty_parameter_rejected(self):
        bridge = PDSBridge(FakePDS())
        result = bridge.apply_nudge("", 0.01)
        assert result.applied_delta == 0.0

    def test_non_numeric_delta_rejected(self):
        bridge = PDSBridge(FakePDS())
        result = bridge.apply_nudge("love_of_learning", "huge")
        assert result.applied_delta == 0.0
        assert "non-numeric" in result.reason.lower()

    def test_weekly_budget_exhausts(self):
        pds = FakePDS()
        bridge = PDSBridge(pds)
        now = datetime(2026, 4, 14, tzinfo=timezone.utc)
        # Apply 6 loop-cap nudges at spaced times (≈0.12 total absolute)
        # — weekly cap is 0.10 so we should saturate.
        for i in range(7):
            bridge.apply_nudge(
                "love_of_learning", 0.02,
                now=now + timedelta(minutes=i),
            )
        # Next nudge in same week should be clamped further to 0
        result = bridge.apply_nudge(
            "love_of_learning", 0.02,
            now=now + timedelta(minutes=8),
        )
        assert result.applied_delta == 0.0
        assert "weekly" in result.reason.lower()

    def test_dry_run_mode(self):
        bridge = PDSBridge(pds=None)  # no backend
        result = bridge.apply_nudge("anything", 0.01)
        # No PDS to validate dimension; accepted + applied in-memory
        # (but new_value is None because no backend)
        assert result.applied_delta != 0.0
        assert result.new_value is None

    def test_to_dict_shape(self):
        bridge = PDSBridge(FakePDS())
        bridge.apply_nudge("love_of_learning", 0.01)
        payload = bridge.to_dict()
        assert "parameters" in payload
        assert "weekly_usage" in payload
        assert payload["max_per_loop"] == 0.02


# ── SIA #2: Phronesis bridge ─────────────────────────────────────

class TestPhronesisBridge:
    def _make_kernel(self):
        k = SubjectivityKernel()
        k.homeostasis = HomeostaticState(
            variables={"safety": 0.80, "trustworthiness": 0.70,
                       "social_alignment": 0.60, "overload": 0.30},
        )
        return k

    def test_commitment_breach_penalizes_trust(self):
        k = self._make_kernel()
        audit_calls = []
        result = apply_phronesis_event(
            k, "commitment_breach",
            narrative_audit_fn=lambda **kw: audit_calls.append(kw),
        )
        assert result.variable == "trustworthiness"
        assert result.applied_delta == -0.20
        assert k.homeostasis.variables["trustworthiness"] == 0.50
        assert audit_calls
        assert audit_calls[0]["severity"] == "warn"

    def test_epistemic_boundary_near_miss(self):
        k = self._make_kernel()
        result = apply_phronesis_event(
            k, "epistemic_boundary_near_miss",
            narrative_audit_fn=lambda **kw: None,
        )
        assert result.variable == "safety"
        assert k.homeostasis.variables["safety"] == 0.65  # 0.80 - 0.15

    def test_humanist_violation(self):
        k = self._make_kernel()
        apply_phronesis_event(
            k, "humanist_principle_violated",
            narrative_audit_fn=lambda **kw: None,
        )
        assert k.homeostasis.variables["social_alignment"] == 0.35

    def test_successful_commitment_restores_trust(self):
        k = self._make_kernel()
        k.homeostasis.variables["trustworthiness"] = 0.50
        apply_phronesis_event(
            k, "successful_commitment",
            narrative_audit_fn=lambda **kw: None,
        )
        assert k.homeostasis.variables["trustworthiness"] == 0.55

    def test_unknown_event_rejected(self):
        k = self._make_kernel()
        result = apply_phronesis_event(k, "made_up_event",
                                        narrative_audit_fn=lambda **kw: None)
        assert result.clamped
        assert "unknown" in result.reason.lower()
        # No variable modified
        assert k.homeostasis.variables["safety"] == 0.80

    def test_homeostasis_missing_safe(self):
        k = SubjectivityKernel()
        k.homeostasis = None
        result = apply_phronesis_event(
            k, "commitment_breach",
            narrative_audit_fn=lambda **kw: None,
        )
        assert result.applied_delta == -0.20   # set before update
        assert "no homeostasis" in result.reason

    def test_variable_clamps_to_unit_range(self):
        k = self._make_kernel()
        k.homeostasis.variables["safety"] = 0.05
        apply_phronesis_event(
            k, "epistemic_boundary_near_miss",
            narrative_audit_fn=lambda **kw: None,
        )
        # 0.05 - 0.15 → clamped to 0.0
        assert k.homeostasis.variables["safety"] == 0.0

    def test_event_registry(self):
        events = registered_events()
        assert "commitment_breach" in events
        assert "successful_recovery" in events
        assert event_policy("commitment_breach") == ("trustworthiness", -0.20)
        assert event_policy("not_a_thing") is None


# ── SIA #4: Training-signal emitter ──────────────────────────────

class FakeTracker:
    def __init__(self, domains):
        self._domains = domains   # list of (domain, sustained_bool)

    def all_domains_summary(self):
        return {"domains": [
            {"domain": d, "mean_accuracy": 0.3, "n_samples": 20,
             "recent_bad_count": 8}
            for d, _ in self._domains
        ]}

    def has_sustained_error(self, domain):
        for d, sustained in self._domains:
            if d == domain:
                return sustained
        return False


class TestTrainingSignalEmitter:
    def test_emits_for_sustained_domains(self, tmp_path):
        reset_training_singleton()
        emitter = TrainingSignalEmitter(queue_path=tmp_path / "q.jsonl")
        tracker = FakeTracker([
            ("researcher:ingest", True),
            ("coder:lint", False),
        ])
        signals = emitter.emit_from_tracker(tracker, loop_count=10)
        assert len(signals) == 1
        assert signals[0].domain == "researcher:ingest"

    def test_dedup_within_24h(self, tmp_path):
        emitter = TrainingSignalEmitter(queue_path=tmp_path / "q.jsonl")
        tracker = FakeTracker([("d", True)])
        now1 = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
        s1 = emitter.emit_from_tracker(tracker, 1, now=now1)
        # Same call again 5 min later — deduped
        now2 = now1 + timedelta(minutes=5)
        s2 = emitter.emit_from_tracker(tracker, 2, now=now2)
        assert len(s1) == 1
        assert len(s2) == 0

    def test_dedup_expires_after_24h(self, tmp_path):
        emitter = TrainingSignalEmitter(queue_path=tmp_path / "q.jsonl")
        tracker = FakeTracker([("d", True)])
        now1 = datetime(2026, 4, 14, tzinfo=timezone.utc)
        emitter.emit_from_tracker(tracker, 1, now=now1)
        now2 = now1 + timedelta(hours=25)
        s = emitter.emit_from_tracker(tracker, 2, now=now2)
        assert len(s) == 1

    def test_queue_jsonl_valid(self, tmp_path):
        path = tmp_path / "q.jsonl"
        emitter = TrainingSignalEmitter(queue_path=path)
        tracker = FakeTracker([("researcher:ingest", True)])
        emitter.emit_from_tracker(tracker, loop_count=7)
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert lines
        payload = json.loads(lines[0])
        assert payload["domain"] == "researcher:ingest"
        assert payload["loop_count"] == 7
        assert payload["reason"] == "sustained_prediction_error"

    def test_no_tracker_no_emit(self, tmp_path):
        emitter = TrainingSignalEmitter(queue_path=tmp_path / "q.jsonl")
        assert emitter.emit_from_tracker(None, 0) == []

    def test_singleton(self):
        reset_training_singleton()
        a = get_emitter()
        b = get_emitter()
        assert a is b
        reset_training_singleton()

    def test_read_recent(self, tmp_path):
        emitter = TrainingSignalEmitter(queue_path=tmp_path / "q.jsonl")
        tracker = FakeTracker([("d1", True), ("d2", True)])
        emitter.emit_from_tracker(tracker, 1)
        rows = emitter.read_recent()
        assert len(rows) == 2


# ── SIA #6: Firecrawl → Predictor ────────────────────────────────

class FakePredictiveLayer:
    def __init__(self):
        self.calls = []

    def predict_and_compare(self, *, channel, context, actual_content,
                             actual_embedding=None):
        self.calls.append({
            "channel": channel, "context": context,
            "actual_content": actual_content,
        })
        # Return a stub with expected fields.
        from app.subia.prediction.layer import PredictionError
        # Simulate a MAJOR surprise for long content, EXPECTED otherwise.
        level = "MAJOR_SURPRISE" if len(actual_content) > 50 else "EXPECTED"
        return PredictionError(
            channel=channel,
            error_magnitude=0.7 if level == "MAJOR_SURPRISE" else 0.1,
            effective_surprise=0.7 if level == "MAJOR_SURPRISE" else 0.1,
            surprise_level=level,
            routed_to_workspace=(level != "EXPECTED"),
        )


class TestFirecrawlPredictor:
    def test_closed_loop_records_error(self):
        layer = FakePredictiveLayer()
        out = record_firecrawl_outcome(
            source_url="https://example.com/truepic-series-c",
            channel=build_channel_key("archibal", "truepic"),
            actual_content="Truepic announced its Series C raise today " * 5,
            predictive_layer=layer,
            context="Expected Truepic Series C announcement this week",
        )
        assert out.prediction_error_recorded
        assert out.surprise_level == "MAJOR_SURPRISE"
        assert out.routed_to_workspace
        assert len(layer.calls) == 1
        assert layer.calls[0]["channel"] == "firecrawl:archibal:truepic"

    def test_expected_content_no_routing(self):
        layer = FakePredictiveLayer()
        out = record_firecrawl_outcome(
            source_url="https://example.com/",
            channel=build_channel_key("plg", "weekly"),
            actual_content="nothing",
            predictive_layer=layer,
        )
        assert out.prediction_error_recorded
        assert out.surprise_level == "EXPECTED"
        assert not out.routed_to_workspace

    def test_no_predictive_layer_graceful(self):
        out = record_firecrawl_outcome(
            source_url="x", channel="y",
            actual_content="z", predictive_layer=None,
        )
        assert not out.prediction_error_recorded
        assert "not attached" in out.reason

    def test_broken_predictor_graceful(self):
        class Broken:
            def predict_and_compare(self, **kw):
                raise RuntimeError("boom")
        out = record_firecrawl_outcome(
            source_url="x", channel="y",
            actual_content="z", predictive_layer=Broken(),
        )
        assert not out.prediction_error_recorded
        assert "raised" in out.reason.lower()

    def test_channel_key_canonical(self):
        assert build_channel_key("Archibal", " truepic ") == (
            "firecrawl:archibal:truepic"
        )
        assert build_channel_key("", "") == "firecrawl:unknown:"


# ── SIA #7: DGM felt-constraint ──────────────────────────────────

class TestDGMFeltConstraint:
    def _make_kernel(self, safety=0.80):
        k = SubjectivityKernel()
        k.homeostasis = HomeostaticState(
            variables={"safety": safety},
        )
        return k

    def test_all_green_gives_small_positive(self):
        k = self._make_kernel()

        def integrity_ok():
            return MagicMock(ok=True, missing=[])

        def scorecard_ok():
            return {"butlin": {"by_status": {"FAIL": 0}},
                    "rsm": {"by_status": {"FAIL": 0}},
                    "sk": {"by_status": {"FAIL": 0}}}

        audit_calls = []
        res = apply_dgm_felt_constraint(
            k,
            integrity_checker=integrity_ok,
            scorecard_runner=scorecard_ok,
            narrative_audit_fn=lambda **kw: audit_calls.append(kw),
        )
        assert res.integrity_ok
        assert res.probes_ok
        assert "all_green" in res.signals
        assert res.safety_delta > 0
        assert k.homeostasis.variables["safety"] > 0.80
        assert audit_calls

    def test_integrity_drift_penalizes(self):
        k = self._make_kernel()

        def integrity_bad():
            return MagicMock(ok=False, missing=["app/x.py"])

        def scorecard_ok():
            return {"butlin": {"by_status": {"FAIL": 0}},
                    "rsm": {"by_status": {"FAIL": 0}},
                    "sk": {"by_status": {"FAIL": 0}}}

        res = apply_dgm_felt_constraint(
            k,
            integrity_checker=integrity_bad,
            scorecard_runner=scorecard_ok,
            narrative_audit_fn=lambda **kw: None,
        )
        assert not res.integrity_ok
        assert "integrity_drift" in res.signals
        assert res.safety_delta < 0
        assert k.homeostasis.variables["safety"] < 0.80

    def test_missing_manifest_weaker_penalty(self):
        k = self._make_kernel()

        def integrity_missing():
            return MagicMock(ok=False, missing=["<MANIFEST>"])

        def scorecard_ok():
            return {"butlin": {"by_status": {"FAIL": 0}},
                    "rsm": {"by_status": {"FAIL": 0}},
                    "sk": {"by_status": {"FAIL": 0}}}

        res = apply_dgm_felt_constraint(
            k,
            integrity_checker=integrity_missing,
            scorecard_runner=scorecard_ok,
            narrative_audit_fn=lambda **kw: None,
        )
        assert "manifest_missing" in res.signals
        # -0.10 vs -0.20 for drift
        assert res.safety_delta == pytest.approx(-0.10, abs=0.01)

    def test_probe_failures_penalize(self):
        k = self._make_kernel()

        def integrity_ok():
            return MagicMock(ok=True, missing=[])

        def scorecard_with_fails():
            return {"butlin": {"by_status": {"FAIL": 2}},
                    "rsm": {"by_status": {"FAIL": 0}},
                    "sk": {"by_status": {"FAIL": 1}}}

        res = apply_dgm_felt_constraint(
            k,
            integrity_checker=integrity_ok,
            scorecard_runner=scorecard_with_fails,
            narrative_audit_fn=lambda **kw: None,
        )
        assert not res.probes_ok
        assert any("probe_failures=3" in s for s in res.signals)

    def test_per_call_cap_applied(self):
        """Multiple drift signals can't drive safety down more than 0.30."""
        k = self._make_kernel()

        def integrity_bad():
            return MagicMock(ok=False, missing=["x", "y", "z"])

        def scorecard_with_many_fails():
            return {"butlin": {"by_status": {"FAIL": 5}},
                    "rsm": {"by_status": {"FAIL": 2}},
                    "sk": {"by_status": {"FAIL": 3}}}

        res = apply_dgm_felt_constraint(
            k,
            integrity_checker=integrity_bad,
            scorecard_runner=scorecard_with_many_fails,
            narrative_audit_fn=lambda **kw: None,
        )
        assert res.safety_delta == pytest.approx(-0.30, abs=0.001)

    def test_no_homeostasis_graceful(self):
        k = SubjectivityKernel()
        k.homeostasis = None
        res = apply_dgm_felt_constraint(k)
        assert "no homeostasis" in res.reason


# ── Service-health circuit breaker ───────────────────────────────

class TestServiceHealth:
    def setup_method(self):
        reset_service_singleton()

    def test_closed_by_default(self):
        reg = ServiceHealthRegistry()
        assert reg.status("anthropic").state == State.CLOSED
        assert reg.open_services() == []

    def test_trips_open_after_threshold(self):
        reg = ServiceHealthRegistry()
        for _ in range(5):
            reg.report_failure("openrouter")
        assert reg.status("openrouter").state == State.OPEN
        assert "openrouter" in reg.open_services()

    def test_success_closes_and_marks_recovery(self):
        reg = ServiceHealthRegistry()
        for _ in range(5):
            reg.report_failure("firestore")
        assert reg.status("firestore").state == State.OPEN
        reg.report_success("firestore")
        st = reg.status("firestore")
        assert st.state == State.CLOSED
        assert st.recovery_pending

    def test_consume_recoveries_clears_flag(self):
        reg = ServiceHealthRegistry()
        for _ in range(5):
            reg.report_failure("anthropic")
        reg.report_success("anthropic")
        recovered = reg.consume_recoveries()
        assert "anthropic" in recovered
        # Second consume is empty
        assert reg.consume_recoveries() == []

    def test_guarded_call_skips_when_open(self):
        reg = ServiceHealthRegistry()
        for _ in range(5):
            reg.report_failure("x")
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        out = reg.guarded_call("x", fn)
        assert out is None
        assert calls == []   # not invoked

    def test_guarded_call_records_failure(self):
        reg = ServiceHealthRegistry()

        def fn():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            reg.guarded_call("x", fn)
        assert reg.status("x").consecutive_failures == 1

    def test_guarded_call_records_success(self):
        reg = ServiceHealthRegistry()
        assert reg.guarded_call("x", lambda: 42) == 42
        assert reg.status("x").consecutive_failures == 0

    def test_apply_signal_penalizes_open_services(self):
        k = SubjectivityKernel()
        k.homeostasis = HomeostaticState(variables={"safety": 0.80})
        reg = ServiceHealthRegistry()
        for _ in range(5):
            reg.report_failure("anthropic")
        out = apply_service_health_signal(k, registry=reg)
        assert "anthropic" in out["open_services"]
        assert out["safety_delta"] < 0
        assert k.homeostasis.variables["safety"] < 0.80

    def test_apply_signal_clamp(self):
        """Many open services can't drop safety below clamp (-0.20)."""
        k = SubjectivityKernel()
        k.homeostasis = HomeostaticState(variables={"safety": 0.50})
        reg = ServiceHealthRegistry()
        for svc in ("a", "b", "c", "d", "e", "f"):
            for _ in range(5):
                reg.report_failure(svc)
        out = apply_service_health_signal(k, registry=reg)
        assert out["safety_delta"] == pytest.approx(-0.20, abs=0.001)

    def test_singleton(self):
        reset_service_singleton()
        a = get_registry()
        b = get_registry()
        assert a is b
        reset_service_singleton()


# ── Loop integration ────────────────────────────────────────────

class TestLoopIntegration:
    def test_step11_exercises_phase10_bridges(self, tmp_path):
        """When reflect runs, dgm_felt + service_health + training_signal
        results appear in step details. Runs with minimal fixtures.
        """
        from app.subia.loop import SubIALoop
        from app.subia.scene.buffer import CompetitiveGate
        from unittest.mock import patch

        def predict(_ctx):
            return Prediction(
                id="p", operation="o", predicted_outcome={},
                predicted_self_change={}, predicted_homeostatic_effect={},
                confidence=0.7, created_at="",
            )

        loop = SubIALoop(
            kernel=SubjectivityKernel(),
            scene_gate=CompetitiveGate(capacity=5),
            predict_fn=predict,
        )
        loop.kernel.loop_count = 10  # match NARRATIVE_DRIFT_CHECK_FREQUENCY

        with patch(
            "app.subia.wiki_surface.consciousness_state.CONSCIOUSNESS_STATE",
            tmp_path / "c.md",
        ):
            result = loop.post_task(
                agent_role="researcher", task_description="x",
                operation_type="task_execute",
                task_result={"summary": "done"},
            )

        reflect = result.step("11_reflect")
        assert reflect is not None
        # Phase 10 signals present in details
        assert "dgm_felt" in reflect.details
        assert "service_health" in reflect.details
