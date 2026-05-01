"""
Phase 2: PP-1 half-circuit closure regression tests.

Before Phase 2 the PredictiveLayer computed prediction errors and set
`error.routed_to_workspace = True` for NOTABLE/MAJOR/PARADIGM surprises,
but no code ever consumed that flag. My forensic analysis flagged this
as the single biggest "half-circuit" in the consciousness surface —
a computed signal with no downstream consumer.

This test file asserts the closed-loop behaviour of the new
app.subia.prediction.surprise_routing module:

  1. A routed prediction error of sufficient magnitude produces a
     WorkspaceItem and submits it to the CompetitiveGate.
  2. A prediction error without the flag is a no-op.
  3. A prediction error below the router's own floor is a no-op even
     if the flag is set.
  4. Surprise items inherit urgency proportional to surprise_level.
  5. At capacity, a high-surprise error displaces the lowest-salience
     active item — the GWT bottleneck actually notices surprise.
  6. Metadata preserves the round-trip (error_id, level, magnitude).

These tests are the machinery by which PP-1 moves from PARTIAL
(half-circuit) to STRONG on the Butlin scorecard.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub out chromadb / pgvector / control_plane deps that buffer.py
# and layer.py transitively import but don't need for these tests.
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

from app.subia.prediction.layer import PredictionError
from app.subia.prediction.surprise_routing import (
    _MIN_ROUTABLE_SURPRISE,
    _NOVELTY_FOR_SURPRISE,
    _URGENCY_FOR_MAJOR,
    _URGENCY_FOR_NOTABLE,
    _URGENCY_FOR_PARADIGM,
    route_surprise_to_gate,
)
from app.subia.scene.buffer import CompetitiveGate, WorkspaceItem


def _major_error(effective: float = 0.7) -> PredictionError:
    return PredictionError(
        error_id="test-major",
        prediction_id="pred-x",
        channel="researcher_output",
        actual_summary="unexpected API change",
        error_magnitude=0.8,
        effective_surprise=effective,
        surprise_level="MAJOR_SURPRISE",
        routed_to_workspace=True,
    )


# ── Core routing behaviour ─────────────────────────────────────────

class TestRouting:
    def test_routed_major_surprise_is_admitted_when_gate_empty(self):
        gate = CompetitiveGate(capacity=5)
        error = _major_error(effective=0.7)

        result = route_surprise_to_gate(error, gate)

        assert result is not None
        assert result.admitted
        assert result.transition_type == "admitted"

    def test_unrouted_error_is_noop(self):
        """PredictiveLayer did not flag this as routable. Router must not act."""
        gate = CompetitiveGate(capacity=5)
        error = PredictionError(
            error_id="low",
            channel="x",
            effective_surprise=0.7,
            surprise_level="MAJOR_SURPRISE",
            routed_to_workspace=False,   # <- unrouted
        )
        assert route_surprise_to_gate(error, gate) is None
        assert len(gate._active) == 0

    def test_below_floor_effective_surprise_is_noop(self):
        """Even a flagged error must be above _MIN_ROUTABLE_SURPRISE."""
        gate = CompetitiveGate(capacity=5)
        error = PredictionError(
            error_id="weak",
            channel="x",
            effective_surprise=_MIN_ROUTABLE_SURPRISE - 0.01,
            surprise_level="NOTABLE_SURPRISE",
            routed_to_workspace=True,
        )
        assert route_surprise_to_gate(error, gate) is None
        assert len(gate._active) == 0

    def test_routed_item_carries_surprise_signal(self):
        gate = CompetitiveGate(capacity=5)
        error = _major_error(effective=0.75)

        route_surprise_to_gate(error, gate)

        assert len(gate._active) == 1
        item = gate._active[0]
        assert item.surprise_signal == 0.75
        assert item.source_agent == "predictive_layer"
        assert item.source_channel.startswith("pp1:")
        assert item.novelty_score == _NOVELTY_FOR_SURPRISE


# ── Urgency mapping by surprise level ──────────────────────────────

class TestUrgencyScaling:
    def test_notable_urgency(self):
        gate = CompetitiveGate(capacity=5)
        error = PredictionError(
            error_id="n",
            channel="x",
            effective_surprise=0.5,
            surprise_level="NOTABLE_SURPRISE",
            routed_to_workspace=True,
        )
        route_surprise_to_gate(error, gate)
        assert gate._active[0].agent_urgency == _URGENCY_FOR_NOTABLE

    def test_major_urgency(self):
        gate = CompetitiveGate(capacity=5)
        route_surprise_to_gate(_major_error(effective=0.6), gate)
        assert gate._active[0].agent_urgency == _URGENCY_FOR_MAJOR

    def test_paradigm_urgency(self):
        gate = CompetitiveGate(capacity=5)
        error = PredictionError(
            error_id="p",
            channel="x",
            effective_surprise=0.9,
            surprise_level="PARADIGM_VIOLATION",
            routed_to_workspace=True,
        )
        route_surprise_to_gate(error, gate)
        assert gate._active[0].agent_urgency == _URGENCY_FOR_PARADIGM

    def test_paradigm_is_most_urgent(self):
        assert _URGENCY_FOR_PARADIGM > _URGENCY_FOR_MAJOR > _URGENCY_FOR_NOTABLE


# ── GWT bottleneck: surprise actually competes for admission ───────

class TestBottleneckCompetition:
    def _fill_gate_with_low_salience(self, gate: CompetitiveGate) -> None:
        for i in range(gate.capacity):
            gate.evaluate(WorkspaceItem(
                item_id=f"bg-{i}",
                content=f"background item {i}",
                salience_score=0.10,
                agent_urgency=0.10,
                novelty_score=0.10,
            ))

    def test_high_surprise_displaces_low_salience(self):
        gate = CompetitiveGate(capacity=3)
        self._fill_gate_with_low_salience(gate)
        assert len(gate._active) == 3

        result = route_surprise_to_gate(
            _major_error(effective=0.85),
            gate,
        )

        assert result is not None
        assert result.admitted
        assert result.transition_type == "displaced"
        # Surprise item should be in gate; one of the originals in peripheral.
        assert any(i.source_agent == "predictive_layer" for i in gate._active)
        assert result.displaced_item is not None
        assert result.displaced_item.item_id.startswith("bg-")

    def test_surprise_signal_weighted_in_salience(self):
        """The SalienceScorer weights surprise_signal — so a high surprise
        item should out-compete a low-surprise item with similar other
        fields. This is the core PP-1 criterion.
        """
        gate = CompetitiveGate(capacity=2)
        # Two background items, no surprise
        gate.evaluate(WorkspaceItem(
            item_id="bg-a", salience_score=0.20, agent_urgency=0.20,
        ))
        gate.evaluate(WorkspaceItem(
            item_id="bg-b", salience_score=0.22, agent_urgency=0.22,
        ))

        # A MAJOR surprise error: composite salience should include
        # 0.25 × surprise_signal(0.70) + 0.25 × novelty(0.80)
        #   + 0.15 × urgency(0.80) = 0.49
        # which beats 0.22.
        route_surprise_to_gate(_major_error(effective=0.70), gate)

        assert any(i.source_agent == "predictive_layer" for i in gate._active)


# ── Metadata preservation ──────────────────────────────────────────

class TestMetadata:
    def test_metadata_round_trips_error_id(self):
        gate = CompetitiveGate(capacity=5)
        error = _major_error(effective=0.75)
        route_surprise_to_gate(error, gate, context="Truepic unexpected raise")

        item = gate._active[0]
        assert item.metadata["pp1_error_id"] == "test-major"
        assert item.metadata["pp1_surprise_level"] == "MAJOR_SURPRISE"
        assert item.metadata["pp1_channel"] == "researcher_output"
        assert "unexpected" in item.content.lower()


# ── Butlin PP-1 acceptance ─────────────────────────────────────────

class TestPP1Acceptance:
    """These are the tests that, taken together, move PP-1 on the
    Butlin scorecard from PARTIAL (half-circuit) to STRONG.
    """

    def test_surprise_signal_is_consumed_not_just_logged(self):
        """The flag set by PredictiveLayer now produces a gate submission."""
        gate = CompetitiveGate(capacity=5)
        initial = len(gate._active)

        route_surprise_to_gate(_major_error(), gate)

        # The half-circuit is closed iff the gate's state changed.
        assert len(gate._active) > initial

    def test_unrouted_does_not_pollute_gate(self):
        """Negative case: unrouted errors must NOT leak into the scene."""
        gate = CompetitiveGate(capacity=5)
        for _ in range(10):
            unrouted = PredictionError(
                channel="x", effective_surprise=0.9,
                surprise_level="MAJOR_SURPRISE",
                routed_to_workspace=False,
            )
            assert route_surprise_to_gate(unrouted, gate) is None
        assert len(gate._active) == 0


# ── End-to-end integration via PredictiveLayer.set_gate ───────────

class TestPredictiveLayerIntegration:
    """The full half-circuit closure: PredictiveLayer with a gate attached
    automatically routes high-surprise errors via surprise_routing.

    Before Phase 2, PredictiveLayer.predict_and_compare() set
    error.routed_to_workspace = True and returned. Nothing changed in
    the scene. After Phase 2, with set_gate(), the same call populates
    the gate's active set.
    """

    def test_layer_without_gate_does_not_break(self):
        """Legacy behaviour: no gate attached, no routing. No crash."""
        from app.subia.prediction.layer import PredictiveLayer
        layer = PredictiveLayer(surprise_budget_per_cycle=2)
        # ChannelPredictor.generate_prediction + compute_error call
        # chromadb_manager.embed which is MagicMocked above. They'll
        # return something noisy but non-crashing.
        err = layer.predict_and_compare(
            channel="test",
            context="input A",
            actual_content="actual A",
        )
        # The flag may or may not be set depending on the mocked
        # embedding; the important invariant is no exception and no
        # gate was affected.
        assert isinstance(err, PredictionError)

    def test_layer_with_gate_routes_major_surprise_end_to_end(self):
        """Integration: attach gate, simulate a MAJOR surprise, observe
        the gate receives a WorkspaceItem from the predictive_layer agent.
        """
        from app.subia.prediction.layer import PredictiveLayer
        from unittest.mock import patch

        gate = CompetitiveGate(capacity=5)
        layer = PredictiveLayer(surprise_budget_per_cycle=2)
        layer.set_gate(gate)

        # Inject a deterministic MAJOR-surprise outcome by stubbing
        # compute_error to return a high-effective-surprise event.
        stub_error = _major_error(effective=0.75)
        with patch.object(layer, "_persist_error", return_value=None):
            pred = layer._predictors  # touch to force lazy init below
            predictor = layer.get_predictor("test_channel")
            with patch.object(predictor, "generate_prediction") as gen, \
                 patch.object(predictor, "compute_error",
                              return_value=stub_error):
                gen.return_value = MagicMock()
                result = layer.predict_and_compare(
                    channel="test_channel",
                    context="expected X",
                    actual_content="got Y",
                )

        assert result.routed_to_workspace is True
        # The gate should now contain a predictive_layer-sourced item
        assert any(i.source_agent == "predictive_layer"
                   for i in gate._active), "surprise did not reach the scene"

    def test_routing_budget_still_enforced_with_gate(self):
        """surprise_budget_per_cycle must still cap the number of
        surprises routed per cycle even when a gate is attached.
        """
        from app.subia.prediction.layer import PredictiveLayer
        from unittest.mock import patch

        gate = CompetitiveGate(capacity=10)
        layer = PredictiveLayer(surprise_budget_per_cycle=2)
        layer.set_gate(gate)

        with patch.object(layer, "_persist_error", return_value=None):
            predictor = layer.get_predictor("ch")
            with patch.object(predictor, "generate_prediction",
                              return_value=MagicMock()), \
                 patch.object(predictor, "compute_error") as ce:
                ce.side_effect = lambda *_args, **_kw: _major_error(
                    effective=0.70,
                )
                # Three major surprises in one cycle; budget=2.
                layer.predict_and_compare("ch", "a", "b")
                layer.predict_and_compare("ch", "a", "b")
                layer.predict_and_compare("ch", "a", "b")

        # Only the first two should have reached the gate.
        pp1_items = [i for i in gate._active
                     if i.source_agent == "predictive_layer"]
        assert len(pp1_items) == 2, (
            f"budget=2 but {len(pp1_items)} PP-1 items in gate"
        )

    def test_routing_never_crashes_layer(self):
        """If the routing bridge raises, predict_and_compare must still
        return the PredictionError — safety invariant.
        """
        from app.subia.prediction.layer import PredictiveLayer
        from unittest.mock import patch

        gate = CompetitiveGate(capacity=5)
        layer = PredictiveLayer(surprise_budget_per_cycle=2)
        layer.set_gate(gate)

        # Force the bridge to explode on this call path.
        with patch.object(layer, "_persist_error", return_value=None):
            predictor = layer.get_predictor("ch")
            with patch.object(predictor, "generate_prediction",
                              return_value=MagicMock()), \
                 patch.object(predictor, "compute_error",
                              return_value=_major_error(effective=0.70)), \
                 patch(
                     "app.subia.prediction.surprise_routing.route_surprise_to_gate",
                     side_effect=RuntimeError("simulated routing crash")
                 ):
                err = layer.predict_and_compare("ch", "a", "b")

        assert err is not None
        assert err.routed_to_workspace is True  # Flag still set
