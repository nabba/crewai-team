"""
Phase 6: prediction-refinement regression tests.

Three subsystems:

  1. accuracy_tracker — per-domain rolling accuracy with wiki
     serialization. Records errors, computes rolling mean, detects
     sustained-error patterns.

  2. cascade policy — pure function combining single-prediction
     confidence, homeostatic coherence deviation, and sustained-error
     flag into an escalation recommendation.

  3. cache accuracy-driven eviction — cached templates whose recent
     accuracy falls below floor get evicted so live LLM refreshes.

Plus: loop integration. After Phase 6 the CIL loop feeds prediction
errors into the tracker (Step 8) and the tracker's sustained-error
signal into cascade modulation (Step 5b).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub heavy deps
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

import pytest

from app.subia.kernel import Prediction, SubjectivityKernel
from app.subia.prediction.accuracy_tracker import (
    AccuracyTracker,
    DomainStats,
    domain_key,
    get_tracker,
    record_prediction_error,
    reset_singleton,
)
from app.subia.prediction.cache import PredictionCache
from app.subia.prediction.cascade import (
    CascadeDecision,
    decide_cascade,
    highest_recommendation,
)


# ── AccuracyTracker: basics ──────────────────────────────────────

class TestAccuracyTrackerBasics:
    def test_new_tracker_empty(self):
        t = AccuracyTracker()
        stats = t.domain_stats("x")
        assert stats.n_samples == 0
        assert stats.mean_accuracy == 0.0

    def test_record_and_read_stats(self):
        t = AccuracyTracker()
        t.record_outcome("researcher:ingest", 0.2)
        t.record_outcome("researcher:ingest", 0.3)
        t.record_outcome("researcher:ingest", 0.1)
        stats = t.domain_stats("researcher:ingest")
        assert stats.n_samples == 3
        # accuracy = 1 - mean(0.2, 0.3, 0.1) = 0.8
        assert abs(stats.mean_accuracy - 0.8) < 0.01

    def test_accuracy_clamped(self):
        t = AccuracyTracker()
        t.record_outcome("d", -5.0)   # clamped to 0
        t.record_outcome("d", 10.0)   # clamped to 1
        stats = t.domain_stats("d")
        assert 0 <= stats.mean_accuracy <= 1

    def test_rolling_window_drops_old(self):
        t = AccuracyTracker(window_size=5)
        for i in range(20):
            t.record_outcome("d", 0.5)
        stats = t.domain_stats("d")
        assert stats.n_samples == 5   # capped to window

    def test_invalid_error_ignored(self):
        t = AccuracyTracker()
        t.record_outcome("d", "not a number")  # TypeError → ignored
        t.record_outcome("d", None)              # likewise
        stats = t.domain_stats("d")
        assert stats.n_samples == 0

    def test_empty_domain_ignored(self):
        t = AccuracyTracker()
        t.record_outcome("", 0.5)
        assert t.domain_stats("").n_samples == 0


# ── AccuracyTracker: sustained error signal ─────────────────────

class TestSustainedError:
    def test_no_sustained_error_by_default(self):
        t = AccuracyTracker()
        assert not t.has_sustained_error("researcher:ingest")

    def test_scattered_bad_errors_below_threshold(self):
        """Occasional bad errors don't trigger the signal."""
        t = AccuracyTracker()
        # 1 bad among 9 good
        for _ in range(9):
            t.record_outcome("d", 0.1)
        t.record_outcome("d", 0.9)
        assert not t.has_sustained_error("d", window=10, threshold=3)

    def test_clustered_bad_errors_trigger(self):
        """Three or more bad errors in the recent window trigger."""
        t = AccuracyTracker()
        for _ in range(7):
            t.record_outcome("d", 0.1)
        for _ in range(3):
            t.record_outcome("d", 0.8)
        assert t.has_sustained_error("d", window=10, threshold=3)

    def test_min_samples_enforced(self):
        """Even with all-bad errors, fewer than 3 samples doesn't fire."""
        t = AccuracyTracker()
        for _ in range(2):
            t.record_outcome("d", 1.0)
        assert not t.has_sustained_error("d", window=10, threshold=1)


# ── AccuracyTracker: wiki serialization ─────────────────────────

class TestWikiSerialization:
    def test_markdown_contains_frontmatter(self):
        t = AccuracyTracker()
        t.record_outcome("a:b", 0.3)
        md = t.serialize_to_wiki_markdown()
        assert md.startswith("---\n")
        assert 'title: "Prediction Accuracy' in md
        assert "# Prediction Accuracy" in md

    def test_markdown_has_global_mean(self):
        t = AccuracyTracker()
        t.record_outcome("a:b", 0.2)
        t.record_outcome("c:d", 0.1)
        md = t.serialize_to_wiki_markdown()
        assert "Global mean accuracy" in md
        # mean error = 0.15, accuracy = 0.85
        assert "0.850" in md

    def test_markdown_lists_domains(self):
        t = AccuracyTracker()
        t.record_outcome("researcher:ingest", 0.2)
        t.record_outcome("coder:lint", 0.1)
        md = t.serialize_to_wiki_markdown()
        assert "researcher:ingest" in md
        assert "coder:lint" in md

    def test_markdown_empty_state(self):
        t = AccuracyTracker()
        md = t.serialize_to_wiki_markdown()
        assert "_No predictions recorded yet._" in md

    def test_save_to_wiki_atomic(self, tmp_path):
        t = AccuracyTracker()
        t.record_outcome("d", 0.2)
        target = tmp_path / "acc.md"
        t.save_to_wiki(path=target)
        assert target.exists()
        assert "Prediction Accuracy" in target.read_text()

    def test_save_to_wiki_never_raises_on_bad_path(self):
        t = AccuracyTracker()
        t.record_outcome("d", 0.2)
        # Path inside a non-existent directory — safe_write will create
        # parents; we just need to confirm no exception propagates.
        # Using "/dev/null/nope" which is guaranteed-bad on Unix.
        t.save_to_wiki(path=Path("/dev/null/absolutely/not/valid/here.md"))


# ── AccuracyTracker: summary & singleton ────────────────────────

class TestTrackerSummary:
    def test_all_domains_summary_shape(self):
        t = AccuracyTracker()
        t.record_outcome("a:x", 0.2)
        t.record_outcome("b:y", 0.3)
        summary = t.all_domains_summary()
        assert summary["n_domains"] == 2
        assert len(summary["domains"]) == 2
        assert "global_mean_accuracy" in summary
        assert "window_size" in summary

    def test_singleton_stable(self):
        reset_singleton()
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2
        reset_singleton()

    def test_record_via_helper(self):
        reset_singleton()
        record_prediction_error("researcher", "ingest", 0.3)
        stats = get_tracker().domain_stats(domain_key("researcher", "ingest"))
        assert stats.n_samples == 1
        assert stats.last_error == 0.3
        reset_singleton()


# ── Domain-key canonicalization ──────────────────────────────────

class TestDomainKey:
    def test_canonical_form(self):
        assert domain_key("Researcher", " INGEST ") == "researcher:ingest"

    def test_empty_bits(self):
        # Still produces a valid key so reads/writes don't crash
        k = domain_key("", "")
        assert k == ":"


# ── Cascade decision policy ──────────────────────────────────────

class TestCascadeDecision:
    def test_high_confidence_maintains(self):
        d = decide_cascade(prediction_confidence=0.9)
        assert d.recommendation == "maintain"
        assert not d.escalated

    def test_low_confidence_escalates(self):
        d = decide_cascade(prediction_confidence=0.3)
        assert d.recommendation == "escalate"

    def test_very_low_confidence_premium(self):
        d = decide_cascade(prediction_confidence=0.05)
        assert d.recommendation == "escalate_premium"

    def test_homeostatic_deviation_escalates(self):
        d = decide_cascade(
            prediction_confidence=0.9,
            homeostatic_coherence_deviation=0.5,
        )
        assert d.recommendation == "escalate"

    def test_sustained_error_escalates(self):
        d = decide_cascade(
            prediction_confidence=0.9,
            sustained_error=True,
            domain="researcher:ingest",
        )
        assert d.recommendation == "escalate"
        assert any("sustained" in r for r in d.reasons)

    def test_two_signals_premium(self):
        """Low confidence + sustained error = premium (two signals)."""
        d = decide_cascade(
            prediction_confidence=0.3,
            sustained_error=True,
            domain="d",
        )
        assert d.recommendation == "escalate_premium"

    def test_disabled_config_returns_maintain(self):
        d = decide_cascade(
            prediction_confidence=0.05,
            sustained_error=True,
            config={"CASCADE_UNCERTAINTY_ESCALATION": False},
        )
        assert d.recommendation == "maintain"

    def test_decision_serializes(self):
        d = decide_cascade(
            prediction_confidence=0.3,
            sustained_error=True,
            domain="d",
        )
        payload = d.to_dict()
        assert payload["recommendation"] == "escalate_premium"
        assert "reasons" in payload
        assert payload["sustained_error"] is True

    def test_highest_recommendation_helper(self):
        assert highest_recommendation("maintain") == "maintain"
        assert highest_recommendation("maintain", "escalate") == "escalate"
        assert highest_recommendation(
            "maintain", "escalate", "escalate_premium",
        ) == "escalate_premium"


# ── Cache: accuracy-driven eviction ─────────────────────────────

class TestAccuracyEviction:
    def _mk_prediction(self, confidence: float = 0.7) -> Prediction:
        return Prediction(
            id="p-test", operation="o",
            predicted_outcome={}, predicted_self_change={},
            predicted_homeostatic_effect={},
            confidence=confidence, created_at="",
        )

    def test_good_accuracy_keeps_entry(self):
        cache = PredictionCache(
            min_uses=1, eviction_floor=0.2, eviction_min_uses=3,
        )
        sig = PredictionCache.signature("r", "i", ("t",))
        for _ in range(5):
            cache.store(sig, self._mk_prediction())
            cache.update_accuracy(sig, observed_accuracy=0.9)
        assert sig in cache._entries

    def test_sustained_bad_accuracy_evicts(self):
        cache = PredictionCache(
            min_uses=1, eviction_floor=0.3, eviction_min_uses=3,
        )
        sig = PredictionCache.signature("r", "i", ("t",))
        for _ in range(5):
            cache.store(sig, self._mk_prediction())
        # Feed a series of accuracy-0.05 observations
        for _ in range(5):
            cache.update_accuracy(sig, observed_accuracy=0.05)
        # Entry should be evicted since recent_accuracy fell below floor
        # AND use_count >= eviction_min_uses
        assert sig not in cache._entries
        assert cache.accuracy_evictions == 1

    def test_eviction_not_premature(self):
        """Eviction is gated on `eviction_min_uses` so a single bad
        early reading can't evict — this protects against noisy
        warm-up.
        """
        cache = PredictionCache(
            min_uses=1, eviction_floor=0.3, eviction_min_uses=10,
        )
        sig = PredictionCache.signature("r", "i", ("t",))
        cache.store(sig, self._mk_prediction())
        cache.update_accuracy(sig, 0.05)
        assert sig in cache._entries  # only 1 use
        assert cache.accuracy_evictions == 0

    def test_stats_exposes_evictions(self):
        cache = PredictionCache()
        stats = cache.stats()
        assert "accuracy_evictions" in stats
        assert "eviction_floor" in stats


# ── Loop integration ────────────────────────────────────────────

class TestLoopIntegration:
    def _mk_loop(self, tracker=None, gate=None, pred_layer=None, predict=None):
        from app.subia.loop import SubIALoop
        from app.subia.scene.buffer import CompetitiveGate

        def default_predict(_ctx):
            return Prediction(
                id="p", operation="o", predicted_outcome={},
                predicted_self_change={}, predicted_homeostatic_effect={},
                confidence=0.7, created_at="",
            )

        return SubIALoop(
            kernel=SubjectivityKernel(),
            scene_gate=gate or CompetitiveGate(capacity=5),
            predict_fn=predict or default_predict,
            predictive_layer=pred_layer,
            accuracy_tracker=tracker,
        )

    def test_pre_task_sets_current_domain(self):
        t = AccuracyTracker()
        loop = self._mk_loop(tracker=t)
        loop.pre_task(
            agent_role="researcher",
            task_description="x",
            operation_type="task_execute",
        )
        assert loop._current_domain == "researcher:task_execute"

    def test_compare_feeds_tracker(self):
        """Real PredictiveLayer — patch compute_error to return a big
        error, then verify the tracker recorded it.
        """
        from unittest.mock import patch
        from app.subia.prediction.layer import PredictionError, PredictiveLayer

        layer = PredictiveLayer(surprise_budget_per_cycle=2)
        tracker = AccuracyTracker()
        loop = self._mk_loop(tracker=tracker, pred_layer=layer)

        stub_error = PredictionError(
            error_id="x", channel="researcher", prediction_id="",
            effective_surprise=0.7, surprise_level="MAJOR_SURPRISE",
            error_magnitude=0.6,
            routed_to_workspace=False,
        )

        with patch.object(layer, "_persist_error", return_value=None):
            predictor = layer.get_predictor("researcher")
            with patch.object(predictor, "generate_prediction",
                              return_value=MagicMock()), \
                 patch.object(predictor, "compute_error",
                              return_value=stub_error):
                loop.post_task(
                    agent_role="researcher",
                    task_description="x",
                    operation_type="ingest",
                    task_result={"success": True, "summary": "s"},
                )

        stats = tracker.domain_stats("researcher:ingest")
        assert stats.n_samples == 1
        assert stats.last_error == pytest.approx(0.6, abs=0.01)

    def test_cascade_uses_sustained_error(self):
        """If the tracker has sustained error on current domain,
        cascade decision escalates.
        """
        tracker = AccuracyTracker()
        for _ in range(5):
            tracker.record_outcome("researcher:task_execute", 0.9)

        loop = self._mk_loop(tracker=tracker)
        result = loop.pre_task(
            agent_role="researcher",
            task_description="x",
            operation_type="task_execute",
        )
        cascade_rec = result.context_for_agent.get("cascade_recommendation")
        assert cascade_rec in ("escalate", "escalate_premium")
        step = result.step("5b_cascade")
        assert step is not None
        assert step.details["sustained_error"] is True

    def test_no_domain_no_cascade_impact(self):
        """Without a tracker, sustained_error signal is absent; cascade
        falls back to confidence-only behaviour (Phase 4 parity).
        """
        loop = self._mk_loop(tracker=AccuracyTracker())  # empty tracker
        result = loop.pre_task(
            agent_role="researcher",
            task_description="x",
            operation_type="task_execute",
        )
        step = result.step("5b_cascade")
        assert step.details["sustained_error"] is False
