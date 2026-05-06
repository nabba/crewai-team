"""Tests for the 11 general improvements modules.

Covers each module's primary public API plus key edge cases. Mocks ChromaDB,
Signal, and LLM dependencies — all modules must degrade gracefully when
those backends are unavailable.

Tests are organized by module:
  - TestSelfModel
  - TestEvolutionROI
  - TestPatternLibrary
  - TestGoodhartGuard
  - TestMutationStrategies
  - TestDifferentialTest
  - TestTierGraduation
  - TestAlignmentAudit
  - TestKnowledgeCompactor
  - TestImprovementNarrative
  - TestHumanGate
"""
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings  # noqa: E402
import app.config as config_mod  # noqa: E402

config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


# ─────────────────────────────────────────────────────────────────────────────
# self_model
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfModel:
    def test_module_node_immutable(self):
        from app.self_model import ModuleNode
        node = ModuleNode(
            path="app/foo.py",
            imports=("app.bar",),
            exports=("Foo",),
            line_count=42,
            is_hot_path=True,
            capability_tags=("evolution",),
        )
        assert node.path == "app/foo.py"
        assert node.is_hot_path is True
        # Frozen dataclass — assignment raises
        with pytest.raises(Exception):
            node.line_count = 99

    def test_extract_imports_and_exports_from_source(self):
        from app.self_model import _parse_imports_and_exports
        source = (
            "from app.foo import bar\n"
            "from app.baz import qux\n"
            "import os  # not a local import\n"
            "def public_fn(): pass\n"
            "def _private_fn(): pass\n"
            "class PublicCls: pass\n"
        )
        imports, exports = _parse_imports_and_exports(source)
        assert "app.foo" in imports
        assert "app.baz" in imports
        assert "os" not in imports
        assert "public_fn" in exports
        assert "PublicCls" in exports
        assert "_private_fn" not in exports

    def test_extract_capabilities_from_docstring(self):
        from app.self_model import _extract_capabilities
        source = '"""\nhealth_monitor.py — error rate, latency, hallucination tracking.\n"""'
        caps = _extract_capabilities(source)
        assert "monitoring" in caps or "error_handling" in caps

    def test_save_and_load_round_trip(self, tmp_path):
        from app.self_model import SelfModel, ModuleNode, save_self_model, load_self_model
        path = tmp_path / "model.json"
        original = SelfModel(
            modules={"a.py": ModuleNode("a.py", (), (), 1, True, ())},
            built_at=time.time(),
        )
        save_self_model(original, path)
        loaded = load_self_model(path)
        assert loaded is not None
        assert "a.py" in loaded.modules

    def test_classify_hot_paths_bfs(self):
        from app.self_model import _classify_hot_paths, ModuleNode
        modules = {
            "app/main.py": ModuleNode("app/main.py", ("app.foo",), (), 10, False, ()),
            "app/foo.py": ModuleNode("app/foo.py", ("app.bar",), (), 10, False, ()),
            "app/bar.py": ModuleNode("app/bar.py", (), (), 10, False, ()),
            "app/cold.py": ModuleNode("app/cold.py", (), (), 10, False, ()),
        }
        hot = _classify_hot_paths(modules, max_depth=2)
        assert "app/main.py" in hot
        assert "app/foo.py" in hot
        assert "app/bar.py" in hot
        assert "app/cold.py" not in hot


# ─────────────────────────────────────────────────────────────────────────────
# evolution_roi
# ─────────────────────────────────────────────────────────────────────────────

class TestEvolutionROI:
    def test_record_and_load(self, tmp_path, monkeypatch):
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")
        roi.record_evolution_cost(
            experiment_id="exp_1", engine="avo", cost_usd=0.10,
            delta=0.02, status="keep", deployed=True,
        )
        snapshot = roi.get_rolling_roi(days=1)
        assert snapshot.sample_size == 1
        assert snapshot.total_cost_usd == 0.10
        assert snapshot.real_improvements == 1

    def test_throttle_no_improvements(self, tmp_path, monkeypatch):
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")
        # Simulate 10 experiments over the last 14 days with no improvements
        now = time.time()
        for i in range(10):
            roi.record_evolution_cost(
                experiment_id=f"exp_{i}", engine="avo", cost_usd=0.10,
                delta=0.0, status="discard",
            )
        throttled, reason, factor = roi.should_throttle()
        assert throttled is True
        assert factor < 1.0
        assert "No real improvements" in reason or "No improvements" in reason or "improvements" in reason

    def test_throttle_healthy_state(self, tmp_path, monkeypatch):
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")
        # Simulate 5 experiments with real improvements
        for i in range(5):
            roi.record_evolution_cost(
                experiment_id=f"exp_{i}", engine="avo", cost_usd=0.10,
                delta=0.05, status="keep", deployed=True,
            )
        throttled, reason, factor = roi.should_throttle()
        assert throttled is False
        assert factor == 1.0

    def test_engine_recommendation(self, tmp_path, monkeypatch):
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")
        # AVO: 3 successes at $0.10 each → $0.033/improvement
        for i in range(3):
            roi.record_evolution_cost(
                experiment_id=f"avo_{i}", engine="avo", cost_usd=0.10,
                delta=0.05, status="keep",
            )
        # Shinka: 1 success at $1.00 → $1.00/improvement
        roi.record_evolution_cost(
            experiment_id="shinka_1", engine="shinka", cost_usd=1.00,
            delta=0.05, status="keep",
        )
        rec = roi.get_engine_recommendation()
        assert rec == "avo"  # better cost-per-improvement


# ─────────────────────────────────────────────────────────────────────────────
# pattern_library
# ─────────────────────────────────────────────────────────────────────────────

class TestPatternLibrary:
    def test_extract_pattern_below_threshold(self):
        from app.pattern_library import extract_pattern_from_experiment
        # delta=0.01 is below the 0.05 minimum
        experiment = {
            "experiment_id": "exp_low",
            "hypothesis": "small change",
            "delta": 0.01,
            "status": "keep",
            "detail": "tiny",
            "files_changed": ["x.py"],
        }
        result = extract_pattern_from_experiment(experiment)
        assert result is None

    def test_extract_pattern_above_threshold(self):
        from app.pattern_library import extract_pattern_from_experiment
        experiment = {
            "experiment_id": "exp_real",
            "hypothesis": "add caching to LLM calls",
            "delta": 0.08,
            "status": "keep",
            "detail": "reduced latency by 40%",
            "files_changed": ["app/cache.py"],
        }
        result = extract_pattern_from_experiment(experiment)
        assert result is not None
        assert "optimization" in result.target_categories

    def test_categorize_experiment_defensive(self):
        from app.pattern_library import _categorize_experiment
        e = {"hypothesis": "add retry with exponential backoff", "detail": "handle timeout"}
        cats = _categorize_experiment(e)
        assert "defensive" in cats

    def test_pattern_id_stable(self):
        from app.pattern_library import _compute_pattern_id
        a = _compute_pattern_id("Add caching to LLM calls")
        b = _compute_pattern_id("add caching to llm calls")  # same after lowering
        assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# goodhart_guard
# ─────────────────────────────────────────────────────────────────────────────

class TestGoodhartGuard:
    def test_loads_adversarial_tasks_from_workspace(self, monkeypatch):
        # In production the path is /app/workspace/, but in tests we point at the
        # actual checked-in file in the repo workspace/ directory.
        from pathlib import Path as _Path
        repo_path = _Path(__file__).parent.parent / "workspace" / "adversarial_tasks.json"
        import app.goodhart_guard as gg
        monkeypatch.setattr(gg, "ADVERSARIAL_TASKS_PATH", repo_path)
        tasks = gg._load_adversarial_tasks()
        assert len(tasks) >= 10
        for t in tasks[:5]:
            assert "task" in t
            assert "validation" in t
            assert "category" in t

    def test_detect_gaming_signals_kept_ratio_spike(self, monkeypatch):
        from app.goodhart_guard import detect_gaming_signals

        # Mock results: 90% kept but all delta=0
        cosmetic_results = []
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        for i in range(20):
            cosmetic_results.append({
                "ts": now_iso,
                "experiment_id": f"exp_{i}",
                "status": "keep" if i < 18 else "discard",
                "delta": 0.0,  # cosmetic
                "change_type": "skill",
            })

        with patch("app.results_ledger.get_recent_results", return_value=cosmetic_results):
            signals = detect_gaming_signals(window_days=30)
            assert any(s.signal_type == "kept_ratio_spike" for s in signals)


# ─────────────────────────────────────────────────────────────────────────────
# mutation_strategies
# ─────────────────────────────────────────────────────────────────────────────

class TestMutationStrategies:
    def test_select_strategy_returns_valid_spec(self):
        from app.mutation_strategies import select_strategy, MutationStrategy
        spec = select_strategy(seed=42)
        assert spec.name in MutationStrategy
        assert spec.weight >= 0.0
        assert spec.description

    def test_load_strategies_all_six(self):
        from app.mutation_strategies import load_strategies, MutationStrategy
        strategies = load_strategies()
        for strategy in MutationStrategy:
            assert strategy in strategies

    def test_build_strategy_prompt_section(self):
        from app.mutation_strategies import select_strategy, build_strategy_prompt_section
        spec = select_strategy(seed=0)
        section = build_strategy_prompt_section(spec)
        assert spec.name.value.upper() in section
        assert spec.guidance in section

    def test_update_strategy_success(self, tmp_path, monkeypatch):
        import app.mutation_strategies as ms
        monkeypatch.setattr(ms, "STRATEGY_STATS_PATH", tmp_path / "stats.json")
        ms.update_strategy_success("defensive", succeeded=True, delta=0.03)
        ms.update_strategy_success("defensive", succeeded=False)
        rates = ms.get_strategy_success_rates()
        assert rates["defensive"]["total"] == 2
        assert rates["defensive"]["succeeded"] == 1
        assert rates["defensive"]["success_rate"] == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# differential_test
# ─────────────────────────────────────────────────────────────────────────────

class TestDifferentialTest:
    def test_exact_match_strategy(self):
        from app.differential_test import (
            run_differential_test, CompareStrategy,
        )

        def add_one(x):
            return x + 1

        def add_one_v2(x):
            return x + 1  # same behavior

        result = run_differential_test(add_one, add_one_v2, [1, 2, 3, 4, 5], CompareStrategy.EXACT)
        assert result.matches == 5
        assert result.divergences == 0
        assert result.is_safe_change

    def test_detects_divergence(self):
        from app.differential_test import run_differential_test, CompareStrategy

        def old_fn(x):
            return x * 2

        def new_fn(x):
            return x * 3  # behavior change

        result = run_differential_test(old_fn, new_fn, [1, 2, 3], CompareStrategy.EXACT)
        assert result.divergences == 3
        assert not result.is_safe_change

    def test_strategy_for_change_type(self):
        from app.differential_test import select_strategy_for_change_type, CompareStrategy
        assert select_strategy_for_change_type("refactoring") == CompareStrategy.EXACT
        assert select_strategy_for_change_type("optimization") == CompareStrategy.EXACT
        assert select_strategy_for_change_type("capability") == CompareStrategy.SEMANTIC


# ─────────────────────────────────────────────────────────────────────────────
# tier_graduation
# ─────────────────────────────────────────────────────────────────────────────

class TestTierGraduation:
    def test_record_successful_mutation(self, tmp_path, monkeypatch):
        import app.tier_graduation as tg
        monkeypatch.setattr(tg, "TIER_HISTORY_PATH", tmp_path / "history.json")
        tg.record_successful_mutation("app/agents/researcher.py")
        history = tg._load_history()
        assert "app/agents/researcher.py" in history
        assert history["app/agents/researcher.py"].successful_mutations == 1

    def test_rollback_threshold_demotes(self, tmp_path, monkeypatch):
        import app.tier_graduation as tg
        monkeypatch.setattr(tg, "TIER_HISTORY_PATH", tmp_path / "history.json")
        monkeypatch.setattr(tg, "GRADUATION_LOG_PATH", tmp_path / "grad.json")

        # Mock a file that's currently OPEN
        history = tg._load_history()
        history["app/test/foo.py"] = tg.TierHistory(
            filepath="app/test/foo.py",
            static_tier="open",
            dynamic_tier="open",
            dynamic_tier_since=time.time() - 86400,
        )
        tg._save_history(history)

        # Three rollbacks in 7 days should demote
        for _ in range(3):
            tg.record_rollback("app/test/foo.py")

        history = tg._load_history()
        assert history["app/test/foo.py"].dynamic_tier == "gated"

    def test_dynamic_tier_returns_more_restrictive(self, tmp_path, monkeypatch):
        """If static says OPEN but dynamic says GATED, GATED wins."""
        import app.tier_graduation as tg
        monkeypatch.setattr(tg, "TIER_HISTORY_PATH", tmp_path / "history.json")

        with patch("app.tier_graduation._static_tier", return_value="open"):
            history = tg._load_history()
            history["app/test/foo.py"] = tg.TierHistory(
                filepath="app/test/foo.py",
                static_tier="open",
                dynamic_tier="gated",
                dynamic_tier_since=time.time(),
            )
            tg._save_history(history)
            assert tg.get_dynamic_tier("app/test/foo.py") == "gated"


# ─────────────────────────────────────────────────────────────────────────────
# alignment_audit
# ─────────────────────────────────────────────────────────────────────────────

class TestAlignmentAudit:
    def test_load_thresholds_returns_tuple(self):
        from app.alignment_audit import _load_thresholds
        alert, critical = _load_thresholds()
        assert 0.0 < alert <= critical <= 1.0

    def test_audit_returns_safe_report_when_constitution_missing(self, tmp_path, monkeypatch):
        import app.alignment_audit as aa
        monkeypatch.setattr(aa, "CONSTITUTION_PATH", tmp_path / "missing.md")
        monkeypatch.setattr(aa, "ALIGNMENT_REPORTS_PATH", tmp_path / "reports.json")
        report = aa.run_alignment_audit()
        assert report.severity == "ok"
        assert report.drift_score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# knowledge_compactor
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeCompactor:
    def test_suggest_merged_name(self, tmp_path):
        from pathlib import Path
        from app.knowledge_compactor import _suggest_merged_name
        paths = [
            Path("api_credit_management__abc123.md"),
            Path("api_credit_management__def456.md"),
        ]
        name = _suggest_merged_name(paths)
        assert name.endswith(".md")
        assert "api_credit" in name

    def test_find_skill_clusters_handles_empty(self, tmp_path, monkeypatch):
        import app.knowledge_compactor as kc
        monkeypatch.setattr(kc, "SKILLS_DIR", tmp_path / "empty")
        clusters = kc.find_skill_clusters()
        assert clusters == []


# ─────────────────────────────────────────────────────────────────────────────
# improvement_narrative
# ─────────────────────────────────────────────────────────────────────────────

class TestImprovementNarrative:
    def test_generate_narrative_empty_data(self, tmp_path, monkeypatch):
        import app.improvement_narrative as inv
        monkeypatch.setattr(inv, "NARRATIVE_DIR", tmp_path / "narratives")
        monkeypatch.setattr(inv, "NARRATIVE_INDEX", tmp_path / "narratives" / "index.json")

        with patch("app.results_ledger.get_recent_results", return_value=[]):
            with patch("app.healing.error_diagnosis.get_recent_errors", return_value=[]):
                narrative = inv.generate_daily_narrative()
                assert "Evolution Daily" in narrative
                assert "No experiments" in narrative or "idle" in narrative

    def test_summarize_experiments_aggregates_correctly(self):
        from app.improvement_narrative import _summarize_experiments
        experiments = [
            {"status": "keep", "delta": 0.05, "hypothesis": "a"},
            {"status": "keep", "delta": 0.0, "hypothesis": "b"},
            {"status": "discard", "delta": -0.01, "hypothesis": "c"},
            {"status": "stored", "delta": 0.0, "hypothesis": "d"},
        ]
        stats = _summarize_experiments(experiments)
        assert stats["total"] == 4
        assert stats["kept"] == 2
        assert stats["kept_meaningful"] == 1
        assert stats["discarded"] == 1
        assert stats["stored"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# human_gate
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanGate:
    def test_classify_high_confidence(self):
        from app.human_gate import classify_confidence, ConfidenceTier
        tier, reason = classify_confidence(
            delta=0.10, eval_measured=True,
            has_high_centrality_files=False, is_hot_path=False,
        )
        assert tier == ConfidenceTier.HIGH

    def test_classify_borderline_small_delta(self):
        from app.human_gate import classify_confidence, ConfidenceTier
        tier, reason = classify_confidence(delta=0.01, eval_measured=True)
        assert tier == ConfidenceTier.BORDERLINE

    def test_classify_borderline_high_centrality(self):
        from app.human_gate import classify_confidence, ConfidenceTier
        tier, reason = classify_confidence(
            delta=0.10, eval_measured=True, has_high_centrality_files=True,
        )
        assert tier == ConfidenceTier.BORDERLINE

    def test_classify_low_for_zero_delta(self):
        from app.human_gate import classify_confidence, ConfidenceTier
        tier, _ = classify_confidence(delta=0.0)
        assert tier == ConfidenceTier.LOW

    def test_request_and_approve(self, tmp_path, monkeypatch):
        import app.human_gate as hg
        monkeypatch.setattr(hg, "APPROVAL_QUEUE_PATH", tmp_path / "queue.json")
        monkeypatch.setattr(hg, "APPROVAL_HISTORY_PATH", tmp_path / "history.json")

        # Suppress Signal notification
        with patch("app.human_gate._send_approval_notification"):
            req_id = hg.request_approval(
                experiment_id="exp_1",
                hypothesis="test",
                change_type="code",
                files={"a.py": "code"},
                delta=0.02,
            )

        pending = hg.get_pending_requests()
        assert len(pending) == 1

        with patch("app.auto_deployer.schedule_deploy"):
            assert hg.approve_request(req_id) is True

        # Should be moved to history, queue empty
        assert hg.get_pending_requests() == []

    def test_expire_stale_requests(self, tmp_path, monkeypatch):
        import app.human_gate as hg
        monkeypatch.setattr(hg, "APPROVAL_QUEUE_PATH", tmp_path / "queue.json")
        monkeypatch.setattr(hg, "APPROVAL_HISTORY_PATH", tmp_path / "history.json")

        # Manually inject an old pending request
        old_request = {
            "request_id": "expired_req",
            "experiment_id": "exp_old",
            "hypothesis": "old",
            "change_type": "skill",
            "files": {},
            "delta": 0.01,
            "confidence_tier": "borderline",
            "confidence_reason": "old",
            "created_at": time.time() - (48 * 3600),  # 2 days ago
            "decision": "pending",
            "decided_at": 0.0,
            "decided_by": "",
        }
        hg._save_queue([old_request])

        expired = hg.expire_stale_requests()
        assert expired == 1
        assert hg.get_pending_requests() == []
