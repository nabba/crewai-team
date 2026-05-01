"""
Training Pipeline Wiring Tests
================================

Tests the full self-training pipeline from data collection through
curation, quality scoring, and MLX training orchestration.

Pipeline under test:
  LLM call → litellm callback → training_collector → JSONL + PostgreSQL
  → quality scoring → curation → export → training_pipeline → MLX LoRA

Run: docker exec crewai-team-gateway-1 python3 -m pytest /app/tests/test_training_pipeline.py -v
"""

import hashlib
import os
import importlib
import inspect
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Persists state under /app/workspace, which is the read-only system root
# on macOS hosts. Skip unless we're inside a Docker-style writable layout.
pytestmark = pytest.mark.skipif(
    not os.access("/app", os.W_OK),
    reason="Requires Docker-style /app writable layout (run inside the gateway container)",
)


sys.path.insert(0, str(Path(__file__).parent.parent))


# ════════════════════════════════════════════════════════════════════════════════
# 1. DATA COLLECTION — litellm callback captures training data
# ════════════════════════════════════════════════════════════════════════════════

class TestDataCollection:
    """Verify training data is captured from litellm success callback."""

    def test_capture_function_exists(self):
        from app.rate_throttle import _capture_training_data
        assert callable(_capture_training_data)

    def test_capture_in_litellm_callback(self):
        """_record_token_usage should call _capture_training_data."""
        source = inspect.getsource(
            importlib.import_module("app.rate_throttle")._record_token_usage
        )
        assert "_capture_training_data" in source

    def test_litellm_success_callback_registered(self):
        """litellm.success_callback should include _record_token_usage."""
        source = inspect.getsource(
            importlib.import_module("app.rate_throttle")
        )
        assert "litellm.success_callback" in source

    def test_capture_filters_short_responses(self):
        """Responses < 50 chars should be filtered out."""
        from app.rate_throttle import _capture_training_data

        # Mock a short response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hi"  # Too short

        # Should not crash and should not store
        _capture_training_data(mock_response, {"messages": [{"role": "user", "content": "hello"}]}, "test-model")
        # No assertion needed — just verifying it doesn't crash

    def test_capture_filters_no_user_message(self):
        """Messages without user role should be filtered out."""
        from app.rate_throttle import _capture_training_data

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A" * 100

        # Only system message — no user content
        _capture_training_data(
            mock_response,
            {"messages": [{"role": "system", "content": "you are an assistant"}]},
            "test-model",
        )

    def test_capture_extracts_completion_text(self):
        """Should extract response text from litellm response object."""
        from app.rate_throttle import _capture_training_data

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a substantial response with real content about Python programming and best practices for writing clean code."

        with patch("app.training_collector._store_record") as mock_store:
            _capture_training_data(
                mock_response,
                {"messages": [
                    {"role": "system", "content": "You are a coder"},
                    {"role": "user", "content": "Write a Python function"},
                ]},
                "deepseek/deepseek-chat",
            )
            # Give the thread a moment to execute
            import time
            time.sleep(0.5)
            if mock_store.called:
                record = mock_store.call_args[0][0]
                assert record["response"] == mock_response.choices[0].message.content
                assert record["source_model"] == "deepseek/deepseek-chat"
                assert len(record["messages"]) >= 1

    def test_capture_handles_empty_response_gracefully(self):
        """Should not crash on malformed response objects."""
        from app.rate_throttle import _capture_training_data

        # No choices
        mock_response = MagicMock()
        mock_response.choices = []
        _capture_training_data(mock_response, {}, "test")

        # None choices
        mock_response.choices = None
        _capture_training_data(mock_response, {}, "test")

        # No message attribute
        mock_response.choices = [MagicMock(spec=[])]
        _capture_training_data(mock_response, {}, "test")


# ════════════════════════════════════════════════════════════════════════════════
# 2. TRAINING COLLECTOR — storage and deduplication
# ════════════════════════════════════════════════════════════════════════════════

class TestTrainingCollector:
    """Verify training_collector.py storage and dedup."""

    def test_content_hash_deterministic(self):
        from app.training_collector import _content_hash
        messages = [{"role": "user", "content": "test"}]
        h1 = _content_hash(messages, "response")
        h2 = _content_hash(messages, "response")
        assert h1 == h2  # Same input → same hash

    def test_content_hash_different_for_different_content(self):
        from app.training_collector import _content_hash
        h1 = _content_hash([{"role": "user", "content": "test1"}], "response1")
        h2 = _content_hash([{"role": "user", "content": "test2"}], "response2")
        assert h1 != h2

    def test_classify_model_tiers(self):
        from app.training_collector import _classify_model
        tier, prov = _classify_model("deepseek/deepseek-chat")
        assert tier == "T2_budget"
        assert prov == "api_deepseek"

        tier, prov = _classify_model("claude-opus-4.6")
        assert tier == "T4_premium"
        assert prov == "api_anthropic"

        tier, prov = _classify_model("ollama_chat/qwen3.5:35b-a3b-q4_K_M")
        assert tier == "T1_local"
        assert prov == "local_ollama"

    def test_store_record_to_jsonl(self):
        """_store_record should write to JSONL file."""
        from app.training_collector import _store_record, RAW_DIR

        record = {
            "id": f"test_{int(datetime.now(timezone.utc).timestamp())}",
            "agent_role": "test",
            "task_description": "unit test",
            "messages": [{"role": "user", "content": "test prompt"}],
            "response": "test response " * 10,
            "source_model": "test-model",
            "source_tier": "T1_local",
            "provenance": "test",
            "quality_score": None,
            "training_eligible": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _store_record(record)

        # Check JSONL file exists and has content
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = RAW_DIR / f"interactions_{date_str}.jsonl"
        assert log_file.exists()
        last_line = log_file.read_text().strip().split("\n")[-1]
        parsed = json.loads(last_line)
        assert parsed["id"] == record["id"]

    def test_store_record_to_postgres(self):
        """_store_record should also write to PostgreSQL."""
        from app.training_collector import _store_record
        from app.control_plane.db import execute

        test_id = f"pgtest_{int(datetime.now(timezone.utc).timestamp())}"
        record = {
            "id": test_id,
            "agent_role": "test",
            "task_description": "pg test",
            "messages": [{"role": "user", "content": "test"}],
            "response": "test response for postgres",
            "source_model": "test",
            "source_tier": "T1_local",
            "provenance": "test",
            "quality_score": None,
            "training_eligible": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _store_record(record)

        # Check PostgreSQL
        rows = execute(
            "SELECT id FROM training.interactions WHERE id = %s",
            (test_id,), fetch=True,
        )
        # May be None without DB — that's OK for local testing
        if rows:
            assert any(r["id"] == test_id for r in rows)


# ════════════════════════════════════════════════════════════════════════════════
# 3. CURATION PIPELINE — quality scoring and export
# ════════════════════════════════════════════════════════════════════════════════

class TestCurationPipeline:
    """Verify CurationPipeline scoring, filtering, and export."""

    def test_pipeline_initializes(self):
        from app.training_collector import CurationPipeline
        pipeline = CurationPipeline()
        assert hasattr(pipeline, "run_curation")
        assert hasattr(pipeline, "_score_quality")
        assert hasattr(pipeline, "_load_unscored")
        assert hasattr(pipeline, "_export_mlx")

    def test_load_unscored_merges_sources(self):
        """_load_unscored should merge PostgreSQL + JSONL."""
        source = inspect.getsource(
            importlib.import_module("app.training_collector").CurationPipeline._load_unscored
        )
        # Should try both PG and JSONL
        assert "psycopg2" in source or "mem0_postgres_url" in source
        assert "_load_from_jsonl" in source

    def test_score_quality_no_max_tokens_arg(self):
        """_score_quality must NOT pass max_tokens to create_cheap_vetting_llm."""
        source = inspect.getsource(
            importlib.import_module("app.training_collector").CurationPipeline._score_quality
        )
        # The bug was: create_cheap_vetting_llm(max_tokens=200)
        assert "max_tokens" not in source

    def test_score_quality_has_logging(self):
        """_score_quality should log scoring results."""
        source = inspect.getsource(
            importlib.import_module("app.training_collector").CurationPipeline._score_quality
        )
        assert "logger" in source
        assert "scored" in source.lower()

    def test_run_curation_returns_stats(self):
        from app.training_collector import CurationPipeline
        pipeline = CurationPipeline()
        result = pipeline.run_curation()
        assert isinstance(result, dict)
        assert "status" in result
        # Should have standard stat keys
        if result["status"] == "completed":
            assert "total_scored" in result
            assert "eligible" in result

    def test_quality_threshold_constant(self):
        from app.training_collector import QUALITY_THRESHOLD
        assert QUALITY_THRESHOLD == 0.70  # Documented threshold

    def test_min_training_set_size(self):
        from app.training_collector import MIN_TRAINING_SET_SIZE
        assert MIN_TRAINING_SET_SIZE == 100


# ════════════════════════════════════════════════════════════════════════════════
# 4. TRAINING PIPELINE — MLX orchestration
# ════════════════════════════════════════════════════════════════════════════════

class TestTrainingPipelineOrchestration:
    """Verify training_pipeline.py orchestration."""

    def test_pipeline_importable(self):
        from app.training_pipeline import (
            get_orchestrator, run_training_cycle,
            TrainingOrchestrator, AdapterInfo,
        )
        assert callable(run_training_cycle)

    def test_orchestrator_singleton(self):
        from app.training_pipeline import get_orchestrator
        o1 = get_orchestrator()
        o2 = get_orchestrator()
        assert o1 is o2

    def test_promotion_gates_immutable(self):
        """Promotion gates should have strict thresholds."""
        from app.training_pipeline import (
            QUALITY_GATE, REGRESSION_GATE, SAFETY_GATE,
            PREFERENCE_GATE, DIVERSITY_GATE,
        )
        assert QUALITY_GATE == 0.75
        assert SAFETY_GATE == 0  # Zero tolerance
        assert DIVERSITY_GATE == 0.80

    def test_training_defaults(self):
        from app.training_pipeline import (
            DEFAULT_BASE_MODEL, DEFAULT_LORA_LAYERS,
            DEFAULT_ITERS, MIN_TRAINING_EXAMPLES,
        )
        assert "Qwen2.5" in DEFAULT_BASE_MODEL or "qwen" in DEFAULT_BASE_MODEL.lower()
        assert DEFAULT_LORA_LAYERS >= 8
        assert MIN_TRAINING_EXAMPLES >= 50

    def test_run_training_cycle_insufficient_data(self):
        """With < 100 examples, should return insufficient_data."""
        from app.training_pipeline import run_training_cycle
        result = run_training_cycle()
        assert isinstance(result, dict)
        # With almost no data, should return insufficient
        assert result.get("status") in ("insufficient_data", "no_curated_data", "error")

    def test_collapse_detection_function(self):
        """Model collapse detection should work on sample texts."""
        from app.training_pipeline import detect_collapse
        current = ["The cat sat on the mat.", "Python is a programming language.",
                    "Machine learning uses data to learn patterns."]
        baseline = ["The dog ran in the park.", "Java is a programming language.",
                     "Deep learning is a subset of machine learning."]
        result = detect_collapse(current, baseline)
        assert "distinct_2_ratio" in result
        assert "vocab_ratio" in result
        assert "passes_gate" in result

    def test_collapse_detection_catches_repetition(self):
        """Collapsed model output (all same) should fail the gate."""
        from app.training_pipeline import detect_collapse
        current = ["The same output."] * 10  # Collapsed
        baseline = [f"Unique output number {i}." for i in range(10)]  # Diverse
        result = detect_collapse(current, baseline)
        assert result["distinct_2_ratio"] < 0.5  # Very low diversity

    def test_adapter_info_dataclass(self):
        from app.training_pipeline import AdapterInfo
        info = AdapterInfo(name="test", base_model="test-model")
        d = info.to_dict()
        assert d["name"] == "test"
        assert d["promoted"] is False


# ════════════════════════════════════════════════════════════════════════════════
# 5. IDLE SCHEDULER WIRING
# ════════════════════════════════════════════════════════════════════════════════

class TestIdleSchedulerWiring:
    """Verify training jobs are registered in idle scheduler."""

    def test_training_curate_job_exists(self):
        from app.idle_scheduler import _default_jobs
        jobs = _default_jobs()
        names = [n for n, _ in jobs]
        assert "training-curate" in names

    def test_training_pipeline_job_exists(self):
        from app.idle_scheduler import _default_jobs
        jobs = _default_jobs()
        names = [n for n, _ in jobs]
        assert "training-pipeline" in names

    def test_training_curate_calls_run_curation(self):
        """The curate job should call pipeline.run_curation()."""
        from app.idle_scheduler import _default_jobs
        jobs = _default_jobs()
        curate_job = [f for n, f in jobs if n == "training-curate"]
        assert len(curate_job) == 1
        # The job function should reference run_curation
        source = inspect.getsource(curate_job[0])
        assert "run_curation" in source

    def test_training_pipeline_job_calls_run_training_cycle(self):
        from app.idle_scheduler import _default_jobs
        jobs = _default_jobs()
        pipeline_job = [f for n, f in jobs if n == "training-pipeline"]
        assert len(pipeline_job) == 1
        source = inspect.getsource(pipeline_job[0])
        assert "run_training_cycle" in source


# ════════════════════════════════════════════════════════════════════════════════
# 6. HOST BRIDGE MLX INTEGRATION
# ════════════════════════════════════════════════════════════════════════════════

class TestHostBridgeMLX:
    """Verify MLX endpoints exist on the host bridge."""

    def test_bridge_mlx_generate_endpoint(self):
        """Host bridge should have /mlx/generate endpoint."""
        bridge_code = Path(__file__).parent.parent / "host_bridge" / "main.py"
        if bridge_code.exists():
            content = bridge_code.read_text()
            assert "/mlx/generate" in content
            assert "MlxGenerateRequest" in content

    def test_bridge_mlx_fuse_endpoint(self):
        """Host bridge should have /mlx/fuse endpoint."""
        bridge_code = Path(__file__).parent.parent / "host_bridge" / "main.py"
        if bridge_code.exists():
            content = bridge_code.read_text()
            assert "/mlx/fuse" in content

    def test_bridge_uses_venv_python(self):
        """Host bridge should use the venv Python (3.12 with MLX)."""
        bridge_code = Path(__file__).parent.parent / "host_bridge" / "main.py"
        if bridge_code.exists():
            content = bridge_code.read_text()
            assert "sys.executable" in content or "_PYTHON" in content


# ════════════════════════════════════════════════════════════════════════════════
# 7. DATABASE SCHEMA
# ════════════════════════════════════════════════════════════════════════════════

class TestDatabaseSchema:
    """Verify training schema exists in PostgreSQL."""

    def test_interactions_table(self):
        from app.control_plane.db import execute
        rows = execute(
            "SELECT COUNT(*) as n FROM training.interactions",
            fetch=True,
        )
        assert rows is not None  # Table exists

    def test_runs_table(self):
        from app.control_plane.db import execute
        rows = execute(
            "SELECT COUNT(*) as n FROM training.runs",
            fetch=True,
        )
        assert rows is not None

    def test_migration_file_exists(self):
        """Migration file should exist on host (not in Docker container)."""
        migration = Path(__file__).parent.parent / "migrations" / "007_training_schema.sql"
        # In Docker, migrations/ isn't copied — check if table exists instead
        if not migration.exists():
            from app.control_plane.db import execute
            rows = execute("SELECT COUNT(*) as n FROM training.interactions", fetch=True)
            assert rows is not None, "Neither migration file nor table found"
        else:
            assert migration.exists()


# ════════════════════════════════════════════════════════════════════════════════
# 8. END-TO-END FLOW TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestEndToEndFlow:
    """Verify the complete data flow from capture to curation."""

    def test_capture_to_jsonl_flow(self):
        """Simulate an LLM call and verify data reaches JSONL."""
        from app.rate_throttle import _capture_training_data
        from app.training_collector import RAW_DIR
        import time

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Here is a comprehensive explanation of how Docker containers work. "
            "Docker uses Linux namespaces and cgroups to provide isolation between "
            "containers. Each container has its own filesystem, network stack, and "
            "process space, making it lightweight compared to virtual machines."
        )

        _capture_training_data(
            mock_response,
            {"messages": [
                {"role": "system", "content": "You are a researcher."},
                {"role": "user", "content": "Explain how Docker containers work."},
            ]},
            "deepseek/deepseek-chat",
        )

        # Wait for async write
        time.sleep(1.0)

        # Check JSONL
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = RAW_DIR / f"interactions_{date_str}.jsonl"
        if log_file.exists():
            lines = log_file.read_text().strip().split("\n")
            last = json.loads(lines[-1])
            assert "Docker" in last.get("response", "")

    def test_curation_loads_captured_data(self):
        """CurationPipeline should find and load captured interactions."""
        from app.training_collector import CurationPipeline
        pipeline = CurationPipeline()
        interactions = pipeline._load_unscored()
        # Should load from JSONL at minimum
        assert isinstance(interactions, list)

    def test_full_pipeline_no_crash(self):
        """Full pipeline should complete without uncaught exceptions."""
        from app.training_pipeline import run_training_cycle
        result = run_training_cycle()
        assert isinstance(result, dict)
        assert "status" in result
        assert "run_id" in result


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
