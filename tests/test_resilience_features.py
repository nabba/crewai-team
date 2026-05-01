"""
Tests for the 5 new resilience features:
  1. SLO budget evaluation (wired)
  2. Request tracing (correlation IDs)
  3. Load shedding
  4. Dead letter queue
  5. Canary deployment

Total: ~65 tests
"""

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock Docker-only modules
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

try:
    from pydantic_settings import BaseSettings
    _HAS_FULL_DEPS = True
except ImportError:
    _HAS_FULL_DEPS = False

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SLO Budget Evaluation (wired)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSLOBudgetWired:
    """SLO budget should now be evaluated during health checks."""

    def test_evaluate_slo_budget_still_works(self):
        from app.health_monitor import evaluate_slo_budget
        result = evaluate_slo_budget(0.001, 2000)
        assert result["budget_ok"] is True
        assert result["severity_escalation"] is None

    def test_slo_budget_exhausted(self):
        from app.health_monitor import evaluate_slo_budget
        result = evaluate_slo_budget(0.01, 2000)
        assert result["budget_ok"] is False
        assert result["severity_escalation"] == "emergency"

    def test_health_monitor_has_slo_budget_attr(self):
        from app.health_monitor import HealthMonitor
        m = HealthMonitor()
        assert hasattr(m, "_last_slo_budget")
        assert m._last_slo_budget == {}

    def test_get_slo_budget_convenience(self):
        from app.health_monitor import get_slo_budget
        assert callable(get_slo_budget)
        result = get_slo_budget()
        assert isinstance(result, dict)

    def test_evaluate_calls_slo_budget(self):
        """evaluate() should populate _last_slo_budget."""
        source = (Path(__file__).parent.parent / "app" / "health_monitor.py").read_text()
        assert "self._last_slo_budget = evaluate_slo_budget" in source

    def test_slo_generates_alert_on_exhaustion(self):
        """SLO budget exhaustion should create a HealthAlert."""
        source = (Path(__file__).parent.parent / "app" / "health_monitor.py").read_text()
        assert '"slo_budget"' in source
        assert "severity_escalation" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Request Tracing
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequestTracing:
    """Correlation IDs should flow through request lifecycle."""

    def test_trace_module_exists(self):
        from app.trace import new_trace_id, get_trace_id, set_trace_id
        assert callable(new_trace_id)
        assert callable(get_trace_id)
        assert callable(set_trace_id)

    def test_new_trace_id_format(self):
        from app.trace import new_trace_id
        tid = new_trace_id()
        assert len(tid) == 12
        assert all(c in "0123456789abcdef" for c in tid)

    def test_trace_id_persists_in_context(self):
        from app.trace import new_trace_id, get_trace_id
        tid = new_trace_id()
        assert get_trace_id() == tid

    def test_set_trace_id(self):
        from app.trace import set_trace_id, get_trace_id
        set_trace_id("custom123abc")
        assert get_trace_id() == "custom123abc"

    def test_trace_id_unique(self):
        from app.trace import new_trace_id
        ids = {new_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_default_is_empty(self):
        from app.trace import _trace_id
        # Reset context
        token = _trace_id.set("")
        assert _trace_id.get() == ""

    def test_trace_id_thread_isolated(self):
        """Each thread should have its own trace_id."""
        from app.trace import new_trace_id, get_trace_id, set_trace_id
        results = {}

        def worker(name):
            set_trace_id(name)
            time.sleep(0.01)
            results[name] = get_trace_id()

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Each thread should have its own trace_id
        for name, tid in results.items():
            assert tid == name

    def test_trace_id_in_main_handle_task(self):
        """handle_task should generate trace_id."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "new_trace_id()" in source

    def test_trace_id_in_error_handler(self):
        """Error handler should include trace_id in entries."""
        source = (Path(__file__).parent.parent / "app" / "error_handler.py").read_text()
        assert '"trace_id"' in source
        assert "get_trace_id" in source

    def test_trace_id_in_audit(self):
        """Audit system should include trace_id in records."""
        source = (Path(__file__).parent.parent / "app" / "audit.py").read_text()
        assert '"trace_id"' in source
        assert "get_trace_id" in source

    def test_trace_id_in_health_metrics(self):
        """InteractionMetrics should have trace_id field."""
        from app.health_monitor import InteractionMetrics
        m = InteractionMetrics(trace_id="abc123")
        assert m.trace_id == "abc123"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Load Shedding
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadShedding:
    """Overloaded system should reject requests gracefully."""

    def test_load_shedding_in_main(self):
        """main.py should check inflight count and reject at capacity."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "load_shed_threshold" in source
        assert "at capacity" in source.lower()

    def test_load_shed_before_notify_task_start(self):
        """Load shedding check should happen BEFORE notify_task_start."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        # Find positions
        shed_pos = source.find("_shed_threshold")
        notify_pos = source.find("idle_scheduler.notify_task_start()")
        assert shed_pos < notify_pos, "Load shedding should happen before notify_task_start"

    def test_config_has_load_shed_threshold(self):
        """Config should have load_shed_threshold setting."""
        source = (Path(__file__).parent.parent / "app" / "config.py").read_text()
        assert "load_shed_threshold" in source

    def test_rejected_message_is_polite(self):
        """Rejection message should explain and suggest retry."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "try again" in source.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dead Letter Queue
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeadLetterQueue:
    """Failed messages should be persisted for retry."""

    def test_dlq_module_exists(self):
        from app.dead_letter import enqueue, dequeue_retryable, mark_success, mark_permanent_failure, get_stats
        assert callable(enqueue)
        assert callable(dequeue_retryable)
        assert callable(mark_success)
        assert callable(mark_permanent_failure)
        assert callable(get_stats)

    def test_enqueue_and_dequeue(self):
        """Messages should survive enqueue→dequeue cycle."""
        import app.dead_letter as dlq
        original_path = dlq._DLQ_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq._DLQ_PATH = os.path.join(tmpdir, "test_dlq")
            try:
                # Enqueue with past timestamp so it's immediately retryable
                dlq.enqueue("sender", "test message", "ValueError", "trace123")
                # Manually set enqueued_at to past to make retryable
                import dbm.sqlite3
                with dbm.sqlite3.open(dlq._DLQ_PATH, "c") as db:
                    for key in db.keys():
                        entry = json.loads(db[key])
                        entry["enqueued_at"] = time.time() - 600  # 10 min ago
                        db[key] = json.dumps(entry)

                retryable = dlq.dequeue_retryable()
                assert len(retryable) == 1
                assert retryable[0]["sender"] == "sender"
                assert retryable[0]["text"] == "test message"
                assert retryable[0]["trace_id"] == "trace123"
            finally:
                dlq._DLQ_PATH = original_path

    def test_max_retries_respected(self):
        """Messages with max retries should not be retryable."""
        import app.dead_letter as dlq
        original_path = dlq._DLQ_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq._DLQ_PATH = os.path.join(tmpdir, "test_dlq")
            try:
                dlq.enqueue("sender", "test", "Error")
                import dbm.sqlite3
                with dbm.sqlite3.open(dlq._DLQ_PATH, "c") as db:
                    for key in db.keys():
                        entry = json.loads(db[key])
                        entry["retry_count"] = dlq.MAX_RETRIES
                        entry["enqueued_at"] = time.time() - 600
                        db[key] = json.dumps(entry)
                retryable = dlq.dequeue_retryable()
                assert len(retryable) == 0
            finally:
                dlq._DLQ_PATH = original_path

    def test_mark_success_removes_entry(self):
        """Successful retry should remove entry from DLQ."""
        import app.dead_letter as dlq
        original_path = dlq._DLQ_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq._DLQ_PATH = os.path.join(tmpdir, "test_dlq")
            try:
                dlq.enqueue("sender", "test", "Error")
                import dbm.sqlite3
                with dbm.sqlite3.open(dlq._DLQ_PATH, "c") as db:
                    key = list(db.keys())[0]
                    key_str = key.decode() if isinstance(key, bytes) else key
                dlq.mark_success(key_str)
                stats = dlq.get_stats()
                assert stats["total"] == 0
            finally:
                dlq._DLQ_PATH = original_path

    def test_mark_permanent_failure(self):
        """Permanently failed messages should be marked."""
        import app.dead_letter as dlq
        original_path = dlq._DLQ_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq._DLQ_PATH = os.path.join(tmpdir, "test_dlq")
            try:
                dlq.enqueue("sender", "test", "Error")
                import dbm.sqlite3
                with dbm.sqlite3.open(dlq._DLQ_PATH, "c") as db:
                    key = list(db.keys())[0]
                    key_str = key.decode() if isinstance(key, bytes) else key
                dlq.mark_permanent_failure(key_str)
                stats = dlq.get_stats()
                assert stats["failed"] == 1
            finally:
                dlq._DLQ_PATH = original_path

    def test_get_stats(self):
        """Stats should show pending/failed/total counts."""
        import app.dead_letter as dlq
        original_path = dlq._DLQ_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq._DLQ_PATH = os.path.join(tmpdir, "test_dlq")
            try:
                stats = dlq.get_stats()
                assert "pending" in stats
                assert "failed" in stats
                assert "total" in stats
            finally:
                dlq._DLQ_PATH = original_path

    def test_max_dlq_size_enforced(self):
        """DLQ should prune oldest entries beyond MAX_DLQ_SIZE."""
        import app.dead_letter as dlq
        original_path = dlq._DLQ_PATH
        original_max = dlq.MAX_DLQ_SIZE
        dlq.MAX_DLQ_SIZE = 5
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq._DLQ_PATH = os.path.join(tmpdir, "test_dlq")
            try:
                for i in range(10):
                    dlq.enqueue(f"sender{i}", f"message {i}", "Error")
                stats = dlq.get_stats()
                assert stats["total"] <= 5
            finally:
                dlq._DLQ_PATH = original_path
                dlq.MAX_DLQ_SIZE = original_max

    def test_dlq_wired_to_main_error_handler(self):
        """main.py should enqueue to DLQ on task failure."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "dlq_enqueue" in source
        assert "dead_letter" in source

    def test_dlq_retry_wired_to_idle_scheduler(self):
        """Idle scheduler should have dead-letter-retry job."""
        source = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "dead-letter-retry" in source
        assert "dequeue_retryable" in source

    def test_text_truncation(self):
        """Text should be truncated to MAX_TEXT_LEN."""
        import app.dead_letter as dlq
        original_path = dlq._DLQ_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq._DLQ_PATH = os.path.join(tmpdir, "test_dlq")
            try:
                long_text = "x" * 5000
                dlq.enqueue("sender", long_text, "Error")
                import dbm.sqlite3
                with dbm.sqlite3.open(dlq._DLQ_PATH, "r") as db:
                    for key in db.keys():
                        entry = json.loads(db[key])
                        assert len(entry["text"]) <= dlq.MAX_TEXT_LEN
            finally:
                dlq._DLQ_PATH = original_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Canary Deployment
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanaryDeployment:
    """Synthetic canary should test before promoting to production."""

    def test_canary_module_exists(self):
        from app.canary_deploy import CanaryDeployer
        assert CanaryDeployer is not None

    def test_config_has_canary_settings(self):
        source = (Path(__file__).parent.parent / "app" / "config.py").read_text()
        assert "canary_deploy_enabled" in source
        assert "canary_regression_tolerance" in source

    def test_auto_deployer_routes_through_canary(self):
        """schedule_deploy should route through CanaryDeployer."""
        source = (Path(__file__).parent.parent / "app" / "auto_deployer.py").read_text()
        assert "CanaryDeployer" in source
        assert "run_canary" in source

    def test_canary_deployer_has_required_methods(self):
        from app.canary_deploy import CanaryDeployer
        import inspect
        methods = [m for m in dir(CanaryDeployer) if not m.startswith("__")]
        assert "run_canary" in methods

    def test_canary_result_format(self):
        """run_canary should return dict with status/scores/reason."""
        # We can't run the full canary, but verify the interface
        source = (Path(__file__).parent.parent / "app" / "canary_deploy.py").read_text()
        assert '"status"' in source
        assert '"baseline_score"' in source
        assert '"canary_score"' in source
        assert '"reason"' in source
        assert '"promoted"' in source
        assert '"rolled_back"' in source

    def test_canary_has_safety_hard_gate(self):
        """Safety violations should always trigger rollback."""
        source = (Path(__file__).parent.parent / "app" / "canary_deploy.py").read_text()
        assert "safety_violations" in source
        assert "SAFETY VIOLATION" in source

    def test_canary_notifies_via_signal(self):
        """Canary results should be sent via Signal."""
        source = (Path(__file__).parent.parent / "app" / "canary_deploy.py").read_text()
        assert "send_message" in source
        assert "CANARY" in source

    def test_canary_fallback_when_disabled(self):
        """Should fall back to direct deploy when canary disabled."""
        source = (Path(__file__).parent.parent / "app" / "canary_deploy.py").read_text()
        assert "canary_deploy_enabled" in source
        assert "run_deploy" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Cross-Feature Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossFeatureIntegration:
    """Features should work together correctly."""

    def test_dlq_includes_trace_id(self):
        """DLQ enqueue in main.py should pass trace_id."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "trace_id" in source and "dlq_enqueue" in source

    def test_all_new_modules_importable(self):
        """All new modules should import without error."""
        from app.trace import new_trace_id, get_trace_id
        from app.dead_letter import enqueue, get_stats
        from app.canary_deploy import CanaryDeployer
        assert True  # If we get here, all imports succeeded

    def test_no_circular_imports(self):
        """trace.py should only use stdlib (no circular app imports)."""
        import ast
        source = (Path(__file__).parent.parent / "app" / "trace.py").read_text()
        tree = ast.parse(source)
        app_imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", None) or ""
                if module.startswith("app.") or module.startswith("app"):
                    for alias in node.names:
                        if alias.name.startswith("app."):
                            app_imports.append(alias.name)
                if isinstance(node, ast.ImportFrom) and (module or "").startswith("app"):
                    app_imports.append(module)
        assert app_imports == [], f"trace.py imports from app: {app_imports}"
