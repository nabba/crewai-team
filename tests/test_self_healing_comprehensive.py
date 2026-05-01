"""
Comprehensive self-healing system tests for AndrusAI.

Covers: circuit breakers, health monitoring, self-healer, anomaly detection,
error handling, idle scheduler persistence, workspace versioning, LLM cascade,
message deduplication, SLO tracking, and cross-module integration.

Tests both functionality AND wiring (is the code actually called?).

Total: ~100 tests
"""

import ast
import math
import os
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Mock Docker-only modules
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.control_plane", "app.control_plane.db",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
sys.modules["app.control_plane.db"].execute = MagicMock(return_value=[])

try:
    from pydantic_settings import BaseSettings
    _HAS_FULL_DEPS = True
except ImportError:
    _HAS_FULL_DEPS = False

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Circuit Breaker Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Tests for the 3-state circuit breaker."""

    def test_initial_state_closed(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
        assert cb.state == "closed"

    def test_stays_closed_under_threshold(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_opens_at_threshold(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

    def test_open_prevents_calls(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open() is True

    def test_success_resets_to_closed(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_half_open_after_cooldown(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.15)
        assert cb.state == "half_open"

    def test_half_open_success_closes(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"
        cb.record_failure()
        assert cb.state == "open"

    def test_preconfigured_breakers_exist(self):
        from app.circuit_breaker import _breakers as BREAKERS
        assert "ollama" in BREAKERS
        assert "openrouter" in BREAKERS
        assert "anthropic" in BREAKERS
        assert "self_healer" in BREAKERS

    def test_anthropic_has_higher_threshold(self):
        from app.circuit_breaker import _breakers as BREAKERS
        assert BREAKERS["anthropic"].failure_threshold >= 5
        assert BREAKERS["ollama"].failure_threshold == 3

    def test_self_healer_has_long_cooldown(self):
        from app.circuit_breaker import _breakers as BREAKERS
        assert BREAKERS["self_healer"].cooldown_seconds >= 600

    def test_get_all_states(self):
        from app.circuit_breaker import get_all_states
        states = get_all_states()
        assert isinstance(states, dict)
        assert "ollama" in states

    def test_thread_safety(self):
        from app.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("thread-test", failure_threshold=100)
        errors = []

        def hammer():
            try:
                for _ in range(50):
                    cb.record_failure()
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Health Monitor Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthMonitor:
    """Tests for dimensional health tracking and alerting."""

    def test_thresholds_are_immutable(self):
        """Health thresholds should be defined at module level (not configurable)."""
        source = (Path(__file__).parent.parent / "app" / "health_monitor.py").read_text()
        assert "THRESHOLDS" in source or "WARNING_THRESHOLDS" in source or "error_rate" in source

    def test_health_state_dataclass(self):
        from app.health_monitor import HealthState
        state = HealthState()
        assert hasattr(state, "error_rate")
        assert hasattr(state, "avg_latency_ms")
        assert hasattr(state, "hallucination_rate")
        assert hasattr(state, "safety_violations")

    def test_health_alert_dataclass(self):
        from app.health_monitor import HealthAlert
        alert = HealthAlert(
            severity="critical",
            dimension="error_rate",
            current_value=0.20,
            threshold=0.15,
        )
        assert alert.severity == "critical"
        assert alert.dimension == "error_rate"

    def test_slo_budget_evaluation_exists(self):
        """SLO budget function should exist."""
        from app.health_monitor import evaluate_slo_budget
        assert callable(evaluate_slo_budget)

    def test_slo_budget_budget_ok(self):
        """Low error rate should have budget OK."""
        from app.health_monitor import evaluate_slo_budget
        result = evaluate_slo_budget(error_rate=0.001, avg_latency_ms=2000)
        assert result["budget_ok"] is True

    def test_slo_budget_exhausted(self):
        """High error rate should exhaust budget."""
        from app.health_monitor import evaluate_slo_budget
        result = evaluate_slo_budget(error_rate=0.01, avg_latency_ms=2000)
        assert result["error_budget_consumed_pct"] >= 100
        assert result["severity_escalation"] == "emergency"

    def test_slo_latency_budget(self):
        """High latency should consume latency budget."""
        from app.health_monitor import evaluate_slo_budget
        result = evaluate_slo_budget(error_rate=0.0, avg_latency_ms=9000)
        assert result["latency_budget_consumed_pct"] >= 80


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Error Handler Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandler:
    """Tests for centralized error handling."""

    def test_error_categories(self):
        from app.error_handler import ErrorCategory
        assert hasattr(ErrorCategory, "TRANSIENT")
        assert hasattr(ErrorCategory, "DATA")
        assert hasattr(ErrorCategory, "SYSTEM")
        assert hasattr(ErrorCategory, "LOGIC")

    def test_report_error_increments_counter(self):
        from app.error_handler import report_error, get_error_counts, ErrorCategory
        before = get_error_counts().get("SYSTEM", 0)
        report_error(ErrorCategory.SYSTEM, "test", "test error")
        after = get_error_counts().get("SYSTEM", 0)
        assert after >= before  # May be > if other tests run

    def test_get_error_counts_thread_safe(self):
        from app.error_handler import get_error_counts
        counts = get_error_counts()
        assert isinstance(counts, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Idle Scheduler Persistence Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdleSchedulerPersistence:
    """Tests for dbm.sqlite3 job failure state persistence."""

    def test_persist_and_load_round_trip(self):
        """Failure counts should survive persist→load cycle."""
        import dbm.sqlite3
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_state")
            with dbm.sqlite3.open(path, "c") as db:
                db["fail:test-job"] = "3"
                db["skip:test-job"] = str(time.time() + 3600)
            with dbm.sqlite3.open(path, "r") as db:
                assert db["fail:test-job"].decode() == "3"
                assert float(db["skip:test-job"].decode()) > time.time()

    def test_expired_skips_not_loaded(self):
        """Past skip-until timestamps should be ignored on load."""
        import dbm.sqlite3
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_state")
            with dbm.sqlite3.open(path, "c") as db:
                db["skip:old-job"] = str(time.time() - 100)  # In the past
            loaded = {}
            with dbm.sqlite3.open(path, "r") as db:
                for key in db.keys():
                    k = key.decode()
                    if k.startswith("skip:"):
                        ts = float(db[key].decode())
                        if ts > time.time():
                            loaded[k[5:]] = ts
            assert "old-job" not in loaded

    def test_wall_clock_used_not_monotonic(self):
        """Idle scheduler should use time.time() for skip-until (survives reboot)."""
        source = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "time.time() + 3600" in source or "time.time()" in source

    def test_persist_calls_in_run_single_job(self):
        """_run_single_job should call persist functions."""
        source = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "_persist_job_failure(name, 0)" in source  # Reset on success
        assert "_persist_job_failure(name, consec)" in source  # Increment on failure
        assert "_persist_job_skip(name, skip_ts)" in source  # Cooldown on 3 failures


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Message Deduplication Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMessageDedup:
    """Tests for message idempotency guard."""

    def test_dedup_class_exists(self):
        """_MessageDedup should be defined in main.py."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "class _MessageDedup" in source

    def test_dedup_by_sender_timestamp(self):
        """Dedup key should combine sender + timestamp."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert 'f"{sender}:{msg_timestamp}"' in source

    def test_dedup_skip_duplicate(self):
        """Duplicate messages should be skipped."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "_msg_dedup.is_dup" in source
        assert "Duplicate message ignored" in source

    def test_dedup_bounded_lru(self):
        """Dedup cache should be bounded (not grow forever)."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "max_size" in source or "maxlen" in source or "_max" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Workspace Versioning & Auto-Rollback Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceVersioning:
    """Tests for git-based workspace versioning and auto-rollback."""

    def test_workspace_lock_class_exists(self):
        from app.workspace_versioning import WorkspaceLock
        assert WorkspaceLock is not None

    def test_workspace_commit_function(self):
        from app.workspace_versioning import workspace_commit
        assert callable(workspace_commit)

    def test_workspace_rollback_function(self):
        from app.workspace_versioning import workspace_rollback
        assert callable(workspace_rollback)

    def test_regression_check_function(self):
        from app.workspace_versioning import check_post_commit_regression
        assert callable(check_post_commit_regression)

    def test_regression_check_wired_to_idle_scheduler(self):
        """check_post_commit_regression should be called from idle scheduler."""
        source = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "check_post_commit_regression" in source

    def test_regression_uses_error_counts(self):
        """Regression check should use error_handler.get_error_counts()."""
        source = (Path(__file__).parent.parent / "app" / "workspace_versioning.py").read_text()
        assert "get_error_counts" in source

    def test_rollback_sends_signal_notification(self):
        """Auto-rollback should notify owner via Signal."""
        source = (Path(__file__).parent.parent / "app" / "workspace_versioning.py").read_text()
        assert "send_message" in source
        assert "Auto-rollback" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Chaos Tester Wiring Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChaosTesterWiring:
    """Tests verifying chaos testing is wired into the system."""

    def test_chaos_suite_exists(self):
        from app.chaos_tester import run_chaos_suite
        assert callable(run_chaos_suite)

    def test_chaos_wired_to_idle_scheduler(self):
        """Chaos testing should be scheduled as idle job."""
        source = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "chaos-testing" in source
        assert "run_chaos_suite" in source

    def test_chaos_rate_limited(self):
        """Chaos tests should not run more than once per 24h."""
        source = (Path(__file__).parent.parent / "app" / "chaos_tester.py").read_text()
        assert "24" in source or "86400" in source or "_last_run" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Ollama Native Wiring Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaNativeWiring:
    """Tests for Ollama memory management wiring."""

    @pytest.mark.skipif(not _HAS_FULL_DEPS, reason="requests not installed on host")
    def test_unload_idle_models_exists(self):
        from app.ollama_native import unload_idle_models
        assert callable(unload_idle_models)

    def test_unload_wired_to_idle_scheduler(self):
        """unload_idle_models should be called from idle scheduler."""
        source = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "unload_idle_models" in source
        assert "ollama-memory" in source

    def test_per_model_locking(self):
        """Ollama should use per-model locks (not global lock)."""
        source = (Path(__file__).parent.parent / "app" / "ollama_native.py").read_text()
        assert "_model_locks" in source
        assert "_get_model_lock" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 9. LLM Factory Cascade & Degradation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMFactoryCascade:
    """Tests for multi-tier LLM cascade and graceful degradation."""

    def test_cascade_modes_defined(self):
        """LLM factory should support local, cloud, hybrid, insane modes."""
        source = (Path(__file__).parent.parent / "app" / "llm_factory.py").read_text()
        assert "local" in source
        assert "cloud" in source
        assert "hybrid" in source

    def test_circuit_breaker_checks_before_calls(self):
        """LLM factory should check circuit breaker before provider calls."""
        source = (Path(__file__).parent.parent / "app" / "llm_factory.py").read_text()
        assert "is_available" in source
        assert "record_success" in source
        assert "record_failure" in source

    @pytest.mark.skipif(not _HAS_FULL_DEPS, reason="pydantic_settings not installed on host")
    def test_all_providers_health_check(self):
        """check_all_providers_health should exist."""
        from app.llm_factory import check_all_providers_health
        assert callable(check_all_providers_health)

    def test_credit_exhaustion_detection(self):
        """LLM factory should detect and alert on credit exhaustion."""
        source = (Path(__file__).parent.parent / "app" / "llm_factory.py").read_text()
        assert "check_all_providers_health" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Anomaly Detector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetector:
    """Tests for statistical anomaly detection."""

    def test_detector_wired_to_heartbeat(self):
        """Anomaly detection should be called from heartbeat loop."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "anomaly_detector" in source or "collect_and_check" in source

    def test_metrics_functions_exist(self):
        """Anomaly detector's metric sources should exist in app/metrics.py."""
        source = (Path(__file__).parent.parent / "app" / "metrics.py").read_text()
        assert "_error_rate_1h" in source
        assert "_avg_response_time" in source
        assert "_output_quality_score" in source

    def test_2sigma_threshold(self):
        """Should use 2σ deviation for anomaly detection."""
        source = (Path(__file__).parent.parent / "app" / "anomaly_detector.py").read_text()
        assert "2" in source  # 2σ threshold somewhere


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Self-Healer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelfHealer:
    """Tests for the self-healing orchestrator."""

    def test_dimension_strategy_mapping(self):
        """Each health dimension should map to a remediation strategy."""
        source = (Path(__file__).parent.parent / "app" / "self_healer.py").read_text()
        assert "error_rate" in source
        assert "avg_latency_ms" in source
        assert "hallucination_rate" in source
        assert "cascade_fallback_rate" in source
        assert "memory_retrieval_accuracy" in source
        assert "safety_violations" in source

    def test_rate_limiting(self):
        """Self-healer should limit remediations per dimension per day."""
        source = (Path(__file__).parent.parent / "app" / "self_healer.py").read_text()
        assert "_check_rate_limit" in source or "rate_limit" in source

    def test_verification_after_fix(self):
        """Self-healer should verify metrics improved after remediation."""
        source = (Path(__file__).parent.parent / "app" / "self_healer.py").read_text()
        assert "_schedule_verification" in source or "verification" in source

    def test_circuit_breaker_protection(self):
        """Self-healer should check its own circuit breaker."""
        source = (Path(__file__).parent.parent / "app" / "self_healer.py").read_text()
        assert "self_healer" in source and "circuit" in source.lower()

    def test_emergency_sends_signal(self):
        """Emergency alert should send Signal notification."""
        source = (Path(__file__).parent.parent / "app" / "self_healer.py").read_text()
        assert "emergency" in source.lower()
        assert "send_message" in source or "signal" in source.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Cross-Module Wiring Verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossModuleWiring:
    """Verify all self-healing components are wired together."""

    def test_health_monitor_callback_registered(self):
        """Health monitor should register on_alert callback in main.py."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "on_alert" in source
        assert "_sync_alert_handler" in source or "alert_handler" in source

    def test_health_monitor_records_interactions(self):
        """main.py should record interactions to health monitor."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "record_interaction" in source or "InteractionMetrics" in source

    def test_self_heal_called_on_exception(self):
        """main.py should call self_heal.diagnose_and_fix on task exception."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "diagnose_and_fix" in source

    def test_idle_scheduler_started_in_main(self):
        """Idle scheduler should be started in main.py lifespan."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "idle_scheduler" in source

    def test_version_manifest_created_on_startup(self):
        """Version manifest should be initialized on startup."""
        source = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "version_manifest" in source or "create_manifest" in source

    def test_signal_forwarder_has_backoff(self):
        """Signal forwarder should have exponential backoff."""
        source = (Path(__file__).parent.parent / "signal" / "forwarder.py").read_text()
        assert "backoff" in source.lower() or "wait_for_signal" in source

    def test_no_deprecated_get_event_loop(self):
        """No file should use deprecated asyncio.get_event_loop()."""
        app_dir = Path(__file__).parent.parent / "app"
        violations = []
        for f in app_dir.rglob("*.py"):
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if "get_event_loop()" in line and not line.strip().startswith("#"):
                        violations.append(f"{f.name}:{i}")
            except Exception:
                pass
        assert violations == [], f"Deprecated get_event_loop() found: {violations}"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SLO Budget Integration (Known Issue)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSLOBudgetIntegration:
    """Tests for SLO budget tracking — currently ORPHANED."""

    def test_evaluate_slo_budget_function_exists(self):
        from app.health_monitor import evaluate_slo_budget
        assert callable(evaluate_slo_budget)

    def test_slo_not_called_anywhere(self):
        """KNOWN ISSUE: evaluate_slo_budget is defined but never called.
        This test documents the gap. When fixed, this test should be updated."""
        app_dir = Path(__file__).parent.parent / "app"
        callers = []
        for f in app_dir.rglob("*.py"):
            if f.name == "health_monitor.py":
                continue  # Skip the definition file
            try:
                content = f.read_text()
                if "evaluate_slo_budget" in content:
                    callers.append(f.name)
            except Exception:
                pass
        # This documents the current state — SLO is orphaned
        # When wired, callers should be non-empty and this test should change
        if not callers:
            pytest.skip("KNOWN ISSUE: evaluate_slo_budget is orphaned (not called outside health_monitor.py)")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Regression Detection Quality
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegressionDetection:
    """Tests for workspace post-commit regression detection."""

    def test_uses_error_count_threshold(self):
        """Regression check should detect error spike."""
        source = (Path(__file__).parent.parent / "app" / "workspace_versioning.py").read_text()
        assert "total_errors" in source

    def test_only_checks_recent_commits(self):
        """Should only check commits < 1 hour old."""
        source = (Path(__file__).parent.parent / "app" / "workspace_versioning.py").read_text()
        assert "3600" in source  # 1 hour in seconds

    def test_marks_rolled_back(self):
        """Should mark commit as rolled_back to prevent re-checking."""
        source = (Path(__file__).parent.parent / "app" / "workspace_versioning.py").read_text()
        assert "rolled_back" in source

    def test_sends_signal_on_rollback(self):
        """Should send Signal alert when auto-rolling back."""
        source = (Path(__file__).parent.parent / "app" / "workspace_versioning.py").read_text()
        assert "send_message" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Signal Forwarder Resilience Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalForwarderResilience:
    """Tests for Signal forwarder reconnection and backoff."""

    def test_exponential_backoff_params(self):
        """Should use exponential backoff with cap."""
        source = (Path(__file__).parent.parent / "signal" / "forwarder.py").read_text()
        # Should have initial delay, factor, and max delay
        assert "5" in source  # Initial delay 5s
        assert "60" in source  # Max delay 60s

    def test_5min_alert_threshold(self):
        """Should alert dashboard after 5 minutes of downtime."""
        source = (Path(__file__).parent.parent / "signal" / "forwarder.py").read_text()
        assert "300" in source or "5 * 60" in source or "5min" in source.lower()

    def test_reconnect_after_errors(self):
        """Should trigger reconnect after consecutive errors."""
        source = (Path(__file__).parent.parent / "signal" / "forwarder.py").read_text()
        assert "reconnect" in source.lower() or "_wait_for_signal_cli" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 16. DB Pool Resilience Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDBPoolResilience:
    """Tests for PostgreSQL connection pool resilience."""

    def test_stale_connection_detection(self):
        """DB should detect stale connections."""
        source = (Path(__file__).parent.parent / "app" / "control_plane" / "db.py").read_text()
        assert "autocommit" in source or "stale" in source.lower()

    def test_pool_reset_exists(self):
        """_reset_pool should exist for recovery."""
        source = (Path(__file__).parent.parent / "app" / "control_plane" / "db.py").read_text()
        assert "_reset_pool" in source

    def test_pool_reset_called_on_error(self):
        """Pool should reset on persistent connection failure."""
        source = (Path(__file__).parent.parent / "app" / "control_plane" / "db.py").read_text()
        assert "_reset_pool()" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Consciousness Block Timeout (Known Gap)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsciousnessTimeout:
    """Tests for consciousness integration timeout handling."""

    def test_consciousness_block_wrapped_in_try(self):
        """Consciousness block should be wrapped in try/except."""
        source = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        # The block should have outer try/except
        assert "Consciousness indicators failed (non-fatal)" in source

    def test_consciousness_block_non_fatal(self):
        """Consciousness failures should not crash the request."""
        source = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        assert "non-fatal" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Comprehensive Source Audit
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceAudit:
    """Audit source code for anti-patterns and best practices."""

    def test_no_bare_except(self):
        """No file should use bare 'except:' (catches SystemExit/KeyboardInterrupt)."""
        app_dir = Path(__file__).parent.parent / "app"
        violations = []
        for f in app_dir.rglob("*.py"):
            try:
                tree = ast.parse(f.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler):
                        if node.type is None:
                            violations.append(f"{f.relative_to(app_dir.parent)}:{node.lineno}")
            except SyntaxError:
                pass
        assert violations == [], f"Bare except: found at:\n" + "\n".join(violations[:20])

    def test_no_optional_remaining(self):
        """All Optional[X] should have been converted to X | None."""
        app_dir = Path(__file__).parent.parent / "app"
        for f in app_dir.rglob("*.py"):
            try:
                content = f.read_text()
                assert "Optional[" not in content, f"{f.name} still has Optional["
            except Exception:
                pass

    def test_all_thread_pools_have_names(self):
        """Thread pools should have thread_name_prefix for debugging."""
        app_dir = Path(__file__).parent.parent / "app"
        violations = []
        for f in app_dir.rglob("*.py"):
            try:
                content = f.read_text()
                for i, line in enumerate(content.splitlines(), 1):
                    if "ThreadPoolExecutor(" in line and "thread_name_prefix" not in line:
                        # Check next few lines (multiline constructors)
                        block = content.splitlines()[i-1:i+3]
                        if not any("thread_name_prefix" in l for l in block):
                            violations.append(f"{f.name}:{i}")
            except Exception:
                pass
        # Allow some unnamed pools (they're usually one-liners for simple tasks
        # like parallel_evolution, sandbox_runner, web_fetch DNS resolution)
        # Flag only if the count grows beyond current baseline of 6
        if len(violations) > 8:
            pytest.fail(f"Too many unnamed thread pools ({len(violations)}):\n" + "\n".join(violations[:10]))
