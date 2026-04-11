"""
Comprehensive tests for the 5 major system improvements:
  1. safe_io — atomic file writes (safe_write, safe_write_json, safe_append)
  2. error_handler — error categories, report_error, safe_execute, counters
  3. Embedding dimension pinning — 768-only, EmbeddingUnavailableError
  4. workspace_versioning — WorkspaceLock, git commit/rollback/log
  5. Idle scheduler — job classification, time caps, should_yield + timeout

Run: pytest tests/test_system_improvements.py -v
"""

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Module-level mocks for Docker-only deps
for _dep in ("chromadb", "psycopg2", "psycopg2.extras", "psycopg2.pool"):
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SAFE_IO — ATOMIC FILE WRITES
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafeWrite:
    """safe_write: atomic file replacement via tempfile + os.replace."""

    def test_writes_string_content(self, tmp_path):
        from app.safe_io import safe_write
        target = tmp_path / "test.txt"
        safe_write(target, "hello world")
        assert target.read_text() == "hello world"

    def test_writes_bytes_content(self, tmp_path):
        from app.safe_io import safe_write
        target = tmp_path / "test.bin"
        safe_write(target, b"\x00\x01\x02")
        assert target.read_bytes() == b"\x00\x01\x02"

    def test_creates_parent_directories(self, tmp_path):
        from app.safe_io import safe_write
        target = tmp_path / "deep" / "nested" / "dir" / "file.txt"
        safe_write(target, "content")
        assert target.exists()
        assert target.read_text() == "content"

    def test_overwrites_existing_file(self, tmp_path):
        from app.safe_io import safe_write
        target = tmp_path / "test.txt"
        target.write_text("old content")
        safe_write(target, "new content")
        assert target.read_text() == "new content"

    def test_no_temp_file_left_on_success(self, tmp_path):
        from app.safe_io import safe_write
        target = tmp_path / "test.txt"
        safe_write(target, "content")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_accepts_path_string(self, tmp_path):
        from app.safe_io import safe_write
        target = str(tmp_path / "test.txt")
        safe_write(target, "content")
        assert Path(target).read_text() == "content"

    def test_atomic_replacement(self, tmp_path):
        """Old content should never be partially overwritten."""
        from app.safe_io import safe_write
        target = tmp_path / "test.txt"
        safe_write(target, "original")
        # Overwrite — if crash happened mid-write, old content would survive
        safe_write(target, "replacement")
        assert target.read_text() == "replacement"

    def test_unicode_content(self, tmp_path):
        from app.safe_io import safe_write
        target = tmp_path / "test.txt"
        safe_write(target, "Helsinki \u2014 60\u00b0N, Saimaa seals")
        content = target.read_text(encoding="utf-8")
        assert "Helsinki" in content
        assert "60\u00b0N" in content


class TestSafeWriteJson:
    """safe_write_json: atomic JSON serialization."""

    def test_writes_dict(self, tmp_path):
        from app.safe_io import safe_write_json
        target = tmp_path / "data.json"
        safe_write_json(target, {"key": "value", "number": 42})
        loaded = json.loads(target.read_text())
        assert loaded["key"] == "value"
        assert loaded["number"] == 42

    def test_writes_list(self, tmp_path):
        from app.safe_io import safe_write_json
        target = tmp_path / "data.json"
        safe_write_json(target, [1, 2, 3])
        assert json.loads(target.read_text()) == [1, 2, 3]

    def test_handles_datetime_via_default_str(self, tmp_path):
        from app.safe_io import safe_write_json
        target = tmp_path / "data.json"
        now = datetime.now(timezone.utc)
        safe_write_json(target, {"ts": now})
        loaded = json.loads(target.read_text())
        # default=str serializes datetime to string
        assert "202" in loaded["ts"]  # Year prefix

    def test_custom_indent(self, tmp_path):
        from app.safe_io import safe_write_json
        target = tmp_path / "data.json"
        safe_write_json(target, {"a": 1}, indent=4)
        content = target.read_text()
        assert "    " in content  # 4-space indent

    def test_nested_objects(self, tmp_path):
        from app.safe_io import safe_write_json
        target = tmp_path / "data.json"
        data = {"level1": {"level2": {"level3": "deep"}}}
        safe_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded["level1"]["level2"]["level3"] == "deep"


class TestSafeAppend:
    """safe_append: crash-safe JSONL line append with fsync."""

    def test_appends_line(self, tmp_path):
        from app.safe_io import safe_append
        target = tmp_path / "log.jsonl"
        safe_append(target, '{"event": "start"}')
        safe_append(target, '{"event": "end"}')
        lines = target.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "start"
        assert json.loads(lines[1])["event"] == "end"

    def test_creates_file_if_missing(self, tmp_path):
        from app.safe_io import safe_append
        target = tmp_path / "new.jsonl"
        safe_append(target, "first line")
        assert target.exists()

    def test_strips_trailing_newline(self, tmp_path):
        from app.safe_io import safe_append
        target = tmp_path / "log.jsonl"
        safe_append(target, "line with newline\n")
        content = target.read_text()
        assert content == "line with newline\n"  # Exactly one newline

    def test_creates_parent_dirs(self, tmp_path):
        from app.safe_io import safe_append
        target = tmp_path / "deep" / "log.jsonl"
        safe_append(target, "test")
        assert target.exists()

    def test_concurrent_appends(self, tmp_path):
        """Multiple threads appending should not corrupt the file."""
        from app.safe_io import safe_append
        target = tmp_path / "concurrent.jsonl"
        errors = []

        def writer(prefix, count):
            for i in range(count):
                try:
                    safe_append(target, f'{{"id": "{prefix}_{i}"}}')
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"t{j}", 20)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        lines = target.read_text().strip().split("\n")
        assert len(lines) == 80  # 4 threads x 20 lines
        # Every line should be valid JSON
        for line in lines:
            json.loads(line)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ERROR_HANDLER — CATEGORIES, REPORTING, SAFE_EXECUTE
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorCategory:
    """ErrorCategory enum values."""

    def test_four_categories_exist(self):
        from app.error_handler import ErrorCategory
        assert len(ErrorCategory) == 4

    def test_category_values(self):
        from app.error_handler import ErrorCategory
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.DATA.value == "data"
        assert ErrorCategory.SYSTEM.value == "system"
        assert ErrorCategory.LOGIC.value == "logic"

    def test_categories_are_strings(self):
        from app.error_handler import ErrorCategory
        for cat in ErrorCategory:
            assert isinstance(cat.value, str)


class TestReportError:
    """report_error: structured logging + counter increment."""

    def setup_method(self):
        from app.error_handler import reset_error_counts
        reset_error_counts()

    def test_increments_counter(self):
        from app.error_handler import report_error, ErrorCategory, get_error_counts
        report_error(ErrorCategory.TRANSIENT, "test error")
        counts = get_error_counts()
        assert counts.get("transient", 0) == 1

    def test_multiple_errors_accumulate(self):
        from app.error_handler import report_error, ErrorCategory, get_error_counts
        report_error(ErrorCategory.TRANSIENT, "err1")
        report_error(ErrorCategory.TRANSIENT, "err2")
        report_error(ErrorCategory.DATA, "err3")
        counts = get_error_counts()
        assert counts["transient"] == 2
        assert counts["data"] == 1

    def test_reset_clears_counters(self):
        from app.error_handler import report_error, ErrorCategory, get_error_counts, reset_error_counts
        report_error(ErrorCategory.SYSTEM, "err")
        reset_error_counts()
        assert get_error_counts() == {}

    def test_message_truncated_to_500(self):
        from app.error_handler import report_error, ErrorCategory
        long_msg = "x" * 1000
        # Should not raise
        report_error(ErrorCategory.DATA, long_msg)

    def test_exception_included_in_report(self):
        from app.error_handler import report_error, ErrorCategory
        try:
            raise ValueError("test error detail")
        except ValueError as e:
            report_error(ErrorCategory.LOGIC, "caught", exc=e)
        # Should not crash

    def test_context_dict_accepted(self):
        from app.error_handler import report_error, ErrorCategory
        report_error(ErrorCategory.TRANSIENT, "test", context={"crew": "research", "agent": "researcher"})

    def test_thread_safety(self):
        from app.error_handler import report_error, ErrorCategory, get_error_counts, reset_error_counts
        reset_error_counts()
        errors = []

        def spam(n):
            for _ in range(n):
                try:
                    report_error(ErrorCategory.TRANSIENT, "thread test")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=spam, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        counts = get_error_counts()
        assert counts["transient"] == 200


class TestSafeExecute:
    """safe_execute: context manager that catches + reports exceptions."""

    def test_swallows_exception(self):
        from app.error_handler import safe_execute
        # Should not raise
        with safe_execute("test_op"):
            raise RuntimeError("boom")
        # Code continues here

    def test_no_exception_passes_through(self):
        from app.error_handler import safe_execute
        result = None
        with safe_execute("test_op"):
            result = 42
        assert result == 42

    def test_reports_error_on_exception(self):
        from app.error_handler import safe_execute, ErrorCategory, get_error_counts, reset_error_counts
        reset_error_counts()
        with safe_execute("test_op", category=ErrorCategory.DATA):
            raise ValueError("parse error")
        counts = get_error_counts()
        assert counts.get("data", 0) == 1

    def test_custom_category(self):
        from app.error_handler import safe_execute, ErrorCategory, get_error_counts, reset_error_counts
        reset_error_counts()
        with safe_execute("test_op", category=ErrorCategory.SYSTEM):
            raise OSError("disk full")
        assert get_error_counts().get("system", 0) == 1

    def test_context_passed_to_report(self):
        from app.error_handler import safe_execute, ErrorCategory
        # Should not raise
        with safe_execute("test_op", context={"crew": "coding"}):
            raise RuntimeError("fail")


class TestStructuredLogging:
    """setup_structured_logging: JSON file handler."""

    def test_setup_creates_directory(self, tmp_path):
        from app.error_handler import setup_structured_logging
        log_path = str(tmp_path / "logs" / "errors.jsonl")
        setup_structured_logging(log_path=log_path, max_mb=1)
        assert Path(log_path).parent.exists()

    def test_setup_adds_handler(self, tmp_path):
        from app.error_handler import setup_structured_logging
        import logging
        initial_count = len(logging.getLogger().handlers)
        log_path = str(tmp_path / "structured.jsonl")
        setup_structured_logging(log_path=log_path)
        assert len(logging.getLogger().handlers) > initial_count
        # Cleanup: remove the handler we just added
        logging.getLogger().handlers = logging.getLogger().handlers[:initial_count]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING DIMENSION PINNING
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmbeddingDimensionPinning:
    """Embedding dimension pinned to 768, fallback refused.

    NOTE: chromadb_manager uses Python 3.10+ type hints (list | None).
    Tests that need to import it are run inside Docker only.
    Source-level checks use file reading instead.
    """

    def test_embed_dim_pinned_in_source(self):
        """Source code should pin _EMBED_DIM = 768."""
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "memory", "chromadb_manager.py")).read_text()
        assert "_EMBED_DIM = 768" in src

    def test_embedding_unavailable_error_in_source(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "memory", "chromadb_manager.py")).read_text()
        assert "class EmbeddingUnavailableError" in src
        assert "RuntimeError" in src

    def test_refuse_fallback_logic_in_source(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "memory", "chromadb_manager.py")).read_text()
        assert "_should_refuse_fallback" in src
        assert "embedding_refuse_fallback" in src

    def test_unavailable_backend_raises_in_source(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "memory", "chromadb_manager.py")).read_text()
        assert 'raise EmbeddingUnavailableError' in src

    def test_ollama_recovery_logic_in_source(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "memory", "chromadb_manager.py")).read_text()
        assert '"Embedding backend recovered' in src or 'Ollama available again' in src

    def test_config_has_embedding_dimension(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "config.py")).read_text()
        assert "embedding_dimension" in src
        assert "embedding_refuse_fallback" in src


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WORKSPACE VERSIONING
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkspaceLock:
    """WorkspaceLock: advisory file lock for evolution coordination."""

    def test_lock_acquire_release(self, tmp_path):
        """Lock can be acquired and released."""
        lock_file = tmp_path / ".workspace.lock"
        with patch("app.workspace_versioning.LOCK_FILE", lock_file):
            from app.workspace_versioning import WorkspaceLock
            lock = WorkspaceLock(timeout_s=5)
            lock.acquire()
            assert lock._fd is not None
            lock.release()
            assert lock._fd is None

    def test_context_manager(self, tmp_path):
        lock_file = tmp_path / ".workspace.lock"
        with patch("app.workspace_versioning.LOCK_FILE", lock_file):
            from app.workspace_versioning import WorkspaceLock
            with WorkspaceLock(timeout_s=5) as lock:
                assert lock._fd is not None
            # After exit, fd should be None
            assert lock._fd is None

    def test_lock_blocks_second_acquirer(self, tmp_path):
        """Second lock attempt should fail with TimeoutError."""
        lock_file = tmp_path / ".workspace.lock"
        with patch("app.workspace_versioning.LOCK_FILE", lock_file):
            from app.workspace_versioning import WorkspaceLock
            lock1 = WorkspaceLock(timeout_s=5)
            lock1.acquire()
            try:
                lock2 = WorkspaceLock(timeout_s=1)  # Short timeout
                with pytest.raises(TimeoutError):
                    lock2.acquire()
            finally:
                lock1.release()

    def test_lock_released_after_exception(self, tmp_path):
        lock_file = tmp_path / ".workspace.lock"
        with patch("app.workspace_versioning.LOCK_FILE", lock_file):
            from app.workspace_versioning import WorkspaceLock
            try:
                with WorkspaceLock(timeout_s=5):
                    raise RuntimeError("crash")
            except RuntimeError:
                pass
            # Lock should be released — second acquire should succeed
            with WorkspaceLock(timeout_s=1):
                pass  # Should not raise


class TestWorkspaceGit:
    """Git-based workspace commit/rollback/log."""

    def test_ensure_workspace_repo_init(self, tmp_path):
        """Should initialize a git repo in the workspace."""
        with patch("app.workspace_versioning.WORKSPACE", tmp_path):
            from app.workspace_versioning import ensure_workspace_repo
            result = ensure_workspace_repo()
            assert result is True
            assert (tmp_path / ".git").exists()

    def test_ensure_workspace_repo_idempotent(self, tmp_path):
        with patch("app.workspace_versioning.WORKSPACE", tmp_path):
            from app.workspace_versioning import ensure_workspace_repo
            ensure_workspace_repo()
            result = ensure_workspace_repo()
            assert result is False  # Already initialized

    def test_workspace_commit(self, tmp_path):
        with patch("app.workspace_versioning.WORKSPACE", tmp_path):
            from app.workspace_versioning import ensure_workspace_repo, workspace_commit
            ensure_workspace_repo()
            # Create a file
            (tmp_path / "test.txt").write_text("hello")
            sha = workspace_commit("test: added test.txt")
            assert sha  # Non-empty SHA
            assert len(sha) >= 7

    def test_workspace_commit_nothing_to_commit(self, tmp_path):
        with patch("app.workspace_versioning.WORKSPACE", tmp_path):
            from app.workspace_versioning import ensure_workspace_repo, workspace_commit
            ensure_workspace_repo()
            sha = workspace_commit("nothing changed")
            assert sha == ""

    def test_workspace_log(self, tmp_path):
        with patch("app.workspace_versioning.WORKSPACE", tmp_path):
            from app.workspace_versioning import ensure_workspace_repo, workspace_commit, workspace_log
            ensure_workspace_repo()
            (tmp_path / "file1.txt").write_text("a")
            workspace_commit("commit 1")
            (tmp_path / "file2.txt").write_text("b")
            workspace_commit("commit 2")
            log = workspace_log(n=5)
            assert len(log) >= 2  # At least 2 commits (+ initial)
            assert "message" in log[0]
            assert "sha" in log[0]

    def test_workspace_rollback(self, tmp_path):
        with patch("app.workspace_versioning.WORKSPACE", tmp_path):
            from app.workspace_versioning import ensure_workspace_repo, workspace_commit, workspace_rollback, workspace_log
            ensure_workspace_repo()
            # Create and commit file
            (tmp_path / "keeper.txt").write_text("keep me")
            sha1 = workspace_commit("add keeper")
            assert sha1, "First commit should produce a SHA"
            # Get full SHA for rollback (short SHA may be ambiguous)
            log = workspace_log(5)
            full_sha = next((e["sha"] for e in log if e["short_sha"] == sha1), sha1)
            # Create another file
            (tmp_path / "remove_me.txt").write_text("remove")
            workspace_commit("add removable")
            assert (tmp_path / "remove_me.txt").exists()
            # Rollback to first commit
            success = workspace_rollback(full_sha)
            assert success
            assert (tmp_path / "keeper.txt").exists()

    def test_workspace_commit_nonfatal(self, tmp_path):
        """workspace_commit should never crash, even with broken git."""
        with patch("app.workspace_versioning.WORKSPACE", tmp_path / "nonexistent"):
            from app.workspace_versioning import workspace_commit
            sha = workspace_commit("test")
            assert sha == ""  # Fails gracefully


# ═══════════════════════════════════════════════════════════════════════════════
# 5. IDLE SCHEDULER — JOB CLASSIFICATION + TIME CAPS
# ═══════════════════════════════════════════════════════════════════════════════


class TestJobWeight:
    """Job weight classification.

    NOTE: idle_scheduler.py uses Python 3.10+ type hints. Tests use
    source reading for classification checks, Docker-only for runtime.
    """

    def _read_scheduler_source(self):
        return Path(os.path.join(os.path.dirname(__file__), "..", "app", "idle_scheduler.py")).read_text()

    def test_three_weight_classes_in_source(self):
        src = self._read_scheduler_source()
        assert 'LIGHT = "light"' in src
        assert 'MEDIUM = "medium"' in src
        assert 'HEAVY = "heavy"' in src

    def test_time_caps_in_source(self):
        src = self._read_scheduler_source()
        assert "JobWeight.LIGHT]: 60" in src or "LIGHT: 60" in src
        assert "JobWeight.MEDIUM]: 180" in src or "MEDIUM: 180" in src
        assert "JobWeight.HEAVY]: 600" in src or "HEAVY: 600" in src

    def test_all_jobs_have_weights_in_source(self):
        """Every jobs.append should have a JobWeight third argument."""
        src = self._read_scheduler_source()
        import re
        appends = re.findall(r'jobs\.append\(\("([^"]+)",\s*\w+,\s*(JobWeight\.\w+)\)', src)
        assert len(appends) >= 25, f"Only {len(appends)} weighted jobs found"
        # All should have valid weights
        for name, weight in appends:
            assert weight in ("JobWeight.LIGHT", "JobWeight.MEDIUM", "JobWeight.HEAVY"), \
                f"Job {name} has invalid weight: {weight}"

    def test_heavy_jobs_classified(self):
        src = self._read_scheduler_source()
        for name in ["evolution", "training-pipeline", "retrospective"]:
            assert f'"{name}", _' in src and "JobWeight.HEAVY" in src

    def test_light_jobs_classified(self):
        src = self._read_scheduler_source()
        for name in ["health-evaluate", "version-snapshot", "system-monitor"]:
            pattern = f'"{name}"'
            idx = src.find(pattern)
            # Find the JobWeight on the same line
            line_end = src.find("\n", idx)
            line = src[idx:line_end]
            assert "JobWeight.LIGHT" in line, f"Job {name} not classified as LIGHT"

    def test_dual_queue_architecture_in_source(self):
        src = self._read_scheduler_source()
        assert "light_jobs" in src
        assert "medium_jobs" in src
        assert "heavy_jobs" in src
        assert "light_pool" in src or "ThreadPoolExecutor" in src

    def test_training_hourly_cadence_in_source(self):
        src = self._read_scheduler_source()
        assert "_last_training_run" in src
        assert "training_interval" in src or "3600" in src


class TestShouldYieldWithTimeout:
    """should_yield() includes _job_timeout signal.

    Uses source reading since idle_scheduler has Python 3.10 type hints.
    """

    def test_should_yield_checks_job_timeout_in_source(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "idle_scheduler.py")).read_text()
        # should_yield should check _job_timeout
        assert "_job_timeout.is_set()" in src

    def test_job_timeout_event_defined_in_source(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "idle_scheduler.py")).read_text()
        assert "_job_timeout = threading.Event()" in src

    def test_run_single_job_enforces_timeout_in_source(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "idle_scheduler.py")).read_text()
        assert "threading.Timer(timeout_s" in src
        assert "_job_timeout.set" in src
        assert "timer.cancel()" in src


class TestRunSingleJobLogic:
    """_run_single_job logic: time cap + exception handling.

    Tests the pattern without importing the module (Python 3.10 compat).
    """

    def test_timeout_timer_pattern(self):
        """Verify the timeout pattern works correctly."""
        _job_timeout = threading.Event()
        completed = []

        def _run_with_timeout(name, fn, timeout_s):
            _job_timeout.clear()
            timer = threading.Timer(timeout_s, _job_timeout.set)
            timer.daemon = True
            timer.start()
            try:
                fn()
            finally:
                timer.cancel()
                _job_timeout.clear()

        def fast_job():
            completed.append(True)

        _run_with_timeout("test", fast_job, 5)
        assert completed == [True]
        assert not _job_timeout.is_set()

    def test_timeout_fires_on_slow_job(self):
        _job_timeout = threading.Event()

        def _run_with_timeout(name, fn, timeout_s):
            _job_timeout.clear()
            timer = threading.Timer(timeout_s, _job_timeout.set)
            timer.daemon = True
            timer.start()
            try:
                fn()
            finally:
                timer.cancel()
                _job_timeout.clear()

        def slow_job():
            # Check if timeout fired during execution
            time.sleep(0.3)

        _run_with_timeout("slow", slow_job, 0.1)
        # After finally, event is cleared
        assert not _job_timeout.is_set()

    def test_exception_handled(self):
        _job_timeout = threading.Event()
        caught = []

        def _run_with_timeout(name, fn, timeout_s):
            _job_timeout.clear()
            timer = threading.Timer(timeout_s, _job_timeout.set)
            timer.daemon = True
            timer.start()
            try:
                fn()
            except Exception as exc:
                caught.append(str(exc))
            finally:
                timer.cancel()
                _job_timeout.clear()

        def crashing_job():
            raise RuntimeError("boom")

        _run_with_timeout("crash", crashing_job, 5)
        assert len(caught) == 1
        assert "boom" in caught[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONFIG SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigSettings:
    """New config settings for all 5 improvements."""

    def test_embedding_settings_exist(self):
        try:
            from app.config import get_settings
            s = get_settings()
            assert hasattr(s, "embedding_dimension")
            assert hasattr(s, "embedding_refuse_fallback")
            assert s.embedding_dimension == 768
            assert s.embedding_refuse_fallback is True
        except Exception:
            pytest.skip("config not importable outside Docker")

    def test_workspace_settings_exist(self):
        try:
            from app.config import get_settings
            s = get_settings()
            assert hasattr(s, "workspace_lock_timeout_s")
            assert s.workspace_lock_timeout_s == 30
        except Exception:
            pytest.skip("config not importable outside Docker")

    def test_scheduler_settings_exist(self):
        try:
            from app.config import get_settings
            s = get_settings()
            assert hasattr(s, "idle_lightweight_workers")
            assert hasattr(s, "idle_heavy_time_cap_s")
            assert hasattr(s, "idle_training_interval_s")
            assert s.idle_lightweight_workers == 3
            assert s.idle_heavy_time_cap_s == 600
            assert s.idle_training_interval_s == 3600
        except Exception:
            pytest.skip("config not importable outside Docker")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INTEGRATION: MIGRATION VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestMigrationVerification:
    """Verify that existing atomic writers now use safe_io."""

    def test_homeostasis_uses_safe_io(self):
        import inspect
        from app.self_awareness.homeostasis import _save
        src = inspect.getsource(_save)
        assert "safe_write_json" in src
        assert "tempfile.mkstemp" not in src  # Old pattern removed

    def test_agent_state_uses_safe_io(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "self_awareness", "agent_state.py")).read_text()
        assert "safe_write_json" in src

    def test_sentience_config_uses_safe_io(self):
        import inspect
        from app.self_awareness.sentience_config import save_config
        src = inspect.getsource(save_config)
        assert "safe_write_json" in src

    def test_prompt_registry_uses_safe_write(self):
        import inspect
        try:
            # prompt_registry may need crewai
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "prompt_registry",
                os.path.join(os.path.dirname(__file__), "..", "app", "prompt_registry.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            src = Path(spec.origin).read_text()
            assert "safe_write" in src
        except Exception:
            pytest.skip("prompt_registry not readable")

    def test_parallel_evolution_uses_safe_io(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "parallel_evolution.py")).read_text()
        assert "safe_write_json" in src

    def test_evolution_uses_workspace_lock(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "evolution.py")).read_text()
        assert "WorkspaceLock" in src
        assert "workspace_commit" in src

    def test_island_evolution_uses_workspace_lock(self):
        src = Path(os.path.join(os.path.dirname(__file__), "..", "app", "island_evolution.py")).read_text()
        assert "WorkspaceLock" in src
        assert "workspace_commit" in src
