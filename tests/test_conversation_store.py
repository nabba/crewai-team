"""Tests for app.conversation_store — task timing/success tracking."""
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock config
from tests.test_metrics import _FakeSettings
import app.config as config_mod
config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


class TestTaskTracking:
    def setup_method(self):
        """Reset thread-local connection before each test."""
        import app.conversation_store as cs
        if hasattr(cs._local, "conn"):
            cs._local.conn = None

    def test_start_and_complete_task(self, tmp_path, monkeypatch):
        import app.conversation_store as cs
        monkeypatch.setattr(cs, "DB_PATH", tmp_path / "test.db")
        cs._local.conn = None  # force reconnect

        task_id = cs.start_task("+10000000001", crew="research")
        assert task_id > 0

        cs.complete_task(task_id, success=True)

        total, successful = cs.count_recent_tasks(hours=1)
        assert total == 1
        assert successful == 1

    def test_failed_task(self, tmp_path, monkeypatch):
        import app.conversation_store as cs
        monkeypatch.setattr(cs, "DB_PATH", tmp_path / "test.db")
        cs._local.conn = None

        task_id = cs.start_task("+10000000001", crew="coding")
        cs.complete_task(task_id, success=False, error_type="ValueError")

        total, successful = cs.count_recent_tasks(hours=1)
        assert total == 1
        assert successful == 0

    def test_multiple_tasks(self, tmp_path, monkeypatch):
        import app.conversation_store as cs
        monkeypatch.setattr(cs, "DB_PATH", tmp_path / "test.db")
        cs._local.conn = None

        for i in range(5):
            tid = cs.start_task("+10000000001")
            cs.complete_task(tid, success=(i % 2 == 0))

        total, successful = cs.count_recent_tasks(hours=1)
        assert total == 5
        assert successful == 3  # 0, 2, 4

    def test_avg_response_time(self, tmp_path, monkeypatch):
        import app.conversation_store as cs
        monkeypatch.setattr(cs, "DB_PATH", tmp_path / "test.db")
        cs._local.conn = None

        tid = cs.start_task("+10000000001")
        time.sleep(0.1)  # small delay to get nonzero duration
        cs.complete_task(tid, success=True)

        avg = cs.avg_response_time(hours=1)
        assert avg >= 0.05  # should be at least ~0.1s

    def test_messages_still_work(self, tmp_path, monkeypatch):
        """Ensure original message functionality isn't broken."""
        import app.conversation_store as cs
        monkeypatch.setattr(cs, "DB_PATH", tmp_path / "test.db")
        cs._local.conn = None

        cs.add_message("+10000000001", "user", "Hello")
        cs.add_message("+10000000001", "assistant", "Hi there!")

        history = cs.get_history("+10000000001", n=5)
        assert "Hello" in history
        assert "Hi there!" in history
