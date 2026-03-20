"""Tests for app.metrics — composite scalar metric system."""
import os
import sys
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Mock the config module before importing anything else ────────────────────

class _FakeSecretStr:
    def __init__(self, v):
        self._v = v
    def get_secret_value(self):
        return self._v

class _FakeSettings:
    anthropic_api_key = _FakeSecretStr("fake-key")
    brave_api_key = _FakeSecretStr("fake-key")
    signal_bot_number = "+10000000000"
    signal_owner_number = "+10000000001"
    signal_cli_path = "/bin/true"
    signal_socket_path = "/tmp/test.sock"
    signal_http_url = ""
    signal_attachment_path = ""
    gateway_secret = _FakeSecretStr("a" * 64)
    gateway_port = 8765
    gateway_bind = "127.0.0.1"
    commander_model = "claude-sonnet-4-6"
    specialist_model = "claude-sonnet-4-6"
    sandbox_image = "test:latest"
    sandbox_timeout_seconds = 30
    sandbox_memory_limit = "512m"
    sandbox_cpu_limit = 0.5
    self_improve_cron = "0 3 * * *"
    self_improve_topic_file = "/tmp/test_learning_queue.md"
    max_parallel_crews = 3
    max_sub_agents = 4
    thread_pool_size = 6
    workspace_backup_repo = ""
    workspace_sync_cron = "0 * * * *"
    conversation_history_turns = 10
    evolution_iterations = 5
    evolution_deep_iterations = 15

import app.config as config_mod
config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_brave_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


# ── Tests ────────────────────────────────────────────────────────────────────

class TestCompositeScore:
    def test_score_range(self):
        from app.metrics import composite_score
        score = composite_score()
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] range"

    def test_compute_metrics_returns_all_keys(self):
        from app.metrics import compute_metrics
        m = compute_metrics()
        expected_keys = {
            "task_success_rate", "error_rate_24h", "error_rate_1h",
            "error_trend", "self_heal_rate",
            "output_quality", "skill_count", "avg_response_time_s",
            "evolution_efficiency", "avg_request_cost_usd",
            "composite_score", "measured_at",
        }
        assert expected_keys == set(m.keys()), f"Missing keys: {expected_keys - set(m.keys())}"

    def test_composite_score_consistency(self):
        from app.metrics import compute_metrics, composite_score
        m = compute_metrics()
        s = composite_score()
        assert m["composite_score"] == s

    def test_format_metrics(self):
        from app.metrics import compute_metrics, format_metrics
        m = compute_metrics()
        text = format_metrics(m)
        assert "Composite Score:" in text
        assert "Task Success Rate:" in text
        assert "Error Rate" in text


class TestSkillCount:
    def test_skill_count_with_no_dir(self, tmp_path, monkeypatch):
        import app.metrics as metrics_mod
        monkeypatch.setattr(metrics_mod, "SKILLS_DIR", tmp_path / "nonexistent")
        assert metrics_mod._skill_count() == 0

    def test_skill_count_with_files(self, tmp_path, monkeypatch):
        import app.metrics as metrics_mod
        monkeypatch.setattr(metrics_mod, "SKILLS_DIR", tmp_path)
        (tmp_path / "skill_a.md").write_text("# Skill A")
        (tmp_path / "skill_b.md").write_text("# Skill B")
        (tmp_path / "learning_queue.md").write_text("topic1")  # should be excluded
        assert metrics_mod._skill_count() == 2
