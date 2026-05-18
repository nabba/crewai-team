"""Unit tests for app.deploy_staging.

Closes the audit-2026-05-18 staging contract:
  - stage_for_deploy writes full content into APPLIED_CODE_DIR
  - refuses path traversal, absolute paths, escapes-after-resolution
  - refuses TIER_IMMUTABLE files (defense in depth)
  - unstage is idempotent and silent on missing files
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings  # noqa: E402
import app.config as config_mod  # noqa: E402

config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


@pytest.fixture
def staging_sandbox(tmp_path, monkeypatch):
    """Redirect APPLIED_CODE_DIR into a tmp sandbox."""
    from app import deploy_staging

    applied = tmp_path / "applied_code"
    applied.mkdir(parents=True)
    monkeypatch.setattr(deploy_staging, "APPLIED_CODE_DIR", applied)
    return applied


# ── Happy path ──────────────────────────────────────────────────────────────


class TestStageForDeploySuccess:
    def test_stages_an_open_file(self, staging_sandbox):
        from app.deploy_staging import stage_for_deploy

        dest = stage_for_deploy(
            "app/agents/researcher.py",
            "x = 1\n",
            source="test",
        )
        assert dest.exists()
        assert dest.read_text() == "x = 1\n"
        assert (staging_sandbox / "app/agents/researcher.py").exists()

    def test_creates_parent_directories(self, staging_sandbox):
        from app.deploy_staging import stage_for_deploy

        stage_for_deploy(
            "app/agents/subdir/deeply/nested.py",
            "y = 2\n",
            source="test",
        )
        assert (staging_sandbox / "app/agents/subdir/deeply/nested.py").exists()

    def test_writes_full_content_not_truncated(self, staging_sandbox):
        """Regression: human_gate.to_dict truncates to 5000 chars for display;
        stage_for_deploy must NOT truncate — auto_deployer needs the full file
        for AST + constitutional checks."""
        from app.deploy_staging import stage_for_deploy

        big = "# comment\n" * 1000  # ~10k chars
        dest = stage_for_deploy("app/agents/big.py", big, source="test")
        assert dest.read_text() == big
        assert len(dest.read_text()) > 5000

    def test_overwrites_existing_staged_file(self, staging_sandbox):
        from app.deploy_staging import stage_for_deploy

        stage_for_deploy("app/agents/x.py", "v1\n", source="t1")
        dest = stage_for_deploy("app/agents/x.py", "v2\n", source="t2")
        assert dest.read_text() == "v2\n"


# ── Safety refusals ─────────────────────────────────────────────────────────


class TestStageForDeployRefusal:
    def test_refuses_path_traversal(self, staging_sandbox):
        from app.deploy_staging import stage_for_deploy

        with pytest.raises(ValueError, match="unsafe path"):
            stage_for_deploy("../../etc/passwd", "x", source="evil")

    def test_refuses_absolute_path(self, staging_sandbox):
        from app.deploy_staging import stage_for_deploy

        with pytest.raises(ValueError, match="unsafe path"):
            stage_for_deploy("/etc/passwd", "x", source="evil")

    def test_refuses_empty_path(self, staging_sandbox):
        from app.deploy_staging import stage_for_deploy

        with pytest.raises(ValueError, match="unsafe path"):
            stage_for_deploy("", "x", source="evil")

    def test_refuses_immutable_file(self, staging_sandbox):
        """Defense in depth: auto_deployer._validate_deploy_batch would
        catch this at deploy time, but failing at stage time gives a
        clearer stack trace and keeps the IMMUTABLE file out of
        APPLIED_CODE_DIR entirely."""
        from app.deploy_staging import stage_for_deploy

        with pytest.raises(ValueError, match="IMMUTABLE"):
            stage_for_deploy("app/sanitize.py", "x = 1\n", source="evil")

    def test_refuses_immutable_evaluation_module(self, staging_sandbox):
        from app.deploy_staging import stage_for_deploy

        # experiment_runner.py is IMMUTABLE — evaluation infrastructure.
        with pytest.raises(ValueError, match="IMMUTABLE"):
            stage_for_deploy("app/experiment_runner.py", "x = 1\n", source="evil")

    def test_refuses_path_that_resolves_outside_applied_code(
        self, staging_sandbox, tmp_path
    ):
        """Belt-and-suspenders: ../../ in the middle of a path that doesn't
        start with .. but still escapes after resolution."""
        from app.deploy_staging import stage_for_deploy

        # Path with embedded .. in middle
        with pytest.raises(ValueError, match="unsafe path"):
            stage_for_deploy("app/agents/../../../escape.py", "x", source="evil")


# ── Unstage ─────────────────────────────────────────────────────────────────


class TestUnstage:
    def test_removes_staged_file(self, staging_sandbox):
        from app.deploy_staging import stage_for_deploy, unstage

        stage_for_deploy("app/agents/x.py", "x\n", source="test")
        assert (staging_sandbox / "app/agents/x.py").exists()
        unstage("app/agents/x.py")
        assert not (staging_sandbox / "app/agents/x.py").exists()

    def test_silent_on_missing_file(self, staging_sandbox):
        from app.deploy_staging import unstage

        # Should not raise
        unstage("app/agents/never_existed.py")

    def test_silent_on_unsafe_path(self, staging_sandbox):
        from app.deploy_staging import unstage

        # Should not raise, should not touch anything outside APPLIED_CODE_DIR
        unstage("../../etc/passwd")
        unstage("/etc/passwd")
        unstage("")
