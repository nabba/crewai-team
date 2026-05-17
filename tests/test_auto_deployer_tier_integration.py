"""Integration tests for the live-deploy tier-protection boundary.

Verifies that auto_deployer._deploy_locked() consults the three-tier
protection model via DeployEvidence and refuses GATED / IMMUTABLE files
that lack the required evidence. Closes the historical gap where
validate_mutation_for_tier() was defined and tested but never called at
the boundary (audit 2026-05-12; productization plan WP A1).
"""
import json
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


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def deploy_sandbox(tmp_path, monkeypatch):
    """Redirect auto_deployer's path constants into a tmp sandbox.

    The auto_deployer module reads APPLIED_CODE_DIR / LIVE_CODE_DIR /
    BACKUP_DIR / DEPLOY_LOG at module level. We rebind these on the
    module object so each test gets an isolated filesystem.
    """
    from app import auto_deployer

    applied = tmp_path / "applied_code"
    live = tmp_path / "live"
    backups = tmp_path / "backups"
    log = tmp_path / "deploy_log.json"

    applied.mkdir(parents=True)
    live.mkdir(parents=True)
    backups.mkdir(parents=True)
    # Pre-populate live/app/ so backups have something to copy
    (live / "app").mkdir(parents=True)

    monkeypatch.setattr(auto_deployer, "APPLIED_CODE_DIR", applied)
    monkeypatch.setattr(auto_deployer, "LIVE_CODE_DIR", live)
    monkeypatch.setattr(auto_deployer, "BACKUP_DIR", backups)
    monkeypatch.setattr(auto_deployer, "DEPLOY_LOG", log)

    # Silence side-effect notifications during tests
    monkeypatch.setattr(
        "app.signal_client.send_message",
        lambda *a, **kw: None,
        raising=False,
    )

    # Avoid hot-reload + post-deploy monitor side effects in the success path.
    # _hot_reload_modules_safe walks sys.modules; in tests we don't want it
    # touching real modules. _post_deploy_monitor starts a thread that
    # sleeps 60s, which we don't want either.
    monkeypatch.setattr(
        auto_deployer,
        "_hot_reload_modules_safe",
        lambda deployed, backup: ([], []),
    )
    monkeypatch.setattr(
        auto_deployer,
        "_post_deploy_monitor",
        lambda deployed, backup, reason: None,
    )

    return {
        "applied": applied,
        "live": live,
        "backups": backups,
        "log": log,
    }


def _write_applied_file(sandbox, relpath: str, content: str = "x = 1\n") -> Path:
    """Stage a file in applied_code/<relpath> for deploy."""
    dest = sandbox["applied"] / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    return dest


# ── Test cases ──────────────────────────────────────────────────────────────


class TestOpenFileDirectDeploy:
    """Case 1: direct deploy of an OPEN file succeeds with normal checks."""

    def test_open_file_deploys_through_direct_evidence(self, deploy_sandbox):
        from app.auto_deployer import run_deploy, DeployEvidence

        _write_applied_file(deploy_sandbox, "app/agents/researcher.py")

        result = run_deploy(
            reason="test-open-direct",
            evidence=DeployEvidence.direct("test-open-direct"),
        )

        assert "Deployed 1 files" in result
        assert (deploy_sandbox["live"] / "app/agents/researcher.py").exists()
        # applied_code/ cleaned up on success
        assert not (deploy_sandbox["applied"] / "app/agents/researcher.py").exists()


class TestGatedRefusedWithoutCanary:
    """Cases 2, 6, 7, 8: GATED files refused at the boundary without canary evidence.

    Regardless of which source tag the evidence carries (direct / proposal /
    human_gate / canary_fallback), the boundary refuses GATED. This is the
    invariant that makes the gate trustworthy.
    """

    def test_direct_evidence_blocks_gated(self, deploy_sandbox, monkeypatch):
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        _write_applied_file(deploy_sandbox, "app/evolution.py")

        result = run_deploy(
            reason="direct-gated",
            evidence=DeployEvidence.direct("direct-gated"),
        )

        assert "Deploy blocked" in result
        assert "tier protection" in result
        assert "canary" in result.lower()
        # Live file was NOT overwritten
        assert not (deploy_sandbox["live"] / "app/evolution.py").exists()

    def test_proposal_evidence_cannot_bypass_gated(self, deploy_sandbox, monkeypatch):
        """Operator approval on a proposal does not replace canary evidence."""
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        _write_applied_file(deploy_sandbox, "app/evolution.py")

        # Even with an operator approval ID, has_canary_pass=False blocks GATED.
        evidence = DeployEvidence(
            reason="proposal-approved",
            source="proposal",
            has_canary_pass=False,
            operator_approval_id="proposal-42",
        )
        result = run_deploy(reason="proposal-approved", evidence=evidence)

        assert "Deploy blocked" in result
        assert "tier protection" in result

    def test_human_gate_evidence_cannot_bypass_gated(self, deploy_sandbox, monkeypatch):
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        _write_applied_file(deploy_sandbox, "app/avo_operator.py")

        evidence = DeployEvidence(
            reason="human-approved-xyz",
            source="human_gate",
            has_canary_pass=False,
            operator_approval_id="approval-xyz",
        )
        result = run_deploy(reason="human-approved-xyz", evidence=evidence)

        assert "Deploy blocked" in result

    def test_canary_fallback_evidence_cannot_bypass_gated(self, deploy_sandbox, monkeypatch):
        """When canary is disabled or has no baseline, the fallback goes direct.
        That fallback must NOT carry canary evidence."""
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        _write_applied_file(deploy_sandbox, "app/evolution.py")

        evidence = DeployEvidence.direct("canary-disabled", source="canary_fallback")
        result = run_deploy(reason="canary-disabled", evidence=evidence)

        assert "Deploy blocked" in result


class TestGatedRefusedWithoutAutoDeploy:
    """Case 3: GATED file refused when EVOLUTION_AUTO_DEPLOY=false."""

    def test_gated_blocked_without_env(self, deploy_sandbox, monkeypatch):
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "false")
        # Use a fake settings instance to enforce the env value (the validator
        # reads BOTH the env var AND get_settings().evolution_auto_deploy).
        fs = _FakeSettings()
        fs.evolution_auto_deploy = False
        config_mod.get_settings = lambda: fs
        try:
            _write_applied_file(deploy_sandbox, "app/evolution.py")
            evidence = DeployEvidence(
                reason="real-canary",
                source="canary",
                has_canary_pass=True,
                canary_id="canary-1",
            )
            result = run_deploy(reason="real-canary", evidence=evidence)
            assert "Deploy blocked" in result
            assert "EVOLUTION_AUTO_DEPLOY" in result
        finally:
            config_mod.get_settings = lambda: _FakeSettings()


class TestGatedAllowedWithCanary:
    """Case 4: GATED file with valid canary evidence + auto_deploy=true deploys."""

    def test_gated_deploys_with_canary_evidence(self, deploy_sandbox, monkeypatch):
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        fs = _FakeSettings()
        fs.evolution_auto_deploy = True
        config_mod.get_settings = lambda: fs
        try:
            _write_applied_file(deploy_sandbox, "app/evolution.py")
            evidence = DeployEvidence.from_canary(
                reason="real-canary",
                canary_id="canary-real-1",
            )
            result = run_deploy(reason="real-canary", evidence=evidence)
            assert "Deployed 1 files" in result, f"expected success, got: {result}"
            assert (deploy_sandbox["live"] / "app/evolution.py").exists()
        finally:
            config_mod.get_settings = lambda: _FakeSettings()


class TestImmutableAlwaysBlocked:
    """Case 5: IMMUTABLE file refused even with canary evidence."""

    def test_immutable_blocked_with_canary(self, deploy_sandbox, monkeypatch):
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        # Use a known IMMUTABLE file
        _write_applied_file(deploy_sandbox, "app/sanitize.py")

        evidence = DeployEvidence.from_canary(
            reason="canary-attempt",
            canary_id="canary-2",
        )
        result = run_deploy(reason="canary-attempt", evidence=evidence)

        assert "Deploy blocked" in result
        assert "IMMUTABLE" in result
        # Live file was NOT created
        assert not (deploy_sandbox["live"] / "app/sanitize.py").exists()


class TestDeployLogRecordsBlockingReason:
    """Case 9: the deploy log captures the tier-violation reason + evidence."""

    def test_log_includes_tier_violation(self, deploy_sandbox, monkeypatch):
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        _write_applied_file(deploy_sandbox, "app/evolution.py")

        run_deploy(
            reason="log-test",
            evidence=DeployEvidence.direct("log-test", source="proposal"),
        )

        assert deploy_sandbox["log"].exists()
        log = json.loads(deploy_sandbox["log"].read_text())
        assert len(log) >= 1
        last = log[-1]
        assert last["status"] == "blocked"
        assert "tier protection" in last["error"]
        assert last["evidence"]["source"] == "proposal"
        assert last["evidence"]["has_canary_pass"] is False


class TestBlockedDeployLeavesAppliedCodeIntact:
    """Case 10: blocked deploys do NOT clean up applied_code/ — operator can inspect."""

    def test_blocked_deploy_preserves_applied_code(self, deploy_sandbox, monkeypatch):
        from app.auto_deployer import run_deploy, DeployEvidence

        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        applied_file = _write_applied_file(deploy_sandbox, "app/sanitize.py")

        run_deploy(
            reason="block-and-keep",
            evidence=DeployEvidence.from_canary("block-and-keep", canary_id="c-keep"),
        )

        # IMMUTABLE was refused; applied_code/ entry must still exist for review
        assert applied_file.exists(), "blocked deploy must preserve applied_code/ for operator review"


# ── Boundary helper coverage ────────────────────────────────────────────────


class TestValidateDeployBatch:
    """Unit test of the batch validator helper."""

    def test_mixed_batch_returns_per_file_violations(self):
        from app.auto_deployer import (
            _validate_deploy_batch,
            DeployEvidence,
        )

        files = [
            (Path("/tmp/x.py"), Path("app/sanitize.py")),    # IMMUTABLE
            (Path("/tmp/y.py"), Path("app/agents/x.py")),    # OPEN
            (Path("/tmp/z.py"), Path("app/evolution.py")),   # GATED
        ]
        # Direct evidence — no canary pass
        violations = _validate_deploy_batch(files, DeployEvidence.direct("test"))
        # IMMUTABLE + GATED should both be refused; OPEN passes through
        assert any("app/sanitize.py" in v for v in violations)
        assert any("app/evolution.py" in v for v in violations)
        assert not any("app/agents/x.py" in v for v in violations)
