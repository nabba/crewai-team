"""End-to-end isolation test for the kept-mutation → staged → deployed loop.

Closes audit-2026-05-18 recommendation #6: prove that a kept AVO code
mutation actually reaches live deploy. Before the staging contract
existed, evolution._trigger_code_auto_deploy called schedule_deploy
without first writing the mutation into APPLIED_CODE_DIR, so the
deploy was a silent no-op ("No files to deploy.").

Scope: this is the contract test — does the evolution → applied_code →
run_deploy chain end-to-end carry the file through. Real eval / metrics
/ canary are out of scope (mocked).
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


# ── Sandbox + mocks ─────────────────────────────────────────────────────────


@pytest.fixture
def deploy_chain_sandbox(tmp_path, monkeypatch):
    """Redirect path constants in BOTH deploy_staging and auto_deployer to a
    tmp sandbox so the staging → deploy chain runs without touching /app.

    Suppresses hot-reload + post-deploy monitor (they walk sys.modules /
    start threads; not useful in tests).
    """
    from app import auto_deployer, deploy_staging

    applied = tmp_path / "applied_code"
    live = tmp_path / "live"
    backups = tmp_path / "backups"
    log = tmp_path / "deploy_log.json"

    applied.mkdir(parents=True)
    live.mkdir(parents=True)
    backups.mkdir(parents=True)
    (live / "app").mkdir(parents=True)

    # Both modules read APPLIED_CODE_DIR from module scope — redirect both.
    monkeypatch.setattr(deploy_staging, "APPLIED_CODE_DIR", applied)
    monkeypatch.setattr(auto_deployer, "APPLIED_CODE_DIR", applied)
    monkeypatch.setattr(auto_deployer, "LIVE_CODE_DIR", live)
    monkeypatch.setattr(auto_deployer, "BACKUP_DIR", backups)
    monkeypatch.setattr(auto_deployer, "DEPLOY_LOG", log)

    # Silence side effects.
    monkeypatch.setattr(
        "app.signal_client.send_message", lambda *a, **kw: None, raising=False
    )
    monkeypatch.setattr(
        auto_deployer, "_hot_reload_modules_safe",
        lambda deployed, backup: ([], []),
    )
    monkeypatch.setattr(
        auto_deployer, "_post_deploy_monitor",
        lambda deployed, backup, reason: None,
    )

    return {
        "applied": applied,
        "live": live,
        "backups": backups,
        "log": log,
    }


@pytest.fixture
def evolution_mocks(monkeypatch):
    """Mock the human_gate / self_model dependencies of
    evolution._trigger_code_auto_deploy and record what schedule_deploy /
    request_approval are called with."""
    schedule_calls: list[dict] = []
    approval_calls: list[dict] = []

    def fake_schedule_deploy(reason, evidence=None, **kw):
        schedule_calls.append({"reason": reason, "evidence": evidence, "extra": kw})

    def fake_request_approval(**kw):
        approval_calls.append(kw)
        return f"approval_{kw.get('experiment_id', 'x')}_t"

    # auto_deployer.schedule_deploy is imported fresh inside the function
    # under test, so patching the module attribute is enough.
    from app import auto_deployer
    monkeypatch.setattr(auto_deployer, "schedule_deploy", fake_schedule_deploy)

    # human_gate is also imported inside the function. Patch the symbols
    # the function will pick up.
    from app import human_gate
    monkeypatch.setattr(human_gate, "request_approval", fake_request_approval)

    # self_model centrality probes — keep them deterministic and low-risk
    from app import self_model
    monkeypatch.setattr(self_model, "is_hot_path", lambda p: False, raising=False)
    monkeypatch.setattr(
        self_model, "get_centrality_score", lambda p: 0.1, raising=False
    )

    return {
        "schedule_calls": schedule_calls,
        "approval_calls": approval_calls,
    }


# ── Test data factories ─────────────────────────────────────────────────────


def _make_kept_result(experiment_id: str = "exp_test_abc", delta: float = 0.10):
    """Build an ExperimentResult shaped like a real kept code mutation."""
    from app.experiment_runner import ExperimentResult
    return ExperimentResult(
        experiment_id=experiment_id,
        hypothesis="Test mutation: rewrite a comment.",
        change_type="code",
        metric_before=0.7,
        metric_after=0.7 + delta,
        delta=delta,
        status="keep",
        files_changed=["app/agents/researcher.py"],
        detail=f"Improvement: {delta:+.4f}",
    )


def _make_mutation_spec(
    experiment_id: str = "exp_test_abc",
    files: dict | None = None,
):
    """Build a MutationSpec matching what experiment_runner would produce."""
    from app.experiment_runner import MutationSpec
    if files is None:
        files = {"app/agents/researcher.py": "# updated\nx = 2\n"}
    return MutationSpec(
        experiment_id=experiment_id,
        hypothesis="Test mutation.",
        change_type="code",
        files=files,
    )


# ── HIGH confidence path: stage → schedule_deploy → run_deploy → live ──────


class TestKeptHighConfidence:
    """The audit's core scenario: AVO kept a code mutation, eval confirmed
    delta is large, no high-centrality files. Must stage + schedule_deploy.
    """

    def test_high_confidence_stages_then_schedules(
        self, deploy_chain_sandbox, evolution_mocks
    ):
        # Force HIGH confidence by patching classify_confidence
        from app import human_gate
        from unittest.mock import patch

        with patch.object(
            human_gate, "classify_confidence",
            return_value=(human_gate.ConfidenceTier.HIGH, "test high")
        ):
            from app.evolution import _trigger_code_auto_deploy
            _trigger_code_auto_deploy(
                _make_kept_result(),
                _make_mutation_spec(),
            )

        # Staging step must have written the file
        staged = deploy_chain_sandbox["applied"] / "app/agents/researcher.py"
        assert staged.exists(), "stage_for_deploy must run before schedule_deploy"
        assert staged.read_text() == "# updated\nx = 2\n"

        # schedule_deploy must have been called with the kept-experiment reason
        assert len(evolution_mocks["schedule_calls"]) == 1
        call = evolution_mocks["schedule_calls"][0]
        assert "exp_test_abc" in call["reason"]

        # request_approval NOT called (HIGH path bypasses human gate)
        assert evolution_mocks["approval_calls"] == []

    def test_staged_file_actually_deploys_to_live(
        self, deploy_chain_sandbox, evolution_mocks
    ):
        """The complete chain: stage → run_deploy → live file present.
        Bypasses schedule_deploy (which schedules a 5s delayed thread)
        and calls run_deploy directly with the kept-experiment evidence
        the production path would construct."""
        from app import human_gate
        from unittest.mock import patch

        with patch.object(
            human_gate, "classify_confidence",
            return_value=(human_gate.ConfidenceTier.HIGH, "test high")
        ):
            from app.evolution import _trigger_code_auto_deploy
            _trigger_code_auto_deploy(
                _make_kept_result(),
                _make_mutation_spec(),
            )

        # Now invoke run_deploy directly with canary-fallback evidence
        # (mimicking the canary-disabled production path)
        from app.auto_deployer import run_deploy, DeployEvidence
        result = run_deploy(
            "evolution-keep-exp_test_abc",
            evidence=DeployEvidence.direct(
                "evolution-keep-exp_test_abc",
                source="canary_fallback",
            ),
        )

        # Researcher.py is OPEN tier → direct evidence is sufficient
        assert "Deployed 1 files" in result, f"expected success, got: {result}"
        live_file = deploy_chain_sandbox["live"] / "app/agents/researcher.py"
        assert live_file.exists()
        assert live_file.read_text() == "# updated\nx = 2\n"


# ── BORDERLINE path: stage → request_approval (no schedule_deploy yet) ─────


class TestKeptBorderlineConfidence:
    """Borderline mutations stage BEFORE going to the human gate. When the
    owner later approves, schedule_deploy fires and finds the file already
    staged."""

    def test_borderline_stages_then_requests_approval(
        self, deploy_chain_sandbox, evolution_mocks
    ):
        from app import human_gate
        from unittest.mock import patch

        with patch.object(
            human_gate, "classify_confidence",
            return_value=(human_gate.ConfidenceTier.BORDERLINE, "small delta")
        ):
            from app.evolution import _trigger_code_auto_deploy
            _trigger_code_auto_deploy(
                _make_kept_result(delta=0.02),
                _make_mutation_spec(),
            )

        # File staged even on borderline path
        staged = deploy_chain_sandbox["applied"] / "app/agents/researcher.py"
        assert staged.exists(), (
            "borderline path must also stage — owner approval triggers "
            "schedule_deploy which scans applied_code/"
        )

        # request_approval was called, schedule_deploy was NOT
        assert len(evolution_mocks["approval_calls"]) == 1
        assert evolution_mocks["schedule_calls"] == []


# ── Failure: staging error must NOT proceed to schedule_deploy ──────────────


class TestStagingFailureSafeguard:
    """If stage_for_deploy raises, _trigger_code_auto_deploy must NOT call
    schedule_deploy — otherwise we'd be back to the silent-no-op behavior
    we're trying to close."""

    def test_staging_failure_skips_deploy(
        self, deploy_chain_sandbox, evolution_mocks, monkeypatch
    ):
        # Patch stage_for_deploy to raise
        from app import deploy_staging

        def _boom(*a, **kw):
            raise ValueError("staging boom (test)")

        monkeypatch.setattr(deploy_staging, "stage_for_deploy", _boom)

        from app.evolution import _trigger_code_auto_deploy
        _trigger_code_auto_deploy(
            _make_kept_result(),
            _make_mutation_spec(),
        )

        # schedule_deploy NOT called — function short-circuited on staging failure
        assert evolution_mocks["schedule_calls"] == []
        # applied_code/ left clean
        assert not (
            deploy_chain_sandbox["applied"] / "app/agents/researcher.py"
        ).exists()


# ── Pre-flight IMMUTABLE skip still fires before staging ────────────────────


class TestImmutablePreFlightSkip:
    """The pre-existing IMMUTABLE pre-flight check at the top of
    _trigger_code_auto_deploy must still short-circuit before staging —
    we don't want IMMUTABLE files even attempting to stage."""

    def test_immutable_file_skips_everything(
        self, deploy_chain_sandbox, evolution_mocks
    ):
        from app.evolution import _trigger_code_auto_deploy
        _trigger_code_auto_deploy(
            _make_kept_result(),
            _make_mutation_spec(
                files={"app/sanitize.py": "# evil rewrite\n"},  # IMMUTABLE
            ),
        )

        # Nothing staged, nothing scheduled
        assert not (deploy_chain_sandbox["applied"] / "app/sanitize.py").exists()
        assert evolution_mocks["schedule_calls"] == []
        assert evolution_mocks["approval_calls"] == []
