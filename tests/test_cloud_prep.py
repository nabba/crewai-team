"""Tests for app.substrate.cloud_prep — WP D Phase 5a.

The cloud_prep module is the "click-button-and-things-happen" automation
that previously required the operator to type 3 gcloud commands. Tests
pin:
  * Each step succeeds/fails for the right reasons.
  * The full pipeline halts on first failure with a clear fail_reason.
  * Tokens never leak into the dict-serialized output (only key names).
  * Subprocess execute-gate is honored (test runs don't shell out).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.substrate import cloud_prep as cp


# ── Subprocess execute-gate ─────────────────────────────────────────


class TestShellGate:
    def test_neither_flag_returns_dry_placeholder(self, monkeypatch):
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        rc, out, err = cp._shell(["echo", "hi"], timeout=5.0)
        assert rc == 0
        assert "<dry:" in out
        assert "hi" in out

    def test_execute_true_runs_for_real(self, monkeypatch):
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        rc, out, _ = cp._shell(["echo", "real"], timeout=5.0, execute=True)
        assert rc == 0
        assert "real" in out
        assert "<dry:" not in out

    def test_env_var_runs_for_real(self, monkeypatch):
        monkeypatch.setenv("BOTARMY_MIGRATE_LIVE_EXECUTE", "1")
        rc, out, _ = cp._shell(["echo", "envgo"], timeout=5.0)
        assert "envgo" in out
        assert "<dry:" not in out

    def test_never_raises_on_empty_argv(self, monkeypatch):
        monkeypatch.setenv("BOTARMY_MIGRATE_LIVE_EXECUTE", "1")
        rc, out, err = cp._shell([], timeout=5.0)
        assert rc != 0


# ── list_authenticated_accounts ──────────────────────────────────────


class TestListAuthenticatedAccounts:
    def test_parses_user_and_service_accounts(self, monkeypatch):
        def _fake(argv, timeout=10.0):
            return (
                0,
                "andrus@raudsalu.com *\n"
                "ci-runner@my-project.iam.gserviceaccount.com\n"
                "third@example.org\n",
                "",
            )
        monkeypatch.setattr(cp, "_shell", lambda argv, **kw: _fake(argv))
        accts = cp.list_authenticated_accounts()
        # The * marks the active account
        assert len(accts) == 3
        active = next(a for a in accts if a["active"] == "yes")
        assert active["account"] == "andrus@raudsalu.com"
        assert active["type"] == "user"
        sa = next(a for a in accts if "gserviceaccount" in a["account"])
        assert sa["type"] == "service_account"

    def test_empty_when_no_accounts(self, monkeypatch):
        monkeypatch.setattr(cp, "_shell", lambda argv, **kw: (0, "", ""))
        assert cp.list_authenticated_accounts() == []

    def test_skips_dry_placeholder(self, monkeypatch):
        """When subprocess is in dry-mode, output has '<dry: ...>'.
        Parser must not return that as an account."""
        monkeypatch.setattr(
            cp, "_shell",
            lambda argv, **kw: (0, "<dry: gcloud auth list>", ""),
        )
        assert cp.list_authenticated_accounts() == []


# ── Step-level behavior ─────────────────────────────────────────────


class TestSetActiveAccount:
    def test_empty_account_fails_immediately(self):
        step = cp._step_set_active_account("")
        assert step.status == "fail"
        assert "empty" in step.detail

    def test_success_returns_ok(self, monkeypatch):
        monkeypatch.setattr(cp, "_shell", lambda argv, **kw: (0, "", ""))
        step = cp._step_set_active_account("andrus@raudsalu.com")
        assert step.status == "ok"
        assert "andrus@raudsalu.com" in step.detail

    def test_subprocess_failure_returns_fail(self, monkeypatch):
        monkeypatch.setattr(cp, "_shell", lambda argv, **kw: (1, "", "boom"))
        step = cp._step_set_active_account("andrus@raudsalu.com")
        assert step.status == "fail"
        assert "boom" in step.detail


class TestEnableApis:
    def test_success_returns_ok(self, monkeypatch):
        monkeypatch.setattr(cp, "_shell", lambda argv, **kw: (0, "", ""))
        step = cp._step_enable_apis("p", cp.REQUIRED_GCP_APIS)
        assert step.status == "ok"
        assert "9 APIs" in step.detail

    def test_permission_denied_surfaces_actionable_message(self, monkeypatch):
        monkeypatch.setattr(
            cp, "_shell",
            lambda argv, **kw: (1, "", "ERROR: PERMISSION_DENIED on api"),
        )
        step = cp._step_enable_apis("p", cp.REQUIRED_GCP_APIS)
        assert step.status == "fail"
        assert "serviceUsageAdmin" in step.detail
        # Must tell the operator what to do
        assert "Owner/Editor" in step.detail


class TestMintTerraformToken:
    def test_token_minted(self, monkeypatch):
        monkeypatch.setattr(
            cp, "_shell",
            lambda argv, **kw: (0, "ya29.actualtokenvaluehere", ""),
        )
        step, tok = cp._step_mint_terraform_token()
        assert step.status == "ok"
        assert tok == "ya29.actualtokenvaluehere"
        # Token length is in detail but token itself is NOT
        assert tok not in step.detail
        assert "length" in step.detail

    def test_dry_mode_returns_skipped(self, monkeypatch):
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        # _shell will return the dry placeholder for us
        step, tok = cp._step_mint_terraform_token()
        assert step.status == "skipped"
        assert "dry-shell" in step.detail


# ── Full pipeline ──────────────────────────────────────────────────


class TestPrepareGcpForMigrate:
    def _ok_shell(self, monkeypatch, token="ya29.testtoken"):
        def _fake(argv, *, timeout, execute=False):
            if argv[:4] == ["gcloud", "auth", "print-access-token"]:
                return (0, token, "")
            return (0, "", "")
        monkeypatch.setattr(cp, "_shell", _fake)

    def test_happy_path_returns_terraform_env_with_token(self, monkeypatch):
        self._ok_shell(monkeypatch)
        result = cp.prepare_gcp_for_migrate(
            active_account="andrus@raudsalu.com",
            project_id="botarmy-495107",
        )
        assert result.succeeded is True
        assert result.fail_reason == ""
        # 4 steps recorded: set_account, set_project, enable_apis, mint_token
        names = [s.name for s in result.steps]
        assert names == [
            "set_active_account", "set_project", "enable_apis", "mint_terraform_token",
        ]
        # Token populated into terraform_env
        assert result.terraform_env["GOOGLE_OAUTH_ACCESS_TOKEN"] == "ya29.testtoken"
        assert result.terraform_env["GOOGLE_PROJECT"] == "botarmy-495107"

    def test_halts_on_first_failure(self, monkeypatch):
        calls: list[list[str]] = []

        def _fake(argv, *, timeout, execute=False):
            calls.append(list(argv))
            # Fail at set_active_account
            if argv[:4] == ["gcloud", "config", "set", "account"]:
                return (1, "", "could not set account")
            return (0, "", "")

        monkeypatch.setattr(cp, "_shell", _fake)
        result = cp.prepare_gcp_for_migrate(
            active_account="andrus@raudsalu.com",
            project_id="p",
        )
        assert result.succeeded is False
        # Only the first step was recorded
        assert len(result.steps) == 1
        assert result.steps[0].name == "set_active_account"
        assert result.steps[0].status == "fail"
        # No follow-up subprocess invocations after the failure
        assert all(c[:4] != ["gcloud", "services", "enable"] for c in calls)
        assert "set_active_account" in result.fail_reason

    def test_failed_at_apis_step(self, monkeypatch):
        def _fake(argv, *, timeout, execute=False):
            if argv[:3] == ["gcloud", "services", "enable"]:
                return (1, "", "PERMISSION_DENIED")
            return (0, "", "")
        monkeypatch.setattr(cp, "_shell", _fake)
        result = cp.prepare_gcp_for_migrate(
            active_account="andrus@raudsalu.com", project_id="p",
        )
        assert result.succeeded is False
        # First two steps ran, third failed, fourth never ran
        assert len(result.steps) == 3
        assert result.steps[-1].name == "enable_apis"
        assert result.steps[-1].status == "fail"
        # No terraform_env populated
        assert result.terraform_env == {}


# ── Security: token never leaks into serialization ──────────────────


class TestTokenNotInSerialization:
    """The terraform OAuth token MUST stay in-memory only. The dict
    serialization (used for REST responses + workspace persistence)
    must only expose the env-var KEY NAMES, not the values.
    """

    def test_to_dict_redacts_token_value(self, monkeypatch):
        def _fake(argv, *, timeout, execute=False):
            if argv[:4] == ["gcloud", "auth", "print-access-token"]:
                return (0, "ya29.SUPER_SENSITIVE_TOKEN_VALUE", "")
            return (0, "", "")
        monkeypatch.setattr(cp, "_shell", _fake)

        result = cp.prepare_gcp_for_migrate(
            active_account="andrus@raudsalu.com", project_id="p",
        )
        assert result.succeeded
        # In-memory env has the token
        assert "ya29.SUPER_SENSITIVE_TOKEN_VALUE" in result.terraform_env.values()
        # But to_dict serialization MUST redact the values
        d = result.to_dict()
        serialized = str(d)
        assert "SUPER_SENSITIVE_TOKEN_VALUE" not in serialized, (
            "to_dict must NOT leak the OAuth token — only env-var KEY names"
        )
        # The key names should still appear so the operator can see
        # which env vars terraform would receive
        assert "GOOGLE_OAUTH_ACCESS_TOKEN" in d["terraform_env_keys"]
