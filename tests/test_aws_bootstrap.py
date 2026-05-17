"""Contract tests for scripts/install/aws_bootstrap.sh + the
/api/cp/migrate/bootstrap-aws-account REST endpoint."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "install" / "aws_bootstrap.sh"


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, env=e, timeout=30.0,
    )


class TestScript:
    def test_script_is_executable(self):
        assert SCRIPT.is_file()
        assert os.access(SCRIPT, os.X_OK), "script must be executable"

    def test_help_prints_usage(self):
        r = _run("--help")
        assert r.returncode == 0
        assert "Stage 0a of the AWS migrate wizard" in r.stdout

    def test_refuses_without_email(self):
        r = _run("--account-name", "test")
        assert r.returncode == 2
        assert "--email" in r.stderr

    def test_refuses_malformed_email(self):
        r = _run("--email", "not-an-email")
        assert r.returncode == 3
        assert "--email format" in r.stderr

    def test_refuses_invalid_role_name(self):
        r = _run(
            "--email", "ops@example.com",
            "--role-name", "has spaces in it",
        )
        assert r.returncode == 3
        assert "--role-name" in r.stderr

    def test_refuses_invalid_ou_id(self):
        r = _run(
            "--email", "ops@example.com",
            "--org-unit-id", "not-an-ou",
        )
        assert r.returncode == 3
        assert "--org-unit-id" in r.stderr

    def test_refuses_without_typed_phrase(self):
        r = _run("--email", "ops@example.com")
        # rc=5 = typed-phrase refusal; rc=6 = no aws CLI on test host
        assert r.returncode in (5, 6)

    def test_refuses_wrong_typed_phrase(self):
        r = _run(
            "--email", "ops@example.com",
            "--confirm", "WRONG",
        )
        assert r.returncode in (5, 6)


class TestRuntimeSettings:
    def test_aws_bootstrap_default_off(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.delenv("BOTARMY_AWS_BOOTSTRAP_ENABLED", raising=False)
        assert rs.get_aws_bootstrap_enabled() is False

    def test_setter_emits_ledger_event(self, monkeypatch, tmp_path):
        import json
        from app import runtime_settings as rs
        from app import paths as _paths
        from app.identity import continuity_ledger as cl

        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        monkeypatch.setattr(cl, "_path_override", tmp_path / "identity" / "continuity_ledger.jsonl")
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

        rs.set_aws_bootstrap_enabled(True)

        ledger = tmp_path / "identity" / "continuity_ledger.jsonl"
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        cm = [e for e in events if e["kind"] == "cloud_migration"]
        assert len(cm) == 1
        assert cm[0]["detail"]["phase"] == "aws_bootstrap_policy_changed"
        assert cm[0]["detail"]["new"] is True


class TestEndpoint:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.control_plane.migrate_api import router
        from app.control_plane import auth_dep
        monkeypatch.setattr(auth_dep, "require_gateway_auth", lambda: None)
        app = FastAPI()
        app.include_router(router)
        from app import runtime_settings as rs
        from app import paths as _paths
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        return TestClient(app)

    def test_refuses_when_aws_bootstrap_disabled(self, client):
        from app import runtime_settings as rs
        assert rs.get_aws_bootstrap_enabled() is False
        r = client.post(
            "/api/cp/migrate/bootstrap-aws-account",
            json={
                "email": "ops@example.com",
                "confirm_phrase": "CREATE AWS ACCOUNT",
            },
        )
        assert r.status_code == 403
        assert "aws_bootstrap_enabled is OFF" in r.json()["detail"]

    def test_refuses_wrong_phrase(self, client):
        from app import runtime_settings as rs
        rs.set_aws_bootstrap_enabled(True)
        r = client.post(
            "/api/cp/migrate/bootstrap-aws-account",
            json={
                "email": "ops@example.com",
                "confirm_phrase": "WRONG",
            },
        )
        assert r.status_code == 400
        assert "CREATE AWS ACCOUNT" in r.json()["detail"]

    def test_dry_run_invokes_script(self, client, monkeypatch):
        from app import runtime_settings as rs
        rs.set_aws_bootstrap_enabled(True)

        called = {}

        def fake_run(argv, **kwargs):
            called["argv"] = argv
            return subprocess.CompletedProcess(
                argv, 0, "Dry-run output\n123456789012\n", "",
            )

        monkeypatch.setattr("subprocess.run", fake_run)

        r = client.post(
            "/api/cp/migrate/bootstrap-aws-account",
            json={
                "email": "ops@example.com",
                "account_name": "botarmy-prod",
                "org_unit_id": "ou-abcd-12345678",
                "confirm_phrase": "CREATE AWS ACCOUNT",
                "dry_run": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["dry_run"] is True
        assert body["new_account_id"] == "123456789012"
        assert "--dry-run" in called["argv"]
        # Confirm phrase forwarded verbatim
        confirm_idx = called["argv"].index("--confirm")
        assert called["argv"][confirm_idx + 1] == "CREATE AWS ACCOUNT"
