"""Tests for the hardening + bootstrap REST endpoints in migrate_api."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Build a FastAPI app with the migrate_router + auth-bypass for tests."""
    from app.control_plane.migrate_api import router
    from app.control_plane import auth_dep
    # Bypass Bearer auth in tests
    monkeypatch.setattr(auth_dep, "require_gateway_auth", lambda: None)
    # Re-instantiate the FastAPI app so the patched dependency is honored
    app = FastAPI()
    app.include_router(router)
    # Re-route runtime_settings to tmp
    from app import runtime_settings as rs
    monkeypatch.setattr(rs, "_cache", None)
    monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    return TestClient(app)


# ── /hardening-preview ────────────────────────────────────────


class TestHardeningPreviewEndpoint:
    def test_default_profile_returns_payload(self, client):
        from app.substrate import cloud_hardening as ch
        with patch.object(ch, "detect_tailnet_cidr", return_value="100.64.0.0/10"), \
             patch.object(ch, "detect_laptop_public_ip", return_value="1.2.3.4"), \
             patch.object(ch, "detect_org_id", return_value="987654321012"):
            r = client.get("/api/cp/migrate/hardening-preview")
            assert r.status_code == 200
            data = r.json()
            assert data["profile"] == "strict"
            assert data["tailnet_reachable"] is True
            assert data["org_id"] == "987654321012"
            assert len(data["recommended_cidrs"]) == 2

    def test_query_param_overrides_profile(self, client):
        from app.substrate import cloud_hardening as ch
        with patch.object(ch, "detect_tailnet_cidr", return_value=None), \
             patch.object(ch, "detect_laptop_public_ip", return_value=None), \
             patch.object(ch, "detect_org_id", return_value=None):
            r = client.get("/api/cp/migrate/hardening-preview?profile=basic&binauthz_mode=AUDIT")
            assert r.status_code == 200
            data = r.json()
            assert data["profile"] == "basic"
            # basic profile suppresses the "no Tailnet" + "no allowlist" notes
            # — they're strict-only concerns
            assert all("master_authorized_networks" not in n for n in data["notes"])

    def test_strict_enforce_emits_warning(self, client):
        from app.substrate import cloud_hardening as ch
        with patch.object(ch, "detect_tailnet_cidr", return_value="100.64.0.0/10"), \
             patch.object(ch, "detect_laptop_public_ip", return_value="1.2.3.4"), \
             patch.object(ch, "detect_org_id", return_value="987654321012"):
            r = client.get("/api/cp/migrate/hardening-preview?profile=strict&binauthz_mode=ENFORCE")
            assert r.status_code == 200
            assert any("ENFORCE will reject" in n for n in r.json()["notes"])

    def test_refuses_unknown_profile(self, client):
        r = client.get("/api/cp/migrate/hardening-preview?profile=garbage")
        # FastAPI's query validator rejects unmatched pattern with 422
        assert r.status_code == 422

    def test_refuses_unknown_binauthz_mode(self, client):
        r = client.get("/api/cp/migrate/hardening-preview?binauthz_mode=relaxed")
        assert r.status_code == 422


# ── /bootstrap-project ────────────────────────────────────────


class TestBootstrapEndpoint:
    def test_refuses_when_gcp_bootstrap_disabled(self, client, monkeypatch):
        from app import runtime_settings as rs
        # Default: gcp_bootstrap_enabled is False
        assert rs.get_gcp_bootstrap_enabled() is False
        r = client.post(
            "/api/cp/migrate/bootstrap-project",
            json={
                "project_id": "botarmy-test-abc",
                "billing_account": "01ABCD-EFGH12-IJ34KL",
                "confirm_phrase": "CREATE GCP PROJECT",
            },
        )
        assert r.status_code == 403
        assert "gcp_bootstrap_enabled is OFF" in r.json()["detail"]

    def test_refuses_wrong_typed_phrase(self, client, monkeypatch):
        from app import runtime_settings as rs
        rs.set_gcp_bootstrap_enabled(True)
        r = client.post(
            "/api/cp/migrate/bootstrap-project",
            json={
                "project_id": "botarmy-test-abc",
                "billing_account": "01ABCD-EFGH12-IJ34KL",
                "confirm_phrase": "WRONG",
            },
        )
        assert r.status_code == 400
        assert "CREATE GCP PROJECT" in r.json()["detail"]

    def test_dry_run_with_proper_phrase_invokes_script(self, client, monkeypatch):
        from app import runtime_settings as rs
        rs.set_gcp_bootstrap_enabled(True)

        # Stub subprocess.run to avoid calling real gcloud
        import app.control_plane.migrate_api as ma
        import subprocess

        called = {}

        def fake_run(argv, **kwargs):
            called["argv"] = argv
            return subprocess.CompletedProcess(
                argv, 0, "Stage 0a complete (dry-run)\n", "",
            )

        # Patch subprocess.run AT the call site (inside the endpoint),
        # not the module-global subprocess — the endpoint imports it locally.
        monkeypatch.setattr("subprocess.run", fake_run)

        r = client.post(
            "/api/cp/migrate/bootstrap-project",
            json={
                "project_id": "botarmy-test-abc",
                "billing_account": "01ABCD-EFGH12-IJ34KL",
                "confirm_phrase": "CREATE GCP PROJECT",
                "dry_run": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["rc"] == 0
        assert body["dry_run"] is True
        assert "--dry-run" in called["argv"]
        assert "--confirm" in called["argv"]
        # Typed phrase must be passed verbatim
        confirm_idx = called["argv"].index("--confirm")
        assert called["argv"][confirm_idx + 1] == "CREATE GCP PROJECT"
