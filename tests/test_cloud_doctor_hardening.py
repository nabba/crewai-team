"""Tests for the hardening probes added to cloud_doctor (project_exists,
tailnet_reachable, binauthz_signing_ready)."""
from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.substrate import cloud_doctor as cd


class TestProjectExistsProbe:
    def test_active_project_returns_ok(self, monkeypatch):
        monkeypatch.setenv("CLOUDSDK_CORE_PROJECT", "botarmy-test")
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(
                ["gcloud"], 0, "botarmy-test\tACTIVE", "",
            )
            p = cd._probe_gcp_project_exists()
            assert p.status == "OK"
            assert "ACTIVE" in p.detail

    def test_missing_project_returns_missing(self, monkeypatch):
        monkeypatch.setenv("CLOUDSDK_CORE_PROJECT", "no-such-project")
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(["gcloud"], 1, "", "not found")
            p = cd._probe_gcp_project_exists()
            assert p.status == "MISSING"

    def test_missing_project_optional_when_bootstrap_on(self, monkeypatch):
        monkeypatch.setenv("CLOUDSDK_CORE_PROJECT", "no-such-project")
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "get_gcp_bootstrap_enabled", lambda: True)
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(["gcloud"], 1, "", "not found")
            p = cd._probe_gcp_project_exists()
            assert p.status == "MISSING"
            assert p.required is False  # not blocking when bootstrap is on
            assert "Stage 0a" in p.detail


class TestTailnetProbe:
    def test_reachable_yields_ok(self, monkeypatch):
        from app.substrate import cloud_hardening as ch
        monkeypatch.setattr(ch, "tailnet_reachable", lambda: True)
        p = cd._probe_tailnet_reachable()
        assert p.status == "OK"
        assert "100.64.0.0/10" in p.detail

    def test_unreachable_yields_missing(self, monkeypatch):
        from app.substrate import cloud_hardening as ch
        monkeypatch.setattr(ch, "tailnet_reachable", lambda: False)
        p = cd._probe_tailnet_reachable()
        assert p.status == "MISSING"
        assert p.required is False  # surfaced only as a heads-up


class TestBinauthzSigningReadyProbe:
    def test_attestor_env_var_present(self, monkeypatch):
        monkeypatch.setenv("BINAUTHZ_ATTESTOR", "projects/p/attestors/my-attestor")
        # Make sure we don't accidentally pick up a real cosign.pub
        monkeypatch.setenv("INSTALL_ROOT", "/nonexistent")
        p = cd._probe_gcp_binauthz_signing_ready()
        assert p.status == "OK"
        assert "my-attestor" in p.detail

    def test_no_signing_pipeline_yields_missing(self, monkeypatch):
        monkeypatch.delenv("BINAUTHZ_ATTESTOR", raising=False)
        monkeypatch.setenv("INSTALL_ROOT", "/definitely/does/not/exist")
        p = cd._probe_gcp_binauthz_signing_ready()
        assert p.status == "MISSING"
        assert p.required is False
        assert "Keep binauthz_mode=AUDIT" in p.detail

    def test_cosign_pub_file_present_yields_ok(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BINAUTHZ_ATTESTOR", raising=False)
        bz_dir = tmp_path / "deploy" / "k8s" / "binauthz"
        bz_dir.mkdir(parents=True)
        (bz_dir / "cosign.pub").write_text("-----BEGIN PUBLIC KEY-----")
        monkeypatch.setenv("INSTALL_ROOT", str(tmp_path))
        p = cd._probe_gcp_binauthz_signing_ready()
        assert p.status == "OK"


class TestCheckReadinessIntegration:
    def test_hardening_probes_wired_into_check_readiness(self, monkeypatch):
        """The three new probes must show up in the rollup for target=gcp."""
        # Mock subprocess to return failure on gcloud commands — the rollup
        # will be MISSING but we just want the probe names to appear.
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(["gcloud"], 1, "", "")
            r = cd.check_readiness("gcp")
            names = [p.name for p in r.probes]
            assert "gcloud project exists" in names
            assert "tailnet reachable" in names
            assert "binauthz signing ready" in names
