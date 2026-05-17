"""Pinning tests for the three pieces shipped 2026-05-17:
  1. AWS member-account bootstrap
  2. Cosign attestor pipeline (binauthz attestor wiring)
  3. VPC Service Controls perimeter
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = Path(__file__).resolve().parents[1]
GCP_DIR = REPO / "deploy" / "terraform" / "gcp"


# ── 1. AWS bootstrap script + endpoint covered in test_aws_bootstrap.py


# ── 2. Cosign attestor terraform wiring ──────────────────────


class TestCosignAttestorTerraform:
    def test_attestor_name_variable_exists(self):
        text = (GCP_DIR / "variables.tf").read_text()
        assert 'variable "binauthz_attestor_name"' in text

    def test_attestor_referenced_in_policy(self):
        text = (GCP_DIR / "hardening.tf").read_text()
        assert "require_attestations_by" in text
        assert "var.binauthz_attestor_name" in text

    def test_enforce_without_attestor_falls_back_to_allow(self):
        """ENFORCE mode without an attestor wired must default to
        ALWAYS_ALLOW so deploys don't brick. This is the load-bearing
        assertion in hardening.tf."""
        text = (GCP_DIR / "hardening.tf").read_text()
        # The conditional expression must check BOTH conditions
        assert 'var.binauthz_mode == "ENFORCE" && var.binauthz_attestor_name != ""' in text

    def test_attestor_summary_field(self):
        text = (GCP_DIR / "hardening.tf").read_text()
        assert "binauthz_attestor_wired" in text


class TestCosignSetupScript:
    SCRIPT = REPO / "scripts" / "install" / "cosign_setup.sh"

    def test_script_exists_and_is_executable(self):
        assert self.SCRIPT.is_file()
        assert os.access(self.SCRIPT, os.X_OK)

    def test_help_prints_overview(self):
        r = subprocess.run(
            ["bash", str(self.SCRIPT), "--help"],
            capture_output=True, text=True, timeout=10.0,
        )
        assert r.returncode == 0
        assert "Binary Authorization" in r.stdout
        assert "cosign keypair" in r.stdout

    def test_refuses_without_project_id(self):
        r = subprocess.run(
            ["bash", str(self.SCRIPT)],
            capture_output=True, text=True, timeout=10.0,
        )
        assert r.returncode == 2
        assert "--project-id" in r.stderr

    def test_refuses_without_typed_phrase(self):
        r = subprocess.run(
            ["bash", str(self.SCRIPT), "--project-id", "botarmy-test-abc"],
            capture_output=True, text=True, timeout=10.0,
        )
        # rc=5 = typed phrase; rc=6 = cosign/gcloud missing in test env
        assert r.returncode in (5, 6)


class TestRuntimeSettings:
    def test_binauthz_attestor_name_default_empty(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.delenv("BOTARMY_BINAUTHZ_ATTESTOR_NAME", raising=False)
        assert rs.get_binauthz_attestor_name() == ""

    def test_setter_strips_whitespace(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        rs.set_binauthz_attestor_name("  my-attestor  ")
        assert rs.get_binauthz_attestor_name() == "my-attestor"


# ── 3. VPC-SC terraform wiring ───────────────────────────────


class TestVpcScTerraform:
    def test_file_exists(self):
        assert (GCP_DIR / "vpc_sc.tf").is_file()

    def test_access_level_resource(self):
        text = (GCP_DIR / "vpc_sc.tf").read_text()
        assert "google_access_context_manager_access_level" in text

    def test_perimeter_resource(self):
        text = (GCP_DIR / "vpc_sc.tf").read_text()
        assert "google_access_context_manager_service_perimeter" in text

    def test_dry_run_default(self):
        """The dry-run variable defaults to true so first apply is observational."""
        text = (GCP_DIR / "variables.tf").read_text()
        assert 'variable "vpc_sc_dry_run"' in text
        # Find the block and confirm default = true
        idx = text.find('variable "vpc_sc_dry_run"')
        block = text[idx : idx + 400]
        assert "default     = true" in block

    def test_vpc_sc_enabled_default_off(self):
        text = (GCP_DIR / "variables.tf").read_text()
        idx = text.find('variable "vpc_sc_enabled"')
        block = text[idx : idx + 400]
        assert "default     = false" in block

    def test_only_active_with_all_preconditions(self):
        """vpc_sc_active is the AND of strict + enabled + org_id + access_policy_id."""
        text = (GCP_DIR / "vpc_sc.tf").read_text()
        assert "local.hardening_strict" in text
        assert "var.vpc_sc_enabled" in text
        assert 'var.org_id != ""' in text
        assert 'var.access_policy_id != ""' in text

    def test_use_explicit_dry_run_spec(self):
        """Toggling enforce vs dry-run uses the same configuration block — terraform native."""
        text = (GCP_DIR / "vpc_sc.tf").read_text()
        assert "use_explicit_dry_run_spec = var.vpc_sc_dry_run" in text

    def test_default_restricted_services(self):
        text = (GCP_DIR / "variables.tf").read_text()
        for svc in (
            "storage.googleapis.com",
            "secretmanager.googleapis.com",
            "sqladmin.googleapis.com",
            "artifactregistry.googleapis.com",
            "cloudkms.googleapis.com",
        ):
            assert svc in text, f"missing restricted service in defaults: {svc}"

    def test_hardening_summary_reports_vpc_sc(self):
        text = (GCP_DIR / "hardening.tf").read_text()
        assert "vpc_sc_enabled" in text
        assert "vpc_sc_dry_run" in text


class TestAccessPolicyDetection:
    def test_returns_id_when_policy_exists(self):
        from app.substrate import cloud_hardening as ch
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(
                ["gcloud"], 0,
                json.dumps([{"name": "accessPolicies/987654321012", "title": "test"}]),
                "",
            )
            assert ch.detect_access_policy_id("123456789012") == "987654321012"

    def test_returns_none_when_no_policies(self):
        from app.substrate import cloud_hardening as ch
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(["gcloud"], 0, "[]", "")
            assert ch.detect_access_policy_id("123456789012") is None

    def test_returns_none_without_org_id(self):
        from app.substrate import cloud_hardening as ch
        assert ch.detect_access_policy_id("") is None
        assert ch.detect_access_policy_id(None) is None  # type: ignore[arg-type]


class TestVpcScRuntimeSettings:
    def test_defaults(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.delenv("BOTARMY_VPC_SC_ENABLED", raising=False)
        monkeypatch.delenv("BOTARMY_VPC_SC_DRY_RUN", raising=False)
        assert rs.get_vpc_sc_enabled() is False
        assert rs.get_vpc_sc_dry_run() is True  # safe default

    def test_setters_emit_ledger_events(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        from app import paths as _paths
        from app.identity import continuity_ledger as cl

        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        monkeypatch.setattr(cl, "_path_override", tmp_path / "identity" / "continuity_ledger.jsonl")
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

        rs.set_vpc_sc_enabled(True)
        rs.set_vpc_sc_dry_run(False)

        ledger = tmp_path / "identity" / "continuity_ledger.jsonl"
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        kinds = [(e["kind"], e["detail"].get("phase")) for e in events]
        assert ("cloud_migration", "vpc_sc_policy_changed") in kinds
        assert ("cloud_migration", "vpc_sc_dry_run_changed") in kinds


class TestTfvarsRenderingWithVpcSc:
    def test_attestor_name_appears_in_tfvars_when_set(self, monkeypatch, tmp_path):
        from app.substrate.migration import _write_per_run_tfvars
        from app import paths as _paths
        from app import runtime_settings as rs
        from app.substrate import cloud_hardening as ch

        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.setattr(ch, "detect_tailnet_cidr", lambda: "100.64.0.0/10")
        monkeypatch.setattr(ch, "detect_laptop_public_ip", lambda: "1.2.3.4")
        monkeypatch.setattr(ch, "detect_org_id", lambda: "987654321012")
        monkeypatch.setattr(ch, "detect_access_policy_id", lambda org: None)
        rs.set_binauthz_attestor_name("botarmy-attestor")

        path = _write_per_run_tfvars("gcp", "botarmy-test", "europe-north1", "cheapest", "run-1")
        body = path.read_text()
        assert 'binauthz_attestor_name = "botarmy-attestor"' in body

    def test_vpc_sc_block_appears_when_policy_detected(self, monkeypatch, tmp_path):
        from app.substrate.migration import _write_per_run_tfvars
        from app import paths as _paths
        from app import runtime_settings as rs
        from app.substrate import cloud_hardening as ch

        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.setattr(ch, "detect_tailnet_cidr", lambda: "100.64.0.0/10")
        monkeypatch.setattr(ch, "detect_laptop_public_ip", lambda: "1.2.3.4")
        monkeypatch.setattr(ch, "detect_org_id", lambda: "987654321012")
        monkeypatch.setattr(ch, "detect_access_policy_id", lambda org: "1234567890")
        rs.set_vpc_sc_enabled(True)

        path = _write_per_run_tfvars("gcp", "botarmy-test", "europe-north1", "cheapest", "run-1")
        body = path.read_text()
        assert "vpc_sc_enabled    = true" in body
        assert 'access_policy_id  = "1234567890"' in body

    def test_vpc_sc_block_omitted_when_no_access_policy(self, monkeypatch, tmp_path):
        from app.substrate.migration import _write_per_run_tfvars
        from app import paths as _paths
        from app import runtime_settings as rs
        from app.substrate import cloud_hardening as ch

        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.setattr(ch, "detect_tailnet_cidr", lambda: None)
        monkeypatch.setattr(ch, "detect_laptop_public_ip", lambda: None)
        monkeypatch.setattr(ch, "detect_org_id", lambda: "987654321012")
        monkeypatch.setattr(ch, "detect_access_policy_id", lambda org: None)
        rs.set_vpc_sc_enabled(True)

        path = _write_per_run_tfvars("gcp", "botarmy-test", "europe-north1", "cheapest", "run-1")
        body = path.read_text()
        # Without an access policy we can't emit vpc_sc_enabled=true safely
        assert "vpc_sc_enabled    = true" not in body
