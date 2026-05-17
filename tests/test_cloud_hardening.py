"""Tests for the cloud-hardening substrate helpers.

Covers:
  * Tailnet + laptop-IP detection (mocked subprocess + HTTPS)
  * CIDR validation refuses world-open ranges
  * build_allowed_cidrs dedupes + preserves order
  * hardening_preview composes auto-detected pieces
  * verify_hardening reads the terraform output back
  * Notes are non-empty for the cases the operator cares about
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

from app.substrate import cloud_hardening as ch


# ── CIDR validation ────────────────────────────────────────────


class TestValidateCidr:
    def test_accepts_valid_tailnet_cidr(self):
        ok, why = ch.validate_cidr("100.64.0.0/10")
        assert ok and why == ""

    def test_accepts_host_address(self):
        ok, _ = ch.validate_cidr("1.2.3.4/32")
        assert ok

    def test_refuses_world_open_ipv4(self):
        ok, why = ch.validate_cidr("0.0.0.0/0")
        assert not ok
        assert "world-open" in why

    def test_refuses_world_open_ipv6(self):
        ok, why = ch.validate_cidr("::/0")
        assert not ok

    def test_refuses_halves_of_the_internet(self):
        ok, _ = ch.validate_cidr("0.0.0.0/1")
        assert not ok
        ok, _ = ch.validate_cidr("128.0.0.0/1")
        assert not ok

    def test_refuses_garbage(self):
        ok, why = ch.validate_cidr("not-a-cidr")
        assert not ok and "invalid CIDR" in why

    def test_refuses_empty(self):
        ok, _ = ch.validate_cidr("")
        assert not ok


# ── build_allowed_cidrs ────────────────────────────────────────


class TestBuildAllowedCidrs:
    def test_empty_inputs_yield_empty_list(self):
        out = ch.build_allowed_cidrs()
        assert out == []

    def test_tailnet_only(self):
        out = ch.build_allowed_cidrs(tailnet_cidr="100.64.0.0/10")
        assert len(out) == 1
        assert out[0].cidr_block == "100.64.0.0/10"
        assert "Tailnet" in out[0].display_name

    def test_laptop_only_becomes_slash_32(self):
        out = ch.build_allowed_cidrs(laptop_public_ip="1.2.3.4")
        assert len(out) == 1
        assert out[0].cidr_block == "1.2.3.4/32"

    def test_tailnet_first_then_laptop(self):
        out = ch.build_allowed_cidrs(
            tailnet_cidr="100.64.0.0/10",
            laptop_public_ip="1.2.3.4",
        )
        assert [c.cidr_block for c in out] == [
            "100.64.0.0/10",
            "1.2.3.4/32",
        ]

    def test_dedup_by_cidr_block(self):
        out = ch.build_allowed_cidrs(
            tailnet_cidr="100.64.0.0/10",
            extra=[
                ch.AllowedCidr(cidr_block="100.64.0.0/10", display_name="dup"),
                ch.AllowedCidr(cidr_block="10.0.0.0/8", display_name="lan"),
            ],
        )
        assert len(out) == 2
        assert out[0].display_name.startswith("Tailnet")  # first wins
        assert out[1].cidr_block == "10.0.0.0/8"

    def test_drops_world_open_extras(self):
        out = ch.build_allowed_cidrs(
            extra=[
                ch.AllowedCidr(cidr_block="0.0.0.0/0", display_name="open"),
                ch.AllowedCidr(cidr_block="10.0.0.0/8", display_name="lan"),
            ],
        )
        assert [c.cidr_block for c in out] == ["10.0.0.0/8"]

    def test_drops_invalid_laptop_ip(self):
        out = ch.build_allowed_cidrs(laptop_public_ip="not-an-ip")
        assert out == []


# ── Tailnet detection ──────────────────────────────────────────


class TestTailnetDetection:
    def test_reachable_when_tailscale_exits_0(self):
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["tailscale"], 0, "100.x.y.z   …", ""
            )
            assert ch.tailnet_reachable() is True

    def test_unreachable_when_tailscale_missing(self):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError("tailscale")):
            assert ch.tailnet_reachable() is False

    def test_unreachable_when_tailscale_exits_nonzero(self):
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["tailscale"], 1, "", "logged out"
            )
            assert ch.tailnet_reachable() is False

    def test_unreachable_when_timeout(self):
        with patch.object(
            subprocess, "run",
            side_effect=subprocess.TimeoutExpired(["tailscale"], 3.0),
        ):
            assert ch.tailnet_reachable() is False

    def test_detect_cidr_returns_constant_when_reachable(self):
        with patch.object(ch, "tailnet_reachable", return_value=True):
            assert ch.detect_tailnet_cidr() == "100.64.0.0/10"

    def test_detect_cidr_none_when_unreachable(self):
        with patch.object(ch, "tailnet_reachable", return_value=False):
            assert ch.detect_tailnet_cidr() is None


# ── Laptop public IP probe ─────────────────────────────────────


class TestLaptopPublicIp:
    def test_returns_ip_on_first_endpoint(self):
        from io import BytesIO
        from urllib import request as urlreq

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"1.2.3.4\n"

        with patch.object(urlreq, "urlopen", return_value=_Resp()):
            assert ch.detect_laptop_public_ip() == "1.2.3.4"

    def test_returns_none_when_all_endpoints_fail(self):
        from urllib import error as urlerr
        from urllib import request as urlreq

        with patch.object(urlreq, "urlopen", side_effect=urlerr.URLError("network")):
            assert ch.detect_laptop_public_ip() is None

    def test_skips_non_ip_response(self):
        from urllib import request as urlreq

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"<html>error</html>"

        with patch.object(urlreq, "urlopen", return_value=_Resp()):
            assert ch.detect_laptop_public_ip() is None


# ── Org ID detection ──────────────────────────────────────────


class TestOrgIdDetection:
    def test_returns_numeric_id_on_match(self):
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["gcloud"], 0,
                json.dumps([
                    {"name": "organizations/987654321012", "displayName": "raudsalu.com"}
                ]),
                "",
            )
            assert ch.detect_org_id() == "987654321012"

    def test_returns_none_when_no_orgs(self):
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(["gcloud"], 0, "[]", "")
            assert ch.detect_org_id() is None

    def test_returns_none_when_gcloud_missing(self):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError("gcloud")):
            assert ch.detect_org_id() is None

    def test_returns_none_on_invalid_json(self):
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(["gcloud"], 0, "not-json", "")
            assert ch.detect_org_id() is None


# ── Preview composition ──────────────────────────────────────


class TestHardeningPreview:
    def test_strict_with_nothing_detected_yields_notes(self):
        with patch.object(ch, "detect_tailnet_cidr", return_value=None), \
             patch.object(ch, "detect_laptop_public_ip", return_value=None), \
             patch.object(ch, "detect_org_id", return_value=None):
            prev = ch.hardening_preview(profile="strict", binauthz_mode="AUDIT")
            assert prev.profile == "strict"
            assert prev.recommended_cidrs == []
            assert prev.tailnet_reachable is False
            # Both "no allowlist" and "no Tailnet" notes should fire
            assert any("master_authorized_networks will be empty" in n for n in prev.notes)
            assert any("Tailnet not detected" in n for n in prev.notes)
            # "no org" note should also fire
            assert any("Workspace org not detected" in n for n in prev.notes)

    def test_strict_with_tailnet_and_org_has_clean_notes(self):
        with patch.object(ch, "detect_tailnet_cidr", return_value="100.64.0.0/10"), \
             patch.object(ch, "detect_laptop_public_ip", return_value="1.2.3.4"), \
             patch.object(ch, "detect_org_id", return_value="987654321012"):
            prev = ch.hardening_preview(profile="strict", binauthz_mode="AUDIT")
            assert prev.tailnet_reachable is True
            assert len(prev.recommended_cidrs) == 2
            # No notes should be needed
            assert prev.notes == []

    def test_strict_enforce_emits_warning_note(self):
        with patch.object(ch, "detect_tailnet_cidr", return_value="100.64.0.0/10"), \
             patch.object(ch, "detect_laptop_public_ip", return_value="1.2.3.4"), \
             patch.object(ch, "detect_org_id", return_value="987654321012"):
            prev = ch.hardening_preview(profile="strict", binauthz_mode="ENFORCE")
            assert any("ENFORCE will reject unsigned images" in n for n in prev.notes)

    def test_to_dict_is_json_serializable(self):
        prev = ch.hardening_preview(profile="strict", binauthz_mode="AUDIT")
        d = prev.to_dict()
        # JSON round-trip must work — the FastAPI response uses jsonable_encoder
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded["profile"] == "strict"
        assert isinstance(decoded["recommended_cidrs"], list)


# ── verify_hardening ───────────────────────────────────────────


class TestVerifyHardening:
    def test_missing_run_dir_returns_not_ok(self):
        result = ch.verify_hardening("/definitely/does/not/exist", project_id="x")
        assert result["ok"] is False
        assert "terraform output failed" in result["reason"]

    def test_parses_terraform_output_summary(self, tmp_path):
        # Patch subprocess.run to mock `terraform output -json hardening_summary`
        fake_summary = {
            "profile": "strict",
            "binauthz_mode": "AUDIT",
            "cmek_enabled": True,
            "cloud_armor_enabled": True,
            "master_authorized_networks": 2,
        }
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["terraform"], 0, json.dumps(fake_summary), "",
            )
            result = ch.verify_hardening(str(tmp_path), project_id="botarmy-495107")
            assert result["ok"] is True
            assert result["summary"]["profile"] == "strict"
            assert result["summary"]["master_authorized_networks"] == 2

    def test_terraform_nonzero_returns_not_ok(self, tmp_path):
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                ["terraform"], 1, "", "no state file",
            )
            result = ch.verify_hardening(str(tmp_path), project_id="x")
            assert result["ok"] is False
            assert "terraform output rc=1" in result["reason"]


# ── Runtime-settings integration ──────────────────────────────


class TestRuntimeSettings:
    def test_three_new_keys_load_with_defaults(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.delenv("BOTARMY_GCP_BOOTSTRAP_ENABLED", raising=False)
        monkeypatch.delenv("BOTARMY_HARDENING_PROFILE", raising=False)
        monkeypatch.delenv("BOTARMY_BINAUTHZ_MODE", raising=False)
        # Drive the cache through every getter
        assert rs.get_gcp_bootstrap_enabled() is False
        assert rs.get_hardening_profile() == "strict"
        assert rs.get_binauthz_mode() == "AUDIT"

    def test_set_hardening_profile_validates(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        with pytest.raises(ValueError, match="hardening_profile"):
            rs.set_hardening_profile("bogus")

    def test_set_binauthz_mode_validates(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        with pytest.raises(ValueError, match="binauthz_mode"):
            rs.set_binauthz_mode("loose")

    def test_set_hardening_profile_emits_ledger_event(self, monkeypatch, tmp_path):
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        from app import paths as _paths
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        from app.identity import continuity_ledger as cl
        monkeypatch.setattr(cl, "_path_override", tmp_path / "identity" / "continuity_ledger.jsonl")
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

        rs.set_hardening_profile("basic")
        rs.set_hardening_profile("strict")

        ledger = tmp_path / "identity" / "continuity_ledger.jsonl"
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        cm = [e for e in events if e["kind"] == "cloud_migration"]
        assert len(cm) == 2
        assert cm[0]["detail"]["phase"] == "hardening_profile_changed"
        assert cm[0]["detail"]["prior"] == "strict"
        assert cm[0]["detail"]["new"] == "basic"


# ── Per-run tfvars rendering ──────────────────────────────────


class TestTfvarsRendering:
    def test_strict_profile_emits_hardening_block(self, monkeypatch, tmp_path):
        from app.substrate.migration import _write_per_run_tfvars
        from app import paths as _paths
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        # Force strict profile via runtime_settings
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        # Stub the auto-detect helpers so the test is deterministic
        monkeypatch.setattr(ch, "detect_tailnet_cidr", lambda: "100.64.0.0/10")
        monkeypatch.setattr(ch, "detect_laptop_public_ip", lambda: "1.2.3.4")
        monkeypatch.setattr(ch, "detect_org_id", lambda: "987654321012")

        path = _write_per_run_tfvars("gcp", "botarmy-495107", "europe-north1", "cheapest", "run-1")
        body = path.read_text()
        assert 'hardening_profile = "strict"' in body
        assert 'binauthz_mode     = "AUDIT"' in body
        assert "allowed_cidrs = [" in body
        assert '100.64.0.0/10' in body
        assert '1.2.3.4/32' in body
        assert 'org_id            = "987654321012"' in body

    def test_off_profile_emits_default_block(self, monkeypatch, tmp_path):
        from app.substrate.migration import _write_per_run_tfvars
        from app import paths as _paths
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        rs.set_hardening_profile("off")

        path = _write_per_run_tfvars("gcp", "x", "europe-north1", "cheapest", "run-1")
        body = path.read_text()
        assert 'hardening_profile = "off"' in body
        # No allowed_cidrs detection when profile=off
        assert "allowed_cidrs     = []" in body

    def test_aws_target_drops_gcp_only_vars(self, monkeypatch, tmp_path):
        """AWS tfvars must NOT include project_id, binauthz_mode, or org_id —
        those are GCP-only and the AWS terraform module would reject them."""
        from app.substrate.migration import _write_per_run_tfvars
        from app import paths as _paths
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.setattr(ch, "detect_tailnet_cidr", lambda: "100.64.0.0/10")
        monkeypatch.setattr(ch, "detect_laptop_public_ip", lambda: "1.2.3.4")
        monkeypatch.setattr(ch, "detect_aws_org_root_id", lambda: "r-abcd1234")

        path = _write_per_run_tfvars("aws", "", "eu-north-1", "cheapest", "aws-run-1")
        body = path.read_text()
        # GCP-only keys must be absent
        assert "project_id" not in body
        assert "binauthz_mode" not in body
        assert "\norg_id" not in body  # `\n` anchors so 'aws_org_*' doesn't false-positive
        assert "enable_monitoring" not in body  # GCP-only var
        # AWS-specific keys present
        assert 'aws_org_enabled   = true' in body
        assert 'aws_org_root_id   = "r-abcd1234"' in body
        # Shared keys
        assert 'region            = "eu-north-1"' in body
        assert 'tier              = "cheapest"' in body
        assert 'hardening_profile = "strict"' in body

    def test_aws_target_without_org_detected(self, monkeypatch, tmp_path):
        """AWS standalone account (no Organizations) → aws_org_enabled=false."""
        from app.substrate.migration import _write_per_run_tfvars
        from app import paths as _paths
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
        monkeypatch.setattr(ch, "detect_tailnet_cidr", lambda: None)
        monkeypatch.setattr(ch, "detect_laptop_public_ip", lambda: None)
        monkeypatch.setattr(ch, "detect_aws_org_root_id", lambda: None)

        path = _write_per_run_tfvars("aws", "", "eu-north-1", "cheapest", "aws-run-2")
        body = path.read_text()
        assert 'aws_org_enabled   = false' in body


# ── AWS-side detection ───────────────────────────────────────────


class TestAwsOrgDetection:
    def test_returns_root_id_when_management_account(self):
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(
                ["aws"], 0,
                json.dumps({"Roots": [{"Id": "r-abcd1234", "Name": "Root", "Arn": "arn:..."}]}),
                "",
            )
            assert ch.detect_aws_org_root_id() == "r-abcd1234"

    def test_returns_none_for_workload_account(self):
        # Workload account can't see org structure → AccessDenied
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(
                ["aws"], 254, "", "AccessDeniedException: not a member of an org",
            )
            assert ch.detect_aws_org_root_id() is None

    def test_returns_none_when_cli_missing(self):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError("aws")):
            assert ch.detect_aws_org_root_id() is None

    def test_returns_none_when_empty_roots(self):
        with patch.object(subprocess, "run") as mr:
            mr.return_value = subprocess.CompletedProcess(
                ["aws"], 0, json.dumps({"Roots": []}), "",
            )
            assert ch.detect_aws_org_root_id() is None
