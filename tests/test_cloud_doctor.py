"""Tests for app.substrate.cloud_doctor — WP D Phase 1.

The doctor never raises and never spends money. Probes shell out via
``subprocess.run``; tests monkeypatch that single seam so behavior is
deterministic regardless of what's on the operator's PATH.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.substrate import cloud_doctor as cd
from app.substrate.cloud_doctor import (
    CloudReadiness,
    ProbeResult,
    check_readiness,
    format_readiness,
)


# ── Fake-subprocess fixtures ────────────────────────────────────────


@pytest.fixture
def fake_shell(monkeypatch):
    """Drop-in replacement for cloud_doctor._run that the test parameterizes.

    Returns a dict that tests mutate to set ``(argv[0]) → (rc, out, err)``.
    Tools not in the dict appear MISSING (returncode 127).
    """
    fake: dict[str, tuple[int, str, str]] = {}

    def _fake_run(argv, timeout=10.0):
        return fake.get(argv[0], (127, "", f"{argv[0]}: not found"))

    monkeypatch.setattr(cd, "_run", _fake_run)
    # shutil.which is also consulted before _run — short-circuit on
    # whichever tools the test marked as present.
    import shutil as _shutil

    def _fake_which(name):
        return f"/usr/local/bin/{name}" if name in fake else None

    monkeypatch.setattr(_shutil, "which", _fake_which)
    return fake


# ── Public API shape ─────────────────────────────────────────────────


class TestPublicShape:
    def test_check_readiness_returns_cloud_readiness(self, fake_shell):
        r = check_readiness("gcp")
        assert isinstance(r, CloudReadiness)
        assert r.target == "gcp"
        assert r.timestamp  # ISO 8601
        assert isinstance(r.probes, list)
        assert len(r.probes) >= 6   # generic 4 + gcp 4 + continuity 2

    def test_unknown_target_raises(self, fake_shell):
        with pytest.raises(ValueError, match="unknown target"):
            check_readiness("azure")  # type: ignore[arg-type]

    def test_to_dict_serializable(self, fake_shell):
        r = check_readiness("gcp")
        d = r.to_dict()
        assert d["target"] == "gcp"
        assert "probes" in d
        for p in d["probes"]:
            assert "name" in p
            assert "status" in p
            assert "detail" in p
            assert "required" in p


# ── Status roll-up ──────────────────────────────────────────────────


class TestOverallStatus:
    def test_all_tools_present_yields_ok(self, fake_shell, tmp_path, monkeypatch):
        # Mark every required tool as present with successful output.
        for tool, banner in {
            "terraform": "Terraform v1.5.0",
            "kubectl":   "Client Version: v1.28.0",
            "helm":      "v3.13.0+g000",
            "docker":    "Docker version 24.0.0",
            "gcloud":    "Google Cloud SDK 450.0.0",
        }.items():
            fake_shell[tool] = (0, banner, "")
        # gcloud auth + project + ADC all need separate stubs since
        # they invoke `gcloud` with different subcommands — _fake_run
        # keys on argv[0] only, so the same stub satisfies all three.
        # Make the active-account / project responses come through:
        # the fake_run for gcloud will return banner; but the doctor
        # parses specific output for auth/project/ADC. Override _run
        # via a callable that switches on argv tail.
        original_run = cd._run

        def _smart_run(argv, timeout=10.0):
            if argv[0] != "gcloud":
                return original_run(argv, timeout=timeout)
            if argv[:3] == ["gcloud", "auth", "list"]:
                return (0, "operator@example.com", "")
            if argv[:3] == ["gcloud", "config", "get"]:
                return (0, "test-project", "")
            if argv[:4] == ["gcloud", "auth", "application-default", "print-access-token"]:
                return (0, "ya29.fake-token", "")
            # Gap-3 probes: project access + required APIs
            if argv[:3] == ["gcloud", "projects", "describe"]:
                return (0, "test-project", "")
            if argv[:3] == ["gcloud", "services", "list"]:
                # Return ALL required APIs as enabled
                return (0, "\n".join(cd._REQUIRED_GCP_APIS), "")
            return (0, "Google Cloud SDK 450.0.0", "")

        monkeypatch.setattr(cd, "_run", _smart_run)

        # Continuity bundle: make WORKSPACE_ROOT point at a tmp with a
        # fresh-ish tarball so the bundle probe is OK.
        from app import paths as _paths
        backup_dir = tmp_path / "backups" / "dr"
        backup_dir.mkdir(parents=True)
        (backup_dir / "dr_test.tar.gz").write_bytes(b"x")
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)

        # SubIA integrity: mock as OK
        from app.subia import integrity as _integ

        class _FakeOK:
            ok = True
            n_files = 164
            mismatched = []
            extra = []
            missing = []

        monkeypatch.setattr(_integ, "verify_integrity", lambda strict=False: _FakeOK())

        # Gap-3 probe: ADC account — write a populated json into tmp HOME
        import json as _json
        adc_dir = tmp_path / ".config" / "gcloud"
        adc_dir.mkdir(parents=True)
        (adc_dir / "application_default_credentials.json").write_text(
            _json.dumps({"account": "operator@example.com"})
        )
        monkeypatch.setenv("HOME", str(tmp_path))

        r = check_readiness("gcp")
        assert r.overall == "OK", [
            (p.name, p.status, p.detail) for p in r.probes
        ]

    def test_missing_terraform_yields_missing_overall(self, fake_shell):
        # Mark everything except terraform as present
        for tool in ("kubectl", "helm", "docker", "gcloud"):
            fake_shell[tool] = (0, "fake-banner", "")
        # terraform stays unset → MISSING
        r = check_readiness("gcp")
        assert r.overall == "MISSING"
        tf = next(p for p in r.probes if p.name == "terraform")
        assert tf.status == "MISSING"

    def test_overall_aws_branch_runs_aws_probes(self, fake_shell):
        for tool in ("terraform", "kubectl", "helm", "docker", "aws"):
            fake_shell[tool] = (0, "fake-banner", "")
        r = check_readiness("aws")
        names = {p.name for p in r.probes}
        assert "aws" in names or "aws identity" in names
        # GCP probes should NOT be present in AWS readiness
        assert not any(p.name.startswith("gcloud") for p in r.probes)


# ── Specific probe semantics ────────────────────────────────────────


class TestProbeSemantics:
    def test_continuity_bundle_missing_when_no_backup(self, fake_shell, tmp_path, monkeypatch):
        from app import paths as _paths
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        # No backup dir exists
        r = check_readiness("gcp")
        bundle = next(p for p in r.probes if p.name == "continuity bundle")
        assert bundle.status == "MISSING"

    def test_continuity_bundle_stale_when_old(self, fake_shell, tmp_path, monkeypatch):
        import os as _os
        import time as _time
        from app import paths as _paths
        monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
        backup_dir = tmp_path / "backups" / "dr"
        backup_dir.mkdir(parents=True)
        tar = backup_dir / "dr_old.tar.gz"
        tar.write_bytes(b"x")
        old = _time.time() - (30 * 86400)   # 30 days old
        _os.utime(tar, (old, old))
        r = check_readiness("gcp")
        bundle = next(p for p in r.probes if p.name == "continuity bundle")
        assert bundle.status == "STALE"
        assert "30." in bundle.detail or "29." in bundle.detail

    def test_gcloud_no_active_account_marks_missing(self, fake_shell, monkeypatch):
        # gcloud installed but no active account
        fake_shell["gcloud"] = (0, "Google Cloud SDK", "")

        def _empty_auth_run(argv, timeout=10.0):
            if argv[0] != "gcloud":
                return (127, "", "")
            if argv[:3] == ["gcloud", "auth", "list"]:
                return (0, "", "")   # no active account
            return (0, "irrelevant", "")

        monkeypatch.setattr(cd, "_run", _empty_auth_run)
        r = check_readiness("gcp")
        auth = next(p for p in r.probes if p.name == "gcloud auth")
        assert auth.status == "MISSING"
        assert "gcloud auth login" in auth.detail


# ── Format ───────────────────────────────────────────────────────────


class TestFormatReadiness:
    def test_format_includes_target_and_overall(self, fake_shell):
        r = check_readiness("gcp")
        out = format_readiness(r)
        assert "gcp" in out
        assert r.overall in out
        # Every probe should appear by name
        for p in r.probes:
            assert p.name in out


# ── Identity-ledger event-kind smoke ────────────────────────────────


class TestPermissionProbes:
    """Gap-3 fix (2026-05-17): catch identity↔project↔API issues that
    pre-existing probes missed.

    The original 4 GCP probes only checked "gcloud is technically
    authenticated." All three could be technically true while the
    active identity lacked the project-level IAM roles that terraform
    apply needs — so the doctor said DEGRADED, the operator ran live
    migrate, and terraform failed partway, leaving cost-accruing
    partial state.
    """

    def test_active_account_type_user_is_ok(self, monkeypatch):
        """Active account `someone@domain.com` → OK."""
        from app.substrate import cloud_doctor as cd
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda n: f"/usr/local/bin/{n}")

        def _fake(argv, timeout=10.0):
            if argv[:3] == ["gcloud", "auth", "list"]:
                return (0, "andrus@raudsalu.com", "")
            return (0, "", "")

        monkeypatch.setattr(cd, "_run", _fake)
        result = cd._probe_gcp_active_account_type()
        assert result.status == "OK"
        assert "user account" in result.detail

    def test_active_account_type_service_account_warns(self, monkeypatch):
        """Active account ending in `.gserviceaccount.com` → STALE warning."""
        from app.substrate import cloud_doctor as cd
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda n: f"/usr/local/bin/{n}")

        def _fake(argv, timeout=10.0):
            if argv[:3] == ["gcloud", "auth", "list"]:
                return (0, "ci-runner@some-project.iam.gserviceaccount.com", "")
            return (0, "", "")

        monkeypatch.setattr(cd, "_run", _fake)
        result = cd._probe_gcp_active_account_type()
        assert result.status == "STALE"
        assert "service account" in result.detail
        assert "ci-runner" in result.detail

    def test_project_access_403_marks_missing(self, monkeypatch):
        """gcloud projects describe → 403 means active identity lacks
        even Viewer role."""
        from app.substrate import cloud_doctor as cd
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda n: f"/usr/local/bin/{n}")

        def _fake(argv, timeout=10.0):
            if argv[:3] == ["gcloud", "config", "get"]:
                return (0, "my-project", "")
            if argv[:3] == ["gcloud", "projects", "describe"]:
                return (1, "", "ERROR: ... 403: does not have permission ...")
            return (0, "", "")

        monkeypatch.setattr(cd, "_run", _fake)
        result = cd._probe_gcp_project_access()
        assert result.status == "MISSING"
        assert "Viewer" in result.detail or "403" in result.detail

    def test_project_access_ok_when_describe_succeeds(self, monkeypatch):
        from app.substrate import cloud_doctor as cd
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda n: f"/usr/local/bin/{n}")

        def _fake(argv, timeout=10.0):
            if argv[:3] == ["gcloud", "config", "get"]:
                return (0, "my-project", "")
            if argv[:3] == ["gcloud", "projects", "describe"]:
                return (0, "my-project", "")
            return (0, "", "")

        monkeypatch.setattr(cd, "_run", _fake)
        result = cd._probe_gcp_project_access()
        assert result.status == "OK"
        assert "my-project" in result.detail

    def test_required_apis_all_enabled_is_ok(self, monkeypatch):
        from app.substrate import cloud_doctor as cd
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda n: f"/usr/local/bin/{n}")

        def _fake(argv, timeout=10.0):
            if argv[:3] == ["gcloud", "services", "list"]:
                # Return every required API plus some extras
                lines = list(cd._REQUIRED_GCP_APIS) + ["other.googleapis.com"]
                return (0, "\n".join(lines), "")
            return (0, "", "")

        monkeypatch.setattr(cd, "_run", _fake)
        result = cd._probe_gcp_required_apis()
        assert result.status == "OK"

    def test_required_apis_missing_marks_missing(self, monkeypatch):
        from app.substrate import cloud_doctor as cd
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda n: f"/usr/local/bin/{n}")

        def _fake(argv, timeout=10.0):
            if argv[:3] == ["gcloud", "services", "list"]:
                # Only some required APIs enabled
                return (0, "iam.googleapis.com\nstorage.googleapis.com", "")
            return (0, "", "")

        monkeypatch.setattr(cd, "_run", _fake)
        result = cd._probe_gcp_required_apis()
        assert result.status == "MISSING"
        assert "APIs disabled" in result.detail
        # Should give the operator the exact gcloud command to run
        assert "gcloud services enable" in result.detail

    def test_adc_populated_when_account_set(self, tmp_path, monkeypatch):
        """ADC json with a non-empty `account` field → OK."""
        import json
        adc_dir = tmp_path / ".config" / "gcloud"
        adc_dir.mkdir(parents=True)
        adc_file = adc_dir / "application_default_credentials.json"
        adc_file.write_text(json.dumps({"account": "user@example.com", "type": "authorized_user"}))
        monkeypatch.setenv("HOME", str(tmp_path))

        from app.substrate import cloud_doctor as cd
        result = cd._probe_gcp_adc_populated()
        assert result.status == "OK"
        assert "user@example.com" in result.detail

    def test_adc_empty_account_warns(self, tmp_path, monkeypatch):
        """ADC json with empty `account` field → STALE warning."""
        import json
        adc_dir = tmp_path / ".config" / "gcloud"
        adc_dir.mkdir(parents=True)
        adc_file = adc_dir / "application_default_credentials.json"
        adc_file.write_text(json.dumps({"account": "", "client_id": "anonymous"}))
        monkeypatch.setenv("HOME", str(tmp_path))

        from app.substrate import cloud_doctor as cd
        result = cd._probe_gcp_adc_populated()
        assert result.status == "STALE"
        assert "no explicit account" in result.detail

    def test_adc_missing_file_marks_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from app.substrate import cloud_doctor as cd
        result = cd._probe_gcp_adc_populated()
        assert result.status == "MISSING"
        assert "application-default login" in result.detail

    def test_check_readiness_registers_all_new_probes_for_gcp(self, fake_shell):
        """Make sure the new probes are wired into check_readiness."""
        # Mark gcloud + tools present so we get to the new probes
        for tool in ("terraform", "kubectl", "helm", "docker", "gcloud"):
            fake_shell[tool] = (0, "fake", "")
        from app.substrate.cloud_doctor import check_readiness
        r = check_readiness("gcp")
        names = [p.name for p in r.probes]
        # Original four
        assert "gcloud auth" in names
        assert "gcloud project" in names
        assert "ADC" in names
        # New gap-3 four
        assert "gcloud account type" in names
        assert "gcloud project access" in names
        assert "gcloud required APIs" in names
        assert "ADC account" in names

    def test_required_apis_failure_blocks_overall(self, fake_shell, monkeypatch):
        """If required APIs are missing, overall must be MISSING.

        This pins the operator-visible promise: if any required probe
        fails, the doctor refuses to greenlight a live migrate.
        """
        from app.substrate import cloud_doctor as cd
        # Tools present
        for tool in ("terraform", "kubectl", "helm", "docker", "gcloud"):
            fake_shell[tool] = (0, "fake-banner", "")

        # Compose smart-run: gcloud auth + project succeed, services list
        # returns nothing (= APIs disabled), describe is 200.
        original_run = cd._run

        def _smart(argv, timeout=10.0):
            if argv[:3] == ["gcloud", "auth", "list"]:
                return (0, "andrus@raudsalu.com", "")
            if argv[:3] == ["gcloud", "config", "get"]:
                return (0, "my-project", "")
            if argv[:4] == ["gcloud", "auth", "application-default", "print-access-token"]:
                return (0, "ya29.fake", "")
            if argv[:3] == ["gcloud", "projects", "describe"]:
                return (0, "my-project", "")
            if argv[:3] == ["gcloud", "services", "list"]:
                # No required APIs enabled
                return (0, "", "")
            return original_run(argv, timeout=timeout)

        monkeypatch.setattr(cd, "_run", _smart)

        r = cd.check_readiness("gcp")
        # Required APIs failure should pull overall to MISSING
        assert r.overall == "MISSING", [
            (p.name, p.status, p.detail) for p in r.probes
        ]
        apis_probe = next(p for p in r.probes if p.name == "gcloud required APIs")
        assert apis_probe.status == "MISSING"


class TestCloudMigrationEventKindRegistered:
    """Phase 1 also registers the cloud_migration identity-event kind."""

    def test_kind_is_in_set(self):
        from app.identity.continuity_ledger import IDENTITY_EVENT_KINDS
        assert "cloud_migration" in IDENTITY_EVENT_KINDS

    def test_event_can_be_emitted(self, tmp_path, monkeypatch):
        from app.identity import continuity_ledger as cl

        # Force-enable + redirect ledger to a tmp path via the public
        # ``path`` parameter so we don't rely on private overrides.
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
        ledger_path = tmp_path / "ledger.jsonl"

        ok = cl.record_event(
            kind="cloud_migration",
            actor="botarmy_cli",
            summary="migrate started → gcp europe-north1",
            detail={"phase": "started", "target": "gcp", "region": "europe-north1"},
            path=ledger_path,
        )

        assert ok is True
        assert ledger_path.exists()
        import json
        lines = ledger_path.read_text().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["kind"] == "cloud_migration"
        assert row["actor"] == "botarmy_cli"
        assert row["detail"]["phase"] == "started"
        assert row["detail"]["region"] == "europe-north1"
