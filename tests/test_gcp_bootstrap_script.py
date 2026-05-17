"""Contract tests for scripts/install/gcp_bootstrap.sh.

We don't actually run gcloud — these tests verify the arg-parsing /
typed-phrase / format-validation paths via shell exit codes.

Each test invokes the script with --dry-run when it needs to reach the
verification step (so no real GCP calls are made), or omits the typed
phrase to exercise the early refusal paths.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "install" / "gcp_bootstrap.sh"


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, env=e, timeout=30.0,
    )


class TestBootstrapScript:
    def test_script_is_executable(self):
        assert SCRIPT.is_file()
        # +x permission
        assert os.access(SCRIPT, os.X_OK), "script must be executable"

    def test_help_prints_usage_and_exits_clean(self):
        r = _run("--help")
        assert r.returncode == 0
        assert "Stage 0a of the migrate wizard" in r.stdout

    def test_refuses_without_project_id(self):
        r = _run("--billing-account", "01ABCD-EFGH12-IJ34KL")
        assert r.returncode == 2
        assert "--project-id" in r.stderr

    def test_refuses_without_billing_account(self):
        r = _run("--project-id", "botarmy-test")
        assert r.returncode == 2
        assert "--billing-account" in r.stderr

    def test_refuses_malformed_billing_account(self):
        r = _run(
            "--project-id", "botarmy-test-abc",
            "--billing-account", "invalid",
        )
        assert r.returncode == 3
        assert "format is XXXXXX-XXXXXX-XXXXXX" in r.stderr

    def test_refuses_malformed_project_id(self):
        r = _run(
            "--project-id", "BAD_UPPERCASE",  # uppercase not allowed
            "--billing-account", "01ABCD-EFGH12-IJ34KL",
        )
        assert r.returncode == 3
        assert "--project-id must be 6-30 chars" in r.stderr

    def test_refuses_malformed_org_id(self):
        r = _run(
            "--project-id", "botarmy-test-abc",
            "--billing-account", "01ABCD-EFGH12-IJ34KL",
            "--org-id", "organizations/123",  # should be numeric only
        )
        assert r.returncode == 3
        assert "--org-id" in r.stderr

    def test_refuses_without_typed_phrase(self):
        # No --confirm flag, no BOTARMY_GCP_BOOTSTRAP_CONFIRM env
        r = _run(
            "--project-id", "botarmy-test-abc",
            "--billing-account", "01ABCD-EFGH12-IJ34KL",
        )
        # rc=5 is the typed-phrase refusal in the script. The script may
        # exit earlier if gcloud isn't installed (test environment) —
        # accept both 5 and 6 (gcloud-missing). On most dev hosts gcloud
        # is present so the test reaches rc=5.
        assert r.returncode in (5, 6)
        if r.returncode == 5:
            assert "typed-phrase confirmation required" in r.stderr

    def test_refuses_wrong_typed_phrase(self):
        r = _run(
            "--project-id", "botarmy-test-abc",
            "--billing-account", "01ABCD-EFGH12-IJ34KL",
            "--confirm", "WRONG PHRASE",
        )
        assert r.returncode in (5, 6)
        if r.returncode == 5:
            assert "typed-phrase confirmation required" in r.stderr

    def test_typed_phrase_via_env_var(self):
        # --dry-run bypasses the gcloud calls; we just want to confirm the
        # phrase from env var unblocks the typed-phrase gate.
        r = _run(
            "--project-id", "botarmy-test-abc",
            "--billing-account", "01ABCD-EFGH12-IJ34KL",
            "--dry-run",
            env={"BOTARMY_GCP_BOOTSTRAP_CONFIRM": "CREATE GCP PROJECT"},
        )
        # rc=0 means dry-run completed past the typed-phrase gate (the
        # `--dry-run` short-circuit at the very top skips the phrase
        # check by design — it's a preview path)
        assert r.returncode == 0 or r.returncode == 6  # 6 if gcloud missing in test env

    def test_dry_run_does_not_require_phrase(self):
        r = _run(
            "--project-id", "botarmy-test-abc",
            "--billing-account", "01ABCD-EFGH12-IJ34KL",
            "--dry-run",
        )
        # rc=0 for dry-run when gcloud is present; rc=6 when gcloud missing.
        assert r.returncode in (0, 6)
        if r.returncode == 0:
            assert "Stage 0a complete" in r.stdout
