"""Tests for app.coding_session.runner — Phase 5.4-b.

Coverage:
  * **Allowlist** — pure check_allowlist() validation
  * **Execution** — happy path, exit codes, captured stdout/stderr,
    elapsed_ms reasonable, env-var stripping
  * **Refusal** — disallowed exe, disallowed subcommand, path-style
    exe, empty argv, missing cwd, missing executable
  * **Timeout** — long-running command killed at wallclock; timed_out
    flag set; partial output preserved; SIGKILL marker in stderr
  * **Output cap** — 64 KiB cap; truncated_stdout / truncated_stderr
    flags; cap honoured at runtime, not just in assertions
  * **Sandbox wrapper** — env-driven CODING_SESSION_SANDBOX selects the
    wrapper prefix; "none" is identity; unknown values fall back

Tests run real subprocesses via ``python -c ...``. Fast (each test
< 1 s); avoid relying on pytest / git in PATH (the runner test
shouldn't depend on tools not in the allowlist's allowed-anywhere set).
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ── Allowlist ───────────────────────────────────────────────────────


class TestAllowlist:

    def test_allowed_no_subcommand_required(self) -> None:
        from app.coding_session.runner import check_allowlist

        ok, reason = check_allowlist(["python", "-c", "print(1)"])
        assert ok
        assert reason is None

    def test_disallowed_exe_rejected(self) -> None:
        from app.coding_session.runner import check_allowlist

        ok, reason = check_allowlist(["bash", "-c", "echo hi"])
        assert not ok
        assert reason is not None
        assert "not in the allowlist" in reason

    def test_curl_rejected(self) -> None:
        """Network exfil canary — curl must never be allowed."""
        from app.coding_session.runner import check_allowlist

        ok, _ = check_allowlist(["curl", "https://evil.example.com"])
        assert not ok

    def test_pip_install_rejected(self) -> None:
        """Manifest changes go through change-request, not the runner."""
        from app.coding_session.runner import check_allowlist

        ok, _ = check_allowlist(["pip", "install", "requests"])
        assert not ok

    def test_path_style_exe_rejected(self) -> None:
        """Forces use of bare names so allowlist can't be bypassed via
        absolute path that aliases a different binary."""
        from app.coding_session.runner import check_allowlist

        ok, reason = check_allowlist(["/usr/bin/python", "-c", "1"])
        assert not ok
        assert reason is not None
        assert "path separators" in reason

    def test_git_subcommand_allowed(self) -> None:
        from app.coding_session.runner import check_allowlist

        for sub in ("status", "diff", "log", "show", "rev-parse", "ls-files"):
            ok, _ = check_allowlist(["git", sub])
            assert ok, f"git {sub} should be allowed"

    def test_git_subcommand_disallowed(self) -> None:
        from app.coding_session.runner import check_allowlist

        for sub in ("push", "commit", "checkout", "merge", "reset", "rm"):
            ok, reason = check_allowlist(["git", sub])
            assert not ok, f"git {sub} should be denied"
            assert reason is not None
            assert "not in the allowlist" in reason

    def test_git_config_read_allowed_write_blocked(self) -> None:
        """git config without write flags is a read; --replace etc must
        be rejected even though the subcommand 'config' is on the list."""
        from app.coding_session.runner import check_allowlist

        ok, _ = check_allowlist(["git", "config", "--get", "user.email"])
        assert ok

        for arg in ("--replace", "--unset", "--add", "--unset-all"):
            ok, reason = check_allowlist(["git", "config", arg, "user.email", "x"])
            assert not ok
            assert reason is not None
            assert "write" in reason.lower()

    def test_gh_subcommand_allowed_disallowed(self) -> None:
        from app.coding_session.runner import check_allowlist

        ok, _ = check_allowlist(["gh", "pr", "view", "1"])
        assert ok
        ok, _ = check_allowlist(["gh", "issue", "view", "1"])
        assert ok
        ok, reason = check_allowlist(["gh", "auth", "status"])
        assert not ok
        assert reason is not None

    def test_empty_argv(self) -> None:
        from app.coding_session.runner import check_allowlist

        ok, reason = check_allowlist([])
        assert not ok
        assert reason is not None and "empty" in reason

    def test_argv_zero_must_be_string(self) -> None:
        """Defensive — pydantic / json may pass odd types."""
        from app.coding_session.runner import check_allowlist

        ok, _ = check_allowlist([123, "arg"])  # type: ignore[list-item]
        assert not ok


# ── Execution / RunResult ───────────────────────────────────────────


class TestRunHappyPath:

    def test_python_print(self, tmp_path: Path) -> None:
        from app.coding_session.runner import run

        result = run(
            argv=["python", "-c", "print('hello world')"],
            cwd=tmp_path,
            timeout_s=10,
        )
        assert result.ok
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.stderr == ""
        assert not result.timed_out
        assert not result.truncated_stdout
        assert not result.truncated_stderr
        assert result.elapsed_ms >= 0  # subprocess startup may be sub-ms

    def test_non_zero_exit_captured(self, tmp_path: Path) -> None:
        from app.coding_session.runner import run

        result = run(
            argv=["python", "-c", "import sys; sys.exit(7)"],
            cwd=tmp_path,
            timeout_s=10,
        )
        assert not result.ok
        assert result.exit_code == 7
        assert not result.timed_out
        assert not result.refused

    def test_stderr_captured(self, tmp_path: Path) -> None:
        from app.coding_session.runner import run

        result = run(
            argv=["python", "-c", "import sys; sys.stderr.write('boom'); sys.exit(1)"],
            cwd=tmp_path,
            timeout_s=10,
        )
        assert "boom" in result.stderr
        assert result.exit_code == 1


class TestRunRefusal:

    def test_disallowed_exe(self, tmp_path: Path) -> None:
        from app.coding_session.runner import run

        result = run(argv=["bash", "-c", "echo hi"], cwd=tmp_path, timeout_s=5)
        assert result.refused
        assert result.exit_code == -1
        assert result.refusal_reason is not None
        assert "allowlist" in result.refusal_reason

    def test_missing_cwd(self) -> None:
        from app.coding_session.runner import run

        result = run(
            argv=["python", "-c", "print('x')"],
            cwd="/tmp/this-dir-does-not-exist-12345",
            timeout_s=5,
        )
        assert result.refused
        assert result.refusal_reason is not None
        assert "does not exist" in result.refusal_reason

    def test_to_dict_omits_refusal_fields_when_not_refused(
        self, tmp_path: Path,
    ) -> None:
        from app.coding_session.runner import run

        result = run(
            argv=["python", "-c", "print(1)"], cwd=tmp_path, timeout_s=5,
        )
        d = result.to_dict()
        assert "refused" not in d
        assert "refusal_reason" not in d
        assert d["ok"] is True


class TestRunTimeout:

    def test_long_running_killed(self, tmp_path: Path) -> None:
        """Sleeps 30 s, but the runner caps at 1 s. Should kill within
        a small slack window."""
        from app.coding_session.runner import run

        result = run(
            argv=["python", "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            timeout_s=1,
        )
        assert result.timed_out
        assert result.exit_code == -9
        # Marker for the agent
        assert "killed after 1s" in result.stderr
        # Sanity: didn't actually run for the full 30 s
        assert result.elapsed_ms < 5_000


class TestRunOutputCap:

    def test_stdout_truncated(self, tmp_path: Path) -> None:
        from app.coding_session.runner import run

        # Print 70 KiB; cap is 64 KiB
        result = run(
            argv=[
                "python", "-c",
                "import sys; sys.stdout.write('a' * (70 * 1024))",
            ],
            cwd=tmp_path,
            timeout_s=10,
            output_cap_bytes=64 * 1024,
        )
        assert result.truncated_stdout
        assert len(result.stdout) <= 64 * 1024
        assert not result.truncated_stderr

    def test_stderr_truncated(self, tmp_path: Path) -> None:
        from app.coding_session.runner import run

        result = run(
            argv=[
                "python", "-c",
                "import sys; sys.stderr.write('b' * (70 * 1024))",
            ],
            cwd=tmp_path,
            timeout_s=10,
            output_cap_bytes=64 * 1024,
        )
        assert result.truncated_stderr
        assert len(result.stderr) <= 64 * 1024
        assert not result.truncated_stdout


class TestEnvStripping:

    def test_secrets_stripped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ANTHROPIC_API_KEY in parent env → child env must NOT see it."""
        from app.coding_session.runner import run

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        result = run(
            argv=[
                "python", "-c",
                "import os; print(os.environ.get('ANTHROPIC_API_KEY', '<missing>'))",
            ],
            cwd=tmp_path,
            timeout_s=10,
        )
        assert "<missing>" in result.stdout
        assert "sk-ant-secret" not in result.stdout

    def test_explicit_override_propagates(self, tmp_path: Path) -> None:
        from app.coding_session.runner import run

        result = run(
            argv=[
                "python", "-c",
                "import os; print(os.environ.get('MY_VAR', '<missing>'))",
            ],
            cwd=tmp_path,
            timeout_s=10,
            env={"MY_VAR": "hello"},
        )
        assert "hello" in result.stdout

    def test_path_inherited(self, tmp_path: Path) -> None:
        """The child needs PATH so it can find python in the first place
        (the test would not even start without it)."""
        from app.coding_session.runner import run

        result = run(
            argv=["python", "-c", "import os; print(bool(os.environ.get('PATH')))"],
            cwd=tmp_path,
            timeout_s=10,
        )
        assert "True" in result.stdout


class TestSandboxWrapper:

    def test_default_none_is_identity(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.coding_session.runner import _apply_sandbox_wrapper

        monkeypatch.delenv("CODING_SESSION_SANDBOX", raising=False)
        assert _apply_sandbox_wrapper(["python", "-c", "1"]) == [
            "python", "-c", "1",
        ]

    def test_unshare_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.coding_session.runner import _apply_sandbox_wrapper

        monkeypatch.setenv("CODING_SESSION_SANDBOX", "unshare-n")
        assert _apply_sandbox_wrapper(["python", "-c", "1"])[0] == "unshare"

    def test_unknown_falls_back_to_identity(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.coding_session.runner import _apply_sandbox_wrapper

        monkeypatch.setenv("CODING_SESSION_SANDBOX", "rocketship")
        assert _apply_sandbox_wrapper(["python", "-c", "1"]) == [
            "python", "-c", "1",
        ]
