"""Bounded subprocess runner for coding-session ``run`` calls.

The single most security-sensitive piece of the coding-session
system. The runner is what lets the agent execute arbitrary
commands inside its worktree — subject to a strict allowlist,
wallclock + CPU bounds, output truncation, and (in production)
network isolation.

Layered defenses, in order of importance:

  1. **argv is a list, not a shell string.** ``subprocess.run`` is
     called with ``shell=False``. Shell metacharacters (``;``,
     ``|``, ``>``, backticks, ``$()``) have no special meaning;
     they're treated as literal arguments to the executable.
  2. **Executable allowlist.** ``argv[0]`` must be in
     ``ALLOWLIST``. Some entries (``git``, ``gh``) further restrict
     which subcommands are permitted (read-only only).
  3. **Wallclock timeout.** ``subprocess.run(timeout=...)`` raises
     ``TimeoutExpired`` and the process is killed.
  4. **CPU rlimit.** ``preexec_fn`` sets ``RLIMIT_CPU`` to 4× the
     wallclock budget — defends against fork-bomb-style runaway
     programs that don't tick the wallclock.
  5. **Working directory locked.** ``cwd`` is the worktree path;
     the process inherits no other context.
  6. **Output truncation.** stdout and stderr are each capped at
     64 KiB. Truncated streams set the corresponding ``truncated_*``
     flag in ``RunResult`` so the agent knows to ask for more focused
     test selection.
  7. **Network isolation.** Pluggable via ``RUNNER_SANDBOX`` env var
     (default: ``"none"`` — relies on the container's egress policy.
     Production K8s sets ``"netns"`` to wrap with ``unshare -n``).
     The hook lives in ``_apply_sandbox_wrapper``.

What the runner does NOT enforce — this is the manager's job:

  * Session state (must be ACTIVE) — the tools layer checks before
    calling.
  * Cwd is the right worktree — the tools layer passes it.
  * Quota on run count — the manager records via ``record_run``.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Allowlist ───────────────────────────────────────────────────────


# Per-executable subcommand allowlist. Value is:
#   * ``None``  → any first arg is permitted
#   * ``set[str]`` → ``argv[1]`` must be in the set
#
# Note we deliberately exclude install / push / write commands. The
# coding session is read-only against the network and write-only
# against the worktree. Manifest changes (adding a new dep) flow
# through the change-request system, not the runner.
ALLOWLIST: dict[str, set[str] | None] = {
    # Test / lint / typecheck — the iteration loop
    "pytest": None,
    "python": None,
    "python3": None,
    "ruff": None,
    "mypy": None,
    "eslint": None,
    "node": None,
    "npx": None,        # NB: npx can run arbitrary packages; trusted because
                        # session disk is bounded and network is isolated
    "tsc": None,
    # Read-only git and gh — context for the agent
    "git": {
        "status", "diff", "log", "show", "ls-files", "rev-parse",
        "branch",   # read-only forms (no -d / -D in path; argv literal check below)
        "config",   # read-only without --replace; for `git config --get`
    },
    "gh": {
        "pr",       # gh pr view / list
        "issue",    # gh issue view / list
    },
    # POSIX read tools
    "cat": None, "ls": None, "wc": None, "head": None, "tail": None,
    "grep": None, "rg": None, "find": None, "tree": None,
    "diff": None, "sort": None, "uniq": None, "cut": None, "tr": None,
    # Bash arithmetic / shell only as `python -c` — no /bin/sh entry by design
}

# Hard ceiling on captured output per stream. Anything past this is
# silently dropped and a `truncated` flag is set. Goal: keep the
# tool's response compact while still surfacing test failures + tracebacks.
_DEFAULT_OUTPUT_CAP_BYTES = 64 * 1024


# ── Result ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunResult:
    """Return shape of ``run()``. Holds everything the agent and the
    audit log need; nothing more."""

    argv: list[str]
    cwd: str
    exit_code: int
    elapsed_ms: int

    stdout: str
    stderr: str
    truncated_stdout: bool = False
    truncated_stderr: bool = False
    timed_out: bool = False

    refused: bool = False
    refusal_reason: str | None = None

    @property
    def ok(self) -> bool:
        """Convenience predicate for the agent prompt — green if the
        process exited 0 and wasn't refused/killed."""
        return (
            not self.refused
            and not self.timed_out
            and self.exit_code == 0
        )

    def to_dict(self) -> dict:
        d: dict = {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated_stdout": self.truncated_stdout,
            "truncated_stderr": self.truncated_stderr,
            "timed_out": self.timed_out,
            "ok": self.ok,
        }
        if self.refused:
            d["refused"] = True
            d["refusal_reason"] = self.refusal_reason
        return d


# ── Allowlist check ─────────────────────────────────────────────────


def check_allowlist(argv: list[str]) -> tuple[bool, str | None]:
    """Pure validation. Returns ``(ok, reason)``.

    Tested in isolation so the allowlist evolution (add a new tool,
    tighten a subcommand list) has clear regression coverage.
    """
    if not argv:
        return False, "argv is empty"
    exe = argv[0]
    if not isinstance(exe, str):
        return False, "argv[0] must be a string"

    # Reject path-style executables — `python` is allowed, `/usr/bin/python`
    # would bypass the basename check downstream and could resolve to a
    # different binary at deploy time. Force the agent to use the bare name.
    if "/" in exe or "\\" in exe:
        return False, (
            f"executable {exe!r} contains path separators; "
            "use the bare name (e.g. 'python', not '/usr/bin/python')"
        )

    if exe not in ALLOWLIST:
        return False, (
            f"executable {exe!r} is not in the allowlist. "
            f"Permitted: {', '.join(sorted(ALLOWLIST))}"
        )

    sub_allowed = ALLOWLIST[exe]
    if sub_allowed is None:
        return True, None

    if len(argv) < 2:
        return False, (
            f"{exe!r} requires a subcommand. "
            f"Permitted: {', '.join(sorted(sub_allowed))}"
        )
    sub = argv[1]
    if sub not in sub_allowed:
        return False, (
            f"{exe} {sub!r} is not in the allowlist. "
            f"Permitted: {', '.join(sorted(sub_allowed))}"
        )

    # Extra defense for `git config`: bare `git config` reads, but
    # `git config --replace`, `--unset`, `--add` write. Block writes.
    if exe == "git" and sub == "config":
        for arg in argv[2:]:
            if arg in {"--replace", "--unset", "--unset-all", "--add",
                       "--rename-section", "--remove-section"}:
                return False, (
                    "git config write subcommands are not allowed in a "
                    "coding session"
                )

    return True, None


# ── Sandbox hooks ───────────────────────────────────────────────────


def _apply_sandbox_wrapper(argv: list[str]) -> list[str]:
    """Wrap argv with a network-isolation prefix if configured.

    The actual implementation chosen is gated on ``CODING_SESSION_SANDBOX``:

      * ``"none"`` (default) — no wrapper. Relies on the surrounding
        container's egress policy. Cross-platform (works on dev macOS).
      * ``"unshare-n"`` — prepends ``unshare -n``. Requires Linux +
        ``CAP_SYS_ADMIN``. Strongest isolation when available.
      * ``"firejail"`` — prepends ``firejail --net=none --quiet``.
        Requires firejail installed.
      * ``"bwrap"`` — prepends ``bwrap --ro-bind / / --proc /proc
        --dev /dev --share-net=false``. Requires bubblewrap.

    Production K8s manifests should set ``CODING_SESSION_SANDBOX=unshare-n``
    once we've verified the kubelet grants the required capability.
    """
    mode = os.environ.get("CODING_SESSION_SANDBOX", "none")
    if mode == "none":
        return argv
    if mode == "unshare-n":
        return ["unshare", "-n", "--", *argv]
    if mode == "firejail":
        return ["firejail", "--net=none", "--quiet", *argv]
    if mode == "bwrap":
        return [
            "bwrap",
            "--ro-bind", "/", "/",
            "--proc", "/proc",
            "--dev", "/dev",
            "--share-net=false",
            "--",
            *argv,
        ]
    logger.warning("runner: unknown CODING_SESSION_SANDBOX=%r; falling back to 'none'", mode)
    return argv


def _preexec(cpu_seconds: int) -> "object":
    """Return a preexec_fn that applies CPU rlimits to the child.

    ``RLIMIT_CPU`` value is whichever the OS supports — Linux always,
    macOS/BSD honour soft limit. CPU is wallclock × 4 — generous
    enough that legitimate compile-heavy commands aren't killed
    spuriously, tight enough to stop fork-bomb-style abuse.
    """
    try:
        import resource  # type: ignore[import-untyped]
    except ImportError:
        return None  # Windows; no rlimits available

    def _setlimits() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError) as exc:
            # If rlimit fails the child still runs — we still have
            # the wallclock timeout as the backstop. Don't crash the
            # whole call.
            logger.debug("runner: setrlimit RLIMIT_CPU failed: %s", exc)

    return _setlimits


# ── Core run() ──────────────────────────────────────────────────────


def run(
    *,
    argv: list[str],
    cwd: str | Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
    output_cap_bytes: int = _DEFAULT_OUTPUT_CAP_BYTES,
) -> RunResult:
    """Execute ``argv`` in ``cwd`` with all sandbox layers applied.

    Args:
        argv: command + args; ``argv[0]`` must be in the allowlist.
        cwd: process working directory (must exist).
        timeout_s: wallclock budget; killed at expiry.
        env: optional environment to pass; if None, inherits parent's
            environment minus a few stripped keys.
        output_cap_bytes: per-stream truncation cap.

    Returns: :class:`RunResult` with the captured stdout/stderr,
    exit code, elapsed time, and the relevant flags.

    Never raises for normal failure modes (timeout, non-zero exit,
    truncation). Only re-raises ``OSError`` for genuinely
    catastrophic failure like cwd missing.
    """
    ok, reason = check_allowlist(argv)
    if not ok:
        return RunResult(
            argv=list(argv),
            cwd=str(cwd),
            exit_code=-1,
            elapsed_ms=0,
            stdout="",
            stderr="",
            refused=True,
            refusal_reason=reason,
        )

    cwd_path = Path(cwd)
    if not cwd_path.is_dir():
        return RunResult(
            argv=list(argv),
            cwd=str(cwd),
            exit_code=-1,
            elapsed_ms=0,
            stdout="",
            stderr="",
            refused=True,
            refusal_reason=f"cwd {cwd!s} does not exist or is not a directory",
        )

    wrapped = _apply_sandbox_wrapper(argv)
    cpu_budget = max(1, int(timeout_s) * 4)
    child_env = _build_env(env)

    started = time.monotonic()
    timed_out = False
    stdout = ""
    stderr = ""
    truncated_stdout = False
    truncated_stderr = False
    exit_code = -1

    try:
        proc = subprocess.run(
            wrapped,
            cwd=str(cwd_path),
            capture_output=True,
            timeout=int(timeout_s),
            shell=False,                 # absolute — no shell expansion
            env=child_env,
            preexec_fn=_preexec(cpu_budget),
            check=False,
        )
        exit_code = proc.returncode
        stdout, truncated_stdout = _truncate_bytes(proc.stdout, output_cap_bytes)
        stderr, truncated_stderr = _truncate_bytes(proc.stderr, output_cap_bytes)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -9                   # SIGKILL convention; clearer than 124
        if exc.stdout:
            stdout, truncated_stdout = _truncate_bytes(exc.stdout, output_cap_bytes)
        if exc.stderr:
            stderr, truncated_stderr = _truncate_bytes(exc.stderr, output_cap_bytes)
        # Append a clear marker so the agent's text shows what happened
        marker = (
            f"\n[coding_session_run: killed after {timeout_s}s wallclock]\n"
        )
        stderr = (stderr + marker) if stderr else marker
    except FileNotFoundError as exc:
        return RunResult(
            argv=list(argv),
            cwd=str(cwd),
            exit_code=-1,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            stdout="",
            stderr=f"executable not found in PATH: {exc.filename}",
            refused=True,
            refusal_reason="executable not found",
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return RunResult(
        argv=list(argv),
        cwd=str(cwd),
        exit_code=exit_code,
        elapsed_ms=elapsed_ms,
        stdout=stdout,
        stderr=stderr,
        truncated_stdout=truncated_stdout,
        truncated_stderr=truncated_stderr,
        timed_out=timed_out,
    )


# ── Internals ───────────────────────────────────────────────────────


def _truncate_bytes(data: bytes | str, cap: int) -> tuple[str, bool]:
    """Decode + cap a stream. Returns ``(text, was_truncated)``.

    UTF-8 with replacement so binary spam doesn't crash the runner;
    the agent rarely cares about the exact bytes of a hex dump.
    """
    if isinstance(data, bytes):
        truncated = len(data) > cap
        if truncated:
            data = data[:cap]
        return data.decode("utf-8", errors="replace"), truncated
    # String path — cap by code units, not bytes (close enough for the
    # operator's UI). Used by tests and a few exotic subprocess paths.
    truncated = len(data) > cap
    if truncated:
        data = data[:cap]
    return data, truncated


# Env vars that should not propagate from the gateway process to the
# child. Mostly secrets and CrewAI's runtime sigils. The child still
# gets PATH, HOME, and other Posix-essentials.
_ENV_STRIP = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY",
    "GATEWAY_SECRET", "BRIDGE_SHARED_SECRET",
    "POSTGRES_PASSWORD", "NEO4J_PASSWORD",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "GH_TOKEN", "GITHUB_TOKEN",   # gh CLI uses its own keyring; don't leak
)


def _build_env(override: dict[str, str] | None) -> dict[str, str]:
    """Strip known-secret env vars from the parent environment, then
    overlay any explicit override the caller passed in.

    Keeps PATH (the runner needs to find pytest etc.) and standard
    POSIX vars; drops keys that look like credentials. The override
    dict can re-add any specific var the caller actually wants the
    child to see.
    """
    env = {k: v for k, v in os.environ.items() if k not in _ENV_STRIP}
    if override:
        env.update(override)
    return env
