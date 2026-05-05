"""Worktree backend implementations.

Two concrete classes implementing ``WorktreeBackend`` from
``manager.py``:

  * :class:`LocalWorktreeBackend` — calls ``git`` directly via
    ``subprocess.run``. Used in tests against a fixture repo, and
    by any deployment that runs the gateway with direct filesystem
    access to the repo. Cross-platform.

  * :class:`BridgeWorktreeBackend` — sends commands through the
    host bridge so they execute on the host. Production default
    when the gateway is in a container that doesn't share the
    host's filesystem.

Both implement the same three operations: ``resolve_ref``,
``create_worktree``, ``remove_worktree``. The manager doesn't
care which one it has; tests use Local + a real fixture, prod
uses Bridge + the real repo on the host.

Operations are best-effort idempotent on cleanup: ``remove_worktree``
will fall back to ``rm -rf`` + ``git worktree prune`` if the
``git worktree remove`` itself fails. Better to leak a directory than
to leave a half-removed entry in the git worktree registry.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Local backend ───────────────────────────────────────────────────


class LocalWorktreeBackend:
    """Runs git locally via subprocess. Production-grade when the
    gateway has direct repo access; primary backend used in tests.

    Parameters:
        repo_root: absolute path to the .git-having repo. All git
            commands run with this as ``cwd``.
        git_executable: defaults to "git"; tests can override to a
            specific path if PATH is uncooperative.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        git_executable: str = "git",
    ) -> None:
        self.repo_root = str(repo_root)
        self.git = git_executable

    # ── Operations ────────────────────────────────────────────────

    def resolve_ref(self, ref: str) -> str:
        """Resolve a ref (branch/tag/sha/HEAD~3/etc) to a full sha.

        Uses ``git rev-parse --verify`` so the operation rejects
        unknown refs with a non-zero exit; we translate that into a
        ``ValueError`` for the manager.
        """
        result = self._run(
            [self.git, "rev-parse", "--verify", ref],
            timeout=15,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise ValueError(
                f"cannot resolve ref {ref!r}: {stderr or 'unknown'}"
            )
        return (result.stdout or "").strip()

    def create_worktree(self, *, worktree_path: str, base_sha: str) -> None:
        """Create a worktree checked out at ``base_sha``."""
        Path(worktree_path).parent.mkdir(parents=True, exist_ok=True)

        # If a directory already exists at the path, refuse — the
        # caller will think a fresh worktree was created when it wasn't.
        if Path(worktree_path).exists():
            raise RuntimeError(
                f"worktree path already exists: {worktree_path}"
            )

        result = self._run(
            [self.git, "worktree", "add", "--detach", worktree_path, base_sha],
            timeout=60,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"git worktree add failed: {stderr or 'unknown'}"
            )

    def remove_worktree(
        self, *, worktree_path: str, force: bool = True,
    ) -> None:
        """Tear down a worktree. Best-effort cleanup with two fallbacks
        if ``git worktree remove`` fails."""
        argv = [self.git, "worktree", "remove"]
        if force:
            argv.append("--force")
        argv.append(worktree_path)

        result = self._run(argv, timeout=30)
        if result.returncode == 0:
            return

        # Fallback 1: brute rm. The git registry might still have the
        # entry, so the prune below cleans that up.
        logger.warning(
            "remove_worktree: git worktree remove failed (%s); falling back to rm -rf",
            (result.stderr or "").strip(),
        )
        try:
            shutil.rmtree(worktree_path, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("remove_worktree: rmtree raised: %s", exc)

        # Fallback 2: prune to drop the orphaned registry entry. If
        # this fails too, we don't really have anywhere else to go.
        prune = self._run([self.git, "worktree", "prune"], timeout=15)
        if prune.returncode != 0:
            raise RuntimeError(
                f"git worktree remove + prune both failed: "
                f"remove={result.stderr!r} prune={prune.stderr!r}"
            )

    # ── Internals ─────────────────────────────────────────────────

    def _run(
        self, argv: list[str], *, timeout: int,
    ) -> subprocess.CompletedProcess:
        """Run ``argv`` with cwd = repo_root. Captures output for
        the caller to inspect."""
        return subprocess.run(
            argv,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )


# ── Bridge backend ──────────────────────────────────────────────────


class BridgeWorktreeBackend:
    """Runs git via the host bridge. Production default in K8s — the
    gateway container talks to a tiny on-host bridge service that
    executes commands against the actual repo.

    The bridge is the same one used by the change-request system to
    apply file writes and run ``gh pr create``. We reuse it so there's
    only one host-side surface to operate.

    Parameters:
        repo_root: absolute path to the repo on the host. All bridge
            commands run with this as ``working_dir``.
        agent_id: identifier passed to ``get_bridge`` for audit
            traceability. Defaults to ``"coding_session"``.
    """

    def __init__(
        self,
        *,
        repo_root: str,
        agent_id: str = "coding_session",
    ) -> None:
        self.repo_root = repo_root
        self.agent_id = agent_id
        self._bridge: Any | None = None

    # ── Operations ────────────────────────────────────────────────

    def resolve_ref(self, ref: str) -> str:
        result = self._bridge_execute(
            ["git", "rev-parse", "--verify", ref],
            timeout=15,
        )
        if result.get("returncode", 0) != 0:
            stderr = (result.get("stderr") or "").strip()
            raise ValueError(
                f"cannot resolve ref {ref!r}: {stderr or 'unknown'}"
            )
        return (result.get("stdout") or "").strip()

    def create_worktree(self, *, worktree_path: str, base_sha: str) -> None:
        # Ensure the parent dir exists. The bridge has a separate
        # filesystem operation for this; we just call git worktree
        # add, which makes the leaf directory itself.
        result = self._bridge_execute(
            ["mkdir", "-p", str(Path(worktree_path).parent)],
            timeout=10,
        )
        if result.get("returncode", 0) != 0:
            raise RuntimeError(
                f"mkdir for worktree parent failed: "
                f"{(result.get('stderr') or '').strip()}"
            )

        result = self._bridge_execute(
            ["git", "worktree", "add", "--detach", worktree_path, base_sha],
            timeout=60,
        )
        if result.get("returncode", 0) != 0:
            raise RuntimeError(
                f"git worktree add failed: "
                f"{(result.get('stderr') or '').strip() or 'unknown'}"
            )

    def remove_worktree(
        self, *, worktree_path: str, force: bool = True,
    ) -> None:
        argv = ["git", "worktree", "remove"]
        if force:
            argv.append("--force")
        argv.append(worktree_path)

        result = self._bridge_execute(argv, timeout=30)
        if result.get("returncode", 0) == 0:
            return

        logger.warning(
            "BridgeWorktreeBackend.remove_worktree: %s; falling back",
            (result.get("stderr") or "").strip(),
        )
        # Brute remove via the bridge
        self._bridge_execute(
            ["rm", "-rf", worktree_path], timeout=15,
        )
        prune = self._bridge_execute(
            ["git", "worktree", "prune"], timeout=15,
        )
        if prune.get("returncode", 0) != 0:
            raise RuntimeError(
                f"git worktree remove + prune both failed: "
                f"remove={result.get('stderr')!r} prune={prune.get('stderr')!r}"
            )

    # ── Internals ─────────────────────────────────────────────────

    def _get_bridge(self) -> Any:
        """Lazy-import + cache the bridge client. Re-raise as
        RuntimeError if the bridge isn't reachable — the manager's
        ``start()`` translates this to a clean error for the agent."""
        if self._bridge is not None:
            return self._bridge
        try:
            from app.bridge_client import get_bridge
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cannot import bridge_client: {exc}"
            ) from exc
        bridge = get_bridge(self.agent_id)
        if bridge is None or not bridge.is_available():
            raise RuntimeError("host bridge unavailable")
        self._bridge = bridge
        return bridge

    def _bridge_execute(
        self, argv: list[str], *, timeout: int,
    ) -> dict[str, Any]:
        """Run a command via the bridge. Returns the bridge response
        dict (``{stdout, stderr, returncode, ...}``); never raises
        for command failures — the caller checks ``returncode``."""
        bridge = self._get_bridge()
        result = bridge.execute(
            argv, working_dir=self.repo_root, timeout=timeout,
        )
        return result or {}
