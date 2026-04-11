"""
workspace_versioning.py — Git-based workspace versioning with file locking.

Provides:
  - WorkspaceLock: advisory file lock (fcntl.flock) for evolution coordination
  - workspace_commit(): git add + commit all workspace changes
  - workspace_rollback(): restore workspace to a specific commit
  - workspace_log(): recent commit history

Evolution strategies (Autoresearch, Island, Parallel, MAP-Elites) must
acquire WorkspaceLock before modifying workspace files. This prevents
concurrent mutations from corrupting shared state.
"""

from __future__ import annotations

import fcntl
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WORKSPACE = Path("/app/workspace")
LOCK_FILE = WORKSPACE / ".workspace.lock"

# Git config for workspace commits (not the user's git identity)
_GIT_AUTHOR = "AndrusAI Evolution"
_GIT_EMAIL = "evolution@andrusai.local"


class WorkspaceLock:
    """Advisory file lock using fcntl.flock for workspace mutation coordination.

    Usage:
        with WorkspaceLock():
            modify_workspace_files()
            workspace_commit("evolution: improved researcher prompt")
    """

    def __init__(self, timeout_s: int = 30):
        try:
            from app.config import get_settings
            self._timeout = getattr(get_settings(), "workspace_lock_timeout_s", timeout_s)
        except Exception:
            self._timeout = timeout_s
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        """Acquire the workspace lock with timeout."""
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(LOCK_FILE, "w")
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except (IOError, OSError):
                if time.monotonic() > deadline:
                    self._fd.close()
                    self._fd = None
                    raise TimeoutError(
                        f"WorkspaceLock: could not acquire lock within {self._timeout}s"
                    )
                time.sleep(0.5)

    def release(self) -> None:
        """Release the workspace lock."""
        if self._fd:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                self._fd.close()
            except Exception:
                pass
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


def _git(*args, check: bool = False) -> subprocess.CompletedProcess:
    """Run a git command in the workspace directory."""
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **__import__("os").environ,
            "GIT_AUTHOR_NAME": _GIT_AUTHOR,
            "GIT_AUTHOR_EMAIL": _GIT_EMAIL,
            "GIT_COMMITTER_NAME": _GIT_AUTHOR,
            "GIT_COMMITTER_EMAIL": _GIT_EMAIL,
        },
        check=check,
    )


def ensure_workspace_repo() -> bool:
    """Initialize workspace as a git repo if not already. Returns True if initialized."""
    git_dir = WORKSPACE / ".git"
    if git_dir.exists():
        return False
    try:
        _git("init", check=True)
        # Create .gitignore for large/binary files
        gitignore = WORKSPACE / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                "*.db\n*.db-shm\n*.db-wal\n*.sqlite3\n"
                "*.pyc\n__pycache__/\n"
                "sandbox_workspaces/\n"
                "mem0_pgdata/\nmem0_neo4j/\n"
                "logs/\n.sender_key\n"
            )
        _git("add", "-A")
        _git("commit", "-m", "Initial workspace snapshot")
        logger.info("workspace_versioning: initialized git repo at /app/workspace")
        return True
    except Exception as e:
        logger.warning(f"workspace_versioning: git init failed: {e}")
        return False


def workspace_commit(message: str) -> str:
    """Stage and commit all workspace changes.

    Returns commit SHA on success, empty string if nothing to commit or on error.
    Non-fatal — never crashes the caller.
    """
    try:
        ensure_workspace_repo()
        # Stage all changes
        _git("add", "-A")
        # Check if there's anything to commit
        status = _git("status", "--porcelain")
        if not status.stdout.strip():
            return ""  # Nothing to commit
        # Commit
        result = _git("commit", "-m", message[:200])
        if result.returncode == 0:
            # Get SHA
            sha_result = _git("rev-parse", "--short", "HEAD")
            sha = sha_result.stdout.strip()
            logger.info(f"workspace_versioning: committed {sha} — {message[:60]}")
            return sha
        return ""
    except Exception as e:
        logger.debug(f"workspace_versioning: commit failed: {e}")
        return ""


def workspace_rollback(sha: str) -> bool:
    """Restore workspace to a specific commit. Returns True on success."""
    try:
        ensure_workspace_repo()
        result = _git("checkout", sha, "--", ".")
        if result.returncode == 0:
            logger.info(f"workspace_versioning: rolled back to {sha}")
            return True
        logger.warning(f"workspace_versioning: rollback to {sha} failed: {result.stderr[:200]}")
        return False
    except Exception as e:
        logger.warning(f"workspace_versioning: rollback failed: {e}")
        return False


def workspace_log(n: int = 20) -> list[dict]:
    """Recent commit history as structured data."""
    try:
        ensure_workspace_repo()
        result = _git("log", f"--max-count={n}", "--format=%H|%h|%s|%ai")
        if result.returncode != 0:
            return []
        entries = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) == 4:
                entries.append({
                    "sha": parts[0],
                    "short_sha": parts[1],
                    "message": parts[2],
                    "date": parts[3],
                })
        return entries
    except Exception:
        return []
