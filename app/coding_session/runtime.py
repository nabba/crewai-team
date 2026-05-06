"""Runtime singleton for the coding-session system.

The seven ``coding_session_*`` tools all need access to the same
``Manager`` instance — they share state via the store + backend.
This module owns that singleton + the env-driven configuration that
selects which ``WorktreeBackend`` is in use.

Three escape hatches:

  * ``set_manager_for_tests(mgr)`` — tests can inject a Manager
    wired to a ``FakeBackend``.
  * ``CODING_SESSION_BACKEND=local`` — switches to
    ``LocalWorktreeBackend`` (gateway runs git directly; useful in
    dev where the gateway has filesystem access to the repo).
  * ``CODING_SESSION_BACKEND=bridge`` (default) — uses
    ``BridgeWorktreeBackend`` for the production K8s topology.

Other env vars:

  * ``HOST_REPO_PATH`` — absolute path to the repo on the host
    (mirrors the change-request system's existing convention).
    Default: ``/Users/andrus/BotArmy/crewai-team`` to match the dev
    laptop.
  * ``CODING_SESSION_WORKTREE_ROOT`` — where worktrees are created.
    Default: ``/tmp/agent-sessions``. The reconciler eventually
    cleans these up; a cron in production also tmpwatches the dir.
"""
from __future__ import annotations

import logging
import os
import threading

from app.coding_session.backends import (
    BridgeWorktreeBackend,
    LocalWorktreeBackend,
)
from app.coding_session.manager import Manager, WorktreeBackend
from app.coding_session.quotas import QuotaConfig

logger = logging.getLogger(__name__)


_HOST_REPO_PATH_DEFAULT = "/Users/andrus/BotArmy/crewai-team"
_WORKTREE_ROOT_DEFAULT = "/tmp/agent-sessions"


# ── Configuration helpers ───────────────────────────────────────────


def host_repo_path() -> str:
    """The absolute path to the repo on the host. Mirrors the
    change-request system's HOST_REPO_PATH convention."""
    return os.environ.get("HOST_REPO_PATH") or _HOST_REPO_PATH_DEFAULT


def worktree_root() -> str:
    """Where new worktrees are created. The reconciler GCs these
    on TTL/idle; a host-side tmpwatch also clears them weekly."""
    return os.environ.get("CODING_SESSION_WORKTREE_ROOT") or _WORKTREE_ROOT_DEFAULT


def _build_backend() -> WorktreeBackend:
    """Pick a backend based on ``CODING_SESSION_BACKEND``. Default
    'bridge' for production; 'local' for dev / tests with direct
    filesystem access."""
    kind = (os.environ.get("CODING_SESSION_BACKEND") or "bridge").lower()
    repo = host_repo_path()
    if kind == "local":
        logger.info("coding_session.runtime: using LocalWorktreeBackend at %s", repo)
        return LocalWorktreeBackend(repo_root=repo)
    if kind == "bridge":
        logger.info("coding_session.runtime: using BridgeWorktreeBackend at %s", repo)
        return BridgeWorktreeBackend(repo_root=repo)
    logger.warning(
        "coding_session.runtime: unknown CODING_SESSION_BACKEND=%r; "
        "falling back to BridgeWorktreeBackend",
        kind,
    )
    return BridgeWorktreeBackend(repo_root=repo)


# ── Singleton ───────────────────────────────────────────────────────


_MANAGER: Manager | None = None
_LOCK = threading.Lock()


def get_manager() -> Manager:
    """Return the process-wide ``Manager`` singleton, building it
    lazily on first access. Thread-safe via double-checked locking."""
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER
    with _LOCK:
        if _MANAGER is not None:
            return _MANAGER
        _MANAGER = Manager(
            backend=_build_backend(),
            config=QuotaConfig.from_env(),
        )
        return _MANAGER


def set_manager_for_tests(manager: Manager | None) -> None:
    """Inject a test-built Manager (or clear the singleton with
    ``None`` so the next ``get_manager()`` re-builds from env).

    Tests should pair this with ``finally: set_manager_for_tests(None)``
    so the next test starts from a clean slate.
    """
    global _MANAGER
    with _LOCK:
        _MANAGER = manager
