"""Coding-session system — Phase 5.4.

Ephemeral worktree primitive that gives agents the iteration loop
(read / write / run / iterate) before submission. The single escape
hatch from a worktree is ``submit``, which routes through the
existing change-request human gate (Phase 5.3a).

This phase (5.4-a) ships the data layer + lifecycle: models, store,
quotas, manager, reconciler. No tools yet — those are wired in
Phase 5.4-d after the runner (5.4-b) and submit (5.4-c) land.

See ``docs/CODING_SESSIONS.md`` for the full design.

Public API::

    from app.coding_session import (
        CodingSession, Status, SubmitResult,
        Manager, WorktreeBackend, QuotaConfig, QuotaResult,
        QuotaExceeded, IllegalTransition,
        ReconcileReport, run_once,
    )
"""
from app.coding_session.backends import (
    BridgeWorktreeBackend,
    LocalWorktreeBackend,
)
from app.coding_session.manager import (
    IllegalTransition,
    Manager,
    QuotaExceeded,
    WorktreeBackend,
)
from app.coding_session.models import (
    CodingSession,
    Status,
    SubmitResult,
)
from app.coding_session.quotas import (
    DEFAULTS,
    QuotaConfig,
    QuotaResult,
    can_start_session,
    can_write_bytes,
    cap_run_timeout,
)
from app.coding_session.reconciler import (
    ReconcileReport,
    run_once,
)
from app.coding_session.runner import (
    ALLOWLIST,
    RunResult,
    check_allowlist,
    run,
)

__all__ = [
    # models
    "CodingSession", "Status", "SubmitResult",
    # manager
    "Manager", "WorktreeBackend",
    "IllegalTransition", "QuotaExceeded",
    # quotas
    "QuotaConfig", "QuotaResult", "DEFAULTS",
    "can_start_session", "can_write_bytes", "cap_run_timeout",
    # reconciler
    "ReconcileReport", "run_once",
    # runner
    "ALLOWLIST", "RunResult", "check_allowlist", "run",
    # backends
    "LocalWorktreeBackend", "BridgeWorktreeBackend",
]
