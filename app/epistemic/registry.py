"""Claim-hook registry.

Decouples the Ledger from the detector subsystems that observe it. The
Ledger has no knowledge of what runs after a claim is emitted; detectors
self-register via :func:`register`.

This is intentionally a small module: it owns the hook list and the
:class:`ClaimHook` Protocol, and nothing else. Phase 0 ships with no
hooks registered. Phase 1's ``app.epistemic.detectors.realtime`` will
register the realtime detectors here at import time.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.epistemic.ledger import Claim, Ledger

logger = logging.getLogger(__name__)


class ClaimHook(Protocol):
    """Callable invoked once per emitted Claim.

    A hook MUST be cheap (target: < 5 ms p95). The Ledger calls hooks
    synchronously on the user-facing path — anything slower belongs in
    a post-hoc detector run by the Self-Improver loop, not here.

    A hook MUST NOT raise. The Ledger guards against this with a
    try/except, but a misbehaving hook is still a bug worth fixing.
    """

    def __call__(self, claim: "Claim", ledger: "Ledger") -> None: ...


_hooks: list[ClaimHook] = []


def register(hook: ClaimHook) -> ClaimHook:
    """Register a claim hook. Returns the hook unchanged so this can be
    used as a decorator::

        @register
        def my_detector(claim, ledger):
            ...
    """
    if hook in _hooks:
        return hook
    _hooks.append(hook)
    return hook


def unregister(hook: ClaimHook) -> None:
    """Remove a hook. No-op if it was never registered.

    Primarily used by tests to isolate the registry between cases.
    """
    try:
        _hooks.remove(hook)
    except ValueError:
        pass


def claim_hooks() -> tuple[ClaimHook, ...]:
    """Return a snapshot of the current hooks.

    Returns a tuple so callers can iterate without worrying about
    concurrent registration mutating the list mid-loop.
    """
    return tuple(_hooks)


def _reset_for_tests() -> None:
    """Drop all hooks. Tests only — the public API has no use for this."""
    _hooks.clear()
