"""Pluggable verifier executor.

When the pushback handler decides to re-check a foundational claim, it
calls :func:`execute` with the claim's :class:`VerifyingAction`. This
module owns the abstraction; the concrete subprocess/shell wiring lives
elsewhere (Phase 5+ wires a real executor; tests inject fakes).

The default executor returns ``settles=False`` — meaning "no execution
wired, treat as UNVERIFIABLE". This keeps the protocol structurally
correct without coupling Phase 3 to shell-execution machinery.

Mirrors the :mod:`app.epistemic.grounding` pattern: pluggable provider,
no-op default, single coupling point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from app.epistemic.ledger import VerifyingAction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifierResult:
    """Outcome of running a :class:`VerifyingAction`.

    ``settles=True`` means the executor produced exact-answer evidence.
    The caller then reads ``confirms`` to decide whether the original
    claim's polarity is correct (REVERIFIED) or wrong (FALSIFIED).

    ``settles=False`` means the executor couldn't run, returned an
    ambiguous result, or the verifier tool isn't supported. The pushback
    handler treats this as UNVERIFIABLE and surfaces to the user with a
    hedge.
    """

    settles: bool
    confirms: bool
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0


VerifierExecutor = Callable[[VerifyingAction], VerifierResult]


def _default_executor(action: VerifyingAction) -> VerifierResult:
    """No real executor wired — returns ``settles=False``.

    The pushback handler maps this to UNVERIFIABLE. Real execution is
    plugged in via :func:`set_executor` once a sandboxed shell runner
    is wired (Phase 5+).
    """
    logger.debug(
        "epistemic verifier_executor: default no-op for tool %r",
        action.tool,
    )
    return VerifierResult(settles=False, confirms=False)


_executor: VerifierExecutor = _default_executor


def set_executor(executor: VerifierExecutor) -> None:
    """Replace the current executor.

    Called by ``app.tools`` (Phase 5) to wire a sandboxed shell runner.
    The function must respect the verifier's read-only contract — the
    safety boundary around :data:`~app.epistemic.verification.DESTRUCTIVE_TOOL_NAMES`
    is enforced at registry load time, but a malicious executor that
    interprets ``readlink`` as ``rm -rf`` would be a hole. Code review
    on any executor wiring is the operative control.
    """
    global _executor
    _executor = executor


def execute(action: VerifyingAction) -> VerifierResult:
    """Run a VerifyingAction via the configured executor.

    Swallows exceptions: a buggy executor must not break the pushback
    handler's user-facing path. On exception, returns
    ``settles=False`` with the exception text in ``stderr`` so it shows
    in the dashboard.
    """
    try:
        return _executor(action)
    except Exception as exc:
        logger.warning(
            "epistemic verifier_executor: %r raised on %s: %s",
            _executor, action.tool, exc,
        )
        return VerifierResult(
            settles=False, confirms=False, stderr=f"executor raised: {exc}",
        )


def _reset_for_tests() -> None:
    """Restore the default executor. Tests only."""
    global _executor
    _executor = _default_executor
