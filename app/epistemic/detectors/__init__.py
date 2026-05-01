"""Cognitive-bias detectors.

Two phases:

* **Realtime** detectors run on every Claim emission via a hook
  registered with :func:`app.epistemic.registry.register`. They gate
  user-facing output via :mod:`app.epistemic.calibration`. Cheap by
  contract — see ``CALIBRATION_HOOK_BUDGET_MS`` (Phase 1: 50 ms p95).
* **Post-hoc** detectors run inside the Self-Improver's 6-stage loop
  via :func:`app.epistemic.postmortem.synthesize_report`. They tolerate
  more complexity (full ledger graph traversal, cross-task patterns)
  but never gate output. Wired in Phase 4.

This module owns the :class:`Detector` ABC and the per-phase registries.
The actual detector implementations live in
:mod:`app.epistemic.detectors.realtime` and
:mod:`app.epistemic.detectors.posthoc`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

if TYPE_CHECKING:
    from app.epistemic.biases import BiasMatch
    from app.epistemic.ledger import Claim, Ledger


class MatchObserver(Protocol):
    """Observer called by the realtime meta-hook after each batch of
    matches has been persisted. Used for downstream subsystems
    (e.g. :mod:`app.epistemic.affect_bridge`) that want to react to
    bias firings without re-running the detectors.
    """

    def __call__(
        self,
        matches: "list[BiasMatch]",
        claim: "Claim",
        ledger: "Ledger",
    ) -> None: ...


class Detector(ABC):
    """Base class for both realtime and post-hoc bias detectors.

    Subclasses set the class-level ``bias_id`` to the id of the
    :class:`~app.epistemic.biases.BiasDefinition` they detect.

    The :meth:`detect` method is overloaded by argument shape:
      * Realtime: ``detect(ledger, claim=current_claim)`` — incremental.
      * Post-hoc: ``detect(ledger)`` — full scan of the ledger.

    Implementations should yield ``BiasMatch`` instances rather than
    returning a list, so callers can short-circuit on first match if
    they want to (the calibration hook does, on critical severity).
    """

    bias_id: str  #: id of the BiasDefinition this detector matches

    @abstractmethod
    def detect(
        self,
        ledger: "Ledger",
        *,
        claim: "Claim | None" = None,
    ) -> Iterable["BiasMatch"]:
        ...


# ── Per-phase registries ────────────────────────────────────────────

_REALTIME: list[Detector] = []
_POSTHOC: list[Detector] = []


def register_realtime(detector: Detector) -> Detector:
    """Register a detector to run on every claim emission. Returns the
    detector unchanged so this can be used at module scope::

        inference_as_fact = register_realtime(InferenceAsFactDetector())
    """
    if detector not in _REALTIME:
        _REALTIME.append(detector)
    return detector


def register_posthoc(detector: Detector) -> Detector:
    """Register a detector to run during post-mortem analysis (Phase 4)."""
    if detector not in _POSTHOC:
        _POSTHOC.append(detector)
    return detector


def realtime_detectors() -> tuple[Detector, ...]:
    return tuple(_REALTIME)


def posthoc_detectors() -> tuple[Detector, ...]:
    return tuple(_POSTHOC)


# ── Match observers ─────────────────────────────────────────────────
# Called by the realtime meta-hook AFTER matches are persisted.
# Observers are best-effort — exceptions are swallowed at the call site.

_MATCH_OBSERVERS: list[MatchObserver] = []


def register_match_observer(observer: MatchObserver) -> MatchObserver:
    if observer not in _MATCH_OBSERVERS:
        _MATCH_OBSERVERS.append(observer)
    return observer


def match_observers() -> tuple[MatchObserver, ...]:
    return tuple(_MATCH_OBSERVERS)


def _reset_for_tests() -> None:
    """Drop all registered detectors and match observers. Tests only."""
    _REALTIME.clear()
    _POSTHOC.clear()
    _MATCH_OBSERVERS.clear()
