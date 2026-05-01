"""Cognitive bias library — types, YAML loader, and Phase 2 library.

The vocabulary of named failure modes lives in
``data/biases.yaml`` (versioned, CODEOWNERS-gated). The detector
predicates that *fire* each bias live in
:mod:`app.epistemic.detectors` (Python code, infrastructure-level).
That separation is the safety boundary: an agent that could edit
``biases.yaml`` could only rename or remove vocabulary, not weaken
the predicates.

The in-code ``_INFERENCE_AS_FACT`` definition is kept as a fallback
so the realtime detector's correctness is never tied to a YAML file
being readable on disk: if the YAML loader fails, the library falls
back to the in-code starter and emits a WARNING. Phase 7's
blocking-mode toggle adds a hard fail on YAML-load failure to make
the dependency explicit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import yaml

logger = logging.getLogger(__name__)


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def severity_rank(s: Severity) -> int:
    """Total order on severity for ``max(...)`` selection."""
    return _SEVERITY_RANK[s]


class DetectorPhase(StrEnum):
    REALTIME = "realtime"
    POSTHOC = "posthoc"


@dataclass(frozen=True)
class BiasDefinition:
    """A named cognitive failure mode.

    Phase 1 defines these in code; Phase 2 adds a YAML loader.
    """

    id: str
    name: str
    description: str
    severity: Severity
    phase: DetectorPhase
    corrective_action: str | None = None  # "hedge_or_verify" | "peer_review_required" | ...
    blocking: bool = False                # Phase 7: turn the calibration gate hard


@dataclass(frozen=True)
class BiasMatch:
    """An instance of a bias firing on a specific claim (or set of claims)."""

    bias_id: str
    matched_claim_ids: tuple[str, ...]
    severity: Severity
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "bias_id": self.bias_id,
            "matched_claim_ids": list(self.matched_claim_ids),
            "severity": self.severity.value,
            "detail": dict(self.detail),
        }


# ── Phase 1 starter library ──────────────────────────────────────────

_INFERENCE_AS_FACT = BiasDefinition(
    id="inference_as_fact",
    name="Inference labeled as fact",
    description=(
        "Claim is stated in declarative register but the ledger shows status="
        "inferred and a cheap exact-answer verifier is available. The canonical "
        "failure mode from the April 2026 reference incident: an agent ran an "
        "adjacent observation, drew the wrong inference, asserted it as fact, "
        "and only verified after a second user pushback. This bias detects the "
        "same shape before the assertion reaches the user."
    ),
    severity=Severity.HIGH,
    phase=DetectorPhase.REALTIME,
    corrective_action="hedge_or_verify",
    blocking=False,  # Phase 1 = warn-only; Phase 7 flips to True.
)


class BiasLibrary:
    """Process-wide collection of :class:`BiasDefinition`.

    Immutable from the agent's perspective — the only mutation API is
    :meth:`extend_from_yaml` (Phase 2), which is called from the loader
    at startup and never at runtime.
    """

    def __init__(self, definitions: Mapping[str, BiasDefinition]) -> None:
        self._defs: dict[str, BiasDefinition] = dict(definitions)

    def get(self, bias_id: str) -> BiasDefinition:
        try:
            return self._defs[bias_id]
        except KeyError:
            raise KeyError(f"unknown bias id: {bias_id!r}") from None

    def all(
        self, *, phase: DetectorPhase | None = None,
    ) -> tuple[BiasDefinition, ...]:
        defs = self._defs.values()
        if phase is not None:
            defs = (d for d in defs if d.phase is phase)
        return tuple(defs)

    def __contains__(self, bias_id: object) -> bool:
        return isinstance(bias_id, str) and bias_id in self._defs

    def __len__(self) -> int:
        return len(self._defs)


_DEFAULT_PATH = Path(__file__).parent / "data" / "biases.yaml"


class BiasLibraryLoadError(RuntimeError):
    """Raised when the YAML library is structurally invalid.

    Like :class:`VerifierRegistryLoadError`, the loader refuses partial
    loads — a half-loaded library would hide bugs from the dashboard
    and from post-hoc detectors.
    """


def _definition_from_entry(entry: Mapping[str, Any]) -> BiasDefinition:
    """Validate one YAML entry into a BiasDefinition."""
    required = ("id", "name", "description", "severity", "detector")
    missing = [k for k in required if k not in entry]
    if missing:
        raise BiasLibraryLoadError(
            f"bias entry missing required keys {missing}: id={entry.get('id')!r}"
        )
    try:
        severity = Severity(entry["severity"])
    except ValueError as exc:
        raise BiasLibraryLoadError(
            f"bias {entry['id']!r} has invalid severity {entry['severity']!r}"
        ) from exc
    try:
        phase = DetectorPhase(entry["detector"])
    except ValueError as exc:
        raise BiasLibraryLoadError(
            f"bias {entry['id']!r} has invalid detector phase "
            f"{entry['detector']!r}"
        ) from exc
    return BiasDefinition(
        id=str(entry["id"]),
        name=str(entry["name"]),
        description=str(entry["description"]).strip(),
        severity=severity,
        phase=phase,
        corrective_action=entry.get("corrective_action"),
        blocking=bool(entry.get("blocking", False)),
    )


def _load_yaml(path: Path) -> dict[str, BiasDefinition]:
    """Parse the YAML file into a dict keyed by bias_id."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "biases" not in raw:
        raise BiasLibraryLoadError(
            f"bias library at {path} missing top-level 'biases' key"
        )
    out: dict[str, BiasDefinition] = {}
    for entry in raw["biases"]:
        d = _definition_from_entry(entry)
        if d.id in out:
            raise BiasLibraryLoadError(f"duplicate bias id {d.id!r} in {path}")
        out[d.id] = d
    return out


def _build_default_library() -> BiasLibrary:
    """Load the YAML library with safe in-code fallback.

    Returns the YAML-loaded library if the file parses cleanly.
    Otherwise emits a WARNING and falls back to the in-code starter
    (just ``inference_as_fact``) so the realtime detector's bias_id
    lookup still resolves.
    """
    try:
        defs = _load_yaml(_DEFAULT_PATH)
        # Belt-and-suspenders: the in-code starter must be present in
        # the YAML, since the realtime detector references it by id.
        if _INFERENCE_AS_FACT.id not in defs:
            raise BiasLibraryLoadError(
                f"bias library at {_DEFAULT_PATH} missing required "
                f"{_INFERENCE_AS_FACT.id!r} entry"
            )
        logger.info(
            "epistemic bias library loaded: %d definitions from %s",
            len(defs), _DEFAULT_PATH,
        )
        return BiasLibrary(defs)
    except (OSError, yaml.YAMLError, BiasLibraryLoadError) as exc:
        logger.warning(
            "epistemic bias library YAML load failed (%s); "
            "falling back to in-code starter",
            exc,
        )
        return BiasLibrary({_INFERENCE_AS_FACT.id: _INFERENCE_AS_FACT})


BIAS_LIBRARY = _build_default_library()


def _reload_for_tests(path: Path | None = None) -> None:
    """Reload the library from a custom YAML path. Tests only."""
    global BIAS_LIBRARY
    if path is None:
        BIAS_LIBRARY = _build_default_library()
    else:
        BIAS_LIBRARY = BiasLibrary(_load_yaml(path))
