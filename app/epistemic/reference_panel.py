"""Reference panel — replay harness for the realtime detectors.

Loads ``data/reference_panel.yaml`` and runs each scenario through the
live detector registry. Returns a structured :class:`PanelReport` so
the dashboard and CI can both consume it.

Parallels :mod:`app.affect.reference_panel`: each scenario specifies a
synthetic ledger setup, a trigger claim, and the expected bias matches.
A regression in any scenario means the bias library has drifted and
should block promotion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from app.epistemic.biases import BiasMatch
from app.epistemic.detectors import realtime_detectors
from app.epistemic.grounding import set_grounding_provider, _reset_for_tests as _reset_grounding
from app.epistemic.ledger import (
    Claim,
    Evidence,
    Ledger,
    Register,
    VerificationStatus,
    VerifyingAction,
)

logger = logging.getLogger(__name__)


_DEFAULT_PATH = Path(__file__).parent / "data" / "reference_panel.yaml"


# ── Result types ────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    passed: bool
    expected_bias_ids: tuple[str, ...]
    actual_bias_ids: tuple[str, ...]
    diff: str = ""

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "expected_bias_ids": list(self.expected_bias_ids),
            "actual_bias_ids": list(self.actual_bias_ids),
            "diff": self.diff,
        }


@dataclass(frozen=True)
class PanelReport:
    total: int
    passed: int
    failed: int
    results: tuple[ScenarioResult, ...] = field(default_factory=tuple)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "results": [r.as_jsonable() for r in self.results],
        }


# ── Loader ──────────────────────────────────────────────────────────

class ReferencePanelLoadError(RuntimeError):
    """Raised when the YAML reference panel is structurally invalid."""


def load_panel(path: Path = _DEFAULT_PATH) -> list[dict[str, Any]]:
    """Read and structurally validate the reference panel YAML.

    Returns the list of scenarios; the caller drives them through
    :func:`replay_one` or :func:`replay_panel`.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ReferencePanelLoadError(
            f"failed to read reference panel from {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or "scenarios" not in raw:
        raise ReferencePanelLoadError(
            f"reference panel at {path} missing top-level 'scenarios' key"
        )
    scenarios = list(raw["scenarios"])
    seen_ids: set[str] = set()
    for s in scenarios:
        if "id" not in s:
            raise ReferencePanelLoadError(f"scenario missing id: {s}")
        if s["id"] in seen_ids:
            raise ReferencePanelLoadError(f"duplicate scenario id {s['id']!r}")
        seen_ids.add(s["id"])
        if "trigger" not in s:
            raise ReferencePanelLoadError(
                f"scenario {s['id']!r} missing required 'trigger' key"
            )
    return scenarios


# ── Scenario → in-memory objects ────────────────────────────────────

_STATUS_BY_NAME = {s.value: s for s in VerificationStatus}
_REGISTER_BY_NAME = {r.value: r for r in Register}


def _build_claim(spec: dict[str, Any], task_id: str) -> Claim:
    status = _STATUS_BY_NAME[spec["status"]]
    register = _REGISTER_BY_NAME.get(spec.get("register", "internal"), Register.INTERNAL)
    verifier = None
    if "verifying_action" in spec and spec["verifying_action"]:
        va = spec["verifying_action"]
        verifier = VerifyingAction(
            tool=va["tool"],
            args=dict(va.get("args", {})),
            expected_signal=va.get("expected_signal", ""),
            estimated_seconds=float(va.get("estimated_seconds", 1.0)),
        )
    evidence = tuple(
        Evidence(
            kind=e["kind"],
            source_ref=e.get("source_ref", ""),
            excerpt=e.get("excerpt", ""),
            confidence=float(e.get("confidence", 0.5)),
        )
        for e in spec.get("evidence", [])
    )
    return Claim.new(
        task_id=task_id,
        agent_role=spec.get("agent_role", "researcher"),
        statement=spec["statement"],
        status=status,
        register=register,
        evidence=evidence,
        verifying_action=verifier,
        load_bearing=bool(spec.get("load_bearing", False)),
        tags=tuple(spec.get("tags", [])),
    )


# ── Replay ──────────────────────────────────────────────────────────

def replay_one(scenario: dict[str, Any]) -> ScenarioResult:
    """Run a single scenario through the realtime detectors.

    Builds a fresh in-memory ledger, seeds the setup claims (these
    fire detectors but their matches are discarded — only the trigger
    claim's matches count), then emits the trigger and collects
    matches. Any ``grounding`` override is wired before the trigger
    via :func:`set_grounding_provider` and reset afterward.
    """
    scenario_id = scenario["id"]
    task_id = f"refpanel:{scenario_id}"
    ledger = Ledger(task_id=task_id)

    # Wire the grounding override if present.
    grounding = scenario.get("grounding")
    if grounding is not None:
        set_grounding_provider(lambda g=float(grounding): g)

    try:
        # Seed setup claims directly into the ledger, bypassing emit().
        # Setup claims are scaffolding — they should appear in queries
        # like ``ledger.unverified_load_bearing()`` but their detector
        # output is irrelevant. Only the trigger's matches are
        # compared against expected_matches.
        for setup_spec in (scenario.get("setup", {}) or {}).get("claims", []) or []:
            c = _build_claim(setup_spec, task_id)
            ledger._claims[c.claim_id] = c

        trigger_claim = _build_claim(scenario["trigger"], task_id)

        # Run every realtime detector on the trigger; collect matches.
        actual: list[BiasMatch] = []
        for detector in realtime_detectors():
            try:
                actual.extend(detector.detect(ledger, claim=trigger_claim))
            except Exception as exc:
                logger.warning(
                    "reference panel: detector %s raised on scenario %s: %s",
                    detector.__class__.__name__, scenario_id, exc,
                )

        expected_ids = tuple(
            m["bias_id"] for m in (scenario.get("expected_matches", []) or [])
        )
        actual_ids = tuple(m.bias_id for m in actual)

        passed = sorted(expected_ids) == sorted(actual_ids)
        diff = ""
        if not passed:
            extra = sorted(set(actual_ids) - set(expected_ids))
            missing = sorted(set(expected_ids) - set(actual_ids))
            parts = []
            if missing:
                parts.append(f"missing: {missing}")
            if extra:
                parts.append(f"unexpected: {extra}")
            diff = "; ".join(parts)

        return ScenarioResult(
            scenario_id=scenario_id,
            passed=passed,
            expected_bias_ids=expected_ids,
            actual_bias_ids=actual_ids,
            diff=diff,
        )
    finally:
        if grounding is not None:
            _reset_grounding()


def replay_panel(path: Path = _DEFAULT_PATH) -> PanelReport:
    """Run every scenario from the YAML and return an aggregate report."""
    scenarios = load_panel(path)
    results = tuple(replay_one(s) for s in scenarios)
    passed = sum(1 for r in results if r.passed)
    return PanelReport(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )


def iter_scenarios(path: Path = _DEFAULT_PATH) -> Iterator[dict[str, Any]]:
    """Yield scenarios one by one. Used by pytest parametrization."""
    yield from load_panel(path)
