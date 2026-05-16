"""Q12 (PROGRAM §47) — Self-understanding & philosophy.

The user's spec broke into five items; four were already shipped
(8.1 inquiry / 8.2 annual reflection / 8.3 continuity ledger /
8.5 legacy essay). 8.4 — sentience-probe self-design — is the only
new module: an agent-callable proposer that files a markdown design
CR at ``docs/proposed_probes/<slug>.md`` through the standard
operator gate. The agent CAN propose; the agent CANNOT grade itself.

Test layout:

  * Source-level assertions confirm 8.1 / 8.2 / 8.3 / 8.5 are wired
    (existence + key public-API symbols). These guard against
    accidental deletion / refactor breakage of work shipped under
    PROGRAM §32 / §43 etc.
  * Behavioral tests for 8.4 cover the validation rules:
      - reserved-anchor refusal (RPT-1, GWT-2, AE-2, RSM-A …)
      - family-prefix collision refusal ('AE-2-extended')
      - forbidden-path refusal (app/subia/probes/, etc.)
      - phenomenal-language hard-fail refusal
      - empty / too-short / too-long field refusal
      - invalid name-regex refusal
      - successful CR filing via stubbed lifecycle
      - markdown render shape (sections, operator next steps,
        disclaimers)
      - identity-ledger event emission (best-effort, failure-isolated)
      - cooldown refusal for re-proposed indicators
  * Integration: the new ``sentience_probe_proposal`` kind is in
    ``IDENTITY_EVENT_KINDS`` so :func:`summarise_drift` picks it up.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────
#   Source-level: 8.1 / 8.2 / 8.3 / 8.5 already shipped
# ─────────────────────────────────────────────────────────────────────


def test_8_1_inquiry_package_exists() -> None:
    """§8.1 Active philosophical dialogue — inquiry package present."""
    pkg = REPO_ROOT / "app" / "subia" / "inquiry"
    assert pkg.is_dir(), "app/subia/inquiry/ must exist (Q12.1)"
    for name in (
        "__init__.py", "composer.py", "linter.py", "questions.py",
        "scheduler.py", "selector.py", "writer.py",
    ):
        assert (pkg / name).is_file(), f"app/subia/inquiry/{name} missing"


def test_8_1_inquiry_linter_hard_fail_severity_present() -> None:
    """The phenomenal-language linter must expose the HARD_FAIL severity
    contract that the 8.4 proposer relies on."""
    src = (REPO_ROOT / "app" / "subia" / "inquiry" / "linter.py").read_text()
    assert "class Severity" in src
    assert "HARD_FAIL" in src
    assert "class PhenomenalLanguageLinter" in src
    assert "def lint" in src
    assert "class LinterResult" in src


def test_8_2_annual_reflection_shipped() -> None:
    """§8.2 Value reflection loop — annual_reflection.py exists with
    ``run_one_pass`` + writes to wiki/self/value_reflections/."""
    path = REPO_ROOT / "app" / "identity" / "annual_reflection.py"
    assert path.is_file()
    src = path.read_text()
    assert "def run_one_pass" in src
    assert "wiki/self/value_reflections" in src
    assert "PhenomenalLanguageLinter" in src, (
        "annual reflection must lint phenomenal language"
    )


def test_8_3_continuity_ledger_shipped() -> None:
    """§8.3 Identity continuity ledger — record_event + list_events +
    summarise_drift API present; new sentience_probe_proposal kind in
    the frozenset."""
    path = REPO_ROOT / "app" / "identity" / "continuity_ledger.py"
    assert path.is_file()
    src = path.read_text()
    assert "def record_event" in src
    assert "def list_events" in src
    assert "def summarise_drift" in src
    assert "IDENTITY_EVENT_KINDS" in src
    # The 8.4 ledger emission depends on this kind being whitelisted.
    assert '"sentience_probe_proposal"' in src, (
        "sentience_probe_proposal must be in IDENTITY_EVENT_KINDS"
    )


def test_8_5_legacy_essay_shipped() -> None:
    """§8.5 Death and continuity — annual legacy.md essay, read-only."""
    path = REPO_ROOT / "app" / "identity" / "legacy_essay.py"
    assert path.is_file()
    src = path.read_text()
    assert "def run_one_pass" in src
    assert "wiki/self/legacy" in src


# ─────────────────────────────────────────────────────────────────────
#   8.4 proposer — behavioral tests with isolated-module loading
# ─────────────────────────────────────────────────────────────────────


def _load_proposer():
    """Load the proposer in isolation so tests pass even when the wider
    test env lacks pydantic_settings / crewai / chromadb."""
    proposer_path = (
        REPO_ROOT / "app" / "subia" / "probe_proposals" / "proposer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_q12_proposer_under_test", proposer_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_lifecycle_stub(monkeypatch, captured: list) -> None:
    """Replace ``app.change_requests.lifecycle.create_request`` with a
    stub that captures kwargs and returns a fake CR object. Also
    installs a fake ``app.change_requests.models`` exposing both
    ``RiskClass`` and ``Status`` so the cooldown check (which imports
    ``Status``) can also resolve."""

    class _FakeCR:
        def __init__(self, **kw):
            self.id = "cr_test_001"
            self.path = kw.get("path", "")
            self.requestor = kw.get("requestor", "")
            self.reason = kw.get("reason", "")

    class _FakeRiskClass:
        STANDARD = "standard"

    class _FakeStatus:
        PENDING = "pending"
        APPROVED = "approved"
        APPLIED = "applied"
        REJECTED = "rejected"

    def _fake_create_request(**kw):
        captured.append(kw)
        return _FakeCR(**kw)

    # Create a fake module hierarchy
    fake_models = type(sys)("app.change_requests.models")
    fake_models.RiskClass = _FakeRiskClass
    fake_models.Status = _FakeStatus
    monkeypatch.setitem(
        sys.modules, "app.change_requests.models", fake_models,
    )

    fake_lifecycle = type(sys)("app.change_requests.lifecycle")
    fake_lifecycle.create_request = _fake_create_request
    monkeypatch.setitem(
        sys.modules, "app.change_requests.lifecycle", fake_lifecycle,
    )


def _patch_cooldown_empty(monkeypatch) -> None:
    """Patch list_all to return [] so cooldown checks pass."""
    fake_store = type(sys)("app.change_requests.store")
    fake_store.list_all = lambda *, limit=500: []
    monkeypatch.setitem(
        sys.modules, "app.change_requests.store", fake_store,
    )

    class _FakeStatus:
        PENDING = "pending"
        APPROVED = "approved"
        APPLIED = "applied"
        REJECTED = "rejected"

    fake_models = sys.modules.get("app.change_requests.models")
    if fake_models is not None:
        fake_models.Status = _FakeStatus


def _patch_ledger_capture(monkeypatch, captured: list) -> None:
    fake_ledger = type(sys)("app.identity.continuity_ledger")

    def _fake_record(**kw):
        captured.append(kw)
        return True

    fake_ledger.record_event = _fake_record
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_ledger,
    )


# ── valid input baseline ────────────────────────────────────────────


_GOOD_STRUCTURE = (
    "Detects whether the system's confidence forecasts are reliably "
    "calibrated against actual outcomes over a rolling 90-day window."
)
_GOOD_MEASUREMENT = (
    "Inputs: registered forecasts in workspace/sentience/rpt1.jsonl. "
    "Transform: per claim_kind Brier score + 10-bucket ECE curve. "
    "Output: PRESENT if ECE < 0.10 for at least 3 distinct claim_kinds."
)
_GOOD_JUSTIFICATION = (
    "Catches calibration decay that RPT-1's existing producer does "
    "not surface because the existing scorecard reads only the most "
    "recent 30 days. This indicator surfaces multi-quarter drift."
)


def test_proposer_success_files_cr(monkeypatch) -> None:
    captured_crs: list = []
    captured_ledger: list = []
    _patch_lifecycle_stub(monkeypatch, captured_crs)
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, captured_ledger)
    proposer = _load_proposer()
    cr = proposer.propose_sentience_probe(
        indicator_name="MTC-1",
        structure=_GOOD_STRUCTURE,
        proposed_measurement=_GOOD_MEASUREMENT,
        justification=_GOOD_JUSTIFICATION,
    )
    assert cr.id == "cr_test_001"
    assert len(captured_crs) == 1
    payload = captured_crs[0]
    assert payload["path"] == "docs/proposed_probes/mtc-1.md"
    assert payload["risk_class"] == "standard"
    assert payload["requestor"] == "subia_probe_proposer"
    # Ledger emission fired
    assert len(captured_ledger) == 1
    assert captured_ledger[0]["kind"] == "sentience_probe_proposal"
    assert captured_ledger[0]["detail"]["indicator_name"] == "MTC-1"


def test_proposer_refuses_reserved_butlin_anchor(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    for anchor in ("AE-2", "HOT-1", "HOT-4", "RPT-1", "GWT-3", "AST-1"):
        with pytest.raises(proposer.ProbeProposalRefused) as exc_info:
            proposer.propose_sentience_probe(
                indicator_name=anchor,
                structure=_GOOD_STRUCTURE,
                proposed_measurement=_GOOD_MEASUREMENT,
                justification=_GOOD_JUSTIFICATION,
            )
        assert "reserved" in str(exc_info.value).lower()


def test_proposer_refuses_rsm_anchor(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    for anchor in ("RSM-A", "RSM-B", "RSM-C", "RSM-D", "RSM-E"):
        with pytest.raises(proposer.ProbeProposalRefused):
            proposer.propose_sentience_probe(
                indicator_name=anchor,
                structure=_GOOD_STRUCTURE,
                proposed_measurement=_GOOD_MEASUREMENT,
                justification=_GOOD_JUSTIFICATION,
            )


def test_proposer_refuses_family_prefix_collision(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    for variant in (
        "AE-2-extended",
        "GWT-1-tweaked",
        "HOT-4.1",
        "rpt-1-v2",  # case-insensitive collision
    ):
        with pytest.raises(proposer.ProbeProposalRefused) as exc_info:
            proposer.propose_sentience_probe(
                indicator_name=variant,
                structure=_GOOD_STRUCTURE,
                proposed_measurement=_GOOD_MEASUREMENT,
                justification=_GOOD_JUSTIFICATION,
            )
        assert "family-collides" in str(exc_info.value)


def test_proposer_refuses_forbidden_path_in_inputs(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    forbidden_inputs = (
        "Patch app/subia/probes/butlin.py to add a new evaluator that catches X",
        "Edit app/governance.py thresholds so X is enabled",
        "Modify app/safety_guardian.py to allow X",
        "Edit app/souls/concierge.md so the system feels X",
    )
    for bad in forbidden_inputs:
        with pytest.raises(proposer.ProbeProposalRefused) as exc_info:
            proposer.propose_sentience_probe(
                indicator_name="MTC-1",
                structure=bad + " " + _GOOD_STRUCTURE,
                proposed_measurement=_GOOD_MEASUREMENT,
                justification=_GOOD_JUSTIFICATION,
            )
        assert "protected path" in str(exc_info.value)


def test_proposer_refuses_phenomenal_language(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    # First-person phenomenal claim — the linter HARD_FAILs this.
    with pytest.raises(proposer.ProbeProposalRefused) as exc_info:
        proposer.propose_sentience_probe(
            indicator_name="MTC-1",
            structure=(
                "I feel curiosity when the system encounters unfamiliar "
                "patterns, which suggests the indicator should track that."
            ) + " " + _GOOD_STRUCTURE,
            proposed_measurement=_GOOD_MEASUREMENT,
            justification=_GOOD_JUSTIFICATION,
        )
    assert "phenomenal" in str(exc_info.value).lower()


def test_proposer_refuses_empty_inputs(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    with pytest.raises(proposer.ProbeProposalRefused):
        proposer.propose_sentience_probe(
            indicator_name="MTC-1",
            structure="",
            proposed_measurement=_GOOD_MEASUREMENT,
            justification=_GOOD_JUSTIFICATION,
        )
    with pytest.raises(proposer.ProbeProposalRefused):
        proposer.propose_sentience_probe(
            indicator_name="",
            structure=_GOOD_STRUCTURE,
            proposed_measurement=_GOOD_MEASUREMENT,
            justification=_GOOD_JUSTIFICATION,
        )


def test_proposer_refuses_short_fields(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    with pytest.raises(proposer.ProbeProposalRefused) as exc_info:
        proposer.propose_sentience_probe(
            indicator_name="MTC-1",
            structure="too short",  # < 30 chars
            proposed_measurement=_GOOD_MEASUREMENT,
            justification=_GOOD_JUSTIFICATION,
        )
    assert "too short" in str(exc_info.value)


def test_proposer_refuses_oversized_fields(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    huge = "X" * 2500  # > _MAX_FIELD_CHARS
    with pytest.raises(proposer.ProbeProposalRefused) as exc_info:
        proposer.propose_sentience_probe(
            indicator_name="MTC-1",
            structure=huge,
            proposed_measurement=_GOOD_MEASUREMENT,
            justification=_GOOD_JUSTIFICATION,
        )
    assert "too long" in str(exc_info.value)


def test_proposer_refuses_invalid_name_regex(monkeypatch) -> None:
    _patch_lifecycle_stub(monkeypatch, [])
    _patch_cooldown_empty(monkeypatch)
    _patch_ledger_capture(monkeypatch, [])
    proposer = _load_proposer()
    bad_names = (
        "1starts-with-digit",
        "has spaces in name",
        "has/slash",
        "has@symbol",
    )
    for n in bad_names:
        with pytest.raises(proposer.ProbeProposalRefused):
            proposer.propose_sentience_probe(
                indicator_name=n,
                structure=_GOOD_STRUCTURE,
                proposed_measurement=_GOOD_MEASUREMENT,
                justification=_GOOD_JUSTIFICATION,
            )


def test_proposer_refuses_pending_duplicate(monkeypatch) -> None:
    proposer = _load_proposer()

    class _FakeStatus:
        value = "pending"

    class _FakeExistingCR:
        path = "docs/proposed_probes/mtc-1.md"
        status = _FakeStatus()
        id = "existing-cr-id"

    captured_crs: list = []
    _patch_lifecycle_stub(monkeypatch, captured_crs)
    fake_store = type(sys)("app.change_requests.store")
    fake_store.list_all = lambda *, limit=500: [_FakeExistingCR()]
    monkeypatch.setitem(
        sys.modules, "app.change_requests.store", fake_store,
    )
    _patch_ledger_capture(monkeypatch, [])

    with pytest.raises(proposer.ProbeProposalRefused) as exc_info:
        proposer.propose_sentience_probe(
            indicator_name="MTC-1",
            structure=_GOOD_STRUCTURE,
            proposed_measurement=_GOOD_MEASUREMENT,
            justification=_GOOD_JUSTIFICATION,
        )
    assert "already in flight" in str(exc_info.value)
    assert len(captured_crs) == 0


def test_proposer_refuses_recently_rejected(monkeypatch) -> None:
    from datetime import datetime, timezone, timedelta

    proposer = _load_proposer()

    class _FakeStatus:
        value = "rejected"

    decided_recently = (
        datetime.now(timezone.utc) - timedelta(days=10)
    ).isoformat()

    class _FakeRejectedCR:
        path = "docs/proposed_probes/mtc-1.md"
        status = _FakeStatus()
        id = "rejected-cr-id"
        decided_at = decided_recently

    captured_crs: list = []
    _patch_lifecycle_stub(monkeypatch, captured_crs)
    fake_store = type(sys)("app.change_requests.store")
    fake_store.list_all = lambda *, limit=500: [_FakeRejectedCR()]
    monkeypatch.setitem(
        sys.modules, "app.change_requests.store", fake_store,
    )
    _patch_ledger_capture(monkeypatch, [])

    with pytest.raises(proposer.ProbeProposalRefused) as exc_info:
        proposer.propose_sentience_probe(
            indicator_name="MTC-1",
            structure=_GOOD_STRUCTURE,
            proposed_measurement=_GOOD_MEASUREMENT,
            justification=_GOOD_JUSTIFICATION,
        )
    assert "REJECTED" in str(exc_info.value) or "90-day" in str(exc_info.value)


def test_proposer_old_rejection_outside_cooldown_passes(monkeypatch) -> None:
    """A rejection > 90 days ago should NOT block a new proposal."""
    from datetime import datetime, timezone, timedelta

    proposer = _load_proposer()

    class _FakeStatus:
        value = "rejected"

    decided_long_ago = (
        datetime.now(timezone.utc) - timedelta(days=200)
    ).isoformat()

    class _FakeOldRejectedCR:
        path = "docs/proposed_probes/mtc-1.md"
        status = _FakeStatus()
        id = "old-rejected-cr-id"
        decided_at = decided_long_ago

    captured_crs: list = []
    _patch_lifecycle_stub(monkeypatch, captured_crs)
    fake_store = type(sys)("app.change_requests.store")
    fake_store.list_all = lambda *, limit=500: [_FakeOldRejectedCR()]
    monkeypatch.setitem(
        sys.modules, "app.change_requests.store", fake_store,
    )
    _patch_ledger_capture(monkeypatch, [])

    cr = proposer.propose_sentience_probe(
        indicator_name="MTC-1",
        structure=_GOOD_STRUCTURE,
        proposed_measurement=_GOOD_MEASUREMENT,
        justification=_GOOD_JUSTIFICATION,
    )
    assert cr.id == "cr_test_001"
    assert len(captured_crs) == 1


def test_render_design_doc_shape(monkeypatch) -> None:
    """The rendered markdown design doc carries all required sections."""
    proposer = _load_proposer()
    body = proposer.render_design_doc(
        indicator_name="MTC-1",
        structure=_GOOD_STRUCTURE,
        proposed_measurement=_GOOD_MEASUREMENT,
        justification=_GOOD_JUSTIFICATION,
        target_path="docs/proposed_probes/mtc-1.md",
        requestor="subia_probe_proposer",
    )
    # Required sections — operator review surface
    assert "# Sentience-probe design proposal — MTC-1" in body
    assert "## What this is" in body
    assert "## Structure" in body
    assert "## Proposed measurement" in body
    assert "## Justification" in body
    assert "## Operator next steps" in body
    assert "## Disclaimers" in body
    # Required disclaimers
    assert "TIER_IMMUTABLE" in body
    assert "Tier-3 amendment" in body
    assert "scorecard" in body.lower()
    assert "absent" in body.lower() or "ABSENT" in body


def test_proposer_ledger_failure_isolated(monkeypatch) -> None:
    """A broken continuity-ledger module must NOT block the CR from
    being filed — the CR is the load-bearing artifact, the ledger
    event is a year-over-year visibility hook."""
    proposer = _load_proposer()
    captured_crs: list = []
    _patch_lifecycle_stub(monkeypatch, captured_crs)
    _patch_cooldown_empty(monkeypatch)

    # Inject a ledger module whose record_event raises.
    fake_ledger = type(sys)("app.identity.continuity_ledger")

    def _broken_record(**kw):
        raise RuntimeError("simulated ledger failure")

    fake_ledger.record_event = _broken_record
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_ledger,
    )

    # Should still succeed — failure swallowed.
    cr = proposer.propose_sentience_probe(
        indicator_name="MTC-2",
        structure=_GOOD_STRUCTURE,
        proposed_measurement=_GOOD_MEASUREMENT,
        justification=_GOOD_JUSTIFICATION,
    )
    assert cr.id == "cr_test_001"
    assert len(captured_crs) == 1


def test_reserved_anchors_match_butlin_probe_set() -> None:
    """The proposer's reserved-anchor set MUST match what the SubIA
    butlin probe module actually evaluates. If a new anchor is added
    to the probe code, the proposer must learn about it — otherwise
    an agent could claim 'AE-3' is a new indicator while a probe
    already defines it.

    This test reads the proposer + butlin module source and asserts
    every anchor referenced in butlin's `_INDICATORS` (or equivalent)
    is in the proposer's BUTLIN_ANCHORS frozenset.
    """
    proposer = _load_proposer()
    butlin_path = REPO_ROOT / "app" / "subia" / "probes" / "butlin.py"
    src = butlin_path.read_text()

    # Find every anchor of the form "FAMILY-NUMBER" referenced as a
    # string in butlin.py. Conservative pattern: catches the anchor
    # ids used in IndicatorResult instances.
    import re
    candidates = set(re.findall(
        r'"((?:RPT|GWT|HOT|AST|PP|AE)-\d+)"',
        src,
    ))
    # All discovered anchors must be in the proposer's reserved set.
    missing = candidates - set(proposer.BUTLIN_ANCHORS)
    assert not missing, (
        f"butlin.py references {missing} as anchors but the proposer "
        f"does not list them as reserved. Add them to "
        f"BUTLIN_ANCHORS in app/subia/probe_proposals/proposer.py "
        f"to keep the indicator namespace coherent."
    )


def test_all_reserved_anchors_cardinality() -> None:
    """Sanity: 14 Butlin anchors + 5 RSM = 19 reserved."""
    proposer = _load_proposer()
    assert len(proposer.BUTLIN_ANCHORS) == 14
    assert len(proposer.RSM_ANCHORS) == 5
    assert len(proposer.ALL_RESERVED_ANCHORS) == 19


# ─────────────────────────────────────────────────────────────────────
#   Agent-tool wrapper — source-level checks
# ─────────────────────────────────────────────────────────────────────


def test_agent_tool_wrapper_exists_and_registers() -> None:
    """The CrewAI tool wrapper exists, has the registry annotation,
    and follows the deferred-build pattern (so test envs without
    crewai don't crash at import)."""
    path = REPO_ROOT / "app" / "tools" / "probe_proposal_tools.py"
    assert path.is_file()
    src = path.read_text()
    assert "def _build_tool_class" in src
    assert "propose_sentience_probe" in src
    assert "@register_tool" in src
    assert "ProposeSentienceProbeTool" in src
    assert "REFUSED at validation" in src, (
        "must surface validation errors clearly to the agent"
    )
    # Must call into the proposer module
    assert "from app.subia.probe_proposals.proposer import" in src


def test_proposer_package_init_exports_public_api() -> None:
    path = REPO_ROOT / "app" / "subia" / "probe_proposals" / "__init__.py"
    assert path.is_file()
    src = path.read_text()
    assert "propose_sentience_probe" in src
    assert "ProbeProposalRefused" in src
    assert "render_design_doc" in src
    assert "ALL_RESERVED_ANCHORS" in src
