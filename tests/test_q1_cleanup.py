"""PROGRAM §40.4 — Q1 cleanup pass regression tests.

Targets the two remaining items after the full Q1+Q2 audit:
  * Q1#3 pattern-eligibility audit documenting schema-drift handlers
    do NOT qualify for auto-apply
  * Q1#6 Goodhart-Enforcing proposer Global-Workspace publish

The other Q1+Q2 items (1, 2, 4, 5, 7, 8, 9) were already shipped in
§38 / §39 / §40.x and have their own test coverage in those passes.
"""
from __future__ import annotations

from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
#   Q1#3 — Pattern-eligibility audit documentation
# ─────────────────────────────────────────────────────────────────────────


def test_validator_documents_schema_drift_ineligibility():
    """The validator source documents why the schema-drift handlers
    don't qualify for auto-apply — so future Q-N audits don't re-
    propose populating the allowlist with them."""
    src = Path("app/change_requests/validator.py").read_text()
    assert "Q1.4 (PROGRAM §40.4) — pattern-eligibility audit" in src
    assert "schema-drift" in src.lower()
    assert "_handle_numeric_overflow" in src
    assert "_handle_missing_column" in src
    # Both disqualifiers documented:
    assert "migrations/" in src and "_AUTO_APPLY_FORBIDDEN_PREFIXES" in src
    assert "TODO scaffolds" in src or "<TABLE>" in src


def test_auto_apply_doc_has_eligibility_audit_section():
    """The operator-facing doc explains the rationale + what WOULD
    qualify in the future."""
    src = Path("docs/AUTO_APPLY.md").read_text()
    assert "Pattern-eligibility audit (PROGRAM §40.4" in src
    assert "Disqualifier 1" in src
    assert "Disqualifier 2" in src
    assert "What WOULD qualify" in src
    # Concrete forward path documented:
    assert "ADD COLUMN IF NOT EXISTS" in src or "IF NOT EXISTS" in src


def test_validator_allowlists_remain_empty():
    """Sanity: the allowlists must STAY empty. If a future change adds
    a requestor or path, this test fails — forcing review."""
    # Source-level check (no chromadb/pydantic deps).
    src = Path("app/change_requests/validator.py").read_text()
    assert "_AUTO_APPLY_ALLOWED_REQUESTORS: frozenset[str] = frozenset()" in src
    assert "_AUTO_APPLY_ALLOWED_PATHS: tuple[str, ...] = ()" in src


def test_migrations_remains_in_forbidden_prefixes():
    """The categorical forbidden-prefix list must still include
    `migrations/`. Removing it would silently re-enable schema-drift
    auto-apply attempts."""
    src = Path("app/change_requests/validator.py").read_text()
    assert '"migrations/",' in src


# ─────────────────────────────────────────────────────────────────────────
#   Q1#6 — Goodhart-Enforcing proposer GW publish
# ─────────────────────────────────────────────────────────────────────────


def test_goodhart_enforcing_proposer_publishes_to_gw():
    """Source-level: the proposer's run() calls
    _publish_proposal_to_gw after filing the proposal. Without this,
    SubIA never sees the pending substrate-governance event — only
    the operator does (via Signal)."""
    src = Path("app/governance_ratchet/goodhart_enforcing_proposer.py").read_text()
    assert "def _publish_proposal_to_gw" in src
    assert "publish_to_workspace" in src
    # The GW publish must follow the Signal alert in run():
    run_start = src.find("def run(")
    assert run_start > 0
    run_body = src[run_start:run_start + 5000]
    signal_idx = run_body.find("send_signal_alert(body")
    publish_idx = run_body.find("_publish_proposal_to_gw(proposal)")
    assert signal_idx > 0, "Signal alert call missing in run()"
    assert publish_idx > 0, "GW publish call missing in run()"
    assert publish_idx > signal_idx, (
        "GW publish should follow Signal alert so both surfaces fire"
    )


def test_goodhart_enforcing_proposer_gw_publish_is_failure_isolated():
    """The GW publish helper must swallow exceptions — never crash
    the proposer if the workspace is unavailable."""
    src = Path("app/governance_ratchet/goodhart_enforcing_proposer.py").read_text()
    helper_start = src.find("def _publish_proposal_to_gw")
    helper_end = src.find("\ndef ", helper_start + 1)
    if helper_end < 0:
        helper_end = len(src)
    body = src[helper_start:helper_end]
    assert "try:" in body
    assert "except Exception:" in body
    assert "Never raises" in body or "non-fatal" in body or "Best-effort" in body


def test_goodhart_proposer_signal_alert_still_present():
    """Regression guard: the Signal-alert path was already shipped
    pre-Q1.4. We added GW publish on top; the Signal alert must
    NOT have been replaced."""
    src = Path("app/governance_ratchet/goodhart_enforcing_proposer.py").read_text()
    assert "send_signal_alert" in src
    assert "gov_auto_propose:goodhart_enforcing" in src
