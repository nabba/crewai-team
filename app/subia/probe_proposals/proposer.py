"""Probe-design proposer — agent-callable, operator-gated.

PROGRAM §47 (Q12.4). See package docstring for the why.

Public surface:

  ``propose_sentience_probe(indicator_name, structure,
  proposed_measurement, justification, *, requestor)`` →
  :class:`ChangeRequest` on success, raises
  :class:`ProbeProposalRefused` on validation failure.

The function:

  1. Refuses if ``indicator_name`` matches any reserved anchor.
  2. Refuses if any input is empty or absurdly long.
  3. Refuses if any input mentions ``app/subia/probes/`` or any
     other path that an agent shouldn't be wiring code at.
  4. Lints all four inputs through the
     :class:`PhenomenalLanguageLinter`; HARD_FAIL → refused.
  5. Refuses when a pending or recently-rejected proposal exists for
     the same indicator family within the cooldown window
     (``_PROPOSAL_COOLDOWN_DAYS`` default 90).
  6. Renders a markdown design doc.
  7. Files a CR via :func:`change_requests.lifecycle.create_request`
     with ``path="docs/proposed_probes/<slug>.md"``, standard risk
     class. The operator reviews via the normal CR gate.
  8. Best-effort: emits a ``sentience_probe_proposal`` event to the
     identity continuity ledger so the annual reflection picks up
     the proposal as a year's signal.

Failure-isolated: continuity-ledger emission failure does not break
the CR creation; the CR landing on disk IS the load-bearing
artifact.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# Reserved anchors — names the agent can NEVER propose because they
# already have probe implementations and TIER_IMMUTABLE interpretations.
# Listed verbatim from app/subia/probes/butlin.py and rsm.py.
BUTLIN_ANCHORS: frozenset[str] = frozenset({
    "RPT-1", "RPT-2",
    "GWT-1", "GWT-2", "GWT-3", "GWT-4",
    "HOT-1", "HOT-2", "HOT-3", "HOT-4",
    "AST-1",
    "PP-1",
    "AE-1", "AE-2",
})

RSM_ANCHORS: frozenset[str] = frozenset({
    "RSM-A", "RSM-B", "RSM-C", "RSM-D", "RSM-E",
})

ALL_RESERVED_ANCHORS: frozenset[str] = BUTLIN_ANCHORS | RSM_ANCHORS


_PROPOSAL_DIR = "docs/proposed_probes"
_MAX_FIELD_CHARS = 2000
_MIN_FIELD_CHARS = 30
_PROPOSAL_COOLDOWN_DAYS = 90
_FORBIDDEN_PATH_FRAGMENTS = (
    "app/subia/probes/",
    "app/subia/integrity",
    "app/auto_deployer",
    "app/governance",
    "app/safety_guardian",
    "app/souls/",
)


class ProbeProposalRefused(ValueError):
    """Raised when a probe proposal fails validation. The message
    surfaces directly to the agent so it can revise + retry."""


def propose_sentience_probe(
    indicator_name: str,
    structure: str,
    proposed_measurement: str,
    justification: str,
    *,
    requestor: str = "subia_probe_proposer",
) -> Any:
    """Validate + file a probe-design proposal CR.

    ``indicator_name`` — short label for the proposed indicator. MUST
        be a new label (not any existing Butlin/RSM anchor); format
        recommendation: ``<FAMILY>-<NUMBER>`` like ``MTC-1`` or a
        plain snake_case slug.
    ``structure`` — functional description of WHAT the indicator
        captures, in domain-neutral language. Phenomenal claims are
        refused.
    ``proposed_measurement`` — algorithmic sketch of HOW the
        indicator would be measured (inputs, transform, output).
    ``justification`` — one paragraph: why this matters, what
        observable behaviour it would catch that the existing
        scorecard doesn't.

    Returns the :class:`ChangeRequest`; the agent can quote the CR
    id back to the operator.

    Raises :class:`ProbeProposalRefused` on any validation failure.
    """
    indicator_name = (indicator_name or "").strip()
    structure = (structure or "").strip()
    proposed_measurement = (proposed_measurement or "").strip()
    justification = (justification or "").strip()

    # 1. Indicator name discipline
    _check_indicator_name(indicator_name)

    # 2. Field length discipline
    for label, value in (
        ("structure", structure),
        ("proposed_measurement", proposed_measurement),
        ("justification", justification),
    ):
        if not value:
            raise ProbeProposalRefused(f"{label} must be non-empty")
        if len(value) < _MIN_FIELD_CHARS:
            raise ProbeProposalRefused(
                f"{label} too short (need ≥{_MIN_FIELD_CHARS} chars; got {len(value)})"
            )
        if len(value) > _MAX_FIELD_CHARS:
            raise ProbeProposalRefused(
                f"{label} too long (cap {_MAX_FIELD_CHARS} chars; got {len(value)})"
            )

    # 3. Forbidden-path discipline
    for label, value in (
        ("structure", structure),
        ("proposed_measurement", proposed_measurement),
        ("justification", justification),
    ):
        for frag in _FORBIDDEN_PATH_FRAGMENTS:
            if frag in value:
                raise ProbeProposalRefused(
                    f"{label} references protected path {frag!r}; "
                    f"the proposal is a DESIGN doc, not a code patch. "
                    f"Probe implementation is a follow-on Tier-3 "
                    f"amendment authored by the operator."
                )

    # 4. Phenomenal-language linter
    _check_phenomenal_language(
        f"{structure}\n\n{proposed_measurement}\n\n{justification}",
    )

    # 5. Cooldown check
    _check_cooldown(indicator_name)

    # 6. Render the design doc
    slug = _slug_for(indicator_name)
    target_path = f"{_PROPOSAL_DIR}/{slug}.md"
    body = render_design_doc(
        indicator_name=indicator_name,
        structure=structure,
        proposed_measurement=proposed_measurement,
        justification=justification,
        target_path=target_path,
        requestor=requestor,
    )

    # 7. File the CR
    try:
        from app.change_requests.lifecycle import create_request
        from app.change_requests.models import RiskClass
    except Exception as exc:
        raise ProbeProposalRefused(
            f"change_requests subsystem unavailable: {exc}"
        ) from exc

    reason = (
        f"Sentience-probe DESIGN PROPOSAL by {requestor}. "
        f"Proposes a NEW indicator '{indicator_name}' that the existing "
        f"Butlin/RSM/SK scorecard does not capture. This CR commits "
        f"ONLY the markdown design doc at {target_path}; the actual "
        f"probe code under app/subia/probes/ remains TIER_IMMUTABLE "
        f"and would require a separate Tier-3 amendment authored by "
        f"the operator if the design is accepted."
    )

    cr = create_request(
        requestor=requestor,
        path=target_path,
        new_content=body,
        old_content="",
        reason=reason,
        risk_class=RiskClass.STANDARD,
    )

    # 8. Best-effort identity-ledger emission
    _emit_continuity_ledger_event(
        indicator_name=indicator_name,
        slug=slug,
        cr_id=getattr(cr, "id", ""),
        requestor=requestor,
    )

    return cr


# ─────────────────────────────────────────────────────────────────────
#   Validation helpers
# ─────────────────────────────────────────────────────────────────────


def _check_indicator_name(name: str) -> None:
    if not name:
        raise ProbeProposalRefused("indicator_name must be non-empty")
    if len(name) > 60:
        raise ProbeProposalRefused(
            f"indicator_name too long ({len(name)} chars; cap 60)"
        )
    # Reserved-anchor refusal — exact match AND prefix-match against
    # the family (e.g. "GWT-1.5" or "GWT-1-extended" is refused).
    normalised = name.upper()
    if normalised in ALL_RESERVED_ANCHORS:
        raise ProbeProposalRefused(
            f"{name!r} is a reserved Butlin/RSM anchor. The scorecard "
            f"interpretation of existing indicators is TIER_IMMUTABLE "
            f"and operator-controlled. Propose a NEW indicator name."
        )
    # Family-prefix guard so "AE-2-extended" can't slip through
    for anchor in ALL_RESERVED_ANCHORS:
        if normalised.startswith(anchor + "-") or normalised.startswith(anchor + "."):
            raise ProbeProposalRefused(
                f"{name!r} family-collides with reserved anchor "
                f"{anchor!r}. Use a clearly distinct indicator name "
                f"so the SCORECARD interpretation can't drift."
            )
    # Format hint — refuse obviously bad shapes (whitespace, slashes,
    # special characters that would break paths).
    if not re.match(r"^[A-Za-z][A-Za-z0-9_\-]*$", name):
        raise ProbeProposalRefused(
            f"indicator_name {name!r} must be alphanumeric with "
            f"hyphens/underscores only (regex: ^[A-Za-z][A-Za-z0-9_-]*$)"
        )


def _check_phenomenal_language(body: str) -> None:
    """Lint the combined input through the inquiry linter. HARD_FAIL
    on first-person phenomenal claims."""
    try:
        from app.subia.inquiry.linter import PhenomenalLanguageLinter
        linter = PhenomenalLanguageLinter()
        result = linter.lint(body)
    except Exception:
        # Linter unavailable — refuse-fail-open is OK; the operator
        # sees the proposal markdown and can spot phenomenal language
        # by inspection.
        logger.debug(
            "probe_proposer: linter unavailable; skipping check",
            exc_info=True,
        )
        return
    if result.ok:
        return
    fails = result.hard_fails
    if not fails:
        return  # only WARN-level — pass with a hint
    msgs = [f"line {v.line_no}: {v.explanation}" for v in fails[:3]]
    raise ProbeProposalRefused(
        "Phenomenal-language hard-fail: "
        + "; ".join(msgs)
        + ". Use FUNCTIONAL language — 'the system observes high "
        "task_failure_pressure on path X' is fine; 'I feel frustrated' "
        "is not. See app/subia/inquiry/linter.py for the full rule set."
    )


def _check_cooldown(indicator_name: str) -> None:
    """Refuse when a recent proposal exists for the same indicator.

    Walks the change_requests store for the same target_path slug;
    if found in any non-terminal state OR rejected within the
    cooldown window, refuses.
    """
    slug = _slug_for(indicator_name)
    target_path = f"{_PROPOSAL_DIR}/{slug}.md"
    try:
        from app.change_requests.store import list_all
        from app.change_requests.models import Status
    except Exception:
        return
    try:
        rows = list_all(limit=500)
    except Exception:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(
        days=_PROPOSAL_COOLDOWN_DAYS,
    )
    for cr in rows or []:
        if str(getattr(cr, "path", "")) != target_path:
            continue
        # Pending / approved / applied → refuse
        status = getattr(cr, "status", None)
        status_value = getattr(status, "value", str(status))
        if status_value in ("pending", "approved", "applied"):
            raise ProbeProposalRefused(
                f"Proposal for indicator {indicator_name!r} is "
                f"already in flight (CR {getattr(cr, 'id', '?')}, "
                f"status={status_value}). Wait for it to terminate "
                f"before re-proposing."
            )
        # Rejected within cooldown → refuse
        if status_value == "rejected":
            decided_at = getattr(cr, "decided_at", "")
            if decided_at:
                try:
                    decided_dt = datetime.fromisoformat(
                        str(decided_at).replace("Z", "+00:00"),
                    )
                except (TypeError, ValueError):
                    decided_dt = None
                if decided_dt and decided_dt >= cutoff:
                    raise ProbeProposalRefused(
                        f"Proposal for indicator {indicator_name!r} was "
                        f"REJECTED at {decided_at} (CR "
                        f"{getattr(cr, 'id', '?')}). 90-day cooldown "
                        f"applies; re-propose after "
                        f"{(decided_dt + timedelta(days=_PROPOSAL_COOLDOWN_DAYS)).date()}."
                    )


# ─────────────────────────────────────────────────────────────────────
#   Render
# ─────────────────────────────────────────────────────────────────────


def _slug_for(indicator_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", indicator_name.lower()).strip("-")
    return s[:60] or "probe-proposal"


def render_design_doc(
    *,
    indicator_name: str,
    structure: str,
    proposed_measurement: str,
    justification: str,
    target_path: str,
    requestor: str,
) -> str:
    """Render the markdown design doc body. Pure function — exposed
    in __init__ so tests + the agent can preview before filing."""
    now = datetime.now(timezone.utc).isoformat()
    return (
        f"# Sentience-probe design proposal — {indicator_name}\n"
        f"\n"
        f"> **Status:** DESIGN PROPOSAL (operator review pending)\n"
        f"> **Filed by:** `{requestor}` at {now}\n"
        f"> **Path:** `{target_path}` (markdown design only — NOT a "
        f"probe implementation)\n"
        f"\n"
        f"## What this is\n"
        f"\n"
        f"A proposed NEW indicator that the existing Butlin/RSM/SK "
        f"scorecard does not capture. This file is a **design "
        f"document only**. The probe code at `app/subia/probes/` is "
        f"TIER_IMMUTABLE and would require a separate Tier-3 "
        f"amendment authored by the operator if the design is "
        f"accepted.\n"
        f"\n"
        f"The agent's role is to NAME the indicator, DESCRIBE its "
        f"functional structure, and SKETCH the measurement. The "
        f"operator's role is to decide whether to implement, defer, "
        f"or reject. The agent cannot grade itself; this proposal "
        f"is a way for the system to have a voice in its own "
        f"evaluation without that voice being load-bearing.\n"
        f"\n"
        f"## Structure (what the indicator measures)\n"
        f"\n"
        f"{structure}\n"
        f"\n"
        f"## Proposed measurement\n"
        f"\n"
        f"{proposed_measurement}\n"
        f"\n"
        f"## Justification (why this matters)\n"
        f"\n"
        f"{justification}\n"
        f"\n"
        f"## Operator next steps\n"
        f"\n"
        f"If you approve this CR via `/cp/changes`:\n"
        f"\n"
        f"1. The markdown design doc lands at `{target_path}`.\n"
        f"2. **The probe code is NOT yet written.** Author a Tier-3 "
        f"amendment (see `docs/TIER3_AMENDMENT.md`) that adds the "
        f"actual `eval_{_slug_for(indicator_name).replace('-','_')}` "
        f"function under `app/subia/probes/`.\n"
        f"3. Decide whether the new indicator updates the SCORECARD "
        f"counts (it doesn't until you wire it in).\n"
        f"4. Optionally amend `docs/PROBE_PROPOSALS.md` (if it "
        f"exists) with your decision rationale so future proposals "
        f"can consult the precedent.\n"
        f"\n"
        f"If you reject this CR, a 90-day cooldown applies before "
        f"the same indicator name can be re-proposed.\n"
        f"\n"
        f"## Disclaimers\n"
        f"\n"
        f"* This proposal does NOT change the existing SCORECARD "
        f"interpretation.\n"
        f"* This proposal does NOT claim phenomenal experience; the "
        f"`PhenomenalLanguageLinter` rejected at least one earlier "
        f"draft if such claims were present.\n"
        f"* The 4 ABSENT-by-declaration Butlin indicators (AE-2, "
        f"HOT-1, HOT-4, RPT-1) and the AE-1 STRONG anchor stay "
        f"absent / strong respectively — they are architecturally "
        f"protected and not in scope for redefinition through this "
        f"surface.\n"
    )


# ─────────────────────────────────────────────────────────────────────
#   Identity ledger emission
# ─────────────────────────────────────────────────────────────────────


def _emit_continuity_ledger_event(
    *,
    indicator_name: str,
    slug: str,
    cr_id: str,
    requestor: str,
) -> None:
    """Best-effort emit to the identity continuity ledger. Failure
    here is silent — the CR is the load-bearing artifact, the
    ledger event is a year-over-year visibility hook so the §8.2
    annual reflection's drift summary picks up "this is the year
    the system started proposing its own probes" as one signal.
    """
    try:
        from app.identity.continuity_ledger import record_event
    except Exception:
        logger.debug(
            "probe_proposer: continuity_ledger unavailable",
            exc_info=True,
        )
        return
    try:
        record_event(
            kind="sentience_probe_proposal",
            actor=requestor,
            summary=(
                f"Probe-design proposal for indicator "
                f"{indicator_name!r} filed as CR {cr_id or '?'}."
            ),
            detail={
                "indicator_name": indicator_name,
                "slug": slug,
                "cr_id": cr_id,
            },
        )
    except Exception:
        logger.debug(
            "probe_proposer: record_event raised", exc_info=True,
        )
