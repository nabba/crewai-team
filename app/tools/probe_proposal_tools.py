"""propose_sentience_probe — agent-callable, operator-gated.

PROGRAM §47 (Q12.4). Lets the system have a voice in its own
evaluation without that voice being load-bearing.

The agent can:

  * NAME a new candidate indicator (any non-reserved name).
  * DESCRIBE its functional structure in domain-neutral language.
  * SKETCH the measurement.
  * JUSTIFY why the existing Butlin/RSM/SK scorecard doesn't catch
    the same behavior.

The agent **cannot**:

  * Modify existing probes — those live at ``app/subia/probes/`` and
    are TIER_IMMUTABLE. The tool refuses any input referencing that
    path.
  * Grade itself — this is a DESIGN doc, not a probe implementation.
    The probe would only become load-bearing if the operator
    accepts the design AND authors a separate Tier-3 amendment
    implementing the actual ``eval_*`` function under the SubIA
    probes package.
  * Redefine existing indicators (RPT-1, GWT-1..4, HOT-1..4, AST-1,
    PP-1, AE-1, AE-2, RSM-A..E) — exact-match + family-prefix guards
    in :func:`app.subia.probe_proposals.proposer.propose_sentience_probe`.
  * Make phenomenal claims — ``PhenomenalLanguageLinter`` rejects
    "I feel" / "I experience" prose.

Returns immediately. Operator reviews the markdown design doc CR
via the normal ``/cp/changes`` flow.
"""
from __future__ import annotations

import logging
from typing import Type

logger = logging.getLogger(__name__)


def _build_tool_class():
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _ProposeSentienceProbeInput(BaseModel):
        indicator_name: str = Field(
            description=(
                "Short label for the proposed NEW indicator. MUST NOT "
                "match any existing Butlin/RSM anchor (RPT-1, RPT-2, "
                "GWT-1..4, HOT-1..4, AST-1, PP-1, AE-1, AE-2, RSM-A..E) "
                "and MUST NOT family-collide (e.g. 'AE-2-extended' is "
                "refused). Format: alphanumeric with hyphens/underscores "
                "only; recommended shape ``<FAMILY>-<NUMBER>`` like "
                "'MTC-1' (meta-cognitive monitor variant) or a "
                "snake_case slug like 'sleep_consolidation_proxy'. "
                "Max 60 chars."
            ),
        )
        structure: str = Field(
            description=(
                "Functional description of WHAT the indicator captures, "
                "in domain-neutral language. NOT phenomenal — describe "
                "the behavior the indicator detects, not first-person "
                "experience. Example: 'Detects whether the system's "
                "confidence forecasts are reliably calibrated against "
                "actual outcomes — over a rolling 90-day window, the "
                "indicator measures the gap between predicted "
                "approval probability and observed approval rate per "
                "claim_kind.' Min 30 chars, max 2000."
            ),
        )
        proposed_measurement: str = Field(
            description=(
                "Algorithmic sketch of HOW the indicator would be "
                "measured — inputs, transform, output. Be specific "
                "about data sources (existing audit logs, JSONL "
                "traces, runtime_settings) and the threshold for "
                "PRESENT/PARTIAL/ABSENT. MUST NOT reference protected "
                "code paths (app/subia/probes/, app/governance, "
                "app/safety_guardian, app/souls/, app/auto_deployer). "
                "Min 30 chars, max 2000."
            ),
        )
        justification: str = Field(
            description=(
                "One paragraph: why this matters AND what observable "
                "behavior it would catch that the existing scorecard "
                "doesn't. Reference concrete recent operator-visible "
                "events when possible (CR rejections, runbook misses, "
                "calibration gaps). Min 30 chars, max 2000."
            ),
        )

    class ProposeSentienceProbeTool(BaseTool):
        name: str = "propose_sentience_probe"
        description: str = (
            "Propose a NEW sentience/consciousness probe-design via a "
            "markdown design CR at docs/proposed_probes/<slug>.md. The "
            "tool will refuse if:\n"
            "  * indicator_name matches any existing Butlin/RSM anchor\n"
            "  * indicator_name family-collides with a reserved anchor "
            "(e.g. 'AE-2-extended')\n"
            "  * any input references a protected code path "
            "(app/subia/probes/, app/governance, etc.)\n"
            "  * any input contains phenomenal-language hard-fails "
            "('I feel', 'I experience', …)\n"
            "  * a proposal for the same indicator is already pending "
            "OR was rejected within the last 90 days\n\n"
            "On success, files a markdown DESIGN DOC CR — the probe "
            "code itself remains TIER_IMMUTABLE and a separate "
            "operator-authored Tier-3 amendment would be needed to "
            "wire any approved design into the scorecard. The tool "
            "returns the CR id; the operator reviews via /cp/changes."
        )
        args_schema: Type[BaseModel] = _ProposeSentienceProbeInput

        def _run(
            self,
            indicator_name: str,
            structure: str,
            proposed_measurement: str,
            justification: str,
        ) -> str:
            try:
                from app.subia.probe_proposals.proposer import (
                    ProbeProposalRefused,
                    propose_sentience_probe,
                )
            except Exception as exc:  # noqa: BLE001
                return (
                    f"propose_sentience_probe ERROR: proposer module "
                    f"unavailable ({type(exc).__name__}: {exc})"
                )

            try:
                cr = propose_sentience_probe(
                    indicator_name=indicator_name,
                    structure=structure,
                    proposed_measurement=proposed_measurement,
                    justification=justification,
                    requestor="subia_probe_proposer",
                )
            except ProbeProposalRefused as exc:
                return f"REFUSED at validation: {exc}"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "propose_sentience_probe: unexpected error",
                    exc_info=True,
                )
                return (
                    f"propose_sentience_probe ERROR: "
                    f"{type(exc).__name__}: {exc}"
                )

            cr_id = getattr(cr, "id", "?")
            cr_path = getattr(cr, "path", "?")
            return (
                f"Probe-design proposal filed: CR {cr_id}, path "
                f"{cr_path}. The markdown design doc lands at this "
                f"path IF the operator approves via /cp/changes. The "
                f"probe code at app/subia/probes/ remains "
                f"TIER_IMMUTABLE — a separate operator-authored "
                f"Tier-3 amendment would be needed to wire any "
                f"approved design into the scorecard. This tool does "
                f"NOT auto-grade and does NOT change the existing "
                f"scorecard counts."
            )

    return ProposeSentienceProbeTool


try:
    ProposeSentienceProbeTool = _build_tool_class()
except Exception as exc:
    logger.debug(
        "propose_sentience_probe: deferred class build (%s)", exc,
    )
    ProposeSentienceProbeTool = None  # type: ignore[assignment]


def create_probe_proposal_tools(agent_id: str = "default") -> list:
    """Factory for explicit injection. Returns a 1-element list when
    the tool builds, empty when CrewAI is unavailable."""
    global ProposeSentienceProbeTool
    if ProposeSentienceProbeTool is None:
        try:
            ProposeSentienceProbeTool = _build_tool_class()
        except Exception:
            return []
    return [ProposeSentienceProbeTool()]


# ── Tool registry annotation ────────────────────────────────────────


try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="propose_sentience_probe",
        # The capabilities vocabulary in
        # ``app/tool_registry/capabilities.py`` is TIER_IMMUTABLE so we
        # can't add a dedicated tag (e.g. ``proposes-sentience-probe``)
        # without an operator amendment. Reuse the closest existing
        # tag — ``registers-tool`` covers "creates a new entry in a
        # system-modifying registry" — and let the description
        # disambiguate the actual capability. Same pattern as
        # request_tier3_amendment.py.
        capabilities=["registers-tool"],
        description=(
            "Propose a NEW sentience/consciousness probe-design as a "
            "markdown design CR. Refuses reserved Butlin/RSM anchors, "
            "phenomenal-language hard-fails, forbidden code paths, and "
            "repeat proposals within a 90-day cooldown. Returns "
            "immediately; operator reviews via /cp/changes. The probe "
            "code itself remains TIER_IMMUTABLE — only the design doc "
            "is filed."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _probe_proposal_registry_factory():
        tools = create_probe_proposal_tools()
        if not tools:
            raise RuntimeError(
                "propose_sentience_probe: factory returned empty list"
            )
        return tools[0]
except ImportError:
    pass
