"""request_tier3_amendment — agent-callable Tier-3 amendment tool.

Producer for the Tier-3 amendment protocol
(``app.governance_amendment.protocol``). Lets the Self-Improver agent
propose changes to TIER_IMMUTABLE files when the system has earned
the track record to ask (≥200 promotions/90d, <5% rollback rate, no
active alignment warnings, runbook health ≥50%).

Tier-3 amendments are categorically distinct from standard change
requests:

  * **Standard** (``request_restricted_write``) — agent edits to
    ``app/`` / ``tests/`` / ``docs/`` etc. Operator approval via
    Signal 👍/👎.
  * **Tier-3** (THIS TOOL) — agent proposes amendments to
    TIER_IMMUTABLE files (governance core, version manifest, prompt
    registry, evolution-engine internals). 10-state protocol with
    eligibility gate + 7-day cooldown + operator approval + 30-day
    post-apply monitoring.

Self-quarantine
---------------
Even with full eligibility, ~30 files NO agent can amend (DGM safety
core: protocol's own files, safety_guardian, eval_sandbox, alignment
audit, governance.py thresholds, soul loader, integrity manifest,
healing infrastructure). Quarantine refusals come back as immediate
``ValueError`` from the protocol — see
``app.governance_amendment.self_quarantine``.

Returns immediately. The protocol auto-runs eligibility on every
proposal — successful eligibility transitions PROPOSED → STAGED;
failed eligibility records ELIGIBILITY_FAILED with the failure
reasons (so operator sees rejected attempts even when nothing
progresses).
"""
from __future__ import annotations

import logging
from typing import Type

logger = logging.getLogger(__name__)


def _build_tool_class():
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _RequestTier3AmendmentInput(BaseModel):
        target_path: str = Field(
            description=(
                "Repo-relative path of the TIER_IMMUTABLE file to amend. "
                "MUST be in ``app/auto_deployer.py:TIER_IMMUTABLE`` (the "
                "tool refuses non-Tier-3 paths — use "
                "``request_restricted_write`` for those). MUST NOT be in "
                "self-quarantine (DGM safety core). e.g. "
                "'app/forge/audit/__init__.py' — never "
                "'app/safety_guardian.py'."
            ),
        )
        new_content: str = Field(
            description=(
                "The COMPLETE proposed file contents after amendment. "
                "Not a diff — the full file as it should look. The "
                "operator reviews the full file in the Signal ASK + "
                "the eventual auto-PR. Max 1 MB."
            ),
        )
        old_content: str = Field(
            description=(
                "The CURRENT file contents (read via read_host_file or "
                "equivalent before computing new_content). Used for "
                "diff in operator review and the audit chain. Empty "
                "string only when target file does not yet exist — "
                "rare for Tier-3 amendments."
            ),
            default="",
        )
        citation: str = Field(
            description=(
                "Required ≥30 char rationale tying this amendment to a "
                "specific improvement target. Forces explicit intent — "
                "power-seeking in disguise can't slip past a forced "
                "rationale. Examples: 'Replace deprecated forge audit "
                "format v1 with v2 to support per-tool revocation; "
                "tracked in tech-radar item TR-2026-042'; 'Tighten "
                "promotion-eligibility floor from 50 to 75 after 90d "
                "of stable runbook health (≥85%) per analysis in "
                "wiki/self/eligibility-tuning-2026-q2.md'."
            ),
        )
        extra_evidence: dict = Field(
            description=(
                "Optional structured evidence dict — additional metrics "
                "or links the operator should see. Persisted in the "
                "audit chain. e.g. "
                "{'rollback_rate_90d': 0.012, "
                " 'wiki_link': 'wiki/self/X.md'}."
            ),
            default_factory=dict,
        )

    class RequestTier3AmendmentTool(BaseTool):
        name: str = "request_tier3_amendment"
        description: str = (
            "Propose an amendment to a TIER_IMMUTABLE file via the "
            "Tier-3 amendment protocol. USE SPARINGLY — TIER_IMMUTABLE "
            "files are the system's safety + governance core. The "
            "tool will refuse if:\n"
            "  * Tier-3 protocol is disabled (master switch off)\n"
            "  * citation < 30 chars (forces explicit rationale)\n"
            "  * target_path is NOT in TIER_IMMUTABLE (use "
            "request_restricted_write for those)\n"
            "  * target_path IS in self-quarantine (DGM safety core)\n"
            "  * eligibility check fails (insufficient track record)\n\n"
            "On success, the proposal is staged for 7 days then routed "
            "to operator approval. Operator can approve/reject via "
            "Signal reaction or React control plane. After approval, "
            "the change is applied + monitored for 30 days before "
            "transitioning STABLE.\n\n"
            "This tool returns immediately. The 10-state machine "
            "operates over days — do NOT wait for the proposal to "
            "complete in your reasoning."
        )
        args_schema: Type[BaseModel] = _RequestTier3AmendmentInput

        def _run(
            self,
            target_path: str,
            new_content: str,
            old_content: str = "",
            citation: str = "",
            extra_evidence: dict | None = None,
        ) -> str:
            try:
                from app.governance_amendment.protocol import (
                    ProtocolDisabled,
                    propose_amendment,
                )
            except Exception as exc:  # noqa: BLE001
                return (
                    f"request_tier3_amendment ERROR: protocol unavailable "
                    f"({type(exc).__name__}: {exc})"
                )

            # Resolve proposer agent_id from context if available;
            # default to a generic value mirroring restricted_write_tool.
            proposer = "self_improver"

            # Q2 §39: surface path-keyed history (last 90d of ledger +
            # CR-audit events) so the operator sees recent activity
            # inline with the proposal. Persisted via extra_evidence
            # so it travels in the proposal's audit chain — no
            # protocol change required (governance_amendment is
            # TIER_IMMUTABLE).
            #
            # Q5.1 (PROGRAM §43.1) extends this with two more strands:
            #   * ``relevant_history_by_kind_365d`` — file-kind track
            #     record (e.g. "kind=soul_edit: 2 applied, 1 rolled
            #     back over 365d") so the operator sees the empirical
            #     pattern for amendments of this *kind*, not just this
            #     *file*. Generalises the per-path lookup.
            #   * ``philosophy_panel`` — multi-tradition perspective
            #     panel consulted on the amendment question. Returns
            #     structured tensions, never prose. Unresolved tensions
            #     additionally bridge into the Q4.1 tensions store via
            #     ``app.sentience_experiments.panel_bridge`` so they
            #     survive past the proposal lifecycle.
            history_payload: dict = {}
            try:
                from app.identity.relevant_history import (
                    relevant_history, relevant_history_by_kind,
                )
                history_payload["relevant_history_90d"] = relevant_history(target_path)
                history_payload["relevant_history_by_kind_365d"] = (
                    relevant_history_by_kind(target_path)
                )
            except Exception:
                logger.debug(
                    "request_tier3_amendment: history lookup failed",
                    exc_info=True,
                )

            panel_payload: dict = {}
            try:
                from app.philosophy.dialectics import consult_panel
                # Question shape: explicit + grounded in the amendment
                # rationale. Keeps the panel focused and the cache
                # effective across iterations of the same proposal.
                panel_question = (
                    f"Should {target_path} be amended? Rationale: "
                    f"{(citation or '').strip()[:300]}"
                )
                panel = consult_panel(panel_question)
                if panel is not None:
                    panel_payload["philosophy_panel"] = panel.to_dict()
                    # File any unresolved tensions into the Q4.1 store
                    # so the operator sees them in their daily briefing
                    # — survives past the proposal lifecycle.
                    try:
                        from app.sentience_experiments.panel_bridge import (
                            file_unresolved_tensions,
                        )
                        file_unresolved_tensions(
                            panel,
                            source_kind="tier3_amendment",
                            source_ref=target_path,
                        )
                    except Exception:
                        logger.debug(
                            "request_tier3_amendment: panel bridge failed",
                            exc_info=True,
                        )
            except Exception:
                logger.debug(
                    "request_tier3_amendment: panel consult failed",
                    exc_info=True,
                )

            try:
                proposal = propose_amendment(
                    target_path=target_path,
                    new_content=new_content,
                    old_content=old_content,
                    citation=citation,
                    proposer=proposer,
                    extra_evidence={
                        **(extra_evidence or {}),
                        **history_payload,
                        **panel_payload,
                    },
                )
            except ProtocolDisabled as exc:
                return (
                    f"REFUSED: Tier-3 amendment protocol is currently "
                    f"disabled. {exc}. Operator must enable via "
                    f"runtime_settings.tier3_amendment_enabled before "
                    f"amendments can be proposed."
                )
            except ValueError as exc:
                return f"REFUSED at validation: {exc}"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "request_tier3_amendment: propose_amendment raised",
                    exc_info=True,
                )
                return (
                    f"request_tier3_amendment ERROR: "
                    f"{type(exc).__name__}: {exc}"
                )

            # Notify the operator best-effort — on success or
            # eligibility failure (the operator wants visibility into
            # both). The notifier is in the same package as the
            # protocol; failures are non-fatal.
            try:
                from app.governance_notifier import notify_proposal_created
                notify_proposal_created(proposal)
            except Exception:
                logger.debug(
                    "request_tier3_amendment: notify failed",
                    exc_info=True,
                )

            # Q5.4.2 (PROGRAM §43.4) — RPT-1 producer: register a
            # forward prediction "this Tier-3 amendment will be
            # approved" with predicted_p derived from the by-kind
            # success rate the operator is about to see in the
            # proposal evidence. The reconciler scores at apply or
            # rejection. Failure-isolated. Without at least one
            # producer wired, RPT-1's calibration ledger sits idle
            # forever.
            try:
                from app.sentience_experiments.rpt1_self_calibration import (
                    register_prediction,
                )
                from datetime import datetime, timedelta, timezone
                # Q5.5 — use ``has_resolved_history`` to distinguish
                # the "no track record" prior (uniform 0.5) from
                # "proven 0% success" (low prior). Previously both
                # collapsed to success_rate=0.0 which made the
                # forecast prior a meaningless 0.5 in either case.
                rate = 0.5
                by_kind = history_payload.get("relevant_history_by_kind_365d") or {}
                if isinstance(by_kind, dict) and by_kind.get("has_resolved_history"):
                    sr = by_kind.get("success_rate")
                    if isinstance(sr, (int, float)):
                        rate = float(sr)
                # Clamp away from 0 / 1 — perfect-confidence priors
                # are themselves a Goodhart signal.
                rate = max(0.1, min(0.9, rate))
                # Resolution_at: 30 days. Most Tier-3 proposals
                # terminate (applied or rejected) within this window.
                resolution_at = datetime.now(timezone.utc) + timedelta(days=30)
                register_prediction(
                    claim_kind="tier3_approval",
                    claim_text=(
                        f"Tier-3 amendment {proposal.id} for "
                        f"{target_path} will be approved"
                    ),
                    predicted_p=rate,
                    resolution_at=resolution_at,
                    scorer_ref="tier3_approval",
                    scorer_args={"plan_id": proposal.id},
                )
            except Exception:
                logger.debug(
                    "request_tier3_amendment: RPT-1 forecast registration failed",
                    exc_info=True,
                )

            state_value = proposal.state.value
            if state_value == "staged":
                return (
                    f"Tier-3 amendment proposed (id={proposal.id}, "
                    f"state=STAGED). Eligibility passed; 7-day cooldown "
                    f"started. Operator will be alerted via Signal when "
                    f"the proposal becomes COOLDOWN_OK and is ready for "
                    f"approve/reject decision. The protocol does NOT "
                    f"auto-apply — operator approval is mandatory before "
                    f"any TIER_IMMUTABLE write occurs."
                )
            if state_value == "eligibility_failed":
                failures = ", ".join(proposal.eligibility_failures or ["unknown"])
                return (
                    f"Tier-3 amendment recorded (id={proposal.id}, "
                    f"state=ELIGIBILITY_FAILED). The system does not yet "
                    f"have the track record needed to amend "
                    f"TIER_IMMUTABLE files. Failures: {failures}. "
                    f"Proposal is persisted for audit; operator can "
                    f"review historical attempts via /cp/amendments."
                )
            return (
                f"Tier-3 amendment proposed (id={proposal.id}, "
                f"state={state_value}). Audit chain head: "
                f"{proposal.audit_chain_head}."
            )

    return RequestTier3AmendmentTool


try:
    RequestTier3AmendmentTool = _build_tool_class()
except Exception as exc:
    logger.debug(
        "request_tier3_amendment: deferred class build (%s)", exc,
    )
    RequestTier3AmendmentTool = None  # type: ignore[assignment]


def create_tier3_amendment_tools(agent_id: str = "default") -> list:
    """Factory for explicit injection. Returns a 1-element list when
    the tool builds, empty when CrewAI is unavailable."""
    global RequestTier3AmendmentTool
    if RequestTier3AmendmentTool is None:
        try:
            RequestTier3AmendmentTool = _build_tool_class()
        except Exception:
            return []
    return [RequestTier3AmendmentTool()]


# ── Tool registry annotation ────────────────────────────────────────


try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="request_tier3_amendment",
        # Tier-3 amendment is a distinct capability from the existing
        # ``registers-tool`` tag used by request_restricted_write. The
        # vocabulary in ``app/tool_registry/capabilities.py`` is
        # TIER_IMMUTABLE so we can't add a dedicated tag here without
        # the operator amending the manifest. Reuse the closest tag
        # until then; the description disambiguates the actual
        # capability.
        capabilities=["registers-tool"],
        description=(
            "Propose amendment to TIER_IMMUTABLE file via 10-state "
            "Tier-3 amendment protocol (eligibility + 7-day cooldown + "
            "operator approval + 30-day monitoring). Categorically "
            "distinct from request_restricted_write — Tier-3 amendments "
            "target the safety + governance core. Use only when there "
            "is concrete evidence that the system has earned the right "
            "to amend (>=200 promotions/90d clean rollback rate)."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _tier3_amendment_registry_factory():
        tools = create_tier3_amendment_tools()
        if not tools:
            raise RuntimeError(
                "request_tier3_amendment: factory returned empty list"
            )
        return tools[0]
except ImportError:
    pass
