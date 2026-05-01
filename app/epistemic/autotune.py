"""Autotune for the bias library and verifier registry.

Step 2 of the recommended rollout (`SELF_REFLECTION.md` §6) was a manual
tuning loop: operator reads the bias-feed + override panels, decides
which biases fire too often (false positives) or too rarely (potential
retirement), opens a CODEOWNERS PR with proposed YAML changes. This
module automates the *analysis and proposal* — humans still own the
approve/reject decision via the existing PR review process.

Architecture mirrors the other coupling modules (``grounding``,
``verifier_executor``, ``peer_review``):

* Pluggable strategy thresholds (module constants below) — set in
  Python so the agent cannot widen its own gates by editing YAML.
* Read-only analysis: queries the existing tables, computes metrics,
  emits :class:`TuningProposal` records.
* Persistence: ``epistemic_tuning_proposals`` (migration 032) keyed
  on a content hash so re-running the analyzer doesn't duplicate
  proposals — it superchanges the freshest evidence in place.
* No auto-apply path. Proposals are surfaced to the dashboard with
  the YAML patch text; the operator reviews the patch, accepts it,
  and it goes through the standard CODEOWNERS PR flow.

The analyzer never modifies YAML. The CLI helper
(:func:`apply_proposal`) writes the patch to disk only after explicit
operator confirmation; the actual PR-opening is a separate
:func:`open_pr_for_proposal` call (also explicit) so the system
cannot push code without a human at the keyboard.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping
from uuid import uuid4

logger = logging.getLogger(__name__)


# ── Tuning strategy thresholds (infrastructure-level constants) ─────
# Why constants and not Settings: tuning the *tuner* is a meta-decision
# the agent must not make at runtime. Changing these requires a
# code-review PR — same boundary as the bias library predicates.

#: Default analysis window. Aligns with the soak-window recommendation.
DEFAULT_WINDOW_DAYS: int = 7

#: A bias whose force-proceed override rate exceeds this fraction is a
#: "too strict" candidate — propose downgrading severity or relaxing.
FORCE_PROCEED_RATE_TOO_STRICT: float = 0.30

#: A bias with force-proceed rate below this and adequate fire volume
#: is well-calibrated — no proposal generated.
FORCE_PROCEED_RATE_GOOD: float = 0.05

#: Minimum fire count for severity proposals to fire. Below this we
#: don't have enough data; the proposal is statistical noise.
MIN_FIRES_FOR_SEVERITY_PROPOSAL: int = 20

#: A bias firing fewer than this many times in the window is a
#: retirement candidate (might be obsolete or too narrow to be useful).
RETIREMENT_FIRE_FLOOR: int = 3

#: A verifier shape that matched zero claims in the window is a
#: retirement candidate.
VERIFIER_RETIREMENT_FIRE_FLOOR: int = 0

#: A peer-review allow rate above this on a bias's vetoes means the
#: gate's intuition is consistently wrong — the bias should be relaxed.
PEER_REVIEW_ALLOW_RATE_TOO_AGGRESSIVE: float = 0.50

#: Minimum peer-review count for the allow-rate signal to be valid.
MIN_PEER_REVIEWS_FOR_AGGRESSIVE_PROPOSAL: int = 5


# ── Public types ────────────────────────────────────────────────────

class ProposalKind(StrEnum):
    SEVERITY_DOWNGRADE = "severity_downgrade"      # HIGH → MEDIUM, etc.
    SEVERITY_UPGRADE = "severity_upgrade"          # rare; signaled by post-mortem trends
    RETIREMENT_CANDIDATE = "retirement_candidate"  # low fire rate
    VERIFIER_RETIREMENT = "verifier_retirement"    # verifier never matches


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"      # awaiting operator review
    ACCEPTED = "accepted"      # operator accepted; YAML patch applied (manually or via CLI)
    REJECTED = "rejected"      # operator rejected; pattern noted
    SUPERSEDED = "superseded"  # a fresher proposal replaced this one


_SEVERITY_ORDER = ("low", "medium", "high", "critical")


def _downgrade_severity(s: str) -> str | None:
    try:
        i = _SEVERITY_ORDER.index(s)
    except ValueError:
        return None
    return _SEVERITY_ORDER[i - 1] if i > 0 else None


def _upgrade_severity(s: str) -> str | None:
    try:
        i = _SEVERITY_ORDER.index(s)
    except ValueError:
        return None
    return _SEVERITY_ORDER[i + 1] if i < len(_SEVERITY_ORDER) - 1 else None


@dataclass(frozen=True)
class TuningProposal:
    """A specific tuning suggestion for a single bias or verifier.

    The ``content_hash`` field deduplicates: re-running the analyzer
    on the same evidence produces the same hash; the persistence
    layer upserts on it.
    """

    proposal_id: str
    target_kind: Literal["bias", "verifier"]
    target_id: str                  # bias_id or verifier_id
    kind: ProposalKind
    rationale: str
    metric_evidence: Mapping[str, Any]
    yaml_patch: str                 # unified-diff-like text the operator can review
    confidence: float               # 0.0–1.0; how strong the signal is
    content_hash: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "kind": self.kind.value,
            "rationale": self.rationale,
            "metric_evidence": dict(self.metric_evidence),
            "yaml_patch": self.yaml_patch,
            "confidence": self.confidence,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
        }


def _hash_proposal(
    target_kind: str,
    target_id: str,
    kind: ProposalKind,
    yaml_patch: str,
) -> str:
    """Stable content hash so re-running the analyzer doesn't duplicate."""
    payload = f"{target_kind}|{target_id}|{kind.value}|{yaml_patch}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _new_proposal(
    *,
    target_kind: Literal["bias", "verifier"],
    target_id: str,
    kind: ProposalKind,
    rationale: str,
    metric_evidence: Mapping[str, Any],
    yaml_patch: str,
    confidence: float,
) -> TuningProposal:
    h = _hash_proposal(target_kind, target_id, kind, yaml_patch)
    return TuningProposal(
        proposal_id=f"prop_{uuid4().hex[:12]}",
        target_kind=target_kind,
        target_id=target_id,
        kind=kind,
        rationale=rationale,
        metric_evidence=dict(metric_evidence),
        yaml_patch=yaml_patch,
        confidence=confidence,
        content_hash=h,
    )


# ── Bias-library analyzer ───────────────────────────────────────────

def analyze_bias_library(
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[TuningProposal]:
    """Generate tuning proposals for the eight named biases.

    Walks four data sources over the window:
      * ``epistemic_bias_matches`` — fire counts per bias_id
      * ``epistemic_overrides`` — joined to bias matches via task_id
      * ``epistemic_peer_reviews`` — per-bias triggering decisions
      * ``epistemic_incidents`` — per-bias root-cause counts

    Returns proposals; the caller persists.
    """
    metrics = _compute_bias_metrics(window_days)
    out: list[TuningProposal] = []
    for bias_id, m in metrics.items():
        out.extend(_proposals_for_bias(bias_id, m, window_days))
    return out


def _proposals_for_bias(
    bias_id: str,
    metrics: Mapping[str, Any],
    window_days: int,
) -> Iterable[TuningProposal]:
    fires = int(metrics.get("fires", 0))
    severity = str(metrics.get("severity", "medium"))

    # ── Retirement candidate: low fire volume.
    if fires <= RETIREMENT_FIRE_FLOOR:
        yield _new_proposal(
            target_kind="bias",
            target_id=bias_id,
            kind=ProposalKind.RETIREMENT_CANDIDATE,
            rationale=(
                f"Bias {bias_id!r} fired {fires} time(s) in the last "
                f"{window_days} days (threshold: {RETIREMENT_FIRE_FLOOR}). "
                "Either the pattern is no longer relevant or the "
                "detector has drifted. Consider retiring or rewriting."
            ),
            metric_evidence=dict(metrics),
            yaml_patch=_yaml_patch_retirement(bias_id),
            confidence=0.6 if fires == 0 else 0.4,
        )
        return  # don't also propose severity downgrades on near-empty data

    # Below the volume floor for severity decisions, don't propose.
    if fires < MIN_FIRES_FOR_SEVERITY_PROPOSAL:
        return

    overrides = int(metrics.get("overrides", 0))
    force_proceed = int(metrics.get("force_proceed", 0))
    fp_rate = (force_proceed / overrides) if overrides else 0.0

    # ── Severity downgrade: too many force_proceed overrides.
    if fp_rate >= FORCE_PROCEED_RATE_TOO_STRICT and overrides >= 5:
        next_sev = _downgrade_severity(severity)
        if next_sev is not None:
            yield _new_proposal(
                target_kind="bias",
                target_id=bias_id,
                kind=ProposalKind.SEVERITY_DOWNGRADE,
                rationale=(
                    f"Bias {bias_id!r} has a force-proceed override rate "
                    f"of {fp_rate:.0%} ({force_proceed}/{overrides}) over "
                    f"the last {window_days} days. Threshold for "
                    f"'too strict' is {FORCE_PROCEED_RATE_TOO_STRICT:.0%}. "
                    f"The user is consistently overruling the gate; "
                    f"propose downgrading severity from {severity} to "
                    f"{next_sev}."
                ),
                metric_evidence=dict(metrics),
                yaml_patch=_yaml_patch_severity_change(
                    bias_id, severity, next_sev,
                ),
                confidence=min(0.95, 0.5 + (fp_rate - FORCE_PROCEED_RATE_TOO_STRICT)),
            )

    # ── Aggressive bias: peer-review allowed > N% of vetoes.
    pr_total = int(metrics.get("peer_reviews_triggered", 0))
    pr_allow = int(metrics.get("peer_reviews_allowed", 0))
    pr_allow_rate = (pr_allow / pr_total) if pr_total else 0.0
    if (
        pr_total >= MIN_PEER_REVIEWS_FOR_AGGRESSIVE_PROPOSAL
        and pr_allow_rate >= PEER_REVIEW_ALLOW_RATE_TOO_AGGRESSIVE
    ):
        next_sev = _downgrade_severity(severity)
        if next_sev is not None:
            yield _new_proposal(
                target_kind="bias",
                target_id=bias_id,
                kind=ProposalKind.SEVERITY_DOWNGRADE,
                rationale=(
                    f"Bias {bias_id!r} triggered {pr_total} peer-review "
                    f"escalations; {pr_allow} ({pr_allow_rate:.0%}) "
                    "were ALLOWED by the reviewer — the gate's veto "
                    f"intuition is wrong more than {PEER_REVIEW_ALLOW_RATE_TOO_AGGRESSIVE:.0%} "
                    "of the time. Propose downgrading."
                ),
                metric_evidence=dict(metrics),
                yaml_patch=_yaml_patch_severity_change(
                    bias_id, severity, next_sev,
                ),
                confidence=min(0.9, 0.5 + (pr_allow_rate - PEER_REVIEW_ALLOW_RATE_TOO_AGGRESSIVE)),
            )


def _compute_bias_metrics(window_days: int) -> dict[str, dict[str, Any]]:
    """Aggregate per-bias metrics from the persistence layer.

    Returns ``{bias_id: {fires, overrides, force_proceed,
                          peer_reviews_triggered, peer_reviews_allowed,
                          incidents_as_root_cause, severity}}``.
    """
    from app.epistemic.biases import BIAS_LIBRARY
    from app.epistemic.span_writer import (
        bias_match_counts,
        override_counts_by_bias,
        peer_review_counts_by_bias,
        incident_counts_by_root_cause,
    )

    fires = bias_match_counts(window_days=window_days)
    override_counts = override_counts_by_bias(window_days=window_days)
    pr_counts = peer_review_counts_by_bias(window_days=window_days)
    inc_counts = incident_counts_by_root_cause(window_days=window_days)

    out: dict[str, dict[str, Any]] = {}
    # Cover every bias in the library, even ones that didn't fire —
    # zero-fire biases get retirement proposals.
    for bdef in BIAS_LIBRARY.all():
        ov = override_counts.get(bdef.id, {})
        pr = pr_counts.get(bdef.id, {})
        out[bdef.id] = {
            "severity": bdef.severity.value,
            "fires": int(fires.get(bdef.id, 0)),
            "overrides": int(ov.get("total", 0)),
            "force_proceed": int(ov.get("force_proceed", 0)),
            "peer_reviews_triggered": int(pr.get("total", 0)),
            "peer_reviews_allowed": int(pr.get("allow", 0)),
            "peer_reviews_vetoed": int(pr.get("veto", 0)),
            "incidents_as_root_cause": int(inc_counts.get(bdef.id, 0)),
        }
    return out


# ── Verifier-registry analyzer ──────────────────────────────────────

def analyze_verifier_registry(
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[TuningProposal]:
    """Walk verifier-shape match counts; propose retirement for stale shapes.

    Wired-up additions (proposing NEW shapes from common unmatched
    claim patterns) is harder and lands in a follow-up. For now the
    analyzer is conservative: it only proposes removal of dead shapes.
    """
    from app.epistemic.span_writer import verifier_match_counts
    from app.epistemic.verification import VERIFIER_REGISTRY

    counts = verifier_match_counts(window_days=window_days)
    out: list[TuningProposal] = []
    for shape in VERIFIER_REGISTRY():
        n = int(counts.get(shape.id, 0))
        if n <= VERIFIER_RETIREMENT_FIRE_FLOOR:
            out.append(_new_proposal(
                target_kind="verifier",
                target_id=shape.id,
                kind=ProposalKind.VERIFIER_RETIREMENT,
                rationale=(
                    f"Verifier shape {shape.id!r} matched {n} claim(s) "
                    f"in the last {window_days} days. Either no claims "
                    "have used this shape or its regex has drifted "
                    "from how the agent phrases the claim. Consider "
                    "retiring or rewriting the pattern."
                ),
                metric_evidence={"matches": n, "tool": shape.tool},
                yaml_patch=_yaml_patch_verifier_retirement(shape.id),
                confidence=0.5,
            ))
    return out


# ── YAML patch generators ───────────────────────────────────────────
# These render plain-text unified-diff-style patches the operator
# reviews in the dashboard. The actual file write happens via
# ``apply_proposal`` (CLI), and the PR open via ``open_pr_for_proposal``
# — both explicit user actions, never automatic.

def _yaml_patch_retirement(bias_id: str) -> str:
    return (
        f"# {bias_id} — retirement candidate\n"
        f"# Manually delete this entry from data/biases.yaml after\n"
        f"# confirming the pattern is obsolete:\n"
        f"#\n"
        f"#   - id: {bias_id}\n"
        f"#     name: ...\n"
        f"#     description: ...\n"
        f"#     severity: ...\n"
        f"#     detector: ...\n"
        f"#\n"
        f"# Then remove the corresponding detector class from\n"
        f"# app/epistemic/detectors/{{realtime,posthoc}}.py and the\n"
        f"# scenarios from data/reference_panel.yaml.\n"
    )


def _yaml_patch_severity_change(bias_id: str, old: str, new: str) -> str:
    return (
        f"# {bias_id} — severity adjustment\n"
        f"#\n"
        f"# In data/biases.yaml, change the entry for `{bias_id}`:\n"
        f"#\n"
        f"# - id: {bias_id}\n"
        f"#   ...\n"
        f"# -   severity: {old}\n"
        f"# +   severity: {new}\n"
        f"#\n"
        f"# This may also warrant lowering the `blocking` flag if it\n"
        f"# is currently true and the new severity is below CRITICAL.\n"
    )


def _yaml_patch_verifier_retirement(verifier_id: str) -> str:
    return (
        f"# {verifier_id} — verifier retirement candidate\n"
        f"#\n"
        f"# In data/verifier_registry.yaml, delete the entry for\n"
        f"# `{verifier_id}` (it has not matched any claim in the\n"
        f"# observation window). Keep the entry only if you expect\n"
        f"# new claim shapes that will exercise it soon.\n"
    )


# ── Top-level entry point ───────────────────────────────────────────

def run_full_analysis(
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    persist: bool = True,
) -> list[TuningProposal]:
    """Generate proposals for the bias library AND the verifier registry.

    If ``persist`` is True, calls
    :func:`app.epistemic.span_writer.persist_tuning_proposal` on each
    proposal. Returns the full list either way (so callers can render
    them without needing to re-query).
    """
    proposals = analyze_bias_library(window_days=window_days)
    proposals += analyze_verifier_registry(window_days=window_days)

    if persist:
        try:
            from app.epistemic.span_writer import persist_tuning_proposal
            for p in proposals:
                persist_tuning_proposal(p)
        except Exception as exc:
            logger.debug(
                "epistemic autotune: persist failed (%s); proposals "
                "computed but not stored",
                exc,
            )
    return proposals


# ── Operator actions: apply / open PR ───────────────────────────────
# These are explicit, manual operations — invoked from the CLI or
# dashboard with a confirmed proposal_id. They never run automatically.

class ProposalApplyError(RuntimeError):
    pass


def apply_proposal_to_disk(
    proposal: TuningProposal,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Apply a proposal's YAML change to disk.

    For severity changes: in-place edit of the relevant YAML key.
    For retirements: prints the manual deletion instructions; we do
    not auto-delete entries (the YAML file is human-curated and
    retirement-candidates often reflect missing test coverage rather
    than obsolescence).

    Returns the path of the modified file. Raises
    :class:`ProposalApplyError` if the change can't be safely applied.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    if proposal.target_kind == "bias":
        return _apply_bias_change(proposal, repo_root)
    if proposal.target_kind == "verifier":
        return _apply_verifier_change(proposal, repo_root)
    raise ProposalApplyError(
        f"unknown target_kind: {proposal.target_kind!r}",
    )


def _apply_bias_change(proposal: TuningProposal, repo_root: Path) -> Path:
    if proposal.kind not in (
        ProposalKind.SEVERITY_DOWNGRADE,
        ProposalKind.SEVERITY_UPGRADE,
    ):
        raise ProposalApplyError(
            f"in-place apply not supported for kind={proposal.kind.value!r} "
            "— follow the manual instructions in the YAML patch.",
        )
    path = repo_root / "app" / "epistemic" / "data" / "biases.yaml"
    text = path.read_text()
    new_severity = (
        proposal.metric_evidence.get("severity_proposed")
        or _extract_new_severity_from_patch(proposal.yaml_patch)
    )
    if not new_severity:
        raise ProposalApplyError(
            f"could not determine new severity for {proposal.target_id!r}",
        )
    new_text = _replace_yaml_severity(text, proposal.target_id, new_severity)
    if new_text == text:
        raise ProposalApplyError(
            f"YAML for {proposal.target_id!r} did not change — "
            "perhaps already at the proposed severity",
        )
    path.write_text(new_text)
    return path


def _apply_verifier_change(proposal: TuningProposal, repo_root: Path) -> Path:
    raise ProposalApplyError(
        "verifier-registry retirements are manual: review the entry "
        "in app/epistemic/data/verifier_registry.yaml, then delete it "
        "by hand (and any test scenarios that referenced it).",
    )


def _extract_new_severity_from_patch(patch: str) -> str | None:
    """Pull the post-change severity from the unified-diff text."""
    for line in patch.splitlines():
        line = line.strip()
        if line.startswith("# +   severity:"):
            return line.split(":", 1)[1].strip()
    return None


def _replace_yaml_severity(yaml_text: str, bias_id: str, new_severity: str) -> str:
    """Surgical edit: find the entry for ``bias_id`` and replace its
    ``severity`` line. Preserves comments and indentation.

    Conservative — bails if it can't find a single unambiguous match.
    """
    lines = yaml_text.splitlines(keepends=True)
    in_target = False
    out_lines: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- id:"):
            current_id = stripped.split(":", 1)[1].strip()
            in_target = (current_id == bias_id)
        if in_target and stripped.startswith("severity:") and not replaced:
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}severity: {new_severity}\n")
            replaced = True
            in_target = False  # only first severity in the entry
            continue
        out_lines.append(line)
    return "".join(out_lines)


def open_pr_for_proposal(
    proposal: TuningProposal,
    *,
    repo_root: Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Compose the git+gh commands to open a CODEOWNERS PR.

    By design, returns the command sequence rather than executing
    automatically. The operator runs the commands (or sets
    ``dry_run=False`` after explicit confirmation).

    The PR title cites the proposal_id and rationale; the branch name
    is ``autotune/<content_hash>`` so re-running is idempotent.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    branch = f"autotune/{proposal.content_hash}"
    title = f"epistemic: {proposal.kind.value} for {proposal.target_id}"
    body = (
        f"# Autotune proposal: {proposal.proposal_id}\n\n"
        f"**Target:** `{proposal.target_id}` ({proposal.target_kind})\n"
        f"**Kind:** {proposal.kind.value}\n"
        f"**Confidence:** {proposal.confidence:.2f}\n\n"
        f"## Rationale\n\n{proposal.rationale}\n\n"
        f"## Metric evidence\n\n```json\n"
        f"{json.dumps(dict(proposal.metric_evidence), indent=2)}\n"
        f"```\n\n"
        f"## YAML patch\n\n```\n{proposal.yaml_patch}\n```\n\n"
        f"---\n"
        f"Generated by `app.epistemic.autotune.run_full_analysis()`. "
        f"Manual review required — see "
        f"`crewai-team/docs/SELF_REFLECTION.md` §11 (safety boundaries).\n"
    )

    plan = {
        "branch": branch,
        "title": title,
        "body": body,
        "commands": [
            f"git checkout -b {branch}",
            f"# (manually apply YAML patch — see body above)",
            f"git add app/epistemic/data/",
            f"git commit -m '{title}'",
            f"git push -u origin {branch}",
            f"gh pr create --title '{title}' --body-file -",
        ],
        "executed": False,
    }
    if dry_run:
        return plan
    raise ProposalApplyError(
        "auto-execution of PR commands is intentionally NOT implemented. "
        "Operator must run the commands listed in the dry-run plan."
    )
