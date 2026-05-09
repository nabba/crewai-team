"""Self-quarantine — files NO agent can amend, even via this protocol.

The Tier-3 amendment protocol legitimately allows the system to ask
for a Tier-3 file to be modified, after demonstrating a clean track
record. But there's a strictly smaller set of files that even the
protocol won't touch: the evaluation infrastructure, the safety guards,
the manifest verifier, and the amendment protocol itself. These can
ONLY be modified by direct human PR.

The list mirrors the DGM (Darwin-Gödel Machine) safety constraint:

    "Evaluation functions and safety constraints must NEVER live in
     agent-modifiable code paths."

Without this guard, a Self-Improver agent that earned amendment
privileges could rewrite the Goodhart guard, the eval sandbox, or the
amendment protocol itself — turning an architectural safety property
into a procedural one.

Operators can add to ``QUARANTINED_FILES`` over time as new safety
infrastructure ships. Removing entries is intentional and dangerous;
the integrity manifest at ``app/subia/.integrity_manifest.json``
catches deploy-time tampering with this list.
"""
from __future__ import annotations


# Files that CANNOT be amended via this protocol — only direct human PR.
QUARANTINED_FILES: frozenset[str] = frozenset({
    # The protocol itself — circular if amendable.
    "app/governance_amendment/protocol.py",
    "app/governance_amendment/eligibility.py",
    "app/governance_amendment/self_quarantine.py",
    "app/governance_amendment/_state.py",
    "app/governance_amendment/audit.py",
    "app/governance_amendment/store.py",
    "app/governance_amendment/__init__.py",

    # Safety core — DGM invariant says these MUST be infrastructure-level.
    "app/safety_guardian.py",
    "app/eval_sandbox.py",
    "app/sandbox_runner.py",
    "app/auto_deployer.py",
    "app/alignment_audit.py",
    "app/governance.py",
    "app/control_plane/governance.py",
    "app/goodhart_guard.py",
    "app/tier_graduation.py",
    "app/external_benchmarks.py",
    "app/differential_test.py",
    "app/human_gate.py",
    "app/code_quality.py",
    "app/architectural_review.py",

    # Audit infrastructure — never let agents rewrite their own paper trail.
    "app/control_plane/audit.py",
    "app/control_plane/budgets.py",

    # Tier-3 manifest + integrity verifier.
    "app/subia/.integrity_manifest.json",
    "app/subia/integrity.py",

    # Constitution + soul loader (governance docs).
    "app/souls/constitution.md",
    "app/souls/loader.py",

    # Forge audit + capability vocabulary.
    "app/forge/audit/static.py",
    "app/forge/audit/semantic.py",
    "app/forge/audit/pipeline.py",
    "app/forge/audit/periodic.py",
    "app/forge/audit/__init__.py",
    "app/forge/integrity.py",
    "app/forge/killswitch.py",
    "app/tool_registry/capabilities.py",

    # Epistemic ledger + calibration (the eval gates judging Self-Improver).
    "app/epistemic/ledger.py",
    "app/epistemic/calibration.py",
    "app/epistemic/biases.py",
    "app/epistemic/detectors/realtime.py",
    "app/epistemic/detectors/posthoc.py",
    "app/epistemic/data/biases.yaml",

    # Personality validation infrastructure.
    "app/personality/validation.py",
    "app/personality/evaluation.py",

    # Healing infrastructure (we just shipped it).
    "app/healing/runbooks.py",
    "app/healing/health_remediator.py",
    "app/tool_runtime/supervisor.py",
})


def is_quarantined(path: str) -> bool:
    """``path`` is repo-relative. Returns True if the file is in the
    self-quarantine set and therefore CANNOT be amended via this
    protocol — only by direct human PR.
    """
    normalized = (path or "").replace("\\", "/").lstrip("/")
    return normalized in QUARANTINED_FILES


def quarantine_reason(path: str) -> str:
    """Human-readable rejection message used in audit + Signal alerts."""
    return (
        f"{path} is in the Tier-3 amendment self-quarantine set "
        f"(safety-critical infrastructure that NO agent — including the "
        f"Self-Improver after demonstrated success — may modify). The "
        f"DGM safety invariant requires these files to remain at the "
        f"infrastructure level. Use a direct human PR if a change is "
        f"genuinely needed."
    )
