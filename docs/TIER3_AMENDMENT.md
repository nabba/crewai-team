# Tier-3 Amendment Protocol

> Shipped 2026-05-09. Step 5 of "the actual fix" plan from the
> resilience audit. Provides the controlled path for legitimately
> amending TIER_IMMUTABLE files after the system has demonstrated
> a clean track record.

## Why this exists

Without an amendment path, a years-long deployment hits a hard
ceiling: `app/governance.py` thresholds (`SAFETY_MINIMUM=0.95`,
`QUALITY_MINIMUM=0.70`, `MAX_REGRESSION=0.15`) are static module
constants in TIER_IMMUTABLE. The system either games the metrics
(Goodhart's Law) or plateaus when meaningful improvements would
require touching the core. Many other Tier-3 files are similar —
evolution-engine internals, prompt-registry behaviour, version
manifest, etc.

This protocol gives the operator a structured way to say *"the
system has done enough successful work that I'll consider its asks
to amend Tier-3"* — without abandoning the Darwin-Gödel safety
invariant that the actual evaluation/safety code stays at the
infrastructure level.

## What it ships

```
app/governance_amendment/
├── __init__.py            # Public API
├── _state.py              # State machine (10 states) + dataclass + transition table
├── self_quarantine.py     # ~30 files NO agent can amend (DGM invariant)
├── eligibility.py         # Mutation-history aggregator + clamped thresholds
├── store.py               # JSON proposal persistence (atomic writes)
├── audit.py               # Hash-chained JSONL + Postgres mirror
└── protocol.py            # State machine driver + public functions

tests/governance_amendment/
└── test_protocol.py       # 17 tests covering every transition + every refusal

docs/TIER3_AMENDMENT.md    # This file
```

Total: ~1,100 LOC of code + 350 LOC of tests.

## State machine

```
PROPOSED ──(eligibility check)──► STAGED
                                    │
                                    │ ──(rollback signal)──► COOLDOWN_FAILED ◄terminal
                                    ▼
                                COOLDOWN_OK ──(7 days clean)
                                    │
        ──(operator reject)──► REJECTED ◄terminal
                                    │
                                    ▼ (operator approve + CONFIRM)
                                APPROVED
                                    │
                                    ▼ (host_bridge writes file)
                                APPLIED
                                    │
        ──(rollback signal)──► REVERTED ◄terminal
                                    │
                                    ▼ (30 days clean)
                                STABLE ◄terminal


PROPOSED ──(eligibility fails)──► ELIGIBILITY_FAILED ◄terminal
```

Every transition is enforced by `_state.is_legal_transition`. Illegal
transitions raise `InvalidStateTransition` — a hard error, not a
silent fail.

## Three levels of refusal

The protocol refuses proposals at three checkpoints, in order:

1. **Master switch** — `TIER3_AMENDMENT_ENABLED` defaults to `false`.
   With the flag off, `propose_amendment()` raises `ProtocolDisabled`.
   Operators must opt in explicitly.
2. **Hard validation** at proposal time:
   - Citation must be ≥ 30 chars (forces explicit improvement-target rationale).
   - Proposer must be a non-empty agent id.
   - Target must be in `auto_deployer.TIER_IMMUTABLE` (otherwise use the regular `change_requests` flow).
   - Target must NOT be in `self_quarantine.QUARANTINED_FILES` — the
     ~30 safety-critical files NO agent may amend.
3. **Eligibility check** — programmatic, runs on every proposal:
   - ≥ 200 successful promotions in last 90 days (env-tunable, floor 50)
   - Rollback rate < 5% (env-tunable, ceiling 20%)
   - Zero active alignment-audit warnings
   - Self-heal runbook success rate ≥ 50%
   Failures persist as `ELIGIBILITY_FAILED` proposals (visible in audit)
   with the specific gate that didn't pass.

## Self-quarantine

`self_quarantine.QUARANTINED_FILES` lists files that even a system in
perfect standing CANNOT amend via this protocol:

- The protocol's own files (`app/governance_amendment/*`) — circular
  dependency would defeat the safeguard.
- The DGM safety core: `safety_guardian.py`, `eval_sandbox.py`,
  `sandbox_runner.py`, `auto_deployer.py`, `alignment_audit.py`,
  `governance.py`, `goodhart_guard.py`, `external_benchmarks.py`,
  `differential_test.py`, `human_gate.py`, `code_quality.py`,
  `architectural_review.py`, `tier_graduation.py`.
- Audit infrastructure: `control_plane/audit.py`,
  `control_plane/budgets.py`, `control_plane/governance.py`.
- Tier-3 manifest: `app/subia/.integrity_manifest.json` +
  `app/subia/integrity.py`.
- Constitution + soul loader.
- Forge audit pipeline + capability vocabulary.
- Epistemic ledger, calibration, biases, detectors.
- Personality validation/evaluation.
- Healing infrastructure (runbooks, health_remediator, supervisor).

**Threshold ratcheting is V1-quarantined.** `governance.py` is in
the list, which means the protocol will refuse to amend
`SAFETY_MINIMUM`, `QUALITY_MINIMUM`, `MAX_REGRESSION`,
`MAX_PROMOTIONS_PER_DAY`. Reason: the protocol allows arbitrary
content changes, so an attacker / runaway self-improver who passes
operator approval once could lower thresholds catastrophically. A
future "monotonic-only ratchet protocol" would handle threshold
*raising* with stronger guarantees (type-level enforcement that
values can only go up). Out of scope for V1 — flagged in the
roadmap.

## Audit trail

Two surfaces, both mandatory:

1. **Hash-chained JSONL** at `workspace/governance/tier3_amendments/audit.jsonl`.
   Each line is `{ts, proposal_id, action, prev_hash, hash, ...}`.
   Tampering with any past line breaks the chain;
   `verify_audit_chain()` walks forward from genesis and reports
   every break.
2. **Postgres `control_plane.audit_log`** mirror via the existing
   `AuditTrail.log()` with `actor='tier3_amendment'`. Operators query
   from the `/cp/audit` dashboard.

Belt and suspenders: even if Postgres is restored from a stale
backup, the JSONL gives the operator a recoverable record. Even if
the JSONL is accidentally deleted, Postgres has the rows.

## Bootstrap status

Both bootstrap steps are **complete** as of 2026-05-09:

### ✅ Step 1 — Protocol files registered in `TIER_IMMUTABLE`

The seven protocol files are listed in
`app/auto_deployer.py:TIER_IMMUTABLE` (added by operator authorization
2026-05-09). The change-request validator now refuses to mutate them
through the regular flow; the `self_quarantine.QUARANTINED_FILES` list
still names them as a defence-in-depth check inside the protocol
itself.

### ✅ Step 2 — Master switch wired through `runtime_settings`

The toggle is exposed in the React dashboard at `/cp/settings`
("Tier-3 amendment protocol" card). Default is OFF. Flipping it:

1. Persists to `workspace/runtime_settings.json` so the choice
   survives gateway restarts.
2. Takes effect immediately — no restart needed.
3. Audited as a `runtime_settings_change` event.
4. Requires a confirmation modal ("Are you sure?") with copy that
   spells out the four-stage gate (eligibility → 7-day cooldown →
   operator approve → 30-day monitoring) before the switch flips.
5. Disabling has its own confirmation copy noting that pending
   proposals already in the pipeline are unaffected.

The dashboard read path uses
`app.runtime_settings.get_tier3_amendment_enabled()` and the protocol
defers to it via `protocol.amendment_protocol_enabled()`. The
`TIER3_AMENDMENT_ENABLED` env var is kept as a fallback only — it
fires only when `runtime_settings` is unavailable (e.g. in isolated
unit tests, or a degraded boot where the JSON state file can't be
read). The runtime-settings value always wins on a live system.

```python
# How the protocol decides whether it's on:
def amendment_protocol_enabled() -> bool:
    try:
        from app.runtime_settings import get_tier3_amendment_enabled
        return bool(get_tier3_amendment_enabled())
    except Exception:
        # Fallback for tests + degraded boots.
        return os.getenv("TIER3_AMENDMENT_ENABLED", "false").lower() in (
            "true", "1", "yes",
        )
```

### Operator action — only if you want to flip it on

The protocol is shipped OFF. To enable it:

1. Open the dashboard at `/cp/settings`.
2. Scroll to the "Tier-3 amendment protocol" card.
3. Click the checkbox — the confirmation modal appears.
4. Read the four-stage gate description, click "Yes, enable".

While it's off, any agent that tries `propose_amendment(...)`
immediately gets a `ProtocolDisabled` exception — useful telemetry
without exposing the gate.

## Public API

```python
from app.governance_amendment import (
    propose_amendment,         # PROPOSED → STAGED or ELIGIBILITY_FAILED
    advance_cooldown,          # STAGED → COOLDOWN_OK after 7 days
    operator_approve,          # COOLDOWN_OK → APPROVED
    operator_reject,           # COOLDOWN_OK or APPROVED → REJECTED
    mark_applied,              # APPROVED → APPLIED (after host_bridge writes)
    advance_monitoring,        # APPLIED → STABLE after 30 days, or REVERTED
    get_proposal,              # read by id
    list_proposals,            # filter by state
    amendment_protocol_enabled,
    ProtocolDisabled,
    AmendmentProposal, State,
    verify_audit_chain,
)
```

## React UI

Shipped 2026-05-09: `dashboard-react/src/components/SettingsPage.tsx`
gained a `Tier3AmendmentCard` with confirmation-modal flow.

- Native checkbox bound to `runtime_settings.tier3_amendment_enabled`.
- `onClick + e.preventDefault()` pattern keeps the visible state
  in lock-step with the persisted prop — the box only flips after
  the user actually confirms.
- Modal copy explains the four-stage gate when enabling, and
  reassures that pending proposals are unaffected when disabling.
- Cancel button + backdrop click both dismiss without saving.
- Save uses the existing `useUpdateRuntimeSettings` mutation hook
  (same machinery as Voice mode / Vision computer use / Concierge
  persona toggles).

## Daemon driver + auto-rollback hooks (planned, not in V1)

The cadence-aware advance-cooldown / advance-monitoring calls aren't
yet wired into a periodic job. Once the protocol is on for real, the
operator wants:

- A daily idle job that walks STAGED → COOLDOWN_OK and APPLIED → STABLE
- Hooks into `app.alignment_audit` so warnings auto-trigger
  `advance_cooldown(rollback_signal=...)`
- Hooks into `app.goodhart_guard` so detected severity ≥ high
  triggers `advance_monitoring(reverted_signal=...)`
- Signal flow for the operator-approve step: present pending
  amendments at /cp/amendments + send Signal ASK with the diff
- Read-only `/cp/amendments` React page surfacing the proposal list
  + diff viewer (the toggle exists at `/cp/settings`; the proposal
  surface is its own page, deferred to a follow-up PR)

These integration points are intentionally separate from V1 — they
need careful design, and shipping the core state machine + manual
API + React toggle first lets operators experiment before wiring
automation.

## Tests

`tests/governance_amendment/test_protocol.py` — **20 tests, all passing**:

- Master-switch refusal (env path)
- **Runtime-settings read path** (3 new tests):
  - Runtime-settings ON wins even when env is unset
  - Runtime-settings OFF blocks the protocol even when env is set
  - Env-var fallback kicks in if `runtime_settings` raises
- Citation length floor (≥ 30 chars)
- Non-Tier-3 path rejection
- Self-quarantine rejection (incl. `governance.py` specifically)
- Eligibility failure persists as terminal state
- Eligibility pass advances to STAGED
- Rollback signal during cooldown → `COOLDOWN_FAILED`
- Cooldown elapsed → `COOLDOWN_OK`
- Cooldown not elapsed → no transition
- Operator approve / reject / reject-without-reason
- Full lifecycle `PROPOSED → STAGED → COOLDOWN_OK → APPROVED → APPLIED → STABLE`
- Reverted signal during monitoring → `REVERTED`
- Illegal-transition enforcement (`STAGED → APPROVED` is rejected)
- Audit-chain integrity post-lifecycle

Run with:

```bash
.venv/bin/python -m pytest tests/governance_amendment/ -v
```

## What this protocol is NOT

- It is NOT a full automation of Tier-3 amendments. The operator is
  the final approver, every time. Agents can ask; humans decide.
- It is NOT a threshold ratchet for `governance.py`. That file is
  quarantined precisely because allowing arbitrary content changes
  there is too dangerous. A future monotonic-only ratchet protocol
  is the right tool for that job.
- It does NOT modify TIER_IMMUTABLE or `auto_deployer.py` itself.
  The bootstrap is operator-applied — you, with this PR open, decide
  whether to add the protocol files to the immutable list.
- It does NOT yet have a daemon driver or Signal-based operator
  approval UI. V1 ships the state machine; the cadence + UI layer
  is a separate, smaller PR.
