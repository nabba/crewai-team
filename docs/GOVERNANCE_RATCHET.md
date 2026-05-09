# Governance Ratchet + Goodhart Hard Gate

> Shipped 2026-05-09. Two governance subsystems that ride on top of
> `app/governance.py`'s `evaluate_promotion()` — the universal
> promotion gate every improvement system (evolution, modification,
> training, ATLAS) routes through.
>
> See PROGRAM.md §25 for the chronological change-log.

This is the canonical reference for both:

* **Governance Ratchet** (`app/governance_ratchet/`) — operator-controlled
  raising / relaxing of `SAFETY_MINIMUM` and `QUALITY_MINIMUM`.
* **Goodhart Hard Gate** (`app/goodhart_guard.py` +
  `app/governance.py:_evaluate_goodhart_gate`) — Gate-0 that runs
  before safety + quality and blocks promotion when the gaming detector
  reports `severity == "high"`.

They share a common architectural pattern (typed exceptions, hash-chained
audit, runtime-toggleable React control) and both originate in the
9-item resilience-gap closure plan from 2026-05-09. Distinct concerns,
distinct modules.

---

## Governance Ratchet

### Why it exists

Closes resilience gap #6. The four governance constants in
`app/governance.py` (`SAFETY_MINIMUM = 0.95`, `QUALITY_MINIMUM = 0.70`,
`MAX_REGRESSION = 0.15`, `MAX_PROMOTIONS_PER_DAY = 20`) are static
module constants in TIER_IMMUTABLE. Without a ratchet path, multi-year
operation hits a hard ceiling: the system either games the metrics
(Goodhart's law) or plateaus. The ratchet lets the bar rise as the
system earns trust.

### What's ratchet-controlled

V1 covers the two **floors**:

| Threshold | Floor | Effective |
|---|---|---|
| `SAFETY_MINIMUM_FLOOR = 0.95` | inviolable | `max(0.95, ratcheted_safety)` |
| `QUALITY_MINIMUM_FLOOR = 0.70` | inviolable | `max(0.70, ratcheted_quality)` |

`MAX_REGRESSION` and `MAX_PROMOTIONS_PER_DAY` are **ceilings** — V1
defers them. Ratcheting a ceiling DOWN (stricter direction) needs the
same protocol shape but inverted; future V2.

The hardcoded floor in `governance.py` is the post-bootstrap safety
contract: even a corrupted state file or maliciously-edited JSON
can't drop the effective threshold below it (`max(FLOOR, ratcheted)`
clamp = type-level guarantee).

### Two operations

```python
from app.governance_ratchet import set_ratchet, relax_ratchet

# Raise (monotonic up, ≤ 1.0)
set_ratchet(
    name="safety_minimum",
    new_value=0.97,
    source="operator_react",
    reason="last 100 promotions all > 0.98",
)
# → MonotonicViolation if new_value <= current
# → CeilingViolation if new_value > 1.0

# Lower (monotonic down, ≥ FLOOR; mandatory reason)
relax_ratchet(
    name="safety_minimum",
    new_value=0.96,
    source="operator_react",
    reason="rolled back after eval-set v2 regression",
)
# → MonotonicViolation if new_value >= current
# → FloorViolation  if new_value < FLOOR
# → ValueError      if reason is empty
```

Typed exceptions: `MonotonicViolation`, `FloorViolation`,
`CeilingViolation`, `UnknownThresholdViolation`.

### Effective-value lookup

```python
from app.governance_ratchet import effective_value

effective_value("safety_minimum")
# Returns max(SAFETY_MINIMUM_FLOOR, ratcheted_current).
# Always ≥ FLOOR — even with missing or corrupted state file.
```

`governance.evaluate_promotion()` reads
`effective_safety_minimum()` and `effective_quality_minimum()` (which
delegate to `effective_value`) for every promotion check. The gate
result includes both `threshold` (effective) and `floor` for
transparency.

### Audit + history

* `workspace/governance/ratchet_state.json` — current state per
  threshold + full history (every entry has timestamp, direction,
  old/new values, source, reason, audit-chain hash).
* `workspace/governance/ratchet_audit.jsonl` — hash-chained audit log
  (`actor='governance_ratchet'` mirrored to Postgres
  `control_plane.audit_log`). `verify_chain()` walks forward from
  genesis and reports any tamper.

### React UI

`/cp/settings` → **Governance ratchet** card. Shows every threshold's
floor, current, effective, and last-change footer. Two action buttons
per threshold:

| Button | Behaviour | UX gate |
|---|---|---|
| **Ratchet up ↑** | Opens `RatchetUpModal`. Number input snapped to `current + 0.01` default. Validates `new_value > current` AND `≤ 1.0`. Optional reason. | Simple confirm. |
| **Relax ↓** | Opens `RatchetRelaxModal`. Red-bordered destructive UX. Three gates before submit enables: (a) `floor ≤ new_value < current`, (b) reason ≥ 10 chars, (c) typed phrase `RELAX <THRESHOLD>` matches exactly. | **Typed confirmation phrase.** |

The typed-phrase check is enforced server-side too (the React form
is the UX; the gateway re-checks the body's `confirmation` field).

### REST endpoints

* `GET /config/governance_ratchet/state` — snapshot of every threshold
  + history.
* `POST /config/governance_ratchet/set` — ratchet up. Body
  `{name, new_value, reason}`. Gateway-bearer-secret gated; rate-limited.
* `POST /config/governance_ratchet/relax` — ratchet down. Body
  `{name, new_value, reason, confirmation}`. Same gating + the
  typed-phrase check.

### V1 scope notes

* **Operator-only.** Agents have NO mutating API path here. The only
  mutation flows through React → `/config/governance_ratchet/*`
  (gateway-bearer-secret gated) → `set_ratchet` / `relax_ratchet`.
* **No agent-proposed amendments via Tier-3.** `governance.py` is in
  the Tier-3 amendment self-quarantine (`app/governance_amendment/
  self_quarantine.py:QUARANTINED_FILES`) — the amendment protocol
  refuses to act on it. V1 keeps amendment and ratchet as separate
  protocols. A future V2 might allow agents to propose ratchet-up
  through the amendment protocol's eligibility + cooldown gates,
  while operator-only for relax.

### Tests

`tests/governance_ratchet/test_protocol.py` — 18 tests. Highlights:
`effective_value` clamps below floor (key invariant),
`set_ratchet` rejects non-strict-greater / above 1.0 / below 0,
`relax_ratchet` rejects below FLOOR / non-strict-less / empty reason,
governance.py integration tests confirming `evaluate_promotion` uses
the ratcheted value AND that the FLOOR clamp survives a corrupted
state file, audit-chain integrity post-lifecycle.

---

## Goodhart Hard Gate

### Why it exists

Closes resilience gap #2. `app/goodhart_guard.py` was already
shipped — it scans recent evolution history every cycle and persists
severity-tagged signals (`kept_ratio_spike`, `category_concentration`,
`rollback_silence`) to `workspace/goodhart_reports.json`. But nothing
in the promotion path read those signals: gaming was logged and
ignored. Detection without enforcement. The hard gate wires it in.

### Where it sits

`app/governance.py:evaluate_promotion()` runs gates in order:

1. **Goodhart hard gate** ← Gate 0, NEW
2. Safety (effective_safety_minimum)
3. Quality (effective_quality_minimum)
4. Regression check
5. Rate limit

Gate 0 reads
`app.goodhart_guard.recent_signal_summary(lookback_hours=24)` and
classifies the system into one of three modes:

| Mode | UI label | Behaviour |
|---|---|---|
| `disabled` | Off (emergency disable) | Gate skipped entirely. Use only when a buggy detector is blocking legitimate promotions. |
| `advisory` | Advisory (default) | Gate runs and records severity in `gate_results["goodhart"]`. Does NOT block. Use for ~2 weeks after enabling to characterise false-positive rates. |
| `enforcing` | Enforcing (blocking) | `severity == "high"` BLOCKS promotion. `severity == "medium" / "low"` are still advisory by design (the detector's medium / low signals are advisory by nature). |

When blocking fires:

```python
PromotionResult(
    approved=False,
    reason='Goodhart hard gate blocked: severity=\'high\' — kept_ratio=...',
    gate_results={'goodhart': {'phase': 'enforcing', 'severity': 'high', ...}},
)
```

The gate **fails OPEN** when the detector itself raises — a buggy
detector should never halt every promotion.

### Two read-only API additions to `goodhart_guard.py`

Operator-authorized TIER_IMMUTABLE edits added:

```python
def recent_severity(lookback_hours: int = 24) -> str:
    """Returns 'none' | 'low' | 'medium' | 'high' — highest severity
    in the rolling window. Reads workspace/goodhart_reports.json
    read-only; never modifies."""

def recent_signal_summary(lookback_hours: int = 24) -> dict:
    """Audit-friendly snapshot:
    {lookback_hours, highest_severity, counts, highest_description}"""
```

The detection logic (`detect_gaming_signals` + `run_goodhart_check`)
is unchanged.

### React UI

`/cp/settings` → **Goodhart hard gate** card. 3-way segmented
control. Each option has a one-paragraph explanation in the card so
operators see *why* each mode exists before flipping. The segmented
choice translates to two underlying booleans
(`goodhart_hard_gate_disabled`, `goodhart_hard_gate_enforcing`)
which `governance.py` reads via `runtime_settings.get_*()`.

### Three env / runtime flags

| Flag | Default | Purpose |
|---|---|---|
| `goodhart_hard_gate_disabled` (runtime_settings) | `false` | Emergency disable. Bypasses both phases. |
| `goodhart_hard_gate_enforcing` (runtime_settings) | `false` | Flip from advisory → blocking. Default OFF: ship in advisory mode for ~2 weeks to characterise FP rate. |
| `GOODHART_HARD_GATE_DISABLED` / `_ENFORCING` (env) | unset | Fallback only — fires when `runtime_settings` raises. |

### Tests

`tests/governance_ratchet/test_goodhart_gate.py` — 18 tests. Covers
`recent_severity` window math, `recent_signal_summary` counts,
advisory phase records-but-doesn't-block, enforcing phase blocks on
high, enforcing doesn't block on medium/low, emergency disable
short-circuits, no-signals doesn't block, detector failure fails
OPEN, gate-evaluated-before-safety (high blocks even when
`safety_score=0.50`), runtime-settings → env-fallback hierarchy
across all 5 master switches.

---

## Operator activation

Both subsystems ship in their default-OFF safe state. To activate:

1. **Restart the gateway** once to pick up the new endpoints
   (`/config/governance_ratchet/*`, `/config/runbook_settings`),
   the new env-flag readers in `governance.py` / `runbooks.py` /
   `supervisor.py` / `recovery/loop.py`, and the new
   `runtime_settings` keys.
2. **Open `/cp/settings`** in the dashboard.
3. **Governance ratchet card** — appears between Tier-3 amendment
   and Goodhart hard gate. No action needed at first; thresholds
   start at FLOOR.
4. **Goodhart hard gate card** — defaults to **Advisory**. Watch
   `gate_results["goodhart"]` in `evaluate_promotion` results for
   ~2 weeks; if FP rate is acceptable, flip to **Enforcing**.
5. **Self-heal subsystems card** — toggle the three master switches
   (Runbook dispatcher / Tool exception supervisor / Refusal
   recovery loop) per your preference. Per-runbook list below shows
   the eight registered handlers with their `min_recurrence`.

### Rollback paths

* **Governance ratchet too high** (system plateaus, no promotions
  pass) — operator manually edits
  `workspace/governance/ratchet_state.json` to lower `current` (the
  edit is detected and audited). The hardcoded FLOOR in
  `governance.py` provides the safety net so even a rushed manual
  edit can't drop below FLOOR.
* **Goodhart blocking false positives** — flip to Off via the React
  3-way control, OR set `GOODHART_HARD_GATE_DISABLED=true` env var
  for a permanent OFF that survives runtime-settings file deletion.
* **Self-heal subsystem misbehaving** — uncheck its master toggle
  on `/cp/settings`. Per-runbook toggles let you disable individual
  handlers without disabling the whole dispatcher.

### Audit query examples

```bash
# Goodhart gate decisions over the last week
gh api '/api/cp/audit?actor=governance&action_prefix=goodhart&since=...'

# Ratchet history
gh api '/api/cp/audit?actor=governance_ratchet&since=...'

# Verify the ratchet audit chain is intact
python -c 'from app.governance_ratchet import verify_audit_chain; print(verify_audit_chain())'
```

---

## Auto-proposers (Phase C #5 + Phase D #3, 2026-05-09)

Two daily proposers watch governance state and suggest operator
actions when conditions warrant. **Neither auto-applies.** Both
write to `workspace/governance_proposals.jsonl` (capped 1000 via
`jsonl_retention`); operator approves through the existing React
`/cp/settings` ratchet card.

### `app/governance_ratchet/auto_propose.py` — meta-governance ratchet

Daily cadence. For each of `safety_minimum` and `quality_minimum`:

1. Reads `control_plane.audit_log` for `action_prefix=promotion_`
   over the last 7 days.
2. Computes `score_avg` from each row's `detail_json`.
3. Checks rollback rate (action prefix `rollback_` / `rejected_` /
   `revert_` over the same window).
4. Proposes raise from `effective` to `effective + 0.01` when:
   * ≥20 promotions in window
   * `score_avg ≥ effective + 0.03`
   * rollback_rate < 5%
   * no proposal for this threshold in last 14 days
5. Caps at `0.99` (never propose 1.0 — nothing reaches it).

Master switch: `GOVERNANCE_AUTO_PROPOSE_ENABLED` (default ON).

### `app/governance_ratchet/goodhart_enforcing_proposer.py`

Daily cadence. Fires only when the gate is in **Advisory** mode
(skip Off, skip Enforcing). 14-day observation window.

Proposes `Advisory → Enforcing` flip when:

* ≥30 promotions in window.
* `would_block_pct ≤ 5%` — fewer than 1 in 20 promotions would
  have been blocked under Enforcing.
* No sustained high-severity incidents (>2 high signals in window
  blocks the proposal).
* No proposal in last 14 days.

The fire-once-per-flip-need pattern means the operator can dismiss
the proposal by simply not acting; it won't re-propose for 14 days.

Master switch: `GOODHART_ENFORCING_PROPOSER_ENABLED` (default ON).

### Why proposers, not auto-applies

The ratchet edits and the gate flip both touch governance behavior
that affects every promotion. A bad auto-apply could quietly disqualify
all future work. The cost of one operator click vs the cost of
silent over-tightening is small — operator-in-the-loop on every
governance change is the right default.
