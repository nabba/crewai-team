# Resilience drills v2 (Q18 / PROGRAM §60)

Architectural rewrite of the Q6 (PROGRAM §44) resilience drill subsystem
that replaces the "scan audit log every pass" model with an explicit
state machine + operator-ratified baselines + CR lifecycle deduplication.

## Why

On 2026-05-16 the Q17.2 `local_only` drill shipped. Within 48 hours it
had produced **3580 `fail` rows + 1163 `error` rows in `drill_audit.jsonl`**
and **1204 identical `local_only_drill` CRs in `workspace/change_requests/`**.
Three drills (`embedding_migration`, `vendor_independence`, `local_only`)
were running every ~30 seconds in a hot loop because:

1. The §44 scheduler logic was "if past-due, auto-run". A drill that failed
   never updated `last_successful` → stayed past-due → ran on every idle
   pass.
2. `embedding_migration` had a CODE_ERROR bug (`AttributeError: 'DryRunStep'
   object has no attribute 'get'`) — the drill treated `DryRunStep` as a
   dict.
3. `vendor_independence` and `local_only` failed because their hardcoded
   thresholds didn't match the operator's deployment (only Anthropic +
   OpenRouter + Groq configured; no Ollama, DeepSeek, Gemini, MiniMax).
   Real production state, real "this is a failure" verdict from the
   drill's built-in opinion. But the operator's posture *accepts* this
   minimal cascade.
4. `local_only.run()` filed a CR on every fail. `change_requests.lifecycle.
   create_request` accepted all 1204 identical proposals.

The §60 redesign makes the hot loop **architecturally impossible** — a
failing drill can run at most once per backoff window regardless of master
switch wiring, and identical CRs auto-deduplicate at the lifecycle layer.

## The five primitives

### 1. State machine — `app/resilience_drills/state.py`

Per-drill persistent state in `workspace/resilience/drill_state/<name>.json`:

```
WARMING_UP → HEALTHY ↔ WATCH ↔ DEGRADED → QUARANTINED
                                            ↓
                                          MUTED (operator)
```

- **WARMING_UP** — newly-registered drill. 7-day grace by default (per-drill
  override via `DrillSpec.warmup_days`). Drill runs and accumulates
  observations but doesn't fire alerts and doesn't auto-escalate state.
  The operator ratifies a baseline before active monitoring kicks in.
- **HEALTHY** — passed last run; on cadence.
  `next_attempt_after = last_success_at + cadence_days`.
- **WATCH** — 1 recent failure. 15-min backoff.
- **DEGRADED** — 2+ consecutive failures. Exponential backoff 1h → 2h → 4h
  → 8h ... capped at `cadence_days`.
- **QUARANTINED** — 3+ consecutive `CODE_ERROR` outcomes. Scheduler will
  **not** auto-run a quarantined drill regardless of master switch. Operator
  must explicitly call `unquarantine()` via `/cp/drills/<name>` after
  fixing the bug.
- **MUTED** — operator silenced. No auto-runs, no alerts. Reversible.

The scheduler reads state and respects `next_attempt_after`. The hot loop
pattern can no longer happen — a DEGRADED drill won't run until backoff
expires.

### 2. Failure classification — `app/resilience_drills/protocol.FailureClass`

```python
CODE_ERROR          # uncaught exception in drill code → quarantine fast
STRUCTURAL_FAIL     # drill produced FAIL with stable findings → backoff
TRANSIENT_FAIL      # network/timing → short retry
BASELINE_REGRESSION # observation drifted from operator-ratified baseline
```

The drill runner returns a `DrillResult` with `failure_class` set on
FAIL/ERROR outcomes. The scheduler uses this to pick state-machine
transitions and alert prefixes (🐛 / ❌ / ⚠️ / 📉).

The orchestrator also infers `failure_class` from `DrillStatus` when the
drill doesn't supply one — back-compat for the §44 drills before
conversion.

### 3. Operator-ratified baselines — `app/resilience_drills/baseline.py`

A drill is an *observer*. Each run emits a structured `Observation` — a
dict of measurements + an optional `summary`. Pass/fail is no longer the
drill's output; it's a *comparison result* between the latest observation
and the operator-ratified `Baseline`.

```python
# vendor_independence emits per run:
observation = {
    "n_fallbacks": 1,
    "providers_with_keys": ["groq"],
    "ollama_reachable": False,
    "configured_fallbacks": ["GROQ_API_KEY"],
    "selector_has_blocklist": True,
}
```

The operator visits `/cp/drills/vendor_independence`, reviews recent
observations, clicks **Ratify baseline** — picks one observation as the
expected/acceptable state plus per-key tolerance rules:

```python
baseline.tolerances = {
    "n_fallbacks": {"rule": "min", "value": 1},
    "providers_with_keys": {"rule": "superset_of", "value": ["groq"]},
    "ollama_reachable": {"rule": "exact"},
}
```

Future observations compare to the baseline. Drift beyond tolerance →
`BASELINE_REGRESSION` failure → standard scheduler escalation. The
operator's policy wins over the drill's built-in opinion: a STRUCTURAL_FAIL
that matches the baseline is **promoted** to PASS by the orchestrator.

Tolerance grammar: `exact`, `min`, `max`, `range`, `subset_of`,
`superset_of`. See `baseline.py:_check_rule` for the canonical list.

This is the same model already used by `app/healing/monitors/embedding_drift`
(anchor queries vs baseline), `app/epistemic/calibration` (Brier drift vs
baseline), and `app/companion/interest_model` (drift vs baseline).
Generalising it to resilience drills closes a conceptual gap.

### 4. CR lifecycle deduplication — `app/change_requests/lifecycle.py`

`create_request(requestor, path, new_content, ...)` is now idempotent over
the (requestor, content_hash) key while the original CR is non-terminal.
`content_hash = sha256(requestor || path || diff)` — reason text is
deliberately excluded so distinct motivations for the same structural
change accumulate in one CR.

```
1st call:  cr_abc (PENDING, recurrence_count=0)
2nd call:  cr_abc (PENDING, recurrence_count=1)
...
1000th call: cr_abc (PENDING, recurrence_count=999)
```

Dedup window: while status is in `{PENDING, APPROVED, APPLY_FAILED}`.
Terminal statuses (`REJECTED, APPLIED, ROLLED_BACK, TIER_IMMUTABLE_REFUSED,
TIMEOUT`) release the hold so a genuine new occurrence creates a fresh
CR — the previous decision was about a past occurrence.

This is a **system-wide** property. Every producer benefits — not just
drills. Closes the parallel-codepath gap with `proposal_bridge` which had
its own dedup but was bypassed by direct `create_request` callers.

New fields on `ChangeRequest`:

- `content_hash` — sha256 dedup key
- `recurrence_count` — N - 1 for N total occurrences
- `first_seen_at` — preserved across recurrences
- `last_recurrence_at` — bumped on each duplicate

### 5. Orchestrator — `app/resilience_drills/runner.py`

Every drill invocation (scheduler, REST, CLI) goes through `invoke_drill`
which threads:

1. Master-switch check
2. State-machine permission check (`is_runnable_now`)
3. In-flight lock acquisition
4. Prior-state snapshot (for landmark emission)
5. `_safe_run` — calls drill, captures uncaught exception as CODE_ERROR
   with traceback
6. Baseline comparison (`_apply_baseline_check`) — promotes PASS-with-
   regression to FAIL/BASELINE_REGRESSION, demotes FAIL-with-matching-
   baseline to PASS
7. State transition (`_apply_state_transitions`) — `record_pass` or
   `record_failure` with backoff calculation
8. Audit append + continuity-ledger landmark (suppressed during warmup)
9. Lock release

**The drill runner now returns a bare `DrillResult`** — no `append_result`,
no `acquire_drill_lock`, no `emit_landmark_for` calls inside drill code.
The orchestrator owns all of that. Every drill in `app/resilience_drills/
drills/` has been converted (9 drills total: `backup_restore`,
`embedding_migration`, `embedding_rotation`, `kill_the_gateway`,
`local_only`, `secret_rotation`, `source_ledger_replay`, `task_recovery`,
`vendor_independence`).

`invoke_drill_by_name(name, triggered_by="operator")` bypasses the backoff
gate — operator explicitly asked. `triggered_by="scheduler"` keeps the
gate engaged. QUARANTINED and MUTED states are never auto-bypassed.

## Operator surfaces

### REST

| Endpoint | Purpose |
|---|---|
| `GET /api/cp/drills/registry` | List all drills with state + v2 fields |
| `GET /api/cp/drills/<name>` | Detail: state, baseline, recent observations, transitions |
| `POST /api/cp/drills/run/<name>` | Operator-triggered run via orchestrator |
| `POST /api/cp/drills/<name>/ratify-baseline` | Ratify latest observation as baseline |
| `POST /api/cp/drills/<name>/unquarantine` | Lift quarantine; drill returns to WATCH |
| `POST /api/cp/drills/<name>/mute` | Operator mute (optional until_iso) |
| `POST /api/cp/drills/<name>/unmute` | Lift mute; drill returns to HEALTHY |

### React

`/cp/drills` — new dedicated page (`dashboard-react/src/components/
DrillsPage.tsx`):
- State-coded drill list grouped by category (Needs attention / Warming up
  / Healthy / Muted)
- Click-through detail drawer: state badge, runnable-now reason,
  recent observations, baseline (or "Ratify baseline" form), traceback
  (when QUARANTINED), state-transition history
- Per-drill action buttons: Run now / Unquarantine / Mute / Unmute /
  Ratify baseline

The legacy `ResilienceDrillsCard.tsx` in `/cp/settings` keeps the master
switches.

## Migration

### One-shot CR spam consolidator — `app/change_requests/spam_cleanup.py`

```bash
docker exec crewai-team-gateway-1 python -m app.change_requests.spam_cleanup \
    --requestor local_only_drill --dry-run    # preview
docker exec crewai-team-gateway-1 python -m app.change_requests.spam_cleanup \
    --requestor local_only_drill              # consolidate
```

The tool:
1. Groups PENDING CRs by `(requestor, path)`
2. Picks the oldest as canonical and sets `recurrence_count = N - 1`
3. Backfills `content_hash` + `first_seen_at` + `last_recurrence_at`
4. Moves the others to `workspace/change_requests/archive/<date>_drill_spam/`
   (reversible — never deletes)
5. Records the consolidation in the audit log

Idempotent. Safe to re-run. Only operates on PENDING — terminal CRs are
left alone.

The gateway's in-memory CR index needs to be invalidated after running
(restart the gateway, OR call this from inside the running process via a
new REST endpoint — not currently wired).

### Drill state initialization

The state machine initializes lazily — on first scheduler pass, each
registered drill gets a WARMING_UP state record. No explicit migration
step needed; existing drills with `warmup_days=0` skip the warmup grace
and enter HEALTHY on first PASS (or DEGRADED on first FAIL).

## Test coverage

| File | Count | What's pinned |
|---|---|---|
| `tests/test_drill_state_machine.py` | 22 | State transitions, backoff growth + cap, quarantine + unquarantine, mute + unmute, warmup grace, hot-loop regression |
| `tests/test_drill_baseline.py` | 15 | Round-trip, every tolerance rule, vendor_independence baseline use case, missing-key surfacing |
| `tests/test_drill_scheduler_v2.py` | 11 | Hot-loop impossible, quarantine never re-runs, HIGH-risk never auto-runs, baseline promotes/demotes |
| `tests/test_cr_lifecycle_dedup.py` | 12 | Recurrence bumping, terminal-release, 1000-duplicate collapse, content_hash stability |
| `tests/test_cr_spam_cleanup.py` | 6 | Consolidation, idempotency, archive-not-delete, terminal-CR-skip |
| `tests/test_drill_routes_v2.py` | 11 (skipped on host) | REST endpoint shapes (skipped without psycopg2) |

**66 unit tests pass on the host. 11 route tests skip without
gateway-deps. 0 fail.** Plus pinning: the hot-loop regression test
(`test_no_hot_loop_regression_2026_05_16`) and the 1000-duplicate test
(`test_thousand_duplicates_collapse_to_one_record`) encode the exact
2026-05-16 incident pattern.

## Operator usage

After Q18 lands and the gateway is rebuilt:

1. **Inspect drill state** — visit `/cp/drills`. The 3 problematic
   drills will appear in "Needs attention" (vendor_independence in
   WATCH, embedding_migration + local_only in WARMING_UP awaiting
   master-switch re-enable).
2. **Re-enable disabled drills** — via `/cp/settings` Resilience Drills
   card, flip `drill_vendor_independence_enabled` and
   `drill_local_only_enabled` back on. The drills will run on next
   scheduler pass.
3. **Run once on demand** — `/cp/drills/<name>` → "Run now". Bypasses
   backoff (operator explicit).
4. **Ratify baselines** — for `vendor_independence` and `local_only`,
   review the observation, lock in tolerances:
   - `vendor_independence`: `n_fallbacks ≥ 1`, `providers_with_keys ⊇ ["groq"]`
   - `local_only`: `n_providers_ready ≥ 1`, `ready_providers ⊇ ["groq"]`
   Notes: "single non-dominant fallback acceptable for our posture".
5. **Drills go HEALTHY** — future runs compare to the baseline. If
   Groq key expires or a new provider configuration drifts below the
   ratified state, the drill alerts as `BASELINE_REGRESSION`.

## Deliberate non-decisions

- **No backward-compat flag.** Hard cutover when v2 lands; the legacy
  scheduler is gone. Per the operator's preference for clean code over
  shims.
- **No new IDENTITY_EVENT_KIND.** The `resilience_drill` event from
  §44.1 covers landmark emissions; the state-machine transitions are
  internal audit-log detail, not identity-shaping.
- **No new master switches.** The 8 drill-level + master toggles from
  §44 are unchanged. Q18 adds operator actions (ratify, unquarantine,
  mute) rather than knobs.
- **Healing-monitor count is unchanged.** No new monitors. The Q18
  scheduler IS itself the rate limiter — adding a separate
  `drill_scheduler_health` monitor would be redundant.
- **No annual-reflection wiring.** Q18 is a refactor of an existing
  observation pipeline, not a new identity-shaping subsystem. The
  existing `resilience_drill` ledger event already feeds the annual
  reflection's `summarise_drift` Counter.
- **No `task_recovery` deeper conversion.** The drill is converted to
  the v2 runner contract; its complex fixtures + injection points are
  preserved as-is.
