# Vacation Mode

**Status (2026-05-16):** Q16 Theme 3 — decade-resilience hardening,
operator-unavailable autonomy. Companion to the `operator_anomaly`
healing monitor (the observational piece). This document covers the
defense lever: pre-staged auto-apply during a time-bounded operator-
absence window.

## Why this exists

Almost every action requires operator approval — change requests
(CRs), Tier-3 amendments, the `kill_the_gateway` typed phrase, and
so on. That gate is the right default for a system that takes its
own code seriously. But it has a known failure mode: **if the
operator is unavailable for a stretch (hospital, retreat, deep work,
travel without internet), the self-improvement axis halts.**

The §38.3 auto-apply infrastructure was shipped dormant — `RiskClass.
AUTO_APPLY`, `validate_auto_apply`, `auto_revert.py` all exist, but
the requestor + path allowlists are deliberately EMPTY (P1-P5
principles say so). Vacation mode is the **staged, time-bounded,
narrower-than-§38.3** path to enable a tightly-scoped subset of
auto-application during operator absence.

## What it is, in one sentence

A master switch + a pre-staged allowlist + a sweep daemon that
auto-approves matching PENDING CRs while engaged, with loud Signal
alerts, the existing auto-revert 60-min rollback window, and an
auto-expiring engagement.

## What it is NOT

- **Not a way to disable the operator gate permanently.** Every
  engagement expires (max 30 days). After expiry, all CRs flow back
  through the operator gate.
- **Not a security-relaxation lever for TIER_IMMUTABLE.** Standard
  `validate()` runs first; TIER_IMMUTABLE paths are refused
  regardless of allowlist.
- **Not a substitute for the §38.3 auto-apply infrastructure.** That
  pathway stays as-is (also EMPTY-allowlisted). Vacation mode is a
  separate path with operator-managed allowlists rather than
  module-constant ones.
- **Not a remote-management tool.** The operator must be present to
  ENGAGE. Engagement is a deliberate human act.

## Design contract

1. **Time-bounded.** Engagement requires an explicit `until_ts`.
   Auto-expires on the next state read past that timestamp. Hard
   cap: 30 days per engagement (`MAX_DURATION_DAYS`).
2. **Pre-staged.** Allowlist must be staged WHEN DISENGAGED. Adding
   to the allowlist while engaged is refused. (No chicken-and-egg
   "engage → expand allowlist → broaden auto-apply" attack vector.)
3. **Default OFF.** Master switch ON by default (the kill-switch),
   but ENGAGEMENT is OFF until the operator explicitly engages.
4. **TIER_IMMUTABLE absolute.** Standard validator runs first.
5. **Tighter than §38.3.** Default line cap is 10 (vs §38.3's 20);
   the forbidden-prefix list is a super-set of §38.3's (additional
   prefixes: `app/governance_amendment/`, `app/governance_ratchet/`,
   `app/change_requests/`, `app/vacation_mode/`, `app/subia/`,
   `app/auto_deployer.py`, `.github/`).
6. **Loud audit.** Every auto-apply during vacation emits a critical
   Signal alert (NOT arbitrated — bypasses suppression) plus a row
   in `workspace/vacation_mode/auto_apply_log.jsonl` (capped at
   1000 rows).
7. **Anomaly-aware.** Suspends auto-apply for 24h after a
   `operator_anomaly` `new_sender` critical alert that fired AFTER
   engagement began. The reasoning: "something looks weird; let me
   wait until things look normal again."
8. **Composes with auto-revert.** Every vacation-approved CR goes
   through the existing 60-min auto-revert watcher in
   `app/change_requests/auto_revert.py`. If the same error pattern
   recurs within the window, the change rolls back automatically.

## Lifecycle

```
            ┌───────────────────────────┐
            │  not staged, not engaged  │  ← initial state
            └─────────────┬─────────────┘
                          │ stage_allowlist(...)
                          ▼
            ┌───────────────────────────┐
            │  staged, not engaged      │  ← operator can re-stage freely
            └─────────────┬─────────────┘
                          │ engage(until_ts, engaged_by, reason)
                          ▼
            ┌───────────────────────────┐
            │  staged, engaged          │  ← sweep daemon active
            │  (allowlist FROZEN)       │     auto-apply window open
            └─────────────┬─────────────┘
                          │ disengage() OR auto-expiry past until_ts
                          ▼
            ┌───────────────────────────┐
            │  staged, not engaged      │  ← state visible for review
            └───────────────────────────┘
```

## Operator interface

### Stage an allowlist

Before engaging, pre-stage:

```python
from app import vacation_mode
vacation_mode.stage_allowlist(
    requestor_allowlist=[
        "wiki_index_reconciler",
        "capability_gap_analyzer",
        "library_radar",
    ],
    path_prefix_allowlist=[
        "wiki/companion/",
        "wiki/self/",
        "docs/proposed_fixes/",
    ],
    max_diff_lines=10,
)
```

**Validation:**

- `path_prefix_allowlist` entries MUST end with `/`. No
  whole-file allowlisting via bare paths.
- Entries cannot be too broad: `app/`, `tests/`, `docs/`,
  `wiki/`, `deploy/` are refused. Specify at least one subdirectory.
- `max_diff_lines` ∈ [1, 50]. Default 10.
- Refused while engaged.

### Engage

```python
import time
engagement = vacation_mode.engage(
    until_ts=time.time() + 14 * 86400,  # 14 days
    engaged_by="andrus",
    reason="medical leave; back 2026-06-01",
)
```

**Refusals:**

- Already engaged.
- Staged allowlist is empty.
- `until_ts` in the past.
- Duration exceeds 30 days.

After engagement, the allowlist snapshot is frozen in
`state.engagement.frozen_allowlist`. Subsequent `stage_allowlist`
calls (which would only succeed AFTER `disengage()`) do not affect
the live engagement.

### Disengage

```python
vacation_mode.disengage(disengaged_by="andrus")
```

Idempotent. Returns the post-state with the staged allowlist
preserved (for future re-engagement) but `engaged=False` and
`engagement=None`.

### Query

```python
vacation_mode.is_active()       # bool — engaged and not auto-expired
vacation_mode.current_state()   # full state (auto-expires on read)
vacation_mode.current_allowlist()  # frozen if engaged, staged if not
```

## What gets auto-applied

The sweep daemon runs every 5 min (`SWEEP_INTERVAL_SECONDS`). On
each pass, for each PENDING CR:

1. Run `validate_vacation_apply` — composes:
   - Standard `validate()` (TIER_IMMUTABLE + repo-root + sensitive-
     name + content-size).
   - Vacation forbidden-prefix list (super-set of §38.3's).
   - Requestor allowlist match.
   - Path prefix allowlist match.
   - Net line delta ≤ `max_diff_lines`.
   - Additive-only (no deleted lines).
2. If ok: `lifecycle.approve(cr.request_id,
   source=DecisionSource.VACATION_AUTO_APPLY, ...)`.
3. `apply.apply_change(cr.request_id)` synchronously.
4. Log row to `workspace/vacation_mode/auto_apply_log.jsonl`.
5. Loud Signal alert (`critical=True`, `arbitrate=False`).

**Per-sweep cap:** 5 CRs max. If more match, they wait for the next
sweep (5 min later). This prevents a malicious / buggy producer
from filing many CRs at once and getting them all auto-applied
before the operator notices.

**Cross-sweep daily rate limits (PROGRAM §51 follow-on):** persisted
in `workspace/vacation_mode/sweep_rate_limit.json`, reset at UTC
day rollover.

- Per-requestor daily cap: 6 auto-applies/day. A single buggy
  agent cannot file many small additive CRs across many sweeps
  and saturate the auto-apply window.
- Global daily cap: 20 auto-applies/day across all requestors.
  Tighter than §38.3's global cap of 10 was intentional — vacation
  mode's allowlist is operator-staged, so the operator has explicit
  insight into what's eligible.

Both caps share the same daily bucket; whichever fires first
returns the requestor's CR to PENDING (operator will see it in
the standard CR review flow on return).

## Composition with operator_anomaly

The 37th healing monitor `operator_anomaly` observes hour-of-day,
cadence, length, and new-sender shifts. The sweep daemon consults
its state file (`workspace/healing/operator_anomaly_state.json`) on
every pass:

- If the monitor's `last_alert_at["new_sender"]` timestamp is AFTER
  the current engagement started AND within the last 24h, the sweep
  pauses (returns `paused_on_anomaly=True` without auto-applying).
- Once the 24h window expires (no further new_sender alert), the
  sweep resumes.

This is the only anomaly that pauses vacation mode automatically.
Hour/cadence/length shifts surface as alerts but do not pause
auto-apply (they're noisier; pausing on each would defeat the
purpose of vacation mode).

The operator can also manually `disengage()` at any time, including
remotely via Signal slash command (when wired) or the React UI
(when wired) — both deferred follow-ons.

## Forbidden prefixes (super-set of §38.3)

```
app/memory/
app/souls/
wiki/governance/
migrations/
deploy/
host_bridge/
app/governance_amendment/
app/governance_ratchet/
app/change_requests/
app/vacation_mode/
app/subia/
app/auto_deployer.py
.github/
```

Operator-staged allowlists can include any prefix EXCEPT these
(rejected at `validate_vacation_apply` time with `reason="path X
is under forbidden prefix Y"`).

## Auto-apply log

`workspace/vacation_mode/auto_apply_log.jsonl` accumulates one row
per auto-apply attempt. Schema:

```json
{
  "ts": "2026-05-16T12:34:56+00:00",
  "request_id": "cr-abc123",
  "path": "wiki/companion/notes.md",
  "requestor": "wiki_index_reconciler",
  "status_pre": "pending",
  "status_post": "applied",
  "ok": true,
  "error": null,
  "elapsed_s": 1.23
}
```

Capped at 1000 rows (oldest dropped on overflow). Survives gateway
restart.

## When vacation mode ends — end-of-vacation digest

On every disengagement (manual or auto-expiry), the system composes
a markdown digest at `workspace/vacation_mode/digests/<engaged_at-
iso>.md` summarising the window:

- Engagement metadata (start, end, duration_actual, reason, frozen
  allowlist).
- Auto-apply log rows in the window (total, by requestor, by
  coarse path prefix, success/failure split).
- Per-row detail table (newest first).
- Operator next-steps reminders.

The composer is `app/vacation_mode/digest.py:compose_digest`. It
reads only `workspace/vacation_mode/auto_apply_log.jsonl` (no LLM,
no network) so it always succeeds. Failure-isolated — if the write
fails, the disengagement still completes.

After return, the operator should:

1. Read the latest digest: Signal `/vacation digest` or
   `GET /api/cp/vacation/digests/<name>`.
2. Cross-reference each row against `/cp/changes` for full diffs
   (CRs with `decided_by=vacation-auto-apply`).
3. Check the auto-revert log
   (`/api/cp/changes/audit?actor=auto_revert`) for any rollbacks
   triggered within the 60-min window post-apply.

## Security considerations

### Engagement requires presence

To engage, the operator must call `engage(...)` directly — currently
via Python REPL or a Signal slash command (when wired). Recovery
from a compromised Signal account: the operator can disengage via
direct REST/Python access since vacation_mode_state is in
runtime_settings.

### The frozen allowlist is the contract

Once engaged, the allowlist is the ONLY surface the sweep daemon
consults. An attacker who somehow modifies `staged_allowlist` after
engagement will find it doesn't affect the active engagement —
they'd have to disengage first (which requires the same authorization
as engaging).

### Forbidden prefixes are belt-and-suspenders

Even with a maximally broad operator-staged allowlist, the forbidden
prefixes (especially `app/governance_amendment/`,
`app/governance_ratchet/`, `app/change_requests/`,
`app/vacation_mode/`, `app/subia/`) are refused. Self-modification of
the vacation mode system itself is impossible through this path.

### Loud alerts on every auto-apply

Critical Signal notifications fire on every auto-apply, bypassing
arbiter suppression. If the operator is checking Signal at any
frequency during vacation, they'll see vacation auto-applies
landing in real-time and can disengage immediately if anything
looks wrong.

### Auto-revert is the safety net

Every vacation-approved CR is registered with the existing
`auto_revert` watcher. If the same error pattern recurs within
the 60-min window, the change rolls back automatically. This is
the §38.3 contract — vacation mode reuses it.

## Operator interfaces

### Signal slash command `/vacation`

Live as of the PROGRAM §51 follow-on. Subcommands:

```
/vacation                              status (engaged / staged / time left)
/vacation status                       same as above
/vacation show                         show current allowlist (frozen if engaged)
/vacation engage <hours> <reason>      engage for N hours (1 ≤ N ≤ 720)
/vacation disengage                    disengage now; writes end-of-vacation digest
/vacation digest [N]                   read recent digest (N=0 = newest, default)
```

Staging the allowlist is deliberately NOT exposed via Signal —
parsing multi-list inputs safely from free-form text is fragile.
Stage via Python REPL, REST, or the (future) React UI.

### REST endpoints under `/api/cp/vacation/*`

```
GET  /state                  current state blob (engaged + staged_allowlist)
GET  /allowlist              currently-applicable allowlist
POST /allowlist/stage        stage a new allowlist (409 if engaged)
POST /engage                 engage vacation mode (409 if validation fails)
POST /disengage              disengage immediately
GET  /digests                list digest filenames
GET  /digests/{name}         read one digest (404 if not found, 400 if invalid name)
GET  /audit-log?limit=N      recent rows from auto_apply_log.jsonl
```

All mutating endpoints require Bearer auth (the same dependency
`/api/cp/changes` uses). Path-traversal-guarded on `/digests/{name}`.

### Python API

```python
from app import vacation_mode as vm
vm.stage_allowlist(...)
vm.engage(until_ts=..., engaged_by=..., reason=...)
vm.disengage()
vm.is_active()
vm.current_state()
vm.current_allowlist()
vm.list_digests()
vm.read_digest(path)
```

## Continuity-ledger emission

PROGRAM §51 follow-on. The 19th `IDENTITY_EVENT_KIND`,
`vacation_window`, is emitted on:

- `engage()` → `detail.event = "engage"`, summary with duration +
  reason, full allowlist snapshot in detail.
- `disengage()` → `detail.event = "disengage"`, actual duration.
- Auto-expiry via `current_state()` → `detail.event = "auto_expire"`.

The annual reflection's `summarise_drift` Counter auto-surfaces
this kind. Vacation history becomes visible in
`wiki/self/value_reflections/<year>.md` without further wiring.

## Master switches

| Setting | Default | Purpose |
|---|---|---|
| `vacation_mode_enabled` | ON | Master kill-switch. When OFF, the sweep daemon does nothing regardless of engagement state. |
| `vacation_mode_state` | `{}` | Full state blob (staged allowlist + engagement). Operator-managed via `engage` / `disengage` / `stage_allowlist`. |

## Composing with the rest of the system

- **Auto-revert** (`app/change_requests/auto_revert.py`) — every
  vacation-approved CR is registered for the standard 60-min
  rollback window.
- **Operator anomaly** (`app/healing/monitors/operator_anomaly.py`)
  — `new_sender` critical alerts after engagement pause the sweep
  for 24h.
- **Continuity ledger** — vacation auto-applies do NOT emit a
  dedicated `vacation_event` kind (deferred); the standard CR audit
  log + the auto-apply log are the operator-facing surface.

## Shipped follow-ons (PROGRAM §51 second batch)

All five items from the original "Deferred follow-ons" list shipped
in the same Q16 cycle:

1. ✅ **Signal slash command** — six subcommands; see "Operator
   interfaces" above.
2. ⏸ **React UI at /cp/settings** — REST endpoints in place; React
   wiring follows the existing card-pattern (still operator-driven;
   not part of this batch because the REST surface is sufficient
   for command-line and any HTTP client).
3. ✅ **End-of-vacation digest composer** — `app/vacation_mode/
   digest.py:compose_digest`; auto-called on every disengagement
   (manual or auto-expiry).
4. ✅ **Continuity-ledger event kind `vacation_window`** — 19th
   identity-event kind; emitted on engage / disengage / auto_expire.
5. ✅ **Per-pattern cross-sweep rate limit** — 6/requestor/day and
   20/global/day, reset at UTC day rollover, persisted to
   `workspace/vacation_mode/sweep_rate_limit.json`.
