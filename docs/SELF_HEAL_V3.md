# Self-Heal v3 — Operational Runbooks + Proactive Monitors + Auditor Bridge

> Shipped 2026-05-09. Builds on Self-Heal v2 (Tool Supervisor + Runbook
> Dispatcher; PROGRAM.md §14) by registering operational runbooks against
> the dispatcher, adding five proactive monitors that observe what
> reactive runbooks can't see, and surfacing the auditor's silently
> accumulating fix proposals to Signal.

---

## What this PR ships

### 1. Six operational runbooks under `app/healing/handlers/`

The runbook framework in `app/healing/runbooks.py` (TIER_IMMUTABLE) was
shipped with only a `log_only` no-op reference handler. This wires real
handlers against the SHA-1 signatures the error-monitor produces.

| File | Runbook(s) | Pattern | Action |
|---|---|---|---|
| `db_pool.py` | `db_pool_reset` | Specific hash | Calls `app.control_plane.db._reset_pool()`. Rate-limited 1/30 min internally. **Auto-apply.** |
| `scheduler_overrun.py` | `apscheduler_overrun_alert` | Specific hash | Tracks per-job overruns; Signal alert after 3 in 24 h. |
| `schema_drift.py` (E) | `numeric_overflow_widen_cr` | Specific hash | **CR-gated:** files a change-request with a generated widening migration (`migrations/<ts>_widen_numeric_*.sql`). Operator approves via Signal 👍 / `/cp/changes`. |
| `schema_drift.py` (F) | `schema_missing_column_cr` (via router) | substring | **CR-gated:** files a marker migration file (`migrations/<ts>_missing_column_<col>.sql`) telling the operator to run `alembic upgrade head`. One CR per missing column; deduped after first filing. |
| `code_drift.py` (C) | `cost_mode_undefined_alert` (×2) | Specific hashes | Signal alerts; deduped 1/day. (Auto-CR would be wrong without source-level analysis of intent.) |
| `code_drift.py` (G) | `anthropic_str_content_cr` | Specific hash | **CR-gated:** files `app/llms/anthropic_response_guard.py` with a defensive `coerce_response()` helper. Operator wires it into the call path after approving. |
| `model_capability.py` (B + H) | (functions consumed by router) | n/a | Tracks misrouted models in `workspace/self_heal/model_capabilities.json`. |
| `multi_router.py` | `self_heal_router` | `.*` (catch-all, registered LAST) | Routes by sample-substring to: embed-misroute (B), no-function-calling (H), missing-column (F). |
| `__init__.py` | (registration) | — | Unregisters default `log_only`, installs handlers in correct insertion order, re-registers a fallback `log_only` last. |

Together these address ~99% of the recurring error volume in
`workspace/logs/errors.jsonl`.

### 2. Five proactive monitors under `app/healing/monitors/`

Reactive runbooks fire on errors. Some failure modes never throw — these
monitors close that gap. All run in a single daemon thread with per-monitor
cadences.

| Monitor | Cadence | Detects |
|---|---|---|
| `disk_quota` | 5 min | Workspace free-space below WARN/CRIT thresholds. |
| `listener_heartbeat` | 10 min | Newest workspace activity older than 30 min → gateway / pollers stuck. |
| `cron_liveness` | 30 min | APScheduler cron jobs whose footprint hasn't been touched in 3× expected interval. |
| `vendor_sunset` | 1 week | In-use models no longer listed by their provider's `/v1/models`. |
| `idle_cooldown` | 1 hour | Idle-scheduler jobs with `skip_until` > 24 h or `failures` > 15. |

Tunable via env vars; defaults are conservative (alerts fire only on
sustained signal, not transients).

### 3. Auditor bridge under `app/healing/auditor_bridge.py`

The single biggest finding in the audit: `auditor.run_error_resolution`
has been running every 30 min for the past month, logging
*"0 resolved, 1 attempted, 23 total patterns"* — diagnoses produced,
proposals written, never read. This bridge polls
`workspace/audit_journal.json` every 5 min for `error_fix_proposed`
events and surfaces each proposal two ways:

  1. **Signal alert** — single message per (pattern, attempt), with the
     proposal text + CR id + deep-link to `/cp/proposals`.
  2. **CR mirror** — files a change-request landing
     `docs/proposed_fixes/<pattern>__attempt_<n>.md` with a structured
     markdown record. Approving the CR creates a permanent paper trail
     in the repo of "the auditor proposed X for pattern Y on date Z."
     Rejecting the CR signals the bridge to try a different angle on
     the next attempt.

Dedup window: 14 days. Considers proposals from the last 7 days.
CR-system failure is non-fatal — Signal alert always goes out so the
operator is never blind because of CR plumbing trouble.

---

## Wiring

`app/main.py` already imports `from app.healing.error_diagnosis import
diagnose_and_fix`, which triggers `app/healing/__init__.py`. That now
also imports `handlers`, `monitors`, and `auditor_bridge` — each
self-registers via import side-effects. **No TIER_IMMUTABLE files were
modified.**

```
app/main.py (IMMUTABLE)
   └─ from app.healing.error_diagnosis import diagnose_and_fix
       └─ triggers app/healing/__init__.py
            ├─ from app.healing import handlers          # registers runbooks
            ├─ from app.healing import monitors           # starts daemon
            └─ from app.healing import auditor_bridge     # starts daemon
```

---

## Master switches

You need to flip three env flags for self-heal to actually take action.
None require code changes.

| Variable | Default | Purpose |
|---|---|---|
| `ERROR_RUNBOOKS_ENABLED` | `false` (off) | Master gate for the runbook dispatcher. **Set this `true` to enable runbooks at all.** |
| `TOOL_SUPERVISOR_ENABLED` | `false` (off) | Tool exception classify→retry→substitute (already shipped). Recommend `true`. |
| `RECOVERY_LOOP_ENABLED` | `false` (off) | LLM-refusal recovery (already shipped). Recommend `true`. |
| `HEALING_MONITORS_ENABLED` | `true` (on) | Master gate for the new monitors. Set `false` to disable the daemon. |
| `HEALING_AUDITOR_BRIDGE_ENABLED` | `true` (on) | Master gate for the auditor → Signal bridge. |
| `HEALING_DISK_FREE_WARN_GB` | `5.0` | Disk-quota WARN threshold. |
| `HEALING_DISK_FREE_CRIT_GB` | `1.0` | Disk-quota CRITICAL threshold. |
| `HEALING_LISTENER_STALE_MIN` | `30` | Listener-heartbeat staleness threshold. |
| `HEALING_VENDOR_SUNSET_ENABLED` | `true` | Set `false` if outbound HTTP is restricted. |

Per-runbook `enabled` + `min_recurrence` live in
`workspace/self_heal/runbook_settings.json` — already populated with
sensible defaults.

To enable everything: in your `.env` (or whichever config the gateway
reads), add:

```bash
ERROR_RUNBOOKS_ENABLED=true
TOOL_SUPERVISOR_ENABLED=true
RECOVERY_LOOP_ENABLED=true
```

(Monitor + auditor-bridge daemons are ON by default.)

---

## Observability

- **Audit trail.** All handlers and monitors emit
  `actor='self_heal_handler'` audit events. Query:
  `gh api ... /api/cp/audit?actor=self_heal_handler` (or hit the `/cp/audit`
  React page).
- **State files.** `workspace/self_heal/` accumulates per-handler state:
  `runbook_settings.json`, `runbook_stats.json`, `model_capabilities.json`,
  `scheduler_overruns.json`, `schema_drift.json`, `code_drift_alerts.json`,
  `disk_quota_alerts.json`, `listener_heartbeat_alerts.json`,
  `cron_liveness_alerts.json`, `vendor_sunset.json`, `idle_cooldown_alerts.json`,
  `auditor_bridge.json`. All atomic-written, all best-effort, all human-readable.
- **Signal alerts.** Each handler tags its alerts (`tag="db_pool_reset"`
  etc.) so you can grep your message history for self-heal activity.

---

## Safety properties

The runbook framework's seven dispatch gates apply to every handler:

1. Master env flag (`ERROR_RUNBOOKS_ENABLED`).
2. Severity must be ≥ `warning`.
3. Pattern (signature) must match.
4. Per-runbook `enabled: true` in settings.
5. Recurrence ≥ `min_recurrence`.
6. Recent success-rate ≥ 50% (auto-degraded after repeated failures).
7. Concurrency cap of 1 in flight.

Plus, every handler:

- Is wrapped in try/except — a bug never propagates back to the dispatcher.
- Sample-substring guards confirm the SHA-1 hash isn't a collision before
  taking action.
- Never modifies TIER_IMMUTABLE files. Schema-drift / code-drift fixes
  go through the `change_requests` system, which validates against the
  immutable list before persisting.
- Per-handler rate limiters prevent self-DOS on sustained spikes.

---

## What's deliberately NOT auto-applied

| Gap | Why not auto-applied |
|---|---|
| Code fixes from `auditor` proposals | Auditor outputs free-form descriptions, not (path, content) tuples. Surfacing to Signal lets the operator decide. |
| Vendor model migration | High-risk; operator approves via change-request after seeing the deprecation list. |
| Idle-job cooldown clearing | The cooldown exists for a reason (3 consecutive failures). Auto-clearing would re-storm a known-bad upstream. |
| Schema migrations (E, F) | E files a CR with the proposed migration scaffold — operator fills in TABLE/COLUMN before approving. F is an alert recommending `alembic upgrade head`. |
| Adapter retirement, governance threshold ratchet | These need Tier-3 amendment protocol; out of scope for this PR. |

---

## Tests

`tests/healing/test_handlers.py` and `tests/healing/test_monitors.py`
cover:

- `compute_signature` matches the live `_signature` formula.
- `db_pool_reset` calls `_reset_pool` exactly once per allowed window.
- Multi-router dispatches to correct sub-handler by substring.
- Specific-hash handlers refuse on sample mismatch (no false-positive
  on hash collision).
- Disk quota / listener heartbeat / cron liveness alert thresholds fire
  exactly once per cooldown.
- Auditor bridge dedup keys behave correctly across re-runs.

Run with:

```
pytest tests/healing/ -v
```

---

## Rollback

To disable everything without touching code:

```bash
unset ERROR_RUNBOOKS_ENABLED   # disables runbook dispatch
HEALING_MONITORS_ENABLED=false # disables monitor daemon
HEALING_AUDITOR_BRIDGE_ENABLED=false  # disables Signal bridge
```

Or, surgically:

- `workspace/self_heal/runbook_settings.json` → set `"enabled": false`
  on individual runbooks. Effect is immediate (re-read every dispatch).
