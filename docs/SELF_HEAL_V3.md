# Self-Heal v3 — Comprehensive Reference

> Shipped 2026-05-09. Builds on Self-Heal v2 (Tool Supervisor + Runbook
> Dispatcher; PROGRAM.md §14) by registering operational runbooks against
> the dispatcher, adding ten proactive monitors that observe what
> reactive runbooks can't see, surfacing the auditor's silently
> accumulating fix proposals to Signal, supervising every healing daemon
> with a watchdog reaper, and exposing the master switches as
> runtime-toggleable React controls.
>
> See PROGRAM.md §23 for the chronological change-log entry.

## Architecture in one diagram

```
                   error_monitor.py (errors.jsonl scanner)
                                │
                                ▼
                       app/healing/runbooks.py  (TIER_IMMUTABLE — dispatcher)
                                │
                                ▼  (matches signature → calls handler)
                       app/healing/handlers/   ← 6 reactive runbooks
                       (db_pool, scheduler, schema_drift, code_drift,
                        model_capability, multi_router)
                                │
                       ┌────────┼────────┐
                       │        │        │
                  Signal alert  CR file  audit log
                                │
                  (operator action: /cp/changes for CRs)


                       app/healing/monitors/  ← 10 proactive monitors
                       │
                       ├─ disk_quota          (5 min)
                       ├─ listener_heartbeat  (10 min)
                       ├─ cron_liveness       (30 min)
                       ├─ vendor_sunset       (1 wk)
                       ├─ idle_cooldown       (1 h)
                       ├─ audit_chain_check   (~1 d)
                       ├─ lock_housekeeper    (6 h)
                       ├─ adapter_lifecycle   (~30 d)
                       ├─ retention.run_chromadb / _worktrees / _attachments
                       └─ signal_heartbeat    (1 d)
                                │
                       Single daemon driver in __init__.py
                       (60 s tick + per-monitor cadence guards)
                                │
                       Watched by app/healing/watchdog.py
                                │
                       (re-spawns dead daemons; gives up after
                        3 crashes/hr; emits Signal alert + Heartbeat
                        footprint at workspace/healing/watchdog_heartbeat
                        consumed by cron_liveness)


                       app/healing/auditor_bridge.py
                       (polls audit_journal.json → Signal + CR mirror)


                       app/safe_io.py
                       (raises DiskQuotaError below DISK_FREE_THRESHOLD_MB)


                       app/coding_session/audit_verify.py
                       (read-only chain integrity for the existing
                        coding-sessions audit log; called by audit_chain_check)
```

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

### 2. Seven proactive monitors under `app/healing/monitors/`

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
| `audit_chain_check` ⭐ | ~daily | Coding-session audit-chain integrity. Reads `workspace/coding_sessions/audit.jsonl` via `app.coding_session.audit_verify` and alerts on any chain break (tampered payload, prev-hash mismatch, malformed JSON). Read-only — never modifies the chain. |
| `lock_housekeeper` ⭐ | 6 h | Orphaned `.lock` files. Walks `workspace/locks/`, `workspace/dreams/`, `workspace/`. fcntl-probes each to confirm uncontested before deletion (defends against PID-reuse) AND requires age ≥ 1 h. Pile-up alert at >50 files. |

⭐ added 2026-05-09 as Wave 1 of the resilience-gap closure plan.

Tunable via env vars; defaults are conservative (alerts fire only on
sustained signal, not transients).

### 2a. Disk-quota guard in `app.safe_io` (Wave 1, 2026-05-09)

`safe_write` and `safe_append` now refuse to write when free disk space
on the target volume drops below `DISK_FREE_THRESHOLD_MB` (default
200 MB). Refusal raises `DiskQuotaError`, a subclass of `OSError` so
existing handlers route correctly. Set the env var to `0` to disable
the check entirely; the guard fails OPEN if `shutil.disk_usage` itself
errors. Each refusal is best-effort audited as
`actor='safe_io', action='disk_quota_block'`.

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

### 4. Auditor → CR + Signal bridge

(See `app/healing/auditor_bridge.py`.) Polls `audit_journal.json` for
`error_fix_proposed` events; emits one Signal alert per (pattern,
attempt) with a deep-link to `/cp/proposals`, AND files a CR mirror
at `docs/proposed_fixes/<pattern>__attempt_<n>.md` so approving
puts a permanent paper trail in the repo. CR failure is non-fatal
(Signal alert always goes out).

### 5. Daemon-thread watchdog (Wave 2 #7)

`app/healing/watchdog.py` is the reaper for everything above. 60 s
loop walks `threading.enumerate()`; for any registered daemon that's
not alive, calls its `start()` (idempotent). 3-crashes-per-hour
backoff with give-up + Signal alert. Touches a heartbeat footprint
at `workspace/healing/watchdog_heartbeat` so the cron_liveness
monitor can detect a watchdog death (closes the "watches the
watchman" gap).

For the watchdog's re-spawn to actually work, every supervised
`start()` had to be made truly idempotent — gating on thread
liveness via `threading.enumerate()` rather than a stale `_started`
flag. Updated:
`app/healing/monitors/__init__.py:start`,
`app/healing/auditor_bridge.py:start`,
`app/healing/watchdog.py:start`.

### 6. Disk-quota guard (Wave 1 #1)

`app/safe_io.py` `safe_write` and `safe_append` raise
`DiskQuotaError(OSError)` when free space drops below
`DISK_FREE_THRESHOLD_MB` (default 200 MB). Probes fail OPEN. Each
refusal audited as `actor='safe_io', action='disk_quota_block'`.

### 7. Coding-session audit-chain verifier (Wave 1 #5)

`app/coding_session/audit_verify.py` walks
`workspace/coding_sessions/audit.jsonl` forward from genesis,
recomputing each `entry_hash`. Read-only — never modifies the chain.
Wired through the daily `audit_chain_check` monitor; alerts on
breaks (cooldown 24 h or new break-line).

### 8. Lock-file housekeeper (Wave 1 #9)

`app/healing/monitors/lock_housekeeper.py` walks watched dirs for
`.lock` files. Two-condition guard before unlink: age > 1 h AND
fcntl-uncontested (defends against PID reuse). Pile-up alert at
>50 files (signals a leak); 24 h cooldown.

### 9. Adapter lifecycle (Wave 2 #4)

`app/training/adapter_lifecycle.py` (open-tier) does monthly
orphan cleanup of `workspace/training_adapters/` +
`workspace/trained_models/`, dead-pointer detection (registry
entries whose path doesn't exist on disk), bloat alert >5 GB,
history snapshot trail at
`workspace/healing/adapter_lifecycle_history.jsonl`.

### 10. Retention jobs (Wave 2 #8)

`app/healing/monitors/retention.py`. Three independent monitors:

  * `run_chromadb` (weekly) — per-collection record cap (default
    100k); deletes oldest by metadata timestamp.
  * `run_worktrees` (daily) — terminal coding-session sessions
    (submitted / discarded / expired / failed) >7 d old → unlink
    JSON + rmtree worktree.
  * `run_attachments` (daily) — Pass 1 age-delete (>30 d), Pass 2
    size-cap oldest-first (>1 GB total).

`RETENTION_DRY_RUN=true` for the first month is recommended.

### 11. Signal-channel heartbeat (Wave 2 #3)

`app/healing/monitors/signal_heartbeat.py` watches
`workspace/conversations.db` mtime (inbound proxy) +
`workspace/signal_outbound.json` (outbound proxy). Flags
asymmetric (recent-inbound + stale-outbound > 7 d) or likely-dead
(no traffic > 7 d). Multi-channel escalation: Signal → PWA push
after 3 fails → email after 7 fails. Streak resets on recovery.

### 12. DB backup engine + monitor (Phase A #A1, 2026-05-09)

`app/healing/db_backup.py` — Postgres + Neo4j + ChromaDB backup
engine driven from the gateway via Docker SDK (postgres + neo4j
exec inside sibling containers; chromadb tarball directly).
Manifest at `workspace/backups/manifest.json` records every run.

`app/healing/monitors/db_backup.py` — opt-in (default OFF) weekly
runner + freshness alerter at >14 d since last successful backup.

`deploy/scripts/backup.sh` — operator-runnable equivalent (host-side
docker exec) for environments where the gateway isn't the backup
runner.

See `crewai-team/deploy/RESTORE.md` for the operator-facing restore
runbook.

### 13. conversations.db monthly VACUUM (Phase A #A6, 2026-05-09)

`app/healing/monitors/db_vacuum.py` — daily probe + 30-day internal
cadence. Calls `app.conversation_store.vacuum()` which reclaims
SQLite pages freed by `prune_old_*`. Alerts when >50 MB freed.

### 14. Log archival (Phase A #A5 + Phase D #2, 2026-05-09)

`app/healing/monitors/log_archival.py` — daily pass:

* `errors.jsonl.{1,2,3}` rotated suffixes → gzip into monthly
  archive at `workspace/logs/archive/<YYYY-MM>.jsonl.gz`.
* `audit_journal.json` >10 MB → gzip + truncate.
* `workspace/shinka_results/run_*` dirs >90 days → tarball into
  `workspace/shinka_results/archive/<run>.tar.gz` (Phase D #2).
* All archive subdirs purged at `LOG_ARCHIVE_RETENTION_DAYS`
  (default 90).

### 15. Silent-regression detector (Phase C #2, 2026-05-09)

`app/healing/silent_regression_detector.py` — 4 h monitor. Walks
`workspace/audit_journal.json` for cron-like events (error_resolution,
code_audit, self_improve, evolution, retrospective, benchmark_snapshot,
workspace_sync). For each, computes 13-day baseline rate; alerts
on last-24h count <70% of baseline. Pairs the alert with recent
git commits + control-plane CR/ratchet/amendment actions as
suspects so the operator gets the WHAT and a likely WHY.

### 16. Failure-pattern learner (Phase C #4, 2026-05-09)

`app/healing/pattern_learner.py` — daily pass over
`workspace/logs/errors.jsonl`. SHA-1 signature clustering using
the same `compute_signature()` the runbook dispatcher uses.
Filters out signatures already covered by a registered runbook
(by reading `_REGISTERED_RUNBOOKS` directly). Flags signatures
with ≥10 occurrences in the last 7 days; writes a markdown
scaffold to `docs/proposed_fixes/learner_<sig>.md` for the
operator to flesh into a real handler.

### 17. Semantic LLM-output drift detector (Phase D #6, 2026-05-09)

`app/healing/llm_output_drift.py` — weekly checkpoint. Runs a
fixed set of golden probes (`workspace/healing/llm_drift_probes.json`,
defaults at module-level), captures answer + embedding (real
chroma if available, hash-trick fallback), persists baseline at
`workspace/healing/llm_drift_baseline.json`. Subsequent runs cosine-
compare new vs baseline; alerts at avg < 0.85.

Embedder source recorded on each baseline entry; cross-source
comparisons (chroma 768-d ↔ hash 256-d) refuse to compare and
instead alert "rebase needed" — the dim/scale mismatch would
otherwise cosine to 0 and fire false-positive drift.

---

## Wiring

`app/main.py` already imports `from app.healing.error_diagnosis import
diagnose_and_fix`, which triggers `app/healing/__init__.py`. That now
imports four side-effect modules:

```
app/main.py (TIER_IMMUTABLE — no edits required)
   └─ from app.healing.error_diagnosis import diagnose_and_fix
       └─ triggers app/healing/__init__.py
            ├─ from app.healing import handlers          # registers runbooks
            ├─ from app.healing import monitors           # starts monitors daemon
            ├─ from app.healing import auditor_bridge     # starts auditor-bridge daemon
            └─ from app.healing import watchdog           # starts watchdog reaper
```

No TIER_IMMUTABLE / TIER_GATED files were modified for the wiring.
Reader updates (master-switch sources) ARE Tier-3 edits — operator-
authorized as part of the React-toggle work below.

---

## Master switches

Five env-only flags became runtime-toggleable from `/cp/settings`
on 2026-05-09. The reader functions try `runtime_settings.get_*()`
first and fall back to the env var if the runtime-settings module
raises (tests / degraded boot). New keys are env-seeded on first
read so an existing `.env` setup keeps its current behaviour.

| Variable | Reader location | React toggle | Default |
|---|---|---|---|
| `ERROR_RUNBOOKS_ENABLED` | `app/healing/runbooks.py:runbooks_enabled` | Self-heal subsystems → Runbook dispatcher | `false` |
| `TOOL_SUPERVISOR_ENABLED` | `app/tool_runtime/supervisor.py:is_enabled` | Self-heal subsystems → Tool exception supervisor | `false` |
| `RECOVERY_LOOP_ENABLED` | `app/recovery/loop.py:is_enabled` | Self-heal subsystems → Refusal recovery loop | `false` |
| `HEALING_MONITORS_ENABLED` | `app/healing/monitors/__init__.py:_enabled` | (env-only) | `true` |
| `HEALING_AUDITOR_BRIDGE_ENABLED` | `app/healing/auditor_bridge.py:_enabled` | (env-only) | `true` |
| `HEALING_WATCHDOG_ENABLED` | `app/healing/watchdog.py:_enabled` | (env-only) | `true` |
| `GOODHART_HARD_GATE_DISABLED` | `app/governance.py:_goodhart_hard_gate_disabled` | Goodhart hard gate → Off | `false` |
| `GOODHART_HARD_GATE_ENFORCING` | `app/governance.py:_goodhart_hard_gate_enforcing` | Goodhart hard gate → Enforcing | `false` |

Per-runbook on/off lives in `workspace/self_heal/runbook_settings.json`,
toggleable via the runbook list in the React Self-heal subsystems
card OR by direct file edit.
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
