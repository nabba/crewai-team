# Self-Heal v3 — Comprehensive Reference

> Shipped 2026-05-09. Builds on Self-Heal v2 (Tool Supervisor + Runbook
> Dispatcher; PROGRAM.md §14) by registering operational runbooks against
> the dispatcher, adding ten proactive monitors that observe what
> reactive runbooks can't see, surfacing the auditor's silently
> accumulating fix proposals to Signal, supervising every healing daemon
> with a watchdog reaper, and exposing the runbook-related master
> switches (`ERROR_RUNBOOKS_ENABLED`, `TOOL_SUPERVISOR_ENABLED`,
> `RECOVERY_LOOP_ENABLED`) as runtime-toggleable React controls. The
> healing-daemon switches (`HEALING_MONITORS_ENABLED`,
> `HEALING_AUDITOR_BRIDGE_ENABLED`, `HEALING_WATCHDOG_ENABLED`) are
> env-only and require gateway restart — see Master Switches §.
>
> The monitor inventory has grown from 10 → 22 since the v3 ship. See
> the Architecture diagram and `app/healing/monitors/__init__.py` for
> the current list.
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


                       app/healing/monitors/  ← 22 proactive monitors
                       │  (v3-shipped 10)
                       ├─ disk_quota          (5 min)
                       ├─ listener_heartbeat  (10 min)
                       ├─ cron_liveness       (30 min)
                       ├─ vendor_sunset       (1 wk)
                       ├─ idle_cooldown       (1 h)
                       ├─ audit_chain_check   (~1 d)
                       ├─ lock_housekeeper    (6 h)
                       ├─ adapter_lifecycle   (~30 d)
                       ├─ retention.run_chromadb / _worktrees / _attachments
                       ├─ signal_heartbeat    (1 d)
                       │  (added in subsequent passes)
                       ├─ silent_regression_detector
                       ├─ pattern_learner
                       ├─ llm_output_drift
                       ├─ signal_keepalive
                       ├─ restore_drill
                       ├─ version_upgrade_drill
                       ├─ provider_contract_drift
                       ├─ db_vacuum
                       ├─ log_archival
                       ├─ db_backup
                       ├─ crypto_rotation_drill
                       ├─ diagnosis_auto_tune              (PROGRAM §39)
                       └─ chromadb_hygiene                 (PROGRAM §40 Item 10)
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

### 12. DB backup engine + monitor (Phase A #A1, 2026-05-09; host-managed split 2026-05-16)

`app/healing/db_backup.py` — backup engine. Originally drove all
three databases from the gateway via the Docker SDK. The 2026-05-16
split moves postgres + neo4j to a host launchd LaunchAgent because
the `tecnativa/docker-socket-proxy` sidecar denies `/exec/.../start`
without an explicit `EXEC: 1` flag (which would widen the gateway's
blast radius). When `DB_BACKUP_HOST_MANAGED=1` (set in
`docker-compose.yml` by default) the gateway's `_backup_postgres` /
`_backup_neo4j` short-circuit and return a `skipped` placeholder.
ChromaDB stays gateway-owned (volume is bind-mounted; no exec needed).
Manifest at `workspace/backups/manifest.json` is shared by all
writers.

`app/healing/monitors/db_backup.py` — opt-in (default OFF) weekly
runner + per-component freshness alerter. Walks the manifest for
each of postgres/neo4j/chromadb and finds the most recent
NON-SKIPPED success; alerts when any component is older than
`DB_BACKUP_STALE_DAYS` (default 14 d). The per-component check is
what prevents a happy gateway-only chromadb stream from masking a
dead host LaunchAgent — without it, the gateway's all_ok=True
runs would have silently hidden missing pg/neo4j backups.

`deploy/scripts/backup.sh` — host-side script with three env knobs
(`BACKUP_SKIP_POSTGRES`, `BACKUP_SKIP_NEO4J`, `BACKUP_SKIP_CHROMADB`).
The Neo4j step does `docker compose stop neo4j → dump in one-shot
container → docker compose start neo4j` because Neo4j Community
refuses `neo4j-admin database dump` while the database is mounted
in a running server (~10–30 s downtime per run).

`scripts/db_backup_host.plist` + `scripts/install_db_backup.sh` —
launchd LaunchAgent (daily 04:30 local) that invokes `backup.sh`
with `BACKUP_SKIP_CHROMADB=1`. Operator install:

  ./scripts/install_db_backup.sh install
  ./scripts/install_db_backup.sh start    # smoke-test
  ./scripts/install_db_backup.sh status

Logs land at `workspace/backups/.db_backup_host.log`.

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

### 18. Restore-drill freshness monitor (Phase H #1, 2026-05-10)

`app/healing/monitors/restore_drill.py` — daily probe that watches
`workspace/backups/restore_drill_manifest.json` (written by
`deploy/scripts/restore-drill.sh`).

Closes the gap "backups exist but the restore path has never been
tested." The monitor never RUNS the drill (compose-from-inside-the-
gateway risks cross-resource issues) — operator runs it from cron /
launchd quarterly:

```
@quarterly cd /path/to/crewai-team && bash deploy/scripts/restore-drill.sh
```

Alert conditions (14-day per-tag dedup):

* **Manifest missing** → "no drill has ever run" (first-boot case).
* **Stale** → most recent drill > `RESTORE_DRILL_STALE_DAYS`
  (default 100 days).
* **Last-failed** → most recent drill's `all_ok: false`.

The drill itself runs Postgres + Neo4j + ChromaDB into a separate
compose project (`andrusai-restore-drill`), restores the freshest
`all_ok` backup set, runs smoke checks (PG row count, Neo4j node
count, Chroma heartbeat), tears down on success or failure.

### 19. Signal 120-day re-registration keepalive (Phase H #2, 2026-05-10)

`app/healing/monitors/signal_keepalive.py` — 30-day cadence,
sends a tagged self-message to keep signal-cli registration warm:

```
[andrusai-keepalive] 2026-05-10T01:23:45+00:00 — registration keepalive (~30d).
```

The `[andrusai-keepalive]` tag lets the operator filter / mute the
thread on their phone. After 3 consecutive failed keepalives
(~90 days) the monitor escalates via Signal alert — composes with
the existing `signal_heartbeat` (Wave 2 #3) PWA-push + email
escalation chain.

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

Three runbook-related env flags became runtime-toggleable from
`/cp/settings` on 2026-05-09: `ERROR_RUNBOOKS_ENABLED`,
`TOOL_SUPERVISOR_ENABLED`, `RECOVERY_LOOP_ENABLED`. Their reader
functions try `runtime_settings.get_*()` first and fall back to the
env var if the runtime-settings module raises (tests / degraded
boot). New keys are env-seeded on first read so an existing `.env`
setup keeps its current behaviour.

The three healing-daemon switches (`HEALING_MONITORS_ENABLED`,
`HEALING_AUDITOR_BRIDGE_ENABLED`, `HEALING_WATCHDOG_ENABLED`) and
the two Goodhart switches remain **env-only and require gateway
restart**. The daemons start at module import of `app.healing` (see
`app/healing/__init__.py`), so flipping the env at runtime has no
effect; redeploy with the env updated to change state. The
`/cp/settings` Self-heal-subsystems card therefore shows three
toggles, not five.

The eager wiring of `app.healing` is anchored by an explicit
`import app.healing` in `app/main.py` (next to the
`from app.healing.error_diagnosis import diagnose_and_fix` line).
This was added 2026-05-10 to remove a structural fragility — the
22-monitor driver, the auditor bridge, and the watchdog all
previously depended on a single transitive from-import staying
eager; a refactor that lazy-imported `error_diagnosis` would have
silently disabled the entire healing surface. Do NOT remove the
explicit import without re-anchoring the eager wiring elsewhere.

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


## Q2 2026 amendments — three runbook auto-actions + runtime threshold flip

Background work in PROGRAM.md §38 elevated three monitors from
alert-only to auto-action, plus lowered one runbook threshold that
was effectively never firing. None of these go through the
auto-apply CR system (which ships dormant per `docs/AUTO_APPLY.md`)
— they are runbook-tier auto-actions in the existing self-heal
infrastructure.

### Q2-A — `db_pool_reset.min_recurrence` 5 → 1

`workspace/self_heal/runbook_settings.json` change. Pool exhaustion
at instance #1 is already actionable; the previous threshold of 5
meant the runbook never triggered in normal operation. Three
independent gates remain as safety nets:

  * RateLimiter (1800s) — at most 2 resets/hour
  * Recent-success-rate gate (≥50%) — auto-disables on persistent
    failure
  * Concurrency cap (max 1) — no overlapping dispatches

### Q2-B — `disk_quota_immediate_retention` (auto-action)

`app/healing/monitors/disk_quota.py` patched to invoke
`retention.run_chromadb()` + `run_worktrees()` + `run_attachments()`
immediately when free space drops below `HEALING_DISK_FREE_WARN_GB`,
instead of waiting days for those runners' own cadence. Per-target
failure isolation, audit-event with outcomes. The Signal alert
appended outcome line (`Auto-retention ran: chromadb=ok,
worktrees=ok, attachments=ok`) so the operator sees the result
inline. Master switch `HEALING_DISK_AUTO_RETENTION_ENABLED` (default
`true`).

### Q2-C — `model_capability_runtime_block` (auto-action)

`app/healing/handlers/model_capability.py` (B + H handlers) now
write blocked models to two new runtime_settings lists:

  * `chat_blocked_models` — populated when an embed-only model is
    routed to chat (e.g. `nomic-embed-text`). The LLM selector's
    new Step 5.5 in `app/llm_selector.py:select_model` consults
    this list at default-tier selection (the existing breaker
    check was only in the pareto fallback path).
  * `no_function_calling_models` — populated when Mem0 LLM
    extraction hits a model that doesn't accept `tool_choice`.
    Consumer subsystems consult to fall back to unstructured
    extraction.

New idempotent setters:
`add_chat_blocked_model` / `remove_chat_blocked_model` /
`add_no_function_calling_model` / `remove_no_function_calling_model`
in `app/runtime_settings.py`.

Addresses ~6,758 errors/month observed in production.

### Q2-D — `stuck_idle_diagnostic_dump` (auto-action)

`app/healing/monitors/idle_cooldown.py` patched to write
`workspace/self_heal/stuck_idle_jobs.json` whenever any job is in
deep cooldown. Snapshot includes name, remaining cooldown (seconds
+ hours), failure count, diagnosis hint (`chronic` for >15 failures
vs `long_cooldown`), operator-action recommendation.

**Forensics-only.** This auto-action does NOT clear any cooldown —
the whole point of the cooldown is to avoid storming a known-bad
upstream. Pinned by test `test_does_not_clear_any_cooldown`.

### Q2-E — Eager-wire fix (boot anchor)

Verification revealed a pre-existing latent bug:
`app.self_improvement.capability_gap_analyzer` and
`app.library_radar` had eager-start patterns at module import time
but nothing in the boot chain ever imported them — their daemons
never ran in production. `app/healing/__init__.py` now anchors all
eager-start subsystems including the new ones:

  * `app.proposal_bridge` (PROGRAM §38.1)
  * `app.change_requests.auto_revert` (PROGRAM §38.3)
  * `app.governance_notifier` (PROGRAM §38.4)
  * `app.self_improvement.capability_gap_analyzer` (latent fix)
  * `app.library_radar` (latent fix)

All anchor through the existing `from app.healing.error_diagnosis
import diagnose_and_fix` line in `app/main.py:96` (TIER_IMMUTABLE)
which already triggers the `app.healing` package init.

### Q2 master switches

| Variable | Default | Purpose |
|---|---|---|
| `HEALING_DISK_AUTO_RETENTION_ENABLED` | `true` | Q2-B: disable to revert disk_quota to alert-only behaviour |
| `CHANGE_REQUESTS_AUTO_REVERT_ENABLED` | `true` | Q2/§38.3: disable the auto-revert watcher (auto_apply CRs become unsupervised) |
| `TIER3_GOVERNANCE_NOTIFIER_ENABLED` | `true` | §38.4: disable governance_notifier daemon (Tier-3 amendments fall back to audit-only) |
| `PROPOSAL_BRIDGE_ENABLED` | `true` | §38.1: disable proposal_bridge promoter (proposals stage but never promote to CR) |

### Cross-references

  * Full change log + test results + files touched: PROGRAM.md §38
  * Auto-apply CR infrastructure (separate, dormant): `docs/AUTO_APPLY.md`
  * Q1 + Q2 narrative: `tests/test_q1_unclog.py` and
    `tests/test_q2_auto_actions.py`


## Q2 §39 amendments — structured-diagnosis CRs + Item 9 history lookup

PROGRAM.md §39 closes the May 2026 audit's "0 resolved, 1 attempted"
finding. The historical `error_diagnosis` path produced PROSE
proposals that operators had to read, parse, and apply manually.
This pipeline replaces the prose path (for `fix_type="code"`) with
a structured `(path, new_content)` proposal that goes through the
standard CR gate.

### Q2-A — Structured diagnosis pipeline

`app/healing/structured_diagnosis.py` calls Claude Sonnet 4.5 with
the error + traceback + full file content and asks for a strict
JSON output: `new_content` + `confidence` + `reasoning` (or
`declined: true` with a `decline_reason`). Multi-site bugs,
destructive patches, missing-context cases all decline → caller
falls back to the existing prose path (preserves degraded-mode
behavior).

Confidence-gated: filing a CR requires `fix.confidence >=
current_threshold()`. The threshold auto-adjusts based on operator
approval rate (see Q2-C).

`app/healing/error_diagnosis.py:_try_structured_path` wires the
structured path BEFORE the prose `create_proposal` call. Returns
True on successful CR file → caller marks the error diagnosed and
exits early.

### Q2-B — Telemetry ledger

`app/healing/diagnosis_telemetry.py` records three event kinds in
`workspace/healing/structured_diagnosis_telemetry.jsonl`:

  * `filed`     — CR created (above-threshold fix produced)
  * `declined`  — LLM declined OR confidence below threshold OR
                  guard fired (file too large / rate limit)
  * `resolution` — CR transitioned (applied / rejected / rolled-back / timeout)

Resolution rows fire automatically from
`change_requests/lifecycle.py:_maybe_emit_diagnosis_telemetry`,
which checks `cr.requestor == "error_diagnosis"` and emits from
all four resolution paths. Rollbacks count as
rejection-equivalent for the auto-tuner.

5000-line cap via `app.utils.jsonl_retention.append_with_cap`
(approx 3 years of cadence).

### Q2-C — Self-adjusting confidence threshold

`app/healing/diagnosis_auto_tune.py` registered as the 24th healing
monitor. Algorithm:

  * Target band: approval rate ∈ [0.65, 0.85]
  * Step: ±0.02 per adjustment
  * Cadence guard: at most one adjustment per 24h
  * Hysteresis: ≥ 5 NEW resolutions since last change
  * Clamps to `[floor, ceiling]` from runtime_settings

Runtime knobs (settable from React `/cp/settings`):

| Key | Default | Purpose |
|---|---|---|
| `structured_diagnosis_threshold_floor` | 0.50 | Auto-tuner can't go below |
| `structured_diagnosis_threshold_ceiling` | 0.95 | Auto-tuner can't go above |
| `structured_diagnosis_threshold_override` | None | Manual operator pin (bypasses auto-tune entirely) |
| `structured_diagnosis_auto_tune_enabled` | true | Master switch |

Read precedence in `current_threshold()`:
  1. operator override (when set)
  2. auto-tune state file (`workspace/healing/structured_diagnosis_threshold.json`)
  3. fallback default 0.70

**Signal alerting is option B**: silent on routine adjustments,
fires only when auto-tune wants to move beyond the operator-set
band (`pinned_at_floor` / `pinned_at_ceiling`), deduped 7d. Routine
auto-tune drift goes silent; pin events surface as actionable
"consider widening the band" alerts.

### Q2-D — REST + React surface

Two new endpoints under `/api/cp/config/`:

  * `GET /structured_diagnosis/state` — threshold state + 30d
    telemetry summary (filed / approved / rejected / pending
    + rolling approval rate)
  * `GET /structured_diagnosis/telemetry?window=N` — paginated
    rolling-window rows joined with their resolutions

React `StructuredDiagnosisCard` in `SettingsPage.tsx` renders the
state inline: active threshold, recent approval rate, telemetry
breakdown, floor/ceiling inputs, override pin/clear, auto-tune
toggle.

### Q2-E — HOT-1 observation hook

Every `generate_structured_fix` invocation emits one row to
`workspace/subia/observations/metacognitive_repair.jsonl` (capped
at 5000 lines). Schema documented in
`docs/CONSCIOUSNESS_HOT1_OBSERVATIONS.md`. The future HOT-1 Butlin
indicator probe will read this log to compute its score; for now
the log is **passive collection only** — adding it costs nothing
and gives the future probe months of pre-built observations to
score against. Goodhart-of-the-indicator is explicitly avoided:
the auto-tuner optimises for operator approval rate, not for
hypothesis length / diversity.

### Q2-F — Item 9: continuity-ledger lookup

`app/identity/relevant_history.py:relevant_history(path,
window_days=90)` aggregates path-keyed history from BOTH the
continuity ledger AND the CR audit log. Wired into:

  * `change_requests.lifecycle.create_request` — appends a "📜 Recent
    activity on this path" markdown block to `cr.reason`
  * `tools.request_tier3_amendment` — passes the lookup as
    `relevant_history_90d` in `extra_evidence`
  * `governance_notifier` — Signal alerts include a one-line
    history summary

Read-only against both sources. The continuity ledger remains a
narrative artefact emitted by identity-shaping events, not augmented
by retrospective queries.

### Q2 master switches recap

| Variable | Default | Purpose |
|---|---|---|
| `structured_diagnosis_auto_tune_enabled` (runtime_settings) | `true` | Master switch for the auto-tuner |
| `STRUCTURED_DIAGNOSIS_TELEMETRY_LOG` (env) | `workspace/healing/structured_diagnosis_telemetry.jsonl` | Override telemetry log path (test fixtures) |
| `STRUCTURED_DIAGNOSIS_THRESHOLD_STATE` (env) | `workspace/healing/structured_diagnosis_threshold.json` | Override state-file path |
| `HOT1_OBSERVATION_LOG` (env) | `workspace/subia/observations/metacognitive_repair.jsonl` | Override HOT-1 log path |


## §53 amendments — destructive_advisory guardrail + audit-driven monitor hardening (2026-05-16)

Same-day post-incident hardening pass (full change-log in
PROGRAM.md §53). Two live-data incidents (migration drill corrupting
postgres; chromadb_hygiene mis-classifying live segment dirs as
orphans) prompted an audit of every monitor under
`app/healing/monitors/`. Audit doc:
`docs/AUDIT_2026_05_16_DESTRUCTIVE_MONITORS.md`. Eight priority
items closed across PRs #113 / #114 / #116 / #117.

### §53.A The structural guardrail — `app/healing/destructive_advisory.py`

A monitor whose alert recommends a destructive action constructs a
`DestructiveAdvisory` dataclass. Construction REFUSES if any of the
five discipline fields is missing or if the snapshot file is not on
disk:

| Field | Discipline |
|---|---|
| `snapshot_path` | Pre-action tar snapshot, MUST exist before the alert is sent, MUST be > 100 bytes (refuses 0-byte placeholders) |
| `apply_command` | Operator-runnable shell command (paste-and-run) |
| `undo_command` | Reversal path — typically `tar -xzf <snapshot>` |
| `verify_command` | Schema check operator runs BEFORE acting. The chromadb_hygiene incident would have been caught here — verify would have run the same query the monitor used and shown 0 orphans |
| `schema_assumption` | One sentence on what classification depends on |

`format()` puts verify BEFORE apply in the alert body — operator's
eye lands on verification first. `emit()` routes through the
standard Signal alert + audit log path with a dedicated
`destructive_advisory:<monitor_name>` tag for arbiter dedup.
`snapshot_paths()` is the canonical tar helper.

Use sites today: `chromadb_hygiene._alert_orphan_segments` (the
load-bearing example — orphan dirs are snapshotted to
`workspace/.snapshots/chromadb_orphans_<ts>.tar.gz` before any
destructive advice reaches the operator).

### §53.B Monitor changes

| Monitor | Change | PR |
|---|---|---|
| `chromadb_hygiene` (orphan scan) | Query `segments.id` not `collections.id`; alert text updated; 8 dirs become true orphans (down from 50+ false positives) | #113 |
| `chromadb_hygiene` (orphan alert) | Routes through `DestructiveAdvisory`; pre-snapshot tarball lives in `workspace/.snapshots/` | #114 |
| `chromadb_hygiene` (VACUUM) | Per-path consecutive-failure tracking; one-shot Signal alert after 4 quarters of consecutive failure | #117 |
| `db_vacuum` | Same chronic-failure shape for `conversations.db` | #117 |
| `retention.run_chromadb` | `_oldest_ids` returns `(ids, stats)`; records lacking timestamp excluded (not silently classified as oldest); audit row surfaces skipped count | #114 |
| `retention.run_worktrees` | `_validate_worktree_path(wt, *, expected_session_id)` before rmtree — refuses unless absolute + inside worktree_root + basename match + 1 segment deep | #116 |
| `retention.run_attachments` | `_is_attachments_dir_safe(p)` refuses unless `SIGNAL_ATTACHMENTS_DIR` resolves inside `/app/attachments`, `/app/workspace/`, or `/tmp/` | #116 |
| `lock_housekeeper` | `_LOCK_RULES` per-directory basename patterns; unknown-shape `.lock` files alerted + never auto-deleted | #116 |
| `log_archival` | Retention floor 7d → 30d; `_PURGE_PER_PASS_CAP = 100` oldest-first | #116 |
| `tz_drift` | `_synthesize_zoneinfo_patch(src)` produces a real CR diff (no longer empty); refuses on unexpected source shape | #117 |

### §53.C The discipline (codified)

Future monitors that emit destructive recommendations:

1. **Snapshot first.** Take a tar snapshot of the targets BEFORE
   emitting the alert. The operator sees the snapshot path in the
   alert text and has a one-line undo command.
2. **Verify-before-act.** Include a concrete shell command the
   operator can run to validate the monitor's classification
   against the current state of the world.
3. **Declare the assumption.** One sentence on what schema
   assumption the classification depends on.
4. **Route through `DestructiveAdvisory`.** The dataclass refuses
   construction if any of (1)-(3) is missing — discipline enforced
   at compile time, not relied on memory of past incidents.

### §53.D Tests

Three new pinning files, 74 tests, all pass without gateway-deps env:

* `tests/test_destructive_advisory.py` (19) — snapshot helper
  edges, dataclass refusal per-field, format ordering (verify
  before apply), emit integration.
* `tests/test_risk_items_followup.py` (22) — path-validation
  refusals, safe-prefix refusals incl. suffix-attack, lock-rule
  basenames, source-level ordering pins.
* `tests/test_remaining_risk_items.py` (15) — chronic-failure
  source-level pins, synth-helper pure-function tests
  (`ast.parse`-verified output), refusal-shape tests.

Plus the regression pin in `tests/test_q3_2_cleanup.py`:
`test_orphan_scan_does_not_flag_segment_dirs_2026_05_16_regression`
exercises the exact incident scenario — a `segments` row whose UUID
matches a dir on disk must NEVER be flagged orphan, regardless of
`collections.id`.
