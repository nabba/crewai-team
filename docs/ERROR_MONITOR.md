# Permanent Error Monitor

Surfaces signature-grouped error patterns and detected anomalies from
`workspace/logs/errors.jsonl` to the React dashboard at `/cp/ops` →
"📈 Error Monitor" tab. Shipped May 2026.

---

## 1. Why this design exists

Before May 2026 the only window into runtime errors was `errors.jsonl` —
a 21 MB JSON-lines file written by `app/error_handler.py` from every
`logger.error` and `logger.warning` site. Tailing it manually surfaced a
real signal (which fixes worked, which didn't) but had three weaknesses:

1. **No grouping.** A single root cause (e.g. OpenRouter routing through
   its "Stealth" provider class) would emit 3 000+ near-identical
   records over a few days, drowning out other signals.
2. **No baseline.** Anomalies were only visible by manual time-series
   analysis: was 50 errors/hour normal, or 10× the steady state?
3. **No surface in the dashboard.** The existing `/cp/ops` tabs covered
   the self-heal journal, the 2σ metric anomaly detector, and the deploy
   pipeline — but the raw error log itself had no UI.

The monitor closes the loop by reading errors.jsonl on a schedule,
extracting a stable per-error signature, and letting three statistical
detectors flag spikes / new patterns / total-rate σ-anomalies.

---

## 2. Architecture (one screen)

```
errors.jsonl  ──>  error_monitor.scan()  ──>  pattern signatures
                          │                          │
                          ├──>  feeds metrics  ──>  app.anomaly_detector
                          │     (per-pattern rate)  (existing 2σ engine)
                          │
                          └──>  detects spikes/new  ──>  control_plane.error_anomalies
                                                              │
                                                              ▼
                                          GET /api/cp/error_audit  ──>  React <ErrorMonitor>
                                                                         (default tab on /cp/ops)
```

The whole thing is a single backend module
(`app/observability/error_monitor.py`), one Postgres table
(`control_plane.error_anomalies`, migration 034), two FastAPI routes
under `/api/cp/`, and one React component
(`dashboard-react/src/components/ErrorMonitor.tsx`).

---

## 3. Pattern signature extraction

Every error record is reduced to a stable 16-character SHA-1 prefix:

```python
module = record["module"].lower()         # e.g. "db", "completion"
msg    = record["message"][:200]          # raw first 200 chars
norm   = strip_ids_timestamps_paths(msg)  # see _STRIP_PATTERNS
sig    = sha1(f"{module}::{norm[:120]}")[:16]
```

`_STRIP_PATTERNS` collapses these varying parts so structurally-identical
errors share a signature:

- UUIDs → `<uuid>`
- ISO timestamps → `<ts>`
- Bare integers → `<n>`
- Single/double-quoted strings → `'<str>'` / `"<str>"`
- File paths ending `.py | .md | .json | .yaml | .sql` → `<path>`

The first event for a signature also stores the original message
unredacted as `pattern_sample` so the dashboard shows real context.

---

## 4. Detection rules

| Rule | Condition | Severity scaling |
|---|---|---|
| `new_pattern` | Signature first observed in last 60 min AND > 5 occurrences in that window | Ratio of current vs (very small) baseline |
| `rate_spike`  | Existing signature's 1h count > 3× its 24h rolling avg AND > 5/hour absolute | 3× → info, 5× → warning, 10× → critical |
| `total_rate`  | Total errors/hour > 2σ from 24h rolling mean (delegated to `app/anomaly_detector.py`) | Surfaces via existing `/api/cp/anomalies` too |

**Auto-resolve.** An open anomaly transitions to `resolved` when its
signature's rate falls below 50 % of the detection threshold for two
consecutive scans (≈10 min).

**Acknowledge.** A user can manually mark `open` → `acknowledged` via
the dashboard's "Ack" button. Distinct from auto-resolve: history is
preserved, the anomaly is silenced for the user but not declared healed.

---

## 5. Lifecycle

```
open  ──(rate falls < 50% × threshold for 2h)──>  resolved
  │
  └──(user clicks "Ack" in dashboard)──>  acknowledged
```

Once an anomaly is `resolved` or `acknowledged`, the same signature
firing again later creates a *new* anomaly row — the dashboard de-dupes
on `(signature, status='open')` so you don't get duplicate cards mid-event.

---

## 6. State persistence

| Path | Contents | Survives restart |
|---|---|---|
| `workspace/observability/error_monitor_state.json` | Last-read byte offset of errors.jsonl + cap of 5 000 most-recent known signatures | ✅ |
| `control_plane.error_anomalies` (Postgres) | Anomaly history with `open` / `acknowledged` / `resolved` lifecycle | ✅ |
| In-memory rolling window | 24 h of (timestamp, signature) tuples; capped at 50 000 events / 2 000 per signature | ❌ — back-filled from the last 5 MB of errors.jsonl on first scan after process start |

**Warm-up back-fill.** Without warm-up, the first 5 minutes after every
restart would show "0 errors" and rate-spike detection would have no
baseline. `_warmup_window()` reads the last 5 MB of errors.jsonl on
first scan and marks every observed signature as known — this prevents
spurious "new pattern" alarms on signatures that have actually been
around for hours/days.

---

## 7. File map

| Layer | Path | Notes |
|---|---|---|
| Backend module | `app/observability/error_monitor.py` | `scan()`, `snapshot()`, `acknowledge(id)`, `_warmup_window()` |
| Schema | `migrations/034_error_anomalies.sql` | `control_plane.error_anomalies` + 2 indexes |
| API | `app/control_plane/dashboard_api.py` | `GET /api/cp/error_audit` + `POST /api/cp/error_audit/anomaly/{id}/acknowledge` |
| Cron | `app/main.py` | `scheduler.add_job(..., minutes=5, next_run_time=now)` so first scan is immediate |
| React component | `dashboard-react/src/components/ErrorMonitor.tsx` | Polls every 30 s |
| React types/queries | `dashboard-react/src/api/queries.ts` | `useErrorAuditQuery`, `useAcknowledgeAnomaly`, `ErrorAudit*` types |
| OpsPage tab | `dashboard-react/src/components/OpsPage.tsx` | Default landing tab on `/cp/ops` |

---

## 8. Configuration

| Setting | Default | Where |
|---|---|---|
| Scan interval | 5 min | `app/main.py` (`scheduler.add_job(...)`) |
| React poll interval | 30 s | `dashboard-react/src/api/queries.ts` (`POLL.verySlow`) |
| Spike ratio threshold | 3× | `error_monitor.SPIKE_RATIO` |
| Min hourly count for spike | 5 | `error_monitor.SPIKE_MIN_HOURLY` |
| New-pattern window | 60 min | `error_monitor.NEW_PATTERN_WINDOW_MIN` |
| New-pattern min count | 5 | `error_monitor.NEW_PATTERN_MIN_COUNT` |
| Auto-resolve threshold | 50 % × spike floor | `error_monitor.AUTO_RESOLVE_RATIO` |
| Auto-resolve duration | 2 consecutive scans | `error_monitor.AUTO_RESOLVE_HOURS` |
| Warm-up back-fill | last 5 MB of errors.jsonl | `error_monitor._WARMUP_TAIL_BYTES` |
| In-memory event cap | 50 000 total / 2 000 per signature | `error_monitor.MAX_EVENTS_IN_MEMORY` |

All thresholds are module-level constants — no env vars yet. Adjust by
editing the module and rebuilding the gateway image.

---

## 9. Risk to protected systems

The monitor is **strictly additive** infrastructure:

- **Reads** errors.jsonl (read-only).
- **Writes** to its own state file + the new `control_plane.error_anomalies` table.
- **Adds** one cron job (5-min interval, ~5 ms CPU per run on a typical scan).
- **Touches none** of: SubIA, beliefs, MAP-Elites, affective layer,
  KBs (episteme/experiential/aesthetics/tensions/philosophy),
  self-improvement crews, LLM cascade, lifecycle hooks.

The total-rate σ-detection feeds metrics into the existing
`app/anomaly_detector.py` deque so total-rate anomalies surface via the
pre-existing `/api/cp/anomalies` endpoint — no parallel pipeline.

---

## 10. Future enhancements (deliberately out of v1 scope)

- **Permanent ignore-list** for known-noise signatures the monitor
  should never flag (workspace JSON or new column).
- **Signal forwarding** of `severity=critical` anomalies via the
  existing `app/signal/forwarder.py`.
- **Per-pattern severity overrides** (some patterns matter at 3×, others
  not until 10×).
- **Trend-direction context** on the dashboard ("rising over last 6h vs
  first 18h") instead of just summary trend.
- **ML-based pattern grouping** to catch near-identical messages our
  regex normalizer misses.

---

## 11. Runbook dispatch

Anomaly rows are useful for humans on the dashboard, but for
operationally-known failure modes — pool exhaustion, outbox stalls,
schema drift — there is value in *automated* remediation. The
Runbook Dispatcher (`app/healing/runbooks.py`, opt-in via
`ERROR_RUNBOOKS_ENABLED=true`) is the hook for that.

It is wired into `_record_anomaly`, *after* the INSERT into
`control_plane.error_anomalies`. A runbook failure can never break
anomaly recording — the dispatcher call is wrapped in a bare
`try/except` that logs at debug.

### Architecture

```
errors.jsonl ──► error_monitor scan ──► _record_anomaly ──► INSERT
                                              │
                                              └──► maybe_run_runbook(anomaly)
                                                              │
                                                              ▼
                                                  ┌────── 7 gates ───────┐
                                                  │ env flag             │
                                                  │ severity ≠ info      │
                                                  │ pattern match        │
                                                  │ runbook enabled      │
                                                  │ recurrence ≥ N       │
                                                  │ success rate ≥ 50%   │
                                                  │ concurrency cap ≤ 1  │
                                                  └──────────┬───────────┘
                                                             ▼
                                                  daemon thread → handler
                                                             │
                                                             ▼
                                                  _record_runbook_outcome
                                                  + dispatch.{started,
                                                    finished, skipped} audit
```

### The seven gates (and their skip reasons)

Each skipped dispatch emits a `dispatch.skipped` audit row with a
`reason` field, so traceability is end-to-end:

| # | Gate | Skip reason |
|---|---|---|
| 1 | `ERROR_RUNBOOKS_ENABLED` env var | (returns None silently) |
| 2 | severity ≠ `info` | `severity_info` |
| 3 | Anomaly's `pattern_signature` matches a registered runbook's regex | `no_pattern_match` |
| 4 | Per-runbook `enabled=true` in `runbook_settings.json` | `runbook_disabled` |
| 5 | Signature's 24h recurrence ≥ runbook's `min_recurrence` | `below_recurrence_threshold` |
| 6 | Recent success rate (last 10 outcomes) ≥ 50% | `recent_success_rate_low` |
| 7 | At most 1 runbook in flight (concurrency cap) | `concurrency_cap` |

First-registered-wins on pattern collision; insertion order is the
operator-controlled tie-breaker. Empty success-rate history is
treated as passing (1.0) so a freshly-registered runbook gets its
first chance.

### Audit actions

Actor: `self_heal_runbook`. Three actions:

* `dispatch.started` — gates passed, daemon thread launched.
* `dispatch.finished` — handler returned (or raised). Includes
  `success`, `duration_ms`, `error?`, `detail?`.
* `dispatch.skipped` — one of the seven gates rejected. Includes
  `reason` plus context (signature, severity, runbook_name,
  recurrence vs threshold, etc.).

Query: `/api/cp/audit?actor=self_heal_runbook`.

### Reference handler — `log_only`

Auto-registered at import time with the catch-all `.*` pattern and
`enabled: true` in shipped defaults. It logs the trigger and writes
an outcome row but takes no system action. Purpose: verify the
dispatch wiring end-to-end before any real runbook is wired.
Operators replace or narrow it once production runbooks
(`restart_pool`, `force_reconcile_outbox`, …) are registered from
boot code.

### Adding a real runbook

```python
# app/main.py (or any boot-time module)
from app.healing.runbooks import register_runbook, RunbookResult

def restart_pool(anomaly: dict) -> RunbookResult:
    try:
        # ... operator-authored remediation ...
        return RunbookResult(name="restart_pool", success=True,
                             detail="pool reset, healthcheck green")
    except Exception as exc:
        return RunbookResult(name="restart_pool", success=False,
                             error=str(exc))

register_runbook(
    name="restart_pool",
    pattern=r"db_pool::.*pool exhaust",
    handler=restart_pool,
)
```

Then add an entry to `workspace/self_heal/runbook_settings.json`:

```json
{
  "runbooks": {
    "restart_pool": {"enabled": true, "min_recurrence": 3}
  }
}
```

### State files

Both live under `workspace/self_heal/`:

* **`runbook_settings.json`** — operator-authored. Per-runbook
  `enabled` flag and `min_recurrence`. Missing entries default to
  disabled (safe-by-default).
* **`runbook_stats.json`** — runtime-managed. Last 10 outcomes per
  runbook, used by the success-rate gate. Auto-pruned at write.

### Constraints

* **No LLM calls.** The dispatcher and handlers are pure Python by
  contract. If a remediation needs reasoning, propose a code change
  via `app/proposals.py` (the same path `diagnose_and_fix` uses for
  `fix_type=code`).
* **TIER_IMMUTABLE respected.** Runbook handlers must NOT modify any
  file in `app/auto_deployer.TIER_IMMUTABLE`. The dispatcher itself
  is in TIER_IMMUTABLE.
* **Bounded concurrency.** Cap of 1 in flight prevents a runbook
  storm from amplifying an outage.
* **Self-tuning gate.** A runbook that fails > 50% of the time over
  its last 10 dispatches gets benched until its history rolls forward
  enough to recover. This keeps a broken handler from hammering a
  signature.

### Code pointers

* `app/healing/runbooks.py` — dispatcher module.
* `app/observability/error_monitor.py:_record_anomaly` — the call
  site (post-INSERT, try/except wrapped).
* See PROGRAM.md §14 for the change-log entry; the dispatcher
  exists as the *Track B* half of the May 2026 self-healing pass.
* See `docs/RECOVERY_LOOP.md` §17 for composition with the Tool
  Supervisor (Track A).
