# Resilience Drills

**Update 2026-05-18 (Q18 / PROGRAM §60):** The §44 drill framework
was rewritten as a state machine + operator-ratified baseline
system. The drill registry, drill names, cadence semantics, and
posture decision documented below are unchanged. What changed is
the orchestration:

* Every drill invocation now goes through
  `app/resilience_drills/runner.py:invoke_drill` which threads
  state + baseline comparison.
* Failed drills enter a backoff state machine (`WATCH` →
  `DEGRADED` → `QUARANTINED`) so the §44 hot-loop pattern is
  architecturally impossible.
* Drills emit structured `Observation` dicts; operators ratify
  them as baselines via `/cp/drills/<name>`.
* CR creation is deduplicated at the lifecycle layer.

See **`docs/RESILIENCE_DRILLS_V2.md`** for the full v2
architecture. The legacy §44 documentation below describes the
registry + cadence + posture decisions that remain canonical.

---

**Status (2026-05-16):** Q6 shipped four drills in
`app/resilience_drills/` at PROGRAM §44.1-§44.3 (the operationally
canonical registry). Q13 added three SIBLING drill+monitor pairs
outside that registry (PROGRAM §48 Q13.1, §44.5 P2#3, §48 Q13.3)
because each runs as an operator-invoked shell script rather than
through the Q6 registry.

The drills exist to verify recovery procedures actually work. They
operationalize the posture decision documented in
`docs/RESILIENCE_POSTURE.md`: identity is data, not uptime.

## The two drill patterns

**Pattern A (Q6 registry):** drills live in `app/resilience_drills/`
+ are auto-run by the scheduler. Each has a `DrillSpec` + an audit
JSONL + landmark emission to the continuity ledger. Four drills
listed below.

**Pattern B (sibling shell scripts):** operator-runnable
`deploy/scripts/*-drill.sh` + paired healing monitor watching the
manifest file. Each monitor alerts on never-run / stale / failed
states + emits its own continuity-ledger landmark on transitions.
Three drill+monitor pairs:

| Drill | Shell script | Monitor | PROGRAM ref |
|---|---|---|---|
| Restore | `deploy/scripts/restore-drill.sh` | `restore_drill` | Phase H #1 |
| Version upgrade | `deploy/scripts/version-upgrade-drill.sh` | `version_upgrade_drill` | §2.5 |
| **Schema migration** | `deploy/scripts/migration-drill.sh` | `migration_drill` | **§48 Q13.1** |

Pattern A is appropriate when the drill is safe to auto-run from
inside the gateway. Pattern B is appropriate when the drill spawns
scratch containers, applies SQL, or otherwise needs operator
scheduling. **Both patterns compose** with the same posture
declaration (`docs/RESILIENCE_POSTURE.md`) and the same continuity-
ledger landmark mechanism.

The TZ-drift monitor (`tz_drift`, PROGRAM §48 Q13.3, default ON) is
arguably a fourth Pattern-B drill but without a shell script — it's
a pure-Python comparison every day between the hand-rolled
`_helsinki_tz` function and `zoneinfo.ZoneInfo("Europe/Helsinki")`.
Files a CR proposing consolidation on first divergence. See
`docs/RISK_REGISTER.md` for the year-2+ sibling monitors.

## The four Q6 drills

### `backup_restore`

**Risk: LOW.** Wraps `app/dr/boot_drill.py`. Takes the most recent DR
tarball, imports it into an ephemeral temp directory, validates
ChromaDB collections + ledger round-trip integrity, smoke-retrieves
from at least one collection. Never touches the live workspace.

**Cadence:** 90 days. Auto-runs on schedule when LOW risk.

**Output:** `workspace/dr/drill_<timestamp>.json` (existing) +
`workspace/resilience/drill_audit.jsonl` (new Q6 audit) + Signal alert
on completion + continuity-ledger landmark on FAIL / first-pass /
recovery transitions.

### `embedding_migration`

**Risk: LOW.** Wraps `app/memory/embedding_migration/dry_run.py`.
Exercises the full 8-step state machine
(IDLE → PLANNED → DUAL_WRITE → BACKFILLING → SHADOW_READ → READY →
CUTOVER → APPLIED) against an isolated sandbox collection. Real HTTP
calls to embedding model; no production-KB writes.

**Cadence:** 90 days. Auto-runs on schedule.

**Default target spec:** `{ollama, mxbai-embed-large, dim=1024}`.
Operator can override per-run via the REST endpoint.

### `secret_rotation`

**Risk: LOW. DRY-RUN ONLY.** This drill NEVER rotates any production
secret. It verifies the rotation procedure:

- Candidate `gateway_secret` generation via `secrets.token_urlsafe(32)`
- Constant-time Bearer-token validator accepts candidate format
- Per-agent `BRIDGE_TOKEN_<AGENT>` slot enumeration matches the
  agent registry
- Vendor API key format patterns (Anthropic, OpenAI, OpenRouter)
  validate correctly

Format-check booleans only — NO secret values ever appear in the
audit log. The drill is a regression-detector: if someone adds a new
secret class that doesn't follow the rotation procedure, this drill
catches it.

Actual rotation is operator-driven (out of Q6 scope).

**Cadence:** 90 days. Auto-runs on schedule.

### `kill_the_gateway`

**Risk: HIGH. DISRUPTIVE.** The only drill that actually stops the
gateway container.

Default state: **OFF.** Operator must explicitly enable via
`/cp/settings` (toggle: `Resilience drills → kill_the_gateway`).
Toggling ON enables scheduler "due" notifications but does NOT
auto-execute. Execution is always operator-initiated.

**Cadence:** 90 days (when opted in).

**Procedure:**

1. Schedule a maintenance window
2. Verify Settings → Resilience drills → kill_the_gateway is ON
3. Operator runs OUTSIDE the gateway:
   ```
   scripts/drills/kill_the_gateway.sh "EXECUTE KILL DRILL"
   ```
4. The script:
   - Hits `POST /api/cp/drills/run/kill_the_gateway` for pre-drill
     readiness check (DR backup recent, no active Tier-3 monitoring,
     persistent stores healthy)
   - Records T0
   - `docker compose stop gateway`
   - Waits 30 seconds (simulating operator-react time)
   - `docker compose start gateway`
   - Polls `/health` until 200 OK (timeout 5 min)
   - Records T1 + computes recovery time
   - Smoke-checks `/api/cp/sentience/scorecard-pinning` returns
     `anti_goodhart_intact: true`
   - Writes `workspace/resilience/kill_drill_<ts>.json`
5. When the restored gateway comes back up, its companion loop
   detects the report on first idle pass and:
   - Ingests it into the audit log
   - Emits the continuity-ledger landmark (PASS, FAIL, or ERROR)
   - Signal-notifies the operator

**Recovery time target: 30 minutes.** Three consecutive drills
exceeding this triggers a posture re-review (see
`docs/RESILIENCE_POSTURE.md` escape condition #2).

## Master switches

All exposed via `/cp/settings`:

| Setting | Default | Effect |
|---|---|---|
| `resilience_drills_enabled` | ON | Top-level master |
| `drill_backup_restore_enabled` | ON | LOW-risk; auto-runs |
| `drill_embedding_migration_enabled` | ON | LOW-risk; auto-runs |
| `drill_secret_rotation_enabled` | ON | LOW-risk; auto-runs |
| `drill_kill_the_gateway_enabled` | **OFF** | Opt-in for the disruptive drill |
| `drill_staleness_monitor_enabled` | ON | Healing monitor; daily alerts past-due |

## Scheduler behavior

`app/resilience_drills/scheduler.py` runs as an idle job in
`companion.loop`. For each registered drill:

- If past `cadence_days`, emit a "drill due" Signal notification
- LOW + MEDIUM risk drills auto-run (in dry-run mode)
- HIGH risk drills (`kill_the_gateway`) **NEVER auto-run** — operator
  runs the external script

Also: on each pass the scheduler calls
`kill_the_gateway.ingest_external_report()` to detect any recent
external-script report and convert it into an audit row.

## Staleness monitor

`app/healing/monitors/drill_staleness.py` was added in Q6.2 as the
27th healing monitor; Q6.5 added `backup_freshness` as the 28th.
Daily probe. Alerts when any registered drill is past
`cadence_days + grace_days` without a successful run.

The scheduler's "due" notifications and the staleness monitor's
"past-due" alerts are complementary:
- Scheduler fires at cadence (catches the operator early)
- Monitor fires after cadence + grace (catches drift)

## Operator surfaces

### REST endpoints

```
GET  /api/cp/drills/registry        — list drills + last-run + cadence
GET  /api/cp/drills/audit?limit=50  — recent drill outcomes
POST /api/cp/drills/run/{name}      — manual trigger (LOW/MEDIUM only)
GET  /api/cp/drills/posture         — current posture (#22 decision)
```

### React `/cp/settings` → Resilience drills

Master toggle + per-drill switches. The `kill_the_gateway` toggle has
explicit warning text about the disruptive nature and the external-
script requirement.

### Weekly briefing

The weekly composer surfaces:
- Count of drills passed this week
- Count of drills FAILED this week (named)
- Count of drills past-due (named)

Section disappears when nothing happened.

## Continuity ledger emission

New event kind `resilience_drill` (10th in `IDENTITY_EVENT_KINDS`).
Emitted ONLY on landmark events:

- Drill FAIL or ERROR — operator must know
- First-ever PASS for a drill — identity-shaping
- Recovery (previous run FAIL/ERROR, this one PASS) — also identity-shaping

Routine PASS-then-PASS does NOT emit (it stays in the audit log).

Annual reflection's `summarise_drift` Counter auto-surfaces the new
kind — year-end self-narrative includes "the year we ran N drills,
M passed."

## DR export inclusion

Drill audit is INCLUDED in DR export tarballs (operator decision,
2026-05-13). The system's resilience history is part of its identity;
restoring from backup preserves the operator's view of which drills
have run when.

The `workspace/resilience/` directory is in
`app/dr/export_kbs.py:_LEDGER_INCLUDES`. Existing secret-denylist
guards apply (no `.env`, no `token` files, etc.).

## Off-host backup policy

Per posture decision: dual-target (S3 + Google Drive). The
`backup_restore` drill verifies the LOCAL tarball restores correctly;
off-host integrity verification is a separate operator-managed flow
(out of Q6 scope, deferred for future).

## Anti-Goodhart guards

The drills are PROCEDURES; they don't have a metric the system can
optimize for. Specifically:

- Drill pass-rate is NOT auto-tuned by any module
- Drill scenarios are FIXED (no parameterized "easy mode")
- Failed drills emit ledger events the operator sees, not just metrics
- The `secret_rotation` drill has a `test_secret_rotation_never_leaks_secret_values`
  test that verifies audit rows are bounded and don't contain candidate tokens

## Repository pointers

| File | Purpose |
|---|---|
| `app/resilience_drills/__init__.py` | Package overview |
| `app/resilience_drills/protocol.py` | DrillSpec / DrillResult / Registry |
| `app/resilience_drills/audit.py` | JSONL audit + ledger landmark emission |
| `app/resilience_drills/posture.py` | #22 decision encoded as constants |
| `app/resilience_drills/scheduler.py` | Cadence + auto-run + notifications |
| `app/resilience_drills/drills/` | Per-drill implementations |
| `scripts/drills/kill_the_gateway.sh` | External script for the disruptive drill |
| `app/healing/monitors/drill_staleness.py` | Past-due alerter |
| `docs/RESILIENCE_POSTURE.md` | Architectural decision (#22) |

## Who watches the watchers?

**The drill subsystem has no probe of its own health.** Open question
flagged in the Q6.4 audit. Two practical mitigations close it in
practice without adding meta-monitor complexity:

1. **Continuity ledger emission is operator-visible.** If drills stop
   emitting landmark events for 6+ months — visible via
   `summarise_drift` in the annual reflection — something is wrong
   with the drill subsystem itself, not just an individual drill.

2. **`drill_staleness` monitor depends on the audit file being
   readable.** If the audit file becomes corrupt or is silently
   deleted, the monitor sees "no successful runs ever" → fires
   alerts (after the 7-day boot-grace window). This is the
   degraded-mode signal.

**Operator heuristic**: if you see no `resilience_drill` ledger events
for 120+ days AND no `drill_staleness` alerts have fired, something
in the drill subsystem is broken — investigate. Routine
verification of subsystem health is operator-driven; no
gateway-side meta-monitor exists by design (avoids infinite
regress).

## CLI entry point

For manual operator-trigger outside the REST surface (useful during
gateway debugging or recovery):

```
# List drills with last-run state
python -m app.resilience_drills list

# Run a specific drill
python -m app.resilience_drills run backup_restore
python -m app.resilience_drills run kill_the_gateway --dry-run

# Show the posture decision
python -m app.resilience_drills posture

# Show recent audit entries
python -m app.resilience_drills audit --limit 10
python -m app.resilience_drills audit --drill backup_restore
```

The CLI runs from the same Python process as the gateway, so
runtime-settings master switches apply identically. For
`kill_the_gateway`, the CLI runs ONLY the pre-drill check — the
LIVE drill remains gated to `scripts/drills/kill_the_gateway.sh`
outside the gateway.

## Audit-log corruption recovery (Q6.5 doc#4)

If `workspace/resilience/drill_audit.jsonl` becomes corrupt or is
silently deleted, the symptoms + recovery procedure are:

**Symptoms:**

- After the 7-day boot grace, `drill_staleness` alerts fire for ALL
  registered drills simultaneously (the monitor sees "no successful
  runs ever" because audit-log iteration returns empty)
- `/api/cp/drills/audit` returns `{"results": []}` even though drills
  have actually run
- React `/cp/settings` ResilienceDrillsCard shows "never run" for
  every drill

**Diagnosis:**

```bash
# 1. Verify file exists + is readable
ls -la workspace/resilience/drill_audit.jsonl

# 2. Walk each line; the first failure tells you the corruption shape
python3 -c "
import json
with open('workspace/resilience/drill_audit.jsonl') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue
        try: json.loads(line)
        except json.JSONDecodeError as e:
            print(f'line {i}: {e}; preview: {line[:80]}')
            break
    else:
        print('no JSON corruption found; check file age + permissions')
"
```

**Recovery:**

The audit log is local history; the **continuity ledger** at
`workspace/identity/continuity_ledger.jsonl` retains all
`resilience_drill` landmark events (failures, first-passes, recoveries)
regardless of audit-log state. So:

1. Move the corrupt file aside:
   `mv workspace/resilience/drill_audit.jsonl{,.corrupt-$(date +%s)}`
2. The drills will re-populate the audit log on next run (auto-runs
   continue on the scheduler's normal cadence)
3. Landmark history from before the corruption is still in the
   continuity ledger and surfaces in the annual reflection

**What is lost:** routine pass/skip rows from before the corruption.
**What survives:** every landmark event ever emitted (FAIL/ERROR,
first-pass, recovery), via the continuity ledger.

## How annual reflection consumes drill events (Q6.5 doc#5)

The `resilience_drill` event kind (10th in `IDENTITY_EVENT_KINDS`)
is emitted to `workspace/identity/continuity_ledger.jsonl` on:

- Drill FAIL or ERROR — operator needs to know
- First-ever PASS for a drill — identity-shaping moment
- Recovery: previous run was FAIL/ERROR, this one is PASS

The annual reflection (`app/identity/annual_reflection.py`) reads
the ledger via `summarise_drift(window_days=365)`. The
`by_kind: Counter` field is dynamic — it counts every event kind
that appears, so the `resilience_drill` count surfaces automatically
in the year-end self-reflection at `wiki/self/value_reflections/<year>.md`.

The narrative typically renders this as a one-line summary:

> "This year the system ran N resilience drills (X passed first-ever,
> Y recovered after prior failures, Z still failing)."

Drill-specific narrative (per-drill counts, trend per drill) is NOT
auto-rendered — the annual reflection treats drill events at the
event-kind level. For per-drill detail, query the audit log directly
via `python -m app.resilience_drills audit --drill <name>` or
`/api/cp/drills/audit`.

## Q6 closure criteria (Q6.5 doc#6)

After five ship cycles (Q6.1 foundation → Q6.5 polish) Q6 is
declared CLOSED. Trajectory of audit findings:

| Cycle | Findings | Severity |
|---|---|---|
| §44.1 (Q6.1) | — | initial foundation |
| §44.2 (Q6.2) | — | four drills + monitor |
| §44.3 (Q6.3) | — | operator surfaces |
| §44.4 (Q6.4) | 10 | 1 P0 + 1 P0 test + 5 P1 + 3 P2 |
| §44.5 (Q6.5) | 6 | 1 P1 + 3 P2 + 2 docs |

The trajectory has flattened from architectural to polish/docs.

**Re-open the audit cycle ONLY under one of these specific
conditions:**

1. **Live operator-visible failure**: a drill silently failed for
   weeks and the operator only discovered it via consequence (e.g.,
   real disaster + restore failed). Investigate why the alert
   pipeline didn't fire.

2. **Posture violation**: scorecard-pinning-style audit shows
   `is_ha_proposed_for_subsystem` returning a violation OR the
   posture decision constants in `app/resilience_drills/posture.py`
   have drifted. This is the architectural-firewall test failing.

3. **Recovery-time excess**: three consecutive `kill_the_gateway`
   drills exceed 30 minutes. Triggers posture re-review per the
   escape conditions in `docs/RESILIENCE_POSTURE.md`.

4. **New drill class needed**: a 5th drill becomes worth running
   quarterly (e.g., a future "cloud-credential-rotation drill"
   when off-host SDKs are wired). That's its own design cycle, not
   an audit of existing drills.

5. **Concrete operator concern**: not "let's audit Q6 again" but
   "drill X behavior surprised me yesterday, here's the specific
   detail." That's an investigation, not a cycle.

**Do NOT re-open** the audit cycle on a cadence or "let's verify Q6
is still ok" pattern. Routine verification is covered by:

- CI tests (anti-Goodhart pinning, recovery-sequence pinning)
- The live `/api/cp/drills/posture` endpoint
- The `drill_staleness` + `backup_freshness` healing monitors

Past Q6.5, identically-prompted audit cycles produce diminishing
returns at risk of inventing problems. The post-Q5.6 closure
discipline applies here too.

## Audit history

- 2026-05-13 — Q6.1 foundation: protocol + audit + posture + tests
- 2026-05-13 — Q6.2 drills + scheduler + staleness monitor
- 2026-05-13 — Q6.3 REST + React + briefing + DR export inclusion + this doc
- 2026-05-13 — Q6.4 post-ship audit: recovery-landmark order +
  test exercises production sequence + per-drill in-flight lock +
  `is_first_run` uses successful-history + `inspect.getsource` +
  SOUL.md guard regex + boot-grace for staleness monitor + CLI
  entry point + React state display + self-monitoring docs
- 2026-05-13 — Q6.5 second-cycle audit: failed-drill Signal alert +
  double-registration warning + backup_freshness monitor + audit-
  log corruption recovery doc + annual reflection consumption doc +
  Q6 closure criteria. **Q6 declared closed.**
