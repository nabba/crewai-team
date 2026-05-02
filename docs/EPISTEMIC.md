# Epistemic Integrity Layer — Comprehensive Guide

> One-stop reference for the AndrusAI subsystem that catches its own
> cognitive failure modes in real time, escalates destructive
> recommendations to peer review, runs deterministic foundation
> re-checks on user pushback, and weaves the failures into the agent's
> daily narrative-self continuity as aviation post-mortems.
>
> Last revised 2026-05-01. Status: **all phases shipped + autotuner;
> live in observe-mode**. 310/310 Python tests + 12/12 reference panel
> scenarios + TS build clean.

This document is the **front door**. It covers what the layer is, how
to operate it, how to extend it, and where to look for deeper material.
Two specialised companions exist:

* **[`EPISTEMIC_INTEGRITY.md`](./EPISTEMIC_INTEGRITY.md)** — engineering
  reference. Schemas, code excerpts, performance budgets, phase-by-phase
  shipping notes, the test count breakdown.
* **[`SELF_REFLECTION.md`](./SELF_REFLECTION.md)** — narrative
  companion. The precipitating story, the closed-loop walkthrough in
  prose, glossary, example scenarios, philosophical framing.

When in doubt, this doc tells you *what to do*; the other two tell you
*why* and *how it's built*.

Related docs in `crewai-team/docs/`:

* [`AFFECT_LAYER.md`](./AFFECT_LAYER.md) — the felt-state layer the
  epistemic system reads its grounding signal from (§9 of this doc).
* [`SELF_IMPROVEMENT.md`](./SELF_IMPROVEMENT.md) — the 6-stage
  Self-Improver pipeline that consumes incident reports as
  `LearningGap` records (§7.4).
* [`RECOVERY_LOOP.md`](./RECOVERY_LOOP.md) — the parallel subsystem
  for refusal recovery; sits next to the epistemic gate at the same
  point in the orchestrator (§7.1).
* [`MEMORY_ARCHITECTURE.md`](./MEMORY_ARCHITECTURE.md) — Mem0 / pgvector
  / Neo4j / ChromaDB; the Self-Improver writes feedback memory entries
  into this stack when an incident triggers one.
* [`ARCHITECTURE.md`](./ARCHITECTURE.md) — the wider request lifecycle
  (POST_PROCESS, vetting, recovery, epistemic gate).

---

## 1. The one-paragraph version

Every assertion the agent makes — from explicit claim emission, from
tool-call observation, or extracted from output text — becomes a row
in a **claim ledger** with verification status, evidence, and the
cheap exact-answer command that would settle it. Eight named cognitive
failure modes (the **bias library**) scan claims in real time and
post-mortem; matches feed a **calibration gate** that ships, hedges,
runs a verifier, or escalates to **peer review** for destructive
recommendations. When the user contradicts a finding, a **pushback
handler** runs the foundational verifier — and only that — and
cascade-invalidates dependents on falsification. Daily, a **post-mortem**
synthesizes incidents and flushes them as **learning gaps** to the
Self-Improver. High-severity firings emit **cognitive-failure salience
events** that weave into the agent's daily narrative chapter as
aviation-style post-mortems. User overrides feed back to the Self-Improver
as USER_CORRECTION signals. An **autotuner** analyses bias, override,
peer-review, and incident data over a window and proposes severity /
retirement adjustments — humans review every YAML change via PR.

---

## 2. Document map

```
EPISTEMIC.md                  ← you are here
   │
   ├── §3  Status              ┐
   ├── §4  Quick start         │  operator
   ├── §5  Architecture        │  reading
   ├── §6  Closed loop         │  path
   ├── §7  Operator runbook    │
   ├── §8  Configuration       ┘
   │
   ├── §9  API reference       ┐
   ├── §10 Database schema     │  reference
   ├── §11 Module map          │  for code
   ├── §14 Cron jobs           │  spelunkers
   ├── §15 Performance         ┘
   │
   ├── §12 Developer guide     ── how to extend
   │
   ├── §13 Safety boundaries   ── why the agent can't widen its own gates
   ├── §16 Glossary
   ├── §17 FAQ
   └── §18 Cross-references and further reading

EPISTEMIC_INTEGRITY.md   ← engineering deep-dive (schemas, code, phase notes)
SELF_REFLECTION.md       ← narrative companion (story, walkthroughs, glossary)
```

---

## 3. Status

| | |
| --- | --- |
| **Master switch** | `EPISTEMIC_ENABLED=true` (live) |
| **Mode** | Observe-mode — `EPISTEMIC_BLOCKING_MODE=<unset>`. Detectors fire and persist; the gate runs but never blocks delivery. |
| **Phases shipped** | 0 (foundation) → 7 (orchestrator gate) + autotuner |
| **Database tables** | 7 (`epistemic_claims`, `epistemic_bias_matches`, `epistemic_pushback_events`, `epistemic_incidents`, `epistemic_peer_reviews`, `epistemic_overrides`, `epistemic_tuning_proposals`) |
| **Migrations** | 026–032, 035 (PCH layer columns) |
| **Bias library** | 9 named biases (5 realtime + 4 post-hoc) |
| **Verifier registry** | 10 starter shapes |
| **Reference panel** | 12 canonical scenarios — must stay 100% green to promote new vocabulary |
| **Python tests** | 329 passing |
| **TypeScript build** | clean |
| **Dashboard** | `/cp/epistemic` — 8 sub-panels |
| **Cron jobs** | post-mortem at 04:40 Helsinki (after affect's daily reflection at 04:30) |

---

## 4. Quick start

The system is **already enabled** as of 2026-05-01 11:30 UTC. This
section is for re-enabling on a fresh deployment, or for someone
following along.

### 4.1 Pre-conditions

* PostgreSQL is running (the `mem0` schema lives there).
* The gateway image is built from a tree containing `app/epistemic/`.
* `docker-compose.yml`'s `gateway` service reads `.env`.

### 4.2 Enable in three steps

```bash
# 1. Apply migrations (idempotent; CREATE TABLE IF NOT EXISTS).
for f in migrations/02{6,7,8,9}_epistemic_*.sql \
         migrations/03{0,1,2}_epistemic_*.sql; do
  docker exec -i crewai-team-postgres-1 psql -U mem0 -d mem0 < "$f"
done

# 2. Append to .env.
cat >> .env <<'EOF'

# Epistemic Integrity Layer
EPISTEMIC_ENABLED=true
EOF

# 3. Restart the gateway so the env var is picked up.
docker compose up -d --force-recreate gateway
```

### 4.3 Verify

```bash
# Wait for health.
until curl -fsS http://localhost:8765/health >/dev/null 2>&1; do
  sleep 2
done

# Sanity-check: 8 biases load.
curl -fsS http://localhost:8765/epistemic/biases \
  | python3 -c "import sys,json; print(len(json.load(sys.stdin)['biases']))"
# → 8

# Sanity-check: empty ledger snapshot for a non-existent task.
curl -fsS "http://localhost:8765/epistemic/now" | python3 -m json.tool
# → {"task_id": null, "ledger": null, ..., "calibration": {...}}
```

The dashboard at `/cp/epistemic` is live the moment the gateway is
healthy. It populates as the agent runs and emits claims.

### 4.4 Disable

```bash
# Reverse the env line and restart. The layer becomes a no-op;
# data already in the tables is preserved.
sed -i '' 's/^EPISTEMIC_ENABLED=true/EPISTEMIC_ENABLED=false/' .env
docker compose up -d --force-recreate gateway
```

---

## 5. Architecture at a glance

### 5.1 Conceptual layers

```
┌─────────────────────────────────────────────────────────────┐
│ orchestrator_hook.py                                        │
│   gate_output() ─ post-vetting, pre-delivery (§7.1)         │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
┌──────▼────────┐    ┌─────────▼─────────┐
│ calibration.py│    │  pushback.py      │
│ gate verdict  │    │  contradiction +  │
│               │    │  foundation check │
└───────┬───────┘    └─────────┬─────────┘
        │                      │
┌───────▼──────────────────────▼────────────┐
│ detectors/{realtime,posthoc}.py           │
│   9 biases (5 realtime + 4 posthoc)       │
└──┬──────────────┬─────────────────┬───────┘
   │              │                 │
┌──▼──┐    ┌──────▼──────┐   ┌──────▼──────────┐
│ledger│    │ verifiers   │   │ peer_review.py  │
│      │    │ (registry)  │   │ destructive gate│
└──┬───┘    └─────────────┘   └─────────────────┘
   │
┌──▼─────────────────────────────────────────┐
│ span_writer.py — PostgreSQL persistence    │
│   7 tables in control_plane.epistemic_*    │
└──┬──┬──┬──┬──┬──┬──┬──────────────────────┘
   │  │  │  │  │  │  │
   ▼  ▼  ▼  ▼  ▼  ▼  ▼
  claims, bias_matches, pushback_events, incidents,
  peer_reviews, overrides, tuning_proposals

┌─────────────────────────────────────────────┐
│ Cross-system bridges (single coupling pts.) │
│  • affect_bridge.py  ←→  app.affect         │
│  • postmortem.py     ─→   app.self_improvement
│  • override.py       ─→   app.self_improvement
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Operator surface                            │
│  • api.py (FastAPI router)                  │
│  • autotune.py + __main__.py (CLI)          │
│  • dashboard-react /cp/epistemic            │
└─────────────────────────────────────────────┘
```

### 5.2 Entry points (where things happen)

| Stage | Code | Trigger |
| --- | --- | --- |
| Claim emission (path 1, 2, 3) | `ledger.emit*` | Agent reasoning, tool-call hooks, output extraction |
| Realtime detection | `detectors/realtime.py::_realtime_meta_hook` | Every `Ledger.emit()` via the claim-hook registry |
| Pushback detection | `pushback.process_user_message` | New user message arrives at orchestrator |
| Calibration gate | `orchestrator_hook.gate_output` | Post-vetting, pre-delivery (orchestrator.py:3151) |
| Peer review | `peer_review.request_peer_review` | Calibration suggests `peer_review` |
| Post-mortem | `postmortem.synthesize_report` | Daily cron at 04:40 Helsinki |
| Self-Improver flush | `postmortem.emit_to_self_improver` + `override._flush_to_self_improver` | After post-mortem; on every override |
| Salience emission | `affect_bridge._emit_cognitive_failure_salience` | After realtime detectors persist matches (HIGH+) |
| Autotune | `autotune.run_full_analysis` | On demand (CLI / dashboard / API / cron-able) |

### 5.3 The pluggable extension points

Three pluggable modules. All follow the same pattern: default no-op,
`set_*` setter for production wiring, `_reset_for_tests` for unit tests.

| Module | Default | Production wires |
| --- | --- | --- |
| `grounding.py` | returns `None` (no signal) | `affect_bridge.bootstrap()` plugs `live_factual_grounding` |
| `verifier_executor.py` | returns `settles=False` | (Phase 5+ optional) sandboxed shell runner |
| `peer_review.py` | `heuristic_executor` (vetoes when ledger shaky) | `creative_crew.discuss_round` via `set_executor` |

---

## 6. The closed loop, in three lines

> Claim emitted → realtime detectors scan → calibration gate decides
> → orchestrator delivers (or blocks/revises in blocking-mode) → user
> may override → post-mortem feeds Self-Improver → narrative chapter
> records the lesson.

The full prose walkthrough lives in
[`SELF_REFLECTION.md` §3](./SELF_REFLECTION.md#3-the-closed-loop-step-by-step).
Read that for the *story*; come back here for *operations*.

---

## 7. Operator runbook

### 7.1 Where the gate runs in the request lifecycle

Inside `app/agents/commander/orchestrator.py` at the post-vetting,
pre-delivery point (line 3151), the call is:

```python
from app.epistemic.orchestrator_hook import gate_output
_gate = gate_output(
    proposal_text=final_result,
    task_id=str(task_id) if task_id else "",
)
if _gate.action == "block":
    final_result = _gate.final_text
    logger.info("epistemic: BLOCKED delivery — %s", _gate.user_visible_reason)
elif _gate.action == "revise":
    final_result = _gate.final_text
    logger.info("epistemic: REVISED delivery — %s", _gate.user_visible_reason)
elif _gate.diagnostic_note:
    logger.debug("epistemic: ship — %s", _gate.diagnostic_note)
```

Sits next to the Recovery Loop (
[`RECOVERY_LOOP.md`](./RECOVERY_LOOP.md)) at the same pre-delivery
point. Recovery handles refusals; epistemic handles cognitive
integrity. They run in series, recovery first.

The block is wrapped in `try/except logger.debug` so any internal
failure preserves the original answer. The user-facing path is
**bulletproof** — gate failures cannot break delivery.

### 7.2 Monitoring (the dashboard panes)

Path: `/cp/epistemic`. Eight sub-panels, top to bottom:

| Pane | Endpoint | What to watch |
| --- | --- | --- |
| **Calibration tile row** | `GET /epistemic/now` | Composite score. Drops below 70% → calibration is shaky; investigate. |
| **Now ledger** | `GET /epistemic/now?task_id=X` | Per-task claim list. Unverified-load-bearing count > 0 in non-trivial tasks → expected during drafting; concerning post-completion. |
| **Bias feed** | `GET /epistemic/feed` | Recent realtime matches. Spike in any single bias → something systematically wrong. |
| **Pushback panel** | `GET /epistemic/pushback/{stats,recent}` | Mean time-to-recheck. Should be sub-second; minutes mean the executor is slow. |
| **Peer review** | `GET /epistemic/peer-reviews/{stats,recent}` | Veto rate. > 30% on `destructive_without_recheck` is healthy paranoia; close to 0% means the gate isn't firing on real destructive proposals. |
| **Overrides** | `GET /epistemic/overrides/{stats,recent}` | **False-positive rate** tile (force_proceed / total). Watch this most. < 10% over 7 days → ready to flip blocking-mode. > 30% → the bias library is too strict. |
| **Incidents** | `GET /epistemic/incidents` | Recent post-mortems. Each links to its full timeline + Self-Improver flush flag. |
| **Bias library** | `GET /epistemic/biases` | Reference. The 8 named failures with severity, phase, corrective action. |
| **Verifier registry** | `GET /epistemic/verifiers` | Reference. The 10 verifier shapes. |
| **Autotune proposals** | `GET /epistemic/tuning/proposals` | Operator queue. Run analysis weekly, accept or reject each, open PR for accepted. |

### 7.3 Rollout phases

```
┌─ Phase A: observe-mode (current) ────────────────────────┐
│  EPISTEMIC_ENABLED=true                                  │
│  EPISTEMIC_BLOCKING_MODE=<unset>                          │
│  Detectors fire, persist, surface to dashboard.          │
│  Gate computes verdict but never blocks delivery.        │
│  Recommended duration: 1-2 weeks.                        │
│  Watch: bias feed, override panel (no overrides yet).    │
└──────────────────────────────────────────────────────────┘
                 │
                 ▼ when patterns are visible, run autotune
┌─ Phase B: tune ──────────────────────────────────────────┐
│  python -m app.epistemic                                 │
│  → review proposals in dashboard                         │
│  → accept the well-supported ones                        │
│  → open CODEOWNERS PRs for accepted YAML changes         │
│  → re-deploy                                             │
└──────────────────────────────────────────────────────────┘
                 │
                 ▼ when override force-proceed rate < 10%
┌─ Phase C: blocking-mode ─────────────────────────────────┐
│  Add to .env: EPISTEMIC_BLOCKING_MODE=true               │
│  Restart gateway.                                         │
│  Veto verdicts now refuse delivery.                      │
│  User can force-proceed; each override is a              │
│  USER_CORRECTION learning signal.                        │
│  Watch: override panel, incidents panel.                 │
└──────────────────────────────────────────────────────────┘
                 │
                 ▼ optional, one at a time
┌─ Phase D: LLM-backed components ─────────────────────────┐
│  EPISTEMIC_PATH3_LLM_EXTRACTION=true                     │
│    — broader claim capture from output text              │
│  EPISTEMIC_PUSHBACK_LLM_DETECTOR=true                    │
│    — catches non-obvious user pushback                   │
│  EPISTEMIC_PEER_REVIEW_LLM=true                          │
│    — Creative-MAS Discuss-phase peer review              │
│  Each adds latency + LLM cost; enable individually       │
│  and watch the latency / cost dashboards.                │
└──────────────────────────────────────────────────────────┘
```

### 7.4 Troubleshooting

**The dashboard shows zero biases.**
The container ran a build that predates the epistemic layer. Rebuild:
```bash
docker compose build gateway && docker compose up -d --force-recreate gateway
```

**Endpoints return 404.**
Same cause as above, OR the router include in `app/main.py` failed
silently. Check:
```bash
docker logs crewai-team-gateway-1 2>&1 | grep -iE "epistemic|router" | head
```
Look for `epistemic verifier registry loaded: 10 shapes` and the
absence of `Epistemic API router registration failed`.

**`is_enabled()` returns False even with the env var set.**
The env var must be `true`, `1`, `yes`, or `on` (case-insensitive).
Anything else — including `enabled` — is treated as off.

**Tables don't exist.**
Migrations 026–032 weren't applied. Run the loop in §4.2 step 1.
All migrations are `CREATE TABLE IF NOT EXISTS` so re-running is safe.

**`/epistemic/now` returns `calibration.factual_grounding: null`.**
Expected unless the affect layer is running. Once `app.affect.core`
emits an `AffectState` (via cron or event hooks), the bridge starts
returning live values. Verify with:
```bash
curl -fsS http://localhost:8765/affect/now | python3 -c "import sys,json; print(json.load(sys.stdin)['affect']['controllability'])"
```

**Bias feed never populates.**
Either no claims are being emitted (path 1 / 2 / 3) — which means the
agent code that should call `Ledger.emit*` isn't wired yet to a hot
path — or detectors are skipping. Check:
```bash
docker exec crewai-team-postgres-1 psql -U mem0 -d mem0 \
  -c "SELECT COUNT(*) FROM control_plane.epistemic_claims;"
```
If zero, no claims. If positive but no `epistemic_bias_matches`,
detectors aren't matching (which can be normal for clean tasks).

**Post-mortem cron didn't run.**
The 04:40 Helsinki cron lives in the orchestrator's idle scheduler.
Verify it's registered:
```bash
docker exec crewai-team-gateway-1 python -c "
from app.idle_scheduler import list_jobs
for j in list_jobs():
    print(j)" | grep -i epistemic
```

**Performance: gate is slow.**
The realtime path target is < 175 ms p95 in observe-mode. If you see
much slower, check whether path-3 LLM extraction (`EPISTEMIC_PATH3_LLM_EXTRACTION`)
is on; that's the only path that adds seconds. Also check the database
connection pool (`CONTROL_PLANE_POOL_MAX`).

### 7.5 Disabling / rollback

Three levels of revert, increasing in scope:

```bash
# Level 1: turn off the gate, keep history.
sed -i '' 's/^EPISTEMIC_ENABLED=true/EPISTEMIC_ENABLED=false/' .env
docker compose up -d --force-recreate gateway

# Level 2: drop blocking-mode but keep observation.
sed -i '' '/^EPISTEMIC_BLOCKING_MODE=/d' .env
docker compose up -d --force-recreate gateway

# Level 3: remove all epistemic data (irreversible — only for dev resets).
docker exec crewai-team-postgres-1 psql -U mem0 -d mem0 <<'EOF'
DROP TABLE IF EXISTS control_plane.epistemic_tuning_proposals CASCADE;
DROP TABLE IF EXISTS control_plane.epistemic_overrides CASCADE;
DROP TABLE IF EXISTS control_plane.epistemic_peer_reviews CASCADE;
DROP TABLE IF EXISTS control_plane.epistemic_incidents CASCADE;
DROP TABLE IF EXISTS control_plane.epistemic_pushback_events CASCADE;
DROP TABLE IF EXISTS control_plane.epistemic_bias_matches CASCADE;
DROP TABLE IF EXISTS control_plane.epistemic_claims CASCADE;
EOF
```

Level 3 is for greenfield re-runs only. The data is small; retention
already CASCADEs from `crew_tasks` so old data ages out naturally.

---

## 8. Configuration reference

### 8.1 Environment variables

All read at process start; require gateway restart to take effect.

| Var | Default | Behavior when set to `true` |
| --- | --- | --- |
| `EPISTEMIC_ENABLED` | unset (off) | Master kill switch. The persistence layer becomes a no-op when off; detectors run but their results aren't stored. |
| `EPISTEMIC_BLOCKING_MODE` | unset (off) | The orchestrator gate enforces verdicts: `block` refuses delivery; `revise` rewrites the response. Off → all verdicts proceed but are recorded. |
| `EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT` | unset (off) | Fine-grained Phase 1 flag. Behaves like `EPISTEMIC_BLOCKING_MODE` but only inside `calibration_check`'s `proceed` field, not the orchestrator hook. Use the Phase 7 flag in production. |
| `EPISTEMIC_PATH3_LLM_EXTRACTION` | unset (off) | Use the LLM-based extractor for path-3 claim capture. Currently falls back to regex; real LLM wiring is a follow-up. |
| `EPISTEMIC_PUSHBACK_LLM_DETECTOR` | unset (off) | Use the LLM-based contradiction detector. Currently falls back to regex. |
| `EPISTEMIC_PEER_REVIEW_LLM` | unset (off) | Use the LLM-backed peer-review executor (Creative-MAS Discuss phase). Currently falls back to heuristic. |

Truthy values are case-insensitive: `1`, `true`, `yes`, `on`. Anything
else is off — including `enabled`, `T`, `Y`.

### 8.2 Module-level constants (infrastructure-level)

These are *not* environment variables. They are Python constants in
the modules where they apply. **Changing them requires a code-review
PR — the agent cannot widen its own gates by editing YAML or env
vars.** This is the safety boundary.

| File | Constant | Default | What it controls |
| --- | --- | --- | --- |
| `app/epistemic/__init__.py` | `LEDGER_MAX_CLAIMS_PER_TASK` | 500 | Per-task soft cap. Beyond this, `Ledger.emit` raises `LedgerFullError`. |
| `app/epistemic/__init__.py` | `POSTHOC_DETECTOR_BUDGET_S` | 30 | Hard timeout for any post-hoc detector during the post-mortem. |
| `app/epistemic/__init__.py` | `CALIBRATION_HOOK_BUDGET_MS` | 50 | Soft budget for the realtime calibration hook (target p95). |
| `app/epistemic/__init__.py` | `PUSHBACK_CLASSIFIER_MIN_CONFIDENCE` | 0.60 | Below this, the regex contradiction detector treats the message as a new question, not a contradiction. |
| `app/epistemic/__init__.py` | `DESTRUCTIVE_PEER_REVIEW_BUDGET_USD` | 0.05 | Hard cap on a single destructive-recommendation peer review. |
| `app/epistemic/detectors/realtime.py` | `_GROUNDING_LOW_THRESHOLD` | 0.40 | The `factual_grounding` floor below which `register_confidence_mismatch` fires. |
| `app/epistemic/detectors/posthoc.py` | `_CONTRADICTING_EVIDENCE_CONFIDENCE` | 0.30 | Below this, evidence is treated as contradicting the claim it attaches to (`anomaly_dismissal`). |
| `app/epistemic/detectors/posthoc.py` | `_DEFENDING_PERIPHERY_MIN_CLAIMS` | 3 | After a UNVERIFIABLE pushback, ≥ this many subsequent claims fires `defending_periphery`. |
| `app/epistemic/detectors/posthoc.py` | `_COHERENCE_MIN_CHAIN` | 3 | Minimum chain length for `coherence_bias`. |
| `app/epistemic/detectors/posthoc.py` | `_TOOL_LAZINESS_MAX_SECONDS` | 5.0 | A verifier under this is "cheap"; multi-step inference for a cheap claim fires `tool_laziness`. |
| `app/epistemic/affect_bridge.py` | `_LOW_GROUNDING_ATTRACTORS` | `{distress, frozen, depletion, overwhelm}` | Attractors that cap `factual_grounding` at 0.5 regardless of raw controllability. |
| `app/epistemic/affect_bridge.py` | `_SALIENCE_SEVERITY_FLOOR` | `"high"` | Match severity must be ≥ this to emit a `cognitive_failure` SalienceEvent. |
| `app/epistemic/extraction.py` | `CAP_PER_OUTPUT` | 8 | Max claims path-3 will extract from a single agent output. |
| `app/epistemic/autotune.py` | `FORCE_PROCEED_RATE_TOO_STRICT` | 0.30 | Override force-proceed rate above which the autotuner proposes downgrade. |
| `app/epistemic/autotune.py` | `MIN_FIRES_FOR_SEVERITY_PROPOSAL` | 20 | Minimum bias fires before severity proposals are worth considering. |
| `app/epistemic/autotune.py` | `RETIREMENT_FIRE_FLOOR` | 3 | Bias fires ≤ this in window → retirement candidate. |
| `app/epistemic/autotune.py` | `PEER_REVIEW_ALLOW_RATE_TOO_AGGRESSIVE` | 0.50 | Peer-review allow rate above which the gate is treated as too aggressive. |
| `app/epistemic/verification.py` | `DESTRUCTIVE_TOOL_NAMES` | `frozenset(...)` | Tools a verifier may NEVER use. The loader hard-rejects entries whose tool head is in this set. |

### 8.3 YAML-defined configuration

Three YAML files in `app/epistemic/data/`:

| File | What it defines | Edit policy |
| --- | --- | --- |
| `biases.yaml` | The 8 named biases (id, name, description, severity, phase, corrective_action, blocking) | CODEOWNERS PR; in-code starter as fallback |
| `verifier_registry.yaml` | 10 claim-shape → verifying-tool mappings | CODEOWNERS PR; loader rejects destructive tools |
| `reference_panel.yaml` | 12 canonical scenarios for detector regression | CODEOWNERS PR; must stay 100% green |

---

## 9. API reference

All endpoints under prefix `/epistemic` (no `/api/cp` prefix —
parallels `/affect`). The frontend Vite proxy already routes these.

### 9.1 Read endpoints

| Method | Path | Query params | Response |
| --- | --- | --- | --- |
| GET | `/epistemic/now` | `?task_id=<str>` (optional) | Ledger snapshot + calibration block. With task_id: full claim list; without: counts only. |
| GET | `/epistemic/feed` | `?window_min=N&limit=N` | Recent bias matches (default 60 min, 200 rows). |
| GET | `/epistemic/biases` | none | The 8 bias definitions. |
| GET | `/epistemic/verifiers` | none | The 10 verifier shapes. |
| GET | `/epistemic/claim/{claim_id}` | path param | Single-claim drill-in. 404 if not found. |
| GET | `/epistemic/pushback/stats` | `?window_min=N` | Counts by outcome (REVERIFIED/FALSIFIED/UNVERIFIABLE) + mean time-to-recheck. |
| GET | `/epistemic/pushback/recent` | `?window_min=N&limit=N` | Recent pushback events with user message + outcome + cascaded ids. |
| GET | `/epistemic/peer-reviews/stats` | `?window_min=N` | Counts by decision (allow/revise/veto) + mean duration. |
| GET | `/epistemic/peer-reviews/recent` | `?window_min=N&limit=N` | Recent peer-review events with proposal excerpt + verdict. |
| GET | `/epistemic/overrides/stats` | `?window_min=N` | Counts by user_action + force-proceed rate. |
| GET | `/epistemic/overrides/recent` | `?window_min=N&limit=N` | Recent override events with user reasoning. |
| GET | `/epistemic/incidents` | `?limit=N` | Recent post-mortem reports (top-level fields only). |
| GET | `/epistemic/incidents/{id}` | path param | Full IncidentReport: timeline, root cause, enabling factors, behavioral changes, missed signals. |
| GET | `/epistemic/tuning/proposals` | `?status=<str>&limit=N` | Tuning proposals (default `status=proposed`). |
| GET | `/epistemic/tuning/proposals/{id}` | path param | Single proposal drill-in. |

### 9.2 Write endpoints

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| POST | `/epistemic/overrides` | `{task_id, blocked_action, user_action, user_reasoning, peer_review_id?, flush_to_self_improver?}` | `OverrideEvent` JSON. Validates the enum fields; flushes to Self-Improver as USER_CORRECTION (signal_strength=0.9) by default. |
| POST | `/epistemic/tuning/proposals/{id}/accept` | `{operator_note?}` | Marks proposal accepted. Does NOT auto-apply YAML. |
| POST | `/epistemic/tuning/proposals/{id}/reject` | `{operator_note?}` | Marks proposal rejected. |
| POST | `/epistemic/tuning/run` | `{window_days?}` (default 7) | Triggers fresh autotune. Idempotent (UPSERT on content_hash). |

Window param convention:
- `window_min` ∈ [1, 1440] for everything-but-incidents (where short windows make sense)
- `window_min` ∈ [1, 10080] for pushback / peer-review / override panels (where 7-day views are useful)
- `window_days` for autotune (the analyzer's natural unit)
- `limit` ∈ [1, 500] universally

### 9.3 Response invariants

* All responses are JSON, UTF-8.
* All timestamps are ISO 8601 with UTC offset.
* `null` is used for "not available" (e.g. `factual_grounding=null`
  when affect not wired). Do not interpret `null` as zero.
* Empty arrays for "no matches in window" — never missing keys.
* All error responses use FastAPI's standard `{"detail": "..."}` shape.

---

## 10. Database schema reference

All tables live in the `control_plane` schema in the `mem0` database
(same PostgreSQL instance as Mem0 + the existing control_plane tables).

### 10.1 Tables

| Migration | Table | Rows when active | Retention |
| --- | --- | --- | --- |
| 026 | `epistemic_claims` | ~10–500 per task | CASCADE from `crew_tasks` (7 days default) |
| 027 | `epistemic_bias_matches` | ~0–10 per task | CASCADE from `crew_tasks` |
| 028 | `epistemic_pushback_events` | ~0–3 per task | CASCADE from `crew_tasks` |
| 029 | `epistemic_incidents` | ~0–1 per task | CASCADE from `crew_tasks` |
| 030 | `epistemic_peer_reviews` | rare (only critical biases) | CASCADE from `crew_tasks` |
| 031 | `epistemic_overrides` | rare (only when user pushes past gate) | CASCADE from `crew_tasks` |
| 032 | `epistemic_tuning_proposals` | ~5–30 per analysis run | UPSERTed on content_hash; never duplicated |

### 10.2 Key columns and indexes

For exact column types, indexes, CHECK constraints, and the
fields-as-columns vs JSONB tradeoffs, see the SQL files in
`crewai-team/migrations/026..032_epistemic_*.sql`. Each migration
opens with a comment block explaining design choices.

The recurring patterns:

* `task_id TEXT NOT NULL REFERENCES control_plane.crew_tasks(id) ON DELETE CASCADE`
  — every table CASCADEs to the parent task.
* `created_at` / `detected_at` / `requested_at` indexed DESC for
  recent-feed queries.
* Per-task indexed `(task_id, ts)` for drill-in.
* Stratified indexes on enums (status, decision, user_action) for
  count-by-X queries.
* Partial indexes where one branch dominates (e.g. unflushed incidents
  WHERE `self_improver_emitted = FALSE`).

### 10.3 Detail JSONB columns

Several tables store complex nested data as JSONB to avoid
schema-thrash:

| Table | Column | Shape |
| --- | --- | --- |
| `epistemic_claims` | `evidence`, `verifying_action`, `tags` | List of Evidence / VerifyingAction / strings |
| `epistemic_bias_matches` | `matched_claim_ids`, `detail` | List of claim_ids; detector-specific detail dict |
| `epistemic_pushback_events` | `invalidated_claim_ids` | List of cascade-invalidated claim_ids |
| `epistemic_incidents` | `report` | Full `IncidentReport.as_jsonable()` |
| `epistemic_peer_reviews` | `reviewers` | List of role names |
| `epistemic_tuning_proposals` | `metric_evidence` | Per-bias metrics dict |

Read patterns over JSONB columns are JSON-path aware (e.g.
`split_part(verifying_action->>'tool', ' ', 1)` for the verifier
match-counts join in autotune).

---

## 11. Module map

The `app/epistemic/` package — 18 modules + 3 YAML data files. Follow
this when reading the code.

### 11.1 Foundation (Phase 0)

| File | Purpose |
| --- | --- |
| `__init__.py` | Public API + `is_enabled()` master gate + bootstrap imports for detectors. |
| `ledger.py` | `Claim`, `Evidence`, `VerifyingAction`, `VerificationStatus`, `Register`, `Ledger` (3 emission paths). |
| `registry.py` | Claim-hook registry (decoupling Ledger from detectors). |
| `span_writer.py` | All PostgreSQL persistence. Bridge between in-process types and the 7 tables. |

### 11.2 Verification + biases (Phase 1, 2)

| File | Purpose |
| --- | --- |
| `verification.py` | Verifier registry (YAML loader with destructive-tool guard). |
| `biases.py` | Bias library types + YAML loader (with in-code fallback). |
| `extraction.py` | Path 3 (regex extractor + LLM-stub). |

### 11.3 Detectors (Phase 1, 2, 4)

| File | Purpose |
| --- | --- |
| `detectors/__init__.py` | Detector ABC + per-phase registries + match-observer pattern. |
| `detectors/realtime.py` | 4 realtime detectors + meta-hook with per-detector failure isolation. |
| `detectors/posthoc.py` | 4 post-hoc detectors. |

### 11.4 Gate + escalations (Phase 1, 3, 6, 7)

| File | Purpose |
| --- | --- |
| `calibration.py` | Realtime calibration_check + escalate (the calibration↔peer_review wiring). |
| `pushback.py` | ContradictionSignal + foundation re-check protocol (structurally narrow). |
| `verifier_executor.py` | Pluggable verifier-execution abstraction. |
| `peer_review.py` | Destructive-recommendation peer review (heuristic default; LLM opt-in). |
| `orchestrator_hook.py` | Single `gate_output(...)` entry point the commander calls. |

### 11.5 Cross-system bridges (Phase 4, 5, 7)

| File | Purpose |
| --- | --- |
| `affect_bridge.py` | The single coupling point between epistemic and affect. |
| `grounding.py` | Pluggable affect-grounding provider. |
| `postmortem.py` | IncidentReport synthesis + Self-Improver flush. |
| `override.py` | User override feedback loop (USER_CORRECTION flush). |

### 11.6 Operator surface

| File | Purpose |
| --- | --- |
| `api.py` | FastAPI router (all `/epistemic/*` endpoints). |
| `autotune.py` | Tuning analyzer, proposal generation, YAML patch + PR plan. |
| `__main__.py` | CLI runner: `python -m app.epistemic`. |
| `reference_panel.py` | Replay harness for canonical scenarios. |
| `data/biases.yaml` | The 8 named biases. |
| `data/verifier_registry.yaml` | The 10 verifier shapes. |
| `data/reference_panel.yaml` | The 12 canonical regression scenarios. |

---

## 12. Developer guide

### 12.1 Adding a new bias

A bias is a *named* failure mode. Adding one means:

1. **Define the vocabulary** — add an entry to
   `app/epistemic/data/biases.yaml`:
   ```yaml
   - id: my_new_bias
     name: "Human-readable name"
     description: |
       Multi-line description. What does this catch? What's the seed
       incident? When does it fire?
     severity: high                     # low | medium | high | critical
     detector: realtime                 # realtime | posthoc
     corrective_action: hedge_or_verify # hedge | verify | peer_review_required | …
     blocking: false                    # Phase 7 default — only critical biases block
   ```

2. **Implement the detector** — in `app/epistemic/detectors/realtime.py`
   (for realtime) or `posthoc.py`. Subclass `Detector`:
   ```python
   class MyNewBiasDetector(Detector):
       bias_id = "my_new_bias"
       def detect(self, ledger, *, claim=None):
           # realtime: claim is the just-emitted claim
           # posthoc: claim is None; walk the full ledger
           if claim is None: return  # if realtime, post-hoc no-op
           if <conditions>:
               yield BiasMatch(
                   bias_id=self.bias_id,
                   matched_claim_ids=(claim.claim_id,),
                   severity=BIAS_LIBRARY.get(self.bias_id).severity,
                   detail={...},
               )
   ```
   Register at module import:
   ```python
   MY_NEW_BIAS = register_realtime(MyNewBiasDetector())
   ```

3. **Add scenarios** to `app/epistemic/data/reference_panel.yaml` — at
   minimum one positive (must fire) and one negative (must NOT fire):
   ```yaml
   - id: my_new_bias_positive_canonical
     description: One sentence on the shape this catches.
     setup:
       claims: []
     trigger:
       statement: "..."
       status: inferred
       register: declarative
       load_bearing: true
     expected_matches:
       - bias_id: my_new_bias
   ```

4. **Write unit tests** in `tests/test_epistemic_*.py` — every bias
   needs:
   - One test for the firing case
   - One test for at least one no-fire case
   - Wire it into `tests/test_epistemic_e2e.py` if the bias has
     cross-component flow

5. **Run the test suite**:
   ```bash
   .venv/bin/python -m pytest tests/test_epistemic_*.py -v
   ```
   Reference panel must stay 100% green; that's the regression gate.

6. **PR review** — `data/biases.yaml` is CODEOWNERS-gated. The PR
   needs review before merge.

### 12.2 Adding a new verifier shape

A verifier is a cheap, exact-answer command for a claim shape.

1. Add entry to `app/epistemic/data/verifier_registry.yaml`:
   ```yaml
   - id: domain.my_verifier
     matches:
       claim_pattern: "regex with capture groups"
       tags_any: [optional, list]
     tool: my_command
     arg_extractor:
       kind: regex_capture       # or: none | template
       groups: { name: 1 }
     expected_signal: "human-readable describing what the output means"
     estimated_seconds: 0.5
   ```

2. The loader hard-rejects entries whose tool head is in
   `DESTRUCTIVE_TOOL_NAMES`. If your tool needs to be allowlisted,
   that's a code change to `verification.py` — and should be
   resisted (verifiers are read-only by contract).

3. Test with the existing pattern in
   `tests/test_epistemic_verification.py`. Your shape gets exercised
   via `match()` against synthetic claims.

### 12.3 Wiring a custom executor

Three pluggable executors. Each follows the same pattern:

```python
# Compute your signal / run your verifier / make your verdict.
def my_provider() -> SomeType:
    ...

# At app startup, wire it.
from app.epistemic.grounding import set_grounding_provider
set_grounding_provider(my_provider)
```

| Executor | Function to set | Default |
| --- | --- | --- |
| Affect grounding | `app.epistemic.grounding.set_grounding_provider` | `affect_bridge.live_factual_grounding` (set by `bootstrap()`) |
| Verifier execution | `app.epistemic.verifier_executor.set_executor` | No-op (returns `settles=False`) |
| Peer review | `app.epistemic.peer_review.set_executor` | `heuristic_executor` (vetoes when ledger shaky) |

Tests inject fakes via the same setters; `_reset_for_tests()` restores
defaults.

### 12.4 Running the full test suite

```bash
cd /Users/andrus/BotArmy/crewai-team
.venv/bin/python -m pytest tests/test_epistemic_*.py -v
# Expected: 310 passed
```

For the reference panel specifically:
```bash
.venv/bin/python -m pytest tests/test_epistemic_phase2.py::TestReferencePanel -v
# Expected: 12 scenarios pass
```

For TypeScript (run from `dashboard-react/`):
```bash
npx tsc --noEmit
# Expected: silent (no errors)
```

### 12.5 Code-style conventions

* Frozen dataclasses for immutable types (`Claim`, `Evidence`,
  `VerifyingAction`, `BiasMatch`, `IncidentReport`, etc.).
* `StrEnum` for enum-like states (`VerificationStatus`, `Register`,
  `Severity`, `DetectorPhase`, etc.).
* All public functions have type hints.
* Docstrings on public functions; comment blocks explaining *why* on
  non-obvious internals.
* Errors are *named* (`LedgerError`, `DuplicateClaimError`,
  `LedgerFullError`, `VerifierRegistryLoadError`,
  `BiasLibraryLoadError`, `ReferencePanelLoadError`,
  `ProposalApplyError`).
* Persistence is *fire-and-forget*: failures log at DEBUG and never
  propagate. The user-facing path must not break on telemetry failures.
* Imports inside functions only when avoiding cycles; otherwise at
  module top.
* No emojis in code. Comments are professional aviation-incident-report
  style.

---

## 13. Safety boundaries

The CLAUDE.md invariant: *"The Self-Improver agent cannot modify its
own evaluation criteria."* Every choice in this layer respects that.
For the full table, see
[`SELF_REFLECTION.md` §12](./SELF_REFLECTION.md#12-safety-boundaries-what-the-agent-cannot-do).

The condensed version:

* **YAML files** (biases, verifiers, reference panel) — CODEOWNERS PR
  only. The Self-Improver may *propose* additions in a PR; never
  auto-applies.
* **Detector predicate code** — code-review PR only.
* **Module-level constants** (severity thresholds, classifier
  confidence floors, severity-to-strength mapping in
  `emit_to_self_improver`) — code-review PR only. Tuning the tuner
  requires a PR.
* **Verifier loader hard guard** — `_VerifierRegistry.load_from`
  refuses entries whose tool head is in `DESTRUCTIVE_TOOL_NAMES`. This
  is a runtime check, not a convention.
* **Settings env vars** — operator can disable; cannot widen.
* **History** — append-only. Supersession is *recorded*, not erased.

---

## 14. Cron jobs

| Time (Helsinki) | Cron | Job | Owner |
| --- | --- | --- | --- |
| 04:30 | existing | Affect daily reflection | `app.affect` |
| 04:35 | existing | Daily chapter consolidator (now includes cognitive_failure episodes) | `app.affect.narrative` |
| 04:40 | new (Phase 4) | Post-mortem cron — `synthesize_report` for every task with realtime matches | `app.epistemic.postmortem` |
| weekly (recommended) | manual / cron-it | Autotune analysis | `python -m app.epistemic` |

The autotune cron is currently manual. To schedule it via the existing
idle scheduler, add a job that calls
`app.epistemic.autotune.run_full_analysis(window_days=7, persist=True)`
once per week. Or run via cron with the Docker exec form:
```bash
docker exec crewai-team-gateway-1 python -m app.epistemic
```

---

## 15. Performance budgets

Asserted in tests; failure of any budget blocks merge.

| Hook | Budget |
| --- | --- |
| Realtime detector dispatch per claim | p95 < 50 ms |
| `calibration_check` | p95 < 75 ms |
| `gate_output` (orchestrator entry) | p95 < 100 ms in observe-mode, < 175 ms in blocking-mode |
| `detect_contradiction` (regex) | p95 < 5 ms |
| `synthesize_report` | p95 < 30 s (post-hoc; runs in cron, not on critical path) |
| Path 3 extraction (regex) | p95 < 50 ms |
| Path 3 extraction (LLM, when enabled) | p95 < 5 s (hard timeout) |
| Ledger write (single SQL INSERT) | p95 < 5 ms |
| Autotune full analysis | p95 < 10 s for typical loads |

The realtime path is intentionally cheap. Anything LLM-backed is
opt-in and gated behind its own env var.

---

## 16. Glossary

For the full glossary with prose, see
[`SELF_REFLECTION.md` §15](./SELF_REFLECTION.md#15-glossary).

The compact version:

* **Claim** — smallest unit of reasoning (statement + status + evidence
  + register + load-bearing flag + optional verifier).
* **Verification status** — `verified` / `inferred` / `assumed` /
  `contradicted`.
* **Register** — how the agent phrased it: `declarative` / `hedged` /
  `unverified` (flagged) / `internal`.
* **Load-bearing** — downstream action depends on this claim being true.
* **Verifier** — cheap exact-answer command (`readlink`, `git status`,
  …) that would settle a claim shape.
* **Bias** — named cognitive failure (8 in the library).
* **Detector** — code that scans the ledger for one bias.
* **Calibration verdict** — `ship` / `hedge` / `verify` / `peer_review`.
* **Gate** — `gate_output()`. Runs after vetting, before delivery.
* **Pushback** — user contradicts a finding; protocol re-runs the
  foundational verifier ONLY.
* **Foundation re-check outcome** — `REVERIFIED` / `FALSIFIED` /
  `UNVERIFIABLE`.
* **Incident** — post-mortem report (timeline + root cause + enabling
  factors + behavioral changes + missed signals).
* **Override** — user pushes past a gate verdict; persisted and flushed
  as USER_CORRECTION.
* **Salience event** — entry in the affect layer's narrative deque;
  `kind="cognitive_failure"` for high-severity bias matches.
* **Tuning proposal** — autotuner-emitted suggestion for a YAML edit
  (severity / retirement / verifier-retirement).
* **Observe-mode vs blocking-mode** — observe (default) detects but
  doesn't gate; blocking enforces verdicts.

---

## 17. FAQ

**Q: Is this on right now?**
Yes, observe-mode. Verify with:
```bash
docker exec crewai-team-gateway-1 sh -c 'echo $EPISTEMIC_ENABLED'
# → true
```

**Q: Why don't I see any bias matches in the dashboard?**
Either the agent isn't emitting claims yet (path 1/2/3 wiring depends
on the agent code that runs in your particular task), or the agent's
output is well-calibrated and biases aren't matching. Both are normal
for short observation windows.

**Q: How do I disable just one bias temporarily?**
Edit `app/epistemic/data/biases.yaml`, remove the entry, restart the
gateway. Or if you want a softer touch: file an autotune proposal
with `kind=retirement_candidate` and accept it (see §10 of
SELF_REFLECTION.md).

**Q: The override force-proceed rate is 60%. Now what?**
The bias library is too strict. Run `python -m app.epistemic` —
the autotuner will likely propose `severity_downgrade` for the
offending bias. Accept and open a CODEOWNERS PR. Re-deploy. Watch
the rate.

**Q: Can the agent edit `biases.yaml` itself?**
No. The Self-Improver's Integrator stage opens a PR with the
proposed change — it never auto-merges. The CODEOWNERS gate on
`app/epistemic/data/*.yaml` enforces this.

**Q: What's the difference between `EPISTEMIC_BLOCKING_MODE` and
`EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT`?**
The first is the Phase 7 master switch read by the orchestrator
hook. The second is a fine-grained Phase 1 flag read by
`calibration_check` itself. In production, use the Phase 7 flag
(`EPISTEMIC_BLOCKING_MODE`). The Phase 1 flag is preserved for
backward compatibility and finer-grained testing.

**Q: How does this interact with the Recovery Loop?**
They sit next to each other at the same orchestrator point.
Recovery handles refusals (the agent saying "I cannot…");
epistemic handles cognitive integrity (the agent stating an
inference as fact). They run in series — recovery first. See
[`RECOVERY_LOOP.md`](./RECOVERY_LOOP.md).

**Q: Does the post-mortem cron cost money?**
Only if path-3 LLM extraction or LLM-backed peer review or LLM
contradiction detection is enabled (and even then only when biases
fire). The default heuristic versions are zero-cost. The
post-mortem itself does not call any LLM — it composes
deterministic IncidentReports from existing data.

**Q: How do I know which biases the autotuner will propose changes
for, before running it?**
Run with `--no-persist` for a dry-run:
```bash
python -m app.epistemic --no-persist --window-days 14
```
You'll see the proposals printed but nothing persisted to the
database.

**Q: Where does the `cognitive_failure` salience event end up?**
In the affect layer's narrative deque. The episode synthesizer
picks it up next time it runs (every 15 min of agent activity by
default), and the daily chapter consolidator at 04:35 Helsinki
weaves it into the chapter. See
[`AFFECT_LAYER.md`](./AFFECT_LAYER.md) for the affect side.

---

## 18. Cross-references and further reading

### 18.1 Sibling docs in `docs/`

* [`EPISTEMIC_INTEGRITY.md`](./EPISTEMIC_INTEGRITY.md) — engineering
  reference. Phase-by-phase shipping notes, full schemas, code
  excerpts, the test count breakdown.
* [`SELF_REFLECTION.md`](./SELF_REFLECTION.md) — narrative companion.
  The precipitating CC incident, prose closed-loop walkthrough,
  example scenarios, full glossary, philosophical framing.
* [`AFFECT_LAYER.md`](./AFFECT_LAYER.md) — felt-state layer.
  `factual_grounding` is read from the affect layer's
  `controllability` (which itself is `certainty.adjusted_certainty`).
* [`SELF_IMPROVEMENT.md`](./SELF_IMPROVEMENT.md) — the 6-stage
  pipeline that consumes incident reports as `LearningGap` records.
* [`RECOVERY_LOOP.md`](./RECOVERY_LOOP.md) — refusal recovery; the
  parallel subsystem at the same orchestrator point.
* [`MEMORY_ARCHITECTURE.md`](./MEMORY_ARCHITECTURE.md) — Mem0 +
  pgvector + Neo4j + ChromaDB.
* [`ARCHITECTURE.md`](./ARCHITECTURE.md) — wider request lifecycle.

### 18.2 Code locations

```
app/epistemic/                      ← the package
   ├── data/*.yaml                  ← bias library, verifier registry, reference panel
   └── tests are in tests/test_epistemic_*.py

app/agents/commander/orchestrator.py:3151
                                    ← the integration point (the gate call)

app/main.py:1782 (approx)
                                    ← router include + affect_bridge.bootstrap()

app/affect/episodes.py
                                    ← daily chapter prompt (extended for cognitive_failure)

migrations/026..032_epistemic_*.sql
                                    ← all 7 tables

dashboard-react/src/components/EpistemicPage.tsx
dashboard-react/src/components/epistemic/*
                                    ← React UI

dashboard-react/src/api/epistemic.ts
dashboard-react/src/types/epistemic.ts
                                    ← React-side types and hooks
```

### 18.3 Test files

| File | Coverage |
| --- | --- |
| `test_epistemic_ledger.py` | Claim / Evidence / VerifyingAction / Ledger emit / supersession |
| `test_epistemic_span_writer.py` | Persistence layer with mocked psycopg2 |
| `test_epistemic_verification.py` | Verifier registry loader + match |
| `test_epistemic_detectors.py` | Realtime + post-hoc detector base |
| `test_epistemic_api.py` | FastAPI router shapes |
| `test_epistemic_phase2.py` | Bias library YAML, new realtime detectors, path 3, reference panel |
| `test_epistemic_phase3.py` | Pushback handler, foundation check, persistence, API |
| `test_epistemic_phase4.py` | Post-hoc detectors, post-mortem, Self-Improver flush, persistence, API |
| `test_epistemic_phase5.py` | Affect bridge, salience emission, calibration block in API |
| `test_epistemic_phase6.py` | Peer review, escalation, persistence, API |
| `test_epistemic_phase7.py` | gate_output, override feedback, observe vs blocking, defensive paths |
| `test_epistemic_e2e.py` | 8 end-to-end stories: reference incident, pushback cascade, affective bridge, destructive blocking, recommendation without measurement, defending periphery, observe-never-blocks |
| `test_epistemic_autotune.py` | Analyzer, proposals, YAML patch, persistence, API |

### 18.4 Original framing

The system was built to answer one question: *can a multi-agent system
catch its own cognitive failures and learn from them with the rigor of
an aviation post-mortem?* The seed was a 2026-04 Claude Code session
that inferred "/etc/foo is not a symlink" from `ls -la`, asserted it
as fact, and only verified after the user pushed back twice. Read the
opening of [`SELF_REFLECTION.md`](./SELF_REFLECTION.md#1-the-story-this-exists-to-tell)
for the full framing.

---

## 19. Closing note

This layer sits next to (not inside) every other subsystem. It coupled
to affect via one function, to Self-Improver via two flush points, and
to the orchestrator via one gate. Everything else is internal.

That isolation is deliberate. The layer can be disabled in observe-mode
in 30 seconds (env var + restart) and disabled entirely in another 30
(plus optional table drops). The agent's reasoning continues unchanged
either way; what changes is how *much honesty* the layer requires of
the agent's output before delivery.

Read [`SELF_REFLECTION.md`](./SELF_REFLECTION.md) for the *why*. Read
[`EPISTEMIC_INTEGRITY.md`](./EPISTEMIC_INTEGRITY.md) for the *how*.
This document covers the *what to do*.
