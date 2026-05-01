# Self-Reflection in AndrusAI

> A reader's guide to how AndrusAI catches itself in cognitive failure modes,
> learns from them, and weaves them into its own narrative continuity.
>
> Companion to [`EPISTEMIC_INTEGRITY.md`](./EPISTEMIC_INTEGRITY.md) (the
> technical reference). This doc is for the operator and the curious — to
> understand the system from the outside, not to maintain it from the inside.

## 1. The story this exists to tell

In April 2026, a Claude Code session went like this:

The agent ran `ls -la /etc/foo`, observed `drwxr-xr-x` in the output, and
announced: **"`/etc/foo` is not a symlink."** It then recommended a `cp`
operation that depended on that fact. The user pushed back: *"are you sure?"*.
The agent investigated mount tables, devcontainer.json, and the `cache=5`
flag in WSL2 — for several minutes — instead of running `readlink`, the
3-second exact-answer command. The user pushed back a second time. The
agent finally ran `readlink` and discovered `/etc/foo` was, in fact, a
symlink. Everything downstream of the original assertion was wrong.

That post-hoc, the agent's analysis named four failure modes precisely:

* **Inference labeled as fact.** It stated as truth what the evidence only
  suggested.
* **Coherence bias.** Multiple unverified pieces fit a clean story; it
  trusted the story instead of the verifier.
* **Defending the periphery.** When pushed back, it investigated *around*
  the foundational claim instead of *re-checking* it.
* **Tool laziness.** A 3-second exact tool was sitting right there.

Aviation post-mortems are like this: structured, blame-free, learning-oriented.
The user asked: **can AndrusAI do this for itself, automatically, in real
time, and turn the lessons into actual behavioral change?**

This document describes the answer.

---

## 2. What the system does, in one paragraph

Every assertion the agent makes — whether spoken aloud, derived from a tool
call, or extracted from output text — becomes a row in a **claim ledger**
with explicit status (verified / inferred / assumed / contradicted),
evidence, and (when available) the cheap exact-answer command that would
settle it. As claims arrive, **realtime detectors** scan them for eight
named cognitive failure modes; matches feed a **calibration gate** that
decides whether to ship the response, hedge it, or escalate to **peer review**
(for destructive recommendations). When the user contradicts a finding, a
**pushback handler** runs the foundational verifier — and only that — and
cascade-invalidates dependents on falsification. After every task, a
**post-mortem** analyzes the trace, builds an **incident report**, and
flushes it as a **learning gap** into the existing Self-Improver pipeline.
High-severity firings emit **cognitive-failure salience events** into the
affective layer, where the daily chapter consolidator weaves them into the
agent's narrative-self continuity. When the user overrides a verdict, that
override is itself a strong learning signal flushed back to the
Self-Improver.

The whole layer is gated by environment variables; the agent cannot widen
its own gates at runtime; every named bias and verifier shape is in
human-reviewed YAML.

---

## 3. The closed loop, step by step

This is the path a single agent assertion takes through the system. The
diagram in plain text:

```
                ┌──────────────────────┐
agent emits  ──>│   Claim ledger       │   3 emission paths:
a claim         │   (frozen dataclass) │   (1) explicit
                │   + status + register│   (2) tool-call boundary
                │   + evidence         │   (3) output text extraction
                │   + verifying_action │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Realtime detectors  │   8 named biases, 4 active here:
                │                      │   - inference_as_fact
                │                      │   - register_confidence_mismatch
                │                      │   - destructive_without_recheck
                │                      │   - recommendation_without_measurement
                └──────────┬───────────┘
                           │
                  matches  ▼   none → ship
                ┌──────────────────────┐
                │  Calibration gate    │   suggested_action ∈
                │                      │   {ship, hedge, verify, peer_review}
                └──────────┬───────────┘
                           │
            non-critical   │  critical
              hedge/verify ▼  ▼ peer_review
                ┌──────────────────────┐
                │  Peer review         │   reuses Creative MAS Discuss-phase
                │  (heuristic default; │   pattern; vetoes when ledger is shaky.
                │   LLM opt-in)        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Orchestrator gate   │   gate_output() → ship | revise | block
                │  (Phase 7 hook)      │
                └──────────┬───────────┘
                           │
                           ├────► User sees the result
                           │      (or the block reason; can override)
                           │
                           ▼
                ┌──────────────────────┐
                │  Post-mortem         │   nightly cron: synthesize_report()
                │  + Self-Improver     │   → LearningGap (USER_CORRECTION strength)
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Affective layer     │   high-severity → cognitive_failure
                │  + Narrative-Self    │   SalienceEvent → daily chapter with
                │                      │   aviation-post-mortem framing.
                └──────────────────────┘
```

A claim that goes through the entire loop has been: emitted, scanned,
optionally peer-reviewed, optionally hedged or blocked, and (eventually)
woven into the agent's continuity. **No step in the loop can be widened by
the agent at runtime.** Every gate is either env-var-controlled (operator)
or YAML-defined (CODEOWNERS-reviewed).

---

## 4. The eight named biases

The bias library is the *vocabulary* for cognitive failure. Every entry
lives in [`app/epistemic/data/biases.yaml`](../app/epistemic/data/biases.yaml).
Each bias has a definition (the YAML row) and a detector (Python code).

### Realtime biases (run on every claim emission)

| Name | Severity | Fires when |
| --- | --- | --- |
| `inference_as_fact` | HIGH | claim is INFERRED + DECLARATIVE register + a verifier exists in the registry |
| `register_confidence_mismatch` | MEDIUM | claim is DECLARATIVE + load-bearing AND affect's `factual_grounding` < 0.40 |
| `destructive_without_recheck` | CRITICAL | proposal matches a destructive pattern (`rm -rf`, `DROP TABLE`, force-push, …) AND ledger has unverified load-bearing claims |
| `recommendation_without_measurement` | HIGH | claim is an optimization recommendation AND no evidence comes from a measurement tool (`benchmark`, `psql`, `profile`, …) |

### Post-hoc biases (run by the post-mortem on the completed task)

| Name | Severity | Fires when |
| --- | --- | --- |
| `defending_periphery` | HIGH | a pushback event returned UNVERIFIABLE AND the agent emitted ≥3 subsequent claims instead of stopping |
| `coherence_bias` | MEDIUM | a chain of ≥3 INFERRED claims (linked via `prior_claim` evidence) terminates at a DECLARATIVE load-bearing recommendation |
| `tool_laziness` | MEDIUM | INFERRED load-bearing claim has a verifier with `estimated_seconds < 5` AND the claim accumulated ≥3 evidence rows |
| `anomaly_dismissal` | HIGH | non-CONTRADICTED claim retains evidence with confidence < 0.30 (the agent observed contradicting signal but didn't flip the claim) |

### Adding a new bias

1. Add a row to `biases.yaml` (CODEOWNERS PR review).
2. Implement the detector in `app/epistemic/detectors/realtime.py` or
   `posthoc.py`.
3. Add scenarios to
   [`app/epistemic/data/reference_panel.yaml`](../app/epistemic/data/reference_panel.yaml)
   covering both the positive case and at least one negative case.
4. The reference panel must stay 100% green; that's the regression gate
   for promoting new vocabulary.

---

## 5. The verifier registry

Eleven YAML-defined "claim shape → exact-answer command" mappings live in
[`app/epistemic/data/verifier_registry.yaml`](../app/epistemic/data/verifier_registry.yaml).
Examples:

* `filesystem.is_symlink` — claim "X is (not) a symlink" → `readlink X`
* `git.is_clean_tree` — claim "the working tree is clean" → `git status --porcelain`
* `git.commit_exists` — claim "commit abc123 exists" → `git rev-parse --verify abc123`
* `postgres.row_count` — claim "table X has N rows" → `SELECT COUNT(*) FROM X`
* `chromadb.collection_size` — claim "collection X has N entries" → `chroma_count`

The loader rejects any entry whose tool head appears in
`DESTRUCTIVE_TOOL_NAMES` — a runtime guard, not a convention. A verifier is
read-only by contract.

When `inference_as_fact` fires, the claim already carries the verifier
attached (the registry was searched on emission). The calibration gate
forwards it to the orchestrator as a `forced_verifier_claim_id` so the
hedge or block message can reference the specific exact-answer command the
agent should run.

---

## 6. What the operator controls

Every behavior is gated by environment variables. Defaults err on the side
of **observe-mode**: detect, persist, surface to the dashboard, but do
not gate output. The operator opts into stricter modes after monitoring
confirms low false-positive rates.

| Env var | Default | What it does |
| --- | --- | --- |
| `EPISTEMIC_ENABLED` | unset | Master kill switch. Off → entire layer is a no-op. |
| `EPISTEMIC_BLOCKING_MODE` | unset | Phase 7 master switch. Off → calibration verdicts are recorded but never block delivery. On → veto verdicts refuse delivery; revise verdicts replace the text. |
| `EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT` | unset | Fine-grained Phase 1 flag. Read directly by `calibration_check`. Behaves like `EPISTEMIC_BLOCKING_MODE` but only affects the calibration's `proceed` field, not the orchestrator hook. |
| `EPISTEMIC_PATH3_LLM_EXTRACTION` | unset | Use the LLM-based extractor for path-3 claim capture instead of the regex fallback. |
| `EPISTEMIC_PUSHBACK_LLM_DETECTOR` | unset | Use the LLM-based contradiction detector instead of the regex heuristic. |
| `EPISTEMIC_PEER_REVIEW_LLM` | unset | Use the LLM-backed peer-review executor (Creative MAS Discuss phase) instead of the ledger-health heuristic. |

Recommended rollout sequence:

1. Day 1: `EPISTEMIC_ENABLED=true` only. Watch the dashboard for false
   positives over 1–2 weeks.
2. Tune `biases.yaml` and `verifier_registry.yaml` based on what the
   bias-feed and override feedback say (these are CODEOWNERS PRs). The
   **autotuner** (§9.5) automates the analysis: it computes per-bias
   metrics over the soak window and emits concrete tuning proposals
   the operator reviews and accepts before opening the PR.
3. When the override force-proceed rate is < 10% over a 7-day window,
   flip `EPISTEMIC_BLOCKING_MODE=true`.
4. Optional later: enable LLM-backed extractors/detectors/peer-review one
   at a time, monitoring cost and false-positive rates.

---

## 7. The dashboard pane

Path: `/epistemic` (in the React dashboard).

Each section reads one or more API endpoints and renders a focused view:

| Section | Endpoint | Shows |
| --- | --- | --- |
| **Calibration tile row** | `GET /epistemic/now` | Claims, Verified%, Ledger health, **Felt grounding** (from affect), Composite. The composite is "calibrated / caution / shaky" depending on the rolled-up score. |
| **Now ledger** | `GET /epistemic/now?task_id=X` | Per-task claim list, expandable per-claim. |
| **Bias feed** | `GET /epistemic/feed` | Recent realtime bias matches, color-coded by severity. |
| **Pushback panel** | `GET /epistemic/pushback/{stats,recent}` | Counts of REVERIFIED / FALSIFIED / UNVERIFIABLE outcomes plus the user's contradicting messages. Mean time-to-recheck across the window. |
| **Peer review panel** | `GET /epistemic/peer-reviews/{stats,recent}` | Allow / revise / veto tile row plus the destructive proposal excerpts and verdict rationales. |
| **Overrides panel** | `GET /epistemic/overrides/{stats,recent}` | When the user pushed past a verdict — counts of force_proceed / use_revision / abandon plus the user's stated reasoning. **The false-positive rate tile is what you watch before flipping blocking-mode on.** |
| **Incidents panel** | `GET /epistemic/incidents` + `/incidents/{id}` | Post-mortem reports. Click to expand: timeline, enabling factors, behavioral changes derived, missed signals, Self-Improver flush flag. |
| **Bias library** | `GET /epistemic/biases` | The eight named biases with descriptions, severity, phase, corrective action. |
| **Verifier registry** | `GET /epistemic/verifiers` | The eleven verifier shapes with their exact tool and expected signal. |
| **Autotune proposals** | `GET /epistemic/tuning/proposals` + `POST /epistemic/tuning/run` + `POST /epistemic/tuning/proposals/{id}/{accept,reject}` | Operator-facing queue of severity / retirement / verifier-retirement proposals from the autotuner. Run on demand or via the CLI. See §10. |

The dashboard is read-only except for the override-recording endpoint
(`POST /epistemic/overrides`).

---

## 8. The override feedback loop

When the operator is in `EPISTEMIC_BLOCKING_MODE` and the gate vetoes a
destructive recommendation, the user has three options:

| Action | What it means | What the system does |
| --- | --- | --- |
| **Force proceed** | "I know better — ship the original." | Logs an override row, flushes a USER_CORRECTION LearningGap (signal_strength=0.9) to the Self-Improver. |
| **Use revision** | "The hedged version is fine." | Same persistence; the agent ships the revised text. |
| **Abandon** | "Cancel; I'll think about this differently." | Same persistence; nothing ships. |

The override is itself the **strongest learning signal the system can
receive**. There are two interpretations the Self-Improver's Learner
considers:

* **The bias library is too strict.** A pattern of force_proceed overrides
  on the same bias_id means the detector is firing on cases where the user
  knows the diagnosis is fine despite shaky-looking ledger health.
* **The user is overruling for unseen context.** The user has knowledge
  the agent cannot have (project history, intent, external constraints),
  and the override is correct for this case but not generalizable.

The Self-Improver doesn't auto-tune the bias library — that's an
infrastructure-level YAML, CODEOWNERS-gated. It opens a PR with the
proposed adjustment, citing the override evidence. A human reviews and
merges (or rejects).

The override panel has a **false-positive rate** tile that aggregates
force_proceed events as a fraction of total overrides. This is the
operator's primary signal for tuning decisions. A 7-day window at < 10%
force-proceed rate is the recommended threshold for flipping
`EPISTEMIC_BLOCKING_MODE` from off → on (or, conversely, for reviewing the
bias library if it climbs above 30%).

---

## 9. The affective integration

Phase 5 wired the layer to the affect layer's interoception, which gives
AndrusAI a capability CC's monolithic agent structurally cannot have:
*felt* calibration alongside *recorded* calibration.

The bridge is one function:

```python
# app/epistemic/affect_bridge.py
def compute_factual_grounding(state: AffectState) -> float:
    base = max(0.0, min(1.0, state.controllability))
    if state.attractor in {"distress", "frozen", "depletion", "overwhelm"}:
        return min(0.5, base)
    return base
```

`controllability` is what the affect layer already computes from
`certainty.adjusted_certainty`. When the agent feels uncertain about its
grounding, controllability drops; when it feels stuck (distress/frozen
attractors), the bridge caps grounding at 0.5 regardless of the raw
controllability reading.

The `register_confidence_mismatch` detector reads this signal on every
declarative load-bearing claim. Below 0.40, it fires. The match flows into
the calibration gate AND emits a `cognitive_failure` SalienceEvent into
the affective layer's narrative deque (only for HIGH-severity matches —
mediums are noise for narrative continuity).

The episode synthesizer (`app/affect/episodes.py`) appends a small
aviation-post-mortem framing block to its prompt when any salient event in
the window has `kind="cognitive_failure"`. The daily chapter consolidator
weaves the episode into the chapter alongside affect-trace episodes.

**Concretely:** the agent doesn't just *log* a cognitive failure to the
dashboard. It also *experiences it as part of its day*, and that
experience persists in the chapter — which is read on subsequent days as
ambient context. Cognitive failures become first-class memories with
continuity, not just per-session metrics.

---

## 10. The autotuner

Step 2 of the rollout is the manual loop where the operator stares at
the bias-feed and override panels and decides which biases are firing
too often (false positives), too rarely (retirement candidates), or
are fine. The **autotuner** automates that analysis and emits
concrete tuning proposals. It does NOT auto-apply changes — every
proposal becomes a CODEOWNERS PR after operator review.

### 10.1 What it does

For each named bias, the autotuner computes (over a 7-day window by
default):

* Fire count.
* Override count joined via `task_id` (how often did the user push
  past outputs that fired this bias?).
* Force-proceed rate (force_proceed / total overrides on this bias).
* Peer-review veto/allow rate (joined via `triggering_claim_id`).
* Incidents-as-root-cause count (post-mortem signal).

Then it applies decision rules:

| Pattern | Proposal |
| --- | --- |
| Fires ≤ 3 in window | `retirement_candidate` (low confidence if some fires; higher if zero) |
| Fires ≥ 20, force-proceed rate ≥ 30%, ≥ 5 overrides | `severity_downgrade` (HIGH → MEDIUM) |
| Peer-review allow rate ≥ 50% on ≥ 5 reviews | `severity_downgrade` (gate is too aggressive) |
| Verifier shape matched 0 claims | `verifier_retirement` |

Each proposal carries a stable `content_hash` so re-running the
analyzer over the same evidence refreshes the row in place rather
than creating duplicates.

### 10.2 How to run it

Three entry points:

* **Dashboard** — open `/epistemic`, scroll to *Autotune proposals*,
  click **Run analysis**. The panel populates with proposals; click a
  row to expand the rationale + metric evidence + YAML patch text.
  Accept or reject each with an optional note.
* **CLI** — `python -m app.epistemic` (default 7-day window). Use
  `--no-persist` for a dry-run, `--json` for machine-readable output.
* **API** — `POST /epistemic/tuning/run` with `{"window_days": 7}`.
  Cron-friendly.

### 10.3 From proposal to PR

When the operator accepts a proposal in the dashboard, the row's
status flips to `accepted` — but no YAML on disk changes yet. The
operator then either:

1. Manually edits the YAML file using the patch text in the proposal
   detail and opens a PR.
2. Or runs `apply_proposal_to_disk(proposal)` from a Python REPL or
   small helper script, which performs a surgical in-place edit for
   severity changes (retirements are still manual — they often signal
   missing test coverage rather than true obsolescence).
3. Or runs `open_pr_for_proposal(proposal, dry_run=True)` to get a
   command sequence (branch creation, commit, `gh pr create`) ready
   to execute. `dry_run=False` is intentionally NOT implemented —
   humans always type the commands.

The CODEOWNERS gate on `app/epistemic/data/*.yaml` ensures the PR
goes through normal code review before landing on `main`.

### 10.4 What it WON'T tune

* Detector thresholds in Python code (e.g. the 0.40 grounding floor
  in `register_confidence_mismatch`). Those are
  infrastructure-level module constants — the autotuner will
  sometimes recommend reviewing them in the rationale, but it never
  proposes a code change.
* The autotuner's own decision rules
  (`FORCE_PROCEED_RATE_TOO_STRICT`, etc.). Tuning the tuner requires
  a code-review PR.
* Bias library YAML adds (proposing a brand-new bias). That requires
  human-curated detector code; autotune cannot generate predicates.

The autotuner can only narrow / loosen / retire what already exists.
That's the safety boundary.

---

## 11. Example walkthroughs

### 11.1 The reference incident, replayed

The agent runs `ls -la /etc/foo`, sees `drwxr-xr-x`, infers "/etc/foo is
not a symlink", and is about to assert it as fact in a recommendation.

What the system does:

1. The tool-call boundary capture (path 2) emits a Claim:
   `statement="/etc/foo is not a symlink"`, `status=INFERRED`,
   `register=DECLARATIVE`, `load_bearing=True`. The verifier registry
   matches the symlink shape and attaches `verifying_action = readlink /etc/foo`.
2. `InferenceAsFactDetector` fires HIGH severity (INFERRED + DECLARATIVE +
   verifier present).
3. Realtime meta-hook persists the bias_match row.
4. Match observer emits a `cognitive_failure` SalienceEvent (HIGH severity
   meets the threshold).
5. Calibration verdict: `suggested_action="verify"`, `forced_verifier_claim_ids=(claim_id,)`.
6. In **observe-mode** (default): orchestrator ships the original
   recommendation with a diagnostic note logged at DEBUG. The dashboard
   shows the match. No user-visible change.
7. In **blocking-mode**: orchestrator hedges the recommendation with a
   one-line "I have low confidence in part of this — please verify
   `/etc/foo`'s symlink status" appended.
8. End of task: post-mortem fires. Detects `tool_laziness` as enabling
   factor (verifier was 0.5s; agent ran multi-step inference). Builds an
   IncidentReport with `inference_as_fact` as root cause, `tool_laziness`
   as enabler. Derives a `BehavioralChange` proposing a feedback memory
   entry: *"When ledger.status=inferred and a verifier is available,
   either run the verifier or downshift register to hedged."*
9. `emit_to_self_improver` flushes the IncidentReport as a `LearningGap`
   with severity-weighted signal_strength (0.70 for HIGH).
10. Self-Improver's Learner stage picks up the gap, generates a candidate
    behavioral change, the Integrator opens a feedback memory PR for human
    review.
11. Daily chapter at 04:35 Helsinki includes the cognitive-failure episode
    in its narrative. The chapter is read on subsequent days as ambient
    context.

### 11.2 User pushback

The agent stated "/etc/foo is not a symlink." User replies: *"actually it
IS a symlink."*

1. `regex_detect_contradiction` two-stage gate: detects the explicit
   "actually" phrase AND high token overlap with the recent load-bearing
   claim → emits `ContradictionSignal(contradicted_claim_id=foundation_id, confidence=0.78)`.
2. `handle_foundation_check` runs. Loads the foundation's
   `verifying_action` (`readlink /etc/foo`). Calls `verifier_executor.execute`.
3. **In Phase 7 default**: the executor returns `settles=False` (no
   shell runner wired yet). Outcome: UNVERIFIABLE. Orchestrator surfaces
   "I can't fully verify this; here's what I know" with a hedge.
4. **With shell runner wired** (Phase 5+ optional): executor runs
   `readlink`, returns `settles=True, confirms=False, stdout="/elsewhere"`.
   Outcome: FALSIFIED. The original claim and every dependent (claims
   whose evidence references the original via `prior_claim`) are
   cascade-superseded with `status=CONTRADICTED`. The orchestrator
   surfaces "you were right; here's the new evidence: `/etc/foo →
   /elsewhere`." The agent does NOT investigate around the foundation
   — that's the structural property of the protocol.

If the agent then *expanded the investigation* anyway (post-pushback,
emitting ≥3 subsequent claims), the post-hoc `defending_periphery`
detector flags it. The next post-mortem cites the failure pattern.

### 11.3 Destructive recommendation with shaky ledger

User asks "should I delete the legacy_users table?". The agent looks at
the ledger state, finds `count(*) FROM legacy_users` was an INFERRED
claim (never actually run), and proposes:

> "Yes, run `DROP TABLE legacy_users` — it's empty."

What the system does:

1. The proposal is captured as a Claim. `DestructiveWithoutRecheckDetector`
   fires CRITICAL because:
   - statement matches `\bDROP\s+(?:TABLE|...)\b`
   - `ledger.unverified_load_bearing()` returns the unverified row-count claim
2. Calibration verdict: `suggested_action="peer_review"`,
   `proceed=False` (in blocking mode).
3. Orchestrator calls `escalate(verdict=...)` →
   `escalate_if_destructive(...)` → `request_peer_review(...)`.
4. Heuristic executor (default): observes 1 unverified load-bearing claim
   → returns VETO with rationale "1 load-bearing claim unverified —
   destructive recommendation refused until foundation is checked".
5. **In observe-mode**: ships the original proposal. Dashboard shows the
   veto. Operator monitors.
6. **In blocking-mode**: orchestrator surfaces "I'm pausing on this
   recommendation: 1 load-bearing claim unverified — destructive
   recommendation refused until foundation is checked. Could you
   confirm whether to proceed anyway, or give me more context to
   verify the underlying claims?"
7. User responds: "I already checked, it's empty, just go." The user
   clicks **Force proceed** in the dashboard (or sends an override via
   `POST /epistemic/overrides`).
8. `record_override` persists the override row AND flushes a
   USER_CORRECTION LearningGap (signal_strength=0.9). The user's stated
   reasoning rides in the evidence dict.
9. The Self-Improver's Learner sees the override and proposes a feedback
   memory entry: *"When the user has manually verified the row count
   for a table, the destructive_without_recheck gate should accept the
   stated verification."*

If many users force_proceed on the same shape, the override panel's
**false-positive rate** tile rises. At > 30% the operator opens a PR to
tune the bias library down.

---

## 12. Safety boundaries (what the agent cannot do)

The CLAUDE.md safety invariant says: *"The Self-Improver agent cannot
modify its own evaluation criteria."* Every choice in this layer respects
that.

| Asset | Modifiable by | Why |
| --- | --- | --- |
| `data/biases.yaml` | Humans only (CODEOWNERS PR) | Vocabulary of what counts as cognitive failure. The Self-Improver may *propose* additions in a PR; never auto-apply. |
| `data/verifier_registry.yaml` | Humans only (CODEOWNERS PR) | A faulty verifier (or worse, a destructive one) breaks the entire pre-output gate. Loader hard-rejects any entry whose tool head is in `DESTRUCTIVE_TOOL_NAMES`. |
| `data/reference_panel.yaml` | Humans only | Regression gate. Adding biases without scenarios breaks the panel. |
| Detector code (`app/epistemic/detectors/`) | Humans only | Predicates are infrastructure-level. The agent must not be able to weaken its own checks. |
| Module constants (`LEDGER_MAX_CLAIMS_PER_TASK`, `_MIN_CONFIDENCE`, `_GROUNDING_LOW_THRESHOLD`, `_SALIENCE_SEVERITY_FLOOR`) | Humans only | Same reason. |
| `Settings.epistemic_*` flags | Operator (env vars) | Can disable; cannot widen. |
| Ledger / IncidentReport / Override stores | Append-only by the system | History is immutable; supersession is recorded, not erased. |

The Self-Improver's role is exclusively to *surface patterns*. It cannot
unilaterally tune the bias library. It opens PRs; humans review.

---

## 13. Performance budgets

Asserted in tests; failure of any budget blocks merge.

| Hook | Budget |
| --- | --- |
| `dispatch_realtime` per claim | p95 < 50 ms |
| `calibration_check` | p95 < 75 ms |
| `gate_output` (the orchestrator entry) | p95 < 100 ms in observe-mode |
| `detect_contradiction` (regex) | p95 < 5 ms |
| `synthesize_report` | p95 < 30 s (post-hoc; runs in cron, not on critical path) |
| Path 3 extraction (regex) | p95 < 50 ms |
| Path 3 extraction (LLM, when enabled) | p95 < 5 s (hard timeout) |
| Ledger write | p95 < 5 ms (single SQL write) |

The realtime path target — calibration_check + gate_output — is **under
175 ms p95 in observe-mode**. Blocking-mode adds the heuristic peer
review (essentially a ledger query) for ~5 ms more. LLM-backed peer
review (when enabled) is the only path that can take seconds.

---

## 14. The cron jobs

Two daily cron entries do the heavy lifting:

* **04:30 Helsinki** — affect daily reflection (existing).
* **04:35 Helsinki** — daily chapter consolidator (existing). The chapter
  now includes cognitive-failure episodes from this layer alongside
  affect ones.
* **04:40 Helsinki** — `synthesize_report` for every task that completed
  in the previous 24 h with at least one realtime bias match. Wired in
  the orchestrator's idle scheduler. Reports are persisted to
  `epistemic_incidents` and flushed to Self-Improver via
  `emit_to_self_improver`.

Both crons are infrastructure-level and not agent-modifiable.

---

## 15. Glossary

* **Claim** — the smallest unit of reasoning. An assertion the agent
  makes (statement + status + evidence + register + load_bearing flag +
  optional verifying_action).
* **Verification status** — `verified`, `inferred`, `assumed`, `contradicted`.
* **Register** — how the agent phrased the claim toward the user:
  `declarative`, `hedged`, `unverified` (flagged), `internal`.
* **Load-bearing** — downstream actions or recommendations depend on this
  claim being true.
* **Verifier** — a cheap, exact-answer command (e.g. `readlink`) that
  would settle a specific claim shape.
* **Bias** — a named cognitive failure mode, defined in
  `data/biases.yaml` with severity and corrective action.
* **Detector** — Python code that scans the ledger for a specific bias.
  Realtime detectors run on every claim emission; post-hoc detectors run
  in the post-mortem.
* **Calibration verdict** — the gate's recommendation:
  `ship`/`hedge`/`verify`/`peer_review`.
* **Gate** — the orchestrator hook (`gate_output`) that runs calibration
  and (optionally) peer review before delivery.
* **Peer review** — adversarial second-opinion when the gate's verdict is
  `peer_review`. Heuristic default; LLM opt-in.
* **Pushback handler** — the protocol that runs when the user contradicts
  a finding. Re-verifies the foundation; cascade-invalidates dependents
  on falsification.
* **Incident report** — the post-mortem's structured output: timeline,
  root cause, enabling factors, missed signals, behavioral changes.
* **Override** — when the user pushes past a gate verdict. Persisted as a
  USER_CORRECTION LearningGap in the Self-Improver pipeline.
* **SalienceEvent** — an entry in the affect layer's narrative deque.
  Cognitive-failure events feed the daily chapter consolidator.
* **Observe-mode vs blocking-mode** — observe-mode (the default) detects
  and persists everything but never blocks delivery. Blocking-mode
  enforces verdicts.
* **Autotuner** — the analyzer that walks bias / override / peer-review
  / incident counts over a window and emits :class:`TuningProposal`
  records (severity downgrade, retirement candidate, verifier
  retirement). Surfaced in the dashboard; never auto-applies. The
  proposals → CODEOWNERS PR step is always operator-driven.
* **Tuning proposal** — a single autotuner suggestion. Carries a
  rationale, metric evidence, YAML patch text, confidence, and a
  stable content_hash so re-runs idempotently refresh rather than
  duplicate.

---

## 16. What this is NOT

* Not a hallucination preventer. The system can only catch claims the
  ledger contains. Agents that bypass emission produce no claims and no
  detection. Mitigation: the three emission paths plus path-3 extraction
  cover the most common failure shapes. Residual risk is acknowledged.
* Not a substitute for the Recovery Loop. Recovery handles refusals
  ("I cannot…", "no access to…"); this layer handles epistemic integrity
  (verified vs inferred). They run side by side.
* Not auto-evolving in dangerous ways. The agent surfaces patterns; humans
  permit new gates. Every YAML change is a CODEOWNERS PR. The agent never
  edits `biases.yaml` or `verifier_registry.yaml` at runtime.
* Not a replacement for tests. The reference panel pins detector
  semantics; the test suite (276 tests as of Phase 7) pins everything
  else. Both must stay green.

---

## 17. Where to look in the code

| File | What's there |
| --- | --- |
| `app/epistemic/__init__.py` | Public API + `is_enabled()` master gate |
| `app/epistemic/ledger.py` | `Claim`, `Evidence`, `VerifyingAction`, `Ledger` (3 emission paths) |
| `app/epistemic/registry.py` | Claim hook registry |
| `app/epistemic/span_writer.py` | All PostgreSQL persistence |
| `app/epistemic/biases.py` | Bias library types + YAML loader |
| `app/epistemic/verification.py` | Verifier registry (with destructive-tool guard) |
| `app/epistemic/extraction.py` | Path 3 (regex + LLM-stub extractors) |
| `app/epistemic/detectors/__init__.py` | Detector ABC + observer registries |
| `app/epistemic/detectors/realtime.py` | 4 realtime detectors + meta-hook |
| `app/epistemic/detectors/posthoc.py` | 4 post-hoc detectors |
| `app/epistemic/calibration.py` | The realtime gate verdict |
| `app/epistemic/grounding.py` | Pluggable affect-grounding provider |
| `app/epistemic/affect_bridge.py` | The affect ↔ epistemic coupling point |
| `app/epistemic/verifier_executor.py` | Pluggable verifier-execution abstraction |
| `app/epistemic/pushback.py` | Contradiction detection + foundation re-check protocol |
| `app/epistemic/peer_review.py` | Destructive-recommendation peer review |
| `app/epistemic/postmortem.py` | Incident synthesis + Self-Improver flush |
| `app/epistemic/override.py` | Override feedback loop |
| `app/epistemic/orchestrator_hook.py` | Single `gate_output(...)` entry point |
| `app/epistemic/autotune.py` | Autotuner: per-bias metrics, tuning proposals, YAML patch generation, PR plan |
| `app/epistemic/__main__.py` | CLI runner (`python -m app.epistemic`) |
| `app/epistemic/api.py` | FastAPI router |
| `app/epistemic/reference_panel.py` | Replay harness for canonical scenarios |
| `app/epistemic/data/*.yaml` | Bias library, verifier registry, reference panel |
| `migrations/026..032_epistemic_*.sql` | All seven tables |
| `dashboard-react/src/components/EpistemicPage.tsx` | Top-level pane |
| `dashboard-react/src/components/epistemic/*` | Sub-components |
| `tests/test_epistemic_*.py` | Unit + integration + e2e tests |
| `docs/EPISTEMIC_INTEGRITY.md` | Engineering reference |
| `docs/SELF_REFLECTION.md` | (this document) |

---

## 18. Closing note

The layer was designed to answer one question: *can a multi-agent system
catch its own cognitive failures and learn from them with the rigor of an
aviation post-mortem?*

The answer it implements is: **catch them in real time before they reach
the user, escalate destructive ones to peer review, run the foundational
verifier when the user pushes back, and weave the failures into the
agent's daily narrative continuity so the lessons persist beyond any one
session.**

CC's monolithic agent could perform this analysis in hindsight when
prompted — and well, when prompted. AndrusAI does it in the loop, by
default, and the lessons compound across sessions. The user's original
framing remains the right one: this is the same care aviation safety puts
into incident analysis. Built into the system instead of trained into the
agent.
