# Meta-Agent Layer

A bounded meta-learning layer over crew dispatch. Inspired by
**Hyperagents** (arXiv:2603.19461, Zhang et al., Meta — March 2026)
but deliberately constrained to respect AndrusAI's safety invariant
that evaluation functions and safety constraints live at infrastructure
level and never become agent-modifiable.

> **Status**: shipped May 2026. Default OFF behind the `META_AGENT`
> feature flag (env or React Org Chart toggle). 32 tests in
> `tests/test_meta_agent.py` (all green); 91 tests across the
> base_crew-touching suites continue passing with no regression.

---

## 1. Why this layer exists

Hyperagents extends the **Darwin Gödel Machine** (DGM) by making the
meta-level modification procedure itself editable, so that gains in
"how to improve" compound on top of gains in "what to do." The
empirical claim that motivated this work: cross-domain improvements in
the meta-procedure — persistent memory + performance tracking that
**accumulate across runs** — transfer between domains.

The faithful Hyperagents formulation directly conflicts with two
hard invariants in CLAUDE.md:

> Evaluation functions and safety constraints live at INFRASTRUCTURE
> level. They must NEVER be in agent-modifiable code paths.

> The Self-Improver agent cannot modify its own evaluation criteria.

So we ship the **non-recursive layer only**. The meta-agent observes
recipe outcomes and chooses among historical recipes for a similar
task; the selector itself, the schema, and the bounded augmentation
set are infrastructure constants. When the meta-agent observes a
recipe pattern that an immutable rule blocks, it generates an
**amendment proposal** — a downloadable Markdown file the operator
reviews before any rule change.

---

## 2. The shape of the system

```
                     (META_AGENT enabled?)
                              │
                              ▼
                   run_single_agent_crew
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
   Selector              Apply (bounded)            Recorder
   (UCB1 +               force_tier·extra_tools     (lifecycle hook,
   similarity)           ·task_hint                  outcome ledger)
      │                  ·max_execution_time
      │                       │
      └───────────┬───────────┘
                  │
                  ▼
            AgentRecipe          ◄── Postgres registry +
                  │                    ChromaDB similarity
                  │                    index
                  ▼
            (used as augmentation
             on top of the agent
             factory's output —
             never replaces it)
```

The **agent factory** (`create_coder`, `create_researcher`,
`create_writer`) is unchanged and unreachable to the meta-agent.
The recipe layer wraps the dispatch in `run_single_agent_crew` and
adjusts only knobs that function already exposes:

| Knob | Why it's safe to tune |
|---|---|
| `force_tier` | Already a factory parameter (`local`/`budget`/`mid`/`premium`) |
| `extra_tools` | Already a `run_single_agent_crew` parameter; tool registry enforces tier and capability gates |
| `task_hint` | Non-destructive prefix on the task template; user input remains the source of truth |
| `max_execution_time` | Advisory ceiling; existing factories already apply per-agent caps |

Anything outside this set (backstory, goal text, the LLM model
selection rules, the agent class) lives in `TIER_GATED` (souls/) or
`TIER_IMMUTABLE` (llm_factory, modification_engine) and cannot be
touched from this layer.

---

## 3. Module layout

```
app/self_improvement/meta_agent/
├── __init__.py                # public re-exports
├── types.py                   # AgentRecipe, RecipeOutcome, RecipeSelection
├── feature_flag.py            # is_meta_agent_enabled (env → JSON → default)
├── meta_agent_settings.py     # JSON persistence for the dashboard toggle
├── store.py                   # Postgres registry + outcome ledger + ChromaDB index
├── selector.py                # UCB1 + similarity argmax (IMMUTABLE thresholds)
├── apply.py                   # bounded augmentation (force_tier × extra_tools × task_hint)
├── recorder.py                # outcome capture from the lifecycle envelope
├── policy_gap.py              # detect "blocked-by-immutable" recipes
└── amendment.py               # render TIER_IMMUTABLE amendment proposals
```

Re-exports are surfaced from `app.self_improvement` so callers see one
subsystem rather than a parallel namespace:

```python
from app.self_improvement import (
    AgentRecipe, RecipeOutcome, RecipeSelection,
    is_meta_agent_enabled,
    select_recipe, apply_recipe, record_recipe_outcome,
    upsert_recipe, list_recipes, list_outcomes,
    scan_for_policy_gaps, propose_immutable_amendment,
    SELECTION_THRESHOLDS,
)
```

---

## 4. Data model

### AgentRecipe

A bounded augmentation. Recipes are **evaluated**, not generated —
the meta-agent searches the historical recipe space (force_tier ×
extra_tool_set × task_hint) by similarity to the incoming task and
selects the highest-UCB option.

```python
@dataclass
class AgentRecipe:
    id: str                              # deterministic from knobs (re-discovery upserts)
    crew_name: str                       # "coding" | "research" | "writing" | ...
    force_tier: str | None
    extra_tool_names: list[str]
    task_hint: str
    max_execution_time: int | None
    task_signature: str                  # the kind of task this was tried on (embedded)
    proposed_by: str                     # "meta_agent" | "operator" | "seed"
    uses: int                            # denormalised counter
    successes: int                       # denormalised counter
    last_used_at: str
```

The **null recipe** for each crew (id `recipe_<crew>_<hash-of-empty-knobs>`)
is the always-available control arm; its `uses`/`successes` accumulate like
any other recipe so the bandit can fairly compare augmentation against
factory defaults.

### RecipeOutcome

Append-only ledger of every dispatch.

```python
@dataclass
class RecipeOutcome:
    id: str
    recipe_id: str
    crew_name: str
    task_id: str                         # joins to crew_lifecycle's firebase id
    success: bool
    confidence: str                      # "high" | "medium" | "low" | ""
    duration_s: float
    cost_estimate: float
    error_signature: str                 # exception type name on failure
    user_feedback: str                   # "👍" / "👎" / "" (if from React reactor)
    task_signature: str                  # snapshot of the task at apply time
```

Outcome inserts use `ON CONFLICT DO NOTHING` so a retry can't
double-count. The recipe row's `uses`/`successes` counters are bumped
in a separate statement; if the process crashes between the two, the
counters stay one behind reality, which is tolerable because the
selector smooths by `(uses + 2)` and the next dispatch compensates.
We deliberately don't introduce a new transactional path — that would
duplicate `app.control_plane.db`'s pool semantics.

### RecipeSelection

The selector's reasoned choice for a single dispatch — useful for
audit trails:

```python
@dataclass
class RecipeSelection:
    chosen: AgentRecipe
    candidates_considered: int
    score: float                         # similarity × UCB
    similarity: float                    # 1 - cosine_distance
    smoothed_success_rate: float         # (successes + 1) / (uses + 2)
    explored: bool                       # True iff ε-greedy explore branch
    rationale: str
```

---

## 5. Selection algorithm

### Algorithm

For each dispatch:

1. **Embed** `task_description` and **query** the ChromaDB recipes
   collection for top-`k` recipes registered to `crew_name`.
2. **Filter** to candidates with `cosine_distance ≤ similarity_tau`.
3. **Add the null recipe** as the always-available control arm
   (similarity = 1.0 by convention since "no augmentation" applies
   to every task).
4. **ε-greedy explore branch**: with probability `epsilon`, return a
   uniformly random candidate, tagged `explored=True` in the
   selection.
5. **Exploit branch**: score each candidate as

   ```
   smoothed_succ_r = (successes_r + 1) / (uses_r + 2)
   ucb_r = smoothed_succ_r + ucb_c × sqrt(log(N + 1) / (uses_r + 1))
   score_r = similarity_r × ucb_r
   ```

   where `N` is the total observed uses across candidates. Return
   `argmax(score_r)`.
6. **Cold start**: when no recipes pass the similarity filter, return
   the null recipe and tag the selection `explored=True` with a
   "cold-start" rationale.

### Why UCB1, not Thompson sampling

UCB1 is deterministic given the same counters — outcome audits become
trivial ("why did the selector pick X for this task?") and the
dashboard's per-recipe ranking stays stable across refreshes. Thompson
would add per-dispatch randomness with no benefit at our scale (low
hundreds of outcomes, not millions).

### Cold-start preference for the null arm

A useful consequence of UCB1's structure: the null recipe's exploration
bonus is **unbounded when its uses = 0**. Until the control arm has
been exercised at least a few dozen times, the bandit will prefer it
over any augmented candidate, no matter how well-evidenced. This is the
design intent — don't apply augmentation before establishing the
factory-default baseline. Verified by `test_cold_start_prefers_unexplored_null_arm`.

### IMMUTABLE thresholds

```python
SELECTION_THRESHOLDS = {
    "similarity_tau": 0.55,    # max cosine distance for candidacy
    "epsilon":         0.10,   # ε-greedy explore rate
    "ucb_c":           1.4,    # UCB1 exploration constant (= sqrt(2))
    "max_candidates":  8,      # similarity-search ceiling
}
```

These live as a module-level constant in `selector.py` — same
convention as `NOVELTY_THRESHOLDS` in `app.self_improvement.novelty`.
Tuning them requires editing the module, which is in
`TIER_IMMUTABLE`-by-convention (operator-only).

---

## 6. Feature flag

Three layers, each overriding the next:

| Priority | Source | Use case |
|---|---|---|
| 1 | `META_AGENT_<CREW>` env var | Ops override (emergency disable, A/B without redeploy) |
| 2 | `META_AGENT` env var       | Ops master switch — all crews ON |
| 3 | `meta_agent_settings.json` | Dashboard Org Chart toggle (everyday surface) |
| 4 | (default)                  | OFF |

The env layer mirrors `app.tool_runtime.feature_flags`. The JSON layer
mirrors `app.crews.delegation_settings`. The React Org Chart panel
detects when an env var is set and disables the toggle with an
`ENV LOCK` badge so the operator knows the JSON change has no effect
until the env var is unset.

---

## 7. HTTP API (Org Chart toggle)

| Method | Path                                | Purpose                                  |
|---|---|---|
| GET  | `/api/cp/meta-agent`                  | Return `{settings, master_env_on, env_overrides}` |
| POST | `/api/cp/meta-agent/{crew}`           | `{enabled: bool}` — toggle a single crew |

The GET response surfaces the env-layer state so the React panel can
render the lock badge:

```json
{
  "settings": {"research": false, "coding": true, "writing": false},
  "master_env_on": false,
  "env_overrides": {}
}
```

POSTing to a crew not in the defaults set returns `404`.

---

## 8. React surface

`OrgChart.tsx` renders **two parallel panels** at the bottom of the
Org Chart tab:

- **Delegation Mode** — green accent, existing structural toggle
- **Meta-Agent** — blue accent, `EXPERIMENTAL` badge, this layer

Both follow the same row pattern (toggle · crew name · state badge ·
description). The Meta-Agent panel additionally:

- Shows a yellow `ENV LOCK` badge when `META_AGENT_<CREW>` is set;
  the toggle is disabled and the tooltip explains the override.
- Shows a yellow banner when the master `META_AGENT=1` is on,
  explaining that all per-crew toggles are forced ON.
- Renders an explanatory message if the endpoint returns no settings
  (e.g. when the gateway hasn't been restarted to pick up the route).

---

## 9. Amendment proposals

When the meta-agent observes a recurring pattern where a high-success
recipe is blocked by an immutable rule, it generates a written
**amendment proposal** instead of attempting any auto-modification.

### Detection (policy_gap.py)

A `PolicyGap` is emitted when a recipe satisfies all of:

- `uses ≥ 5` outcomes
- `smoothed_success_rate ≥ 0.65`
- has `extra_tool_names` that fail to resolve in the registry
  (likely an env-gated tool with an OFF default in the immutable
  config)

The detector best-effort matches the unresolved tool name against
`TIER_IMMUTABLE` / `TIER_GATED` paths to suggest a target for the
amendment.

### Proposal (amendment.py)

For each gap, `propose_immutable_amendment(gap)` does **only**:

1. Render a self-contained Markdown document with diagnosis +
   suggested edit + risk analysis + reversal plan.
2. Submit through the existing `app.proposals.create_proposal` flow
   with `proposal_type="config"` and `files=None` — explicitly NO
   auto-applied code changes.
3. Overwrite the auto-rendered `proposal.md` in
   `workspace/proposals/<NNN>_amend_*/` with the rich body so the
   operator's first view is the meta-agent's diagnosis.

The proposal **always** asks the operator to apply the edit by hand.
No code in this package can edit `app/auto_deployer.py`.

### Operator workflow

1. The proposal lands in `workspace/proposals/<NNN>_amend_*` and
   surfaces in the existing Signal digest + `/cp/proposals` dashboard.
2. The operator reads the diagnosis, reviews the affected recipes,
   and decides whether the success-rate signal warrants a tier
   change.
3. If they agree, they edit `app/auto_deployer.py` by hand. The next
   dispatch picks up the loosened protection.
4. If they disagree, they mark the proposal `rejected`. The dedup key
   prevents the same gap from re-proposing.

---

## 10. Schema

### Postgres tables (auto-created on first use)

```sql
CREATE TABLE meta_agent_recipes (
    id                  TEXT PRIMARY KEY,
    crew_name           TEXT NOT NULL,
    force_tier          TEXT,
    extra_tool_names    JSONB NOT NULL DEFAULT '[]'::jsonb,
    task_hint           TEXT NOT NULL DEFAULT '',
    max_execution_time  INTEGER,
    task_signature      TEXT NOT NULL DEFAULT '',
    proposed_by         TEXT NOT NULL DEFAULT 'meta_agent',
    notes               TEXT NOT NULL DEFAULT '',
    uses                INTEGER NOT NULL DEFAULT 0,
    successes           INTEGER NOT NULL DEFAULT 0,
    last_used_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX meta_agent_recipes_crew_idx       ON meta_agent_recipes (crew_name);
CREATE INDEX meta_agent_recipes_uses_idx       ON meta_agent_recipes (crew_name, uses DESC);

CREATE TABLE meta_agent_outcomes (
    id                  TEXT PRIMARY KEY,
    recipe_id           TEXT NOT NULL,
    crew_name           TEXT NOT NULL,
    task_id             TEXT NOT NULL,
    success             BOOLEAN NOT NULL,
    confidence          TEXT NOT NULL DEFAULT '',
    duration_s          DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_estimate       DOUBLE PRECISION NOT NULL DEFAULT 0,
    error_signature     TEXT NOT NULL DEFAULT '',
    user_feedback       TEXT NOT NULL DEFAULT '',
    task_signature      TEXT NOT NULL DEFAULT '',
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX meta_agent_outcomes_recipe_idx          ON meta_agent_outcomes (recipe_id);
CREATE INDEX meta_agent_outcomes_crew_recorded_idx   ON meta_agent_outcomes (crew_name, recorded_at DESC);
```

### ChromaDB collection

`meta_agent_recipes` — cosine-space HNSW. Documents are
`json.dumps(recipe.to_dict())`; embeddings are `embed(recipe.task_signature)`.
The collection is filtered by `crew_name` on every query so a coding
recipe never bleeds into a research dispatch.

---

## 11. Difference from delegation mode

| | **Delegation** (existing) | **Meta-Agent** (this layer) |
|---|---|---|
| What it changes | **Structure** of one crew dispatch | **Configuration** across crew dispatches |
| When it acts | At dispatch — splits crew into Coordinator + specialists | At dispatch — picks the best learned recipe |
| What it remembers | Nothing (static mode) | Cross-run outcome history (Postgres + Chroma) |
| Knobs it tunes | Whether to split | `force_tier`, `extra_tools`, `task_hint`, `max_execution_time` |
| Cost | ~2× LLM calls per task | ~1 extra embedding call per dispatch |
| Tradeoff | Tool-limit relief at cost of latency | Latency-neutral; learns slowly (cold start prefers null arm) |
| Wired into | `dispatch()` in each crew's `run()` | `run_single_agent_crew` in `base_crew.py` |

They're **orthogonal** and can be on independently per crew. The
meta-agent only fires on the single-agent dispatch path, so when
delegation is ON for a crew the meta-agent skips that dispatch (the
delegated path runs its own multi-agent flow). That's a coherent
boundary, not a bug — recipes are about how to configure one agent,
delegation is about how to fan out across multiple.

---

## 12. Operating procedures

### Enabling the meta-agent for a crew

1. Org Chart → Meta-Agent panel → toggle the crew row.
2. Verify the badge flips to `META-AGENT ON` (blue).
3. The next dispatch on that crew routes through the selector. The
   first dozen runs will mostly use the null recipe (cold start
   protection); augmented recipes start being chosen as `uses` on
   the null arm grows.

### Rolling back

- Toggle off in the dashboard — instant.
- Or set `META_AGENT_<CREW>=0` in env to override the JSON without
  needing a UI session.
- Or kill the master with `META_AGENT=` (unset).

### Reading the recipe state

```python
from app.self_improvement.meta_agent import list_recipes, list_outcomes

# All recipes for a crew, sorted by uses DESC
list_recipes(crew_name="coding", limit=50)

# Outcomes from the last 7 days
list_outcomes(crew_name="coding", since_days=7, limit=200)
```

### Pruning dead recipes

```python
from app.self_improvement.meta_agent.store import prune_dead_recipes

# Drop recipes with < 5 uses older than 90 days. Returns count.
prune_dead_recipes(min_uses_for_keep=5, max_age_days=90)
```

This is wired off the idle-scheduler reconciler list by design; call
it manually or add to `idle_scheduler.py` if pruning becomes load-bearing.

### Reviewing amendment proposals

The meta-agent's `policy_gap.scan_for_policy_gaps()` is not currently
on the idle-scheduler — it's surfaced as an opt-in API for the
operator. To run a sweep manually:

```python
from app.self_improvement.meta_agent import (
    scan_for_policy_gaps, propose_immutable_amendment,
)

gaps = scan_for_policy_gaps(crew_name="coding", since_days=30)
for gap in gaps:
    pid = propose_immutable_amendment(gap)
    print(f"proposal #{pid}: {gap.target_filepath}")
```

Each proposal lands in `/app/workspace/proposals/<NNN>_amend_*/` and
in the `/cp/proposals` dashboard. Apply by hand if you agree.

---

## 13. Safety properties (what holds, by construction)

1. **The agent factory output is the floor.** The recipe layer never
   replaces `create_coder` etc.; it only adds tools and overrides
   already-existing factory parameters. Verified at the apply boundary
   by `RecipeAugmentation.is_noop` for the null recipe and by the
   `test_null_recipe_yields_noop_augmentation` test.

2. **Selector thresholds are infrastructure constants.** Tuning
   requires editing `selector.py`. The package as a whole is marked
   `IMMUTABLE — infrastructure-level module` in every docstring; the
   operator-facing
   [AMENDMENT_protect_meta_agent.md](../workspace/proposals/meta_agent/AMENDMENT_protect_meta_agent.md)
   recommends adding the file paths to `TIER_IMMUTABLE` for
   defense-in-depth (operator action, not auto-applied).

3. **Failsafe by default.** Any error in the meta-agent path
   (selector failure, recorder failure, store unreachable) falls
   through to factory defaults with `logger.debug` and no observable
   effect on the user-facing flow.

4. **Default OFF.** The feature flag's default value is OFF and the
   wiring in `run_single_agent_crew` is guarded by
   `is_meta_agent_enabled(crew_name)` — when not enabled, the
   meta-agent modules aren't even imported at the dispatch site.

5. **Outcomes are append-only.** The recipe row's `uses`/`successes`
   counters are denormalised, but the underlying
   `meta_agent_outcomes` ledger is immutable history. Re-deriving
   counters from the ledger is a single `SELECT` if a counter ever
   drifts.

6. **Amendment proposals never auto-apply.** Every amendment is
   operator-action-only. No code in this package edits
   `app/auto_deployer.py`.

---

## 14. Tests

`tests/test_meta_agent.py` — 32 tests:

- `TestAgentRecipe` (8) — dataclass invariants, smoothing math, dict round-trip
- `TestRecipeSelection` (1) — selection summary properties
- `TestFeatureFlag` (4) — env / per-crew / master / default resolution
- `TestApplyRecipe` (5) — null recipe is no-op, force_tier override,
  task_hint formatting, unresolved tools logged not fatal
- `TestSelector` (5) — UCB+similarity argmax (steady state), cold-start
  null preference, ε-greedy explore branch, null always a candidate
- `TestPolicyGap` (3) — below-threshold no-gap, high-success unresolved-tool
  yields gap, deterministic gap_id dedup
- `TestAmendment` (2) — render includes all required sections,
  unknown target handled
- `TestRecorder` (2) — outcome writes through store, task_signature
  seeded on first observation
- `TestBaseCrewIntegration` (2) — feature-flag default OFF, selector
  is pure given the same rng

Existing crew-touching test suites verified unaffected:
`test_self_improvement_integration.py`, `test_plugin_registry.py`,
`test_tool_first.py` — 91 total tests green after the wiring change.

---

## 15. Future work (out of scope for this commit)

- **Idle-scheduler reconciler** for `policy_gap.scan_for_policy_gaps`
  — currently a manual API call.
- **Recipe nominator** — today recipes are seeded only by operator
  curation or by accumulated null-recipe outcomes that "promote"
  knobs. A nominator that reads `LearningGap` records and proposes
  candidate recipes (still bounded to the four knobs) is a clean
  extension.
- **Cross-crew transfer** — recipes are crew-scoped today. The
  Hyperagents paper's empirical claim about cross-domain transfer
  could be probed by lifting the `where={"crew_name": ...}` filter
  to a similarity-only lookup and tagging recipe origin in the
  outcome log. Worth measuring before shipping.
- **Promote the operator-facing TIER_IMMUTABLE amendment**:
  [`workspace/proposals/meta_agent/AMENDMENT_protect_meta_agent.md`](../workspace/proposals/meta_agent/AMENDMENT_protect_meta_agent.md)
  needs operator review before the meta-agent's own modules are
  enforced as immutable at the auto_deployer level.

---

## References

- arXiv:2603.19461 — Hyperagents (Zhang et al., Meta — March 2026)
- arXiv:2505.22954 — Darwin Gödel Machine (prior work)
- `docs/SELF_IMPROVEMENT.md` — broader self-improvement subsystem
- `docs/RECOVERY_LOOP.md`, `docs/COMPANION_LAYER.md` — sibling per-subsystem docs
- `app/crews/delegation_settings.py` — sibling toggle pattern
- `app.self_improvement.novelty` — IMMUTABLE-thresholds convention
