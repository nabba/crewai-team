# Creativity System

A multi-agent ideation pipeline implementing 7 mechanisms from the
creativity-synthesis research (DMAD ICLR'25, LLM Discussion COLM'24,
min-p sampling, conceptual blending, Torrance evaluation).

It is **dormant by default and activated by intent**: the pipeline only
runs when the user asks for ideation, not for routine writing. Running
costs ~10× a normal `writing` crew, so a per-run budget cap (default
$0.10) acts as a safety net.

---

## Why this exists

Standard single-agent generation is good at routine writing — summaries,
emails, documentation. It is poor at **genuine ideation**:

- Single agents converge fast, often on the first plausible answer.
- Sequential chains lose alternative framings the moment Phase 1 commits.
- Default sampling (T=0.7, top-p=0.95) suppresses the long tail where
  novel ideas live.

The creativity system addresses each failure mode with a specific
mechanism (M1–M7 below). Whether it actually produces more original
output is **measurable** via the Torrance scores returned with every
run; the data exists to evaluate the design rather than trust it.

---

## Activation paths

There are three independent ways the pipeline gets invoked. If any one
fires, the user gets creative output.

| Path | Where | Trigger |
|------|-------|---------|
| **Router selection** | `app/agents/commander/routing.py` ROUTING_PROMPT | LLM router picks `crew=creative` based on the task description (now framed as a default for exploration tasks, not an escape valve) |
| **Auto-promotion** | `routing.maybe_promote_to_creative()` | `crew=writing AND difficulty≥6 AND task contains brainstorm/ideate/novel/breakthrough/etc keyword` → rewritten to `creative` |
| **Direct dispatch** | `POST /config/creative_run` | User clicks "Run Creative" on the dashboard, bypassing routing entirely |

The auto-promotion fix exists because the router LLM was systematically
under-classifying brainstorm tasks as `writing`. Auto-promote is a
deterministic backstop; the budget cap prevents runaway cost from
over-eager promotion.

---

## Three-phase pipeline

Implemented in `app/crews/creative_crew.py:run_creative_crew()`.
Custom orchestrator — **not** `Process.hierarchical` — to retain
per-phase control over sampling, agents, and prompts.

```
                        ┌─────────────────────┐
   user task ─────────► │  Phase 1: Diverge   │  4 agents in parallel
                        │  (Initiation)       │  Different reasoning methods
                        │                     │  Heterogeneous tiers
                        │  4 agents × ≥4 ideas│  Sampling: T=1.3, top-p=0.95
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Phase 2: Discuss   │  N rounds (default 2)
                        │                     │  Round 1: conformity (build on peers)
                        │  Conformity ↔       │  Round 2: anti-conformity (probe flaws)
                        │  Anti-conformity    │  Sampling: T=0.9, top-p=0.92
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Phase 3: Converge  │  1 commander synthesizer
                        │                     │  Premium tier (Claude Sonnet)
                        │  Synthesize 2-4     │  Sampling: T=0.5, top-p=0.90
                        │  strongest ideas    │  Originality × Feasibility ranking
                        └──────────┬──────────┘
                                   │
                                   ▼
                          {final_output, scores, cost}
```

### Phase 1 — Diverge (Initiation)

- 4 agents (researcher, writer, coder, critic) generate ideas independently.
- Each agent receives a different reasoning-method preamble (M1).
- Each agent runs on a different model tier (M4): local → mid → premium.
- High-temperature sampling (M5): T=1.3, min-p=0.05 (Ollama only).
- Output: ≥4 numbered ideas per agent, no convergence.

### Phase 2 — Discuss

- Default 2 rounds, configurable via `discussion_rounds` arg.
- Odd rounds = conformity: each agent sees peer outputs, builds on them.
- Even rounds = anti-conformity: only the critic runs, probing for
  load-bearing assumptions and unexplored alternatives (M3).
- Sampling moderates: T=0.9, top-p=0.92.

### Phase 3 — Converge

- Commander runs as a dedicated synthesizer Agent (not the routing
  Commander class — a fresh Agent built with the commander soul).
- Sees all Phase 1 + Phase 2 outputs as `<candidate_ideas>`.
- Selects 2–4 strongest ideas using **originality × feasibility ×
  structural coherence** as the criteria.
- Low-temperature sampling: T=0.5, top-p=0.90.
- Returns the synthesized output as the final result.

---

## The 7 mechanisms

| # | Mechanism | Implementation | File |
|---|-----------|----------------|------|
| **M1** | Reasoning-method diversity (DMAD) | 5 method preambles injected into agent backstories at runtime | `app/souls/reasoning_methods.md`, `app/souls/loader.py:get_reasoning_method()` |
| **M2** | 3-phase orchestration (Diverge→Discuss→Converge) | Sequential phases with budget checkpointing | `app/crews/creative_crew.py:run_creative_crew()` |
| **M3** | Anti-conformity passes | Critic agent in even discussion rounds with adversarial prompt | `app/crews/creative_prompts.py:render_anti_conformity()` |
| **M4** | Heterogeneous model tiers | Per-role tier assignment (local/budget/mid/premium) | `_TIER_BY_ROLE_CREATIVE` in `creative_crew.py` |
| **M5** | Phase-dependent sampling | Temperature/top-p/min-p/presence-penalty per phase per provider | `app/llm_sampling.py`, threaded via `app/llm_factory.py:create_specialist_llm(phase=…)` |
| **M6** | Conceptual blending | Tool that retrieves philosophy + fiction passages and templates a blend prompt with [PIT]/[PIH] epistemic tags | `app/tools/blend_tool.py` (attached to writer) |
| **M7** | Torrance creativity scoring | Fluency / flexibility / originality / elaboration computed at infrastructure level after each run | `app/personality/creativity_scoring.py` |

### M1 — Reasoning methods

Each agent gets a different reasoning preamble, breaking the default
chain-of-thought monoculture:

| Agent | Method | What it does |
|-------|--------|--------------|
| commander | `meta_reasoning` | Enumerates 2–3 strategies, picks one with justification |
| researcher | `step_back` | Abstracts to general principles before specific case |
| coder | `compositional_cot` | Decomposes → solves parts → recomposes (recomposition is where new constraints surface) |
| writer | `analogical_blending` | Maps relational structure across distant domains |
| critic / introspector | `contrastive` | Inverts load-bearing assumptions |

Mappings live in `_REASONING_METHOD_BY_ROLE` in `creative_crew.py`.

### M5 — Sampling table

| Phase | Temperature | top-p | min-p (Ollama) | presence_penalty |
|-------|-------------|-------|---------------|-------------------|
| diverge | 1.3 | 0.95 | 0.05 | 0.5 |
| discuss | 0.9 | 0.92 | 0.10 | 0.3 |
| converge | 0.5 | 0.90 | 0.10 | 0.0 |

Provider-specific translation in `app/llm_sampling.py:build_llm_kwargs()`:
- Anthropic: temperature + top-p only.
- OpenRouter: + presence_penalty.
- Ollama: + `extra_body.options.min_p` (litellm passthrough).

When `phase=None` the kwargs dict is empty → legacy callers are
byte-identical to before. Verified via cache-identity test.

### M7 — Torrance scoring

Every run returns four metrics:

| Metric | What | How |
|--------|------|-----|
| `fluency` | Number of distinct ideas | List-marker extraction (`extract_ideas()`) |
| `flexibility` | Number of distinct categories | Greedy embedding clustering at 0.35 cosine distance |
| `originality` | Semantic distance from corpus | `0.6 × wiki_distance + 0.4 × mem0_distance` (weights adjustable via dashboard) |
| `elaboration` | Detail depth | 0.6 × length-saturation + 0.4 × detail-marker density |

Originality compares against:
- `wiki_corpus` ChromaDB collection (757 chunks from `wiki/*.md`,
  re-ingested via `scripts/ingest_wiki_corpus.py`).
- The agent's own Mem0 history (catches self-repetition — the
  PDS anti-gaming concern).

When either source is unavailable, the scorer **degrades to zero**
rather than raising — diagnostics surface in the `diagnostics` field
of `CreativityScores`.

---

## Component reference

### Core modules

| File | Lines | Role |
|------|-------|------|
| `app/crews/creative_crew.py` | 510 | Pipeline orchestrator |
| `app/crews/creative_prompts.py` | 145 | Phase prompt templates |
| `app/llm_sampling.py` | 100 | Phase→sampling-kwargs translation |
| `app/creative_mode.py` | 80 | Runtime-mutable budget + originality weight |
| `app/personality/creativity_scoring.py` | 230 | Torrance metrics |
| `app/failure_modes.py` | 200 | 5-mode catalog with detectors |
| `app/agents/observer.py` | 165 | Failure prediction agent |
| `app/tools/blend_tool.py` | 165 | Conceptual blending with [PIT]/[PIH] tags |
| `app/tools/blackboard_tool.py` | 130 | Cross-agent shared workspace |
| `app/memory/scoped_memory.py` | (extended) | `store_finding`/`retrieve_findings`/`promote_to_knowledge_base` |
| `app/subia/belief/internal_state.py` | (extended) | `MetacognitiveStateVector` (5-dim) |
| `app/souls/reasoning_methods.md` | 100 | 5 reasoning-method preambles |
| `app/souls/loader.py` | (extended) | `compose_backstory(role, reasoning_method=...)` |

### Wiring points

| Hook | Where | When |
|------|-------|------|
| Commander routing → creative crew | `orchestrator.py:_run_crew()` `elif crew_name == "creative"` | Every dispatched task |
| Auto-promote `writing→creative` | `orchestrator.py` post-validation, before homeostasis | Every routing decision |
| Observer activation | `orchestrator.py:_run_crew()` after PRE_LLM_CALL hook | When `mcsv.requires_observer == True` |
| MAP-Elites archival | `orchestrator.py` post-crew (`map_elites_wiring.record_crew_outcome`) | Every crew completion |
| Torrance scoring | End of `run_creative_crew()` | Every creative run |
| Failure-mode scan | End of `run_creative_crew()` | Every creative run |

---

## Configuration

### Settings (`app/config.py`)

```python
creative_run_budget_usd: float = 0.10        # hard cap per run, dashboard-adjustable
creative_originality_wiki_weight: float = 0.6 # vs Mem0 weight (= 1 - this)
```

### Runtime settings (`app/creative_mode.py`)

Mutable via `POST /config/creative_mode`:

```json
{
  "creative_run_budget_usd": 0.25,
  "originality_wiki_weight": 0.7
}
```

Read once per run via `get_budget_usd()` to avoid mid-run drift.

### Budget-aware degradation

```python
if budget_usd < 0.05 and creativity == "high":
    creativity = "medium"   # auto-downgrade
if creativity == "medium":
    discussion_rounds = 1   # skip anti-conformity round
```

When the budget cap fires mid-phase, the run aborts and returns the
**best-so-far output** from whichever phase last completed. The
`aborted_reason` field on the result tells the caller what happened.

---

## API

### `GET /config/creative_mode`

Returns current runtime settings.

```json
{
  "creative_run_budget_usd": 0.10,
  "originality_wiki_weight": 0.6,
  "mem0_weight": 0.4
}
```

### `POST /config/creative_mode` (gateway-protected)

Updates settings. Accepts any subset of keys.

### `POST /config/creative_run` (gateway-protected)

Force-dispatches a task to the creative crew, bypassing routing.

```bash
curl -X POST http://localhost:8765/config/creative_run \
  -H "Authorization: Bearer $GATEWAY_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"task": "Brainstorm 5 ways to make Helsinki winter welcoming", "creativity": "high"}'
```

Returns:
```json
{
  "final_output": "...",
  "scores": {"fluency": 5, "flexibility": 4, "originality": 0.74, "elaboration": 0.61, "diagnostics": {...}},
  "cost_usd": 0.087,
  "aborted_reason": null,
  "phases": 7
}
```

---

## Dashboard

`dashboard-react/src/components/CreativeModeSettings.tsx` provides:

1. **Budget cap input** ($USD per run, default $0.10).
2. **Originality weight slider** (% wiki vs % Mem0).
3. **Try Creative Mode panel** — textarea + "Run Creative" button + result display showing the synthesized output and Torrance scores.

Wired into `Dashboard.tsx`. Both sections share the same `useCreativeModeQuery` / `useUpdateCreativeMode` / `useCreativeRun` React Query hooks (`dashboard-react/src/api/queries.ts`).

---

## Safety invariants

The creativity system **must not** violate the CLAUDE.md rule that
evaluation functions and safety constraints live at infrastructure
level, outside agent-modifiable code paths. Audit:

| Concern | Status |
|---------|--------|
| Torrance scoring | ✅ `app/personality/creativity_scoring.py` — infrastructure-level, not callable from agent prompts |
| Failure-mode detectors | ✅ `app/failure_modes.py` — pure-Python detectors, agents cannot rewrite them |
| Anti-conformity prompts | ✅ Prompt-level only, executed by the Critic agent which is non-self-modifying |
| Budget cap | ✅ Settings + `creative_mode.py` runtime guard outside agent reach |
| `SOUL.md` / `CONSTITUTION.md` | ✅ Untouched; new reasoning methods live in separate `reasoning_methods.md` |
| Observer prediction | ✅ Hook-level execution, JSON output consumed by orchestrator (not by the monitored agent) |

---

## Failure modes (`app/failure_modes.py`)

A canonical catalog with detectors that scan every creative run output:

| Mode | Detector trigger | Remediation hint |
|------|------------------|-------------------|
| `confidence_mirage` | 2+ certainty phrases without URLs/citations | "Cite specific sources or downgrade certainty language." |
| `fix_spiral` | 3+ consecutive history entries with >50% word overlap | "Step back: are you fixing the right problem?" |
| `consensus_collapse` | 3+ "I agree" patterns with ≤1 new idea | "Inject anti-conformity. Re-roll with critic in lead." |
| `hallucinated_citation` | 2+ `example.com`-style URLs | "Verify all citations via web_fetch. Remove unverifiable ones." |
| `scope_creep` | >40% of sentences off-topic vs task | "Re-scope: what was the user actually asking?" |

Recorded as benchmark metrics with `record_metric("creative_failure_modes", n, {...})` so distributions are observable over time.

---

## Metacognitive State Vector (MCSV)

`MetacognitiveStateVector` in `app/subia/belief/internal_state.py` —
5 dimensions complementing the existing 6-dim `CertaintyVector`:

| Dimension | Source signal |
|-----------|---------------|
| `emotional_awareness` | `SomaticMarker.intensity` |
| `correctness_evaluation` | `CertaintyVector.factual_grounding` |
| `experience_matching` | Mem0 hit rate on task features |
| `conflict_detection` | Cogito contradiction count |
| `complexity_assessment` | Token depth × tool-call depth |

Derived properties:
- `novelty = (1 - experience_matching) × (0.5 + 0.5 × conflict_detection)`
- `requires_observer = (correctness < 0.4) OR (conflict > 0.6)`

The Observer agent (`app/agents/observer.py`) only fires when
`requires_observer == True`, keeping cost bounded.

---

## Blackboard

Long-lived per-task shared workspace replacing string-passing chains.
Backed by ChromaDB (scope `scope_research_bb--{task_id}`). Findings
carry structured metadata:

```python
{
    "claim": "Helsinki population is 657,674 (2024)",
    "evidence": "Statistics Finland",
    "confidence": "high",        # high | medium | low
    "source_url": "https://...",
    "agent": "researcher",
    "verification_status": "verified",  # verified | unverified | contradicted
    "ts": "2026-04-26T15:32:00Z",
}
```

Two tools auto-attached to the researcher when `task_id` is set:
- `deposit_finding(claim, evidence, confidence, source_url, status)`
- `read_findings(query, n=5, confidence_filter=None)`

Verified findings can be **promoted** to the project knowledge base via
`promote_to_knowledge_base(task_id)` — the bridge from ephemeral
research scratchpad to long-term institutional memory.

---

## MCP integration (`app/mcp/`)

The creativity system exposes its observable surface via MCP at
`/mcp/sse`:

| Resource URI | What it returns |
|--------------|-----------------|
| `mcsv://current` | Live metacognitive state vector |
| `philosophy://...` | Philosophy RAG passages |
| `blackboard://{task_id}` | Findings deposited so far for a research task |
| `personality://current` | PDS state |
| `memory://recent` | Recent Mem0 entries |

Plus tools: `score_creativity(text)`, `read_blackboard(task_id)`, etc.
External MCP clients (Claude Desktop, etc.) can introspect and operate
on system state without going through the Signal gateway.

---

## Testing

117 tests organized into three files:

| File | Class count | Test count | Scope |
|------|-------------|------------|-------|
| `tests/test_creativity.py` | 9 | 76 | Pure-logic units (sampling, prompts, scoring, MCSV, MAP-Elites, auto-promote, etc.) |
| `tests/test_research.py` | 6 | 24 | Mocked external services (blackboard, blend tool, observer, MCP helpers) |
| `tests/test_e2e_creative_research.py` | 8 | 26 | Integration + E2E (creative crew internals, MAP-Elites wiring, Observer flow, real ChromaDB) |

```bash
# Fast unit tests (no Docker needed, ~3s)
pytest tests/test_creativity.py tests/test_research.py -v

# Full suite including E2E (requires Docker stack, ~4s)
docker exec crewai-team-gateway-1 python -m pytest tests/ -v
```

E2E tests are gated by the `e2e` pytest marker. Run with `-m e2e` to
include them, `-m "not e2e"` to skip.

---

## Operational guide

### How to know it's running

1. Dashboard: `Creative Mode` panel shows current budget + scores from any prior runs.
2. Logs: every promotion is logged: `creative auto-promote: writing → creative (difficulty=8, task='Brainstorm…')`.
3. Metrics: query the benchmarks journal for `creative_run_time` / `creative_run_cost` / `creative_failure_modes`.

### How to debug a failed run

1. Check `aborted_reason` in the response. If "budget exceeded", raise the cap.
2. Check `scores.diagnostics` — empty embeddings or missing wiki corpus
   show up here.
3. Check container logs for `creative_crew:` entries — each phase logs
   its agent invocations.
4. The `final_output` may be a best-so-far fallback if the run aborted
   mid-phase; check `aborted_reason` to confirm.

### How to tune

| Knob | Where | Effect |
|------|-------|--------|
| Budget cap | Dashboard → Creative Mode panel | Raises the per-run cost ceiling |
| Originality weight | Dashboard slider | Higher = score against shared corpus more; lower = score against agent's own history more |
| Discussion rounds | `run_creative_crew(discussion_rounds=N)` | More rounds = more exploration, more cost |
| Reasoning methods | `_REASONING_METHOD_BY_ROLE` in `creative_crew.py` | Swap which method each role uses |
| Tier per role | `_TIER_BY_ROLE_CREATIVE` in `creative_crew.py` | Trade cost for quality per agent |
| Auto-promote keywords | `_CREATIVE_PROMOTION_PATTERNS` in `routing.py` | Add/remove triggers |
| Auto-promote min difficulty | `_CREATIVE_PROMOTION_MIN_DIFFICULTY` in `routing.py` | Default 6; lower = more aggressive promotion |

### How to extend

- **New reasoning method**: add a `## METHOD: <name>` section to
  `app/souls/reasoning_methods.md`, then assign it to a role in
  `_REASONING_METHOD_BY_ROLE`.
- **New failure mode**: add a `FailureMode` to `CATALOG` in
  `app/failure_modes.py` with a detector function. It's automatically
  scanned on every run.
- **New phase preset**: add an entry to `_PHASE_PRESETS` in
  `app/llm_sampling.py` and a `render_*` function in
  `app/crews/creative_prompts.py`.

---

## Known limitations

- **MAP-Elites grids fill slowly.** With current crew dispatch volume,
  cells outside `self_improve` accumulate ~1 entry per few hundred
  runs. The wiring is correct; the data is just sparse.
- **Wiki corpus quality determines originality scoring quality.** With
  757 chunks, scores are meaningfully grounded; with the original 7
  chunks they were not. Re-ingest if `wiki/` grows substantially.
- **Ollama `min_p` passthrough is [Unverified] against current Ollama
  version.** Anthropic + OpenRouter use temperature/top-p which are
  guaranteed to work.
- **Torrance flexibility threshold (0.35 cosine)** is a pragmatic
  default, not empirically tuned to your corpus.
- **Cost tracking depends on `app.rate_throttle.get_active_tracker()`.**
  If the tracker isn't initialized for a creative run (e.g. via direct
  function call without HTTP wrapper), the budget cap is a no-op.

---

## Decade-class primitives (`app/creativity/`, May 2026)

The creative-crew pipeline above is the *active* ideation machine —
it runs when the user asks for ideas. The `app/creativity/`
primitives are *passive* infrastructure: small, observational, always-
on data structures the active pipeline (and any other subsystem) can
consult. Three pieces.

### Cross-domain analogy index (`analogy_index.py`, §7.1)

Hofstadter-style structural retrieval. The intuition: many genuinely
new ideas come from noticing that two apparently unrelated systems
share the same underlying *structure* — feedback loop with delay,
producer-consumer with bounded buffer, gradient-descent with momentum
— and porting an insight from one domain to the other.

Data model:

```python
@dataclass(frozen=True)
class AnalogyEntry:
    id: str
    structure_signature: str       # short label
    structure_description: str     # 2-3 sentences
    examples: list[DomainExample]  # ≥1 witness per domain
    created_iso: str

@dataclass(frozen=True)
class DomainExample:
    domain: str          # "biology", "control_theory", ...
    instance: str        # 1-line specific example
    why_it_fits: str     # 1-line structural-match note
```

Storage: append-only JSONL at
`workspace/creativity/analogy_index.jsonl`. Last-write-wins on
duplicate `id`.

Retrieval is hash-trick cosine similarity over the concatenated
signature + description + examples. `query_analogies(text, top_k=5,
min_similarity=0.05, exclude_domains=…)` returns ranked matches.

Use cases:
* Brainstorm pipeline: query before committing — "is the framing I'm
  settling on actually a re-instance of one we've already seen?"
* Concept-blend operator (below): seed input spaces from analogous
  domains.

Master switch: `ANALOGY_INDEX_ENABLED` (default ON; pure read/write,
no LLM, no external call).

### Concept-blend operator (`concept_blend.py`, §7.2)

Fauconnier-Turner four-space conceptual blend. Given two input
concepts, the operator produces a typed `BlendResult` with input
spaces, generic structure (what the inputs share), blend label, blend
description, selected projections from each input, emergent structure
(what's new in the blend, not in inputs), and follow-on questions.

The operator takes a `llm_call(system, user) → str` injectable so
it's testable without an LLM. Output prompt forces strict JSON; the
parser tolerates code-fence wrappers and sets `parse_failed=True`
with a `parse_error` field rather than raising when the LLM emits
malformed JSON.

```python
result = blend_concepts(
    "wikipedia",   # input_a
    "wetland",     # input_b — Hofstadter's classic
    llm_call=my_llm_call,
)
# result.blend_label = "Wikiland"
# result.emergent_structure = ["self-purifying via edit war ecology", ...]
```

Master switch: `CONCEPT_BLEND_ENABLED` (default ON).

### Brainstorm-idea novelty wrapper (`novelty_wrap.py`, §7.4)

Wraps every brainstorm idea with a structural-novelty assessment.
Distinguishes "really new" (high structure-distance from prior ideas)
from "lexically rephrased" (low structure-distance, high lexical
distance — the failure mode the active pipeline's M3 mechanism is
designed to prevent).

Returns a `NoveltyVerdict` with a numeric score + reasoning the
brainstorm session can show alongside the idea. Composes with the
analogy index — a "novel" idea that turns out to match a known
structure-signature gets flagged.

### How the three compose

```
brainstorm idea
  │
  ↓
novelty_wrap.assess_brainstorm_idea(idea)
  │
  ├─→ analogy_index.query_analogies(idea.statement)
  │     ↓
  │   "actually we've seen this structure 3× before in {biology,
  │    economics, fluid dynamics}; you might be re-discovering
  │    Liebig's law"
  │
  └─→ concept_blend.blend_concepts(idea.input_a, idea.input_b)
        ↓
      "the blend that produced this idea has 4 emergent properties
       neither input has alone — that's what makes it novel"

→ NoveltyVerdict(score=0.7, reasoning="...")
```

All three primitives are observational. They surface signals; they
never overrule the active pipeline or the operator.

### Files

```
app/creativity/__init__.py        public API
app/creativity/analogy_index.py    §7.1 Hofstadter-style retrieval
app/creativity/concept_blend.py    §7.2 Fauconnier-Turner blend
app/creativity/novelty_wrap.py     §7.4 brainstorm novelty assessment

tests/creativity/test_analogy_index.py    11 tests
tests/creativity/test_concept_blend.py     9 tests
tests/creativity/test_novelty_wrap.py     12 tests
```

PROGRAM.md §32.3 covers the ship.

---

## See also

- `docs/ARCHITECTURE.md` — overall system architecture
- `app/crews/creative_crew.py` — the active-pipeline orchestrator
- `app/creativity/` — passive primitives (analogy / blend / novelty)
- `app/llm_sampling.py` — sampling parameter details
- `app/personality/creativity_scoring.py` — Torrance scoring details
- `tests/test_creativity.py` — active-pipeline tests
- `tests/creativity/` — passive primitives tests
