# LLM Subsystem

The complete reference for how this system chooses, calls, vets, and
learns from large-language-model usage. Written so a new contributor
can read it once and understand every layer end-to-end.

---

## 1. Why this design exists

Earlier versions hand-curated a 24-model dict and a static
`ROLE_DEFAULTS[mode][role]` table. Whenever a frontier model launched
(Claude Opus 4.7, DeepSeek V4, Kimi K2.6), someone had to edit Python
and ship a release before the system would *even consider* using it.
That defeated the point of having a model-aware agent stack at all.

The current system replaces that with **five interlocking feedback
loops**:

1. **Discovery** — every 24 h the catalog refreshes from live sources
   (Artificial Analysis, OpenRouter, Ollama). New models appear
   automatically.
2. **Promotion** — discovered models that Pareto-dominate an incumbent
   under cost-aware scoring become first-choice picks (free-tier auto;
   paid-tier via governance approval).
3. **External rankings** — third-party leaderboards (Artificial
   Analysis intelligence, coding, math indexes) blend into the
   selector's quality signal at a configurable weight.
4. **Telemetry** — every LLM call records latency, success, error,
   token usage; the per-task-type benchmarks shift the scoring weights
   over time.
5. **Vetting feedback** — when vetting fails on a low-tier model's
   output, the next selection for that role gets a tier bump or a
   different model entirely.

Plus three operator overrides for when humans know better:
**hand-pin** (force a specific model for a (role, mode) pair),
**promote** (mark a discovered model as a first-choice candidate), and
**runtime mode** (constrain the entire candidate pool — Free / Budget
/ Balanced / Quality / Insane / Anthropic).

---

## 2. Architecture (one screen)

```
                       ┌──────────────────────────────┐
                       │     create_specialist_llm    │   ◄── single
                       │   (app/llm_factory.py:299)   │       gateway
                       └──────────────┬───────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
          select_model()       get_default_for_role()    cascade fallback
          (llm_selector.py)    (llm_catalog.py)          (local → API → Claude)
                  │                   │
                  └────────┬──────────┘
                           ▼
            ┌─────────── resolve_role_default ────────────┐
            │  app/llm_catalog.py:533                      │
            │                                              │
            │  Layer 1  Hand-pin  ─►  return directly      │
            │  Layer 2  Promotion ─►  filter candidate set │
            │  Layer 3  Pool      ─►  score & pick best    │
            └──────────────┬───────────────────────────────┘
                           │
        reads policy:                          reads data:
        ─────────────                          ──────────
        _MODE_TIER_WHITELIST (per mode)        CATALOG (367 models)
        _MODE_PROVIDER_WHITELIST (anthropic)   strengths map (9 task types)
        _ROLE_TIER_FLOOR (commander=premium)   external ranks blend
        _ROLE_LOCAL_PREFERRED (planner etc.)   live benchmark scores
        _ROLES_NEEDING_TOOLS                   tool_use_reliability
        _MODE_WEIGHT (cost penalty)            cost_input/output_per_m

                           │
                           ▼
                  one of 367 catalog keys ──► instantiate LLM
```

Every layer fails soft. Every layer is overrideable. No layer is
allowed to be skipped.

---

## 3. The single-gateway invariant

**Every LLM call in the codebase goes through
`create_specialist_llm`.** This is enforced by convention, not the
type system, but verified by grep:

| Bypass vector | Allowed? |
|---|---|
| Direct `LLM(...)` outside `llm_factory.py` | ❌ — only one terminal call exists, at `llm_factory.py:119` |
| `ChatOpenAI` / `ChatAnthropic` / `OpenAI()` / `Anthropic()` direct instantiation | ❌ — zero matches anywhere |
| `Agent(llm="hardcoded-string")` (CrewAI accepts a model_id string) | ❌ — zero matches |
| `select_model()` called outside the factory | ❌ — only callsite is `llm_factory.py:347` |
| `force_tier=...` to constrain candidate pool | ✅ — by design; passes through resolver |

This invariant is what makes the dashboard's "view current pick per
role" report **honest** — it isn't a parallel computation that drifts
from runtime reality.

---

## 4. The 6-mode unified runtime axis

Single user-facing knob. Set via dashboard, Signal command, or env
var. Lives in `app/llm_mode.py` as a runtime-mutable singleton; reads
via `get_mode()`.

| Mode | Tier whitelist | Provider whitelist | Cost ceiling (output) | Cost weight | Use when |
|---|---|---|---|---|---|
| **free** | local + free | any | $0 | 0.50 | offline / zero-cost |
| **budget** | local + free + budget | any | ~$1.5/M | 0.35 | cost-minimised |
| **balanced** *(default)* | every tier | any | ~$6/M | 0.15 | normal operation |
| **quality** | every tier, premium-leaning | any | ~$30/M | 0.05 | best within reason |
| **insane** | premium only, no local | any | ∞ | 0.00 | money no object |
| **anthropic** | mid + premium | Anthropic only | ∞ | 0.15 | vendor lock |

Source of truth: `RUNTIME_MODES` tuple + `_MODE_*` policy dicts in
`app/llm_catalog.py`. Legacy aliases (`hybrid`, `local`, `cloud`) are
auto-normalised by `_normalize_mode()` so old configs keep working.

The mode controls **two** things at once:
1. Which catalog tiers are even eligible (`_MODE_TIER_WHITELIST`).
2. How aggressively the resolver penalises cost in scoring
   (`_MODE_WEIGHT`).

That collapse is the whole point of the unification — previously a
separate `cost_mode` axis (`budget`/`balanced`/`quality`) duplicated
half this signal in a confusing way.

---

## 5. The catalog

`app/llm_catalog.py` exposes a single `CATALOG: dict[str, dict]` that
the resolver scores against. Structure of each entry:

```python
"claude-opus-4.7": {
    "tier": "premium",                 # local | free | budget | mid | premium
    "provider": "anthropic",           # ollama | openrouter | anthropic
    "model_id": "anthropic/claude-opus-4-7",
    "context": 1_000_000,
    "multimodal": True,
    "supports_tools": True,
    "cost_input_per_m": 5.00,
    "cost_output_per_m": 25.00,
    "tool_use_reliability": 0.97,
    "knowledge_cutoff": "2025-09-01",  # optional — see §5.3 below
    "strengths": {
        "coding":       0.95,
        "debugging":    0.93,
        "architecture": 0.94,
        "research":     0.92,
        "writing":      0.93,
        "reasoning":    0.95,
        "multimodal":   1.0,
        "vetting":      0.94,
        "general":      0.94,
    },
    "description": "Claude Opus 4.7 — Anthropic frontier reasoning.",
},
```

### 5.1 Bootstrap (survival minimum)

`_BOOTSTRAP_CATALOG` is a 3-entry hard-coded dict that exists only so
the system boots when every external API is down and no snapshot is on
disk:

| Key | Purpose |
|---|---|
| `claude-sonnet-4.6` | Premium Anthropic fallback |
| `deepseek-v3.2` | Budget OpenRouter fallback |
| `qwen3:30b-a3b` | Local Ollama fallback |

These are mutated in place when the builder refreshes their derived
fields, but never removed. If the resolver's filter set is empty, it
returns `claude-sonnet-4.6` — the universal bootstrap fallback.

### 5.2 Auto-population (the builder)

`app/llm_catalog_builder.py` runs every 24 h via the idle scheduler
job `llm-refresh-catalog`. It:

1. Fetches `https://artificialanalysis.ai/api/v2/data/llms/models`
   (intelligence index, coding index, math index, gpqa, livecodebench,
   ifbench, pricing, median tps).
2. Fetches `https://openrouter.ai/api/v1/models` (cost, context,
   modality, `supported_parameters`).
3. Probes the local Ollama daemon at `/api/tags` for installed models.
4. Cross-references by canonical key (the same `_resolve_catalog_key`
   helper that `llm_external_ranks.py` uses).
5. Calls `derive_strengths(aa_row, is_multimodal, tier)` to map the
   AA evaluation columns onto our 9 canonical task types.
6. Persists to `workspace/cache/llm_catalog_snapshot.json` (24-h TTL).
7. Mutates the live `CATALOG` dict in place — module-level imports of
   `from app.llm_catalog import CATALOG` keep working without restart.

A typical refresh produces ~360 entries. Manual trigger: send Signal
command `refresh catalog` or POST to `/api/cp/llms/discovery/run`.

#### Embedding-only models are filtered out

Both ingestion paths (`_build_local_entry` in the catalog builder and
`scan_ollama` in `app/llm_discovery.py`) reject any model whose name
contains `embed`. Without this guard, embedding-only Ollama models like
`nomic-embed-text` got registered as `ollama_chat/...` chat models, the
selector eventually picked one for a chat role, and `litellm.completion`
returned `400 "<model> does not support chat"` on every call. Working
embeddings already flow through `app/memory/chromadb_manager.py` which
calls Ollama's `/api/embeddings` endpoint directly — never the catalog.

#### OpenRouter "Stealth" provider is excluded

`_cached_llm` injects `extra_body={"provider": {"ignore": ["Stealth"]}}`
into every OpenRouter request (any LLM whose `base_url` matches
`openrouter.ai`). OpenRouter's anonymous "Stealth" provider class
periodically returns `502 "Invalid URL: ''"` — historically the highest-
volume single-cause error class on the system. The exclusion list is
overridable via the `OPENROUTER_IGNORE_PROVIDERS` env var (CSV); set
empty to disable filtering. The mechanism mirrors the parallel block
that injects `extra_headers` for Anthropic prompt-caching, so adding
more provider names later is a one-line change.

### 5.3 Knowledge cutoff

Optional `knowledge_cutoff: str` field (ISO date, `YYYY-MM-DD`) records
the model's training-data horizon. Two sources populate it:

* **Live builder** — `_build_openrouter_entry` extracts `created_at`
  from the OpenRouter `/models` payload (Unix epoch UTC, the day the
  model was added to OpenRouter). This trails the real training cutoff
  by 1-3 months but is a usable lower bound.
* **Bootstrap** — survival entries hardcode published cutoffs from each
  model card (`deepseek-v3.2 → 2024-07-01`, `qwen3.5:35b → 2024-09-01`).
  The merge step refreshes these from live data when available.

The field is optional — entries without it pass selector filters (the
absence of evidence isn't evidence of absence). Anthropic entries
currently have no automatic source for the cutoff; backfilling them
from a static map is a tracked follow-up.

The selector's `min_recency` parameter (see §7.1 step 3b) consumes this
field to drop stale models when the caller needs current information.

### 5.4 Strengths map (9 canonical task types)

```
coding, debugging, architecture, research, writing,
reasoning, multimodal, vetting, general
```

Roles map to task types via `_ROLE_TO_TASK` in `llm_catalog.py`:

| Role | Task type | Notes |
|---|---|---|
| commander | general | Routing reliability, light tokens |
| critic | reasoning | Adversarial review |
| vetting | vetting | Output-quality gate |
| synthesis | writing | Multi-source merge |
| introspector | reasoning | Meta-cognitive |
| self_improve | research | Background reflection |
| planner | architecture | Topic decomposition |
| evo_critic | reasoning | LLM-Judge for evolution variants |
| coding | coding | Crews + standalone agents |
| research | research | Crews + standalone agents |
| writing | writing | Crews + standalone agents |
| media | multimodal | Image / PDF / audio |

Custom hint can override via `task_hint=` keyword to `select_model()`.

---

## 6. The 3-layer resolver

`resolve_role_default(role, mode)` in `app/llm_catalog.py` (≈line
533). The authority cake, strongest first:

### Layer 1 — Hand-pin (hard override)

Active row in `control_plane.role_assignments` with
`priority ≥ HAND_PIN_PRIORITY` (=1000) for `(role, mode)`.

If found and its `model` exists in the live `CATALOG`, the resolver
returns it **directly without scoring**. This is the dashboard's
📌 pin button + the Signal `pin <role> <mode> <model>` command.

`unpin_role()` retires the pin; resolver takes over again.

### Layer 2 — Promotion filter

Models in `control_plane.model_promotions` are "first-choice"
candidates. If any promoted model survives the hard filters
(tier whitelist, provider whitelist, multimodal need, tool need),
candidates **collapse to the promoted set** before scoring.

Promotion sources:
- Free-tier discovery: auto (`_promote_model` in `llm_discovery.py`).
- Paid-tier discovery: governance approval required (writes a
  `governance_requests` row of type `model_promotion`; user clicks
  Approve in the dashboard or Signal).
- Manual: dashboard button or Signal `promote <model>`.

### Layer 3 — Pool scoring

The default path. After hard filters:

```
quality = 0.60 · benchmark_score          (live telemetry)
        + 0.35 · catalog_strengths        (derived from AA / bootstrap)
        + 0.05 · tool_use_reliability     (only if role needs tools)

cost_penalty = mode_weight · (cost_per_m / max_cost_in_candidates)

score = quality − cost_penalty
```

When live benchmarks aren't yet populated for a model, the formula
falls back to `0.80·strengths + 0.20·tool_use_reliability` and skips
the cost normalisation against unknowns. The point is: a 5% quality
bump isn't worth a 20× cost increase under `budget` mode, but is
under `quality`.

`max(candidates, key=score)` returns the winner.

### 6.1 Effective tier floor reconciliation

`_effective_tier_floor(mode, role_tier_floor)`. The role-level floor
gets capped by the mode's max allowed tier so the resolver honours the
user's explicit mode choice rather than silently escalating:

| Role | role_tier_floor | mode | effective_tier_floor |
|---|---|---|---|
| commander | premium | balanced | premium |
| commander | premium | free | free *(capped)* |
| commander | premium | insane | premium |
| coding | budget | free | free *(capped)* |
| coding | budget | quality | budget |

Without this reconciliation, `free + commander` would silently
escalate to Claude premium because `tier_floor=premium` had no
satisfying entries in `{local, free}`.

### 6.2 Local preference

A small set of roles (`_ROLE_LOCAL_PREFERRED` =
{`introspector`, `self_improve`, `planner`, `evo_critic`}) prefers
local-tier picks when the mode allows it (`_MODE_PREFER_LOCAL` =
{`free`, `budget`, `balanced`}). For these roles in those modes, the
resolver narrows candidates to local-only first, then scores. Quality
/ insane / anthropic explicitly opt out.

---

## 7. The selector

`app/llm_selector.py:select_model(role, task_hint, force_tier)` is the
intermediate layer between the factory and the resolver. It applies
**non-mode** constraints — environment overrides, task-type detection,
difficulty bumps, runtime ContextVar — that the resolver doesn't know
about.

### 7.1 Selection sequence

1. **Env override** — `ROLE_MODEL_RESEARCH=kimi-k2.5` (case-insensitive).
   Skips everything else; verified to be in `CATALOG`.
2. **Default from resolver** — `get_default_for_role(role, get_mode())`
   reads the live runtime mode (so dashboard switches take effect on
   the next call, no restart).
3. **Task-type detection** — `canonical_task_type(role, task_hint)`
   keyword-scans the hint (`"debug stack trace"` → debugging,
   `"image"` → multimodal) and may swap the pick.
3b. **Recency filter** — when `min_recency: date` is passed, drop the
   default if its catalog `knowledge_cutoff` is strictly older and walk
   the candidates by score for a replacement. Models without a
   `knowledge_cutoff` field pass through (treated as unknown). If no
   compliant candidate exists, the default is kept and a warning is
   logged (graceful degradation). Single-pass at this step;
   downstream score / cost / budget swaps may still drift, so callers
   wanting strict guarantees should also pass `force_tier`. Filter is
   only active in `mode == "balanced"` — restrictive modes go through
   `_pool_constrained_select` which doesn't yet thread `min_recency`.
4. **Special rules** — e.g. multimodal hint forces a multimodal model
   regardless of strengths score.
5. **Benchmark adjustment** — `get_scores(task_type)` may shift the
   pick if the resolver chose something that's been failing in
   production.
6. **Availability check** — local: ping Ollama; API: verify key set.
7. **Return** — the catalog key.

### 7.2 Difficulty bumps

`difficulty_to_tier(difficulty, mode)` and
`_resolve_difficulty_tier_floor(role, difficulty)` provide a 1-10
difficulty axis that orthogonally bumps tier floors. Difficulty is set
by the orchestrator at task entry and propagated via the
`_active_difficulty` ContextVar so deeply-nested sub-agents inherit it
without explicit threading.

```
difficulty 1-3  →  budget tier floor (local OK if mode allows)
difficulty 4-7  →  default catalog logic
difficulty 8-10 →  premium tier floor

# Plus role-specific overrides:
research at d=8  →  premium (tighter than coding at d=8)
research at d=7  →  mid
coding at d=9    →  premium
writing at d=9   →  premium
```

This catches the case where a sub-agent (e.g. CrewAI's
`delegate_work_to_coworker` spawning a "Web Research Specialist"
inside a coordinator) wouldn't otherwise know how hard the parent
task is.

### 7.3 Resource-aware local picks

`_select_local_resource_aware()` checks Ollama VRAM usage + system
RAM + model `size_gb` from the catalog and prefers models already
loaded. RAM headroom of 16 GB is reserved for the OS + embeddings.

---

## 8. The factory (single gateway)

`app/llm_factory.py:create_specialist_llm(max_tokens, role, task_hint,
force_tier, phase, min_recency)` is the only function that constructs
`LLM` instances.

### Recency auto-default

`_DEFAULT_RECENCY_DAYS_BY_ROLE` maps a small set of roles to a default
lookback in days; when the caller doesn't pass `min_recency`, the
factory derives `date.today() - timedelta(days=N)` for those roles
before calling `select_model`. Currently:

| Role | Lookback | Why |
|---|---|---|
| `research` | 180 d | Web research / dossier collector / tech radar — outputs depend on current information |
| `self_improve` | 180 d | Post-mortems benefit from current best-practice |

Other roles (coding, writing, vetting, etc.) get no auto-default.
Callers can override at any call site by passing an explicit
`min_recency=date(...)`, or opt out with the sentinel
`min_recency=date.min`. This keeps the recency floor in one place
rather than spread across every research-adjacent caller.

### 8.1 Mode dispatch

```python
mode = get_mode()                # live runtime mode

if mode != "balanced":
    # Restrictive modes (free / budget / quality / insane / anthropic)
    # constrain the pool, then run the regular selector inside it.
    chosen = _pool_constrained_select(role, task_hint, mode, force_tier)
    return _build_from_entry(*chosen, ...) or claude_fallback(...)

# Balanced (default): unconstrained selector + full cascade fallback.
model = select_model(role, task_hint, force_tier)
entry = get_model(model)

if tier == "local" and settings.local_llm_enabled:
    llm = try_local(...)
    if llm: return _maybe_race_wrap(llm, ...)   # Stage 4.3 race-with-API
    if settings.api_tier_enabled:
        api_model = get_default_for_role(role, mode)
        try_api(api_model)                       # cascade up
    return claude_fallback(...)                  # universal bottom

if tier in (free, budget, mid):  try_api(...) or claude_fallback(...)
if provider == "anthropic":      try_anthropic(...)
```

### 8.2 LLM object cache

LLM instances are cached by `(builder-tag, model_id, max_tokens,
base_url, sampling_key)`. They're stateless wrappers over `(model_id,
api_key, params)` so sharing across requests is safe. Saves
~50-100 ms per specialist call.

### 8.3 Last-pick tracking (per-thread)

`is_using_local()`, `is_using_api_tier()`, `get_last_model()`,
`get_last_tier()` read a `threading.local` populated by the cascade —
used for telemetry attribution and the dashboard's "currently using"
indicator.

---

## 9. Vetting

`app/vetting.py`. **Not** every LLM call gets vetted; that would
double the cost. Vetting is risk-based across 4 tiers:

| Risk | Action | When |
|---|---|---|
| `none` | skip | Direct user answers, easy + premium model |
| `schema` | format / sanity check, no LLM call | Structured outputs from any tier |
| `cheap` | yes/no via budget model | Mid-tier output on routine task |
| `full` | full Claude Sonnet review | All local Ollama output, all code |

Risk derives from `crew_type + difficulty_score + model_tier`. The
vetting LLM itself is selected via the same resolver path with
`role="vetting"`, so it inherits the runtime mode (e.g. in
`anthropic` mode the vetting model is always Anthropic).

Vetting is bounded — `llm.call()` has a hard timeout to prevent the
2026-04-25 case where gpt-5.5 hung for 17 minutes inside a vetting
call and stalled the parent task lifecycle.

Failed vetting feeds back into the next selection: the role's tier
floor lifts, or the model gets a benchmark penalty, depending on the
failure shape.

---

## 10. Discovery & promotion

### 10.1 Discovery

`app/llm_discovery.py`. Runs on the idle scheduler. Consumes the
external-rank fetchers + the catalog snapshot to identify candidates
that **Pareto-dominate the incumbent**:

> Candidate dominates incumbent if
>   `quality_candidate ≥ quality_incumbent`
> AND
>   `cost_candidate ≤ cost_incumbent · (1 − cost_penalty[mode])`
>
> with at least one strict inequality.

`cost_penalty` is `0.35` in budget, `0.10` in balanced, `0.00` in
quality (so quality requires strict cost reduction; budget tolerates
some cost increase for big quality gains).

A `_discover_judges()` rotation picks one top-intelligence model from
each of three different provider families to act as the LLM-as-Judge
for cross-evaluation, preventing self-reinforcing bias.

### 10.2 Promotion

When a candidate dominates:

- **Free tier** (cost = 0): auto-promoted via
  `_promote_model(model, role, by="discovery")`. Inserts a row in
  `control_plane.model_promotions`.
- **Paid tier**: writes a `governance_requests` row of type
  `model_promotion`. The dashboard's Governance page or the Signal
  `approve <id>` command consumes it. On approval,
  `consume_approved_promotions()` calls `llm_promotions.promote()` and
  triggers a catalog rehydrate.

`_promote_model` correctly merges roles via:

```sql
ARRAY(SELECT DISTINCT unnest(COALESCE(promoted_roles, '{}') || %s::text[]))
```

(Earlier bug: simple assignment clobbered the previous role list when
the same model was approved twice for different roles.)

### 10.3 Hand-pin overlay

`pin_role(role, mode, model)` writes priority=1000 to
`role_assignments`. Returns directly from the resolver's layer 1 —
short-circuiting score and promotion entirely.

`unpin_role(role, mode)` retires every priority≥1000 row for that
pair. Doesn't touch lower-priority auto-promotion artefacts.

The dashboard surfaces 📌 inline on each role card and exposes a
"Pin to role" dialog from each model card in the catalog grid.

### 10.4 Registry scanner (local-model proposals)

`app/llm_registry_scanner.py`. Closes the gap exposed by the
2026-04-25 qwen3.5 incident: the `scan_ollama()` half of discovery
only sees the **local** Ollama daemon's `/api/tags`, so a
strictly-better release like `qwen3.5:35b-a3b-q4_K_M` (3 weeks live,
fixes mem0 function-calling, vision + thinking modes) stayed
invisible for 21 days because nobody had pulled it yet.

The scanner crawls `ollama.com/library/<family>/tags` for an
allowlist of families (default: qwen3.5, qwen3, gemma3, llama3.1,
llama4, deepseek-r1, codestral; tunable via
`LLM_REGISTRY_FAMILIES`), parses the variant rows, applies a host-
capacity-aware size cap (auto-detected via `probe_host_capacity()` —
total RAM minus OS baseline minus Docker overhead, divided by
`OLLAMA_MAX_LOADED_MODELS` and the KV-cache factor), and emits
`local_model_pull` governance requests.

**Safety invariant: the scanner NEVER auto-pulls.** Pulls are 5–50 GB
on disk and hot-load into the same memory as every other Ollama
model — the `deepseek-r1:32b` SIGKILL spiral happened exactly because
that ceiling was ignored. Every candidate becomes a governance
request that the user approves via dashboard or Signal.

After `diff_against_local()` strips already-pulled tags, three
proposal filters chain in `_scan_registry_and_propose()`. They were
added 2026-04-30 in response to a 9-of-9 governance rejection storm
(three model variants × three idle-cycle re-proposals each). All
three parse a tag like `qwen3.5:35b-a3b-q4_K_M` into structural parts
via `_parse_model_id()` → `(family, size_b, quant, variants)` and
degrade gracefully when the parse can't identify the dimension they
care about (no false-positive blocks).

**Filter 1 — dominance by installed** (`filter_dominated_by_installed`):
skips a candidate that's already covered by something the user has
locally. Implements three layered rules — the second and third were
added on a follow-up pass after a different rejection wave (qwen3.5:latest
+ qwen3:14b + qwen3:8b proposed while qwen3.5:35b-a3b was already
installed) showed Rule 1 alone was too narrow.

  *Rule 1 — same family + strictly smaller explicit size*:
  ```text
  installed: qwen3.5:35b-a3b-q4_K_M (35B)
  proposed:  qwen3.5:4b-q8_0          (4B)  → SKIP — strict downgrade
  proposed:  qwen3.5:122b-a10b        (122B) → KEEP — larger, not dominated
  ```

  *Rule 2 — same family + sizeless candidate (`:latest`, `:instruct`)
  when the family already has an installed member*. The user already
  pinned a specific variant; the generic alias is almost always the
  smallest default and not what they want:
  ```text
  installed: qwen3.5:35b-a3b-q4_K_M
  proposed:  qwen3.5:latest           → SKIP — :latest of pinned family
  proposed:  qwen3.5:instruct         → SKIP — same reason
  proposed:  phi3:latest              → KEEP — phi family not installed
  ```

  *Rule 3 — cross-version-within-base dominance*. Family names parse
  via `_family_base_and_version()` into `(base, version)` so
  ``qwen3.5`` → `("qwen", 3.5)` and ``qwen3`` → `("qwen", 3.0)`. A
  candidate at a *lower* lineage version than something installed of
  the same base is treated as dominated:
  ```text
  installed: qwen3.5:35b-a3b-q4_K_M  (base=qwen, ver=3.5)
  proposed:  qwen3:14b-q4_K_M        (base=qwen, ver=3.0) → SKIP
  proposed:  qwen3:8b-q4_K_M         (base=qwen, ver=3.0) → SKIP
  proposed:  qwen4:14b-q4_K_M        (base=qwen, ver=4.0) → KEEP — newer
  proposed:  llama3.1:8b             (base=llama)         → KEEP — diff base
  ```
  Lineage detection covers `llama3.1`/`llama4`, `gemma3`/`gemma4`,
  `phi3`/`phi3.5`, `deepseek-r1` etc. Families without trailing version
  digits (`codestral`, `glm-ocr`) parse as `(family, None)` and skip
  Rule 3 entirely (no false-positive blocks across unrelated families).

When all three rules find nothing to compare against (different
family, candidate has all the data, no lineage relationship) the
candidate passes through.

**Filter 2 — quantization preference** (`filter_quant_dominated`):
skips a candidate that's a higher-quant (bigger / more precise)
version of an installed `(family, size, variants)` triple. Quant
ranking via `_QUANT_RANK` (`q4_K_M=4 < q5_K_M=5 < q8_0=7 < fp16=9`).

```text
installed: qwen3.5:35b-a3b-q4_K_M
proposed:  qwen3.5:35b-a3b-q8_0    → SKIP — q8 doubles disk for marginal quality
proposed:  qwen3.5:35b-a3b-fp16    → SKIP — even larger, same reason

installed: qwen3.5:35b-a3b-q8_0
proposed:  qwen3.5:35b-a3b-q4_K_M  → KEEP — leaner alternative
```

**Filter 3 — rejection learning** (`filter_recently_rejected`):
reads `control_plane.governance_requests` for `local_model_pull` rows
with `status='rejected'` in the last 30 days; skips matching
`model_id`s. DB failure returns an empty set so the scanner stays
loud-over-silent (proposes; doesn't suppress). Suppression window
tunable via the `_REJECTION_SUPPRESSION_DAYS` constant.

The three filters cap top-3 candidates per cycle (`new_candidates[:3]`)
and dedupe against existing pending governance requests
(`_existing_pull_proposal_models`) so an idle cycle that runs every
few minutes doesn't flood the queue.

A telemetry log line records when filtering reduces the candidate
set so operators can audit suppression decisions:

```
registry_scan: filtered 46 → 41 after dominance/quant/rejection checks
```

---

## 11. External rankings

`app/llm_external_ranks.py`. Three sources blended into a single
quality signal:

| Source | What it provides | Weight |
|---|---|---|
| **Artificial Analysis** | Intelligence index (~57-80 frontier), coding index, math index, gpqa, livecodebench, ifbench, throughput | dominant |
| **OpenRouter** | Cost (input/output per M), context size, supported parameters, throughput | cost ground truth |
| **HuggingFace leaderboard** | Open-model rankings | optional, dependency on `pandas` |

Blend formula in `_blend()`:

```
final_score = (1 − external_weight) · internal_score
            + external_weight       · external_score
```

`external_weight` defaults to `0.3` (settings.external_ranks_weight).

AA's intelligence index is rescaled by `/70` because the frontier
tops at ~57-80 (not 100). Models without an AA measurement get a
`0.85×` confidence penalty so the resolver doesn't over-weight a
guess.

---

## 12. Telemetry & feedback loops

`app/llm_benchmarks.py` records per-call metrics keyed by `(model,
task_type)`:

- latency (p50/p95)
- success rate (0/1 from `_benchmark_recorded` ContextVar)
- error type (timeout / 429 / 4xx / 5xx)
- token usage (in / out)
- cost ($, derived)

The selector reads `get_combined_scores(task_type)` on the hot path —
telemetry-driven scores override catalog strengths when present
(`0.60·bench + 0.35·strengths + 0.05·tool` in the resolver formula).

A `_record_token_usage` guard via `_benchmark_recorded` ContextVar
prevents double-recording when wrappers nest (e.g. cascade +
race-wrap + retry).

Re-benchmarking happens periodically via
`app/llm_discovery.py::TestIncumbentRotation` style logic — incumbents
are re-tested, new candidates are tested, scores update.

---

## 13. Span tracking (task-flow drawer)

`app/crews/span_events.py` subscribes to CrewAI's event bus
(`AgentExecutionStartedEvent`, `ToolUsageStartedEvent`,
`LLMCallStartedEvent`, plus their finish/error counterparts) and
persists every fine-grained event to
`control_plane.crew_task_spans`.

Correlation: a ContextVar `_current_crew_task_id` is set by
`crews/lifecycle.py` before `crew.kickoff()` runs; subscribers read it
on every event. CrewAI's own `event_id`/`parent_event_id` fields
reconstruct the agent → tool → llm-call hierarchy for free.

Per-row overhead: ~1 ms INSERT (start) + ~1 ms UPDATE (finish).
Typical crew run = ~45 events ≈ 90 ms across a 30-60 s run.

The dashboard's Tasks tab opens a drawer on row click with two views
(Tree / Timeline) that poll `/api/cp/tasks/{id}/timeline` at 2 s
while the task state is `running`, then stop. Retention: 7-day sweep
via the idle scheduler `spans-retention` job.

A 10-minute watchdog (`close_stale_spans`) marks any span stuck in
`running` longer than that as `failed` — covers the case where CrewAI
crashes mid-tool and never fires the matching `*_Finished` event.

---

## 14. Database schema

All tables in the `control_plane` schema. Migrations under
`migrations/`.

### 14.1 `role_assignments` (016, 019)

The hand-pin + auto-promotion overlay.

| Column | Type | Notes |
|---|---|---|
| role | TEXT NOT NULL | e.g. `commander`, `coding` |
| mode | TEXT NOT NULL | One of the 6 runtime modes (was `cost_mode` pre-019) |
| model | TEXT NOT NULL | Catalog key — must exist in live `CATALOG` |
| priority | INT NOT NULL DEFAULT 100 | ≥1000 = hand-pin |
| source | TEXT NOT NULL | `manual`/`auto_promotion`/`governance`/`rebenchmark` |
| reason | TEXT | Free-form |
| assigned_by | TEXT | `user`/`user:dashboard`/`system`/etc. |
| active | BOOLEAN NOT NULL DEFAULT TRUE | |
| created_at | TIMESTAMPTZ | |
| retired_at | TIMESTAMPTZ | |

Primary key: `(role, mode, model)`.

### 14.2 `model_promotions` (018)

Sticker-list of "first-choice" models. The resolver's layer-2 filter
reads this.

| Column | Type | Notes |
|---|---|---|
| model | TEXT PRIMARY KEY | Catalog key |
| promoted_by | TEXT | `discovery`/`governance`/`user:dashboard` |
| reason | TEXT | |
| promoted_roles | TEXT[] | Optional role-specific scope |
| created_at | TIMESTAMPTZ | |

### 14.3 `external_ranks` (017)

Cached per-model external metrics. Refreshed daily by
`llm-refresh-external-ranks` idle job.

| Column | Type |
|---|---|
| model | TEXT NOT NULL |
| source | TEXT NOT NULL (`aa`/`openrouter`/`hf`) |
| metric | TEXT NOT NULL (`intelligence`/`coding`/`cost_out`/etc.) |
| value | NUMERIC |
| recorded_at | TIMESTAMPTZ |

### 14.4 `discovered_models` (011)

Audit trail of discovery runs — what got considered, dominated, or
rejected.

### 14.5 `crew_task_spans` (022)

Fine-grained event log inside a crew run.

| Column | Type | Notes |
|---|---|---|
| id | BIGSERIAL PRIMARY KEY | |
| task_id | TEXT NOT NULL | FK → crew_tasks.id, ON DELETE CASCADE |
| parent_span_id | BIGINT | Self-FK for tree |
| span_type | TEXT NOT NULL | `agent`/`tool`/`llm_call` |
| name | TEXT NOT NULL | Role / tool name / model id |
| crewai_event_id | TEXT | For Started→Finished pairing |
| started_at, completed_at, state | … | |
| detail | JSONB DEFAULT `'{}'` | Tool args preview, token usage, etc. |
| error | TEXT | |

---

## 15. File inventory

| File | Role |
|---|---|
| `app/llm_catalog.py` | Catalog dict + `RUNTIME_MODES` + policy dicts + resolver |
| `app/llm_catalog_builder.py` | Auto-population from AA + OpenRouter + Ollama |
| `app/llm_mode.py` | Runtime-mutable mode singleton (`get_mode`/`set_mode`) |
| `app/llm_factory.py` | Single LLM gateway (`create_specialist_llm`, `create_vetting_llm`) |
| `app/llm_selector.py` | Selector with env overrides + difficulty + ContextVars |
| `app/llm_role_assignments.py` | DB overlay (hand-pins + auto-promotions) |
| `app/llm_promotions.py` | Promotion CRUD |
| `app/llm_discovery.py` | Pareto-dominance discovery + governance gating |
| `app/llm_registry_scanner.py` | ollama.com crawler + host-capacity probe + 3 proposal filters (dominance/quant/rejection) |
| `app/llm_external_ranks.py` | AA / OpenRouter / HF blending |
| `app/llm_benchmarks.py` | Per-call telemetry + score aggregation |
| `app/llm_rehydrate.py` | Rebuild CATALOG from snapshot at boot |
| `app/llm_sampling.py` | Phase-tuned sampling params (creative-mode) |
| `app/llm_context.py` | Token-budget management |
| `app/vetting.py` | 4-tier risk-based output verification |
| `app/crews/span_events.py` | CrewAI event-bus → crew_task_spans bridge |
| `app/control_plane/crew_task_spans.py` | Span persistence + retention |
| `app/control_plane/dashboard_api.py` | `/api/cp/llms/*` + `/api/cp/tasks/{id}/timeline` |
| `migrations/011_llm_discovery_schema.sql` | discovered_models |
| `migrations/016_llm_role_assignments.sql` | overlay table (col was `cost_mode`) |
| `migrations/017_llm_external_ranks.sql` | rank cache |
| `migrations/018_model_promotions.sql` | promotion list |
| `migrations/019_unified_runtime_mode.sql` | rename cost_mode → mode |
| `migrations/022_crew_task_spans.sql` | task-flow spans |

---

## 16. Operations

### 16.1 Switching mode

```bash
# Dashboard: LLMs tab → Runtime Mode card → click any of the 6 buttons.

# Signal:
mode quality

# API (requires GATEWAY_SECRET):
curl -X POST -H "Authorization: Bearer $GATEWAY_SECRET" \
     -H "Content-Type: application/json" \
     http://localhost:8765/config/llm_mode \
     -d '{"mode":"insane"}'
```

### 16.2 Pinning a model

```bash
# Dashboard: LLMs tab → click "📌 pin to role" on any model card.

# Signal:
pin commander balanced claude-opus-4.7

# API:
curl -X POST -H "Authorization: Bearer $GATEWAY_SECRET" \
     -H "Content-Type: application/json" \
     http://localhost:8765/api/cp/llms/pin \
     -d '{"role":"commander","mode":"balanced","model":"claude-opus-4.7","reason":"prefer Opus reasoning"}'
```

### 16.3 Refreshing the catalog

```bash
# Idle scheduler runs llm-refresh-catalog every 24 h. Manual trigger:
# Signal:
refresh catalog

# API:
curl -X POST -H "Authorization: Bearer $GATEWAY_SECRET" \
     http://localhost:8765/api/cp/llms/discovery/run
```

### 16.4 Inspecting current resolver state

```bash
# Signal:
status

# API (returns 367 models + role assignments + active mode):
curl -H "Authorization: Bearer $GATEWAY_SECRET" \
     http://localhost:8765/api/cp/llms/catalog
```

### 16.5 Watching a task flow live

Dashboard → Tasks tab → click any row. Drawer opens; toggle 🌳 Tree
/ ⏱️ Timeline. Polls every 2 s while the task is running.

---

## 17. Failure modes & recovery

| Failure | What happens | Why it's safe |
|---|---|---|
| All three fetchers (AA, OpenRouter, Ollama) down | Builder skips refresh, snapshot stays stale | 24-h TTL gives long grace; live `CATALOG` keeps last good state |
| Snapshot file missing or corrupt | Falls back to `_BOOTSTRAP_CATALOG` (3 entries) | System still boots, commander/vetting/critic resolve to Sonnet |
| Postgres unreachable | `role_assignments` queries return `None`; resolver layer 1 silently skips | Layer 3 pool scoring still works |
| CrewAI version doesn't expose event types | `span_events.install_listeners()` logs warning, sets installed flag, returns | Crews still run; just no spans get persisted |
| Hand-pin points at retired model | `set_assignment()` rejects writes that aren't in live `CATALOG`; old pins surface as stale | Resolver layer 1 verifies `pin in CATALOG` before returning |
| Vetting hangs | Bounded `llm.call()` timeout (~30 s) | Parent task's soft-timeout still fires |
| Span never finishes (CrewAI bus crash) | 10-min watchdog `close_stale_spans` marks them failed | Dashboard doesn't show eternally-running spans |
| Discovery promotes the wrong model | Demote button on dashboard or `demote <model>` Signal | Catalog rehydrates immediately |
| ollama.com unreachable or HTML format changes | `parse_tags_page()` returns `[]`, no proposals emitted | Degrades silently — local discovery still runs; never raises |
| `governance_requests` table unreachable for rejection lookup | `get_recently_rejected_models()` returns `set()` | Loud-over-silent: scanner still proposes, user re-rejects if needed |
| Scanner repeatedly proposes a smaller-family sibling, `:latest` alias of a pinned family, or older-lineage variant | `filter_dominated_by_installed` (3 rules: size dominance, sizeless-alias, cross-version-base) blocks at source | 2026-04-30 fixes; removes the original 9-of-9 storm AND the follow-up `qwen3.5:latest` / `qwen3:8b` / `qwen3:14b` slip-through |
| Host-capacity probe fails (no env, no Docker, no sysctl) | `_DEFAULT_MAX_SIZE_GB_FALLBACK = 16 GB` cap | Strictly conservative — better to under-propose than blow memory |

---

## 18. Test surfaces

| Test file | Covers |
|---|---|
| `tests/test_llm_catalog.py` | Catalog structure, resolver, role policy, planner/introspector wiring, unified-mode vocabulary |
| `tests/test_llm_catalog_builder.py` | Snapshot building, strength derivation, fetcher fallbacks |
| `tests/test_llm_selector_routing.py` | difficulty_to_tier, detect_task_type |
| `tests/test_llm_role_assignments.py` | Pin/unpin, set_assignment, mode aliasing |
| `tests/test_llm_promotions_and_pins.py` | Promotion CRUD, hand-pin layered priority |
| `tests/test_llm_external_ranks.py` | AA / OR / HF fetchers, blend weighting |
| `tests/test_llm_rebenchmark.py` | Incumbent rotation, judge selection |
| `tests/test_llm_discovery.py` | Pareto-dominance, governance gating |
| `tests/test_llm_registry_scanner.py` | Tags-page parser, host-capacity probe, dominance / quant / rejection-learning filters |
| `tests/test_llm_telemetry.py` | Per-call recording, ContextVar hygiene |
| `tests/test_vetting_feedback.py` | Vetting failure → tier bump |
| `tests/test_crew_task_spans.py` | Span persistence, ContextVar correlation, event-map roundtrip |

Run the whole LLM suite with:

```bash
pytest tests/test_llm_*.py tests/test_vetting_feedback.py tests/test_crew_task_spans.py -v
```

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **Catalog** | The live `dict[str, dict]` of all known models. Mutated in place by the builder. |
| **Bootstrap** | The 3-entry survival catalog. Always present even with zero connectivity. |
| **Resolver** | `resolve_role_default(role, mode)` — the score-based pick function. |
| **Selector** | `select_model(role, task_hint)` — adds env overrides, difficulty, ContextVars on top of the resolver. |
| **Factory** | `create_specialist_llm(...)` — single instantiation gateway. |
| **Hand-pin** | A `priority≥1000` row in `role_assignments`. Returned directly from the resolver, no scoring. |
| **Promotion** | A row in `model_promotions`. Filters the candidate set down to "first-choice" models when any survive. |
| **Mode** | The unified runtime axis: `free`/`budget`/`balanced`/`quality`/`insane`/`anthropic`. |
| **Tier** | `local`/`free`/`budget`/`mid`/`premium`. Mode controls which tiers are allowed. |
| **Strengths map** | Per-model dict from the 9 canonical task types to a 0-1 score. Derived from AA evaluations. |
| **External rank** | A third-party metric (AA intelligence, OpenRouter cost, etc.) blended into the resolver's quality signal. |
| **Vetting** | `app/vetting.py` — 4-tier risk-based output check. Vetting model itself is selected via the resolver with `role="vetting"`. |
| **Span** | A `crew_task_spans` row. One per agent-execution / tool-call / llm-call inside a crew run. |
| **Span watchdog** | `close_stale_spans` — marks spans stuck in `running` past 10 minutes as `failed`. |
| **Incumbent** | The current top-scoring model for a role + mode pair. Discovery proposes replacements. |
| **Pareto dominance** | Quality ≥ AND cost ≤ (with at least one strict inequality, mode-weighted). |
| **Registry scanner** | `app/llm_registry_scanner.py` — crawls ollama.com for new local-model variants, applies host-capacity sizing + three proposal filters, emits governance requests. Never auto-pulls. |
| **Dominance by installed** | An installed model "dominates" a candidate when any of three rules applies: same family + strictly smaller size (qwen3.5:4b vs qwen3.5:35b), sizeless candidate (`:latest`) of an already-pinned family (qwen3.5:latest when qwen3.5:35b is in), or candidate's lineage version is lower than installed under the same base (qwen3:14b dominated by qwen3.5:35b). Dominated proposals are skipped — installing a smaller / sizeless / older-lineage variant when a better one is already in is almost always a mistake. |
| **Lineage base / version** | Result of `_family_base_and_version()`: a family name like `qwen3.5` is split into base (`qwen`) and version (`3.5`). Two families share a lineage iff their bases match; the higher version dominates the lower for governance-proposal purposes. Families without trailing digits (`codestral`, `glm-ocr`) parse as `(family, None)` and don't participate in cross-version dominance — their candidates pass through Rule 3 unchanged. |
| **Quant rank** | Numeric ordering over quantizations (q4_K_M=4, q5_K_M=5, q8_0=7, fp16=9). Used by `filter_quant_dominated` to skip "bigger for marginal gain" variants of an installed base. |
| **Rejection learning** | `filter_recently_rejected` reads `governance_requests` for `local_model_pull` rejections in the last 30 days and suppresses re-proposing the same `model_id`. Stops the idle-cycle nag loop. |

---

## 20. Cross-references

- High-level system architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Control-plane patterns: [`CONTROL_PLANES.md`](CONTROL_PLANES.md)
- Self-improvement pipeline (uses LLM-as-Judge for evolution variants): [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md)
- Dashboard surfaces (LLMs tab + Tasks tab + drawer): React app at `dashboard-react/src/components/{LlmsPage,TasksPage,TaskFlowDrawer}.tsx`
