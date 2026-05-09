# Workspace Companion — Per-Workspace Idle Contemplation

> A cooperative background system that runs during idle windows, contemplates
> a workspace's seed prompt, surfaces unique ideas to the user, accepts
> thumbs-up/down feedback, persists polished ideas as documents + wiki pages,
> and lets abstract structural insights flow between workspaces under two
> independent safety gates.
>
> **Companion stance.** Sits one layer above [SubIA](SUBIA.md) and the
> [Affect Layer](AFFECT_LAYER.md). Reads from both: the affect bridge gates
> cycles in adverse states; SubIA's own primitive [Reverie engine](../app/subia/reverie/)
> is available as one ideation tool among many (Phase 14+ wire). Reuses
> the [Creative MAS](../app/crews/creative_crew.py) pipeline as the
> Diverge/Discuss/Converge engine and the [Transfer Insight](../app/transfer_memory/)
> sanitiser for cross-workspace gating. No parallel structures where a seam
> already existed.

**Status.** Live on `main`. 14 commits across phases 0–13 + production
follow-ups. Module surface in `app/companion/`; FastAPI routes in
`app/api/companion_api.py`; React UI as a sub-tab on `/cp/ops`. **313
backend tests** in `tests/test_companion_*.py`. Verified end-to-end on the
preview server.

---

## Table of contents

1. [TL;DR](#tldr)
2. [Why a separate layer](#why-a-separate-layer)
3. [Reuse map — no parallel structures](#reuse-map)
4. [Architecture](#architecture)
5. [Idea lifecycle](#idea-lifecycle)
6. [The cognitive cycle](#the-cognitive-cycle)
7. [Fairness scheduler](#fairness-scheduler)
8. [Cross-layer memory registration](#cross-layer-memory-registration)
9. [Cross-workspace transfer (hybrid model)](#cross-workspace-transfer)
10. [Configuration](#configuration)
11. [Persistence layout](#persistence-layout)
12. [API surface](#api-surface)
13. [React surface](#react-surface)
14. [Idle scheduler integration](#idle-scheduler-integration)
15. [Operational guide](#operational-guide)
16. [Safety invariants](#safety-invariants)
17. [Failure modes & mitigations](#failure-modes--mitigations)
18. [Research grounding](#research-grounding)
19. [Phase history](#phase-history)
20. [File inventory](#file-inventory)
21. [Open follow-ups](#open-follow-ups)

---

## TL;DR

The Workspace Companion runs as four cooperative idle-time jobs registered
through [`idle_scheduler._default_jobs()`](../app/idle_scheduler.py).
**Cold-start is automatic**: the first cycle on a workspace with no
`seed_prompt` reads `control_plane.projects.mission` + recent tickets the
operator filed against the project, synthesises a candidate seed via one
cheap-tier LLM call, persists it, and continues. The user retains
override via the React Settings tab.

| Job | Weight | Cadence | Purpose |
|---|---|---|---|
| `companion-tick` | MEDIUM | every idle window (CFS-fair) | One ideation cycle for one eligible workspace |
| `companion-ingest` | LIGHT | once per ≤12 h per source | Fetch fresh material from accepted external sources |
| `companion-grand-task` | MEDIUM | once per 12 h per workspace | Synthesise a higher-order grand task from polished ideas |
| `companion-xworkspace` | LIGHT | daily | Propose `GLOBAL_META` kernels to relevant peer workspaces |

Each cycle: pick a workspace via [CFS-fairness](#fairness-scheduler) → compose
context (temporal grounding + KB v2 + companion sources + lessons from past
feedback + diversity hints from MAP-Elites void cells) → run [Creative MAS](../app/crews/creative_crew.py)
3-phase pipeline → score the converged output (novelty + quality +
transferability + 5-persona panel) → persist the lineage → maybe surface to
Signal + React. On thumbs-up + promote, the idea lands as a markdown +
docx + pdf, registered in **four memory layers at once**:

1. Workspace wiki (`workspace/companion/wiki/<ws>/<id>-<slug>.md`)
2. Mem0 (workspace-scoped agent_id, content-searchable)
3. System wiki (`wiki/meta/companion/<id>.md`)
4. ChromaDB `companion_ideas` collection (already done at idea creation)

Cross-workspace transfer is **hybrid**: workspaces stay focused on their
own topics, but `TransferScope.GLOBAL_META` kernels (after the existing
[transfer_memory sanitiser](../app/transfer_memory/sanitizer.py)) propose to
relevant peers based on seed-prompt embedding similarity. Two independent
gates protect focus discipline. Always human-in-loop.

---

## Why a separate layer

Three distinct gaps the existing system didn't fill:

1. **Background ideation per workspace.** SubIA's own primitive
   [Reverie engine](../app/subia/reverie/) does free-association
   mind-wandering globally; it has no workspace concept, no seed-prompt
   anchor, no surfacing pipeline, no user feedback loop, no document
   maturation. Companion is the higher-level orchestration that uses
   such engines as primitives.

2. **User-driven creative co-worker.** Creative MAS runs on user request.
   Companion runs *during user idle time*, contemplates whatever workspace
   the user has been engaged with, and presents only ideas that clear a
   high bar. The user thumbs-up/down to shape what comes next.

3. **Cross-workspace pollination with focus discipline.** The user's spec
   answer #1 ("Estonian forests workspace will not contemplate how to sell
   e-commerce SaaS in SEA region") demands per-workspace topical gravity —
   but the user also wanted abstract/structural insights to flow across
   workspaces. Two-gate model (sanitiser + relevance) achieves both.

Companion does **not** replace any of these concerns; it composes them.

---

## Reuse map

Hard rule per the user's spec: *no parallel structures where a seam exists.*

| Need | Reused | How |
|---|---|---|
| Workspace identity | `app.consciousness.personality_workspace` (CP project ID) | seed_prompt + companion config stored in `control_plane.projects.config_json.companion`; no DB migration |
| Idle execution | [`app.idle_scheduler`](../app/idle_scheduler.py) | Self-registers four jobs through `loop.get_idle_jobs()` returning `(name, fn, JobWeight)` tuples |
| Ideation pipeline | [`app.crews.creative_crew.run_creative_crew`](../app/crews/creative_crew.py) | Cycle composes prompt + calls; cost reported back via `CreativeRunResult.cost_usd` |
| Context retrieval | [`app.retrieval.orchestrator.RetrievalOrchestrator`](../app/retrieval/orchestrator.py) | KB v2 episteme + experiential collections every cycle |
| Quality-Diversity | [`app.map_elites_wiring.record_crew_outcome`](../app/map_elites_wiring.py) | Workspace-scoped role `companion:<ws>`; void cells → diversity hints |
| Curiosity / drive signal | [`app.epistemic.affect_bridge.live_factual_grounding`](../app/epistemic/affect_bridge.py) | Below 0.3 → all cycles paused; otherwise grounding ∈ [0,1] → CFS weight ∈ [0.5, 2.0] |
| Cross-domain insights | [`app.transfer_memory.sanitizer.check`](../app/transfer_memory/sanitizer.py) | `GLOBAL_META`-only propagation to other workspaces; fails closed if sanitiser unimportable |
| Persistent memory | Mem0 / ChromaDB | `companion_ideas` collection (workspace_id metadata); Mem0 facts via `mem0_manager.store_memory(text, agent_id="workspace:<ws>", metadata)` |
| Cost-aware inference | [`app.llm_factory.create_specialist_llm`](../app/llm_factory.py) | All scoring + critique + suggester calls |
| Web search | [`app.tools.web_search.search_brave`](../app/tools/web_search.py) | Existing 3-tier cascade (Brave → SearXNG → DDG) for source ingestion |
| Outbound Signal | `app.conversation_store.enqueue_outbound` | Real wire-up in `surfacing._send_signal` |
| UI shell | `OpsPage.tsx` tab pattern | `CompanionTab.tsx` mounted as 6th sub-tab on `/cp/ops` |
| Per-document formats | `pandoc` subprocess | Optional docx/pdf rendering — silently skipped when pandoc absent |

Net-new modules: `app/companion/` (12 files, one per concern), `app/api/companion_api.py`,
`tests/test_companion_*.py` (23 files), `dashboard-react/src/components/CompanionTab.tsx`,
`dashboard-react/src/api/companion.ts`.

---

## Architecture

```
                    +--- /cp/ops > Workspace Companion (React tab) ---+
                    |  Live | Ideas | Documents | Wiki | Sources       |
                    |  Grand task | Inbox | Settings                   |
                    +---^--------------------------------------^------+
                        | poll                                  | mutate
+-- /api/cp/companion/* (18 endpoints in app/api/companion_api.py) ----+
                        |
+-- app/companion/ (the layer) ---------------------------------------+
|                                                                     |
|   loop.py            — registers 4 idle jobs in idle_scheduler      |
|     |                                                                |
|     v                                                                |
|   scheduler.py       — CFS-fairness + 12 h temporal floor            |
|     |  reads: affect_bridge.live_factual_grounding (Phase 7+)        |
|     v                                                                |
|   cycle.py           — one ideation pass:                            |
|     1. workspace_kb.compose() — temporal + KB v2 + companion_sources |
|     2. reflexion.build_block() — feedback into next prompt           |
|     3. diversity.sparse_cell_hints() — MAP-Elites void cells         |
|     4. _invoke_creative_crew(prompt) → CreativeRunResult             |
|     5. scoring.compute_{novelty,quality,transferability}             |
|     6. critique.run_panel() — 5 personas, aggregate score            |
|     7. _persist_lineage() — fragments + developed + converged        |
|     8. diversity.record_cycle() — workspace-scoped MAP-Elites grid   |
|     9. surfacing.surface() — if all gates pass                       |
|                                                                      |
|   idea_store.py       — IdeaRecord, JSONL + ChromaDB index           |
|   events.py           — event log: SURFACED/FEEDBACK/DOCUMENTED/...  |
|   state.py            — runtime sidecar: vruntime, last_tick_at, $   |
|   budget.py           — daily $ ledger (UTC-day reset)               |
|   surfacing.py        — should_surface + send_signal + card render   |
|   feedback.py         — record thumbs (Polarity.UP/DOWN), summary    |
|   reflexion.py        — Shinn 2023 verbal-RL prompt block            |
|   document_pipeline.py — promote → md/docx/pdf + wiki publish        |
|   wiki.py             — workspace + system wiki + Mem0 register      |
|   sources.py          — source CRUD per workspace                    |
|   source_suggester.py — LLM-driven proposals from seed prompt        |
|   ingest.py           — daily fetch + ChromaDB index                 |
|   grand_task.py       — 12 h synthesis from polished ideas           |
|   diversity.py        — MAP-Elites per-workspace + void hints        |
|   cross_workspace.py  — sanitiser + relevance gates → inbox          |
|   critique.py         — 5-persona panel (Engineer/DomainExpert/      |
|                          Skeptic/Synthesizer/UserAdvocate)           |
|   config.py           — CompanionConfig + load/save into CP          |
|                                                                      |
+----------------------------------------------------------------------+

External components consumed (read-only / fire-and-forget):
  app.crews.creative_crew              app.retrieval.orchestrator
  app.epistemic.affect_bridge          app.map_elites_wiring
  app.transfer_memory.sanitizer        app.memory.mem0_manager
  app.memory.chromadb_manager          app.tools.web_search
  app.temporal_context                 app.conversation_store
  app.llm_factory                      app.control_plane.projects
```

---

## Idea lifecycle

```
   Creative MAS Initiation                Phase 1 outputs
            │                              ─────────────────
            ▼                              FRAGMENT × N
   Creative MAS Discussion                  (no parents)
            │                                    │
            ▼                                    ▼
   Creative MAS Convergence ───────────► DEVELOPED × M
            │                              (parents = all
            ▼                               fragments)
   final_output text                            │
            │                                    │
            ▼                                    ▼
       score                                CONVERGED × 1
       persist                              (parents = all
       maybe surface                         developed)
            │
            │   surface() emits SURFACED event
            ▼
       SURFACED ──── feedback DOWN ────► ARCHIVED
            │
            │  promote() emits DOCUMENTED event
            ▼  + writes md/docx/pdf
       DOCUMENTED ────► wiki.publish_to_wiki()
                            │
                            ├── workspace wiki page
                            ├── system wiki mirror
                            ├── Mem0 fact (agent_id=workspace:<ws>)
                            └── (ChromaDB already indexed at creation)
```

Effective state is computed by [`idea_store.current_state`](../app/companion/idea_store.py)
folding the event log forward over the immutable creation record. The
`ideas.jsonl` file is append-only — every event lands in
`events/<ws>.jsonl` instead.

---

## The cognitive cycle

One pass, 60–180 s wall clock (capped by `idle_scheduler` MEDIUM weight):

1. **Idle gate.** `idle_scheduler.should_yield()` returns `False`; affect-bridge
   `live_factual_grounding()` ≥ 0.3.
2. **Workspace selection.** [CFS scheduler](#fairness-scheduler) picks the
   workspace with lowest `vruntime / weight`, with a 12 h temporal-floor
   override so no active workspace is starved.
3. **Quiet-hours / budget gate.** Skip if local hour is in
   `quiet_hours_{start,end}` or daily budget exhausted.
4. **Context composition.** [`workspace_kb.compose`](../app/companion/workspace_kb.py)
   pulls:
   - [`temporal_context.format_temporal_block`](../app/temporal_context.py) for
     seasonal/lunar/daylight grounding.
   - KB v2 episteme + experiential snippets via
     [`RetrievalOrchestrator.retrieve`](../app/retrieval/orchestrator.py).
   - Workspace-scoped sources from ChromaDB `companion_sources`.
5. **Reflexion.** [`reflexion.build_block`](../app/companion/reflexion.py)
   adds "directions that DID/did NOT resonate" from past thumbs-up/down ideas.
6. **Diversity hints.** [`diversity.sparse_cell_hints`](../app/companion/diversity.py)
   reads MAP-Elites void cells (workspace-scoped role `companion:<ws>`)
   and adds "explore a direction with complexity=…, cost_efficiency=…,
   specialization=…" pointers.
7. **Generate.** [`_invoke_creative_crew(prompt)`](../app/crews/creative_crew.py)
   runs the 3-phase pipeline (Initiation / Discussion × N / Convergence).
8. **Score.** Against PRIOR workspace history only:
   - `novelty` = 1 − max-cosine to past `companion_ideas`.
   - `quality` = LLM-as-judge rubric (cheap tier, 0–10 → [0,1]).
   - `transferability` = distinct rubric (abstract / structural vs
     workspace-specific).
   - `panel_score` = mean of 5 persona scores (Engineer / DomainExpert /
     Skeptic / Synthesizer / UserAdvocate), each 1–5, normalised to [0,1].
9. **Persist.** Fragments + developed + converged with lineage parents.
   ChromaDB upsert + JSONL append. Idempotent via deterministic ids.
10. **Diversity record.** [`diversity.record_cycle`](../app/companion/diversity.py)
    writes the converged fitness onto the workspace's MAP-Elites grid
    so step 6 of the next cycle can sample its voids.
11. **Surfacing decision.** Eligible iff:
    - `novelty ≥ config.novelty_threshold` (default 0.7)
    - `quality ≥ config.surface_threshold` (default 0.7)
    - `panel_score ≥ config.panel_threshold` (default 0.6)
    - workspace not surfaced in the last `SURFACE_COOLDOWN_HOURS` (4 h)
12. **Surface.** [`surfacing.surface`](../app/companion/surfacing.py) renders
    the card, calls `_send_signal` (→ `conversation_store.enqueue_outbound`
    when recipient configured), appends `SURFACED` event regardless of
    delivery success.
13. **Tick recorded.** [`scheduler.record_tick`](../app/companion/scheduler.py)
    advances `vruntime`, charges `budget`, clears any prior skip reason.

Failures of any individual step are absorbed: the cycle reports the failure
in `CycleResult.aborted_reason` or `surface_reason`; the scheduler still
records the tick.

---

## Cold-start seed bootstrap (Phase 11.5)

Before a workspace has any polished Companion ideas, Phase 11 grand-task
synthesis can't fire (its activation gate requires ≥3 polished ideas).
Without that, a workspace whose `seed_prompt` is None would produce
nothing forever — chicken-and-egg.

[`seed_bootstrap.derive_seed`](../app/companion/seed_bootstrap.py) closes
the loop by reading the human signal that's already there:

1. `control_plane.projects.mission` (always populated for real projects)
2. Recent tickets for the project (`control_plane.tickets.title`,
   newest 15)

If either yields material, one cheap-tier LLM call synthesises a seed.
[`cycle._maybe_bootstrap_seed`](../app/companion/cycle.py) persists it
via [Phase 6.5 `config.save`](../app/companion/config.py), emits a
`SEED_DERIVED` event with `source_signal ∈ {mission+tickets, tickets_only,
mission_only}`, then continues the same cycle with the new seed (no
extra tick burnt).

Activation gates (infrastructure-bounded):
- `default` workspace is blocklisted (catch-all, mixed signal)
- Mission below `MIN_MISSION_CHARS = 10` doesn't count as signal
- Need at least one ticket OR a real mission

Failure modes — all absorbed:
- LLM unavailable / parse failure → `derive_seed` returns None →
  cycle aborts with `no_seed_prompt` (same as before Phase 11.5)
- `config.save` fails → bootstrap aborts the cycle without continuing
  (no Creative MAS spend on a seed that didn't persist)
- DB unreachable → `_recent_tickets` returns `[]`, `_get_project`
  returns None — both seams swallow exceptions internally

The user retains full override at the React Settings tab; on edit, the
manually-supplied seed wins (`seed_prompt` is set unconditionally on
POST /config). Phase 11 grand-task synthesis takes over for ongoing
refinement once polished ideas accumulate.

A future Phase 11.5b will add a third source — recent conversation turns
routed to the workspace — once `conversation_store.add_message` captures
`project_id`.

---

## Fairness scheduler

[`scheduler.select_next`](../app/companion/scheduler.py) implements a CFS-style
selector:

```python
def select_next(now=None, now_local_hour=None):
    cands = [w for w in active_workspaces
             if not in_quiet_hours(w, now_local_hour)
             and not budget_exhausted(w)
             and affect_grounding >= AFFECT_GROUNDING_FLOOR]

    # Hard floor: any workspace untouched for 12 h jumps the queue.
    starved = [c for c in cands
               if c.last_tick_at == 0
               or (now - c.last_tick_at) > TEMPORAL_FLOOR_S]
    if starved:
        return min(starved, key=lambda c: c.last_tick_at)

    # Otherwise: lowest effective vruntime wins.
    return min(cands, key=lambda c: c.vruntime / max(c.weight, WEIGHT_FLOOR))
```

Properties:
- **Affect-aware weight.** `weight = clamp(0.5 + grounding, 0.5, 2.0)`. High
  grounding → workspace gets more cycles per unit `vruntime`.
- **Adverse-state pause.** `live_factual_grounding < 0.3` skips ALL cycles —
  the system is signalling distress / frozen / depleted.
- **No starvation.** 12 h temporal floor guarantees every active workspace
  gets at least two cycles per day.
- **Quiet hours.** Per-workspace `quiet_hours_{start,end}` (default 02:00 →
  06:00 local Helsinki); wraps midnight if `start > end`.
- **Budget cap.** Daily UTC ledger; default $1/workspace/day, configurable
  via React Settings tab or `POST /api/cp/companion/config/{ws}`.

---

## Cross-layer memory registration

The user's spec answer #6 explicitly demanded that **big ideas and
contemplations land in workspace memory wiki AND in other memory layers**.
[`wiki.publish_to_wiki`](../app/companion/wiki.py) writes to four layers in
one call (ChromaDB happens earlier at idea creation, so it's three
write-paths from this function):

| Layer | Path | Owner | Why |
|---|---|---|---|
| Workspace wiki | `workspace/companion/wiki/<ws>/<id>-<slug>.md` | Companion | Per-workspace, lineage cross-links via `[[backlinks]]` + `./<parent>.md`, `_index.md` regenerated on every publish |
| Mem0 | `agent_id="workspace:<ws>"` | [Mem0 stack](MEMORY_ARCHITECTURE.md) | Cross-session content-search via `search_agent`; metadata tagged `kind="companion_polished_idea"` |
| System wiki | `wiki/meta/companion/<id>.md` | Existing wiki-synthesis | Surfaces beside human-authored knowledge; tagged `epistemic_status: companion-polished` |
| ChromaDB | `companion_ideas` collection | Phase 3 idea creation | Embedding novelty search, workspace-scoped via `where: {workspace_id}` |

Failures of any single layer are absorbed and listed in `WikiResult.errors`
— the others still proceed. A `WIKI_REGISTERED` event lands on the
workspace event log so `idea_store.current_state` reflects the cross-layer
commit.

See also: [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) for the broader
context on how Mem0, ChromaDB, Neo4j, and pgvector compose.

---

## Cross-workspace transfer

Implements the user's spec answer #1 hybrid model. **Estonian forests
workspace contemplates Estonian forests**, but a structural insight from
KaiCart (e.g., "delayed-reward feedback loops self-stabilise") can propose
to Estonian forests *if* it passes both gates.

Two-gate pipeline in [`cross_workspace.propagate_eligible`](../app/companion/cross_workspace.py):

```
   converged idea in workspace A
            │
            │  transferability ≥ config.transferability_threshold (0.7)
            ▼
   ┌──────────────────────────────────────────┐
   │ Gate 1: TransferScope sanitiser          │
   │   re-uses transfer_memory.sanitizer.check│
   │   FAILS CLOSED on import error           │
   └──────────────────┬───────────────────────┘
                      │ allowed_scope == GLOBAL_META
                      ▼
   ┌──────────────────────────────────────────┐
   │ Gate 2: Relevance to TARGET workspace    │
   │   cosine(kernel_emb, target.seed_emb)    │
   │   ≥ CROSS_WORKSPACE_RELEVANCE_THRESHOLD  │
   │   (0.75)                                  │
   └──────────────────┬───────────────────────┘
                      │
                      ▼  CROSS_WORKSPACE_INBOX event on target's log
              user accepts ─────► kernel rides into next N cycles
                      │
                      └─ user dismisses ─► CROSS_WORKSPACE_DISMISSED
                                             (won't be re-proposed)
```

Hard properties:
- **Sanitiser fails closed.** If `app.transfer_memory.sanitizer` is
  unimportable, `_passes_sanitiser` returns `False` — no kernel propagates.
  Cross-workspace risk is the wrong thing to risk on a degraded import.
- **Per-source-idea cooldown.** `_already_proposed` checks the target
  workspace's event log; the same kernel won't be re-proposed on every
  daily run.
- **Capped per run.** `MAX_PROPAGATIONS_PER_RUN = 5` per source workspace
  per idle invocation.
- **Always human-in-loop.** Kernels never auto-inject. They wait in the
  React Inbox tab; user accepts or dismisses.

See also: [Transfer Insight Layer](../app/transfer_memory/) for the
sanitiser's three-tier promotion ladder.

---

## Configuration

[`CompanionConfig`](../app/companion/config.py) is stored in
`control_plane.projects.config_json.companion`. All bounds are
infrastructure-level — `clamp()` re-applies them on save, so the API
cannot leak out-of-range values into storage. Per CLAUDE.md's
self-improvement invariant, the Self-Improver agent cannot widen them.

| Field | Default | Bounds | Purpose |
|---|---|---|---|
| `enabled` | `True` | — | Master switch; off pauses all 4 idle jobs for this workspace |
| `seed_prompt` | `None` | — | The user-supplied workspace seed; auto-rotated on grand-task accept |
| `daily_budget_usd` | `1.0` | [0, 100] | Per-workspace UTC-day spend ceiling |
| `surface_threshold` | `0.7` | [0.5, 0.95] | Quality gate for Signal/React surface |
| `novelty_threshold` | `0.7` | [0.3, 0.95] | Embedding-novelty gate (cosine vs prior history) |
| `transferability_threshold` | `0.7` | [0.5, 0.95] | Cross-workspace propagation eligibility |
| `panel_threshold` | `0.6` | [0.4, 0.9] | Aggregate of 5-persona critic panel |
| `quiet_hours_start` | `2` | [0, 23] | Local Helsinki hour; wraps midnight if `> end` |
| `quiet_hours_end` | `6` | [0, 23] | Inclusive of start, exclusive of end |
| `signal_recipient` | `None` | — | Per-workspace Signal recipient; falls back to env var |
| `sources` | `[]` | — | Tracked for future expansion; ignore (sources live in `app/companion/sources.py` sidecar) |

Environment variables:

| Variable | Default | Effect |
|---|---|---|
| `COMPANION_SIGNAL_RECIPIENT` | unset | Operator-default Signal recipient when no per-workspace `signal_recipient` set |
| `COMPANION_STATE_DIR` | `workspace/companion/state` | Override runtime state sidecar location |
| `COMPANION_IDEAS_DIR` | `workspace/companion/ideas` | Override JSONL idea store location |
| `COMPANION_EVENTS_DIR` | `workspace/companion/events` | Override event log location |
| `COMPANION_SOURCES_DIR` | `workspace/companion/sources` | Override sources sidecar location |
| `COMPANION_DOCUMENTS_DIR` | `workspace/companion/documents` | Override document maturation output location |
| `COMPANION_WIKI_DIR` | `workspace/companion/wiki` | Override per-workspace wiki output |
| `COMPANION_SYSTEM_WIKI_DIR` | `wiki` | System wiki root for `meta/companion/` mirror |

---

## Persistence layout

```
workspace/companion/
├── state/<workspace_id>.json          ← vruntime, last_tick_at, daily_cost_usd
├── ideas/<workspace_id>.jsonl         ← append-only idea creation log
├── events/<workspace_id>.jsonl        ← append-only event log
├── sources/<workspace_id>.json        ← workspace's external sources (CRUD)
├── documents/<workspace_id>/
│   └── <idea_id>.{md,docx,pdf}        ← document maturation output
└── wiki/<workspace_id>/
    ├── _index.md                       ← regenerated on every publish
    └── <idea_id>-<slug>.md             ← per-idea wiki page

wiki/meta/companion/
└── <idea_id>.md                       ← system-wiki mirror

ChromaDB collections:
  companion_ideas       ← embedding novelty search (workspace-scoped via metadata)
  companion_sources     ← ingested external material (workspace-scoped)

Mem0:
  agent_id="workspace:<ws>"  ← polished-idea facts, content-searchable

Per-workspace MAP-Elites grid (via app.map_elites):
  role="companion:<workspace_id>"  ← strategy diversity archive

CP project (PostgreSQL):
  control_plane.projects.config_json.companion  ← CompanionConfig
```

---

## API surface

All endpoints under `/api/cp/companion/*`, registered through
[`app.api.companion_api.router`](../app/api/companion_api.py). Mounted in
`main.py` next to `workspace_api`. **18 routes:**

### Phase 4 — feedback + ideas
| Method | Path | Purpose |
|---|---|---|
| POST | `/feedback` | Record thumbs up/down + comment |
| GET | `/ideas/{ws}` | List ideas with current state, scores, lineage |

### Phase 6 — sources
| Method | Path | Purpose |
|---|---|---|
| GET | `/sources/{ws}` | List workspace's external sources |
| POST | `/sources/{ws}` | Add a `web_search` source |
| DELETE | `/sources/{ws}/{source_id}` | Remove a source |
| GET | `/sources/{ws}/suggestions` | LLM-driven source proposals |

### Phase 6.5 — config
| Method | Path | Purpose |
|---|---|---|
| GET | `/config/{ws}` | Read `CompanionConfig` for one workspace |
| POST | `/config/{ws}` | Patch any subset of fields |

### Phase 8 — documents
| Method | Path | Purpose |
|---|---|---|
| POST | `/promote/{ws}/{idea_id}` | Generate md/docx/pdf + publish to wiki + Mem0 |
| GET | `/document/{ws}/{idea_id}` | List artifacts already on disk |
| GET | `/document/{ws}/{idea_id}/{format}` | Download as `FileResponse` |

### Phase 9 — wiki
| Method | Path | Purpose |
|---|---|---|
| GET | `/wiki/{ws}` | List wiki pages (idea_id + title from first heading) |
| GET | `/wiki/{ws}/{idea_id}` | Return raw markdown body as `text/markdown` |

### Phase 11 — grand-task synthesis
| Method | Path | Purpose |
|---|---|---|
| GET | `/grand-task/{ws}/proposals` | List recent proposals |
| POST | `/grand-task/{ws}/{proposal_id}/accept` | Rotate `seed_prompt` |
| POST | `/grand-task/{ws}/{proposal_id}/reject` | Record reason for next synthesis |

### Phase 13 — cross-workspace inbox
| Method | Path | Purpose |
|---|---|---|
| GET | `/xworkspace/{ws}/inbox` | List undecided cross-workspace kernels |
| POST | `/xworkspace/{ws}/inbox/{kernel_id}/accept` | Feed kernel into next N cycles |
| POST | `/xworkspace/{ws}/inbox/{kernel_id}/dismiss` | Won't be re-proposed |

---

## React surface

`/cp/ops` → "🌀 Workspace Companion" tab. Eight sub-tabs in
[`CompanionTab.tsx`](../dashboard-react/src/components/CompanionTab.tsx),
backed by typed hooks in
[`api/companion.ts`](../dashboard-react/src/api/companion.ts):

| Sub-tab | Polling | Backend | Behaviour |
|---|---|---|---|
| 🌀 Live | 5 s + 20 s | ideas + config | At-a-glance: seed, budget, surfaced/documented counts; 5 most-recent ideas with inline thumbs + Promote |
| 💡 Ideas | 20 s | ideas | Full list (≤100); state + four scores (novelty/quality/transferability/panel) per idea |
| 📄 Documents | 20 s | ideas | Filter to DOCUMENTED; per-format download links |
| 📚 Wiki | 60 s | wiki | List + raw markdown body view |
| 🔍 Sources | 60 s | sources | List/add/remove + on-demand LLM suggestions |
| 🎯 Grand task | 60 s | proposals | Pending proposals; accept rotates seed |
| 📨 Inbox | 20 s | xworkspace | Cross-workspace kernels with relevance + source attribution |
| ⚙️ Settings | 60 s | config | Full `CompanionConfig` editor with diff-only PATCH on save |

Workspace selector at the top pulls from existing `/api/workspaces`
(reuses Phase 0 CP project listing — no new endpoint).

See also: [`OpsPage.tsx`](../dashboard-react/src/components/OpsPage.tsx) for
the parent tab pattern.

---

## Idle scheduler integration

[`loop.get_idle_jobs()`](../app/companion/loop.py) is called by
`idle_scheduler._default_jobs()` and returns four `(name, fn, JobWeight)`
tuples:

```python
[
  ("companion-tick",       companion_tick,                          JobWeight.MEDIUM),
  ("companion-ingest",     ingest.run_ingest,                        JobWeight.LIGHT),
  ("companion-grand-task", grand_task.run_synthesis_for_all_workspaces, JobWeight.MEDIUM),
  ("companion-xworkspace", cross_workspace.run_propagation_for_all_workspaces, JobWeight.LIGHT),
]
```

The registration block in [`idle_scheduler._default_jobs`](../app/idle_scheduler.py)
imports `app.companion.loop.get_idle_jobs` lazily inside a try/except, so
a Companion package import error never crashes the scheduler thread —
matching the convention used for SubIA's idle jobs.

JobWeight time caps:
- LIGHT: 60 s (ingest, xworkspace)
- MEDIUM: 180 s (tick, grand-task)
- HEAVY: 600 s (Companion does not use HEAVY)

`should_yield()` is checked between phases inside each cycle, so long-running
work cooperatively aborts when a user task arrives.

---

## Operational guide

### Activating in production

The companion router is mounted on every gateway start (see
`app/main.py` near `workspace_api` include). To make outbound Signal
delivery work end-to-end:

```bash
export COMPANION_SIGNAL_RECIPIENT="<your-signal-user-id>"
# or per-workspace:
curl -X POST http://localhost:8000/api/cp/companion/config/<ws_id> \
     -d '{"signal_recipient": "<other-id>"}'
```

Without either, `surface()` logs the would-send card text at INFO and the
`SURFACED` event payload records `signal_sent: false` — operators can see
formatting and replay later.

### Setting a seed prompt

The workspace's `seed_prompt` is the contemplation anchor:

```bash
curl -X POST http://localhost:8000/api/cp/companion/config/<ws_id> \
     -d '{"seed_prompt": "What does long-term Estonian forest stewardship demand of multi-generational decision tools?"}'
```

Or via the React Settings tab. Without a seed, every cycle returns
`aborted_reason="no_seed_prompt"` — the budget isn't charged.

### Scaling controls

- **Lower thresholds** (surface_threshold, novelty_threshold,
  panel_threshold) → more surfaces, more thumbs-up/down churn.
- **Lower `daily_budget_usd`** → fewer cycles per day. Floor is $0
  (effectively pauses).
- **Disable a workspace** (`enabled: false`) → all four idle jobs skip it.

### Reading the event log

Every state transition is auditable:

```bash
cat workspace/companion/events/<ws_id>.jsonl | jq .
```

Event types:
- `surfaced` — card sent to Signal/React (or attempted)
- `feedback` — thumbs up/down + comment
- `archived` — thumbs-down or explicit dismissal
- `documented` — promoted to md/docx/pdf
- `wiki_registered` — published to workspace wiki + Mem0 + system wiki
- `grand_task_proposed` / `_accepted` / `_rejected`
- `cross_workspace_inbox` / `_accepted` / `_dismissed`

---

## Safety invariants

Per the [CLAUDE.md](../../CLAUDE.md) safety guarantee that *evaluators must
live outside the system's ability to modify*:

- **Threshold bounds are infrastructure-level constants** in `config.py`.
  `clamp()` re-applies them on every save. The Self-Improver agent cannot
  widen them.
- **Cross-workspace sanitiser fails closed.** If
  `app.transfer_memory.sanitizer` is unimportable, no kernel propagates.
  Cross-workspace risk is the wrong thing to default-permissive.
- **Always human-in-loop for cross-workspace.** Kernels never
  auto-inject — only land in the inbox until the user accepts.
- **Adverse affect attractors pause cycles.** `live_factual_grounding < 0.3`
  stops every cycle, regardless of weight.
- **Daily budget gate.** Per-workspace UTC-day spend cap; cycles skip when
  exhausted.
- **Cooldown on surfacing.** `SURFACE_COOLDOWN_HOURS = 4` prevents flooding.
- **Idempotent persistence.** ChromaDB + JSONL writes use deterministic ids
  (idea_id, kernel_id, proposal_id); re-running `promote()` overwrites the
  same files.

---

## Failure modes & mitigations

| Failure | Mitigation |
|---|---|
| Creative MAS LLM unavailable | Cycle aborts cleanly with `aborted_reason="creative_crew_failed:<exc>"`; tick still recorded |
| ChromaDB down | `companion_ideas` writes log-and-skip; novelty search returns `1.0` (treat as fully novel); idea still in JSONL |
| Mem0 unavailable | `wiki._invoke_mem0_add` returns None; workspace + system wiki still land |
| pandoc absent | Markdown still written; docx/pdf silently skipped, `formats` dict reflects what landed |
| Signal CLI down | `enqueue_outbound` returns None; SURFACED event payload records `signal_sent: false` |
| Affect bridge unimportable | Treated as silent signal (weight=1.0, no block); cycles continue |
| Sanitiser unimportable | `_passes_sanitiser` returns False (fail closed); no cross-workspace propagation |
| Budget runaway | Daily UTC ledger; cap per workspace per day; cap reset at UTC midnight |
| Mode collapse | MAP-Elites diversity records every cycle; sparse-cell hints push toward voids in next prompt |
| Echo chamber on critic panel | Skeptic persona explicitly adversarial — score 1 = fatally flawed; rubric forbids consensus |
| Same idea re-surfaced | 4 h cooldown per workspace; per-idea event log prevents duplicate `SURFACED` |
| Same kernel re-proposed cross-workspace | `_already_proposed` checks target's `CROSS_WORKSPACE_INBOX` events for source_idea_id |
| User feedback drift / sycophancy | Reflexion uses up to 3 negatives + 2 positives; balanced, time-bounded |

---

## Research grounding

Selected papers and how they map:

- **Park et al. 2023, *Generative Agents***. Periodic reflection over
  memory → daily / 12 h grand-task synthesis (Phase 11) is the same
  mechanism, scoped per workspace.
- **Du et al. 2023, *Multi-Agent Debate***. Five-persona critic panel
  (Phase 7) implements the consensus-resistance idea; Skeptic is the
  adversarial vote.
- **Shinn et al. 2023, *Reflexion***. Phase 5 builds the verbal-RL prompt
  block from past thumbs-up/down ideas — directly the "lessons from past
  episodes in the next prompt" recipe.
- **Madaan et al. 2023, *Self-Refine***. The Discuss phase of Creative MAS
  + critic panel form one round of self-critique; Phase 12+ MAP-Elites
  diversity hints close the loop.
- **Si, Yang & Hashimoto 2024, *Can LLMs Generate Novel Research Ideas?***
  Surface threshold gates output; humans rate LLM ideas as more novel
  than human ideas under matched conditions, justifying "generate then
  gate" over "filter at generation."
- **Mouret & Clune 2015, *MAP-Elites***. Phase 12 wires per-workspace QD
  archives over the existing `app.map_elites_wiring`; void-cell hints in
  the prompt drive coverage.
- **Hu et al. 2025, *Sleep-time Compute***. Justifies the cheap-tier
  Diverge / mid-tier Discuss / premium Converge cascade — idle time is
  the budget for amortised exploration of likely-relevant questions.
- **Bai et al. 2022, *Constitutional AI***. Critic-personas pattern;
  rubric-as-constitution rather than learned-reward.
- **Hayes-Roth 1985, *Blackboard architecture***. Asynchronous,
  opportunistic phase composition — even though no explicit blackboard
  service is shipped, the cycle's failure-tolerant phase chain echoes the
  pattern.

---

## Phase history

Each phase shipped as one or more commits on `feature/reverie-layer-phase-0-1`,
merged to `main` as `f862c9e`.

| Phase | Theme | Commits |
|---|---|---|
| 0+1 | Workspace seed + idle loop skeleton | `b7e13bd` |
| 2 | Cycle wiring + WorkspaceKB + affect bridge | `e7aed1d` |
| 3 | Idea store + scoring + lineage persistence | `ee52611` |
| 4 | Surfacing + feedback intake + event log + react API | `ddb3cc1` |
| 5 | Reflexion: feedback shapes the next cycle's prompt | `d5479ac` |
| 6 | External sources + auto-suggest + daily ingestion | `1335825` |
| 6.5 | Config endpoint — recover seed/budget editability | `6f6de8b` |
| 7 | Five-persona critic panel | `27c6bd9` + `914fe2e` |
| 8 | Document maturation pipeline (md / docx / pdf) | `26ac696` + `fa8c3b2` |
| 9 | Workspace wiki + Mem0 + system-wiki cross-layer registration | `262e264` + `31e924d` |
| 10 | React `/cp/ops/Companion` tab + API client | `b21cb81` |
| 11 | Grand-task synthesis (12 h cadence) | `827a647` + `f6be025` |
| 11.5 | Cold-start seed bootstrap from CP mission + tickets | (see commit log) |
| 12 | Per-workspace MAP-Elites diversity hook | `8a6a51b` |
| 13 | Cross-workspace transfer (hybrid model) | `2656ed2` |
| 4.5 + 9.5 + 10.5 | Production wire-ups (router, signal, mem0) | `e2e89e4` |
| Merge | Workspace Companion → main | `f862c9e` |

---

## File inventory

### Backend (`app/companion/`)

| File | Lines | Concern |
|---|---:|---|
| `__init__.py` | 16 | Package init + re-exports |
| `config.py` | 230 | `CompanionConfig` + bounds + load/save into CP `config_json` |
| `state.py` | 76 | Runtime sidecar (vruntime, last_tick_at, daily_cost_usd) |
| `budget.py` | 45 | Daily UTC-day cost ledger |
| `scheduler.py` | 165 | CFS-fairness selector + 12 h floor + affect gates |
| `loop.py` | 90 | `companion_tick` + `get_idle_jobs` |
| `workspace_kb.py` | 165 | Composer for temporal + KB v2 + companion_sources |
| `cycle.py` | 312 | One ideation pass — scoring + persistence + surfacing dispatch |
| `idea_store.py` | 240 | `IdeaRecord` + JSONL + ChromaDB + `current_state` |
| `events.py` | 100 | Append-only event log per workspace |
| `feedback.py` | 80 | Thumbs recording + summary for Reflexion |
| `surfacing.py` | 175 | `should_surface` + card + Signal outbound |
| `reflexion.py` | 100 | Past-feedback prompt block |
| `document_pipeline.py` | 240 | promote → md + pandoc docx/pdf + wiki publish |
| `wiki.py` | 360 | Workspace + system wiki + Mem0 cross-layer registration |
| `sources.py` | 160 | Per-workspace source CRUD |
| `source_suggester.py` | 110 | LLM-driven source proposals |
| `ingest.py` | 175 | Daily fetch + ChromaDB index |
| `grand_task.py` | 280 | 12 h synthesis from polished ideas |
| `diversity.py` | 185 | Per-workspace MAP-Elites + sparse-cell hints |
| `critique.py` | 150 | Five-persona panel |
| `cross_workspace.py` | 320 | Sanitiser + relevance gate + inbox |

### API (`app/api/companion_api.py`)
~370 lines, 18 routes.

### React (`dashboard-react/`)
- `src/api/companion.ts` — 415 lines (types + 17 hooks).
- `src/components/CompanionTab.tsx` — 885 lines (8 sub-tabs).
- `src/components/OpsPage.tsx` — +11 lines (tab registration).

### Tests (`tests/test_companion_*.py`)
23 files, 313 tests.

---

## Open follow-ups

1. **Auto-promotion on high panel-score + thumbs-up.** Phase 4.5+ —
   currently the user has to click Promote. The hook is straightforward in
   `feedback.record`: when polarity=UP and panel_score ≥ a configurable
   floor, call `document_pipeline.promote` automatically.

2. **HDBSCAN topic clustering for the workspace wiki.** Phase 9 ships
   one page per idea + an `_index.md`. Phase 9.5 could cluster ideas by
   embedding similarity, name each cluster via cheap LLM, and group pages
   into topic folders.

3. **External wiki sync.** One-direction export to BookStack / Outline
   via their APIs; out of scope for v1.

4. **Per-cycle telemetry dashboard.** Per-workspace metrics surfaced on
   /cp/ops: ideas/day, surface rate, panel-score distribution, time-to-document.
   Source data is in `workspace/companion/{ideas,events}/*.jsonl` —
   purely additive.

5. **MAP-Elites coverage telemetry on Live tab.** `diversity.coverage(ws)`
   already returns the report; just plumb it through the React Live view.

6. **Conversation-history seam.** Right now `WorkspaceKB.compose` doesn't
   include user conversation. The `app/conversation_store.py` layer is
   sender-keyed, not workspace-keyed; needs a small workspace-aware view.

---

## Cross-references

- [`AFFECT_LAYER.md`](AFFECT_LAYER.md) — `live_factual_grounding` provider
  consumed by `companion.scheduler`.
- [`SUBIA.md`](SUBIA.md) — primitive Reverie engine + the architectural
  honesty stance the Companion inherits.
- [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md) — Mem0 / ChromaDB /
  Neo4j stack the Companion writes into.
- [`LLM_SUBSYSTEM.md`](LLM_SUBSYSTEM.md) — `create_specialist_llm` cascade
  used by all Companion LLM calls.
- [`RECOVERY_LOOP.md`](RECOVERY_LOOP.md) — refusal detection patterns
  available for surfacing failure analysis.
- [`ERROR_MONITOR.md`](ERROR_MONITOR.md) — operational visibility surface
  the Companion sub-tab sits next to on `/cp/ops`.
- [`CREATIVITY_SYSTEM.md`](CREATIVITY_SYSTEM.md) — Creative MAS internals
  the Companion drives via `run_creative_crew`.
- [`CONTROL_PLANES.md`](CONTROL_PLANES.md) — `control_plane.projects` table
  where `CompanionConfig` lands as `config_json.companion`.
- [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) — the Self-Improver agent's
  bounded-modification rules that `panel_threshold` and friends preserve.
