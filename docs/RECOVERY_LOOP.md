# Capability Recovery Loop

The reference for how this system catches and re-routes refusal-shaped
final answers ("I cannot…", "no access to…", "<unavailable>") instead
of delivering them to the user. Written so a new contributor can read
it once and understand every layer plus the design rationale.

---

## 1. Why this design exists

Across April 2026 the system shipped many per-failure-mode point fixes
— vetting wrong-crew detection, credit failover to local Ollama, sync
ingestion in async handlers, etc. Each one closed a specific failure
class. None of them addressed the **meta** problem: when an agent
delivers an honest "I can't help with that," it almost always means
the system *did* have the capability, but the agent that picked up the
request couldn't reach it.

A single-day audit of refusals on 2026-04-28 found five distinct
"capability-locality failures" — every one of them recoverable by the
system itself if it had a meta-layer to notice and re-try:

| Refusal text | Underlying cause | Where the fix lived |
|---|---|---|
| "I do not have access to your email account" | Research crew picked, lacks `email_tools` | PIM crew has them |
| "EXECUTION OUTPUT: <unavailable>" | Coding crew dumped a script, sandbox not invoked | `app/sandbox_runner.py` exists |
| "Crew pim failed: 402" | OpenRouter out of credits | Credit failover to Ollama (litellm path only) |
| "Sorry, I had trouble understanding" | Vetting parse failed mid-synthesis | Synthesis result was already in hand |
| Coding crew dumped 230-line Python with `<unavailable>` | No execution wired | Sandbox could have run it |

The Recovery Loop closes this gap. It runs **after** vetting and
critic, **before** outbound delivery, as a last-mile defense:

> *"The answer we're about to send to the user looks like a refusal.
> Try harder before bothering them."*

The design philosophy is **read-mostly + escalating-cost**:
* Read the existing crew/tool/skill registries (don't reinvent).
* Try cheap recoveries first (direct tool calls, ~$0).
* Escalate to expensive ones (premium model retry, ~$0.10) only if
  cheap ones fail.
* When *nothing* recovers, replace bare refusal with a structured
  diagnostic + queue a forge experiment so the same gap is cheaper
  next time.

---

## 2. Architecture (one screen)

```
                     vetting + critic complete
                              │
                              ▼
                       ┌──────────────┐
                       │  final_text  │   the synthesised answer
                       └──────┬───────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │   Layer 1 — Detector    │   refusal_detector.py
                  │   detect_refusal(text)  │
                  └────────────┬────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        signal=None     signal=policy*     RefusalSignal
       (no refusal)    (don't recover)    {category, conf}
              │                │                │
              │                │                ▼
              │                │     ┌──────────────────────┐
              │                │     │  Layer 2 — Librarian │  librarian.py
              │                │     │  find_alternatives() │
              │                │     └──────────┬───────────┘
              │                │                │
              │                │                ▼
              │                │     ┌──────────────────────┐
              │                │     │  Layer 3 — Strategies│  strategies/*.py
              │                │     │  ranked, cheap-first │
              │                │     └──────────┬───────────┘
              │                │                │
              │                │      one of  ┌─┴────────────────────┐
              │                │      these:  │ direct_tool          │ ~$0,   ~5s
              │                │              │ sandbox_execute      │ ~$0.005,~20s
              │                │              │ re_route             │ ~$0.02,~30s
              │                │              │ skill_chain          │ ~$0.01,~10s
              │                │              │ escalate_tier        │ ~$0.10,~60s
              │                │              │ forge_queue (always) │ async
              │                │              └─┬────────────────────┘
              │                │                │
              │                │                ▼
              │                │     ┌──────────────────────┐
              │                │     │  Layer 4 — Loop      │  loop.py
              │                │     │  budget + recurse    │
              │                │     │  guard + audit       │
              │                │     └──────────┬───────────┘
              │                │                │
              │                │           ┌────┴────┐
              │                │           │         │
              │                │       success    no-recovery
              │                │           │         │
              │                │           ▼         ▼
              │                │     recovered    diagnostic
              │                │     text +       text from
              │                │     route note   forge_queue
              │                │           │         │
              ▼                ▼           ▼         ▼
        ╔═══════════════════════════════════════════════════╗
        ║              outbound delivery                    ║
        ╚═══════════════════════════════════════════════════╝
```

\* policy refusals respected unless `force=True` (user-driven override
via Signal command — see §8).

Every layer fails soft. Every layer has a dedicated module. The hook
in `orchestrator.py` is wrapped in `try/except` so a recovery bug can
never break delivery of the original answer.

---

## 3. The post-vetting invariant

**Every final answer in the codebase passes through the recovery
loop's hook before delivery.** Enforced by convention + grep. The
single hook point is:

```
app/agents/commander/orchestrator.py
  inside Commander._route() → post-vetting / post-critic block
  immediately before _proactive_notes append
```

| Bypass vector | Allowed? |
|---|---|
| Crews returning answers without going through Commander.handle() | ❌ — every Signal/dashboard inbound path routes through `Commander.handle` |
| Streaming partial results out of multi-crew dispatch | ⚠️ — partials from `parallel_runner` aren't covered today (see §12 limitations) |
| Direct ticket completions via `/api/cp/tickets/{id}` PUT | ⚠️ — manual completions skip Commander; if the typed text is a refusal, it's delivered as-is |
| Disabling via `RECOVERY_LOOP_ENABLED=false` | ✅ — clean opt-out, original pipeline restored |

Off by default historically; flipped to `true` on 2026-04-28 after
verification. The env flag remains as a kill switch.

---

## 4. Layer 1 — Refusal Detector

`app/recovery/refusal_detector.py:detect_refusal`

Conservative pattern matcher. Returns a `RefusalSignal` when the
response text looks like a capability refusal we could plausibly
recover from; `None` otherwise.

### Signal shape

```python
@dataclass
class RefusalSignal:
    category: str        # missing_tool / auth / execution / data_unavailable / generic
    confidence: float    # 0.0 – 1.0
    matched_phrase: str  # the specific phrase we matched
    refusal_density: float  # fraction of text that is refusal language
```

### Categories

Five non-policy categories (a sixth, `policy`, is detected and
**respected** — i.e. NOT recovered). Each maps to different
alternative-route strategies via the librarian.

| Category | Example phrase | Strategy preference |
|---|---|---|
| `missing_tool` | "i do not have access to" / "no tool available" / "<unavailable>" | re_route or direct_tool |
| `auth` | "api key" / "credentials are not" / "authorize a connection" | forge_queue (we can't conjure keys) |
| `execution` | "cannot run" / "no code execution" / "no execution environment" | sandbox_execute |
| `data_unavailable` | "could not find any" / "no records were found" | escalate_tier (stronger LLM may persist) |
| `generic` | "i cannot" / "i'm unable to" / "had trouble understanding" | escalate_tier or re_route |
| `policy` | "violates" / "against my guidelines" / "not appropriate" | **none — respect the refusal** |

The full phrase catalog with per-phrase base-confidence weights lives
in `_PHRASES_BY_CATEGORY` at the top of the module.

### Conservative-by-design — the density check

A long, useful answer that contains "I can't run this code locally,
but here's the algorithm…" would be a false positive under naive
phrase matching. The detector defends against this with a **density
check**:

```python
density = covered_chars_by_refusal_phrases / total_chars
final_confidence = base_phrase_confidence × sqrt(density × 8)
```

A single weak phrase in 1500 characters of useful prose scores
density ~0.005 → density_factor ~0.2 → final confidence multiplied
by 0.2 → almost certainly below threshold. A short response that's
mostly refusal language scores density ~0.3+ → density_factor ~1.0
→ confidence retained.

### Threshold

```bash
RECOVERY_DETECTION_THRESHOLD=0.80   # default (conservative)
```

Anything from 0.0 (never miss) to 1.0 (only ironclad matches). At
0.80, the detector fires on ~5% of responses tested in dev — almost
always actual refusals.

### Force mode

`detect_refusal(text, force=True)` bypasses three guards:
1. Policy guard (`"violates my guidelines"` no longer respected)
2. Density check (single passing mention triggers)
3. Confidence threshold (low-scoring matches still fire)

When the response has no refusal phrase at all under force, the
function returns a low-confidence (0.50) generic signal so the loop
has SOMETHING to dispatch the librarian on. Force is ONLY entered
via the user-driven Signal command (§8) — auto-detection always uses
`force=False`.

### Code pointers

| Function | Purpose |
|---|---|
| `_PHRASES_BY_CATEGORY` | The pattern catalog |
| `_refusal_density(text)` | Compute density (used by confidence formula) |
| `_detection_threshold()` | Read env-tunable threshold |
| `detect_refusal(text, force=False)` | Public entry point |

---

## 5. Layer 2 — Capability Librarian

`app/recovery/librarian.py:find_alternatives`

Read-only inventory of what the system can do. Given a `(task,
refusal_category, used_crew)` tuple, returns a ranked list of
`Alternative` objects — the loop walks this list within budget.

### Alternative shape

```python
@dataclass
class Alternative:
    strategy: str          # which strategy module handles this
    rationale: str         # human-readable why
    est_cost_usd: float    # ranking key
    est_latency_s: float   # ranking key (tiebreak)
    sync: bool             # True = run inside user's request
    crew: str | None       # for re_route
    tier: str | None       # for escalate_tier
    tool: str | None       # for direct_tool
    extra: dict
```

### Capability map

```python
_CAPABILITY_MAP = {
    "email":             {"crews": ["pim"], "tools": ["email_tools.check_email"], ...},
    "calendar":          {"crews": ["pim"], "tools": ["calendar_tools.list_events"], ...},
    "tasks":             {"crews": ["pim"], "tools": ["task_tools.list_tasks"], ...},
    "code_execute":      {"crews": ["coding"], "tools": ["code_executor", "sandbox.run"], ...},
    "research_matrix":   {"crews": ["research"], "tools": ["research_orchestrator"], ...},
    "web":               {"crews": ["research"], "tools": ["web_search", "browser_fetch"], ...},
    "files":             {"crews": ["desktop", "research"], "tools": ["file_manager"], ...},
}
```

Hand-curated for stability + debuggability. Adding a new capability
is one dict entry — see §11.

### Ranking algorithm

1. Sync strategies (direct_tool, sandbox_execute, re_route,
   skill_chain, escalate_tier) sorted ascending by `(est_cost_usd,
   est_latency_s)`.
2. `forge_queue` always appended **last** — it's the async fallback
   that never returns empty, so it serves as the floor of the
   alternative list.

The loop respects this order strictly: cheaper strategies get budget
first, expensive ones only if cheap ones fail.

### Tool-registry bridge

The hand-curated `_CAPABILITY_MAP` is augmented by a semantic-search
fallback against the tool registry's ChromaDB index. After the keyword
path emits its alternatives, `_registry_alternatives(task, used_crew)`
calls `tool_registry.discovery.search_tools(intent=task, limit=5)` and
maps each ranked `ToolMatch` into a `direct_tool` `Alternative` —
provided the tool name is in the recipe-eligibility list
`_DIRECT_TOOL_RECIPE_NAMES` (shared with the keyword path).

Why bridge them at all: the keyword catalog grows by hand on every
new tool. Semantic search picks up `@register_tool`-annotated tools
whose phrasing the operator never anticipated. The 4-layer
contamination defense in `discovery.py` (subjectless guard,
quarantine, tier, workspace, distance ceiling 0.55) applies
uniformly to recovery suggestions — so the bridge inherits the same
safety properties as the agent-side `tool_search` primitive.

Dedup runs after the registry pass: any `(strategy, tool, crew)`
tuple already emitted by the keyword path is skipped, so a registry
hit on the same tool the keyword path found doesn't double-emit.

Today the bridge only emits `direct_tool` alternatives because the
registry doesn't expose a `source_module → crew` map. When that
lands, the bridge can also emit `re_route` for hits whose source
crew differs from `used_crew`.

Failure mode: if the registry / Chroma is unreachable, the bridge
returns an empty list and recovery proceeds with the keyword
alternatives. Registry blips never break recovery.

### Code pointers

| Function | Purpose |
|---|---|
| `_CAPABILITY_MAP` | Hand-curated capability → tools/crews/keywords |
| `_infer_capabilities(task)` | Keyword match against the map |
| `_DIRECT_TOOL_RECIPE_NAMES` | Eligibility list — tools with a recipe in `direct_tool.py:_TOOL_RECIPES` |
| `_registry_alternatives(task, used_crew)` | Semantic-search bridge against the tool registry |
| `_current_tier_for_role(role)` | Best-effort tier lookup for escalate_tier eligibility |
| `find_alternatives(...)` | Public entry point — returns ranked list |

---

## 6. Layer 3 — Strategies

Six concrete strategies. Each lives in its own module under
`app/recovery/strategies/`. All share the contract:

```python
def execute(task: str, alt: Alternative, ctx: dict) -> StrategyResult
```

`ctx` carries the call context the strategy may need:

| Key | Used by | Notes |
|---|---|---|
| `commander` | re_route, escalate_tier | bound `Commander` instance |
| `user_input` | all | original user request |
| `crew_used` | re_route, escalate_tier | the failed crew name |
| `conversation_history` | re_route, escalate_tier | for crew context |
| `difficulty` | re_route, escalate_tier | original difficulty score |
| `refusal_category` | forge_queue | for diagnostic phrasing |
| `original_response` | sandbox_execute | source of code blocks |

### 6.1 `direct_tool`

`app/recovery/strategies/direct_tool.py`

**Cheapest path.** Bypass the LLM entirely and call a known tool
directly with regex-extracted parameters.

#### When it fires

Librarian surfaces this when a `_CAPABILITY_MAP` entry's tools list
contains a name in the **recipe table** (`_TOOL_RECIPES`). Currently
wired:
* `email_tools.check_email`
* `calendar_tools.list_events`

Adding a new tool: add a recipe + add the tool name to the librarian's
hard-coded eligibility list (see §11).

#### Param extraction

Regex-based, deliberately simple. Tool defaults fill in anything the
regex doesn't catch.

| Pattern | Email param | Calendar param |
|---|---|---|
| `"top N"` / `"first N"` / `"give me N"` | `limit=N` (capped 100) | `limit=N` (capped 50) |
| `"today"` / `"this morning"` | `days_back=1` | `days_ahead=1` |
| `"yesterday"` | `days_back=2` | — |
| `"this week"` | `days_back=7` | `days_ahead=7` |
| `"weekend"` | `days_back=3` | — |
| `"unread"` / `"new emails"` | `unread_only=True` | — |

#### Result shape

`StrategyResult(success=True, text="Pulled this directly from your inbox (no LLM in the loop): ...", note="Called email_tools.check_email directly (no LLM layer).", route_changed=True)`

#### Failure modes (returns success=False so loop falls through)

* Tool can't be created (env not configured, e.g. no `EMAIL_PASSWORD`)
* Tool returns empty/tiny output
* Tool returns a string starting with "Error:" or containing "no email config"

### 6.2 `sandbox_execute`

`app/recovery/strategies/sandbox_execute.py`

**Closes the 2026-04-25 coding-crew-dumps-script regression.** When
the response contains a Python code block + `<unavailable>` marker,
extract the largest code block and run it in the existing Docker
sandbox.

#### When it fires

Librarian only surfaces it when the response_text contains a fenced
Python block (` ```python ` or ` ```py `). No code → no spin-up.

#### Sandbox isolation

Same flags as `app/sandbox_runner.py:run_code_check`:

```bash
docker run --rm
  --network none           # no internet
  --memory 512m            # mem cap
  --cpus 1
  --security-opt no-new-privileges
  --read-only              # rootfs locked
  --tmpfs /tmp:size=256m
  -v <tmp>:/work:ro        # script bind-mount, read-only
  crewai-sandbox:latest python /work/script.py
```

60-second wall-clock cap.

#### Pre-flight rejects

The strategy returns `success=False` BEFORE spinning up Docker if the
script:

* Imports anything network-shaped (`requests`, `urllib`, `aiohttp`,
  `httpx`, `socket`, `smtplib`, `imaplib`, `paramiko`, `ftplib`,
  `telnetlib`, `websocket`, `fetch(...)`)
* References host-specific filesystem paths (`/Users/`, `/home/`,
  `C:\\`, `~/`)

These would just timeout or `FileNotFoundError`. Better to skip and
let `re_route` or `skill_chain` try a different angle.

#### Result shape

```
Ran the script the coding crew produced. Output:

```
<stdout>
```

Source script (for reference):

```python
<extracted code, truncated to 2000 chars>
```
```

The user sees BOTH the script and its real output — not just bare
stdout, which would be confusing without context.

### 6.3 `re_route`

`app/recovery/strategies/re_route.py`

**The original Phase 1 strategy.** Re-target the request to a different
crew, reusing `Commander._run_crew` so the new dispatch benefits from
all the existing intelligence (matrix detection, attachment hints,
ToM, etc.).

#### When it fires

For any refusal category where the librarian found a crew with
matching tools that wasn't the failed one. The most-curated case is
"email question routed to research" → re_route to PIM.

#### Failure modes

* Target crew also produces refusal-shaped output (defensive
  re-detection prevents looping)
* Target crew returns empty/tiny output (<10 chars)
* Commander or context unavailable

### 6.4 `skill_chain`

`app/recovery/strategies/skill_chain.py`

**First-cut: surface a single matching skill from the library.**
Multi-skill chaining (run skill A, pass output to skill B) is deferred.

#### When it fires

Librarian always surfaces this — skills are domain-agnostic, so a
match might exist for any task.

#### Search

Reuses `app.self_improvement.integrator.search_skills(query, n=3)` —
the same underlying index the orchestrator pre-task loader queries
(via `search_skills_scored`). The pre-task loader layers four
contamination defences on top — see *MEMORY_ARCHITECTURE.md §6.7.1
Retrieval API and contamination defences* — but the recovery flow's
`skill_chain` strategy uses the unscored convenience wrapper because
it's invoked on demand for an explicit recovery query, not on every
crew dispatch.

Score threshold `_MIN_RELEVANCE_SCORE = 0.55`. Below that, "best
match" is treated as "no match" so we don't apply an irrelevant skill.

#### Result shape

> *"I have an existing skill that addresses this — 'PSP enrichment patterns' (relevance 0.78). [skill body, truncated to 3500 chars]"*

The user sees `route_changed=True` so the answer carries a `_(Surfaced
existing skill 'X' from the skills library.)_` note.

### 6.5 `escalate_tier`

`app/recovery/strategies/escalate_tier.py`

**Same crew, stronger LLM.** Use case: the right specialist was
chosen, but its budget-tier model gave up early on a research-deep
question. A premium model on the same crew often persists.

#### When it fires

Only for `generic` and `data_unavailable` categories — no model can
fix a missing API key. Only when `used_tier` is `None`/`budget`/`mid`/
`free` (already-premium can't escalate).

#### Mechanism

Sets `LLM_FORCE_TIER=premium` env var, re-runs `Commander._run_crew`,
restores the env on completion. The selector respects the env and
picks a premium-tier model from the catalog.

`difficulty` is bumped to `max(difficulty, 7)` so the selector
doesn't demote in the retry.

### 6.6 `forge_queue`

`app/recovery/strategies/forge_queue.py`

**Always-last async fallback.** Never recovers the current request.
Records the gap for offline skill-forge processing AND returns a
structured diagnostic so the user sees actionable info instead of
bare refusal.

#### Frequency tracking

```
workspace/recovery/refusal_frequency.json:
{
  "missing_tool::find sales lead at company": {
    "timestamps": ["2026-04-26T...", "2026-04-27T...", "2026-04-28T..."],
    "last_seen": "2026-04-28T19:42:18+00:00",
    "count_in_window": 3
  }
}
```

Window: 7 days. Old timestamps fall off automatically.

#### Auto-forge threshold

```python
_AUTO_FORGE_THRESHOLD = 3       # same gap 3 times in 7 days...
_AUTO_FORGE_WINDOW_DAYS = 7     # ...auto-queues a forge experiment
```

When threshold hit, append entry to `workspace/skills/learning_queue.md`:

```markdown
## refusal-recovery: missing_tool::find sales lead at...
* added: 2026-04-28T19:42:18+00:00
* category: missing_tool
* refusal count in 7d: 3
* example task: find sales lead at Foo Co
* suggested skill: a tool that addresses 'missing_tool' failures...
```

The forge picks up these entries on its next idle cycle (see
`SELF_IMPROVEMENT.md` §4 for the forge pipeline).

#### Diagnostic answer template

Per category, the answer surfaces actionable next steps:

| Category | Hint to user |
|---|---|
| `missing_tool` | "Set the API key in `.env` if you have one for the underlying service. Or rephrase to route to a crew that has the tool." |
| `auth` | "A required API key is missing. Set in `.env` and restart." |
| `execution` | "No code-execution sandbox wired up. Tell me what output you needed." |
| `data_unavailable` | "Data sources don't have what you asked. Point me at a URL or attachment." |
| `generic` | "Agent gave up without a clear reason. Try rephrasing or use 'force this' to escalate." |

If frequency >= threshold, the answer ends with:
> *"This is the 3rd time I've hit this gap in the last 7 days, so I've queued a skill-forge experiment to close it. Check the Skills tab in ~6h."*

If frequency < threshold:
> *"This is the 2nd time I've hit this gap. 1 more and I'll auto-queue a forge experiment."*

---

## 7. Layer 4 — The Loop

`app/recovery/loop.py:maybe_recover`

The orchestrator. Called from `Commander._route()` post-vetting/post-critic.

### Result shape

```python
@dataclass
class RecoveryResult:
    triggered: bool                   # True if refusal was detected
    success: bool                     # True if a strategy returned an answer
    text: str | None                  # the recovered (or diagnostic) text
    note: str | None                  # short note, shown only if route_changed
    route_changed: bool               # affects whether note is shown
    strategies_tried: list[str]
    refusal_signal: RefusalSignal | None
    elapsed_s: float
```

### Budget guards

Two env-tunable limits:

```bash
RECOVERY_MAX_ATTEMPTS=2     # max strategies per task (default 2, range 1-5)
RECOVERY_BUDGET_SECONDS=90  # wall-clock ceiling (default 90s)
```

Both checked between strategies. Forge_queue always runs last and
costs ~0s, so it's effectively always reachable.

### Recursion guard

```python
_in_recovery: ContextVar[bool] = ContextVar("recovery_in_progress", default=False)
```

A strategy's own LLM calls (e.g. re_route's `_run_crew`) set this
True for their span. If the strategy's LLM call produces a refusal,
that refusal does NOT trigger another recovery loop — the
`maybe_recover` early-returns when the ContextVar is already set.

### Audit trail

Every detection + strategy attempt fires an audit entry:

```python
get_audit().log(
    actor="recovery_loop",
    action="refusal.detected",
    detail={
        "category": signal.category,
        "confidence": signal.confidence,
        "phrase": signal.matched_phrase,
        "crew": crew_used,
        "task": user_input[:300],
    },
)
```

```python
get_audit().log(
    actor="recovery_loop",
    action="strategy.executed",
    detail={
        "strategy": alt.strategy,
        "success": result.success,
        "error": result.error,
        "crew_used": crew_used,
        "target_crew": alt.crew,
        "target_tier": alt.tier,
    },
)
```

Query via `/api/cp/audit?actor=recovery_loop` or by SQL on
`control_plane.audit_log`.

### Hook integration

```python
# orchestrator.py — inside Commander._route(), post-vetting + post-critic
try:
    from app.recovery import maybe_recover, is_enabled as _recov_enabled
    if _recov_enabled():
        rec = maybe_recover(
            final_result, user_input, crew_name,
            commander=self, difficulty=difficulty,
            used_tier=get_last_tier() or "unknown",
            conversation_history=_crew_history,
        )
        if rec.triggered:
            if rec.success and rec.text:
                final_result = rec.text
                if rec.route_changed and rec.note:
                    final_result += f"\n\n_{rec.note}_"
except Exception as _rec_exc:
    logger.debug(f"recovery: loop raised; preserving original answer", exc_info=True)
```

The bare `try/except` is **load-bearing**: a recovery bug must never
break delivery of the original answer.

---

## 8. The user-driven `force this` Signal command

`app/agents/commander/commands.py:_handle_force_recover`

Triggered when the next user message after a refusal-shaped response
matches one of:
* `force this`
* `force recover`
* `force-recover`
* `try harder`
* `try alternative`
* `try another way`
* `find another way`

(Case-insensitive, exact match or `startswith`.)

### Mechanism

1. Walk back through `conversation_store.get_history(sender, limit=20)`
2. Skip the most recent user message (it IS the force-recover command)
3. Find the previous (assistant_response, user_question) pair
4. Call `maybe_recover(...)` with `force=True`

`force=True` bypasses three things in `detect_refusal`:
* Policy guard
* Density check
* Confidence threshold

When the previous response has NO refusal phrases at all under force,
`detect_refusal` returns a low-confidence `generic` signal so the
loop has something to feed the librarian. The librarian's
`skill_chain` and `escalate_tier` strategies are domain-agnostic,
so they always have a chance.

### Result format

If recovery succeeds:

> *[the recovered answer]*
>
> *_(force-recover: direct_tool, re_route)_*

The suffix shows what was tried so the user knows what changed.

If recovery doesn't find anything to do:

> *"Tried 2 alternative routes (skill_chain, escalate_tier) but none produced a better answer. The original response stands."*

---

## 9. Configuration

All env-tunable. Set in `.env` + `docker compose up -d gateway`.

| Var | Default | Purpose |
|---|---|---|
| `RECOVERY_LOOP_ENABLED` | `false` (pre-2026-04-28), `true` (now) | Master switch |
| `RECOVERY_DETECTION_THRESHOLD` | `0.80` | Confidence threshold for `detect_refusal` |
| `RECOVERY_MAX_ATTEMPTS` | `2` | Max strategies per task (1-5) |
| `RECOVERY_BUDGET_SECONDS` | `90` | Wall-clock ceiling for the whole loop |

Forge-related (no env vars; tune in `forge_queue.py` if needed):

```python
_AUTO_FORGE_THRESHOLD = 3       # same gap 3 times...
_AUTO_FORGE_WINDOW_DAYS = 7     # ...in 7 days → queue
```

Sandbox-execute (tune in `sandbox_execute.py`):

```python
_EXEC_TIMEOUT_S = 60
_OUTPUT_CAP_CHARS = 8000
```

Skill-chain (tune in `skill_chain.py`):

```python
_MIN_RELEVANCE_SCORE = 0.55
```

---

## 10. Operating playbook

### 10.1 Turning the loop on

```bash
# .env
RECOVERY_LOOP_ENABLED=true

# restart
docker compose up -d gateway

# verify
docker logs crewai-team-gateway-1 2>&1 | grep "recovery: loop"
# should see no errors; first refusal triggers the first audit entry
```

### 10.2 Watching it work

```bash
# real-time recovery activity
docker logs -f crewai-team-gateway-1 2>&1 | grep -E "recovery:"

# audit log via API
curl -s http://127.0.0.1:8765/api/cp/audit?limit=50 \
  | jq '[.[] | select(.actor=="recovery_loop")]'

# postgres direct
docker exec crewai-team-postgres-1 psql -U mem0 -d mem0 -c "
  SELECT timestamp, action, detail_json
  FROM control_plane.audit_log
  WHERE actor='recovery_loop'
  ORDER BY id DESC LIMIT 20;
"
```

### 10.3 Triaging false positives

Symptom: a useful answer got replaced by something weird.

Check the audit log:
```sql
SELECT detail_json
FROM control_plane.audit_log
WHERE actor='recovery_loop' AND action='refusal.detected'
ORDER BY id DESC LIMIT 10;
```

If you see `confidence` values barely over 0.80 firing on legitimate
answers, raise the threshold:
```bash
RECOVERY_DETECTION_THRESHOLD=0.90
```

If a specific phrase keeps misfiring (e.g. "I can't promise" being
treated as a refusal), edit `_PHRASES_BY_CATEGORY` in
`refusal_detector.py` to lower its base confidence or remove it.

### 10.4 Disabling on the fly

```bash
# kill switch
echo "RECOVERY_LOOP_ENABLED=false" >> .env
docker compose up -d gateway   # 30s downtime, recovery now off
```

The original answer pipeline is untouched, so disabling cleanly reverts.

### 10.5 Reading the frequency log

```bash
cat workspace/recovery/refusal_frequency.json | jq .
```

Each gap key is `<category>::<normalized_task>`. The
`count_in_window` field shows progress toward auto-forge threshold.
When `count_in_window >= 3`, the next refusal of that gap triggers
forge queueing — verify by checking
`workspace/skills/learning_queue.md` afterwards.

### 10.6 Manually clearing a gap

If a refusal is recorded but you've fixed the underlying cause:

```bash
# remove the entry (any valid JSON edit)
python3 -c "
import json
data = json.load(open('workspace/recovery/refusal_frequency.json'))
del data['missing_tool::find sales lead at...']
json.dump(data, open('workspace/recovery/refusal_frequency.json','w'), indent=2)
"
```

The next refusal of that pattern restarts the count from 1.

---

## 11. Adding a new strategy

The minimal contract — one file under `app/recovery/strategies/`:

```python
# app/recovery/strategies/my_strategy.py
from app.recovery.librarian import Alternative
from app.recovery.strategies import StrategyResult

def execute(task: str, alt: Alternative, ctx: dict) -> StrategyResult:
    # ... do something ...
    if success:
        return StrategyResult(
            success=True,
            text="...",
            note="Brief explanation if route_changed.",
            route_changed=True,
        )
    return StrategyResult(success=False, error="reason")
```

Then wire it into the loop's dispatcher:

```python
# app/recovery/loop.py:_execute_strategy
if alt.strategy == "my_strategy":
    from app.recovery.strategies import my_strategy
    return my_strategy.execute(task, alt, ctx)
```

And the librarian's alternative-list builder:

```python
# app/recovery/librarian.py:find_alternatives
out.append(Alternative(
    strategy="my_strategy",
    rationale="Why this might work",
    est_cost_usd=0.01,
    est_latency_s=10.0,
    sync=True,
))
```

Add tests in `tests/test_recovery_strategies.py` following the
existing pattern (mocked dependencies + assertion on
`StrategyResult.success`).

---

## 12. Adding a new capability map entry

When a new tool/crew family lands and you want refusals around it to
auto-route correctly:

```python
# app/recovery/librarian.py:_CAPABILITY_MAP
"new_capability": {
    "crews": ["target_crew_name"],
    "tools": ["module.tool_name"],
    "keywords": ("keyword1", "keyword2", "keyword3"),
},
```

Keywords match against the user's task text (lowercased,
case-insensitive substring). Multiple keywords increase recall;
overlap with existing entries is fine — both will produce
alternatives, the loop tries them cheap-first.

If you also want `direct_tool` to handle it:

1. Add the tool to `_TOOL_RECIPES` in `direct_tool.py` with extract
   + format functions.
2. Add the tool name to the eligibility list inside
   `librarian.find_alternatives`:
   ```python
   if tool in ("email_tools.check_email", "calendar_tools.list_events", "module.tool_name"):
       out.append(Alternative(strategy="direct_tool", ...))
   ```

---

## 13. Known limitations

### 13.1 Single-crew dispatch only

The hook fires inside the single-crew branch of `Commander._route()`.
Multi-crew parallel dispatch (when commander routes to 2+ crews
simultaneously) doesn't currently invoke the recovery loop on the
combined output. Workaround: any single crew's refusal-shaped output
gets caught when its individual final answer flows through the
single-crew vetting+critic path.

### 13.2 Manual ticket completions skip the loop

When a user completes a ticket via dashboard PUT
`/api/cp/tickets/{id}` with refusal-shaped text in `result_summary`,
that text is delivered as-is without going through
`Commander.handle()` — recovery doesn't run. Edge case; rare in
practice (manual completions are operator-driven and the operator
controls the text).

### 13.3 Streaming partial results

CrewAI streams partials from `parallel_runner.run_parallel` to Signal
as each sub-crew completes. Those partials don't pass through
recovery. The final synthesised answer (which combines partials) DOES
pass through recovery, but the partials themselves can leak
refusal-shaped fragments. Fix: route partials through recovery too —
deferred because it would multiply the loop's compute footprint.

### 13.4 Tool-availability blindness

`direct_tool` doesn't pre-check whether the tool's prerequisites are
met (e.g. `EMAIL_PASSWORD` set). It optimistically calls the tool
and treats an "Error: no email config" response as a soft failure
that lets the loop fall through. Sub-optimal because it wastes a
turn — but cheap (no LLM call), so not worth pre-flighting.

### 13.5 Single-skill `skill_chain`

Multi-skill chaining (run skill A, pass output to skill B) is
deferred. The first-cut surfaces ONE matching skill. In practice
that's enough for "remember how we solved this before" cases. Multi-
skill orchestration will be added when this strategy gets used enough
to justify the complexity.

### 13.6 Confidence threshold is global

`RECOVERY_DETECTION_THRESHOLD` applies to all categories uniformly.
A future iteration might want per-category thresholds (e.g.
`missing_tool` fires at 0.7, `generic` at 0.85) for finer control.
Today: one knob.

### 13.7 No recovery for non-refusal failures

The loop only runs on refusal-shaped output. Other failure modes
(timeouts, errors, empty responses) have their own handling layers
(vetting fallback, credit failover, etc.) and aren't double-wrapped.

---

## 14. Future work

| Idea | Status | Notes |
|---|---|---|
| Multi-skill chain orchestration | Deferred | Wait for usage data |
| Per-category confidence thresholds | Backlog | One knob enough for now |
| Streaming-partials recovery | Backlog | Triples compute, low ROI |
| Recovery on dashboard ticket completions | Backlog | Edge case |
| Auto-tune threshold from audit data | Idea | After 1 month of audit data accumulates, fit a precision/recall curve and recommend a threshold |
| `code_compose` strategy | Idea | Take a code-shaped task, generate code via LLM, run via sandbox_execute, return output. Composition of two existing strategies — would benefit from shared helpers. |
| `web_lookup` strategy | Idea | When `data_unavailable` fires, escalate to firecrawl or browser_fetch with a rephrased query. |
| Recovery dashboard panel | Idea | Show `actor='recovery_loop'` audit entries grouped by category + strategy, with success rates, on a dashboard tab. |

---

## 15. Code-pointer index

For rapid navigation:

| Concept | File:line |
|---|---|
| Public API | `app/recovery/__init__.py` |
| `RefusalSignal` dataclass | `app/recovery/refusal_detector.py:36` |
| `_PHRASES_BY_CATEGORY` catalog | `app/recovery/refusal_detector.py:47` |
| Density check | `app/recovery/refusal_detector.py:_refusal_density` |
| Force mode | `app/recovery/refusal_detector.py:detect_refusal` (force= kwarg) |
| `Alternative` dataclass | `app/recovery/librarian.py:25` |
| `_CAPABILITY_MAP` | `app/recovery/librarian.py:52` |
| `find_alternatives` | `app/recovery/librarian.py:find_alternatives` |
| `StrategyResult` | `app/recovery/strategies/__init__.py:24` |
| `direct_tool` strategy | `app/recovery/strategies/direct_tool.py:execute` |
| `direct_tool` recipes | `app/recovery/strategies/direct_tool.py:_TOOL_RECIPES` |
| `sandbox_execute` strategy | `app/recovery/strategies/sandbox_execute.py:execute` |
| `sandbox_execute` pre-flight | `app/recovery/strategies/sandbox_execute.py:_is_runnable_in_sandbox` |
| `re_route` strategy | `app/recovery/strategies/re_route.py:execute` |
| `skill_chain` strategy | `app/recovery/strategies/skill_chain.py:execute` |
| `escalate_tier` strategy | `app/recovery/strategies/escalate_tier.py:execute` |
| `forge_queue` strategy | `app/recovery/strategies/forge_queue.py:execute` |
| Frequency tracker | `app/recovery/strategies/forge_queue.py:_record_and_count` |
| Auto-forge queue write | `app/recovery/strategies/forge_queue.py:_queue_for_forge` |
| Loop orchestrator | `app/recovery/loop.py:maybe_recover` |
| Strategy dispatcher | `app/recovery/loop.py:_execute_strategy` |
| Recursion guard | `app/recovery/loop.py:_in_recovery` (ContextVar) |
| Env-flag gate | `app/recovery/loop.py:is_enabled` |
| Audit logger | `app/recovery/loop.py:_audit` |
| Orchestrator hook | `app/agents/commander/orchestrator.py` (post-vetting block) |
| `force this` Signal command | `app/agents/commander/commands.py:_handle_force_recover` |
| Force-pattern recognition | `app/agents/commander/commands.py:_FORCE_PATTERNS` |
| Tests — Phase 1 (loop core) | `tests/test_recovery_loop.py` (22 tests) |
| Tests — Phase 2 (strategies) | `tests/test_recovery_strategies.py` (32 tests) |

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Refusal-shaped** | Response text that says "I cannot…" / "no access to…" / "<unavailable>" — the system gave up rather than tried |
| **Capability-locality failure** | Refusal where the system *had* the capability, but the agent that picked up the request couldn't reach it |
| **Density** | Fraction of response text that consists of refusal phrases (used to suppress false positives) |
| **Strategy** | A single recovery attempt module; takes (task, alt, ctx), returns StrategyResult |
| **Alternative** | A typed description of one strategy + its parameters, returned by the librarian |
| **Recipe** | (direct_tool only) A typed entry mapping tool_name → (param_extractor, output_formatter) |
| **Gap key** | Stable identifier used by the frequency tracker — `<category>::<normalized_task>` |
| **Force** | User-driven bypass of policy guard + density check + threshold (Signal command "force this") |
| **Route changed** | The recovery materially altered the answer's source (different crew, different tier, direct tool call) — triggers a `_(note)_` annotation in the delivered answer |

---

## 17. Composition with the Tool Supervisor

The Tool Supervisor (`app/tool_runtime/supervisor.py`) is a sibling
layer that handles a different failure shape. **Activation is
double-gated**: the supervisor wraps `available_functions` only
inside `LoadableAgentExecutor`, so it fires only when *both*
`TOOL_SUPERVISOR_ENABLED=true` *and* the calling agent runs on the
LoadableAgent path (`LOADABLE_<AGENT>=1` or master
`LOADABLE_AGENT_EXPERIMENTAL=1`). Agents on the standard CrewAI
executor bypass the supervisor regardless of the flag.

When both gates are open, the two layers compose without overlap:

| Layer | When it fires | What kind of failure | Surface |
|---|---|---|---|
| **Tool Supervisor** | mid-iteration, around each tool call inside `_handle_native_tool_calls` | Raised exception during tool dispatch (`RateLimitError`, `ConnectionError`, `TimeoutError`, schema-validation, etc.) | Wraps `available_functions` in `LoadableAgentExecutor` |
| **Recovery Loop** | post-vetting, after the final answer is composed | Refusal-*shaped* final answer (the agent gave up in prose, no exception) | Hooks `Commander._route` |

A single user request can trigger both. Example timeline:

1. Agent calls `geocode_tool` mid-iteration → raises `TimeoutError`.
2. **Supervisor** catches, classifies as `timeout`, exp-backoff retries
   twice, both fail. Looks up registry alternatives — none. Emits
   `invocation.gave_up` audit event and returns a structured
   `[tool-supervisor] tool 'geocode_tool' failed after 3 attempt(s)`
   string back into the agent's loop.
3. Agent reads the soft-fail observation, decides it can't proceed, and
   replies `"I'm unable to look up that location right now."`
4. **Recovery Loop** detects the refusal-shaped answer, dispatches
   `re_route` strategy → different crew has a working geocode path →
   delivers the corrected answer with `_(note: re-routed via …)_`.

Two safeties keep the layers from fighting:

* **Recursion guard.** The Supervisor sets `_in_substitute = True`
  before calling a substitute tool. Substitute tools therefore run
  *unsupervised* — no nested retry/substitute on a substitute.
  Mirrors the Recovery Loop's `_in_recovery` ContextVar.
* **No double-recovery on substitution.** If a substitute *also* fails,
  the Supervisor returns a soft-fail string (not an exception). The
  agent's downstream prose may then be refusal-shaped, at which point
  the Recovery Loop takes over — but the Supervisor never re-enters.

### Code pointers

* `app/tool_runtime/supervisor.py` — supervisor module.
* `app/tool_runtime/loadable_executor.py:[SUP-1]` — the two wiring
  sites in the sync path (initial render + dirty re-render).
* Audit query: `/api/cp/audit?actor=tool_supervisor`.
* See PROGRAM.md §14 for the change-log entry; the Supervisor
  exists as the *Track A* half of the May 2026 self-healing pass.

---

## See also

| Document | Why |
|---|---|
| `docs/LLM_SUBSYSTEM.md` | The selector, factory, and vetting layers that produce the response we recover |
| `docs/SELF_IMPROVEMENT.md` | The skill-forge that consumes `forge_queue`'s entries |
| `docs/MEMORY_ARCHITECTURE.md` | Where `search_skills` (used by `skill_chain`) reads from |
| `docs/CONTROL_PLANES.md` | The audit log this layer writes to |
| `docs/ERROR_MONITOR.md` §11 | The Runbook Dispatcher — Track B's sibling to the Tool Supervisor |
| `app/agents/commander/orchestrator.py` (Commander._route) | The host that calls `maybe_recover` |
