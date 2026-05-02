"""
base_crew.py — Shared crew execution logic with tool plugin registry.

Tool Plugin Registry:
    register_tool_plugin(factory_fn) — called once per tool source at import
    get_plugin_tools() — returns all plugin tools, cached per-process

    MCP tools, browser tools, and any future tool sources register here.
    All agents get them automatically. No per-agent file modification.

Auto-Skill Creation:
    Complex crew executions (>= _SKILL_CREATION_THRESHOLD tool calls) trigger
    a background distillation into a reusable SkillDraft. The draft is routed
    through the standard Integrator so novelty checking still applies.
"""

import logging
import threading
import time as _time
from pathlib import Path

from crewai import Task, Crew, Process

from app.config import get_settings
# Lifecycle-related sinks (crew_started/completed/failed, belief state,
# completion-time metric) are now reached via app.crews.lifecycle's
# context manager.  ``record_metric`` is still imported directly because
# it's used here for the refusal-retry counters, which are NOT
# envelope-level (they fire mid-body, not at the outer boundary).
from app.benchmarks import record_metric
from app.llm_selector import difficulty_to_tier
from app.sanitize import wrap_user_input
from app.self_heal import diagnose_and_fix

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Tool Plugin Registry ──────────────────────────────────────────────────────

_tool_plugins: list = []  # list of Callable[[], list[BaseTool]]
_plugin_tools_cache: list | None = None
_plugin_lock = threading.Lock()


def register_tool_plugin(factory) -> None:
    """Register a tool factory function. All agents get these tools automatically.

    Factory must return a list of CrewAI tool instances. Called lazily on
    first crew execution (not at registration time).
    """
    global _plugin_tools_cache
    with _plugin_lock:
        _tool_plugins.append(factory)
        _plugin_tools_cache = None  # Invalidate cache


# Ordered priority list for tool capping (highest = drop last).
# When an agent's tool count exceeds MAX_TOOLS_PER_AGENT (default 18, max
# 20 due to Anthropic's strict-tools limit), tools with lower priority
# are pruned first.  Core capabilities — code execution, web search,
# file I/O, memory, knowledge base — always survive.  Peripheral
# introspection/tension/experiential tools drop first.
_TOOL_PRIORITY_ORDER = (
    # Priority 100 — core execution / I/O  (never drop these)
    ("execute_code", 100),
    ("file_manager", 100),
    ("read_attachment", 100),
    ("send_email", 100),
    # Priority 90 — primary information retrieval
    ("web_search", 90),
    ("web_fetch", 90),
    ("search_knowledge_base", 90),
    ("get_youtube_transcript", 90),
    # Firecrawl — structured scraping/extraction of public data sources
    # (stat.ee, FAO, Global Forest Watch etc.).  Without these at high
    # priority, Anthropic's 25-tool cap drops them → researcher falls back
    # to training data and tells the user "I can't access live sources."
    # Kept at 90 so they always survive alongside web_search.
    ("firecrawl_scrape", 90),
    ("firecrawl_search", 90),
    ("firecrawl_extract", 88),
    ("firecrawl_crawl", 86),
    ("firecrawl_map", 84),
    # Google Earth Engine — server-side raster compute for satellite-
    # imagery analysis (Hansen GFC, Sentinel-2/Landsat, NDVI, MODIS,
    # GEDI lidar).  Same logic as firecrawl: without an explicit slot
    # it defaults to priority 10 and the cap silently drops it (Phase 3
    # of the 2026-05-02 audit caught this — coder had 35 tools, 25
    # cap, GEE in the cull zone).  Pinned at 88 to survive alongside
    # firecrawl_extract.
    ("gee_run_script", 88),
    # Priority 80 — memory (one mem0 + one scoped-memory is enough)
    ("mem0_add", 80),
    ("mem0_search", 80),
    ("scoped_memory_add", 80),
    ("scoped_memory_search", 80),
    ("team_memory_store", 70),
    ("team_memory_retrieve", 70),
    # Priority 70 — calendar / email / pim core
    ("check_email", 70),
    ("search_email", 70),
    ("list_calendar_events", 70),
    ("create_calendar_event", 70),
    ("list_tasks", 70),
    ("create_task", 70),
    # Priority 68 — knowledge stores with NO surviving equivalent.
    # search_research_knowledge hits the Episteme ChromaDB collection and
    # search_journal hits the Journal collection — neither is reachable via
    # search_knowledge_base (enterprise collection) or any other retained
    # tool.  Dropping these would remove the agent's proactive access to
    # the system's research memory and its experiential history during a
    # task.  (Automatic post-task hooks still WRITE to both collections
    # regardless — this priority just preserves READ access.)
    ("search_research_knowledge", 68),
    ("search_journal", 68),
    # Priority 65 — MCP manager (needed for dynamic tool discovery)
    ("mcp_search_servers", 65),
    ("mcp_list_servers", 65),
    # Priority 60 — OCR / media
    ("ocr_extract_text", 60),
    ("read_pdf", 60),
    # Priority 55 — browser / host bridge
    ("browser_fetch", 55),
    ("execute_on_host", 55),
    ("http_from_host", 55),
    # Priority 50 — wiki (read OK, write is peripheral)
    ("wiki_read", 55),
    ("wiki_search", 55),
    ("wiki_write", 40),
    ("wiki_slides", 30),
    # Priority 58 — writer-essential dialectics.  These are peripheral for
    # coder/researcher but core for writer — writer tasks that benefit from
    # counter-arguments or conceptual reframing need these.  Ranked ABOVE
    # bridge tools (55) so the writer's 29-tool list preserves them while
    # dropping 2 bridge tools (which writers rarely use anyway).
    ("philosophy_knowledge_base", 58),
    ("find_counter_argument", 58),
    ("conceptual_blend", 58),
    # Priority 35 — introspective surface the agent can *call* during a task
    # (post-task telemetry writes experiential/tension entries automatically
    # via lifecycle hooks regardless, so the search tools matter more than
    # the record ones).
    ("search_tensions", 35),
    ("search_experiential", 35),
    ("search_aesthetic", 30),
    ("record_tension", 25),
    ("record_experience", 25),
    ("record_aesthetic", 25),
    ("self_report", 20),
    # Priority 10 — plugin tools / anything unrecognized (default)
)

_TOOL_PRIORITY_MAP = {name: prio for name, prio in _TOOL_PRIORITY_ORDER}


def _tool_priority(tool) -> int:
    """Return priority 0-100 for a tool.  Unknown names default to 10."""
    name = getattr(tool, "name", "") or ""
    return _TOOL_PRIORITY_MAP.get(name, 10)


# Provider-specific tool caps.  Anthropic has the tightest limits (20 strict
# tools, then separately a schema-complexity budget ≈ 25-40 non-strict).
# OpenAI-compatible providers (Grok, DeepSeek, Kimi, MiniMax, Step, Gemini)
# allow ~128 tools per request with much looser schema complexity.  Local
# Ollama is limited by the model itself but typically unbounded.
#
# These caps are SAFETY margins below each provider's hard limit.  Going
# higher than the cap burns tokens on every system prompt and degrades LLM
# decision quality without meaningful capability gains.
_PROVIDER_TOOL_CAPS = {
    "anthropic": 25,
    "openai": 50,
    "openrouter": 50,   # OpenRouter fronts many providers — all OpenAI-compatible
    "xai": 50,
    "google": 50,       # Gemini
    "gemini": 50,
    "deepseek": 50,
    "minimax": 50,
    "moonshot": 50,
    "stepfun": 50,
    "mistral": 50,
    "meta-llama": 50,
    "ollama": 50,       # Local (nominally unbounded, but keep focused)
    "ollama_chat": 50,
}


def _detect_llm_provider(llm) -> str:
    """Return the provider key (anthropic, openrouter, xai, ollama, …) for
    an agent's LLM instance.  Inspects .model or .model_id first, falls
    back to .provider, ultimately returns 'anthropic' (the most restrictive
    cap) on unknown so we never accidentally exceed a provider's limit."""
    if llm is None:
        return "anthropic"
    for attr in ("model", "model_id"):
        raw = getattr(llm, attr, "") or ""
        if raw:
            raw_lower = str(raw).lower()
            # Split on '/' — first segment is typically the provider
            head = raw_lower.split("/", 1)[0]
            if head in _PROVIDER_TOOL_CAPS:
                return head
            # Some model_ids don't have a provider prefix (e.g. "gpt-4o")
            if "gpt" in head or "o1-" in head:
                return "openai"
            if "claude" in head:
                return "anthropic"
            if "gemini" in head:
                return "google"
            if "grok" in head:
                return "xai"
    prov = getattr(llm, "provider", "") or ""
    if prov and str(prov).lower() in _PROVIDER_TOOL_CAPS:
        return str(prov).lower()
    return "anthropic"  # safest fallback


def _cap_for_llm(llm) -> int:
    """Tool-count cap appropriate to the agent's LLM provider."""
    import os as _os
    # Explicit env override wins over per-provider defaults
    override = _os.environ.get("MAX_TOOLS_PER_AGENT")
    if override and override.isdigit():
        return int(override)
    provider = _detect_llm_provider(llm)
    return _PROVIDER_TOOL_CAPS.get(provider, 25)


def _cap_tools_by_priority(tools: list, cap: int) -> list:
    """Return at most *cap* tools, keeping highest-priority ones.

    Stable within priority bucket: if two tools share a priority,
    the one listed first in *tools* wins.  This lets a factory
    declare its preferred order and have it respected among peers.
    """
    if len(tools) <= cap:
        return tools
    # Sort by (priority DESC, original index ASC).  Then slice.
    indexed = list(enumerate(tools))
    indexed.sort(key=lambda it: (-_tool_priority(it[1]), it[0]))
    kept = [t for _, t in indexed[:cap]]
    # Preserve the caller's original order for the survivors
    survivor_set = {id(t) for t in kept}
    return [t for t in tools if id(t) in survivor_set]


def get_plugin_tools() -> list:
    """Collect tools from all registered plugins. Cached after first call."""
    global _plugin_tools_cache
    if _plugin_tools_cache is not None:
        return _plugin_tools_cache
    with _plugin_lock:
        if _plugin_tools_cache is not None:
            return _plugin_tools_cache
        tools = []
        for factory in _tool_plugins:
            try:
                result = factory()
                if result:
                    tools.extend(result)
            except Exception:
                logger.warning(f"Tool plugin failed: {factory}", exc_info=True)
        _plugin_tools_cache = tools
        logger.info(f"Tool plugin registry: {len(tools)} tools from {len(_tool_plugins)} plugins")
        return tools


# ── Tool Manifest (Tool-First affordance) ──────────────────────────────────

_TOOL_MANIFEST_MARKER = "## Your Tools (Tool-First Protocol)"
_TOOL_MANIFEST_MAX = 40  # cap so the backstory doesn't explode past context limits


def _append_tool_manifest(backstory: str, tools: list) -> str:
    """Append a concise list of tool names + one-liners to the backstory.

    Idempotent — detects the marker and refreshes rather than duplicating.
    Keeps descriptions short so the system prompt stays lean.
    """
    if not tools:
        return backstory
    entries = []
    for t in tools[:_TOOL_MANIFEST_MAX]:
        name = getattr(t, "name", "")
        if not name:
            continue
        desc = (getattr(t, "description", "") or "").strip()
        # Collapse whitespace + trim to first sentence / 100 chars
        desc = " ".join(desc.split())[:120]
        entries.append(f"- `{name}` — {desc}")
    if not entries:
        return backstory
    extra = len(tools) - _TOOL_MANIFEST_MAX
    extra_note = f"\n  (+{extra} more — see function-calling schema)" if extra > 0 else ""
    manifest = (
        f"\n\n{_TOOL_MANIFEST_MARKER}\n"
        "You have these tools attached RIGHT NOW. Scan this list before answering any request.\n"
        "If ANY entry plausibly maps to the user's ask, CALL IT before saying you cannot. "
        "Refusing without a tool attempt violates your operating protocol.\n\n"
        + "\n".join(entries)
        + extra_note
    )
    # Idempotent refresh: strip any previous manifest block
    if _TOOL_MANIFEST_MARKER in backstory:
        backstory = backstory.split(_TOOL_MANIFEST_MARKER)[0].rstrip()
    return backstory + manifest


# ── Refusal Detection (Tool-First enforcement) ──────────────────────────────
# When a specialist answers with a refusal phrase AND has tools AND hasn't
# actually called any, we retry once with an explicit nudge listing available
# tool names. This captures the "LLM defaults to 'I can't' despite having
# working tools" failure mode that users observe most often.

_REFUSAL_PATTERNS = (
    "i don't have access",
    "i do not have access",
    "i can't help",
    "i cannot help",
    "i'm unable to",
    "i am unable to",
    "i cannot do that",
    "i can't do that",
    "i don't have the ability",
    "i do not have the ability",
    "i'm not able to",
    "i am not able to",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i apologize, but i can't",
    "i apologize, but i cannot",
    "unfortunately, i can't",
    "unfortunately, i cannot",
    "as an ai",
    "i don't have real-time",
    "i don't have the capability",
)

_REFUSAL_MAX_LEN = 2000  # only scan the first N chars — long answers are rarely refusals


def _looks_like_refusal(result: str) -> bool:
    """Heuristic: does this response refuse to act without having called a tool?"""
    if not result:
        return False
    head = result[:_REFUSAL_MAX_LEN].lower()
    # If the agent clearly called tools (Action: / Observation: markers present),
    # it's not a pre-emptive refusal — it's a reasoned one after trying.
    if "observation:" in head or "action:" in head:
        return False
    return any(p in head for p in _REFUSAL_PATTERNS)


def _build_retry_prompt(original_task: str, agent_tools: list, refusal_text: str) -> str:
    """Compose an explicit re-prompt listing the tools the agent already has."""
    tool_lines = []
    for t in agent_tools[:25]:
        name = getattr(t, "name", "")
        desc = (getattr(t, "description", "") or "")[:120]
        if name:
            tool_lines.append(f"  - `{name}`: {desc}")
    tool_list = "\n".join(tool_lines) or "  (tool list empty — this is a bug)"
    return (
        "Your previous response refused to act, but you DO have tools that can help. "
        "Retry the task — this time call at least one of your tools before answering.\n\n"
        f"Original task:\n{original_task}\n\n"
        f"Your tools (call one of these):\n{tool_list}\n\n"
        f"What you said last time (don't repeat this):\n{refusal_text[:400]}\n\n"
        "Tool-First protocol: try the most plausible tool, inspect the output, chain if needed, "
        "and synthesise an answer from what the tools actually returned. Refusing again without "
        "a tool call is not acceptable."
    )


# ── Auto-Skill Creation ──────────────────────────────────────────────────────

_SKILL_CREATION_THRESHOLD = 5  # Minimum tool calls to trigger skill creation
_SKILL_EXCLUDED_CREWS = {"self_improvement", "retrospective", "critic"}


def _estimate_tool_calls(result: str) -> int:
    """Estimate tool call count from crew output text."""
    text = str(result)
    # CrewAI outputs "Observation:" after each tool call
    count = text.count("Observation:")
    if count == 0:
        # Fallback: "Action:" markers
        count = text.count("Action:")
    if count == 0 and len(text) > 2000:
        # Heuristic for long results without markers
        count = _SKILL_CREATION_THRESHOLD
    return count


def _auto_create_skill(
    crew_name: str,
    task: str,
    result: str,
    tool_calls: int,
    task_id: str = "",
) -> None:
    """Background: distill a complex crew execution into a reusable skill.

    Vetting-gated.  Phase 3 of the 2026-05-02 audit found this function
    persists skills distilled from FAILED dispatches, polluting the
    experiential KB and biasing future retrievals.  The fix: before
    handing the draft to ``integrate``, check the vetting outcome the
    orchestrator recorded for ``task_id`` via
    ``app.crews.events.set_vetting_outcome``.

    Outcome semantics:
      * vetting passed   → integrate the draft (existing behaviour)
      * vetting failed   → drop the draft (don't pollute the KB)
      * outcome unknown  → drop the draft (conservative — refusing to
                           act on uncertainty is the right default for
                           a persistence sink)

    Backward-compatible: callers that don't pass ``task_id`` get the
    unknown-outcome path, which drops the draft.  This matches the
    desired behaviour for legacy code that may have called this
    function without telemetry — better to lose the skill than pollute.
    """
    try:
        from app.llm_factory import create_specialist_llm
        from app.self_improvement.types import SkillDraft
        from app.self_improvement.integrator import integrate as integrate_draft
        from app.crews.events import get_vetting_outcome
        import uuid

        llm = create_specialist_llm(max_tokens=800, role="synthesis")
        prompt = (
            f"A {crew_name} crew completed a complex task ({tool_calls} tool calls).\n\n"
            f"Task: {task[:500]}\n\nResult excerpt: {result[:1000]}\n\n"
            f"Distill into a reusable SKILL:\n"
            f"1. Topic (one line)\n2. When to use\n3. Procedure (max 5 steps)\n"
            f"4. Pitfalls\n\nMax 300 words."
        )
        skill_text = str(llm.call(prompt)).strip()
        if not skill_text or len(skill_text) < 50:
            return

        # ── Vetting gate (Week 1 audit fix for H6) ──
        # By the time the LLM call above finishes (~10-30s), the
        # orchestrator has typically completed vetting and recorded
        # the verdict via set_vetting_outcome(crew_name, passed).
        # The registry is keyed by crew_name (not task_id) — see
        # app/crews/events.py for why.  Any other state (failed /
        # unknown) drops the draft to avoid KB pollution.
        passed = get_vetting_outcome(crew_name) if crew_name else None
        if passed is not True:
            verdict = "failed" if passed is False else "unknown"
            logger.info(
                f"Auto-skill creation skipped: vetting {verdict} for "
                f"{crew_name} (task_id={task_id or '(unset)'})"
            )
            return

        lines = skill_text.strip().split("\n")
        topic = lines[0].replace("Topic:", "").replace("#", "").strip()[:100] or task[:80]

        draft = SkillDraft(
            id=f"auto_{uuid.uuid4().hex[:8]}",
            topic=topic,
            rationale=f"Auto-captured from {crew_name} ({tool_calls} tool calls)",
            content_markdown=skill_text,
            proposed_kb="experiential",
        )
        integrate_draft(draft)
        logger.info(f"Auto-skill created: '{topic}' from {crew_name}")
    except Exception:
        logger.debug("Auto-skill creation failed", exc_info=True)


# ── Core Crew Execution ──────────────────────────────────────────────────────

def run_single_agent_crew(
    crew_name: str,
    agent_role: str,
    create_agent_fn,
    task_template: str,
    task_description: str,
    expected_output: str,
    parent_task_id: str = None,
    difficulty: int = 5,
    extra_tools: list = None,
) -> str:
    """Run a single-agent crew with all standard boilerplate.

    Args:
        crew_name: Firebase crew name (e.g. "coding", "writing")
        agent_role: Belief state role (e.g. "coder", "writer")
        create_agent_fn: Factory function (force_tier) → Agent
        task_template: Template string with {user_input} placeholder
        task_description: The user's task (gets wrapped in template)
        expected_output: Expected output description for CrewAI
        parent_task_id: Optional parent task for sub-agent tracking
        difficulty: Task difficulty (1-10)
        extra_tools: additional tools specific to this crew (on top of plugin tools).

    Returns:
        The crew's output as a string.
    """
    from app.llm_mode import get_mode
    from app.crews.lifecycle import crew_lifecycle

    force_tier = difficulty_to_tier(difficulty, get_mode())
    agent = create_agent_fn(force_tier=force_tier)

    # Inject plugin tools (MCP, browser, etc.) into the agent.
    # NOTE: The monkey-patched Agent.__init__ already injects these, so
    # only add tools whose names aren't already present (avoids _2 duplicates).
    plugin_tools = get_plugin_tools()
    if plugin_tools:
        existing = list(agent.tools) if agent.tools else []
        existing_names = {getattr(t, "name", "") for t in existing}
        new_plugins = [t for t in plugin_tools if getattr(t, "name", "") not in existing_names]
        if new_plugins:
            agent.tools = existing + new_plugins
    if extra_tools:
        existing = list(agent.tools) if agent.tools else []
        agent.tools = existing + extra_tools

    # Re-cap after late additions.  Provider-aware — agents on Anthropic
    # get 25, on Grok/OpenAI/etc. get 50.
    _cap = _cap_for_llm(getattr(agent, "llm", None))
    if agent.tools and len(agent.tools) > _cap:
        before = len(agent.tools)
        agent.tools = _cap_tools_by_priority(agent.tools, _cap)
        provider = _detect_llm_provider(getattr(agent, "llm", None))
        logger.warning(
            f"{crew_name} [{provider}]: capped tools {before} → {len(agent.tools)} "
            f"(provider cap={_cap})"
        )

    # Extract model name from the agent's LLM for the task record
    _model_name = ""
    try:
        _model_name = getattr(agent.llm, 'model', '') if getattr(agent, 'llm', None) else ''
    except Exception:
        pass

    # Strip injected context from the task summary shown on dashboard.
    _clean_desc = task_description
    for _ctx_marker in ("KNOWLEDGE BASE CONTEXT", "RELEVANT KNOWLEDGE", "RELEVANT TEAM CONTEXT",
                        "<recent_conversation>", "LESSONS FROM PAST"):
        _idx = _clean_desc.find(_ctx_marker)
        if _idx == 0:
            for _end_marker in ("\n\nNOTE:", "\n</", "\nNOTE:"):
                _end = _clean_desc.rfind(_end_marker)
                if _end > 0:
                    _rest = _clean_desc[_end:].split("\n\n", 2)
                    if len(_rest) > 1:
                        _clean_desc = _rest[-1].strip()
                        break
            break

    # Lifecycle envelope — handles crew_started/belief/metric/journal/
    # completed-or-failed/auto-skill uniformly.  Everything between
    # ``with`` and ``return`` is the crew's actual work.
    with crew_lifecycle(
        crew_name=crew_name,
        agent_role=agent_role,
        task_title=f"{crew_name.title()}: {_clean_desc[:100]}",
        task_description=task_description,
        parent_task_id=parent_task_id,
        model=_model_name,
    ) as _ctx:
        task = Task(
            description=task_template.format(user_input=wrap_user_input(task_description)),
            expected_output=expected_output,
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=settings.crew_verbose,
        )

        try:
            result = str(crew.kickoff())
        except Exception as exc:
            # Non-lifecycle side-effect: error-journal diagnosis.  Must
            # fire BEFORE the lifecycle manager's failure path re-raises,
            # so the diagnose_and_fix pipeline sees the original exception
            # with the right task_id (so subsequent retries attach to
            # the same parent).
            diagnose_and_fix(crew_name, task_description, exc, task_id=_ctx.task_id)
            raise

        # Tool-First enforcement: if the agent refused without calling tools, retry once
        # with an explicit nudge listing the tools it has.
        if _looks_like_refusal(result) and agent.tools:
            retry_prompt = _build_retry_prompt(task_description, agent.tools, result)
            logger.info(
                f"base_crew: refusal detected in {crew_name}, retrying with tool-first nudge "
                f"({len(agent.tools)} tools available)"
            )
            try:
                retry_task = Task(
                    description=retry_prompt,
                    expected_output=expected_output,
                    agent=agent,
                )
                retry_crew = Crew(
                    agents=[agent],
                    tasks=[retry_task],
                    process=Process.sequential,
                    verbose=settings.crew_verbose,
                )
                retry_result = str(retry_crew.kickoff())
                if retry_result and not _looks_like_refusal(retry_result):
                    result = retry_result
                    record_metric("refusal_retry_success", 1.0, {"crew": crew_name})
                else:
                    record_metric("refusal_retry_failed", 1.0, {"crew": crew_name})
            except Exception:
                logger.debug(f"base_crew: refusal retry in {crew_name} crashed", exc_info=True)

        _ctx.set_outcome(result)
        return result


# ── Plugin auto-registration at import time ──────────────────────────────────
# These execute only when the factories are first called (lazy), not at import.

def _register_default_plugins() -> None:
    """Register built-in tool plugins. Called once from main.py startup.

    Also patches crewai.Agent so every Agent instance — including those built
    by multi-agent crews that don't use run_single_agent_crew — gets plugin
    tools auto-appended at construction time.
    """
    # MCP tools
    register_tool_plugin(
        lambda: __import__("app.mcp.tool_adapter", fromlist=["create_crewai_tools"]).create_crewai_tools()
    )
    # Browser tools
    register_tool_plugin(
        lambda: __import__("app.tools.browser_tools", fromlist=["create_browser_tools"]).create_browser_tools()
    )
    # Session search tool
    register_tool_plugin(
        lambda: __import__("app.tools.session_search_tool", fromlist=["create_session_search_tools"]).create_session_search_tools()
    )
    # MCP manager tools (search/add/list/remove MCP servers — self-service)
    register_tool_plugin(
        lambda: __import__("app.tools.mcp_manager_tool", fromlist=["create_mcp_manager_tools"]).create_mcp_manager_tools()
    )

    # Patch crewai.Agent so every agent instance gets plugin tools automatically
    _patch_agent_for_plugins()


_agent_patched = False


def _patch_agent_for_plugins() -> None:
    """Monkey-patch crewai.Agent so every Agent auto-appends plugin tools.

    Hooks the constructor so multi-agent crews (research / media / critic /
    creative / etc.) that construct crewai.Agent directly — without going
    through run_single_agent_crew — still get plugin tools.

    Injection happens BEFORE original __init__ runs, so CrewAI's
    field_validator("tools") processes plugin tools uniformly (wrapping
    langchain-style tools, validating BaseTool subclasses). Post-init
    assignment to self.tools bypasses that validator and caused subtle bugs
    where dynamically-added MCP tools weren't picked up by the executor.
    """
    global _agent_patched
    if _agent_patched:
        return
    try:
        from crewai import Agent
    except ImportError:
        logger.debug("crewai not importable; skipping Agent plugin patch")
        return

    original_init = Agent.__init__

    def patched_init(self, *args, **kwargs):
        # Pre-init injection — extend kwargs["tools"] before pydantic validates.
        try:
            plugins = get_plugin_tools()
            if plugins:
                existing = list(kwargs.get("tools") or [])
                existing_names = {getattr(t, "name", "") for t in existing}
                additions = [
                    t for t in plugins
                    if getattr(t, "name", "") and getattr(t, "name", "") not in existing_names
                ]
                if additions:
                    kwargs["tools"] = existing + additions
                    logger.debug(
                        f"Agent('{kwargs.get('role', '?')}') + {len(additions)} plugin tools: "
                        f"{[getattr(t, 'name', '?') for t in additions]}"
                    )
        except Exception:
            logger.debug("Agent plugin pre-init failed (non-fatal)", exc_info=True)

        # Provider-aware tool-count cap.  Different LLM providers have
        # different hard limits:
        #   Anthropic: 20 strict tools, ≈25-40 non-strict (schema complexity)
        #   OpenAI/Grok/Gemini/DeepSeek/Kimi/MiniMax: ~128 tools per request
        #   Ollama: unbounded but still capped for token efficiency
        # _cap_for_llm() picks the right cap by inspecting the agent's LLM.
        # Result: Anthropic-bound agents get 25 tools; Grok/OpenAI-bound
        # agents get 50, which fits every peripheral tool (philosophy,
        # tensions, experiential, aesthetic, self_report, etc.) comfortably
        # even with all MCP plugin tools attached.
        try:
            tools = list(kwargs.get("tools") or [])
            llm = kwargs.get("llm")
            max_tools = _cap_for_llm(llm)
            if len(tools) > max_tools:
                tools_after = _cap_tools_by_priority(tools, max_tools)
                dropped = len(tools) - len(tools_after)
                dropped_names = [
                    getattr(t, "name", "?") for t in tools
                    if t not in tools_after
                ][:8]
                provider = _detect_llm_provider(llm)
                logger.warning(
                    f"Agent('{kwargs.get('role', '?')}' [{provider}]): "
                    f"capped tools {len(tools)} → {len(tools_after)} "
                    f"(dropped {dropped}: {dropped_names})"
                )
                kwargs["tools"] = tools_after
        except Exception:
            logger.debug("Agent tool-cap pre-init failed (non-fatal)", exc_info=True)

        # Tool-First affordance: append a short manifest of available tools to the
        # backstory so the LLM sees "I have tools X, Y, Z — USE THEM" in its system
        # prompt. CrewAI already lists tool schemas for function-calling, but this
        # second-person narrative is much stickier for refusal-prone LLMs.
        # Only modify `backstory` if the caller already supplied one — don't add
        # a kwarg the callee doesn't accept.
        try:
            tools = kwargs.get("tools") or []
            if tools and "backstory" in kwargs and kwargs["backstory"]:
                kwargs["backstory"] = _append_tool_manifest(kwargs["backstory"], tools)
        except Exception:
            logger.debug("Agent tool-manifest injection failed (non-fatal)", exc_info=True)

        original_init(self, *args, **kwargs)

    Agent.__init__ = patched_init
    _agent_patched = True
    logger.info("crewai.Agent patched for plugin-tool auto-injection (pre-init)")

    # Also patch tool-schema emission so strict=False is sent to Anthropic.
    # Anthropic's tool_use API caps STRICT tools at 20 per request; non-strict
    # tools have a much higher limit (~100).  CrewAI's agent_utils hard-codes
    # strict=True on every tool schema, which forces the cap.  Flipping to
    # strict=False removes Anthropic's cap, so agents can see the full set of
    # peripheral tools (introspection, tensions, experiential, philosophy)
    # without the 18-tool priority cull in patched_init.
    _patch_crewai_strict_false()
    _patch_crewai_anthropic_prefill()


def _patch_crewai_anthropic_prefill() -> None:
    """Ensure the Anthropic `messages` array never ends with an assistant
    message.

    Claude Sonnet 4.5+ (and other newer Anthropic models) reject requests
    where the last message has role='assistant' with:

        400 {"type":"invalid_request_error","message":"This model does not
        support assistant message prefill. The conversation must end with
        a user message."}

    CrewAI's hierarchical mode and reflexion-retry paths occasionally feed
    a prior assistant turn back as the tail of the conversation (to
    provide context for a refinement pass), which triggers this.  Rather
    than fixing every CrewAI call site, we wrap
    `AnthropicCompletion._format_messages_for_anthropic` (the real entry
    point for Anthropic message formatting — NOT the base class's
    `_format_messages` which is only called internally) so it appends a
    minimal user continuation message whenever the tail turns out to be
    assistant.  The continuation is a single space — harmless to the
    model, makes Anthropic's API accept the conversation.
    """
    try:
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion
        _orig_fmt = AnthropicCompletion._format_messages_for_anthropic

        def _safe_fmt(self, messages):
            formatted, system = _orig_fmt(self, messages)
            # If the last message is from assistant, Anthropic (Sonnet 4.5+)
            # rejects the request.  Append a tiny user continuation so the
            # conversation ends with a user turn.
            if formatted and formatted[-1].get("role") == "assistant":
                formatted = list(formatted) + [
                    {"role": "user", "content": " "}
                ]
            return formatted, system

        AnthropicCompletion._format_messages_for_anthropic = _safe_fmt
        logger.info(
            "crewai: Anthropic provider patched — trailing assistant messages "
            "now get a user continuation (prevents 'model does not support prefill' 400)"
        )
    except Exception:
        logger.debug("crewai Anthropic prefill patch failed (non-fatal)",
                     exc_info=True)


def _patch_crewai_strict_false() -> None:
    """Monkey-patch CrewAI's convert_tools_to_openai_schema so every tool is
    emitted with strict=False.  Unlocks Anthropic's higher tool limit.

    NOTE: CrewAI's `crew_agent_executor.py` does
        `from crewai.utilities.agent_utils import convert_tools_to_openai_schema`
    which creates a name binding captured at import time.  Replacing the
    attribute on `crewai.utilities.agent_utils` alone is insufficient — the
    executor keeps calling the original.  We must also rebind the name in
    every module that imported it directly.  (We also dig through
    `sys.modules` to cover any future importers we don't know about yet.)
    """
    try:
        import sys
        import crewai.utilities.agent_utils as _cu
        from app.observability import schema_transforms

        # Install the built-in transforms (strict→False + required-list
        # normalisation) into the shared registry.  Any future provider
        # quirks register additional transforms via the same module; the
        # wrapping function below is provider-agnostic and doesn't need
        # to know what's in the registry.
        schema_transforms.install_default_transforms()

        _original_convert = _cu.convert_tools_to_openai_schema

        def _loose_convert(*args, **kwargs):
            schemas, fns, name_map = _original_convert(*args, **kwargs)
            for s in schemas:
                schema_transforms.apply_to_function_schema(s.get("function"))
            return schemas, fns, name_map

        # (1) Patch the canonical location.
        _cu.convert_tools_to_openai_schema = _loose_convert

        # (2) Rebind in every already-imported module that pulled the name in
        #     via `from ... import convert_tools_to_openai_schema`.
        rebound = 0
        for modname, mod in list(sys.modules.items()):
            if mod is None or mod is _cu:
                continue
            if getattr(mod, "convert_tools_to_openai_schema", None) is _original_convert:
                try:
                    setattr(mod, "convert_tools_to_openai_schema", _loose_convert)
                    rebound += 1
                except Exception:
                    pass

        logger.info(
            "crewai: tool schemas patched (rebound in %d additional modules; "
            "transforms registered: %s)",
            rebound, ", ".join(schema_transforms.registered_names()),
        )
    except Exception:
        logger.debug("crewai strict-false patch failed (non-fatal)", exc_info=True)
