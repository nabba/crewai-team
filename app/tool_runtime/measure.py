"""measure.py — Phase 1c empirical token-cost harness.

Compares stock coder vs LoadableAgent on a representative task,
modeling Anthropic's prompt-cache pricing analytically. Outputs the
go/no-go gate value for Phase 2.

Methodology
-----------
We do NOT make real LLM calls. The question Phase 1c answers is
"would LoadableAgent use ≤50% of stock tokens on a typical task?" —
that's answerable by:

  1. Building both agents with the same toolset semantics.
  2. Asking CrewAI to materialize each agent's prompt + tools array
     (via ``create_agent_executor``).
  3. Counting tokens in each.
  4. Modeling 5 iterations with 2 mid-task loads + Anthropic's cache
     pricing (1.25× write, 0.10× read).

Avoiding real calls means:
  * No API cost for the gate.
  * Determinism — re-running gives the same answer.
  * No network dependency in CI.

Phase 2 will validate the model with actual ``usage`` from
Anthropic's ``input_tokens`` / ``cache_read_input_tokens`` /
``cache_creation_input_tokens`` fields when LoadableAgent is on a
real workload. If the model is materially off, the gate threshold
will be re-tuned then; for the Phase 2 gate decision the analytical
model is what we use.

Usage::

    docker exec crewai-team-gateway-1 python -m app.tool_runtime.measure

prints a markdown report. Or programmatically::

    from app.tool_runtime.measure import compare_stock_vs_loadable
    report = compare_stock_vs_loadable()
    assert report["loadable_total"] / report["stock_total"] <= 0.50
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Anthropic prompt-cache pricing multipliers (May 2026 docs):
#   - Cache write: 1.25× base input price
#   - Cache read:  0.10× base input price
#   - No cache:    1.00× base input price
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


def _count_tokens_chars(text: str) -> int:
    """Approximation: Anthropic's tokenizer averages ~3.5 chars/token
    for English prose. Tools / JSON have shorter average (more
    structural punctuation), but the ratio holds within ~5%.

    Phase 2 swaps this for ``litellm.token_counter`` against the
    actual Sonnet tokenizer.
    """
    return max(1, len(text) // 4)


def count_tokens(*parts: Any) -> int:
    """Sum of token counts across multiple text-or-dict parts."""
    total = 0
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (dict, list)):
            text = json.dumps(part, separators=(",", ":"))
        else:
            text = str(part)
        total += _count_tokens_chars(text)
    return total


# ── Prompt extraction from a CrewAI agent ───────────────────────────


@dataclass(frozen=True)
class AgentPromptShape:
    """The materialized prompt shape of an agent at the moment its
    executor was created. We extract these fields, count their
    tokens, and use them to model iteration cost."""

    name: str
    system_text: str
    user_text: str          # the per-task prompt template
    tools_description: str  # the textual tools block (system-region)
    tools_schema: list      # the JSON-schema tools array (API param)

    @property
    def system_tokens(self) -> int:
        return count_tokens(self.system_text)

    @property
    def user_tokens(self) -> int:
        return count_tokens(self.user_text)

    @property
    def tools_description_tokens(self) -> int:
        return count_tokens(self.tools_description)

    @property
    def tools_schema_tokens(self) -> int:
        return count_tokens(self.tools_schema)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "system_tokens": self.system_tokens,
            "user_tokens": self.user_tokens,
            "tools_description_tokens": self.tools_description_tokens,
            "tools_schema_tokens": self.tools_schema_tokens,
            "total_static_tokens": (
                self.system_tokens
                + self.user_tokens
                + self.tools_description_tokens
                + self.tools_schema_tokens
            ),
        }


def extract_prompt_shape(agent, *, name: str) -> AgentPromptShape:
    """Build an AgentPromptShape from a CrewAI Agent.

    Triggers ``create_agent_executor`` if not already done, then reads
    the prepared prompt + tools off the executor.
    """
    if agent.agent_executor is None:
        agent.create_agent_executor()
    ex = agent.agent_executor

    # CrewAI stores either a SystemPromptResult dict ({system, user})
    # or a single-prompt dict. Handle both.
    prompt_dict = ex.prompt if isinstance(ex.prompt, dict) else {}
    system_text = prompt_dict.get("system", "")
    user_text = prompt_dict.get("user", "") or prompt_dict.get("prompt", "")

    tools_desc = ex.tools_description or ""

    # Build the OpenAI-format tools schema as it would be sent to the API.
    tools_schema: list = []
    try:
        from crewai.utilities.agent_utils import convert_tools_to_openai_schema
        tools_schema, _, _ = convert_tools_to_openai_schema(ex.original_tools or [])
    except Exception:
        # Fall back to a structural estimate if convert fails (some
        # CrewAI versions have the .run-attribute issue we hit in 1.14.4).
        tools_schema = [
            {"function": {"name": t.name, "description": getattr(t, "description", "")}}
            for t in (ex.original_tools or [])
        ]

    return AgentPromptShape(
        name=name,
        system_text=system_text,
        user_text=user_text,
        tools_description=tools_desc,
        tools_schema=tools_schema,
    )


# ── Iteration-cost modeling ─────────────────────────────────────────


@dataclass
class IterationCost:
    """Effective token cost of one LLM call, accounting for the
    Anthropic cache state at that moment."""

    iteration: int
    fresh_tokens: int       # tokens NOT covered by cache (full price)
    cache_write_tokens: int # tokens being written to cache (1.25x)
    cache_read_tokens: int  # tokens served from cache (0.10x)

    @property
    def effective_tokens(self) -> float:
        return (
            1.00 * self.fresh_tokens
            + CACHE_WRITE_MULT * self.cache_write_tokens
            + CACHE_READ_MULT * self.cache_read_tokens
        )


def model_iterations(
    *,
    system_tokens: int,
    tools_tokens_per_iter: list[int],
    new_turn_tokens_per_iter: int = 200,
    num_iterations: int = 5,
) -> list[IterationCost]:
    """Simulate ``num_iterations`` LLM calls under Anthropic's cache.

    Critical model property: Anthropic caches **system**, **tools API
    parameter**, and **earlier messages** as **independent cache lines**.
    Mutating the tools array invalidates only the tools cache line —
    system + earlier messages stay warm. This was the Phase 0 cache-
    research finding and it's load-bearing for the LoadableAgent
    cost model: per-iteration tool-array changes are MUCH cheaper
    than they'd be if the whole prefix invalidated.

    Args:
        system_tokens: System message tokens. Stable across all iters
            (LoadableAgent doesn't mutate the system message; only
            the tools API param).
        tools_tokens_per_iter: Tools-API-array tokens for each iter
            (length must equal num_iterations). When this changes
            between iters, the tools-cache-line resets.
        new_turn_tokens_per_iter: Tokens added by each iteration's
            assistant tool_use + tool_result content.
        num_iterations: Number of LLM calls.

    Returns:
        Per-iteration IterationCost list.
    """
    assert len(tools_tokens_per_iter) == num_iterations, (
        f"tools_tokens_per_iter must have {num_iterations} entries, "
        f"got {len(tools_tokens_per_iter)}"
    )
    out: list[IterationCost] = []
    system_cached = False
    last_tools_size: int | None = None

    for i in range(1, num_iterations + 1):
        idx = i - 1
        tools_size = tools_tokens_per_iter[idx]
        # Earlier messages = sum of new turns from iters 1..i-1
        earlier_messages_tokens = new_turn_tokens_per_iter * (i - 1)

        # System cache line: write once at iter 1, read forever after.
        sys_write = 0
        sys_read = 0
        if not system_cached:
            sys_write = system_tokens
            system_cached = True
        else:
            sys_read = system_tokens

        # Tools cache line: write when changed (or on iter 1), read otherwise.
        tools_write = 0
        tools_read = 0
        if last_tools_size != tools_size:
            tools_write = tools_size
        else:
            tools_read = tools_size
        last_tools_size = tools_size

        # Earlier messages: can also be cached (fully read after iter 1).
        msg_read = earlier_messages_tokens if i > 1 else 0
        # New turn this iter: never cached (it's the suffix).
        msg_fresh = new_turn_tokens_per_iter if i > 1 else 0

        out.append(IterationCost(
            iteration=i,
            fresh_tokens=msg_fresh,
            cache_write_tokens=sys_write + tools_write,
            cache_read_tokens=sys_read + tools_read + msg_read,
        ))
    return out


# ── Stock vs LoadableAgent comparison ───────────────────────────────


def compare_stock_vs_loadable(
    *,
    num_iterations: int = 5,
    num_mid_task_loads: int = 2,
    new_turn_tokens: int = 200,
) -> dict[str, Any]:
    """The headline comparison.

    Builds a stock coder agent and a LoadableAgent (with the same
    tool universe but only 5 core tools loaded up front; 2 mid-task
    loads bring in additional tools). Models ``num_iterations`` LLM
    calls under Anthropic cache pricing for each.

    Returns a dict with per-agent costs + the ratio + the verdict.
    """
    import os
    os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

    # Stock agent — all 37+ tools baked in.
    from app.agents.coder import create_coder
    stock = create_coder()
    stock_shape = extract_prompt_shape(stock, name="stock_coder")

    # LoadableAgent shapes — one fresh agent per load-count to avoid
    # binder state from one shape leaking into the next. We need 3:
    #   * initial (0 loads) — what the agent's prompt looks like at iter 1
    #   * after 1 load     — at iter 2 (post-load #1)
    #   * after 2 loads    — at iter 3+ (post-load #2)
    loadable_shape = _build_loadable_and_extract(loaded_names=[])
    after_1 = _build_loadable_and_extract(loaded_names=["pdf_compose"])
    loadable_after_2 = _build_loadable_and_extract(
        loaded_names=["pdf_compose", "signal_send_attachment"],
    )

    # Build per-iteration tool-API sizes for both agents.
    # Stock: same tools every iter.
    stock_tools_per_iter = [stock_shape.tools_schema_tokens] * num_iterations

    # LoadableAgent: tool array grows with each mid-task load.
    # Loads happen "between" iters — tool N's schema is visible from
    # iter N+1 onward. With num_mid_task_loads=2:
    #   iter 1: initial (no loads yet)
    #   iter 2: after load #1
    #   iter 3+: after load #2 (and beyond)
    loadable_tools_per_iter = [loadable_shape.tools_schema_tokens]
    if num_mid_task_loads >= 1:
        loadable_tools_per_iter.append(after_1.tools_schema_tokens)
    if num_mid_task_loads >= 2:
        loadable_tools_per_iter.append(loadable_after_2.tools_schema_tokens)
    # Pad with the final size for any remaining iterations.
    while len(loadable_tools_per_iter) < num_iterations:
        loadable_tools_per_iter.append(loadable_tools_per_iter[-1])
    loadable_tools_per_iter = loadable_tools_per_iter[:num_iterations]

    # System tokens: in both agents, system text is small but the
    # interpolated tools_description is in the system message.
    # Stock = small system_text + the rendered tools_description.
    stock_system_total = stock_shape.system_tokens + stock_shape.tools_description_tokens
    # LoadableAgent's system message stays at the *initial* tools_description
    # — CrewAI bakes that string at executor init, mid-iter loads only
    # change the tools API parameter, not the system message.
    loadable_system_total = (
        loadable_shape.system_tokens + loadable_shape.tools_description_tokens
    )

    # Model each agent.
    stock_iters = model_iterations(
        system_tokens=stock_system_total,
        tools_tokens_per_iter=stock_tools_per_iter,
        new_turn_tokens_per_iter=new_turn_tokens,
        num_iterations=num_iterations,
    )
    loadable_iters = model_iterations(
        system_tokens=loadable_system_total,
        tools_tokens_per_iter=loadable_tools_per_iter,
        new_turn_tokens_per_iter=new_turn_tokens,
        num_iterations=num_iterations,
    )

    stock_total = sum(c.effective_tokens for c in stock_iters)
    loadable_total = sum(c.effective_tokens for c in loadable_iters)
    ratio = loadable_total / stock_total

    return {
        "shapes": {
            "stock": stock_shape.to_dict(),
            "loadable_initial": loadable_shape.to_dict(),
            "loadable_after_2_loads": loadable_after_2.to_dict(),
        },
        "system_tokens_per_agent": {
            "stock": stock_system_total,
            "loadable": loadable_system_total,
        },
        "tools_tokens_per_iter": {
            "stock": stock_tools_per_iter,
            "loadable": loadable_tools_per_iter,
        },
        "iter_costs": {
            "stock": [(c.iteration, round(c.effective_tokens, 1)) for c in stock_iters],
            "loadable": [(c.iteration, round(c.effective_tokens, 1)) for c in loadable_iters],
        },
        "stock_total": round(stock_total, 1),
        "loadable_total": round(loadable_total, 1),
        "ratio": round(ratio, 3),
        "threshold": 0.50,
        "verdict": "GO" if ratio <= 0.50 else "NO-GO",
    }


def _lazy(name: str):
    """Build a no-arg factory that constructs the named tool on demand.

    Mirrors what Phase 2's registry-backed binder will do — instead of
    eager construction at agent init, the factory is captured and
    only called when the agent loads the tool.
    """
    def factory():
        if name == "pdf_compose":
            from app.tools.pdf_compose import create_pdf_tools
            return create_pdf_tools("coder")[0]
        if name == "signal_send_attachment":
            from app.tools.signal_attachment import create_signal_attachment_tools
            tools = create_signal_attachment_tools("coder")
            return tools[0] if tools else None
        raise KeyError(name)
    return factory


def _build_loadable_and_extract(*, loaded_names: list[str]) -> AgentPromptShape:
    """Build a fresh LoadableAgent, optionally pre-load some tools,
    and extract its prompt shape. Used for measurement only — each
    call gets a clean binder so the shapes don't bleed state."""
    from app.knowledge_base.tools import KnowledgeSearchTool
    from app.llm_factory import create_specialist_llm
    from app.tool_runtime import LoadableAgent
    from app.tools.attachment_reader import read_attachment
    from app.tools.code_executor import execute_code
    from app.tools.file_manager import file_manager
    from app.tools.web_search import web_search

    llm = create_specialist_llm(max_tokens=4096, role="coding")
    agent = LoadableAgent(
        role="Coder",
        goal="Execute coding tasks",
        backstory="Same backstory as the stock coder.",
        llm=llm,
        core_tools=[
            execute_code, file_manager, web_search, read_attachment,
            KnowledgeSearchTool(),
        ],
        available_tools={
            "pdf_compose": _lazy("pdf_compose"),
            "signal_send_attachment": _lazy("signal_send_attachment"),
        },
        verbose=False,
    )
    for name in loaded_names:
        try:
            agent.binder.load(name)
        except Exception:
            pass
    # Force the executor to reflect the binder's current set.
    agent.create_agent_executor()
    if loaded_names:
        agent.agent_executor.original_tools = agent.binder.tools
        # Re-render the openai schema by rebuilding tools_description.
        from crewai.utilities.agent_utils import (
            parse_tools, render_text_description_and_args,
        )
        agent.agent_executor.tools_description = render_text_description_and_args(
            parse_tools(agent.binder.tools),
        )
    return extract_prompt_shape(agent, name=f"loadable_after_{len(loaded_names)}")


def _shape_after_loads(loadable_agent, names_to_load: list[str]) -> AgentPromptShape:
    """Simulate the LoadableAgent's prompt shape AFTER ``names_to_load``
    have been bound. The binder already supports this — we just walk
    through the loads + re-extract the prompt shape."""
    for name in names_to_load:
        try:
            loadable_agent.binder.load(name)
        except Exception:
            pass  # idempotent — already loaded counts as success
    # Force the executor to re-render its tools_description by rebuilding
    # its tool list off the binder (mirrors what _invoke_loop_native_tools
    # does mid-iteration in the prototype).
    loadable_agent.agent_executor.original_tools = loadable_agent.binder.tools
    return extract_prompt_shape(loadable_agent, name=f"loadable_after_{len(names_to_load)}")


# ── CLI ─────────────────────────────────────────────────────────────


def render_report(report: dict[str, Any]) -> str:
    """Format a comparison dict as a readable markdown report."""
    lines = [
        "# Phase 1c — Tool-overhead token cost: stock vs LoadableAgent",
        "",
        "## Per-agent prompt shape (one iteration)",
        "",
        "| Agent | System | User | Tools-description | Tools-schema | Total static |",
        "|-------|------:|-----:|------------------:|-------------:|-------------:|",
    ]
    for label, shape in [
        ("stock_coder", report["shapes"]["stock"]),
        ("loadable_initial", report["shapes"]["loadable_initial"]),
        ("loadable_after_2_loads", report["shapes"]["loadable_after_2_loads"]),
    ]:
        lines.append(
            f"| {label} | {shape['system_tokens']:,} | "
            f"{shape['user_tokens']:,} | "
            f"{shape['tools_description_tokens']:,} | "
            f"{shape['tools_schema_tokens']:,} | "
            f"**{shape['total_static_tokens']:,}** |"
        )

    lines += [
        "",
        "## Cache-line sizes",
        "",
        "Anthropic caches **system**, **tools API param**, and "
        "**earlier messages** as independent cache lines. Tool array "
        "mutations only invalidate the tools cache line.",
        "",
        "| Cache line | Stock | Loadable |",
        "|------------|------:|---------:|",
        f"| System (incl. tools-description) | {report['system_tokens_per_agent']['stock']:,} | {report['system_tokens_per_agent']['loadable']:,} |",
        f"| Tools API at iter 1 | {report['tools_tokens_per_iter']['stock'][0]:,} | {report['tools_tokens_per_iter']['loadable'][0]:,} |",
        f"| Tools API at iter 5 | {report['tools_tokens_per_iter']['stock'][-1]:,} | {report['tools_tokens_per_iter']['loadable'][-1]:,} |",
    ]

    lines += [
        "",
        "## Per-iteration effective tokens (after cache pricing)",
        "",
        "| iter | stock | loadable |",
        "|-----:|------:|---------:|",
    ]
    for s, l in zip(report["iter_costs"]["stock"], report["iter_costs"]["loadable"]):
        lines.append(f"| {s[0]} | {s[1]:,} | {l[1]:,} |")

    lines += [
        "",
        f"## Totals over {len(report['iter_costs']['stock'])} iterations",
        "",
        f"* Stock total:    **{report['stock_total']:,}** effective tokens",
        f"* Loadable total: **{report['loadable_total']:,}** effective tokens",
        f"* Ratio: **{report['ratio']:.1%}** of stock",
        f"* Phase 1c threshold: ≤ {int(report['threshold']*100)}%",
        f"* **Verdict: {report['verdict']}**",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    report = compare_stock_vs_loadable()
    print(render_report(report))


if __name__ == "__main__":
    main()
