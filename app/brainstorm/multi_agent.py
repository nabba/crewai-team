"""Multi-agent brainstorm — parallel seed + react rounds with the creative roster.

When a session is in **team** mode, each step runs in two halves:

    1. ``gather_seed``   — agents (researcher, writer, coder, critic) generate
                           initial ideas in parallel, with anti-conformity
                           prompts. Their answers are shown to the user
                           before the user types.
    2. ``gather_react``  — same agents see the user's answer and each other's
                           seeds, then react / extend / disagree. Their
                           reactions are shown after the user, before moving
                           to the next step.

Reuses the high-creativity agent construction from
:mod:`app.crews.creative_crew` (heterogeneous LLM tiers per role + reasoning-
method diversity). Cost is tracked through the same active tracker so the
``creative_run_budget_usd`` setting applies — but a session-specific
``BRAINSTORM_TEAM_BUDGET_USD`` env var can raise the cap (default $0.50).

Failure modes are non-fatal: any agent that raises is recorded in
:class:`AgentResponse.error` and the session continues with the rest.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


DEFAULT_ROSTER: list[str] = ["researcher", "writer", "coder", "critic"]
MAX_ROSTER_SIZE = 5  # researcher, writer, coder, critic, commander
DEFAULT_TIMEOUT_S = 180
DEFAULT_PARALLEL = 4


@dataclass
class AgentResponse:
    """One agent's contribution to one round (seed or react)."""

    role: str
    text: str
    duration_s: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "text": self.text,
            "duration_s": self.duration_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentResponse":
        return cls(
            role=str(data.get("role", "?")),
            text=str(data.get("text", "")),
            duration_s=float(data.get("duration_s", 0.0)),
            error=data.get("error"),
        )


# ── Agent construction ────────────────────────────────────────────────────


def _factory_for(role: str) -> Callable:
    """Map a role name to its agent factory."""
    if role == "researcher":
        from app.agents.researcher import create_researcher
        return create_researcher
    if role == "writer":
        from app.agents.writer import create_writer
        return create_writer
    if role == "coder":
        from app.agents.coder import create_coder
        return create_coder
    if role == "critic":
        from app.agents.critic import create_critic
        return create_critic
    if role == "commander":
        from app.agents.commander.commander import Commander
        # Commander is an orchestrator, not a single-agent factory; we don't
        # use it as a participant unless explicitly requested. For now,
        # accepting only the four creative-crew roles keeps cost predictable.
        raise ValueError(
            "commander role not yet supported as a brainstorm participant"
        )
    raise ValueError(f"Unknown brainstorm role: {role!r}")


def resolve_roster(spec: int | list[str] | None) -> list[str]:
    """Normalise user-supplied roster spec to a list of role names.

    Accepts:
      - ``None`` or 0  → empty list (solo mode)
      - integer N      → first N from DEFAULT_ROSTER (clamped to 4)
      - list[str]      → explicit role names; validated
    """
    if spec is None or spec == 0:
        return []
    if isinstance(spec, int):
        n = max(0, min(spec, len(DEFAULT_ROSTER)))
        return list(DEFAULT_ROSTER[:n])
    if isinstance(spec, (list, tuple)):
        out: list[str] = []
        for r in spec:
            r = str(r).strip().lower()
            if not r:
                continue
            if r not in DEFAULT_ROSTER:
                raise ValueError(
                    f"Unknown brainstorm role {r!r} (allowed: {DEFAULT_ROSTER})"
                )
            if r not in out:
                out.append(r)
        return out
    raise TypeError(f"roster spec must be None, int, or list[str]; got {type(spec)}")


# Tier mapping for brainstorm — distinct from creative_crew's mapping.
#
# Why the divergence from creative_crew's `researcher: local`:
# In creative_crew, the researcher's wild-divergent firehose feeds into a
# downstream discuss/converge phase that cleans up garbage. In brainstorm,
# the researcher's output is shown DIRECTLY to the user, with no
# refinement step in between. Empirically (May 2026 in-app session) the
# local Ollama path on small models would echo CrewAI's internal Task
# scaffolding ("MUST return the actual content, not a summary…") instead
# of producing real ideas. Bumping to `budget` (DeepSeek via OpenRouter)
# fixes the regression while keeping researcher distinct from the other
# roles' tiers.
_TIER_BY_ROLE = {
    "researcher": "budget",   # was "local" — see comment above
    "writer": "mid",
    "coder": "budget",
    "critic": "premium",
}
_METHOD_BY_ROLE = {
    "researcher": "step_back",
    "writer": "analogical_blending",
    "coder": "compositional_cot",
    "critic": "contrastive",
}
_LLM_ROLE = {
    "researcher": "research",
    "writer": "writing",
    "coder": "coding",
    "critic": "critic",
}
# Tier for a role can be overridden via env: BRAINSTORM_TIER_<ROLE>=<tier>
# e.g. BRAINSTORM_TIER_RESEARCHER=local to roll back to the original
# creative_crew behaviour.


def _tier_for(role: str) -> str:
    override = os.environ.get(f"BRAINSTORM_TIER_{role.upper()}")
    if override:
        return override.strip().lower()
    return _TIER_BY_ROLE[role]


def _build_creative_agent(role: str, *, phase: str = "diverge"):
    """Return a CrewAI Agent configured for high-creativity brainstorming.

    Mirrors :func:`app.crews.creative_crew._make_agent` with two
    interactivity-driven differences:

    * ``max_execution_time`` is tighter (rounds need to be interactive).
    * ``researcher`` runs on ``budget`` instead of ``local`` — see the
      comment on ``_TIER_BY_ROLE`` for why.
    """
    from crewai import Agent
    from app.agents._common import optional_tool_group  # noqa: F401  (unused but for parity)
    from app.llm_factory import create_specialist_llm
    from app.souls.loader import compose_backstory

    factory = _factory_for(role)
    base = factory()
    tools = list(getattr(base, "tools", []) or [])

    llm = create_specialist_llm(
        # 4096 matches creative_crew. Lower caps were truncating
        # writer/critic mid-sentence on long-form responses.
        max_tokens=4096,
        role=_LLM_ROLE[role],
        force_tier=_tier_for(role),
        phase=phase,
    )
    backstory = compose_backstory(role, reasoning_method=_METHOD_BY_ROLE[role])
    return Agent(
        role=base.role,
        goal=base.goal,
        backstory=backstory,
        llm=llm,
        tools=tools,
        max_execution_time=DEFAULT_TIMEOUT_S,
        verbose=False,
    )


# ── Prompt templates ──────────────────────────────────────────────────────


def _build_seed_prompt(
    technique_title: str,
    topic: str,
    step_prompt: str,
    role: str,
) -> str:
    return (
        f"You are participating in a {technique_title} brainstorming session "
        f"with a human and three other agents.\n\n"
        f"## Topic\n{topic}\n\n"
        f"## Current step\n{step_prompt}\n\n"
        f"## Your role\n"
        f"You are the **{role}**. Lean into your role's natural perspective "
        f"and habits of thought — what you'd notice or ask that the others "
        f"might not.\n\n"
        f"## Instructions\n"
        f"Give 2-4 short, distinct, specific responses to this step. "
        f"Diverge from obvious answers — surprise the team. No hedging, no "
        f"meta-commentary, no preamble. Output a numbered list only.\n\n"
        f"## Output\n"
        f"A numbered list of 2-4 ideas, one per line. Concrete and specific. "
        f"No explanation of what you're about to do — just the ideas."
    )


def _format_peer_seeds(peers: list[AgentResponse], exclude_role: str) -> str:
    """Render peer seed answers as a markdown block for the react prompt."""
    chunks: list[str] = []
    for p in peers:
        if p.role == exclude_role or p.error:
            continue
        chunks.append(f"### {p.role}\n{p.text.strip()}")
    return "\n\n".join(chunks) if chunks else "(no peer responses available)"


def _build_react_prompt(
    technique_title: str,
    topic: str,
    step_prompt: str,
    role: str,
    user_answer: str,
    peer_seeds: list[AgentResponse],
) -> str:
    peers_text = _format_peer_seeds(peer_seeds, exclude_role=role)
    return (
        f"You are continuing a {technique_title} brainstorm.\n\n"
        f"## Topic\n{topic}\n\n"
        f"## Current step\n{step_prompt}\n\n"
        f"## What just happened\n"
        f"The HUMAN brainstormer answered:\n"
        f"\"\"\"\n{user_answer}\n\"\"\"\n\n"
        f"Your peers' seed responses (from before the human spoke):\n"
        f"{peers_text}\n\n"
        f"## Your role\n"
        f"You are the **{role}**. React. Do NOT just echo or politely agree.\n\n"
        f"## Instructions\n"
        f"Pick the most useful move: extend an idea, sharpen it, push back on "
        f"a weak one, or contribute a missing angle. Reference specific peers "
        f"or the human's point when relevant. Anti-conformity: at least one "
        f"of your contributions should disagree with or complicate something "
        f"already said.\n\n"
        f"## Output\n"
        f"2-3 short numbered points. Be specific. No filler."
    )


# ── Degenerate-response detection ─────────────────────────────────────────
#
# Real-world failure modes observed when a weak model is wrapped in
# CrewAI's Task scaffolding:
#
#   1. "MUST return the actual content, not a summary." repeated forever —
#      the model echoes CrewAI's internal re-prompt instead of answering.
#   2. "This is the expected criteria for your final answer…" — same
#      thing, different scaffolding string.
#   3. A single short phrase repeated 20+ times (small-model degeneration).
#   4. Streams of `")"")"` punctuation (broken JSON/markup attempts).
#
# Detecting these and tagging the response as errored prevents garbage
# from showing up in the UI; the user sees the agent's role-coloured
# error tag instead of a wall of repeated text.

_SCAFFOLDING_PHRASES: tuple[str, ...] = (
    "must return the actual content",
    "must return the actual complete content",
    "this is the expected criteria",
    "expected criteria for your final answer",
    "return the complete content as the final answer",
)


def _is_degenerate(text: str) -> tuple[bool, str]:
    """Return ``(is_bad, reason)`` for an agent response string.

    Pure: no I/O, no side effects. Conservative — designed to false-
    negative (let some bad responses through) rather than false-positive
    (block real ones). Tunable via the constants below.
    """
    if not text:
        return False, ""
    stripped = text.strip()
    if len(stripped) < 30:
        # Too short to gauge; let it through.
        return False, ""

    lower = stripped.lower()

    # 1. Scaffolding echo — these phrases appearing 3+ times means the
    #    model is parroting CrewAI's re-prompt.
    for phrase in _SCAFFOLDING_PHRASES:
        if lower.count(phrase) >= 3:
            return True, f"echoing CrewAI scaffolding ('{phrase[:40]}…')"

    # 2. Repetition loop — 5+ identical non-empty lines.
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) >= 5:
        from collections import Counter

        counts = Counter(lines)
        most_line, count = counts.most_common(1)[0]
        if count >= 5:
            preview = most_line[:60] + ("…" if len(most_line) > 60 else "")
            return True, f"repetition loop ({count}× '{preview}')"

    # 3. Heavy non-text content — long response that is mostly punctuation
    #    or symbols. Threshold: <40% alphanumeric/whitespace and >150 chars.
    if len(stripped) > 150:
        text_chars = sum(1 for c in stripped if c.isalnum() or c.isspace())
        ratio = text_chars / len(stripped)
        if ratio < 0.40:
            return True, f"heavy non-text content ({ratio:.0%} alphanumeric)"

    # 4. Very low word diversity — long response with <15% unique words.
    words = stripped.split()
    if len(words) >= 50:
        unique = len(set(w.lower() for w in words))
        diversity = unique / len(words)
        if diversity < 0.15:
            return True, f"low diversity ({unique}/{len(words)} unique words)"

    return False, ""


# ── Execution ─────────────────────────────────────────────────────────────


def _run_one_agent(
    role: str,
    description: str,
    *,
    expected_output: str,
    phase: str,
) -> AgentResponse:
    """Build the agent, run one Task, return :class:`AgentResponse`.

    Catches all exceptions and converts them into ``AgentResponse.error``.
    Also runs the output through :func:`_is_degenerate` and demotes
    scaffolding-echo / repetition-loop responses to errors so the UI
    doesn't display them as ideas.
    """
    from crewai import Crew, Process, Task

    t0 = time.monotonic()
    try:
        agent = _build_creative_agent(role, phase=phase)
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
        crew = Crew(
            agents=[agent], tasks=[task], process=Process.sequential, verbose=False
        )
        text = str(crew.kickoff()).strip()
    except Exception as exc:
        logger.warning("brainstorm.multi_agent: %s failed: %s", role, exc)
        return AgentResponse(
            role=role,
            text="",
            duration_s=time.monotonic() - t0,
            error=str(exc)[:200],
        )

    bad, reason = _is_degenerate(text)
    if bad:
        logger.warning(
            "brainstorm.multi_agent: %s produced degenerate output (%s)",
            role,
            reason,
        )
        return AgentResponse(
            role=role,
            text="",
            duration_s=time.monotonic() - t0,
            error=f"degenerate output: {reason}",
        )
    return AgentResponse(role=role, text=text, duration_s=time.monotonic() - t0)


def _gather_parallel(
    roster: list[str],
    prompt_builder: Callable[[str], str],
    *,
    phase: str,
    expected_output: str,
    parallel_workers: int = DEFAULT_PARALLEL,
) -> list[AgentResponse]:
    """Run each role in parallel through ``prompt_builder(role)``."""
    if not roster:
        return []
    results: dict[str, AgentResponse] = {}
    workers = min(max(1, parallel_workers), len(roster))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bstorm-agent") as pool:
        future_map = {
            pool.submit(
                _run_one_agent,
                role,
                prompt_builder(role),
                expected_output=expected_output,
                phase=phase,
            ): role
            for role in roster
        }
        for fut in as_completed(future_map):
            role = future_map[fut]
            try:
                results[role] = fut.result()
            except Exception as exc:
                results[role] = AgentResponse(
                    role=role,
                    text="",
                    duration_s=0.0,
                    error=f"executor: {exc!r}",
                )
    # Preserve roster order in output
    return [results[r] for r in roster if r in results]


def _team_budget_usd() -> float:
    """Per-session budget cap. Defaults to $0.50; override via env."""
    raw = os.environ.get("BRAINSTORM_TEAM_BUDGET_USD")
    if not raw:
        return 0.50
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.50


def _budget_exceeded(spent_usd: float) -> bool:
    cap = _team_budget_usd()
    return cap > 0 and spent_usd >= cap


# ── Public API ────────────────────────────────────────────────────────────


def gather_seed(
    *,
    technique_title: str,
    topic: str,
    step_prompt: str,
    roster: list[str],
    spent_so_far_usd: float = 0.0,
) -> list[AgentResponse]:
    """Run the seed phase: agents propose initial answers in parallel.

    Returns an empty list (with a warning logged) when the budget is hit.
    """
    if _budget_exceeded(spent_so_far_usd):
        logger.warning(
            "brainstorm.multi_agent: skipping seed — budget $%.2f reached "
            "(cap $%.2f)", spent_so_far_usd, _team_budget_usd(),
        )
        return []
    return _gather_parallel(
        roster,
        lambda role: _build_seed_prompt(technique_title, topic, step_prompt, role),
        phase="diverge",
        expected_output="A numbered list of 2-4 distinct, specific ideas.",
    )


def gather_react(
    *,
    technique_title: str,
    topic: str,
    step_prompt: str,
    roster: list[str],
    user_answer: str,
    peer_seeds: list[AgentResponse],
    spent_so_far_usd: float = 0.0,
) -> list[AgentResponse]:
    """Run the react phase: agents react to user + peer seeds in parallel."""
    if _budget_exceeded(spent_so_far_usd):
        logger.warning(
            "brainstorm.multi_agent: skipping react — budget $%.2f reached "
            "(cap $%.2f)", spent_so_far_usd, _team_budget_usd(),
        )
        return []
    return _gather_parallel(
        roster,
        lambda role: _build_react_prompt(
            technique_title, topic, step_prompt, role, user_answer, peer_seeds
        ),
        phase="discuss",
        expected_output="2-3 numbered points reacting to what was said.",
    )
