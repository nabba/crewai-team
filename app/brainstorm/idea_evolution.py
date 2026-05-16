"""Population-based idea evolution — ShinkaEvolve pattern, text mode.

PROGRAM §46.22 (Q11.3). ShinkaEvolve's core algorithm — population
+ LLM mutation + verifier + diversity archive + island model —
generalises to ideas. This module is a standalone implementation
that does NOT depend on the ``shinka`` package (which is built for
code); it runs the same pattern over plain-text idea strings.

Population members are short text descriptions; the mutator is an
LLM rewrite; the verifier is an LLM judge against task constraints;
the archive preserves diversity by embedding distance.

Algorithm per generation:

  1. Score every member against task + constraints (LLM judge).
  2. Pick top-K survivors AND a few high-diversity outliers.
  3. Mutate survivors → children (LLM rewrites with diversity hint).
  4. Score children, add to next generation.
  5. Update the archive (max-size, diversity-preserving).
  6. Stop when budget exhausted or generations cap reached.

Cost discipline:

  * Hard cap ``budget_usd`` (default $0.50 per call). Caller can
    request smaller; never larger than ``MAX_BUDGET_USD``.
  * Hard cap ``max_generations`` (default 5, cap 20).
  * Hard cap ``population_size`` (default 6, cap 12).

Output:

  * ``IdeaEvolutionResult.population`` — final population (scored).
  * ``IdeaEvolutionResult.archive`` — diverse strong members across
    generations.
  * ``IdeaEvolutionResult.top_ideas(n)`` — convenience for callers
    that want top-N by judge score with novelty-aware tie-break.

Integration:

  * Operator triggers via ``/brainstorm evolve <topic>`` (slash
    command); the Commander dispatcher invokes
    :func:`evolve_ideas` and pipes the top results into a new
    brainstorm session as seed ideas.
  * Tests inject ``mutator_fn`` and ``judge_fn`` to drive the
    algorithm without LLM calls.

Master switch: ``IDEA_EVOLUTION_ENABLED`` (default ON).
"""
from __future__ import annotations

import logging
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#   Hard caps
# ─────────────────────────────────────────────────────────────────────


MAX_BUDGET_USD = 2.0
MAX_GENERATIONS = 20
MAX_POPULATION = 12
MAX_ARCHIVE = 24

_DEFAULT_BUDGET_USD = 0.50
_DEFAULT_GENERATIONS = 5
_DEFAULT_POPULATION = 6
_DEFAULT_ARCHIVE = 12


def _enabled() -> bool:
    return os.getenv("IDEA_EVOLUTION_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ─────────────────────────────────────────────────────────────────────
#   Data model
# ─────────────────────────────────────────────────────────────────────


@dataclass
class IdeaMember:
    """One individual in the population."""

    id: str
    text: str
    score: float = 0.0       # judge score 0..1
    parent_id: Optional[str] = None
    generation: int = 0
    judge_rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "judge_rationale": self.judge_rationale,
        }


@dataclass
class IdeaEvolutionResult:
    """Output of one evolve_ideas() call."""

    task: str
    population: list[IdeaMember] = field(default_factory=list)
    archive: list[IdeaMember] = field(default_factory=list)
    generations_run: int = 0
    judge_calls: int = 0
    mutate_calls: int = 0
    estimated_cost_usd: float = 0.0
    truncated_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "population": [m.to_dict() for m in self.population],
            "archive": [m.to_dict() for m in self.archive],
            "generations_run": self.generations_run,
            "judge_calls": self.judge_calls,
            "mutate_calls": self.mutate_calls,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "truncated_reason": self.truncated_reason,
        }

    def top_ideas(self, n: int = 5) -> list[IdeaMember]:
        """Top-N by score across population ∪ archive, deduped by id."""
        seen: set[str] = set()
        merged: list[IdeaMember] = []
        for m in sorted(
            list(self.population) + list(self.archive),
            key=lambda x: x.score, reverse=True,
        ):
            if m.id in seen:
                continue
            seen.add(m.id)
            merged.append(m)
            if len(merged) >= max(1, min(n, 50)):
                break
        return merged


# ─────────────────────────────────────────────────────────────────────
#   Injectable hooks
# ─────────────────────────────────────────────────────────────────────


MutatorFn = Callable[[str, str, list[str]], str]
"""(task, parent_text, neighbour_texts) -> mutated text.

The neighbour list lets the mutator nudge toward diversity ("don't
look like these").
"""

JudgeFn = Callable[[str, str, list[str]], tuple[float, str]]
"""(task, idea_text, constraints) -> (score in [0,1], rationale)."""


# ─────────────────────────────────────────────────────────────────────
#   Cost estimation
# ─────────────────────────────────────────────────────────────────────


# Conservative per-call estimates — Anthropic Haiku 4.5 input+output
# typical token cost. Rounded up for headroom.
_COST_PER_JUDGE = 0.0005      # short input, short output
_COST_PER_MUTATE = 0.001      # longer output


def _within_budget(spent: float, budget: float) -> bool:
    return spent < budget


# ─────────────────────────────────────────────────────────────────────
#   Diversity (hash-trick cosine)
# ─────────────────────────────────────────────────────────────────────


def _embed(text: str) -> list[float]:
    """Use the hash-trick embedding already in the codebase."""
    try:
        from app.utils.hash_embedding import embed
        return embed(text or "")
    except Exception:
        return []


def _cosine(a: list[float], b: list[float]) -> float:
    try:
        from app.utils.hash_embedding import cosine
        return cosine(a, b)
    except Exception:
        return 0.0


def _diverse_pick(
    members: list[IdeaMember], k: int,
) -> list[IdeaMember]:
    """Pick ``k`` members that maximize pairwise distance.

    Greedy: start with the highest-scored member; each subsequent
    pick maximizes (member.score - 0.5 × max_cosine_to_picked).
    """
    if not members:
        return []
    k = max(1, min(k, len(members)))
    sorted_members = sorted(members, key=lambda m: m.score, reverse=True)
    picked: list[IdeaMember] = [sorted_members[0]]
    picked_emb: list[list[float]] = [_embed(sorted_members[0].text)]
    remaining = sorted_members[1:]
    while remaining and len(picked) < k:
        best_m = None
        best_obj = -1e9
        for m in remaining:
            emb = _embed(m.text)
            if not emb or not picked_emb[0]:
                # Embedding fallback: just use score order
                obj = m.score
            else:
                max_sim = max(_cosine(emb, e) for e in picked_emb)
                obj = m.score - 0.5 * max_sim
            if obj > best_obj:
                best_obj = obj
                best_m = m
        if best_m is None:
            break
        picked.append(best_m)
        picked_emb.append(_embed(best_m.text))
        remaining = [m for m in remaining if m.id != best_m.id]
    return picked


def _update_archive(
    archive: list[IdeaMember],
    candidates: list[IdeaMember],
    *,
    max_size: int,
) -> list[IdeaMember]:
    """Keep the most-diverse top-scored members. ``max_size`` is the
    cap; ties broken by generation (newer wins)."""
    pool = list(archive) + list(candidates)
    # Dedup by id, keep the higher-scored copy.
    by_id: dict[str, IdeaMember] = {}
    for m in pool:
        existing = by_id.get(m.id)
        if existing is None or m.score > existing.score:
            by_id[m.id] = m
    if len(by_id) <= max_size:
        return list(by_id.values())
    return _diverse_pick(list(by_id.values()), max_size)


# ─────────────────────────────────────────────────────────────────────
#   Public entry point
# ─────────────────────────────────────────────────────────────────────


def evolve_ideas(
    task: str,
    *,
    seed_ideas: list[str],
    constraints: list[str] | None = None,
    generations: int = _DEFAULT_GENERATIONS,
    population_size: int = _DEFAULT_POPULATION,
    archive_size: int = _DEFAULT_ARCHIVE,
    budget_usd: float = _DEFAULT_BUDGET_USD,
    mutator_fn: MutatorFn | None = None,
    judge_fn: JudgeFn | None = None,
    rng: random.Random | None = None,
) -> IdeaEvolutionResult:
    """Population-based idea search.

    ``task`` — the problem statement / question being explored.
    ``seed_ideas`` — initial population (≥1; gets padded if shorter
        than population_size by LLM-mutating the seeds).
    ``constraints`` — optional list of constraint strings the judge
        will score against (e.g. "must be feasible in 1 week", "must
        cost < $100"). Defaults to empty.
    ``generations`` — capped at MAX_GENERATIONS.
    ``population_size`` — capped at MAX_POPULATION.
    ``archive_size`` — capped at MAX_ARCHIVE.
    ``budget_usd`` — hard cap; capped at MAX_BUDGET_USD.
    ``mutator_fn`` / ``judge_fn`` — test seams; production passes
        None and the LLM-backed defaults are used.

    Returns an :class:`IdeaEvolutionResult`. Returns an empty result
    with ``truncated_reason="disabled"`` when the master switch is OFF.
    """
    if not _enabled():
        return IdeaEvolutionResult(
            task=task, truncated_reason="disabled",
        )
    if not task or not task.strip():
        return IdeaEvolutionResult(
            task=task, truncated_reason="empty_task",
        )
    if not seed_ideas:
        return IdeaEvolutionResult(
            task=task, truncated_reason="no_seed_ideas",
        )

    generations = max(1, min(int(generations), MAX_GENERATIONS))
    population_size = max(2, min(int(population_size), MAX_POPULATION))
    archive_size = max(2, min(int(archive_size), MAX_ARCHIVE))
    budget_usd = max(0.05, min(float(budget_usd), MAX_BUDGET_USD))

    constraints = list(constraints or [])
    mutator = mutator_fn or _default_mutator
    judge = judge_fn or _default_judge
    rng = rng or random.Random(42)

    result = IdeaEvolutionResult(task=task)

    # Initialise population from seeds (truncate or pad)
    population: list[IdeaMember] = [
        IdeaMember(
            id=str(uuid.uuid4())[:8],
            text=text.strip(),
            generation=0,
        )
        for text in seed_ideas if text and text.strip()
    ][:population_size]

    # Score initial population
    for m in population:
        try:
            score, rationale = judge(task, m.text, constraints)
        except Exception as exc:
            score, rationale = 0.0, f"judge raised: {exc}"
        m.score = max(0.0, min(1.0, float(score)))
        m.judge_rationale = (rationale or "")[:240]
        result.judge_calls += 1
        result.estimated_cost_usd += _COST_PER_JUDGE
        if not _within_budget(result.estimated_cost_usd, budget_usd):
            result.truncated_reason = "budget_exhausted_initial_score"
            result.population = population
            return result

    archive: list[IdeaMember] = []

    # Pad population to size by LLM-mutating seeds if needed
    while len(population) < population_size:
        if not _within_budget(result.estimated_cost_usd, budget_usd):
            result.truncated_reason = "budget_exhausted_initial_pad"
            break
        parent = rng.choice(population)
        try:
            child_text = mutator(task, parent.text, [m.text for m in population])
        except Exception:
            child_text = ""
        result.mutate_calls += 1
        result.estimated_cost_usd += _COST_PER_MUTATE
        if not child_text.strip():
            continue
        child = IdeaMember(
            id=str(uuid.uuid4())[:8],
            text=child_text.strip()[:1000],
            parent_id=parent.id,
            generation=0,
        )
        try:
            score, rationale = judge(task, child.text, constraints)
        except Exception:
            score, rationale = 0.0, "judge failed"
        child.score = max(0.0, min(1.0, float(score)))
        child.judge_rationale = (rationale or "")[:240]
        result.judge_calls += 1
        result.estimated_cost_usd += _COST_PER_JUDGE
        population.append(child)

    archive = _update_archive(archive, population, max_size=archive_size)

    # Generation loop
    for gen in range(1, generations + 1):
        result.generations_run = gen
        if not _within_budget(result.estimated_cost_usd, budget_usd):
            result.truncated_reason = "budget_exhausted"
            break
        # Pick survivors: top half by score + diversity outliers
        sorted_pop = sorted(population, key=lambda m: m.score, reverse=True)
        elite_n = max(1, len(sorted_pop) // 2)
        elites = sorted_pop[:elite_n]
        diversity = _diverse_pick(sorted_pop[elite_n:], max(1, elite_n // 2))
        survivors = elites + diversity

        # Mutate survivors → children
        children: list[IdeaMember] = []
        for parent in survivors:
            if len(children) + len(survivors) >= population_size * 2:
                break
            if not _within_budget(result.estimated_cost_usd, budget_usd):
                result.truncated_reason = "budget_exhausted_during_mutate"
                break
            neighbour_texts = [
                m.text for m in survivors if m.id != parent.id
            ]
            try:
                child_text = mutator(task, parent.text, neighbour_texts)
            except Exception:
                child_text = ""
            result.mutate_calls += 1
            result.estimated_cost_usd += _COST_PER_MUTATE
            if not child_text.strip():
                continue
            child = IdeaMember(
                id=str(uuid.uuid4())[:8],
                text=child_text.strip()[:1000],
                parent_id=parent.id,
                generation=gen,
            )
            try:
                score, rationale = judge(task, child.text, constraints)
            except Exception:
                score, rationale = 0.0, "judge failed"
            child.score = max(0.0, min(1.0, float(score)))
            child.judge_rationale = (rationale or "")[:240]
            result.judge_calls += 1
            result.estimated_cost_usd += _COST_PER_JUDGE
            children.append(child)

        # Next generation = top survivors + best children, capped
        combined = sorted(
            survivors + children, key=lambda m: m.score, reverse=True,
        )
        population = combined[:population_size]
        archive = _update_archive(archive, population, max_size=archive_size)

    if not result.truncated_reason:
        result.truncated_reason = "generations_complete"
    result.population = population
    result.archive = archive
    return result


# ─────────────────────────────────────────────────────────────────────
#   Default LLM-backed hooks
# ─────────────────────────────────────────────────────────────────────


def _default_mutator(
    task: str, parent_text: str, neighbour_texts: list[str],
) -> str:
    """Anthropic Haiku 4.5 rewriting the parent toward diversity."""
    try:
        import anthropic
    except ImportError:
        return ""
    neighbours_block = "\n".join(
        f"- {t[:240]}" for t in neighbour_texts[:5]
    ) or "(no neighbours)"
    system = (
        "You are evolving ideas in a population-based search. Given "
        "a task, a parent idea, and the parent's neighbours, produce "
        "ONE child idea that:\n"
        "  - addresses the task\n"
        "  - is structurally different from the parent (not a "
        "    paraphrase)\n"
        "  - is far from the listed neighbours\n"
        "  - is concrete (specific actions or mechanisms, not "
        "    abstract slogans)\n"
        "Output ONLY the child idea text — no preamble, no "
        "explanation, no list, no markdown. Max 240 characters."
    )
    user = (
        f"Task: {task}\n\n"
        f"Parent idea:\n{parent_text}\n\n"
        f"Neighbour ideas (avoid resembling these):\n{neighbours_block}\n\n"
        f"Child idea:"
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=240,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_parts = [
            getattr(b, "text", "")
            for b in (msg.content or [])
            if getattr(b, "type", "") == "text"
        ]
        return "".join(text_parts).strip()[:1000]
    except Exception:
        logger.debug("idea_evolution: mutator call failed", exc_info=True)
        return ""


def _default_judge(
    task: str, idea_text: str, constraints: list[str],
) -> tuple[float, str]:
    """Anthropic Haiku 4.5 judging the idea against task + constraints.

    Returns (score in [0..1], rationale). On any failure returns
    (0.0, "judge failed: <reason>") so the caller's iteration
    continues."""
    try:
        import anthropic
    except ImportError:
        return 0.0, "anthropic unavailable"
    constraints_block = "\n".join(
        f"- {c}" for c in constraints[:6]
    ) or "(no explicit constraints — judge on intrinsic merit)"
    system = (
        "You are an idea judge. Score one idea against the task and "
        "constraints. Output STRICTLY JSON of the form:\n"
        '  {"score": <float 0..1>, "rationale": "<one-sentence reason>"}\n\n'
        "Score 1.0 = directly answers the task within all constraints.\n"
        "Score 0.5 = partially addresses task or partially within constraints.\n"
        "Score 0.0 = off-topic, infeasible, or violates a hard constraint.\n\n"
        "Output ONLY the JSON object, no preamble, no markdown fence."
    )
    user = (
        f"Task: {task}\n\n"
        f"Constraints:\n{constraints_block}\n\n"
        f"Idea:\n{idea_text}\n\n"
        f"JSON judgement:"
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_parts = [
            getattr(b, "text", "")
            for b in (msg.content or [])
            if getattr(b, "type", "") == "text"
        ]
        raw = "".join(text_parts).strip()
    except Exception as exc:
        return 0.0, f"judge raised: {exc}"
    # Tolerant parse
    import json
    import re
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return 0.0, "judge produced no JSON"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return 0.0, "judge JSON malformed"
    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    rationale = str(data.get("rationale", "") or "")
    return max(0.0, min(1.0, score)), rationale
