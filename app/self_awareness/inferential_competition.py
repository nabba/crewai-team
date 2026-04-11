"""
inferential_competition.py — Beautiful Loop inferential competition.

Multiple approach plans compete for execution. Only the precision-weighted
winner gets executed. This implements the Beautiful Loop's second criterion:
inferential competition determines which hypothesis enters the reality model.

Design: LIGHTWEIGHT plan competition.
  - Generate N=3 short approach plans via local LLM (~300ms total)
  - Score each plan against the reality model using precision-weighting
  - Execute only the winning plan
  - Triggered selectively: ~15-25% of steps (when uncertainty is high)

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CompetingPlan:
    """A candidate approach plan for inferential competition."""
    plan_id: str
    approach: str
    predicted_outcome: str
    precision_score: float = 0.0
    alignment_score: float = 0.0
    novelty_score: float = 0.0
    affective_score: float = 0.5    # Somatic forecast valence (0=negative, 1=positive)
    free_energy_score: float = 0.5  # Active inference: expected FE reduction (0=increases, 1=reduces)
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "approach": self.approach[:200],
            "predicted_outcome": self.predicted_outcome[:200],
            "precision_score": round(self.precision_score, 3),
            "alignment_score": round(self.alignment_score, 3),
            "novelty_score": round(self.novelty_score, 3),
            "affective_score": round(self.affective_score, 3),
            "free_energy_score": round(self.free_energy_score, 3),
            "composite_score": round(self.composite_score, 3),
        }


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Pure Python cosine similarity, normalized to [0, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.5
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.5
    return (dot / (norm_a * norm_b) + 1.0) / 2.0


class InferentialCompetition:
    """Generates and evaluates competing approach plans."""

    def __init__(
        self,
        n_candidates: int = 3,
        precision_weight: float = 0.25,
        alignment_weight: float = 0.25,
        novelty_weight: float = 0.10,
        affective_weight: float = 0.20,
        free_energy_weight: float = 0.20,
    ):
        self.n_candidates = n_candidates
        self.precision_weight = precision_weight
        self.alignment_weight = alignment_weight
        self.novelty_weight = novelty_weight
        self.affective_weight = affective_weight
        self.free_energy_weight = free_energy_weight

    def should_compete(
        self,
        certainty_fast_path_mean: float,
        somatic_intensity: float,
        step_number: int,
    ) -> bool:
        """Trigger competition ~15-25% of the time."""
        if step_number == 0:
            return True
        if certainty_fast_path_mean < 0.4:
            return True
        if somatic_intensity > 0.6:
            return True
        return False

    def compete(
        self,
        task_description: str,
        reality_model=None,
        available_tools: list[str] = None,
        agent_id: str = "",
        free_energy_pressure: float = 0.0,
    ) -> tuple[CompetingPlan, list[CompetingPlan]]:
        """Generate N plans and select winner. Returns (winner, all)."""
        candidates = self._generate_candidates(task_description, reality_model, available_tools)

        if not candidates:
            default = CompetingPlan(
                plan_id="default", approach="Proceed with standard approach",
                predicted_outcome="Standard execution",
                precision_score=0.5, alignment_score=0.5, composite_score=0.5,
            )
            return default, [default]

        # Score each candidate (includes affective forecasting + free energy)
        scored = [self._score_plan(c, reality_model, candidates, agent_id, free_energy_pressure) for c in candidates]
        scored.sort(key=lambda p: p.composite_score, reverse=True)

        logger.info(
            f"Inferential competition: {len(scored)} plans. "
            f"Winner: '{scored[0].approach[:50]}' (score={scored[0].composite_score:.3f})"
        )
        return scored[0], scored

    def _generate_candidates(self, task_desc, reality_model, tools) -> list[CompetingPlan]:
        """Generate N candidate plans via local LLM."""
        try:
            from app.llm_factory import create_specialist_llm
            from app.utils import safe_json_parse

            llm = create_specialist_llm(max_tokens=400, role="self_improve", force_tier="local")

            model_context = ""
            if reality_model:
                high = reality_model.high_precision_elements[:3]
                model_context = "\n".join(
                    f"- [{e.category}] {e.content[:80]} (conf: {e.precision:.1f})" for e in high
                )

            tools_str = ", ".join((tools or [])[:10]) if tools else "standard tools"
            prompt = (
                f"Generate {self.n_candidates} DIFFERENT approaches to this task.\n"
                f"Each approach: 2-3 sentences max.\n\n"
                f"Task: {task_desc[:300]}\n\n"
                f"Known context:\n{model_context}\n\n"
                f"Tools: {tools_str}\n\n"
                f"Respond ONLY with JSON array:\n"
                f'[{{"plan_id": "plan_1", "approach": "...", "predicted_outcome": "..."}},'
                f' {{"plan_id": "plan_2", "approach": "...", "predicted_outcome": "..."}},'
                f' {{"plan_id": "plan_3", "approach": "...", "predicted_outcome": "..."}}]'
            )

            raw = str(llm.call(prompt)).strip()
            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]

            parsed, _ = safe_json_parse(raw)
            if not parsed:
                # Try parsing as array
                import json as _json
                try:
                    parsed = _json.loads(raw.strip())
                except Exception:
                    return []

            if isinstance(parsed, list):
                plans_data = parsed
            elif isinstance(parsed, dict) and "plans" in parsed:
                plans_data = parsed["plans"]
            else:
                return []

            return [
                CompetingPlan(
                    plan_id=pd.get("plan_id", f"plan_{i}"),
                    approach=pd.get("approach", ""),
                    predicted_outcome=pd.get("predicted_outcome", ""),
                )
                for i, pd in enumerate(plans_data[:self.n_candidates])
            ]

        except Exception as e:
            logger.debug(f"Plan generation failed: {e}")
            return []

    def _score_plan(self, plan, reality_model, all_plans, agent_id: str = "",
                    free_energy_pressure: float = 0.0) -> CompetingPlan:
        """Score a plan using precision-weighting + affective forecasting + free energy."""
        try:
            from app.memory.chromadb_manager import embed
            plan_emb = embed(plan.approach[:200])

            # 1. Precision: alignment with high-precision reality elements
            if reality_model and reality_model.high_precision_elements:
                high_prec = reality_model.high_precision_elements[:3]
                sims = []
                for elem in high_prec:
                    elem_emb = embed(elem.content[:200])
                    sim = _cosine_sim(plan_emb, elem_emb) * elem.precision
                    sims.append(sim)
                plan.precision_score = sum(sims) / len(sims) if sims else 0.5
            else:
                plan.precision_score = 0.5

            # 2. Alignment: similarity to overall reality model
            if reality_model and reality_model.elements:
                model_text = " ".join(e.content[:50] for e in reality_model.elements[:5])
                model_emb = embed(model_text[:300])
                plan.alignment_score = _cosine_sim(plan_emb, model_emb)
            else:
                plan.alignment_score = 0.5

            # 3. Novelty: dissimilarity from other plans
            other_plans = [p for p in all_plans if p.plan_id != plan.plan_id]
            if other_plans:
                other_sims = [_cosine_sim(plan_emb, embed(p.approach[:200])) for p in other_plans]
                plan.novelty_score = 1.0 - (sum(other_sims) / len(other_sims))
            else:
                plan.novelty_score = 0.5

            # 4. Affective forecast: predict emotional outcome of this approach
            # Uses somatic marker forecast (backward experience + forward causal beliefs)
            if agent_id:
                try:
                    from app.self_awareness.somatic_marker import SomaticMarkerComputer
                    smc = SomaticMarkerComputer()
                    forecast = smc.forecast(
                        agent_id=agent_id,
                        proposed_action=plan.approach[:300],
                        context_embedding=plan_emb,
                    )
                    # Map valence [-1, 1] to score [0, 1]
                    plan.affective_score = (forecast.valence + 1.0) / 2.0
                except Exception:
                    plan.affective_score = 0.5
            else:
                plan.affective_score = 0.5

        except Exception:
            plan.precision_score = 0.5
            plan.alignment_score = 0.5
            plan.novelty_score = 0.5
            plan.affective_score = 0.5

        # 5. Free energy minimization (active inference explore/exploit)
        # High pressure + novel plan = good (explore to reduce surprise)
        # Low pressure + precise plan = good (exploit working model)
        if free_energy_pressure > 0.5:
            plan.free_energy_score = plan.novelty_score * 0.6 + (1.0 - plan.precision_score) * 0.4
        else:
            plan.free_energy_score = plan.precision_score * 0.6 + plan.alignment_score * 0.4

        plan.composite_score = (
            self.precision_weight * plan.precision_score
            + self.alignment_weight * plan.alignment_score
            + self.novelty_weight * plan.novelty_score
            + self.affective_weight * plan.affective_score
            + self.free_energy_weight * plan.free_energy_score
        )
        return plan
