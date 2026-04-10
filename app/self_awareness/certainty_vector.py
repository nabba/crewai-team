"""
certainty_vector.py — Certainty Vector computation for AndrusAI agents.

Fast path: 3 dimensions from embeddings + DB lookups (~50ms on M4 Max). Always runs.
Slow path: 3 additional dimensions via local LLM (~100ms). Conditionally triggered.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from app.self_awareness.internal_state import CertaintyVector

logger = logging.getLogger(__name__)


class CertaintyVectorComputer:
    """Computes the CertaintyVector for a reasoning step."""

    def __init__(self):
        self._tool_success_cache: dict[str, float] = {}

    # ── FAST PATH (always runs, ~50ms) ───────────────────────────────────────

    def compute_fast_path(
        self,
        agent_id: str,
        current_output: str,
        rag_source_count: int = 0,
        total_claim_count: int = 0,
        selected_tool: Optional[str] = None,
        recent_output_embeddings: Optional[list[list[float]]] = None,
    ) -> CertaintyVector:
        """Compute 3 fast-path dimensions from DB lookups and embeddings."""
        cv = CertaintyVector()

        # 1. Factual grounding: ratio of sourced claims
        if total_claim_count > 0:
            cv.factual_grounding = min(rag_source_count / total_claim_count, 1.0)
        else:
            cv.factual_grounding = 0.5  # No claims = neutral

        # 2. Tool confidence: historical success rate
        if selected_tool:
            cv.tool_confidence = self._get_tool_confidence(agent_id, selected_tool)
        else:
            cv.tool_confidence = 0.5  # No tool = neutral

        # 3. Coherence: similarity to recent outputs
        if current_output and len(current_output) > 20:
            try:
                from app.memory.chromadb_manager import embed
                current_embedding = embed(current_output[:1000])
                if recent_output_embeddings is None:
                    recent_output_embeddings = self._get_recent_embeddings(agent_id, limit=3)
                cv.coherence = self._compute_coherence(current_embedding, recent_output_embeddings)
            except Exception:
                cv.coherence = 0.5
        else:
            cv.coherence = 0.5

        return cv

    def _get_tool_confidence(self, agent_id: str, tool_name: str) -> float:
        """Historical success rate of this tool for this agent."""
        cache_key = f"{agent_id}:{tool_name}"
        if cache_key in self._tool_success_cache:
            return self._tool_success_cache[cache_key]

        try:
            from app.self_awareness.agent_state import get_agent_stats
            stats = get_agent_stats(agent_id)
            success_rate = stats.get("success_rate", 0.5)
            self._tool_success_cache[cache_key] = success_rate
            return success_rate
        except Exception:
            return 0.5

    def _get_recent_embeddings(self, agent_id: str, limit: int = 3) -> list[list[float]]:
        """Fetch embeddings of recent outputs from agent_experiences."""
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT context_embedding
                FROM agent_experiences
                WHERE agent_id = %s
                  AND context_embedding IS NOT NULL
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (agent_id, limit),
                fetch=True,
            )
            return [list(row[0]) for row in (rows or []) if row[0]]
        except Exception:
            return []

    @staticmethod
    def _compute_coherence(current: list[float], recent: list[list[float]]) -> float:
        """Cosine similarity between current output and recent outputs. Normalized to [0,1]."""
        if not recent:
            return 0.5

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.5
            return (dot / (norm_a * norm_b) + 1.0) / 2.0  # Normalize [-1,1] → [0,1]

        sims = [cosine_sim(current, past) for past in recent]
        return sum(sims) / len(sims)

    # ── SLOW PATH (conditional, ~100ms) ──────────────────────────────────────

    def compute_slow_path(
        self,
        agent_id: str,
        task_description: str,
        current_output: str,
        cv: CertaintyVector,
    ) -> CertaintyVector:
        """Compute 3 slow-path dims via local LLM. Only call when should_trigger_slow_path()."""
        try:
            from app.llm_factory import create_specialist_llm
            from app.utils import safe_json_parse

            llm = create_specialist_llm(max_tokens=100, role="self_improve", force_tier="local")
            prompt = (
                "Rate the following on a scale of 0.0 to 1.0:\n\n"
                f"1. task_understanding: How well does this output address the task?\n"
                f"   Task: {task_description[:200]}\n"
                f"   Output: {current_output[:300]}\n\n"
                "2. value_alignment: How consistent with integrity, safety, transparency?\n\n"
                "3. meta_certainty: How confident are you in these ratings?\n\n"
                'Respond ONLY with JSON: {"task_understanding": 0.X, "value_alignment": 0.X, "meta_certainty": 0.X}'
            )
            raw = str(llm.call(prompt)).strip()
            parsed, _ = safe_json_parse(raw)
            if parsed:
                cv.task_understanding = max(0.0, min(1.0, float(parsed.get("task_understanding", 0.5))))
                cv.value_alignment = max(0.0, min(1.0, float(parsed.get("value_alignment", 0.5))))
                cv.meta_certainty = max(0.0, min(1.0, float(parsed.get("meta_certainty", 0.5))))
        except Exception as e:
            logger.debug(f"Slow-path certainty failed for {agent_id}: {e}")
            # Fallback: meta_certainty from variance
            cv.meta_certainty = max(0.0, 1.0 - (cv.variance * 5.0))

        return cv

    def compute_full(
        self,
        agent_id: str,
        task_description: str,
        current_output: str,
        rag_source_count: int = 0,
        total_claim_count: int = 0,
        selected_tool: Optional[str] = None,
    ) -> CertaintyVector:
        """Full certainty vector: fast path always, slow path conditionally."""
        cv = self.compute_fast_path(
            agent_id=agent_id,
            current_output=current_output,
            rag_source_count=rag_source_count,
            total_claim_count=total_claim_count,
            selected_tool=selected_tool,
        )

        # Read thresholds from sentience config (adjustable by cogito feedback loop)
        try:
            from app.self_awareness.sentience_config import load_config
            cfg = load_config()
            threshold = cfg.get("slow_path_trigger_threshold", 0.4)
            var_threshold = cfg.get("slow_path_variance_threshold", 0.03)
        except Exception:
            threshold, var_threshold = 0.4, 0.03

        if cv.should_trigger_slow_path(threshold=threshold, variance_threshold=var_threshold):
            cv = self.compute_slow_path(agent_id, task_description, current_output, cv)
        else:
            cv.meta_certainty = max(0.0, 1.0 - (cv.variance * 5.0))

        return cv

    def clear_cache(self) -> None:
        self._tool_success_cache.clear()
