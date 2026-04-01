"""
meta_learning.py — Tracks which modification strategies succeed and biases
future proposals toward proven approaches.

Uses UCB1 (Upper Confidence Bound) to balance exploitation (strategies
with high success rates) vs exploration (under-tried strategies).

This module only becomes meaningful after sufficient modification data
accumulates (30+ attempts).  Before that threshold, strategy selection
is uniform random.
"""

import json
import logging
import math
import random
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Minimum sample size before meta-learning takes effect
MIN_SAMPLE_SIZE = 30

# UCB1 exploration weight (higher = more exploration)
EXPLORATION_WEIGHT = 1.414

# All known modification strategies
ALL_STRATEGIES = [
    "additive_instruction",
    "example_injection",
    "instruction_refinement",
    "constraint_addition",
    "persona_calibration",
]


class MetaLearner:
    """Tracks modification strategy effectiveness and suggests the best strategy."""

    def __init__(self, db_url: str):
        self._db_url = db_url
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(self._db_url, pool_size=1)
            except Exception:
                pass
        return self._engine

    def _execute(self, query: str, params: dict = None) -> list:
        engine = self._get_engine()
        if not engine:
            return []
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                if result.returns_rows:
                    return [dict(row._mapping) for row in result]
                conn.commit()
                return []
        except Exception:
            return []

    def record_outcome(self, modification_id: str, strategy: str,
                        category: str, outcome: str,
                        improvement: float = 0.0) -> None:
        """Record whether a modification was successful.

        Args:
            modification_id: UUID of the modification attempt
            strategy: which strategy was used
            category: feedback category that triggered the modification
            outcome: 'promoted', 'rejected', or 'rolled_back'
            improvement: weighted score improvement (if available)
        """
        is_success = outcome == "promoted"

        self._execute(
            """INSERT INTO modification.strategy_stats
               (feedback_category, modification_strategy, attempts, successes, total_improvement)
               VALUES (:cat, :strategy, 1, :success, :improvement)
               ON CONFLICT (feedback_category, modification_strategy)
               DO UPDATE SET
                   attempts = modification.strategy_stats.attempts + 1,
                   successes = modification.strategy_stats.successes + :success,
                   total_improvement = modification.strategy_stats.total_improvement + :improvement,
                   last_updated = now()""",
            {
                "cat": category,
                "strategy": strategy,
                "success": 1 if is_success else 0,
                "improvement": improvement if is_success else 0.0,
            }
        )

    def get_strategy_stats(self) -> list[dict]:
        """Return success rates per (feedback_category, modification_strategy)."""
        rows = self._execute(
            """SELECT feedback_category, modification_strategy,
                      attempts, successes, total_improvement,
                      CASE WHEN attempts > 0
                           THEN successes::float / attempts
                           ELSE 0 END as success_rate,
                      CASE WHEN successes > 0
                           THEN total_improvement / successes
                           ELSE 0 END as avg_improvement
               FROM modification.strategy_stats
               ORDER BY success_rate DESC"""
        )
        return rows

    def suggest_strategy(self, feedback_category: str,
                          target_parameter: str = "") -> str:
        """Use UCB1 to select the best strategy for a given feedback type.

        Balances exploitation (high success rate) vs exploration (untried strategies).
        Falls back to random selection if insufficient data.
        """
        # Get stats for this feedback category
        stats = self._execute(
            """SELECT modification_strategy, attempts, successes,
                      CASE WHEN attempts > 0
                           THEN successes::float / attempts
                           ELSE 0 END as success_rate
               FROM modification.strategy_stats
               WHERE feedback_category = :cat""",
            {"cat": feedback_category}
        )

        # Build strategy stats map
        stats_map = {s["modification_strategy"]: s for s in stats}
        total_attempts = sum(s.get("attempts", 0) for s in stats)

        # If insufficient data, use random selection
        if total_attempts < MIN_SAMPLE_SIZE:
            chosen = random.choice(ALL_STRATEGIES)
            logger.debug(f"meta_learning: insufficient data ({total_attempts}), random: {chosen}")
            return chosen

        # UCB1 selection
        best_strategy = None
        best_ucb = -1.0

        for strategy in ALL_STRATEGIES:
            s = stats_map.get(strategy)
            if s is None or s["attempts"] == 0:
                # Untried strategy gets infinite UCB (must explore)
                best_strategy = strategy
                best_ucb = float("inf")
                break

            success_rate = s["success_rate"]
            attempts = s["attempts"]

            # UCB1 formula
            exploration_bonus = EXPLORATION_WEIGHT * math.sqrt(
                math.log(total_attempts) / attempts
            )
            ucb_score = success_rate + exploration_bonus

            if ucb_score > best_ucb:
                best_ucb = ucb_score
                best_strategy = strategy

        logger.info(f"meta_learning: suggested {best_strategy} for {feedback_category} "
                    f"(UCB={best_ucb:.3f}, total_attempts={total_attempts})")
        return best_strategy or random.choice(ALL_STRATEGIES)

    def generate_meta_report(self) -> str:
        """Generate human-readable summary of meta-learning state."""
        stats = self.get_strategy_stats()
        if not stats:
            return "No modification data yet."

        report = "📈 Meta-Learning Report\n\n"
        report += f"{'Category':<20} {'Strategy':<25} {'Rate':<8} {'N':<5} {'Avg Δ':<8}\n"
        report += "─" * 66 + "\n"

        for s in stats:
            rate = f"{s.get('success_rate', 0):.0%}"
            avg_imp = f"{s.get('avg_improvement', 0):.3f}"
            report += (f"{s['feedback_category']:<20} "
                       f"{s['modification_strategy']:<25} "
                       f"{rate:<8} "
                       f"{s['attempts']:<5} "
                       f"{avg_imp:<8}\n")

        return report
