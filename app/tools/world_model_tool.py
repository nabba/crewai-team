"""
world_model_tool.py — Tool for agents to record causal observations.

Agents use this tool during task execution to note cause→effect patterns
they discover.  These observations are stored in the world model memory
(L2 self-awareness) and can be recalled by future tasks to improve
decision-making.
"""

import logging
from crewai.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


class WorldModelTool(BaseTool):
    name: str = "store_causal_observation"
    description: str = (
        "Record a cause-and-effect observation you discovered during this task. "
        "Use this when you notice a pattern, dependency, or consequence that "
        "would be useful for future tasks. "
        "Args: cause (str) - what triggered the effect, "
        "effect (str) - what happened as a result, "
        "confidence (str) - 'high', 'medium', or 'low'."
    )
    agent_role: str = Field(default="default")

    def _run(
        self,
        cause: str,
        effect: str,
        confidence: str = "medium",
    ) -> str:
        # Normalize confidence
        confidence = confidence.lower().strip()
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        try:
            from app.self_awareness.world_model import store_causal_belief
            store_causal_belief(
                cause=cause[:500],
                effect=effect[:500],
                confidence=confidence,
                source=f"observed by {self.agent_role}",
            )
            return (
                f"Causal observation recorded: {cause[:100]} → {effect[:100]} "
                f"[confidence={confidence}]"
            )
        except Exception as e:
            logger.warning(f"Failed to store causal observation: {e}", exc_info=True)
            return f"Failed to store observation: {e}"


def create_world_model_tool(role: str) -> WorldModelTool:
    """Factory to create a world model tool configured for a specific agent role."""
    return WorldModelTool(agent_role=role)
