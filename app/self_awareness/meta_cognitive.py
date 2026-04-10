"""
meta_cognitive.py — Meta-Cognitive Layer for AndrusAI agents.

Hyperagent-inspired wrapper that assesses strategy effectiveness,
proposes context modifications, and enforces compute-aware planning.

CRITICAL CONSTRAINTS:
  1. NEVER modifies agent code at runtime. Only modifies CONTEXT.
  2. NEVER touches FREEZE-BLOCK regions, priority 0 hooks, SOUL.md.
  3. Self-modification proposals are LOGGED AND DEFERRED to evolution cycle.
  4. Compute-aware: budget phase determines aggressiveness.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from app.self_awareness.internal_state import InternalState, MetaCognitiveState

logger = logging.getLogger(__name__)


class MetaCognitiveLayer:
    """Lightweight meta-cognitive wrapper for CrewAI agents."""

    def __init__(
        self,
        agent_id: str,
        max_history: int = 20,
        reassessment_cooldown_steps: int = 3,
    ):
        self.agent_id = agent_id
        self.strategy_history: deque[dict] = deque(maxlen=max_history)
        self.modification_log: list[dict] = []
        self.steps_since_reassessment: int = 0
        self.reassessment_cooldown = reassessment_cooldown_steps

    def pre_reasoning_hook(
        self,
        task_context: dict,
        previous_state: Optional[InternalState] = None,
    ) -> tuple[dict, MetaCognitiveState]:
        """Runs BEFORE each reasoning step. Can modify context, not code."""
        meta = MetaCognitiveState()

        # 1. Check compute budget
        budget_info = self._get_budget_info()
        meta.compute_phase = self._compute_phase(budget_info)
        meta.compute_budget_remaining_pct = budget_info.get("remaining_pct", 1.0)

        # 2. Decide whether to reassess strategy
        self.steps_since_reassessment += 1
        should_reassess = self._should_reassess(previous_state, meta.compute_phase)
        meta.reassessment_triggered = should_reassess

        if should_reassess:
            # 3. Assess current strategy
            assessment = self._assess_strategy(task_context, previous_state)
            meta.strategy_assessment = assessment.get("assessment", "uncertain")

            if assessment.get("assessment") == "failing" and meta.compute_phase != "late":
                # 4. Propose context modification
                proposal = self._generate_modification(task_context, assessment, meta.compute_phase)
                if proposal:
                    task_context = self._apply_context_modification(task_context, proposal)
                    meta.modification_proposed = True
                    meta.modification_description = proposal.get("description", "")
                    self._log_modification(proposal)

            self.steps_since_reassessment = 0
        else:
            if self.strategy_history:
                meta.strategy_assessment = self.strategy_history[-1].get("assessment", "not_assessed")

        # 5. Inject meta-cognitive state into context
        task_context["_meta_state"] = {
            "phase": meta.compute_phase,
            "strategy_trend": self._get_strategy_trend(),
            "compute_remaining_pct": meta.compute_budget_remaining_pct,
        }

        # Record to history
        self.strategy_history.append({
            "assessment": meta.strategy_assessment,
            "modification_proposed": meta.modification_proposed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return task_context, meta

    def _should_reassess(self, previous_state: Optional[InternalState], compute_phase: str) -> bool:
        """Conservative: only reassess when signals warrant it."""
        if compute_phase == "late":
            return False
        if self.steps_since_reassessment < self.reassessment_cooldown:
            return False
        if previous_state is None:
            return True
        if previous_state.certainty_trend == "falling":
            return True
        if previous_state.action_disposition in ("pause", "escalate"):
            return True
        if self.strategy_history:
            last = self.strategy_history[-1]
            if last.get("modification_proposed") and previous_state.certainty_trend != "rising":
                return True
        return False

    def _assess_strategy(self, task_context: dict, previous_state: Optional[InternalState]) -> dict:
        """Use local LLM to assess strategy effectiveness. ~100ms."""
        try:
            from app.llm_factory import create_specialist_llm
            from app.utils import safe_json_parse

            llm = create_specialist_llm(max_tokens=100, role="self_improve", force_tier="local")

            state_summary = previous_state.to_context_string() if previous_state else "No prior state"
            recent_history = list(self.strategy_history)[-5:]

            prompt = (
                f"Assess the effectiveness of the current strategy for this agent task.\n\n"
                f"Current state: {state_summary}\n"
                f"Recent history: {json.dumps(recent_history, default=str)}\n"
                f"Task: {task_context.get('description', 'unknown')[:300]}\n\n"
                'Respond ONLY with JSON: {"assessment": "effective"|"uncertain"|"failing", "reason": "brief"}'
            )
            raw = str(llm.call(prompt)).strip()
            result, _ = safe_json_parse(raw)
            if result and result.get("assessment") in ("effective", "uncertain", "failing"):
                return result
            return {"assessment": "uncertain", "reason": "unparseable"}
        except Exception as e:
            logger.debug(f"Strategy assessment failed for {self.agent_id}: {e}")
            return {"assessment": "uncertain", "reason": "assessment_error"}

    def _generate_modification(self, task_context: dict, assessment: dict, compute_phase: str) -> Optional[dict]:
        """Generate context modification proposal. APPEND-ONLY, never delete."""
        allowed = ["refine_task_description", "adjust_tool_selection", "add_strategy_hint"]
        if compute_phase == "mid":
            allowed = ["add_strategy_hint"]

        try:
            from app.llm_factory import create_specialist_llm
            from app.utils import safe_json_parse

            llm = create_specialist_llm(max_tokens=200, role="self_improve", force_tier="local")
            prompt = (
                f"Current strategy assessed as: {assessment.get('assessment')}\n"
                f"Reason: {assessment.get('reason', 'unknown')}\n\n"
                f"Propose ONE context modification from: {allowed}\n\n"
                "Rules:\n- Cannot modify safety constraints or SOUL.md\n"
                "- Only suggest additions/refinements to task context\n"
                "- Keep under 100 words\n\n"
                'Respond ONLY with JSON: {"type": "...", "description": "...", "content": "..."}'
            )
            raw = str(llm.call(prompt)).strip()
            proposal, _ = safe_json_parse(raw)
            if proposal and proposal.get("type") in allowed:
                return proposal
            return None
        except Exception:
            return None

    @staticmethod
    def _apply_context_modification(task_context: dict, proposal: dict) -> dict:
        """Apply modification to context. Safe operations only: appending hints."""
        mod_type = proposal.get("type")
        content = proposal.get("content", "")

        if mod_type == "refine_task_description":
            desc = task_context.get("description", "")
            task_context["description"] = f"{desc}\n\n[Meta-cognitive refinement]: {content}"
        elif mod_type == "adjust_tool_selection":
            task_context.setdefault("tool_hints", []).append(content)
        elif mod_type == "add_strategy_hint":
            task_context.setdefault("strategy_hints", []).append(content)

        return task_context

    def _log_modification(self, proposal: dict) -> None:
        """Log for audit trail. Fed to Self-Improver in evolution cycle."""
        entry = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "proposal": proposal,
        }
        self.modification_log.append(entry)

        # Also log to activity journal
        try:
            from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
            get_journal().write(JournalEntry(
                entry_type=JournalEntryType.DECISION,
                summary=f"Meta-cognitive modification: {proposal.get('type', '?')}",
                agents_involved=[self.agent_id],
                details=proposal,
            ))
        except Exception:
            pass

    def _get_budget_info(self) -> dict:
        """Query control plane for current compute budget."""
        try:
            from app.control_plane.budgets import get_status
            statuses = get_status()
            if statuses:
                agent_budget = next((s for s in statuses if s.get("agent_role") == self.agent_id), None)
                if agent_budget:
                    return {"remaining_pct": 1.0 - agent_budget.get("pct_used", 0.0)}
            return {"remaining_pct": 1.0}
        except Exception:
            return {"remaining_pct": 1.0}

    @staticmethod
    def _compute_phase(budget_info: dict) -> str:
        remaining = budget_info.get("remaining_pct", 1.0)
        if remaining > 0.6:
            return "early"
        if remaining > 0.2:
            return "mid"
        return "late"

    def _get_strategy_trend(self) -> str:
        if len(self.strategy_history) < 3:
            return "insufficient_data"
        recent = [h["assessment"] for h in list(self.strategy_history)[-5:]]
        failing = recent.count("failing")
        effective = recent.count("effective")
        if failing > effective:
            return "degrading"
        if effective > failing:
            return "improving"
        return "stable"

    def get_modification_log(self) -> list[dict]:
        return self.modification_log.copy()

    def clear_modification_log(self) -> None:
        self.modification_log.clear()
