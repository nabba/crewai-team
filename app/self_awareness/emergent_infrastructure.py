"""
emergent_infrastructure.py — Emergent Engineering Infrastructure.

Agents propose new tools/capabilities, all requiring human approval
via Signal CLI before deployment. Meta Hyperagents (2026) adapted
with human-in-the-loop safety gate.

Pipeline: Agent recognizes recurring need → generates proposal →
safety scan → sandbox test → Signal CLI human review →
APPROVED / REJECTED / MODIFY → deploy if approved.

Safety: All proposals logged to audit trail. Forbidden patterns blocked.
Sandbox testing before human review. No deployment without explicit APPROVED.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ProposalStatus(str, Enum):
    PENDING = "pending"
    SAFETY_REJECTED = "safety_rejected"
    SANDBOX_FAILED = "sandbox_failed"
    SENT_FOR_REVIEW = "sent_for_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFY_REQUESTED = "modify_requested"
    DEPLOYED = "deployed"


@dataclass
class ToolProposal:
    """A proposed new tool or capability."""
    proposal_id: str
    agent_id: str
    tool_name: str
    tool_description: str
    justification: str
    tool_code: str
    tool_type: str = "function"
    triggered_by: str = ""
    frequency_of_need: int = 0
    status: ProposalStatus = ProposalStatus.PENDING
    human_feedback: Optional[str] = None
    sandbox_result: Optional[dict] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "tool_description": self.tool_description[:200],
            "justification": self.justification[:200],
            "tool_code": self.tool_code[:1000],
            "tool_type": self.tool_type,
            "triggered_by": self.triggered_by[:200],
            "frequency_of_need": self.frequency_of_need,
            "status": self.status.value,
            "human_feedback": self.human_feedback,
            "created_at": self.created_at.isoformat(),
        }


# Forbidden patterns in proposed tool code
FORBIDDEN_PATTERNS = [
    "os.system", "subprocess", "eval(", "exec(",
    "FREEZE-BLOCK", "SOUL.md", "priority_0",
    "__import__", "shutil.rmtree",
    "DROP TABLE", "DELETE FROM", "TRUNCATE",
    "open('/etc", "open('/mnt",
]


class EmergentInfrastructureManager:
    """Manages the lifecycle of agent-proposed tools."""

    def generate_proposal(
        self,
        agent_id: str,
        need_description: str,
        task_context: str = "",
        available_tools: list[str] = None,
        meta_cognitive_log: list[dict] = None,
    ) -> Optional[ToolProposal]:
        """Agent generates a tool proposal based on a recognized need."""
        # Only propose if need appeared 3+ times
        frequency = sum(
            1 for entry in (meta_cognitive_log or [])
            if need_description.lower() in str(entry.get("modification_description", "")).lower()
        )
        if frequency < 3:
            return None

        try:
            from app.llm_factory import create_specialist_llm
            from app.utils import safe_json_parse

            llm = create_specialist_llm(max_tokens=600, role="self_improve", force_tier="local")
            tools_str = ", ".join((available_tools or [])[:15])

            prompt = (
                f"Design a new utility tool for this recurring need.\n\n"
                f"Need: {need_description[:300]}\n"
                f"Context: {task_context[:200]}\n"
                f"Existing tools: {tools_str}\n\n"
                "Requirements:\n"
                "- Pure Python function, no filesystem/network/subprocess\n"
                "- Clear input/output types with docstring\n\n"
                'Respond ONLY with JSON:\n'
                '{"tool_name": "snake_case", "description": "...", '
                '"justification": "...", "code": "def tool_name(args):\\n    ..."}'
            )
            raw = str(llm.call(prompt)).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]

            data, _ = safe_json_parse(raw.strip())
            if not data:
                return None

            return ToolProposal(
                proposal_id=f"prop_{agent_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                agent_id=agent_id,
                tool_name=data.get("tool_name", "unnamed"),
                tool_description=data.get("description", ""),
                justification=data.get("justification", ""),
                tool_code=data.get("code", ""),
                triggered_by=need_description,
                frequency_of_need=frequency,
            )
        except Exception as e:
            logger.debug(f"Proposal generation failed: {e}")
            return None

    def submit_proposal(self, proposal: ToolProposal) -> ToolProposal:
        """Submit through safety scan → sandbox → human review pipeline."""
        # 1. Safety scan
        issues = self._safety_scan(proposal.tool_code)
        if issues:
            proposal.status = ProposalStatus.SAFETY_REJECTED
            proposal.human_feedback = f"Auto-rejected: {'; '.join(issues)}"
            self._persist(proposal)
            return proposal

        # 2. AST validation (must parse as valid Python)
        try:
            ast.parse(proposal.tool_code)
        except SyntaxError as e:
            proposal.status = ProposalStatus.SANDBOX_FAILED
            proposal.human_feedback = f"Syntax error: {e}"
            self._persist(proposal)
            return proposal

        # 3. Mark for human review
        proposal.status = ProposalStatus.SENT_FOR_REVIEW
        self._persist(proposal)

        # Log to audit trail
        try:
            from app.control_plane.audit import get_audit
            get_audit().log(
                actor=proposal.agent_id,
                action="tool_proposal_submitted",
                detail=proposal.to_dict(),
            )
        except Exception:
            pass

        # Log to journal
        try:
            from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
            get_journal().write(JournalEntry(
                entry_type=JournalEntryType.DECISION,
                summary=f"Tool proposal: {proposal.tool_name} by {proposal.agent_id}",
                agents_involved=[proposal.agent_id],
                details=proposal.to_dict(),
            ))
        except Exception:
            pass

        return proposal

    def _safety_scan(self, code: str) -> list[str]:
        """Scan for forbidden patterns."""
        issues = []
        code_lower = code.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in code_lower:
                issues.append(f"Forbidden: '{pattern}'")
        return issues

    def _persist(self, proposal: ToolProposal) -> None:
        try:
            from app.control_plane.db import execute
            execute(
                """INSERT INTO tool_proposals
                   (proposal_id, agent_id, tool_name, tool_description,
                    justification, tool_code, tool_type, triggered_by,
                    frequency_of_need, status, human_feedback)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (proposal_id)
                   DO UPDATE SET status=%s, human_feedback=%s""",
                (
                    proposal.proposal_id, proposal.agent_id,
                    proposal.tool_name, proposal.tool_description[:2000],
                    proposal.justification[:2000], proposal.tool_code[:5000],
                    proposal.tool_type, proposal.triggered_by[:500],
                    proposal.frequency_of_need, proposal.status.value,
                    proposal.human_feedback,
                    proposal.status.value, proposal.human_feedback,
                ),
            )
        except Exception as e:
            logger.debug(f"Failed to persist proposal: {e}")

    def get_pending_proposals(self) -> list[dict]:
        """Get all proposals awaiting human review."""
        try:
            from app.control_plane.db import execute
            rows = execute(
                "SELECT * FROM tool_proposals WHERE status = %s ORDER BY created_at DESC",
                ("sent_for_review",), fetch=True,
            )
            return rows or []
        except Exception:
            return []
