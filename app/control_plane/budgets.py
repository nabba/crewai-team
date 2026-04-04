"""Budget enforcement — atomic per-agent/per-project spending limits.

Budget checks happen at INFRASTRUCTURE level (llm_factory), not inside agent code.
Agents cannot bypass, modify, or access budget internals. (DGM safety invariant)
"""
import logging
import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from app.control_plane.db import execute, execute_one, execute_scalar

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when an LLM call would exceed the budget."""
    def __init__(self, agent_role: str, period: str, reason: str = ""):
        self.agent_role = agent_role
        self.period = period
        super().__init__(reason or f"Budget exceeded for {agent_role} in {period}")


def _current_period() -> str:
    """Return current period string: '2026-04'."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


class BudgetEnforcer:
    """Atomic budget enforcement integrated at the LLM factory level."""

    def __init__(self):
        self._audit = None

    @property
    def audit(self):
        if self._audit is None:
            from app.control_plane.audit import get_audit
            self._audit = get_audit()
        return self._audit

    def check_and_record(
        self,
        project_id: str,
        agent_role: str,
        estimated_cost_usd: float,
        estimated_tokens: int = 0,
    ) -> tuple[bool, Optional[str]]:
        """Atomically check budget and record spend.

        Returns (allowed: bool, reason: Optional[str]).
        Called by llm_factory BEFORE every LLM API call.
        """
        period = _current_period()
        try:
            allowed = execute_scalar(
                "SELECT control_plane.record_spend(%s, %s, %s, %s, %s)",
                (project_id, agent_role, period, estimated_cost_usd, estimated_tokens),
            )
        except Exception as e:
            logger.warning(f"Budget check failed (allowing): {e}")
            return True, None  # Fail-open: if budget system is down, don't block work

        if allowed is False:
            reason = f"Budget exceeded for {agent_role} in {period}"
            self.audit.log(
                actor="system", action="budget.exceeded",
                project_id=project_id,
                resource_type="budget",
                detail={"agent_role": agent_role, "period": period,
                        "estimated_cost": estimated_cost_usd},
            )
            logger.warning(f"BUDGET EXCEEDED: {reason}")
            return False, reason

        return True, None

    def get_status(self, project_id: str = None) -> list[dict]:
        """Dashboard: current budget status for all agents in a project."""
        period = _current_period()
        if project_id:
            rows = execute(
                """SELECT agent_role, period, limit_usd, spent_usd,
                          limit_tokens, spent_tokens, is_paused,
                          ROUND(spent_usd / NULLIF(limit_usd, 0) * 100, 1) as pct_used
                   FROM control_plane.budgets
                   WHERE project_id = %s AND period = %s
                   ORDER BY agent_role""",
                (project_id, period), fetch=True,
            )
        else:
            rows = execute(
                """SELECT b.agent_role, b.period, b.limit_usd, b.spent_usd,
                          b.limit_tokens, b.spent_tokens, b.is_paused,
                          ROUND(b.spent_usd / NULLIF(b.limit_usd, 0) * 100, 1) as pct_used,
                          p.name as project_name
                   FROM control_plane.budgets b
                   JOIN control_plane.projects p ON b.project_id = p.id
                   WHERE b.period = %s
                   ORDER BY p.name, b.agent_role""",
                (period,), fetch=True,
            )
        return rows or []

    def set_budget(self, project_id: str, agent_role: str,
                   limit_usd: float, limit_tokens: int = None) -> None:
        """Set or update a budget for an agent in the current period."""
        period = _current_period()
        execute(
            """INSERT INTO control_plane.budgets
               (project_id, agent_role, period, limit_usd, limit_tokens)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (project_id, agent_role, period) DO UPDATE
               SET limit_usd = EXCLUDED.limit_usd,
                   limit_tokens = COALESCE(EXCLUDED.limit_tokens, control_plane.budgets.limit_tokens),
                   updated_at = NOW()""",
            (project_id, agent_role, period, limit_usd, limit_tokens),
        )
        self.audit.log(
            actor="user", action="budget.set",
            project_id=project_id,
            resource_type="budget",
            detail={"agent_role": agent_role, "limit_usd": limit_usd, "period": period},
        )

    def override_budget(self, project_id: str, agent_role: str,
                        new_limit: float, approver: str = "user") -> None:
        """Override: increase budget and unpause agent."""
        period = _current_period()
        execute(
            """UPDATE control_plane.budgets
               SET limit_usd = %s, is_paused = FALSE, updated_at = NOW()
               WHERE project_id = %s
                 AND (agent_role = %s OR agent_role IS NULL)
                 AND period = %s""",
            (new_limit, project_id, agent_role, period),
        )
        self.audit.log(
            actor=approver, action="budget.override",
            project_id=project_id,
            resource_type="budget",
            detail={"agent_role": agent_role, "new_limit": new_limit, "period": period},
        )

    def ensure_default_budgets(self, project_id: str, default_limit: float = 50.0) -> None:
        """Create default budgets for all agents in a project if not exists."""
        period = _current_period()
        agents = execute(
            "SELECT agent_role FROM control_plane.org_chart ORDER BY sort_order",
            fetch=True,
        )
        for agent in (agents or []):
            role = agent["agent_role"]
            existing = execute_scalar(
                """SELECT id FROM control_plane.budgets
                   WHERE project_id = %s AND agent_role = %s AND period = %s""",
                (project_id, role, period),
            )
            if not existing:
                execute(
                    """INSERT INTO control_plane.budgets
                       (project_id, agent_role, period, limit_usd)
                       VALUES (%s, %s, %s, %s)""",
                    (project_id, role, period, default_limit),
                )

    def format_status(self, project_id: str = None) -> str:
        """Human-readable budget status for Signal."""
        rows = self.get_status(project_id)
        if not rows:
            return "No budgets configured. Use `budget set <agent> <amount>` to set limits."
        lines = ["💰 Budget Status:"]
        for r in rows:
            role = r.get("agent_role") or "project-wide"
            limit_usd = float(r.get("limit_usd") or 0)
            spent = float(r.get("spent_usd") or 0)
            pct = float(r.get("pct_used") or 0)
            paused = "⏸️ PAUSED" if r.get("is_paused") else ""
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            lines.append(f"  {role}: ${spent:.2f}/${limit_usd:.2f} [{bar}] {pct:.0f}% {paused}")
        return "\n".join(lines)


# ── Singleton ────────────────────────────────────────────────────────────────

_enforcer: Optional[BudgetEnforcer] = None
_lock = threading.Lock()


def get_budget_enforcer() -> BudgetEnforcer:
    global _enforcer
    if _enforcer is None:
        with _lock:
            if _enforcer is None:
                _enforcer = BudgetEnforcer()
    return _enforcer
