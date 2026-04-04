"""Control Plane — organizational layer for AndrusAI.

Provides: budget enforcement, ticket tracking, immutable audit trail,
governance gates, multi-project isolation, org chart, and heartbeats.

All tables live in the `control_plane` PostgreSQL schema (existing Mem0 instance).
"""
from app.control_plane.db import get_pool
from app.control_plane.audit import AuditTrail, get_audit
from app.control_plane.tickets import TicketManager, get_tickets
from app.control_plane.budgets import BudgetEnforcer, get_budget_enforcer
from app.control_plane.projects import ProjectManager, get_projects
from app.control_plane.governance import GovernanceGate, get_governance
from app.control_plane.org_chart import get_org_chart
from app.control_plane.cost_tracker import estimate_cost

__all__ = [
    "get_pool", "AuditTrail", "get_audit",
    "TicketManager", "get_tickets",
    "BudgetEnforcer", "get_budget_enforcer",
    "ProjectManager", "get_projects",
    "GovernanceGate", "get_governance",
    "get_org_chart", "estimate_cost",
]
