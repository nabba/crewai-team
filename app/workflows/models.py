"""Workflow template + run data model.

PROGRAM §46.3 — Q8.3. A WorkflowTemplate is a saved DAG description;
a WorkflowRun is one instance of that DAG executing.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class InvalidWorkflow(ValueError):
    """Raised by ``validate_template`` on cycles / missing tools /
    unresolvable references / etc."""


class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowNode:
    """One step in the workflow DAG.

    ``args`` is a JSON-serialisable mapping. Values may contain
    ``${node_id.field}`` references that the executor resolves against
    earlier nodes' outputs, OR ``{arg_name}`` placeholders that the
    caller fills via :meth:`WorkflowTemplate.runtime_args` at run time.
    """
    id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    # Optional human-readable description; surfaced in React.
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "args": dict(self.args),
            "depends_on": list(self.depends_on),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowNode":
        return cls(
            id=str(d["id"]),
            tool_name=str(d["tool_name"]),
            args=dict(d.get("args") or {}),
            depends_on=list(d.get("depends_on") or []),
            description=str(d.get("description", "") or ""),
        )


@dataclass
class WorkflowTemplate:
    """JSON-saveable DAG of tool calls."""
    id: str
    name: str
    description: str
    nodes: list[WorkflowNode] = field(default_factory=list)
    # Declared inputs (substituted via ``{arg_name}`` in node args).
    inputs: list[str] = field(default_factory=list)
    created_at: str = ""
    last_run_at: str | None = None
    run_count: int = 0
    success_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "inputs": list(self.inputs),
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "run_count": self.run_count,
            "success_count": self.success_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowTemplate":
        return cls(
            id=str(d["id"]),
            name=str(d["name"]),
            description=str(d.get("description", "") or ""),
            nodes=[WorkflowNode.from_dict(n) for n in d.get("nodes") or []],
            inputs=list(d.get("inputs") or []),
            created_at=str(d.get("created_at", "") or ""),
            last_run_at=d.get("last_run_at"),
            run_count=int(d.get("run_count", 0) or 0),
            success_count=int(d.get("success_count", 0) or 0),
        )


@dataclass
class WorkflowRun:
    """One execution instance of a WorkflowTemplate.

    Persisted at ``workspace/workflows/runs/<run_id>.json`` so a
    gateway restart doesn't lose history. ``node_outputs`` carries
    per-node return values (truncated for storage)."""
    id: str
    template_id: str
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    node_outputs: dict[str, Any] = field(default_factory=dict)
    node_statuses: dict[str, str] = field(default_factory=dict)
    error: str = ""
    error_node: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "template_id": self.template_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "inputs": dict(self.inputs),
            "node_outputs": dict(self.node_outputs),
            "node_statuses": dict(self.node_statuses),
            "error": self.error,
            "error_node": self.error_node,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowRun":
        return cls(
            id=str(d["id"]),
            template_id=str(d["template_id"]),
            status=RunStatus(d["status"]),
            started_at=str(d["started_at"]),
            finished_at=d.get("finished_at"),
            inputs=dict(d.get("inputs") or {}),
            node_outputs=dict(d.get("node_outputs") or {}),
            node_statuses=dict(d.get("node_statuses") or {}),
            error=str(d.get("error", "") or ""),
            error_node=str(d.get("error_node", "") or ""),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED,
        }
