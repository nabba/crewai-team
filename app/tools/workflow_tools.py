"""Agent-callable workflow tools.

PROGRAM §46.3 (Q8.3). Two CrewAI tools:

  * ``workflow_run`` — enqueue a workflow template by name and wait
    up to a budget for it to finish (default 60s). Returns the run
    record or a "still running, poll /api/cp/workflows/runs/<id>"
    message.
  * ``workflow_list`` — list available workflow templates.

The brainstorm + skills + tool_registry surfaces are unchanged.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Type

try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field
    _CREWAI_AVAILABLE = True
except ImportError:  # pragma: no cover — gateway-deps
    BaseTool = object  # type: ignore[misc,assignment]
    BaseModel = object  # type: ignore[misc,assignment]
    def Field(*a, **k):  # type: ignore[no-redef]
        return None
    _CREWAI_AVAILABLE = False

logger = logging.getLogger(__name__)


_DEFAULT_WAIT_S = 60


def _build_tools() -> list:
    if not _CREWAI_AVAILABLE:
        return []

    class _RunInput(BaseModel):
        template_name: str = Field(
            description=(
                "Workflow template name (case-sensitive). Use "
                "workflow_list to discover available templates."
            ),
        )
        inputs: dict[str, Any] = Field(
            default_factory=dict,
            description=(
                "Map of {input_name: value} for placeholder "
                "substitution in node args."
            ),
        )
        wait_seconds: int = Field(
            default=_DEFAULT_WAIT_S,
            ge=0, le=600,
            description=(
                "How long to block waiting for completion. 0 = "
                "return run_id immediately."
            ),
        )

    class WorkflowRunTool(BaseTool):
        name: str = "workflow_run"
        description: str = (
            "Run a saved workflow template by name. Workflows compose "
            "existing registered tools into a DAG; the result of each "
            "node is available to downstream nodes via ${node_id.field} "
            "references. Returns the run id + status; for long-running "
            "workflows set wait_seconds=0 and poll "
            "/api/cp/workflows/runs/<id>."
        )
        args_schema: Type[BaseModel] = _RunInput

        def _run(
            self,
            template_name: str,
            inputs: dict[str, Any] | None = None,
            wait_seconds: int = _DEFAULT_WAIT_S,
        ) -> str:
            try:
                from app.workflows import enqueue, list_all
                from app.workflows.queue import wait_for_run
            except Exception as exc:
                return f"ERROR workflow subsystem unavailable: {exc}"

            templates = list_all(limit=500)
            template = next(
                (t for t in templates if t.name == template_name),
                None,
            )
            if template is None:
                names = sorted(t.name for t in templates)[:10]
                return (
                    f"ERROR template {template_name!r} not found. "
                    f"Known: {names}"
                )
            try:
                run = enqueue(template, inputs=dict(inputs or {}))
            except Exception as exc:
                return f"ERROR enqueue failed: {exc}"

            if wait_seconds <= 0:
                return json.dumps({
                    "run_id": run.id,
                    "status": run.status.value,
                    "poll": f"/api/cp/workflows/runs/{run.id}",
                }, indent=2)

            final = wait_for_run(run.id, timeout=float(wait_seconds))
            if final is None:
                return f"ERROR run {run.id} lost"
            return json.dumps({
                "run_id": final.id,
                "status": final.status.value,
                "node_statuses": final.node_statuses,
                "node_outputs": final.node_outputs,
                "error": final.error,
                "error_node": final.error_node,
                "finished_at": final.finished_at,
                "is_terminal": final.is_terminal,
            }, indent=2)

    class _ListInput(BaseModel):
        pass

    class WorkflowListTool(BaseTool):
        name: str = "workflow_list"
        description: str = (
            "List saved workflow templates. Returns name + "
            "description + declared inputs for each."
        )
        args_schema: Type[BaseModel] = _ListInput

        def _run(self) -> str:
            try:
                from app.workflows import list_all
                items = list_all(limit=200)
            except Exception as exc:
                return f"ERROR workflow subsystem unavailable: {exc}"
            return json.dumps([
                {
                    "name": t.name,
                    "description": t.description,
                    "inputs": list(t.inputs),
                    "node_count": len(t.nodes),
                    "run_count": t.run_count,
                    "success_count": t.success_count,
                }
                for t in items
            ], indent=2)

    return [WorkflowRunTool(), WorkflowListTool()]


def create_workflow_tools() -> list:
    """Factory called by the tool-registry boot path."""
    try:
        return _build_tools()
    except Exception:
        logger.debug("workflow tools build failed", exc_info=True)
        return []
