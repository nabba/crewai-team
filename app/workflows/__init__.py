"""Workflow templates — compose registered tools into a JSON DAG.

PROGRAM §46.3 (Q8.3). Sits between:

  * **Tool registry** (single-tool discovery via capability)
  * **Skills registry** (single ``task_template`` → one
    ``Commander.handle()`` invocation)
  * **Forge** (synthesise NEW tool source code)

A workflow_template describes a DAG of tool calls in JSON. Each node
runs one registered tool; later nodes can reference earlier nodes'
output via ``${node_id.field}`` placeholders. No new tool code is
generated — the workflow is a recipe over existing capabilities.

Public surface:

  * :mod:`app.workflows.models` — ``WorkflowTemplate`` + ``WorkflowNode``
    + ``WorkflowRun``.
  * :mod:`app.workflows.store` — JSON-per-record persistence.
  * :mod:`app.workflows.validator` — DAG cycle check + reference
    resolution + tool-exists check.
  * :mod:`app.workflows.queue` — async run queue (thread-backed,
    bounded).
  * :mod:`app.workflows.executor` — topologically-sorted node
    execution.

Cross-cuts (wired in separate modules):

  * REST: ``app/control_plane/workflows_api.py``.
  * Agent tool: ``app/tools/workflow_tools.py`` (``workflow_run``).
  * React: ``dashboard-react/src/components/WorkflowsPage.tsx``.

The Brainstorm subsystem could be reframed as a workflow_template —
that refactor is intentionally NOT done here (PROGRAM §46.3, doc-
only mention). The Python state-machine techniques continue to work
unchanged.
"""
from app.workflows.models import (
    InvalidWorkflow,
    RunStatus,
    WorkflowNode,
    WorkflowRun,
    WorkflowTemplate,
)
from app.workflows.queue import enqueue, get_run, list_runs
from app.workflows.store import (
    get,
    list_all,
    reset_for_tests,
    save,
)
from app.workflows.validator import validate_template

__all__ = [
    "InvalidWorkflow",
    "RunStatus",
    "WorkflowNode",
    "WorkflowRun",
    "WorkflowTemplate",
    "enqueue",
    "get",
    "get_run",
    "list_all",
    "list_runs",
    "reset_for_tests",
    "save",
    "validate_template",
]
