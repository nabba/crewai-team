"""PROGRAM §46.3 — Q8.3 workflow_template primitive tests.

Covers:

  1. models — WorkflowTemplate / WorkflowNode / WorkflowRun JSON
     round-trip + RunStatus enum.
  2. validator — cycle detection, missing-dep refusal, ${node.field}
     forward-ref refusal, undeclared input refusal, duplicate node id
     refusal.
  3. store — save / get / list_all / delete + counter persistence.
  4. executor — topological order, reference resolution, failure
     short-circuit (downstream nodes skipped).
  5. queue — enqueue persists immediately, async execution mutates
     status, wait_for_run blocks until terminal.
  6. Source-level wiring (REST + agent tool + React + App route).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from app.workflows import (
    InvalidWorkflow,
    RunStatus,
    WorkflowNode,
    WorkflowRun,
    WorkflowTemplate,
    enqueue,
    get,
    list_all,
    reset_for_tests,
    save,
    validate_template,
)
from app.workflows.executor import execute_run
from app.workflows.queue import wait_for_run


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path):
    reset_for_tests(tmp_path)
    yield
    reset_for_tests(None)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────
#   1. models
# ─────────────────────────────────────────────────────────────────────


def test_template_round_trip() -> None:
    t = WorkflowTemplate(
        id="t1",
        name="echo workflow",
        description="trivial",
        nodes=[
            WorkflowNode(
                id="n1", tool_name="echo",
                args={"text": "{greeting}"},
            ),
        ],
        inputs=["greeting"],
        created_at=_now_iso(),
    )
    d = t.to_dict()
    t2 = WorkflowTemplate.from_dict(d)
    assert t2.id == "t1"
    assert t2.nodes[0].args == {"text": "{greeting}"}
    assert t2.inputs == ["greeting"]


def test_run_status_terminal_predicate() -> None:
    r = WorkflowRun(
        id="r1", template_id="t1",
        status=RunStatus.RUNNING, started_at=_now_iso(),
    )
    assert not r.is_terminal
    r.status = RunStatus.SUCCEEDED
    assert r.is_terminal
    r.status = RunStatus.CANCELLED
    assert r.is_terminal


# ─────────────────────────────────────────────────────────────────────
#   2. validator
# ─────────────────────────────────────────────────────────────────────


def _t(nodes, inputs=None, *, tid="t1", name="t") -> WorkflowTemplate:
    return WorkflowTemplate(
        id=tid, name=name, description="",
        nodes=list(nodes), inputs=list(inputs or []),
        created_at=_now_iso(),
    )


def test_validate_refuses_duplicate_node_id() -> None:
    t = _t([
        WorkflowNode(id="a", tool_name="echo"),
        WorkflowNode(id="a", tool_name="echo"),
    ])
    with pytest.raises(InvalidWorkflow, match="duplicate node id"):
        validate_template(t)


def test_validate_refuses_unknown_depends_on() -> None:
    t = _t([
        WorkflowNode(id="a", tool_name="echo", depends_on=["ghost"]),
    ])
    with pytest.raises(InvalidWorkflow, match="depends_on unknown node"):
        validate_template(t)


def test_validate_refuses_cycle() -> None:
    t = _t([
        WorkflowNode(id="a", tool_name="echo", depends_on=["b"]),
        WorkflowNode(id="b", tool_name="echo", depends_on=["a"]),
    ])
    with pytest.raises(InvalidWorkflow, match="cycle"):
        validate_template(t)


def test_validate_refuses_forward_node_ref() -> None:
    """A node can only reference earlier (topologically) nodes via
    ${nid.field}."""
    t = _t([
        # node 'a' references 'b' but doesn't depend on it
        WorkflowNode(
            id="a", tool_name="echo", args={"text": "${b.output}"},
        ),
        WorkflowNode(id="b", tool_name="echo", depends_on=["a"]),
    ])
    with pytest.raises(InvalidWorkflow, match="not an ancestor"):
        validate_template(t)


def test_validate_refuses_undeclared_input_placeholder() -> None:
    t = _t([
        WorkflowNode(id="a", tool_name="echo", args={"text": "{missing}"}),
    ], inputs=[])
    with pytest.raises(InvalidWorkflow, match="not declared"):
        validate_template(t)


def test_validate_accepts_well_formed_template() -> None:
    t = _t([
        WorkflowNode(id="a", tool_name="echo", args={"text": "{greeting}"}),
        WorkflowNode(
            id="b", tool_name="echo",
            args={"text": "Got: ${a}"},
            depends_on=["a"],
        ),
    ], inputs=["greeting"])
    validate_template(t)  # no raise


def test_validate_refuses_empty_id_or_name() -> None:
    with pytest.raises(InvalidWorkflow, match="id is empty"):
        validate_template(WorkflowTemplate(
            id="", name="x", description="",
            created_at=_now_iso(),
        ))
    with pytest.raises(InvalidWorkflow, match="name is empty"):
        validate_template(WorkflowTemplate(
            id="x", name="", description="",
            created_at=_now_iso(),
        ))


# ─────────────────────────────────────────────────────────────────────
#   3. store
# ─────────────────────────────────────────────────────────────────────


def test_store_save_get_list() -> None:
    t = _t([WorkflowNode(id="n", tool_name="echo")])
    save(t)
    assert get(t.id) is t
    items = list_all()
    assert any(x.id == t.id for x in items)


def test_store_persistence_survives_index_reset(tmp_path: Path) -> None:
    t = _t([WorkflowNode(id="n", tool_name="echo")], tid="persistent")
    save(t)
    # Drop the in-memory cache; the JSON file should re-hydrate it.
    reset_for_tests(tmp_path)
    again = get("persistent")
    assert again is not None
    assert again.nodes[0].id == "n"


def test_store_record_run_outcome_updates_counters() -> None:
    from app.workflows.store import record_run_outcome
    t = _t([WorkflowNode(id="n", tool_name="echo")])
    save(t)
    assert t.run_count == 0
    assert t.success_count == 0
    record_run_outcome(t.id, success=True, finished_at=_now_iso())
    record_run_outcome(t.id, success=False, finished_at=_now_iso())
    saved = get(t.id)
    assert saved.run_count == 2
    assert saved.success_count == 1


# ─────────────────────────────────────────────────────────────────────
#   4. executor
# ─────────────────────────────────────────────────────────────────────


def test_executor_runs_nodes_in_topological_order() -> None:
    t = _t([
        WorkflowNode(id="a", tool_name="echo", args={"text": "{x}"}),
        WorkflowNode(
            id="b", tool_name="echo",
            args={"text": "from-a:${a}"},
            depends_on=["a"],
        ),
    ], inputs=["x"])
    save(t)
    run = WorkflowRun(
        id="run-1", template_id=t.id,
        status=RunStatus.QUEUED, started_at=_now_iso(),
        inputs={"x": "hello"},
    )

    call_order: list[str] = []

    def fake_dispatcher(tool_name: str, args: dict[str, Any]):
        call_order.append(tool_name + "(" + str(args) + ")")
        return args.get("text", "")

    execute_run(t, run, tool_dispatcher=fake_dispatcher)
    assert run.status == RunStatus.SUCCEEDED
    assert run.node_outputs["a"] == "hello"
    assert run.node_outputs["b"] == "from-a:hello"
    # 'a' was called before 'b'
    assert call_order[0].startswith("echo({'text': 'hello'}")


def test_executor_propagates_failure_and_skips_downstream() -> None:
    t = _t([
        WorkflowNode(id="a", tool_name="echo"),
        WorkflowNode(id="b", tool_name="echo", depends_on=["a"]),
        WorkflowNode(id="c", tool_name="echo", depends_on=["b"]),
    ])
    save(t)
    run = WorkflowRun(
        id="run-2", template_id=t.id,
        status=RunStatus.QUEUED, started_at=_now_iso(),
    )

    def fake_dispatcher(tool_name: str, args: dict[str, Any]):
        # Simulate first node succeeding, second crashing
        if "b" in str(args) or args == {} and run.node_statuses.get("a") == "succeeded":
            raise RuntimeError("b crashed")
        return "ok"

    # Simpler: bind failure to node id by lookup over node_statuses
    counter = {"calls": 0}

    def dispatcher(tool_name, args):
        counter["calls"] += 1
        # Fail on the 2nd call (which will be node 'b' in topological order)
        if counter["calls"] == 2:
            raise RuntimeError("b crashed")
        return "ok"

    execute_run(t, run, tool_dispatcher=dispatcher)
    assert run.status == RunStatus.FAILED
    assert run.error_node == "b"
    assert run.node_statuses["a"] == "succeeded"
    assert run.node_statuses["b"] == "failed"
    # 'c' should be skipped, never called
    assert run.node_statuses.get("c") == "skipped"
    assert counter["calls"] == 2  # never called for 'c'


def test_executor_resolves_input_placeholders() -> None:
    t = _t([
        WorkflowNode(
            id="a", tool_name="echo",
            args={"prefix": "{salutation}", "suffix": "{name}"},
        ),
    ], inputs=["salutation", "name"])
    save(t)
    run = WorkflowRun(
        id="run-3", template_id=t.id,
        status=RunStatus.QUEUED, started_at=_now_iso(),
        inputs={"salutation": "Hello", "name": "World"},
    )

    captured = {}

    def dispatcher(_name, args):
        captured["args"] = args
        return args

    execute_run(t, run, tool_dispatcher=dispatcher)
    assert captured["args"]["prefix"] == "Hello"
    assert captured["args"]["suffix"] == "World"


def test_executor_resolves_nested_node_field_references() -> None:
    t = _t([
        WorkflowNode(id="a", tool_name="echo"),
        WorkflowNode(
            id="b", tool_name="echo",
            args={"text": "field=${a.greeting}"},
            depends_on=["a"],
        ),
    ])
    save(t)
    run = WorkflowRun(
        id="run-4", template_id=t.id,
        status=RunStatus.QUEUED, started_at=_now_iso(),
    )

    def dispatcher(_n, args):
        # node 'a' returns a dict; node 'b' references a.greeting
        if not args:
            return {"greeting": "hi", "noise": "ignored"}
        return args.get("text", "")

    execute_run(t, run, tool_dispatcher=dispatcher)
    assert run.status == RunStatus.SUCCEEDED
    assert run.node_outputs["b"] == "field=hi"


# ─────────────────────────────────────────────────────────────────────
#   5. queue
# ─────────────────────────────────────────────────────────────────────


def test_queue_enqueue_returns_immediately() -> None:
    """enqueue persists QUEUED and returns the run record before the
    worker thread runs to completion."""
    t = _t([WorkflowNode(id="n", tool_name="echo")])
    save(t)

    def slow_dispatcher(_name, _args):
        time.sleep(0.05)
        return "done"

    run = enqueue(t, inputs={}, tool_dispatcher=slow_dispatcher)
    # Status may be QUEUED or RUNNING — depends on scheduling — but
    # NOT terminal yet.
    assert run.status in (RunStatus.QUEUED, RunStatus.RUNNING)

    # Block for completion
    final = wait_for_run(run.id, timeout=5.0)
    assert final is not None
    assert final.status == RunStatus.SUCCEEDED
    assert final.node_outputs["n"] == "done"


def test_queue_run_persists_to_disk() -> None:
    """Run JSON is written under runs_dir() so a gateway restart can
    surface the run history."""
    t = _t([WorkflowNode(id="n", tool_name="echo")])
    save(t)
    run = enqueue(t, inputs={}, tool_dispatcher=lambda _n, _a: "x")
    wait_for_run(run.id, timeout=5.0)
    from app.workflows.store import runs_dir
    files = list(runs_dir().glob("*.json"))
    assert any(f.stem == run.id for f in files)


def test_queue_list_runs_filters_by_template() -> None:
    t1 = _t([WorkflowNode(id="n", tool_name="echo")], tid="t-A", name="A")
    t2 = _t([WorkflowNode(id="n", tool_name="echo")], tid="t-B", name="B")
    save(t1)
    save(t2)
    enqueue(t1, tool_dispatcher=lambda _n, _a: "x")
    enqueue(t2, tool_dispatcher=lambda _n, _a: "x")
    # Wait briefly for both
    time.sleep(0.2)
    from app.workflows import list_runs
    runs_a = list_runs(template_id="t-A")
    assert all(r.template_id == "t-A" for r in runs_a)


# ─────────────────────────────────────────────────────────────────────
#   6. Source-level wiring
# ─────────────────────────────────────────────────────────────────────


def test_rest_router_mounted() -> None:
    src = Path("app/main.py").read_text(encoding="utf-8")
    assert "from app.control_plane.workflows_api import router as workflows_router" in src
    assert "app.include_router(workflows_router)" in src


def test_rest_endpoints_exist() -> None:
    src = Path("app/control_plane/workflows_api.py").read_text(encoding="utf-8")
    for route in (
        "/api/cp/workflows",
        '@router.get("")', '@router.get("/{template_id}")',
        '@router.post("")', '@router.delete("/{template_id}")',
        '@router.post("/{template_id}/run")',
        '@router.get("/runs")',
        '@router.get("/runs/{run_id}")',
        '@router.post("/runs/{run_id}/cancel")',
    ):
        assert route in src, f"missing {route}"


def test_agent_tool_module_exists_and_exports_factory() -> None:
    src = Path("app/tools/workflow_tools.py").read_text(encoding="utf-8")
    assert "def create_workflow_tools(" in src
    assert "class WorkflowRunTool" in src
    assert "class WorkflowListTool" in src
    assert "name: str = \"workflow_run\"" in src
    assert "name: str = \"workflow_list\"" in src


def test_react_page_mounted() -> None:
    app_src = Path("dashboard-react/src/App.tsx").read_text(encoding="utf-8")
    assert "WorkflowsPage" in app_src
    assert 'path="/workflows"' in app_src
    layout_src = Path("dashboard-react/src/components/Layout.tsx").read_text(encoding="utf-8")
    assert "/workflows" in layout_src


def test_react_api_module_exposes_hooks() -> None:
    src = Path("dashboard-react/src/api/workflows.ts").read_text(encoding="utf-8")
    for hook in (
        "useWorkflowsListQuery",
        "useWorkflowRunsQuery",
        "useWorkflowRunStatusQuery",
        "useStartWorkflowMutation",
        "useCancelWorkflowRunMutation",
    ):
        assert hook in src
