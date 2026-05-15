"""Workflow executor — topologically-sorted node dispatch.

PROGRAM §46.3 (Q8.3). Driven by :mod:`app.workflows.queue` which
runs the executor in a background thread. The executor itself is a
plain function so unit tests can drive it without spinning up the
queue thread.

Reference resolution:

  * ``{input_name}`` in node args → replaced with the run's
    ``inputs`` dict value.
  * ``${node_id}`` → entire output of ``node_id`` (typically a
    string or dict).
  * ``${node_id.field.subfield}`` → nested attribute / dict-key
    access. For unknown fields the result is an empty string (NOT
    a failure) — workflows should validate at design time.

Each node's tool is invoked via :func:`_invoke_tool`, which falls
back through three strategies:

  1. Tool-registry lookup + ``Tool._run(**args)``.
  2. Direct callable lookup in ``app.tools.<tool_name>`` (legacy
     pure-function tools).
  3. A clear ``ToolNotFound`` error captured in the run record.

The executor mutates the passed-in :class:`WorkflowRun` (sets
status / node_outputs / node_statuses / error). The caller (queue)
persists the run after each transition.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.workflows.models import RunStatus, WorkflowRun, WorkflowTemplate
from app.workflows.validator import _topological_order

logger = logging.getLogger(__name__)


_NODE_REF_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)(?:\.([a-zA-Z_][a-zA-Z0-9_.]*))?\}")
# Negative lookbehind on $ so ``${name}`` (node-reference) doesn't
# double-match as ``{name}`` (input placeholder). The node-ref pass
# runs first and substitutes ``${name}`` away, so the lookbehind
# guard is belt-and-suspenders.
_INPUT_REF_RE = re.compile(r"(?<!\$)\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class ToolNotFound(LookupError):
    """Raised when a node's tool_name has no resolver."""


def execute_run(
    template: WorkflowTemplate,
    run: WorkflowRun,
    *,
    tool_dispatcher=None,
    persist_callback=None,
) -> WorkflowRun:
    """Execute the run in-process. Mutates ``run`` and returns it.

    ``tool_dispatcher`` is an injection seam — production passes
    None (uses ``_default_dispatcher``); tests inject a fake that
    returns canned outputs.

    ``persist_callback(run)`` is called after every node-status
    transition so the queue can persist intermediate state."""
    dispatcher = tool_dispatcher or _default_dispatcher
    persist = persist_callback or (lambda r: None)

    run.status = RunStatus.RUNNING
    persist(run)

    try:
        order = _topological_order(template)
    except Exception as exc:
        run.status = RunStatus.FAILED
        run.error = f"DAG order failed: {exc}"
        run.finished_at = _now_iso()
        persist(run)
        return run

    nodes_by_id = {n.id: n for n in template.nodes}

    # Track skipped/failed upstream propagation. We DON'T early-return
    # on first failure — downstream nodes need to be marked "skipped"
    # so the operator-facing run record reflects the full impact.
    # ``run.status`` is updated to FAILED on first failure but we keep
    # walking the rest of the DAG.
    failed_so_far = False

    for nid in order:
        node = nodes_by_id[nid]
        # Skip if any dependency failed OR was itself skipped
        upstream_bad = any(
            run.node_statuses.get(d) in ("failed", "skipped")
            for d in node.depends_on
        )
        if upstream_bad:
            run.node_statuses[nid] = "skipped"
            persist(run)
            continue

        try:
            resolved_args = _resolve_args(
                node.args, inputs=run.inputs, node_outputs=run.node_outputs,
            )
        except Exception as exc:
            run.node_statuses[nid] = "failed"
            if not failed_so_far:
                run.error = f"arg resolution: {exc}"
                run.error_node = nid
            failed_so_far = True
            persist(run)
            continue

        run.node_statuses[nid] = "running"
        persist(run)
        try:
            output = dispatcher(node.tool_name, resolved_args)
        except ToolNotFound as exc:
            run.node_statuses[nid] = "failed"
            if not failed_so_far:
                run.error = f"tool not found: {exc}"
                run.error_node = nid
            failed_so_far = True
            persist(run)
            continue
        except Exception as exc:
            run.node_statuses[nid] = "failed"
            if not failed_so_far:
                run.error = f"tool raised: {exc}"
                run.error_node = nid
            failed_so_far = True
            persist(run)
            continue

        # Truncate large outputs so the JSON ledger doesn't blow up.
        run.node_outputs[nid] = _truncate_output(output)
        run.node_statuses[nid] = "succeeded"
        persist(run)

    run.status = RunStatus.FAILED if failed_so_far else RunStatus.SUCCEEDED
    run.finished_at = _now_iso()
    persist(run)
    return run


# ─────────────────────────────────────────────────────────────────────
#   Reference resolution
# ─────────────────────────────────────────────────────────────────────


def _resolve_args(
    args: dict[str, Any],
    *,
    inputs: dict[str, Any],
    node_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Walk ``args`` and substitute ``{input}`` + ``${node.field}``
    references. Returns a NEW dict (does not mutate input)."""
    return _resolve_obj(args, inputs=inputs, node_outputs=node_outputs)


def _resolve_obj(
    obj: Any, *, inputs: dict[str, Any], node_outputs: dict[str, Any],
) -> Any:
    if isinstance(obj, str):
        return _resolve_string(obj, inputs=inputs, node_outputs=node_outputs)
    if isinstance(obj, dict):
        return {
            k: _resolve_obj(v, inputs=inputs, node_outputs=node_outputs)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            _resolve_obj(v, inputs=inputs, node_outputs=node_outputs)
            for v in obj
        ]
    return obj


def _resolve_string(
    s: str,
    *,
    inputs: dict[str, Any],
    node_outputs: dict[str, Any],
) -> str:
    # Pass 1: ${node_id.field} references
    def _sub_node(m):
        nid = m.group(1)
        field_path = m.group(2)
        output = node_outputs.get(nid)
        if output is None:
            return ""
        if field_path is None:
            return _stringify(output)
        return _stringify(_dig(output, field_path))

    s = _NODE_REF_RE.sub(_sub_node, s)

    # Pass 2: {input_name} placeholders
    def _sub_input(m):
        name = m.group(1)
        return _stringify(inputs.get(name, ""))

    s = _INPUT_REF_RE.sub(_sub_input, s)
    return s


def _dig(obj: Any, path: str) -> Any:
    """Access ``obj.a.b.c`` via dict-keys or attributes. Unknown
    fields return empty string."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return ""
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur if cur is not None else ""


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    try:
        import json
        return json.dumps(v, default=str)
    except Exception:
        return str(v)


# ─────────────────────────────────────────────────────────────────────
#   Output truncation
# ─────────────────────────────────────────────────────────────────────


_MAX_OUTPUT_CHARS = 8000


def _truncate_output(output: Any) -> Any:
    """Keep node outputs reasonable on disk."""
    if isinstance(output, str):
        return output if len(output) <= _MAX_OUTPUT_CHARS else (
            output[:_MAX_OUTPUT_CHARS] + "…[truncated]"
        )
    if isinstance(output, (dict, list)):
        try:
            import json
            blob = json.dumps(output, default=str)
            if len(blob) <= _MAX_OUTPUT_CHARS:
                return output
            # Mostly-string truncation: round-trip via a string
            # representation for now.
            return {"truncated": True, "preview": blob[:_MAX_OUTPUT_CHARS]}
        except Exception:
            return str(output)[:_MAX_OUTPUT_CHARS]
    return output


# ─────────────────────────────────────────────────────────────────────
#   Tool dispatch
# ─────────────────────────────────────────────────────────────────────


def _default_dispatcher(tool_name: str, args: dict[str, Any]) -> Any:
    """Production tool resolver. Tries the registry first, then
    falls back to importing app.tools.<tool_name>.
    """
    # Strategy 1 — tool registry
    try:
        from app.tool_registry.registry import get_registry
        reg = get_registry()
        tool = None
        try:
            tool = reg.get(tool_name)
        except Exception:
            tool = None
        if tool is None:
            try:
                for t in reg.list_all():
                    if getattr(t, "name", None) == tool_name:
                        tool = t
                        break
            except Exception:
                pass
        if tool is not None:
            run_fn = getattr(tool, "_run", None) or getattr(tool, "run", None) or tool
            if callable(run_fn):
                return run_fn(**args)
    except Exception:
        logger.debug(
            "executor: tool_registry dispatch failed for %s",
            tool_name, exc_info=True,
        )

    # Strategy 2 — direct callable lookup
    try:
        import importlib
        mod = importlib.import_module(f"app.tools.{tool_name}")
        fn = getattr(mod, tool_name, None) or getattr(mod, "run", None)
        if callable(fn):
            return fn(**args)
    except Exception:
        pass

    raise ToolNotFound(
        f"no resolver for tool {tool_name!r} (tried registry + "
        f"app.tools.{tool_name}.{tool_name})"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
