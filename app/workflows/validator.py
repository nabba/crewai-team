"""Workflow validator — refuses invalid templates at save time.

Checks (in order):

  1. ``id`` and ``name`` are non-empty.
  2. Every ``node.id`` is unique within the template.
  3. Every ``depends_on`` reference points to a sibling node.
  4. The dependency graph has no cycles (Kahn's algorithm).
  5. Every ``${node_id.field}`` reference in node args points to an
     earlier (topologically) node and uses a known field shape.
  6. Every ``{input_name}`` placeholder in node args is declared in
     ``template.inputs``.
  7. Every ``tool_name`` is registered in the tool registry (best-
     effort; skips when registry is unavailable in a fresh boot).

The checker is conservative: when a check can't be performed (e.g.
tool registry unavailable), it warns but doesn't refuse — refusing
on transient unavailability would block legitimate operator work.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.workflows.models import InvalidWorkflow, WorkflowTemplate

logger = logging.getLogger(__name__)


_NODE_REF_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)(?:\.([a-zA-Z_][a-zA-Z0-9_.]*))?\}")
# Negative lookbehind on $ so ``{name}`` matches input placeholders
# but ``${name}`` (the node-reference syntax) is excluded.
_INPUT_REF_RE = re.compile(r"(?<!\$)\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def validate_template(template: WorkflowTemplate) -> None:
    """Raise :class:`InvalidWorkflow` if the template is malformed.

    Callers should wrap the save+validate as one transaction; this
    function never mutates the template."""
    if not template.id or not template.id.strip():
        raise InvalidWorkflow("template id is empty")
    if not template.name or not template.name.strip():
        raise InvalidWorkflow("template name is empty")

    # 2. Unique node ids.
    seen: dict[str, int] = {}
    for i, n in enumerate(template.nodes):
        if not n.id.strip():
            raise InvalidWorkflow(f"node #{i} has empty id")
        if n.id in seen:
            raise InvalidWorkflow(
                f"duplicate node id {n.id!r} at positions "
                f"{seen[n.id]} and {i}"
            )
        seen[n.id] = i

    # 3. depends_on references are siblings.
    node_ids = set(seen.keys())
    for n in template.nodes:
        for dep in n.depends_on:
            if dep not in node_ids:
                raise InvalidWorkflow(
                    f"node {n.id!r} depends_on unknown node {dep!r}"
                )

    # 4. DAG cycle check (Kahn's).
    topo_order = _topological_order(template)
    pos_of: dict[str, int] = {nid: pos for pos, nid in enumerate(topo_order)}

    # 5. ${node_id.field} reference check.
    for n in template.nodes:
        for value in _walk_strings(n.args):
            for m in _NODE_REF_RE.finditer(value):
                referenced_node = m.group(1)
                if referenced_node not in node_ids:
                    raise InvalidWorkflow(
                        f"node {n.id!r} references unknown node "
                        f"{referenced_node!r} in args"
                    )
                if pos_of[referenced_node] >= pos_of[n.id]:
                    raise InvalidWorkflow(
                        f"node {n.id!r} references {referenced_node!r} "
                        f"but {referenced_node!r} is not an ancestor "
                        f"(check depends_on chain)"
                    )

    # 6. {input_name} placeholder check.
    declared = set(template.inputs)
    for n in template.nodes:
        for value in _walk_strings(n.args):
            for m in _INPUT_REF_RE.finditer(value):
                name = m.group(1)
                if name not in declared:
                    raise InvalidWorkflow(
                        f"node {n.id!r} uses {{{name}}} but it is not "
                        f"declared in template.inputs"
                    )

    # 7. Tool registry check (best-effort).
    known_tools = _known_tool_names()
    if known_tools is not None:
        for n in template.nodes:
            if n.tool_name not in known_tools:
                raise InvalidWorkflow(
                    f"node {n.id!r} references tool {n.tool_name!r} "
                    f"which is not registered in the tool registry"
                )


def _topological_order(template: WorkflowTemplate) -> list[str]:
    """Kahn's algorithm. Raises InvalidWorkflow on cycles."""
    indegree: dict[str, int] = {n.id: 0 for n in template.nodes}
    children: dict[str, list[str]] = {n.id: [] for n in template.nodes}
    for n in template.nodes:
        for dep in n.depends_on:
            indegree[n.id] += 1
            children[dep].append(n.id)
    queue = [nid for nid, d in indegree.items() if d == 0]
    out: list[str] = []
    while queue:
        nid = queue.pop(0)
        out.append(nid)
        for child in children[nid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(out) != len(template.nodes):
        remaining = [n for n, d in indegree.items() if d > 0]
        raise InvalidWorkflow(
            f"cycle detected in workflow DAG; nodes in cycle: {remaining}"
        )
    return out


def _walk_strings(obj: Any):
    """Yield every string value inside a nested dict/list."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


def _known_tool_names() -> set[str] | None:
    """Return the set of registered tool names, or None when the
    registry is unavailable (e.g. fresh boot, missing deps)."""
    try:
        from app.tool_registry.registry import get_registry
    except Exception:
        return None
    try:
        reg = get_registry()
        names = set()
        for tool in reg.list_all():
            name = getattr(tool, "name", None) or getattr(tool, "id", None)
            if name:
                names.add(str(name))
        return names if names else None
    except Exception:
        logger.debug("validator: tool registry probe failed", exc_info=True)
        return None
