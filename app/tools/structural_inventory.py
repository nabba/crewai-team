"""
structural_inventory — pure AST snapshot of the codebase's public surface.

Productization plan WP G Phase 0. Used as a regression net by every
choke-point refactor PR. The snapshot is deterministic (sorted) so any
unintended drift surfaces as a clean diff against the baseline JSON.

Scope (sufficient for refactor safety; trivially extensible):

  * FastAPI routes (every ``@app.X`` / ``@router.X`` decorator across
    ``app/``), captured as (method, path, handler, module).
  * Per choke-point module: top-level public callables + classes.
  * Idle-job registry (every ``jobs.append((name, fn, weight))`` in
    ``app/idle_scheduler.py``), captured as (name, weight).

Design:

  * **No runtime imports.** Pure ``ast.parse``. Booting the gateway is
    expensive and side-effectful (daemons start). Static analysis is
    enough and deterministic.
  * **No external deps.** Stdlib only.
  * **Idempotent.** ``collect_inventory()`` is a pure function.
  * **Stable output.** Lists sorted by composite key; dicts emit as
    sorted JSON.

Update workflow when a refactor intentionally moves things:

    python -m app.tools.structural_inventory > tests/baselines/structural_inventory.json

This rewrites the baseline; the diff in the PR is the change record.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# Repo root: this file lives at app/tools/structural_inventory.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Modules tracked individually for callable + class inventory. The
# route scanner (below) sweeps the entire app/ tree separately, so
# routes are NOT limited to this list.
CHOKE_POINTS: tuple[str, ...] = (
    "app/main.py",
    "app/agents/commander/orchestrator.py",
    "app/idle_scheduler.py",
    "app/control_plane/dashboard_api.py",
    "app/crews/base_crew.py",
)

# FastAPI verb decorators we care about. Same set on @app.X and @router.X.
_HTTP_VERBS = frozenset({
    "get", "post", "put", "delete", "patch", "options", "head",
})

# Decorator-owner names treated as route registrars. The scanner picks
# up @app.get(...), @router.get(...), and @sub_router.get(...) variants.
_ROUTER_OWNERS = frozenset({"app", "router"})


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return None


# ── Route extraction ────────────────────────────────────────────────────────


def _extract_routes_from_tree(tree: ast.Module, module_rel: str) -> list[dict[str, str]]:
    """Walk a module AST for HTTP-verb decorators."""
    routes: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            r = _decorator_to_route(deco, node.name, module_rel)
            if r is not None:
                routes.append(r)
    return routes


def _decorator_to_route(
    deco: ast.expr, handler_name: str, module_rel: str,
) -> dict[str, str] | None:
    """Return a route dict iff the decorator is an HTTP-verb registration."""
    if not isinstance(deco, ast.Call):
        return None
    fn = deco.func
    if not isinstance(fn, ast.Attribute):
        return None
    verb = fn.attr.lower()
    if verb not in _HTTP_VERBS:
        return None
    obj = fn.value
    # Owner is a plain Name (`app`, `router`) — covers the common case.
    # Names like `sub_router` are also accepted as long as the trailing
    # token starts with `router` (covers `audit_router.get(...)`).
    owner: str
    if isinstance(obj, ast.Name):
        owner = obj.id
    else:
        return None
    if owner not in _ROUTER_OWNERS and not owner.endswith("router") and not owner.startswith("router"):
        return None
    # First positional arg is the path string
    path = ""
    if deco.args:
        first = deco.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            path = first.value
    return {
        "method": verb.upper(),
        "path": path,
        "handler": handler_name,
        "module": module_rel,
    }


def _scan_routes_in_app() -> list[dict[str, str]]:
    """Walk every .py under app/ and collect HTTP routes."""
    routes: list[dict[str, str]] = []
    app_dir = REPO_ROOT / "app"
    for p in sorted(app_dir.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(REPO_ROOT))
        tree = _parse(p)
        if tree is None:
            continue
        routes.extend(_extract_routes_from_tree(tree, rel))
    # Stable sort. Module is NOT in the sort key so a route is still
    # "the same route" after moving between files — this is exactly the
    # property we need for the refactor regression net.
    routes.sort(key=lambda r: (r["method"], r["path"], r["handler"]))
    return routes


# ── Per-module surface ──────────────────────────────────────────────────────


def _public_top_level(tree: ast.Module, kind: str) -> list[str]:
    """Return sorted top-level public names of the given kind.

    kind="def"    → FunctionDef + AsyncFunctionDef
    kind="class"  → ClassDef
    "Public" = name does not start with an underscore.
    """
    if kind == "def":
        types = (ast.FunctionDef, ast.AsyncFunctionDef)
    else:
        types = (ast.ClassDef,)
    names = sorted(
        n.name for n in tree.body
        if isinstance(n, types) and not n.name.startswith("_")
    )
    return names


# ── Idle-job extraction ─────────────────────────────────────────────────────


def _extract_idle_jobs(idle_path: Path) -> list[dict[str, str]]:
    """Parse `jobs.append((NAME, fn, WEIGHT))` patterns from idle_scheduler.py."""
    tree = _parse(idle_path)
    if tree is None:
        return []
    jobs: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr != "append":
            continue
        if not isinstance(fn.value, ast.Name) or fn.value.id != "jobs":
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.Tuple) or len(arg.elts) < 2:
            continue
        # Job name is the first tuple element if it's a string literal.
        name = ""
        if isinstance(arg.elts[0], ast.Constant) and isinstance(arg.elts[0].value, str):
            name = arg.elts[0].value
        # Weight is the third tuple element (JobWeight.LIGHT etc.)
        weight = ""
        if len(arg.elts) >= 3:
            w = arg.elts[2]
            if isinstance(w, ast.Attribute):
                weight = w.attr.lower()
            elif isinstance(w, ast.Constant) and isinstance(w.value, str):
                weight = w.value
        if name:
            jobs.append({"name": name, "weight": weight})
    jobs.sort(key=lambda j: j["name"])
    return jobs


# ── Public entry ────────────────────────────────────────────────────────────


def collect_inventory() -> dict[str, Any]:
    """Return the deterministic snapshot dict.

    Schema:
      {
        "version": 1,
        "choke_points": {
          "app/main.py": {"exists": bool, "public_defs": [...], "classes": [...]},
          ...
        },
        "routes": [{"method", "path", "handler", "module"}, ...],
        "idle_jobs": [{"name", "weight"}, ...],
      }
    """
    inv: dict[str, Any] = {
        "version": 1,
        "choke_points": {},
        "routes": _scan_routes_in_app(),
        "idle_jobs": _extract_idle_jobs(REPO_ROOT / "app/idle_scheduler.py"),
    }
    for rel in CHOKE_POINTS:
        p = REPO_ROOT / rel
        if not p.exists():
            inv["choke_points"][rel] = {"exists": False}
            continue
        tree = _parse(p)
        if tree is None:
            inv["choke_points"][rel] = {"exists": True, "parse_error": True}
            continue
        inv["choke_points"][rel] = {
            "exists": True,
            "public_defs": _public_top_level(tree, "def"),
            "classes": _public_top_level(tree, "class"),
        }
    return inv


def write_baseline(out_path: Path) -> None:
    inv = collect_inventory()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(inv, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Iterable[str] = ()) -> int:
    argv = list(argv)
    if argv and argv[0] == "--write-baseline":
        out = REPO_ROOT / "tests" / "baselines" / "structural_inventory.json"
        write_baseline(out)
        print(f"wrote {out}", file=sys.stderr)
        return 0
    inv = collect_inventory()
    print(json.dumps(inv, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
