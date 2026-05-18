"""Walk ``app/``, extract one ``ModuleEntry`` per Python module.

The scan is intentionally cheap — AST-only, no imports — so it can run
weekly without risking subsystem side-effects (some modules start daemon
threads at import time; ``import app.healing`` is the canonical example).
A scan that ``import``ed every module would defeat the cadence guard.

What we extract per file
------------------------
- ``path``           — repo-relative path
- ``kind``           — ``package`` (an ``__init__.py``) or ``module``
- ``summary``        — first non-blank line of the module docstring
- ``public_symbols`` — top-level ``def``/``async def``/``class`` whose
                       names don't start with ``_``
- ``capabilities``   — tags scraped from ``@register_tool(capability=…)``
                       decorators (parsed structurally, not by import)
- ``loc``            — non-blank, non-comment line count
- ``has_tests``      — ``True`` iff a sibling test file exists under
                       ``tests/`` matching the module's stem
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# Subdirectories under ``app/`` we skip entirely. Pure data / generated.
_SKIP_PARTS: frozenset[str] = frozenset({
    "__pycache__", ".pytest_cache", "souls",
})


@dataclass(frozen=True)
class ModuleEntry:
    """One module's catalogued shape."""
    path: str
    kind: str                      # "package" | "module"
    summary: str                   # first docstring line (may be empty)
    public_symbols: tuple[str, ...]
    capabilities: tuple[str, ...]
    loc: int
    has_tests: bool

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "kind": self.kind,
            "summary": self.summary,
            "public_symbols": list(self.public_symbols),
            "capabilities": list(self.capabilities),
            "loc": self.loc,
            "has_tests": self.has_tests,
        }


@dataclass(frozen=True)
class InventorySnapshot:
    """A timestamped inventory of every catalogued module."""
    generated_at: str              # ISO-8601 UTC
    modules: tuple[ModuleEntry, ...]
    app_root: str

    @property
    def n_modules(self) -> int:
        return len(self.modules)

    @property
    def n_packages(self) -> int:
        return sum(1 for m in self.modules if m.kind == "package")

    @property
    def total_loc(self) -> int:
        return sum(m.loc for m in self.modules)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "app_root": self.app_root,
            "n_modules": self.n_modules,
            "n_packages": self.n_packages,
            "total_loc": self.total_loc,
            "modules": [m.to_dict() for m in self.modules],
        }


# ── extraction primitives ───────────────────────────────────────────────


def _first_docstring_line(tree: ast.Module) -> str:
    doc = ast.get_docstring(tree, clean=True) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _public_symbols(tree: ast.Module) -> tuple[str, ...]:
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                out.append(node.name)
    return tuple(out)


def _capabilities_from_decorators(tree: ast.Module) -> tuple[str, ...]:
    """Scrape ``@register_tool(capabilities=[...])`` calls without importing.

    Matches both ``register_tool(...)`` and ``tool_registry.register_tool(...)``.
    The registry's signature uses the plural ``capabilities`` (a list);
    we accept the singular ``capability`` too for forward-compat.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name != "register_tool":
            continue
        for kw in node.keywords:
            if kw.arg not in ("capabilities", "capability"):
                continue
            # List literal: capabilities=["reads-attachment", ...]
            if isinstance(kw.value, ast.List):
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        found.add(elt.value)
            # Bare string: capability="reads-attachment"
            elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                found.add(kw.value.value)
    return tuple(sorted(found))


def _non_blank_loc(source: str) -> int:
    n = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            n += 1
    return n


def _has_tests(module_path: Path, tests_root: Path) -> bool:
    """Best-effort: a sibling test file exists whose name contains this stem."""
    if not tests_root.exists():
        return False
    stem = module_path.stem
    if stem == "__init__":
        stem = module_path.parent.name
    needle = f"test_{stem}.py"
    for p in tests_root.rglob(needle):
        if p.is_file():
            return True
    return False


def _scan_one(path: Path, app_root: Path, tests_root: Path) -> ModuleEntry | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.debug("system_inventory: skipping %s — SyntaxError", path)
        return None

    rel = str(path.relative_to(app_root.parent))
    return ModuleEntry(
        path=rel,
        kind="package" if path.name == "__init__.py" else "module",
        summary=_first_docstring_line(tree),
        public_symbols=_public_symbols(tree),
        capabilities=_capabilities_from_decorators(tree),
        loc=_non_blank_loc(source),
        has_tests=_has_tests(path, tests_root),
    )


# ── public entry point ──────────────────────────────────────────────────


def build_snapshot(
    app_root: Path | None = None,
    tests_root: Path | None = None,
) -> InventorySnapshot:
    """Walk ``app/`` and produce a fresh ``InventorySnapshot``.

    Repo-relative paths use ``app_root.parent`` as the anchor so entries
    look like ``app/healing/monitors/bit_rot_scan.py``.
    """
    app_root = (app_root or Path("app")).resolve()
    tests_root = (tests_root or app_root.parent / "tests").resolve()

    modules: list[ModuleEntry] = []
    for path in sorted(app_root.rglob("*.py")):
        if any(part in _SKIP_PARTS for part in path.parts):
            continue
        entry = _scan_one(path, app_root, tests_root)
        if entry is not None:
            modules.append(entry)

    return InventorySnapshot(
        generated_at=datetime.now(timezone.utc).isoformat(),
        modules=tuple(modules),
        app_root=str(app_root),
    )
