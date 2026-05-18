"""Persist and query ``InventorySnapshot`` against the on-disk catalogue.

Snapshot lives at ``workspace/system_inventory/snapshot.json``. A
``build_snapshot`` walk costs ~0.5 s for the current ~1200-module
codebase, but we cache so prompt-time queries are O(disk-read).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from app.system_inventory.scanner import (
    InventorySnapshot,
    ModuleEntry,
    build_snapshot,
)

logger = logging.getLogger(__name__)


_lock = threading.Lock()


def _workspace_root() -> Path:
    # Env var takes precedence so tests + operator overrides see the
    # current value without forcing a module reload of ``app.paths``.
    env = os.environ.get("WORKSPACE_ROOT")
    if env:
        return Path(env)
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _snapshot_path() -> Path:
    return _workspace_root() / "system_inventory" / "snapshot.json"


# ── persistence ─────────────────────────────────────────────────────────


def persist_snapshot(snapshot: InventorySnapshot) -> None:
    """Atomic JSON write so concurrent readers never see a half-file."""
    path = _snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True)
    tmp = path.with_suffix(".json.tmp")
    with _lock:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)


def _load() -> InventorySnapshot | None:
    path = _snapshot_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("system_inventory: snapshot.json unreadable", exc_info=True)
        return None
    try:
        modules = tuple(
            ModuleEntry(
                path=m["path"],
                kind=m["kind"],
                summary=m.get("summary", ""),
                public_symbols=tuple(m.get("public_symbols") or ()),
                capabilities=tuple(m.get("capabilities") or ()),
                loc=int(m.get("loc", 0)),
                has_tests=bool(m.get("has_tests", False)),
            )
            for m in raw.get("modules", [])
        )
    except (KeyError, TypeError, ValueError):
        logger.debug("system_inventory: snapshot.json malformed", exc_info=True)
        return None
    return InventorySnapshot(
        generated_at=raw.get("generated_at", ""),
        modules=modules,
        app_root=raw.get("app_root", ""),
    )


def get_snapshot(*, rebuild_if_missing: bool = True) -> InventorySnapshot | None:
    """Return the last persisted snapshot, optionally building a fresh
    one when nothing is on disk yet."""
    snap = _load()
    if snap is not None:
        return snap
    if not rebuild_if_missing:
        return None
    snap = build_snapshot()
    persist_snapshot(snap)
    return snap


# ── query helpers ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Query:
    kind: str | None
    capability: str | None
    keyword: str | None


def _matches(entry: ModuleEntry, q: _Query) -> bool:
    if q.kind and entry.kind != q.kind:
        return False
    if q.capability and q.capability not in entry.capabilities:
        return False
    if q.keyword:
        needle = q.keyword.lower()
        if needle not in entry.path.lower() and needle not in entry.summary.lower():
            return False
    return True


def query_inventory(
    *,
    kind: str | None = None,
    capability: str | None = None,
    keyword: str | None = None,
    limit: int = 50,
) -> list[ModuleEntry]:
    """Filter the persisted inventory. All filters AND together.

    ``kind`` must be ``package`` or ``module`` when set. ``capability``
    matches an exact tag. ``keyword`` is a case-insensitive substring
    against the module path and its docstring summary.
    """
    snap = get_snapshot()
    if snap is None:
        return []
    q = _Query(kind=kind, capability=capability, keyword=keyword)
    out = [e for e in snap.modules if _matches(e, q)]
    return out[:limit]


def inventory_summary() -> str:
    """Compact one-paragraph summary suitable for prompts and Signal."""
    snap = get_snapshot()
    if snap is None:
        return "system_inventory: snapshot unavailable"
    capabilities = sorted({c for m in snap.modules for c in m.capabilities})
    tested_pct = (
        100.0 * sum(1 for m in snap.modules if m.has_tests) / max(1, snap.n_modules)
    )
    return (
        f"system_inventory@{snap.generated_at[:10]}: "
        f"{snap.n_modules} modules ({snap.n_packages} packages) · "
        f"{snap.total_loc:,} LOC · {tested_pct:.0f}% with tests · "
        f"{len(capabilities)} registered-tool capabilities"
    )
