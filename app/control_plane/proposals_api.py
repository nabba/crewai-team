"""Control plane — unified proposals listing at /api/cp/proposals.

Aggregates the three producer outputs that all converge on operator
review through architecture-requests / change-requests:

  - capability_gap_analyzer  →  docs/proposed_capabilities/<sig>.md
  - library_radar            →  docs/proposed_libraries/<sig>-<slug>.md
  - recipe_consolidation     →  workspace/training/
                                recipe_retirement_proposals.jsonl

Endpoints:

  GET /api/cp/proposals                 list summaries from all three
  GET /api/cp/proposals/{kind}          filter to one kind
  GET /api/cp/proposals/{kind}/{name}   full markdown body or row
                                          (capability + library are
                                           markdown files; recipe is
                                           a JSONL row by id)

Read-only. Operators promote proposals manually:
  - capability + library: edit + POST to /api/cp/architecture-requests
                           (or in library's case, file a change-request
                            against requirements.txt)
  - recipe: mark superseded via meta_agent.store ops
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/proposals",
    tags=["control-plane", "proposals"],
    dependencies=[Depends(require_gateway_auth)],
)


_CAPABILITY_DIR = Path("/app/docs/proposed_capabilities")
_LIBRARY_DIR = Path("/app/docs/proposed_libraries")
_RECIPE_PATH = Path("/app/workspace/training/recipe_retirement_proposals.jsonl")


KIND_CAPABILITY = "capability"
KIND_LIBRARY = "library"
KIND_RECIPE = "recipe"
_VALID_KINDS = {KIND_CAPABILITY, KIND_LIBRARY, KIND_RECIPE}


def _summarize_md(path: Path) -> dict[str, Any]:
    """Pull title (first H1) + size + mtime from a markdown file."""
    title = path.stem
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    try:
        stat = path.stat()
        size = stat.st_size
        from datetime import datetime, timezone
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        size = 0
        mtime = ""
    return {
        "name": path.name,
        "title": title,
        "size_bytes": size,
        "modified_at": mtime,
    }


def _list_md_dir(directory: Path, kind: str) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.md"), reverse=True):
        out.append({"kind": kind, **_summarize_md(p)})
    return out


def _list_recipe_proposals() -> list[dict[str, Any]]:
    if not _RECIPE_PATH.exists():
        return []
    rows: dict[str, dict[str, Any]] = {}  # recipe_id → row (latest only)
    try:
        for line in _RECIPE_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("recipe_id", "")
            if not rid:
                continue
            prior = rows.get(rid)
            if prior is None or row.get("proposed_at", "") > prior.get("proposed_at", ""):
                rows[rid] = row
    except OSError:
        return []
    out = []
    for rid, row in rows.items():
        out.append({
            "kind": KIND_RECIPE,
            "name": rid,
            "title": f"retire recipe {rid}",
            "size_bytes": 0,
            "modified_at": row.get("proposed_at", ""),
            "row": row,
        })
    out.sort(key=lambda r: r.get("modified_at", ""), reverse=True)
    return out


@router.get("")
def list_proposals(
    kind: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """List proposals across all three producers (or filter by kind)."""
    if kind is not None and kind not in _VALID_KINDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid kind {kind!r}. Valid: {sorted(_VALID_KINDS)}"
            ),
        )

    items: list[dict[str, Any]] = []
    if kind in (None, KIND_CAPABILITY):
        items.extend(_list_md_dir(_CAPABILITY_DIR, KIND_CAPABILITY))
    if kind in (None, KIND_LIBRARY):
        items.extend(_list_md_dir(_LIBRARY_DIR, KIND_LIBRARY))
    if kind in (None, KIND_RECIPE):
        items.extend(_list_recipe_proposals())

    # Sort all kinds together by modified_at desc.
    items.sort(key=lambda r: r.get("modified_at", ""), reverse=True)
    return {
        "count": len(items[:limit]),
        "proposals": items[:limit],
        "kinds": sorted(_VALID_KINDS),
    }


@router.get("/{kind}/{name}")
def get_proposal(kind: str, name: str):
    """Read one proposal's full body / row."""
    if kind not in _VALID_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"invalid kind {kind!r}. Valid: {sorted(_VALID_KINDS)}",
        )
    # Path-traversal guard.
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid name")

    if kind == KIND_RECIPE:
        for entry in _list_recipe_proposals():
            if entry["name"] == name:
                return entry
        raise HTTPException(status_code=404, detail="recipe proposal not found")

    directory = _CAPABILITY_DIR if kind == KIND_CAPABILITY else _LIBRARY_DIR
    target = (directory / name).resolve()
    canonical = directory.resolve()
    if not (canonical == target or canonical in target.parents):
        raise HTTPException(status_code=400, detail="path escapes directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"{kind} proposal not found")
    try:
        body = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"read failed: {exc}")
    return {"kind": kind, **_summarize_md(target), "body": body}
