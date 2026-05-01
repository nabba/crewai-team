"""Persistence for Companion ideas — JSONL sidecar + ChromaDB embedding store.

Each ``IdeaRecord`` is appended to ``workspace/companion/ideas/<workspace_id>.jsonl``
and indexed in the ChromaDB ``companion_ideas`` collection (one row per
idea, ``workspace_id`` in metadata so novelty search is workspace-scoped).

Neo4j lineage edges are deferred to Phase 13 (cross-workspace transfer)
where they pay off; until then the JSONL's ``lineage_parents`` list gives
us a perfectly serviceable in-workspace lineage graph.

ChromaDB writes use ``chromadb_manager._get_col`` + ``embed`` directly —
the manager's public wrappers don't expose ID-controlled writes or
filtered+metadata queries in one call. Wrapped in defensive seams so any
ChromaDB outage logs and continues; the JSONL remains the source of truth.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_IDEAS_DIR = Path(os.environ.get(
    "COMPANION_IDEAS_DIR", "workspace/companion/ideas"))
_LOCK = Lock()
_CHROMA_COLLECTION = "companion_ideas"


class IdeaState(str, Enum):
    """Lifecycle state of a Companion idea.

    Phase 3 covers FRAGMENT / DEVELOPED / CONVERGED. Phase 4 adds SURFACED.
    Phase 8 adds DOCUMENTED. ARCHIVED is for thumbs-down or stale ideas.
    Transitions are infrastructure-bounded — Self-Improver cannot widen.
    """
    FRAGMENT = "fragment"
    DEVELOPED = "developed"
    CONVERGED = "converged"
    SURFACED = "surfaced"
    DOCUMENTED = "documented"
    ARCHIVED = "archived"


@dataclass
class IdeaRecord:
    """One idea, scored and persisted."""
    idea_id: str = field(
        default_factory=lambda: f"idea_{uuid.uuid4().hex[:12]}")
    workspace_id: str = ""
    cycle_id: str = ""
    text: str = ""
    role: str = ""
    state: IdeaState = IdeaState.FRAGMENT
    lineage_parents: list[str] = field(default_factory=list)
    novelty: float = 0.0
    quality: float = 0.0
    transferability: float = 0.0
    created_at: float = field(default_factory=time.time)


def persist(record: IdeaRecord) -> str:
    """Write the idea to the JSONL sidecar + best-effort ChromaDB index.

    Returns the idea_id. JSONL append is the durable store; ChromaDB
    indexing failure is logged and absorbed.
    """
    _append_jsonl(record)
    _index_chromadb(record)
    return record.idea_id


def find_by_workspace(workspace_id: str, *,
                      state: IdeaState | None = None,
                      limit: int = 100) -> list[IdeaRecord]:
    """Read recent ideas for a workspace from JSONL.

    ``state=None`` means all states. ``limit`` is applied AFTER state
    filtering — caller gets the most recent ``limit`` matching records.
    """
    p = _path_for(workspace_id)
    if not p.exists():
        return []
    out: list[IdeaRecord] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            kwargs = {k: raw[k] for k in IdeaRecord.__dataclass_fields__
                      if k in raw}
            if "state" in kwargs:
                kwargs["state"] = IdeaState(kwargs["state"])
            r = IdeaRecord(**kwargs)
            if state is not None and r.state != state:
                continue
            out.append(r)
        except Exception:
            continue
    return out[-limit:]


def find_by_id(workspace_id: str, idea_id: str) -> IdeaRecord | None:
    """Read one idea record by id. Returns None if not found."""
    for r in find_by_workspace(workspace_id, limit=10_000):
        if r.idea_id == idea_id:
            return r
    return None


def current_state(workspace_id: str, idea_id: str) -> IdeaState | None:
    """Effective state — original record + applied event-log overrides.

    State machine (Phase 4):
        CONVERGED ──surfaced────▶ SURFACED
        SURFACED  ──fb DOWN──▶ ARCHIVED
        SURFACED  ──fb UP─────▶ APPROVED
        any       ──archived─▶ ARCHIVED
        any       ──approved─▶ APPROVED

    Returns None if the idea_id is not in this workspace.
    """
    rec = find_by_id(workspace_id, idea_id)
    if rec is None:
        return None
    state = rec.state
    try:
        from app.companion import events as _events
    except ImportError:
        return state
    for ev in _events.read_for_idea(workspace_id, idea_id):
        if ev.type == _events.EventType.SURFACED:
            state = IdeaState.SURFACED
        elif ev.type == _events.EventType.ARCHIVED:
            state = IdeaState.ARCHIVED
        elif ev.type == _events.EventType.APPROVED:
            state = IdeaState.DOCUMENTED  # treat APPROVED → DOCUMENTED for now
        elif ev.type == _events.EventType.FEEDBACK:
            pol = (ev.payload or {}).get("polarity")
            if pol == "down":
                state = IdeaState.ARCHIVED
            # thumbs UP keeps SURFACED (Phase 5 may introduce APPROVED state).
    return state


def search_similar(workspace_id: str, text: str, *,
                   top_k: int = 5) -> list[dict]:
    """Return top-k most-similar prior ideas for one workspace.

    Each item has ``{document, metadata, distance}``. ``distance`` is
    ChromaDB cosine distance ∈ [0, 2]; lower = more similar. Returns []
    on failure or empty workspace history (so the caller treats absence
    as "no comparable history").
    """
    if not text or not text.strip():
        return []
    try:
        return _chroma_query_for_workspace(text, workspace_id, top_k)
    except Exception as exc:
        logger.debug("companion.idea_store: similar query failed: %s", exc)
        return []


def _chroma_query_for_workspace(text: str, workspace_id: str,
                                 top_k: int) -> list[dict]:
    """Indirection over the ChromaDB filtered query.

    The local seam keeps tests from depending on chromadb being installed.
    """
    from app.memory.chromadb_manager import _get_col, embed
    col = _get_col(_CHROMA_COLLECTION)
    embedding = embed(text)
    results = col.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={"workspace_id": workspace_id},
        include=["documents", "metadatas", "distances"],
    )
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    return [
        {"document": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]


# ── Internal helpers ────────────────────────────────────────────────────────

def _path_for(workspace_id: str) -> Path:
    safe = "".join(c for c in workspace_id if c.isalnum() or c in "-_") \
        or "default"
    return _IDEAS_DIR / f"{safe}.jsonl"


def _append_jsonl(record: IdeaRecord) -> None:
    p = _path_for(record.workspace_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(asdict(record), default=_json_default,
                            sort_keys=True)
    with _LOCK:
        with open(p, "a") as f:
            f.write(serialised + "\n")


def _index_chromadb(record: IdeaRecord) -> None:
    """Best-effort ChromaDB upsert. Failure is logged, not raised."""
    try:
        from app.memory.chromadb_manager import _get_col, embed
        col = _get_col(_CHROMA_COLLECTION)
        embedding = embed(record.text)
        col.upsert(
            ids=[record.idea_id],
            documents=[record.text],
            embeddings=[embedding],
            metadatas=[{
                "workspace_id": record.workspace_id,
                "cycle_id": record.cycle_id,
                "state": record.state.value if isinstance(
                    record.state, IdeaState) else str(record.state),
                "novelty": float(record.novelty),
                "quality": float(record.quality),
                "transferability": float(record.transferability),
                "created_at": float(record.created_at),
            }],
        )
    except Exception as exc:
        logger.debug("companion.idea_store: chromadb upsert failed: %s", exc)


def _json_default(o):
    if isinstance(o, IdeaState):
        return o.value
    raise TypeError(f"Not JSON-serialisable: {type(o)}")
