"""WorkspaceKB — unified retrieval across the workspace's knowledge sources.

Phase 2 wires:
  - temporal_context (always included)
  - KB v2 retrieval orchestrator (episteme + experiential collections)

Phase 6 will add: workspace-scoped sources (RSS / web_search / URL pollers)
ingested into the ``companion_sources`` ChromaDB collection.
Phase 9 will add: accepted ideas, polished documents, wiki pages.

The composer always queries WorkspaceKB even if it returns nothing — the
caller can detect "low workspace knowledge" and broaden context elsewhere.
Failures of individual sources are logged and absorbed; the cycle proceeds
with whatever is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_KB_TOP_K = 6
DEFAULT_KB_COLLECTIONS = ("episteme", "experiential")


@dataclass
class KBSnippet:
    """A single retrieved snippet with its provenance."""
    text: str
    score: float
    source: str

    def to_prompt_line(self) -> str:
        body = self.text.strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:397] + "..."
        return f"[{self.source} · {self.score:.2f}] {body}"


def compose(workspace_id: str, query: str, *,
            top_k: int = DEFAULT_KB_TOP_K) -> list[KBSnippet]:
    """Build a context bundle for one Companion cycle.

    Always returns a list (possibly with only the temporal snippet).
    Failures of any one source are absorbed — the cycle proceeds with
    whatever is available rather than aborting.
    """
    snippets: list[KBSnippet] = []

    temporal = _temporal_snippet()
    if temporal.text:
        snippets.append(temporal)

    try:
        snippets.extend(_kb_v2_snippets(query, top_k=top_k))
    except Exception as exc:
        logger.debug("companion.workspace_kb: kb_v2 helper raised: %s", exc)

    return snippets


def _temporal_snippet() -> KBSnippet:
    """Helsinki-localised seasonal / lunar / daylight context."""
    try:
        from app.temporal_context import format_temporal_block
        block = format_temporal_block()
    except Exception as exc:
        logger.debug("companion.workspace_kb: temporal_context failed: %s", exc)
        return KBSnippet(text="", score=0.0, source="temporal_context")
    return KBSnippet(text=block or "", score=1.0, source="temporal_context")


def _kb_v2_snippets(query: str, *, top_k: int) -> list[KBSnippet]:
    """Pull from KB v2 collections via the retrieval orchestrator."""
    if not query or not query.strip():
        return []
    try:
        from app.retrieval.orchestrator import RetrievalOrchestrator
        results = RetrievalOrchestrator().retrieve(
            query=query,
            collections=list(DEFAULT_KB_COLLECTIONS),
            top_k=top_k,
        )
    except Exception as exc:
        logger.debug("companion.workspace_kb: retrieval orchestrator failed: %s",
                     exc)
        return []
    out: list[KBSnippet] = []
    for r in results:
        meta = getattr(r, "metadata", None) or {}
        prov = getattr(r, "provenance", None) or {}
        source = (
            prov.get("collection")
            if isinstance(prov, dict) else None
        ) or (
            meta.get("collection") if isinstance(meta, dict) else None
        ) or "kb_v2"
        out.append(KBSnippet(
            text=getattr(r, "text", "") or "",
            score=float(getattr(r, "score", 0.0)),
            source=str(source),
        ))
    return out
