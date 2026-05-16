"""recall_past_conversation — agent tool for Q17.8 retrieval."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _format_references(query: str, refs: list, window_months: int) -> str:
    if not refs:
        return f"No prior conversation found matching {query!r} in the last {window_months} months."
    lines = [f"Recall for {query!r} (window: {window_months} months, top {len(refs)}):", ""]
    for r in refs:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        ts = d.get("ts", "")[:19]
        kind = d.get("kind", "?")
        preview = d.get("preview", "")[:200]
        score = d.get("score", 0)
        ref_id = d.get("ref") or "—"
        lines.append(f"  [{ts}] kind={kind} score={score:.2f} ref={ref_id}\n      {preview}")
    return "\n".join(lines)


def recall_past_conversation(query: str, window_months: int = 24, top_k: int = 5) -> str:
    try:
        from app.conversation_memory.retrieval import recall
    except Exception as exc:
        return f"Conversation memory unavailable: {type(exc).__name__}"
    try:
        refs = recall(query, window_months=window_months, top_k=top_k)
    except Exception as exc:
        return f"Recall failed: {type(exc).__name__}: {exc}"
    return _format_references(query, refs, window_months)


try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _RecallSchema(BaseModel):
        query: str = Field(description="Topic or keywords to search for.")
        window_months: int = Field(default=24, description="How far back to search, in months.")
        top_k: int = Field(default=5, description="Maximum number of references to return.")

    class RecallPastConversationTool(BaseTool):
        name: str = "recall_past_conversation"
        description: str = (
            "Search the operator's past conversations for prior context on a topic. "
            "Returns up to top_k matching references with timestamps + previews. Use this "
            "before assuming a topic is novel — there may be prior context worth referencing."
        )
        args_schema: type = _RecallSchema

        def _run(self, query: str, window_months: int = 24, top_k: int = 5) -> str:
            return recall_past_conversation(query, window_months=window_months, top_k=top_k)

    __all__ = ["RecallPastConversationTool", "recall_past_conversation"]
except ImportError:
    __all__ = ["recall_past_conversation"]
