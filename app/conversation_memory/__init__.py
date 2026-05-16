"""conversation_memory — cross-conversation continuity (Q17.8).

Incremental scan of audit.log into a compact searchable index.
Keyword + recency retrieval. Agent tool exposes recall to the LLM.
"""
from __future__ import annotations

from app.conversation_memory.retrieval import (
    ConversationReference,
    recall,
    recent_summary,
)
from app.conversation_memory.temporal_index import (
    rebuild_index,
    scan_audit_log,
)

__all__ = [
    "ConversationReference",
    "recall",
    "recent_summary",
    "rebuild_index",
    "scan_audit_log",
]
