"""retrieval — keyword + recency search over the temporal index (Q17.8)."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_DEFAULT_TOP_K = 10


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _index_path() -> Path:
    return _workspace_root() / "conversation_memory" / "index.jsonl"


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_conversation_memory_enabled
        return get_conversation_memory_enabled()
    except Exception:
        return True


@dataclass
class ConversationReference:
    ts: str
    kind: str
    preview: str
    ref: str | None
    score: float
    tokens_matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "kind": self.kind, "preview": self.preview, "ref": self.ref,
                "score": round(float(self.score), 4), "tokens_matched": self.tokens_matched}


def _query_tokens(query: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(query or "")}


def _score_row(query_set: set[str], tokens: list[str]) -> tuple[float, list[str]]:
    row_set = set(tokens)
    matched = list(query_set & row_set)
    if not matched:
        return 0.0, []
    overlap = len(matched) / max(1, len(query_set))
    density = len(matched) / max(1, len(tokens))
    return (overlap * 0.7) + (density * 0.3), matched


def recall(query: str, *, window_months: int = 24, top_k: int = _DEFAULT_TOP_K, kinds: set[str] | None = None) -> list[ConversationReference]:
    if not _enabled():
        return []
    p = _index_path()
    if not p.exists():
        return []
    qset = _query_tokens(query)
    if not qset:
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(window_months * 30.5))).isoformat()
    scored: list[ConversationReference] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = row.get("ts", "")
                if ts < cutoff:
                    continue
                if kinds and row.get("kind") not in kinds:
                    continue
                tokens = row.get("tokens") or []
                if not isinstance(tokens, list):
                    continue
                score, matched = _score_row(qset, tokens)
                if score <= 0:
                    continue
                scored.append(ConversationReference(
                    ts=ts, kind=row.get("kind", "unknown"),
                    preview=row.get("preview", "")[:240],
                    ref=row.get("ref"), score=score, tokens_matched=matched,
                ))
    except OSError:
        return []
    scored.sort(key=lambda r: (r.score, r.ts), reverse=True)
    return scored[:top_k]


def recent_summary(*, days: int = 90) -> dict[str, Any]:
    if not _enabled():
        return {"enabled": False}
    p = _index_path()
    if not p.exists():
        return {"n_total": 0, "by_kind": {}, "first_ts": None, "last_ts": None}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    by_kind: dict[str, int] = {}
    first, last = None, None
    n_total = 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = row.get("ts", "")
                if ts < cutoff:
                    continue
                n_total += 1
                kind = row.get("kind", "unknown")
                by_kind[kind] = by_kind.get(kind, 0) + 1
                if first is None or ts < first:
                    first = ts
                if last is None or ts > last:
                    last = ts
    except OSError:
        return {"error": "read_failed"}
    return {"n_total": n_total, "by_kind": by_kind, "first_ts": first, "last_ts": last}
