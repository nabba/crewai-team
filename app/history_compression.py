"""
history_compression.py — Three-tier conversation compression.

Inspired by Agent Zero's Message → Topic → Bulk hierarchy.
Adapted for crewai-team's 4-tier LLM cascade with local model context pressure.

Architecture:
    Message  — Individual conversation turns (user/assistant/tool)
    Topic    — Cluster of related messages (one user request + all responses)
    Bulk     — Mega-summary of old topic clusters

Budget allocation (configurable):
    Current Topic:     40% of context budget (full fidelity)
    Historical Topics: 35% (summarized)
    Bulks:             15% (mega-summaries)
    System prompt:     10% reserved

Compression runs in background thread after each agent turn.
Uses budget-tier LLM (DeepSeek via OpenRouter) for summarization.

Integration:
    - Uses llm_factory.py for LLM calls (not raw litellm)
    - Serializes to JSON for PostgreSQL storage alongside conversation_store
    - Output compatible with CrewAI/LangChain message format
    - Auto-adjusts context budget when LLM cascade tier changes

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class CompressionConfig:
    """Tunable knobs for the compression system."""

    # Context budget ratios (must sum to <= 1.0)
    current_topic_ratio: float = 0.40
    historical_topics_ratio: float = 0.35
    bulk_ratio: float = 0.15
    system_prompt_reserved: float = 0.10

    # Compression behavior
    compression_target_ratio: float = 0.80  # Compress to 80% of limit
    topics_merge_count: int = 3             # Merge oldest N topics into bulk
    large_message_trim_chars: int = 4000    # Trim individual messages beyond this

    # Total context window (tokens) — overridden per model tier
    max_context_tokens: int = 8192  # Conservative default for local Ollama

    # Summarization (uses existing llm_factory)
    summarizer_max_tokens: int = 512

    # Async compression
    compress_in_background: bool = True
    compression_timeout_seconds: float = 30.0

    # Per-model-tier context limits (auto-selected by cascade)
    tier_context_limits: dict = field(default_factory=lambda: {
        "local":   8192,     # Ollama qwen3:30b-a3b
        "budget":  32768,    # DeepSeek V3.2 via OpenRouter
        "mid":     65536,    # MiniMax M2.5
        "premium": 200000,   # Anthropic Claude / Gemini
    })


# ── Token estimation ──────────────────────────────────────────────────────────


def approximate_tokens(text: str) -> int:
    """~4 chars/token for English. Fast, no tokenizer needed."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Data model: Message → Topic → Bulk ────────────────────────────────────────


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class Message:
    """Single conversation turn."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: str = "user"  # Use string for compatibility with existing conversation_store
    content: str = ""
    summary: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""
    metadata: dict = field(default_factory=dict)
    _token_cache: Optional[int] = field(default=None, repr=False)

    @property
    def effective_content(self) -> str:
        return self.summary if self.summary else self.content

    @property
    def tokens(self) -> int:
        if self._token_cache is None:
            self._token_cache = approximate_tokens(self.effective_content)
        return self._token_cache

    def set_summary(self, summary: str) -> None:
        self.summary = summary
        self._token_cache = None

    def trim_content(self, max_chars: int) -> None:
        if len(self.content) <= max_chars:
            return
        half = max_chars // 2
        trimmed = len(self.content) - max_chars
        self.content = (
            self.content[:half]
            + f"\n\n... [{trimmed} chars trimmed] ...\n\n"
            + self.content[-half:]
        )
        self._token_cache = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "role": self.role, "content": self.content,
            "summary": self.summary, "timestamp": self.timestamp,
            "agent_id": self.agent_id, "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            summary=data.get("summary"),
            timestamp=data.get("timestamp", time.time()),
            agent_id=data.get("agent_id", ""),
            metadata=data.get("metadata", {}),
        )

    def to_langchain(self) -> dict:
        """Convert to LangChain-compatible message dict."""
        role_map = {"user": "human", "assistant": "ai", "tool": "tool", "system": "system"}
        return {"role": role_map.get(self.role, "human"), "content": self.effective_content}


@dataclass
class Topic:
    """Cluster of related messages — one user request + all responses."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: list[Message] = field(default_factory=list)
    summary: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def tokens(self) -> int:
        if self.summary:
            return approximate_tokens(self.summary)
        return sum(m.tokens for m in self.messages)

    @property
    def is_summarized(self) -> bool:
        return self.summary is not None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Topic":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            summary=data.get("summary"),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class Bulk:
    """Mega-summary of multiple old topics. Most compressed tier."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    summary: str = ""
    topic_count: int = 0
    span_start: float = 0.0
    span_end: float = 0.0

    @property
    def tokens(self) -> int:
        return approximate_tokens(self.summary)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "summary": self.summary,
            "topic_count": self.topic_count,
            "span_start": self.span_start, "span_end": self.span_end,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Bulk":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            summary=data.get("summary", ""),
            topic_count=data.get("topic_count", 0),
            span_start=data.get("span_start", 0.0),
            span_end=data.get("span_end", 0.0),
        )


# ── Summarizer — uses existing llm_factory ───────────────────────────────────

# IMMUTABLE prompts
SUMMARIZE_MESSAGES_PROMPT = """\
You are a conversation summarizer for an AI agent system.
Produce a concise STRUCTURED summary preserving:
- DECISIONS: Key choices and conclusions reached
- DETAILS: Specific technical details (file paths, function names, config values, numbers)
- ACTIONS: Action items, next steps, pending work
- ISSUES: Errors encountered and their resolutions

Be extremely concise. No filler. No pleasantries. Pure information density."""

MERGE_TOPICS_PROMPT = """\
Merge these conversation topic summaries into one ultra-concise overview.
Preserve: key decisions, technical specifics (paths, names, values), unresolved items.
Use DECISIONS / DETAILS / ACTIONS / ISSUES sections. Maximum 200 words."""


class Summarizer:
    """Calls budget-tier LLM for summarization via existing llm_factory."""

    def __init__(self, max_tokens: int = 512):
        self._max_tokens = max_tokens

    def summarize_messages(self, messages: list[Message]) -> str:
        content_block = "\n\n".join(
            f"[{m.role}]: {m.effective_content}" for m in messages
        )
        return self._call_llm(
            SUMMARIZE_MESSAGES_PROMPT,
            f"Summarize this conversation segment:\n\n{content_block}",
            fallback=self._naive_fallback(messages),
        )

    def summarize_topics(self, topics: list[Topic]) -> str:
        topic_block = "\n\n".join(
            f"--- Topic {i+1} ---\n{t.summary or self._topic_text(t)}"
            for i, t in enumerate(topics)
        )
        return self._call_llm(
            MERGE_TOPICS_PROMPT,
            f"Merge:\n\n{topic_block}",
            fallback=" | ".join((t.summary or "")[:150] for t in topics)[:500],
        )

    def _call_llm(self, system: str, user: str, fallback: str) -> str:
        """Use existing llm_factory budget-tier model for summarization."""
        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=self._max_tokens, role="self_improve")
            prompt = f"{system}\n\n{user}"
            result = str(llm.call(prompt)).strip()
            if result and len(result) > 10:
                return result
            return f"[SUMMARY FALLBACK] {fallback}"
        except Exception as e:
            logger.error(f"Summarization LLM call failed: {e}")
            return f"[SUMMARY FALLBACK] {fallback}"

    @staticmethod
    def _topic_text(topic: Topic) -> str:
        return "\n".join(
            f"[{m.role}]: {m.effective_content[:300]}" for m in topic.messages
        )

    @staticmethod
    def _naive_fallback(messages: list[Message]) -> str:
        return " | ".join(m.effective_content[:200] for m in messages)[:500]


# ── History: main container with compression engine ──────────────────────────


class History:
    """Three-tier conversation history with automatic compression.

    Usage:
        history = History(CompressionConfig())

        # On each user message
        history.start_new_topic()
        history.add_message(Message(role="user", content="..."))

        # On each agent response
        history.add_message(Message(role="assistant", content="...", agent_id="commander"))

        # Get compressed context for LLM
        messages = history.to_langchain_messages()

        # Compress in background after turn completes
        history.compress_async()

        # Persist
        json_str = history.serialize()

        # Restore
        history = History.deserialize(json_str, config)
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.summarizer = Summarizer(max_tokens=self.config.summarizer_max_tokens)
        self.bulks: list[Bulk] = []
        self.topics: list[Topic] = []
        self.current: Topic = Topic()
        self._compress_lock = threading.Lock()
        self._compress_thread: Optional[threading.Thread] = None

    # ── Message operations ────────────────────────────────────────────

    def add_message(self, message: Message) -> None:
        """Add a message to the current topic."""
        self.current.messages.append(message)

    def start_new_topic(self) -> None:
        """Start a new conversation topic. Moves current to historical."""
        if self.current.messages:
            self.topics.append(self.current)
            self.current = Topic()

    @property
    def total_tokens(self) -> int:
        return (
            sum(b.tokens for b in self.bulks)
            + sum(t.tokens for t in self.topics)
            + self.current.tokens
        )

    @property
    def needs_compression(self) -> bool:
        target = int(self.config.max_context_tokens * self.config.compression_target_ratio)
        return self.total_tokens > target

    # ── Output for LLM ────────────────────────────────────────────────

    def to_langchain_messages(self) -> list[dict]:
        """Build LangChain-compatible message list respecting context budget."""
        output = []

        # Bulks as system messages (most compressed)
        for bulk in self.bulks:
            output.append({
                "role": "system",
                "content": f"[Earlier context summary]\n{bulk.summary}",
            })

        # Historical topics (summarized or full)
        for topic in self.topics:
            if topic.is_summarized:
                output.append({
                    "role": "system",
                    "content": f"[Previous exchange summary]\n{topic.summary}",
                })
            else:
                for msg in topic.messages:
                    output.append(msg.to_langchain())

        # Current topic (full fidelity)
        for msg in self.current.messages:
            output.append(msg.to_langchain())

        return output

    def get_context_messages(self) -> list[dict]:
        """Alias for to_langchain_messages."""
        return self.to_langchain_messages()

    # ── Compression engine ────────────────────────────────────────────

    def compress(self) -> None:
        """Run synchronous compression pass."""
        with self._compress_lock:
            self._compress_pass()

    def compress_async(self) -> None:
        """Run compression in background thread."""
        if not self.config.compress_in_background:
            self.compress()
            return
        if self._compress_thread and self._compress_thread.is_alive():
            return  # Already compressing
        self._compress_thread = threading.Thread(
            target=self._safe_compress, daemon=True,
            name="history-compress",
        )
        self._compress_thread.start()

    def wait_for_compression(self, timeout: Optional[float] = None) -> None:
        """Wait for background compression to finish."""
        if self._compress_thread and self._compress_thread.is_alive():
            self._compress_thread.join(
                timeout=timeout or self.config.compression_timeout_seconds
            )

    def _safe_compress(self) -> None:
        try:
            self.compress()
        except Exception as e:
            logger.error(f"Background compression failed: {e}")

    def _compress_pass(self) -> None:
        """Multi-strategy compression, iterative until within budget.

        Order: trim large msgs → summarize topics → merge topics→bulk → merge bulks
        """
        target = int(self.config.max_context_tokens * self.config.compression_target_ratio)

        for _ in range(10):  # Max iterations to prevent infinite loop
            if self.total_tokens <= target:
                return

            # Strategy 1: Trim oversized individual messages
            if self._trim_large_messages():
                continue
            # Strategy 2: Summarize oldest unsummarized topic
            if self._summarize_oldest_topic():
                continue
            # Strategy 3: Merge oldest topics into a bulk
            if self._merge_topics_to_bulk():
                continue
            # Strategy 4: Merge old bulks together
            if self._merge_old_bulks():
                continue

            logger.warning(
                f"Compression exhausted: {self.total_tokens} tokens (target: {target})"
            )
            break

    def _trim_large_messages(self) -> bool:
        trimmed = False
        max_chars = self.config.large_message_trim_chars
        for topic in self.topics:
            for msg in topic.messages:
                if len(msg.content) > max_chars:
                    msg.trim_content(max_chars)
                    trimmed = True
        return trimmed

    def _summarize_oldest_topic(self) -> bool:
        for topic in self.topics:
            if not topic.is_summarized and topic.messages:
                before = sum(m.tokens for m in topic.messages)
                topic.summary = self.summarizer.summarize_messages(topic.messages)
                after = approximate_tokens(topic.summary)
                logger.info(f"Summarized topic {topic.id}: {before} → {after} tokens")
                return True
        return False

    def _merge_topics_to_bulk(self) -> bool:
        n = self.config.topics_merge_count
        if len(self.topics) < n:
            return False

        to_merge = self.topics[:n]
        self.topics = self.topics[n:]

        # Ensure all topics are summarized before merging
        for t in to_merge:
            if not t.is_summarized:
                t.summary = self.summarizer.summarize_messages(t.messages)

        bulk = Bulk(
            summary=self.summarizer.summarize_topics(to_merge),
            topic_count=len(to_merge),
            span_start=to_merge[0].created_at,
            span_end=to_merge[-1].created_at,
        )
        self.bulks.append(bulk)
        logger.info(f"Created bulk {bulk.id} from {n} topics: {bulk.tokens} tokens")
        return True

    def _merge_old_bulks(self) -> bool:
        if len(self.bulks) < 2:
            return False
        b1, b2 = self.bulks[0], self.bulks[1]
        self.bulks = self.bulks[2:]

        merged_summary = self.summarizer._call_llm(
            "Merge these two summaries into one. Max 150 words. Preserve only critical info.",
            f"{b1.summary}\n\n---\n\n{b2.summary}",
            fallback=f"{b1.summary[:300]} | {b2.summary[:300]}",
        )
        merged = Bulk(
            summary=merged_summary,
            topic_count=b1.topic_count + b2.topic_count,
            span_start=b1.span_start,
            span_end=b2.span_end,
        )
        self.bulks.insert(0, merged)
        return True

    # ── Model tier switching ──────────────────────────────────────────

    def set_model_tier(self, tier: str) -> None:
        """Update context limit when switching LLM cascade tiers."""
        if tier in self.config.tier_context_limits:
            self.config.max_context_tokens = self.config.tier_context_limits[tier]
            logger.info(f"Context limit → {self.config.max_context_tokens} for tier '{tier}'")

    # ── Serialization ─────────────────────────────────────────────────

    def serialize(self) -> str:
        """Serialize to JSON for PostgreSQL storage."""
        return json.dumps({
            "version": 1,
            "bulks": [b.to_dict() for b in self.bulks],
            "topics": [t.to_dict() for t in self.topics],
            "current": self.current.to_dict(),
        }, ensure_ascii=False)

    @classmethod
    def deserialize(cls, json_data: str, config: Optional[CompressionConfig] = None) -> "History":
        """Restore from JSON."""
        data = json.loads(json_data)
        h = cls(config=config)
        h.bulks = [Bulk.from_dict(b) for b in data.get("bulks", [])]
        h.topics = [Topic.from_dict(t) for t in data.get("topics", [])]
        h.current = Topic.from_dict(data.get("current", {}))
        return h

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.config.max_context_tokens,
            "utilization": f"{self.total_tokens / max(1, self.config.max_context_tokens):.1%}",
            "bulks": len(self.bulks),
            "topics": len(self.topics),
            "current_messages": len(self.current.messages),
            "needs_compression": self.needs_compression,
        }

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"History({s['total_tokens']}/{s['max_tokens']}tok, "
            f"{s['bulks']}B {s['topics']}T {s['current_messages']}M)"
        )


# ── Per-sender history management ────────────────────────────────────────────


_histories: dict[str, History] = {}
_histories_lock = threading.Lock()


def get_history(sender_id: str, config: Optional[CompressionConfig] = None) -> History:
    """Get or create a per-sender compressed history."""
    with _histories_lock:
        if sender_id not in _histories:
            _histories[sender_id] = History(config=config)
        return _histories[sender_id]


def clear_history(sender_id: str) -> None:
    """Clear a sender's compressed history."""
    with _histories_lock:
        _histories.pop(sender_id, None)
