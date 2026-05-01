"""
dead_letter_inbound — bounded DLQ for messages rejected by load shedding.

Phase F3 (initial in-process backend) + multi-pod follow-up
(Redis-backed backend, opt-in via ``REDIS_DLQ_URL``).

Today, when ``handle_task`` is at capacity (inflight ≥ shed_threshold),
the message is dropped. This module buffers it instead so a follow-up
idle scheduler pass can replay when capacity returns.

Two backends, swappable at module-import time:

  1. **In-process deque** (default) — bounded, fast, zero deps, lost on
     restart. Correct for single-pod laptop deploys and dev clusters.

  2. **Redis-backed** (opt-in) — set ``REDIS_DLQ_URL`` (e.g.
     ``redis://botarmy-redis:6379/0``) and the queue moves to a Redis
     LIST keyed by ``REDIS_DLQ_KEY`` (default ``botarmy:dlq:inbound``).
     Multi-pod deployments share one queue. Pod restarts don't lose
     messages. Falls back to the in-process backend silently if Redis
     becomes unreachable mid-run; reconnects on the next operation.

The PUBLIC API is identical for both backends. Callers (``main.py``
load-shed path, ``idle_scheduler.py`` drain job) do not need to know
which backend is active.

Failure modes:
  * Redis URL set but unreachable at import → log + fall back to
    in-process. Re-tries on next operation (no permanent disable).
  * Redis available at start, dies later → individual operation logs
    a warning and falls back to in-process for THAT operation. The
    queue may end up split across the two backends temporarily; the
    drain pass reads from both.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Coroutine, Deque, Iterable

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
_DEFAULT_CAPACITY = 200          # in-process cap; Redis cap enforced via LTRIM
_DEFAULT_MAX_AGE_SECONDS = 1800  # 30 min — older messages dropped on drain

# Redis backend (opt-in). REDIS_DLQ_URL is its own env var so that ops
# can wire a dedicated Redis without affecting any other Redis usage.
_REDIS_URL = os.environ.get("REDIS_DLQ_URL", "").strip()
_REDIS_KEY = os.environ.get("REDIS_DLQ_KEY", "botarmy:dlq:inbound")
_REDIS_CAPACITY = int(os.environ.get("REDIS_DLQ_CAPACITY", "1000"))


# ── In-process backend state ──────────────────────────────────────────
_queue_lock = threading.Lock()
_queue: Deque[dict[str, Any]] = deque(maxlen=_DEFAULT_CAPACITY)


# ── Redis backend wiring ──────────────────────────────────────────────
_redis_client: Any = None
_redis_lock = threading.Lock()


def _get_redis() -> Any:
    """Lazy-create a Redis client. Returns None if unconfigured/unreachable."""
    global _redis_client
    if not _REDIS_URL:
        return None
    if _redis_client is not None:
        return _redis_client
    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            import redis  # type: ignore
        except ImportError:
            logger.warning(
                "dlq: REDIS_DLQ_URL is set but `redis` package is not installed; "
                "falling back to in-process backend"
            )
            return None
        try:
            client = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
            client.ping()  # surface connection errors at first use
            _redis_client = client
            logger.info("dlq: Redis backend active (key=%s, cap=%d)", _REDIS_KEY, _REDIS_CAPACITY)
            return client
        except Exception as exc:
            logger.warning(
                "dlq: Redis at %s unreachable (%s); falling back to in-process",
                _REDIS_URL, exc,
            )
            return None


def _backend_in_use() -> str:
    """Return ``"redis"`` if Redis is wired and reachable, else ``"memory"``."""
    return "redis" if _get_redis() is not None else "memory"


def _entry_to_json(entry: dict[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False)


def _entry_from_json(s: str) -> dict[str, Any] | None:
    try:
        return json.loads(s)
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────


def enqueue(sender: str, text: str, attachments: list | None = None) -> bool:
    """Buffer a load-shedded message for later replay.

    Returns True if enqueued, False if the active backend is full.
    """
    entry: dict[str, Any] = {
        "sender": sender,
        "text": text,
        "attachments": list(attachments or []),
        "enqueued_at": time.time(),
    }

    rd = _get_redis()
    if rd is not None:
        try:
            # LPUSH then LTRIM to enforce the cap (FIFO semantics: RPOP on drain).
            pipe = rd.pipeline()
            pipe.lpush(_REDIS_KEY, _entry_to_json(entry))
            pipe.ltrim(_REDIS_KEY, 0, _REDIS_CAPACITY - 1)
            pipe.llen(_REDIS_KEY)
            _, _, depth = pipe.execute()
            logger.info("dlq[redis]: enqueued message from %s (depth=%d)", sender, depth)
            return True
        except Exception as exc:
            logger.warning("dlq[redis]: enqueue failed (%s) — falling back to in-process", exc)
            # Fall through to in-process backend.

    with _queue_lock:
        if len(_queue) >= (_queue.maxlen or _DEFAULT_CAPACITY):
            logger.warning(
                "dlq[memory]: full (cap=%d) — refusing to buffer message from %s",
                _queue.maxlen, sender,
            )
            return False
        _queue.append(entry)
        depth = len(_queue)
    logger.info("dlq[memory]: enqueued message from %s (depth=%d)", sender, depth)
    return True


def queue_depth() -> int:
    """Number of messages currently buffered (sum across both backends if mixed)."""
    total = 0
    rd = _get_redis()
    if rd is not None:
        try:
            total += int(rd.llen(_REDIS_KEY))
        except Exception as exc:
            logger.debug("dlq[redis]: llen failed: %s", exc)
    with _queue_lock:
        total += len(_queue)
    return total


def peek() -> list[dict[str, Any]]:
    """Read-only snapshot for observability. Strips PII (text summarized)."""
    out: list[dict[str, Any]] = []
    now = time.time()

    def _summarize(e: dict[str, Any]) -> dict[str, Any]:
        return {
            "sender": e.get("sender", ""),
            "text_preview": (e.get("text", "")[:80] + "…")
                if len(e.get("text", "")) > 80 else e.get("text", ""),
            "attachment_count": len(e.get("attachments", [])),
            "enqueued_at": e.get("enqueued_at", 0.0),
            "age_seconds": now - e.get("enqueued_at", now),
        }

    rd = _get_redis()
    if rd is not None:
        try:
            raw = rd.lrange(_REDIS_KEY, 0, -1)
            for s in raw:
                e = _entry_from_json(s)
                if e is not None:
                    out.append(_summarize(e))
        except Exception as exc:
            logger.debug("dlq[redis]: lrange failed: %s", exc)

    with _queue_lock:
        for e in _queue:
            out.append(_summarize(e))
    return out


def _pop_one_from_redis() -> dict[str, Any] | None:
    """RPOP one entry from Redis. Returns None if Redis missing or empty."""
    rd = _get_redis()
    if rd is None:
        return None
    try:
        raw = rd.rpop(_REDIS_KEY)
    except Exception as exc:
        logger.warning("dlq[redis]: rpop failed: %s", exc)
        return None
    if raw is None:
        return None
    return _entry_from_json(raw)


def _pop_one_from_memory() -> dict[str, Any] | None:
    with _queue_lock:
        if _queue:
            return _queue.popleft()
    return None


def _drain_n(n: int, max_age_seconds: float) -> list[dict[str, Any]]:
    """Pop up to ``n`` entries (Redis first, then in-process), dropping
    expired ones. Returns the list of entries to replay.
    """
    out: list[dict[str, Any]] = []
    now = time.time()
    while len(out) < n:
        msg = _pop_one_from_redis() or _pop_one_from_memory()
        if msg is None:
            break
        if now - msg.get("enqueued_at", now) > max_age_seconds:
            logger.info(
                "dlq: dropping expired message from %s (age=%.0fs)",
                msg.get("sender"), now - msg.get("enqueued_at", now),
            )
            continue
        out.append(msg)
    return out


def drain(
    handler: Callable[[str, str, list], Coroutine[Any, Any, Any]],
    *,
    inflight_count: int,
    shed_threshold: int,
    max_to_drain: int = 5,
    max_age_seconds: float = _DEFAULT_MAX_AGE_SECONDS,
) -> dict[str, int]:
    """Replay buffered messages while capacity exists.

    Backend-agnostic. Pulls from Redis first (multi-pod canonical) then
    in-process. Drops expired (> ``max_age_seconds``) entries.
    """
    counts = {"replayed": 0, "expired": 0, "skipped_capacity": 0}
    headroom = shed_threshold - 1 - inflight_count
    if headroom <= 0:
        counts["skipped_capacity"] = queue_depth()
        return counts

    budget = min(max_to_drain, headroom)
    pre_depth = queue_depth()
    to_replay = _drain_n(budget, max_age_seconds)
    counts["expired"] = max(0, (pre_depth - queue_depth()) - len(to_replay))

    import asyncio
    for msg in to_replay:
        try:
            asyncio.run(handler(msg["sender"], msg["text"], msg.get("attachments", [])))
            counts["replayed"] += 1
        except RuntimeError:
            # Already inside an event loop — schedule on it instead.
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(
                    handler(msg["sender"], msg["text"], msg.get("attachments", []))
                )
                counts["replayed"] += 1
            except Exception as exc:
                logger.warning(
                    "dlq: replay failed (loop dispatch) for %s: %s",
                    msg.get("sender"), exc,
                )
        except Exception as exc:
            logger.warning(
                "dlq: replay failed for %s: %s — message dropped",
                msg.get("sender"), exc,
            )

    if counts["replayed"] or counts["expired"]:
        logger.info(
            "dlq: drain done [%s] — replayed=%d expired=%d remaining=%d",
            _backend_in_use(), counts["replayed"], counts["expired"], queue_depth(),
        )
    return counts


def clear() -> int:
    """Drop every buffered message from BOTH backends. Tests + emergency only."""
    n = 0
    rd = _get_redis()
    if rd is not None:
        try:
            n += int(rd.llen(_REDIS_KEY))
            rd.delete(_REDIS_KEY)
        except Exception as exc:
            logger.warning("dlq[redis]: clear failed: %s", exc)
    with _queue_lock:
        n += len(_queue)
        _queue.clear()
    logger.info("dlq: cleared %d messages (backend=%s)", n, _backend_in_use())
    return n


def backend_info() -> dict[str, Any]:
    """Diagnostic snapshot: which backend is active + sizing knobs.

    Surfaced via :func:`app.control_plane.dashboard_api.list_idle_jobs`
    so operators can confirm at a glance whether Redis is being used.
    """
    rd = _get_redis()
    return {
        "backend": "redis" if rd is not None else "memory",
        "redis_url_configured": bool(_REDIS_URL),
        "redis_key": _REDIS_KEY if _REDIS_URL else None,
        "redis_capacity": _REDIS_CAPACITY if _REDIS_URL else None,
        "memory_capacity": _DEFAULT_CAPACITY,
        "max_age_seconds": _DEFAULT_MAX_AGE_SECONDS,
        "depth": queue_depth(),
    }
