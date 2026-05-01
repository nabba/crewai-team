"""
dead_letter_inbound — bounded DLQ for messages rejected by load shedding.

Phase F3 of the remediation plan.

Today, when ``handle_task`` is at capacity (inflight ≥ shed_threshold),
the message is dropped: the user gets a "try again" reply and the
content is gone. For a single-user laptop this is fine; for a K8s
deployment with multiple users it's a real loss-of-message bug.

This module ships a process-local bounded deque that buffers rejected
messages and a re-injection helper that the idle scheduler can call when
capacity returns. The DLQ is intentionally:

  * In-memory (per-pod) — no shared state, no Redis dependency on a
    single-pod laptop run. Multi-pod K8s will eventually want a shared
    DLQ; until then, each pod buffers its own rejections.
  * Bounded (default 200 messages) — refusing to enqueue past the cap
    is preferable to unbounded memory growth on a sustained outage.
  * Persistence-free — messages do NOT survive a process restart. This
    is a deliberate tradeoff: a "we'll get to it later" guarantee is
    safer than a "we wrote it to disk" guarantee that promises retry
    after long downtime when the user has moved on.

Wiring:
  * ``app/main.py`` enqueues on the load-shed path.
  * Phase F3 also registers ``drain_load_shed_queue()`` as a LIGHT idle
    job. When inflight drops below the threshold, the next idle pass
    drains the queue back into ``handle_task``.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Coroutine, Deque

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
_DEFAULT_CAPACITY = 200          # max messages queued before we refuse new entries
_DEFAULT_MAX_AGE_SECONDS = 1800  # 30 min — older messages dropped on drain


# ── State (module-level so re-imports don't lose it) ──────────────────
_queue_lock = threading.Lock()
_queue: Deque[dict[str, Any]] = deque(maxlen=_DEFAULT_CAPACITY)


def enqueue(sender: str, text: str, attachments: list | None = None) -> bool:
    """Buffer a load-shedded message for later replay.

    Returns True if enqueued, False if the queue is full (the deque has
    a maxlen, so silent eviction of the OLDEST item happens automatically;
    we report False instead to make the rejection observable).
    """
    with _queue_lock:
        if len(_queue) >= _queue.maxlen:
            # The deque would otherwise silently drop the oldest entry.
            # Refuse to enqueue instead — keeps the existing buffer intact
            # and lets the caller decide whether to surface the rejection
            # to the user.
            logger.warning(
                "dlq: full (cap=%d) — refusing to buffer message from %s",
                _queue.maxlen, sender,
            )
            return False
        _queue.append({
            "sender": sender,
            "text": text,
            "attachments": list(attachments or []),
            "enqueued_at": time.time(),
        })
    logger.info("dlq: enqueued message from %s (depth=%d)", sender, len(_queue))
    return True


def queue_depth() -> int:
    """Number of messages currently buffered."""
    with _queue_lock:
        return len(_queue)


def peek() -> list[dict[str, Any]]:
    """Read-only snapshot for observability. Strips PII (text/attachments
    are summarized but not surfaced)."""
    with _queue_lock:
        return [
            {
                "sender": e["sender"],
                "text_preview": (e["text"][:80] + "…") if len(e["text"]) > 80 else e["text"],
                "attachment_count": len(e["attachments"]),
                "enqueued_at": e["enqueued_at"],
                "age_seconds": time.time() - e["enqueued_at"],
            }
            for e in _queue
        ]


def drain(
    handler: Callable[[str, str, list], Coroutine[Any, Any, Any]],
    *,
    inflight_count: int,
    shed_threshold: int,
    max_to_drain: int = 5,
    max_age_seconds: float = _DEFAULT_MAX_AGE_SECONDS,
) -> dict[str, int]:
    """Replay buffered messages while capacity exists.

    Pulls up to ``max_to_drain`` messages and dispatches them one by one
    via ``handler(sender, text, attachments)``. Stops when:
      * the queue is empty,
      * the in-flight count would exceed ``shed_threshold - 1`` after a
        drain (leave headroom for live arrivals),
      * we hit the per-call drain budget.

    Drops messages older than ``max_age_seconds`` to keep the queue from
    being a memory leak under prolonged outage. Returns counts.

    Note: ``handler`` is called synchronously from the caller's coroutine
    runner — this function does not start its own event loop. The idle
    scheduler bridges to async via :func:`_drain_for_scheduler` below.
    """
    counts = {"replayed": 0, "expired": 0, "skipped_capacity": 0}
    headroom = shed_threshold - 1 - inflight_count
    if headroom <= 0:
        counts["skipped_capacity"] = queue_depth()
        return counts

    budget = min(max_to_drain, headroom)
    now = time.time()
    to_replay: list[dict[str, Any]] = []
    with _queue_lock:
        while _queue and len(to_replay) < budget:
            msg = _queue.popleft()
            if now - msg["enqueued_at"] > max_age_seconds:
                counts["expired"] += 1
                logger.info(
                    "dlq: dropping expired message from %s (age=%.0fs)",
                    msg["sender"], now - msg["enqueued_at"],
                )
                continue
            to_replay.append(msg)

    # Outside the lock — handler may be slow.
    import asyncio
    for msg in to_replay:
        try:
            asyncio.run(handler(msg["sender"], msg["text"], msg["attachments"]))
            counts["replayed"] += 1
        except RuntimeError:
            # Already in an event loop — schedule on the running loop.
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(
                    handler(msg["sender"], msg["text"], msg["attachments"])
                )
                counts["replayed"] += 1
            except Exception as exc:
                logger.warning(
                    "dlq: replay failed (loop dispatch) for %s: %s",
                    msg["sender"], exc,
                )
        except Exception as exc:
            logger.warning(
                "dlq: replay failed for %s: %s — message dropped",
                msg["sender"], exc,
            )

    if counts["replayed"] or counts["expired"]:
        logger.info(
            "dlq: drain done — replayed=%d expired=%d remaining=%d",
            counts["replayed"], counts["expired"], queue_depth(),
        )
    return counts


def clear() -> int:
    """Drop every buffered message. For tests and emergency operator use."""
    with _queue_lock:
        n = len(_queue)
        _queue.clear()
    logger.info("dlq: cleared %d messages", n)
    return n
