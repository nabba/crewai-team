"""Recovery-loop ↔ long-horizon-threads bridge.

PROGRAM §46.1 (Q8.1). When the recovery loop is selecting a strategy
for a refusal-shaped response, the operator may have open threads
that carry background context — sub-questions, blockers, and
*unblock hints* (what the operator hypothesised might resolve the
current block). This module surfaces those hints to the recovery
loop without imposing any decision: ``maybe_recover`` reads
``ctx["thread_hints"]`` and individual strategies may consult it,
but no strategy is forced to use it.

The bridge is observational + read-only:
  * Never mutates threads.
  * Never blocks recovery on thread availability — broken
    ``app.threads`` returns an empty list.
  * Caps payload size to keep recovery prompts compact.

Design choice: we surface ONLY hints that the operator has
deliberately filed (via ``add_unblock_hint`` / ``/thread hint`` /
``POST /api/cp/threads/<id>/unblock-hint``). We do NOT auto-fish
through ``notes`` or ``description`` — those are unbounded text and
including them would degrade strategy selection on noise.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Hard caps on how many threads we surface to recovery + how much
# text per thread. The recovery prompts are token-sensitive — if we
# inject 30 KB of thread hints we'll just blow the context budget.
_MAX_THREADS = 5
_MAX_HINTS_PER_THREAD = 3
_MAX_HINT_CHARS = 240


def collect_open_thread_hints(*, max_threads: int = _MAX_THREADS) -> list[dict[str, Any]]:
    """Return open-thread unblock-hint payloads for recovery context.

    Each payload::

        {
          "thread_id": "<id>",
          "title": "<thread title>",
          "status": "open" | "in_progress" | "blocked",
          "blockers": [<top-N blocker strings>],
          "hints": [<top-N unblock-hint strings>],
        }

    Threads with no ``unblock_hints`` AND no ``blockers`` are omitted
    — they carry no actionable signal for strategy selection.
    Failure-isolated: returns ``[]`` if the threads module is
    unavailable.
    """
    try:
        from app.threads import list_open
    except Exception:
        return []
    try:
        threads = list_open(limit=max(1, max_threads))
    except Exception:
        logger.debug(
            "thread_consultation: list_open raised", exc_info=True,
        )
        return []
    out: list[dict[str, Any]] = []
    for t in threads:
        hints = list(getattr(t, "unblock_hints", []) or [])
        blockers = list(getattr(t, "blockers", []) or [])
        if not hints and not blockers:
            continue  # nothing actionable
        out.append({
            "thread_id": t.id,
            "title": (t.title or "")[:120],
            "status": getattr(t.status, "value", str(t.status)),
            "blockers": [b[:_MAX_HINT_CHARS] for b in blockers[:_MAX_HINTS_PER_THREAD]],
            "hints": [h[:_MAX_HINT_CHARS] for h in hints[:_MAX_HINTS_PER_THREAD]],
        })
        if len(out) >= max_threads:
            break
    return out


def format_for_prompt(payloads: list[dict[str, Any]]) -> str:
    """Render the payloads as a short markdown block.

    Strategies that LLM-prompt (re_route, escalate_tier, ...) can
    paste this into their prompt under a heading like "Operator's
    open lines of inquiry (may be relevant)". Returns empty string
    when payloads is empty so callers can ``if not text: skip``.
    """
    if not payloads:
        return ""
    lines: list[str] = []
    for p in payloads:
        lines.append(f"- **{p['title']}** ({p['status']}, id={p['thread_id'][:8]})")
        for b in p.get("blockers", []):
            lines.append(f"  - 🚧 blocker: {b}")
        for h in p.get("hints", []):
            lines.append(f"  - 💡 hint: {h}")
    return "\n".join(lines)
