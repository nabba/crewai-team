"""
tools_timeout — Process-wide wall-clock timeout for every CrewAI tool.

Motivation
==========
A CrewAI tool call can block indefinitely if its underlying code
hangs — a stuck DB connection, a LiteLLM call that never returns, a
memory backend that blocks on an empty queue, a network fetch that
times out at 10 minutes instead of the expected 10 seconds.  None of
these are individually catastrophic, but the outer handle_task only
has LLM-activity-level heartbeat instrumentation — it sees the tool
thread as "still alive" and waits up to 45 min for the hard cap to
fire.  Meanwhile the tool's span stays ``running`` forever (see
:mod:`app.control_plane.crew_task_spans` and the 2026-04-22 PSP trace
where ``recall_facts`` was stuck ``running`` for 2h 11m).

This module monkey-patches ``BaseTool.run`` to enforce a **hard
wall-clock timeout** on every tool invocation.  When the timeout
fires:

1. The tool returns a clear string error to the calling agent (CrewAI's
   ReAct loop handles string returns gracefully — the agent can decide
   to retry, switch tools, or give up).
2. The matching ``crew_task_spans`` row is stamped ``failed`` with an
   explanatory error so the dashboard doesn't lie about state.
3. The underlying Python thread is abandoned (Python can't safely kill
   a thread mid-syscall).  It will complete on its own and its return
   value is discarded.  This is the same tradeoff CrewAI's ReAct
   supervisor already makes for long-running tool calls.

Policy
------
Defaults:

* ``_DEFAULT_TIMEOUT_SECONDS = 180`` (3 min) — plenty for any
  single-source fetch + parse; short enough to keep agents unblocked.

Overrides live in ``_PER_TOOL_OVERRIDES`` for tools that legitimately
run long (e.g. the research orchestrator internally fans out to many
subjects and has its own budget logic).

Disabling
---------
Set ``TOOL_TIMEOUTS_DISABLED=1`` in the environment to skip the
monkey-patch entirely.  Useful when debugging a single tool with a
debugger attached.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


# Default per-call timeout.  Applies to every BaseTool unless
# overridden below.
_DEFAULT_TIMEOUT_SECONDS: int = 180

# Explicit per-tool budgets.  Key = tool name (``self.name``).
# Anything not in this map uses ``_DEFAULT_TIMEOUT_SECONDS``.
_PER_TOOL_OVERRIDES: dict[str, int] = {
    # Orchestrators fan out internally and carry their own budget
    # logic — give them the full handle_task window.
    "research_orchestrator": 1500,
    # Memory retrieval should be fast; Mem0 queries that take > 60 s
    # usually mean Neo4j/pgvector is wedged, and we'd rather fail fast.
    "recall_facts": 60,
    "persist_fact": 60,
    "persist_conversation": 60,
    "team_memory_retrieve": 60,
    # Search APIs have their own network timeout but occasionally hang
    # on connection setup.
    "web_search": 30,
    "web_fetch": 60,
    # Firecrawl is slower — scraping + markdown conversion.
    "firecrawl_scrape": 90,
    "firecrawl_search": 60,
    "firecrawl_extract": 120,
    "firecrawl_map": 60,
    "firecrawl_crawl": 180,
    # Knowledge-base searches (pgvector) are fast unless the index is
    # rebuilding.
    "search_knowledge_base": 30,
}

_installed: bool = False
_install_lock = threading.Lock()

# ── Phase E2: tool-activity heartbeat for the handle_task stall checker ──
# A monotonic timestamp updated on every tool ENTRY and EXIT. The stall
# checker in main.py reads this as a fourth diagnostic signal alongside
# output-progress, zero-output, and llm-activity. Purpose: when an agent
# is genuinely cycling through tools (each completing quickly), the LLM
# activity timestamp also moves; but if the LLM is wedged WAITING on a
# tool that is itself idle, only the tool-activity stamp differentiates
# "tool computing" from "tool stuck mid-call".
_last_tool_activity_ts: float = 0.0
_last_tool_timeout_ts: float = 0.0
_tool_timeout_count: int = 0


def _record_tool_activity() -> None:
    """Update the tool heartbeat timestamp. Cheap; called per tool call."""
    global _last_tool_activity_ts
    _last_tool_activity_ts = time.monotonic()


def seconds_since_last_tool_activity() -> float | None:
    """Seconds since the last tool started OR finished, or None if never.

    Mirrors the contract of
    :func:`app.rate_throttle.seconds_since_last_llm_activity`. Used by
    main.py's _stall_check for diagnostic logging on kill decisions.
    """
    if _last_tool_activity_ts == 0.0:
        return None
    return time.monotonic() - _last_tool_activity_ts


def get_tool_timeout_count() -> int:
    """How many tool timeouts have fired in this process. Diagnostic only."""
    return _tool_timeout_count


def seconds_since_last_tool_timeout() -> float | None:
    """Seconds since the most recent tool timeout fired (None if none yet)."""
    if _last_tool_timeout_ts == 0.0:
        return None
    return time.monotonic() - _last_tool_timeout_ts


def _resolve_timeout(tool_name: str) -> int:
    """Look up the wall-clock budget for a tool by name."""
    if not tool_name:
        return _DEFAULT_TIMEOUT_SECONDS
    return _PER_TOOL_OVERRIDES.get(tool_name, _DEFAULT_TIMEOUT_SECONDS)


def register_tool_timeout(tool_name: str, seconds: int) -> None:
    """Override a tool's timeout at runtime.  Useful when a new tool
    lands or a user wants to tune a specific budget without editing
    the default map."""
    if not tool_name or seconds <= 0:
        return
    _PER_TOOL_OVERRIDES[tool_name] = int(seconds)


def install() -> None:
    """Monkey-patch ``BaseTool.run`` to enforce a wall-clock timeout.

    Idempotent.  No-op if ``TOOL_TIMEOUTS_DISABLED=1`` or if CrewAI
    isn't importable (tests without the full stack installed).
    """
    global _installed
    with _install_lock:
        if _installed:
            return
        if os.environ.get("TOOL_TIMEOUTS_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.info("tools_timeout: disabled via TOOL_TIMEOUTS_DISABLED env var")
            _installed = True
            return
        try:
            from crewai.tools import BaseTool
        except ImportError:
            logger.warning("tools_timeout: crewai.tools.BaseTool not importable — skip")
            _installed = True
            return

        original_run = BaseTool.run

        def _timed_run(self: Any, *args: Any, **kwargs: Any) -> Any:
            tool_name = getattr(self, "name", "") or type(self).__name__
            budget = _resolve_timeout(tool_name)
            # Heartbeat: record tool entry. The handle_task stall checker
            # reads this via seconds_since_last_tool_activity().
            _record_tool_activity()
            # Fresh single-worker pool per call.  A hung previous call
            # can't block the next one — abandoning it with
            # ``shutdown(wait=False)`` is fine because the leaked
            # thread eventually completes on its own (same trade-off
            # as research_orchestrator's per-adapter pool).
            pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"tool-{tool_name}",
            )
            try:
                fut = pool.submit(original_run, self, *args, **kwargs)
                try:
                    return fut.result(timeout=budget)
                except TimeoutError:
                    msg = (
                        f"Error: tool '{tool_name}' timed out after {budget}s. "
                        f"The underlying call did not return in time — most "
                        f"likely the upstream service is unreachable or "
                        f"overloaded. Try a different approach or a narrower "
                        f"query."
                    )
                    logger.warning(
                        "tools_timeout: '%s' exceeded %ss budget — returning "
                        "timeout error to agent", tool_name, budget,
                    )
                    # Phase E2: surface tool timeouts to handle_task stall
                    # checker so kill decisions can include "tool just
                    # timed out — give the agent a moment to react."
                    global _last_tool_timeout_ts, _tool_timeout_count
                    _last_tool_timeout_ts = time.monotonic()
                    _tool_timeout_count += 1
                    return msg
                except Exception as exc:
                    # Propagate non-timeout exceptions to CrewAI's
                    # normal error handler.  This preserves the
                    # existing ToolUsageErrorEvent → span failure path.
                    raise exc
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
                # Heartbeat: record tool exit (success, error, OR timeout).
                _record_tool_activity()

        BaseTool.run = _timed_run  # type: ignore[method-assign]
        _installed = True
        logger.info(
            "tools_timeout: installed; default=%ds, overrides=%d tools",
            _DEFAULT_TIMEOUT_SECONDS, len(_PER_TOOL_OVERRIDES),
        )
