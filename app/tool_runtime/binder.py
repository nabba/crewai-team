"""Per-crew, per-run tool binder.

A ToolBinder owns the *current set* of tools available to one agent.
It is the bridge between a (future) registry and CrewAI's
agent_executor.original_tools list. Phase 0 keeps it simple:

  binder = ToolBinder(core_tools=[...], available={"name": factory_callable})
  binder.load("name")          # adds to active set, marks dirty
  binder.tools                  # current active list — what the executor sees
  binder.dirty                  # True if a load happened since last clear
  binder.consume_pending()      # returns names loaded since last call

The dirty flag is the signal LoadableAgentExecutor uses to know it
should re-render the openai_tools schema before the next LLM call.
The pending-loads list feeds the user-turn announcement that tells
the model "you now have access to: ...".

Phase 1 will replace `available` with a real registry + capability
search; the binder API stays the same.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolNotAvailable(KeyError):
    """Tool name unknown to this binder's registry slice."""


class ToolBinder:
    def __init__(
        self,
        *,
        core_tools: list[Any],
        available: dict[str, Callable[[], Any]],
    ) -> None:
        """
        Args:
            core_tools: Always-loaded tools — appear in every iteration.
                These ARE in the system prompt; they never change.
            available: name → factory_callable mapping for lazily-
                loadable tools. Factory takes no args (closures
                capture agent_id etc.) and returns a BaseTool.
        """
        self._core: list[Any] = list(core_tools)
        self._available: dict[str, Callable[[], Any]] = dict(available)
        self._loaded: dict[str, Any] = {}
        self._pending_announce: list[str] = []
        self._dirty = False
        self._lock = threading.Lock()

    # ── Read API ────────────────────────────────────────────────

    @property
    def tools(self) -> list[Any]:
        """Current active toolset (core + dynamically-loaded)."""
        with self._lock:
            return self._core + list(self._loaded.values())

    @property
    def dirty(self) -> bool:
        """True if any load happened since clear_dirty() was last called."""
        return self._dirty

    @property
    def loaded_names(self) -> list[str]:
        with self._lock:
            return list(self._loaded.keys())

    def catalog_names(self) -> list[str]:
        """Names of tools the binder *could* load (for tool_search)."""
        with self._lock:
            return sorted(self._available.keys())

    # ── Write API ───────────────────────────────────────────────

    def load(self, name: str) -> Any:
        """Load a tool by name. Idempotent.

        Returns the tool instance. Raises ToolNotAvailable if the
        name isn't in this binder's registry slice.
        """
        with self._lock:
            if name in self._loaded:
                return self._loaded[name]
            if name in (t.name for t in self._core):
                # Core tool — already loaded, return the existing instance.
                for t in self._core:
                    if t.name == name:
                        return t
            factory = self._available.get(name)
            if factory is None:
                raise ToolNotAvailable(name)
            instance = factory()
            self._loaded[name] = instance
            self._pending_announce.append(name)
            self._dirty = True
            logger.info("ToolBinder: loaded %s (now %d tools)", name, len(self._core) + len(self._loaded))
            return instance

    def clear_dirty(self) -> None:
        """Mark the binder as clean — call after the executor has
        re-rendered its tool schemas."""
        with self._lock:
            self._dirty = False

    def consume_pending(self) -> list[str]:
        """Return + clear the list of tool names loaded since last
        consume_pending() call. Used to build the per-turn
        announcement message."""
        with self._lock:
            out = list(self._pending_announce)
            self._pending_announce.clear()
            return out
