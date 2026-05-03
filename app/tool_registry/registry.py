"""ToolRegistry — the global singleton of all registered tools.

Thread-safe (single _lock; Python GIL covers most reads). Idempotent
``register`` — re-registering the same name with an identical spec is
a no-op; re-registering with a CHANGED spec emits a warning and keeps
the old one (so a hot reload doesn't silently swap behaviour).

The registry is **in-memory primary**. Postgres snapshot (see
``persistence.py``) is for cross-process visibility, not source of
truth. If Postgres is unreachable at boot, the registry still works —
we just don't snapshot.
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable

from app.tool_registry.types import Tier, ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Global singleton. Use ``ToolRegistry.instance()``."""

    _instance: "ToolRegistry | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the singleton — tests can rebuild a clean registry.

        Decorator side-effects from earlier imports won't fire again
        (sys.modules caches the modules), so call ``replay_decorations()``
        afterwards if you want all previously-decorated tools back.
        """
        with cls._instance_lock:
            cls._instance = None

    def replay_decorations(self) -> int:
        """Re-register every spec the @register_tool decorator has ever
        produced this Python process.

        Used by tests + boot fallback: if decorators have already fired
        (modules cached in sys.modules), we still want the registry to
        contain those specs after a singleton reset.

        Returns the count of specs replayed.
        """
        # Local import — decorator imports registry, so this avoids the
        # circular at module-load time.
        from app.tool_registry.decorator import _get_decorated_specs

        specs = _get_decorated_specs()
        for spec in specs.values():
            self.register(spec)
        return len(specs)

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._lock = threading.Lock()
        # Lifecycle caches: SINGLETON keeps one instance per spec name;
        # PER_AGENT keys on (name, agent_id).
        self._singleton_cache: dict[str, object] = {}
        self._per_agent_cache: dict[tuple[str, str], object] = {}

    # ── Registration ────────────────────────────────────────────

    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            existing = self._specs.get(spec.name)
            if existing is not None:
                if existing.description_hash == spec.description_hash and \
                   existing.capabilities == spec.capabilities and \
                   existing.tier == spec.tier:
                    return  # idempotent — same registration
                logger.warning(
                    "ToolRegistry: %r re-registered with different spec "
                    "(existing source=%s, new source=%s). Keeping the "
                    "first registration. This usually means a tool was "
                    "imported twice via different module paths.",
                    spec.name, existing.source_module, spec.source_module,
                )
                return
            self._specs[spec.name] = spec
            logger.debug(
                "ToolRegistry: registered %r [%s] capabilities=%s",
                spec.name, spec.tier.value, list(spec.capabilities),
            )

    def replace_spec(self, spec: ToolSpec) -> None:
        """Force-replace an existing spec (or insert if absent).

        Used by the Forge bridge to update a tool's tier when Forge
        graduates it (SHADOW → CANARY → ACTIVE). Bypasses register's
        "keep the first" semantics — caller assumes responsibility
        for verifying the replacement is intentional.

        Also drops any cached SINGLETON / PER_AGENT instance for the
        tool, since the new factory may have different runtime
        behavior (e.g. a SHADOW Forge tool that was promoted to
        ACTIVE no longer discards results).
        """
        with self._lock:
            self._specs[spec.name] = spec
            self._singleton_cache.pop(spec.name, None)
            for key in [k for k in self._per_agent_cache if k[0] == spec.name]:
                self._per_agent_cache.pop(key, None)
            logger.info(
                "ToolRegistry: replaced %r — now tier=%s capabilities=%s",
                spec.name, spec.tier.value, list(spec.capabilities),
            )

    def unregister(self, name: str) -> bool:
        """Remove a spec by name. Returns True if removed.

        Used when a Forge tool transitions to KILLED / DEPRECATED —
        it should disappear from agent discovery immediately.
        """
        with self._lock:
            if name not in self._specs:
                return False
            del self._specs[name]
            self._singleton_cache.pop(name, None)
            for key in [k for k in self._per_agent_cache if k[0] == name]:
                self._per_agent_cache.pop(key, None)
            logger.info("ToolRegistry: unregistered %r", name)
            return True

    # ── Read API ────────────────────────────────────────────────

    def get(self, name: str) -> ToolSpec | None:
        with self._lock:
            return self._specs.get(name)

    def all(self) -> list[ToolSpec]:
        with self._lock:
            return list(self._specs.values())

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._specs.keys())

    def by_capability(self, tag: str) -> list[ToolSpec]:
        """Tools that declare ``tag`` in their capabilities."""
        with self._lock:
            return [s for s in self._specs.values() if tag in s.capabilities]

    def by_tier(self, tier: Tier) -> list[ToolSpec]:
        with self._lock:
            return [s for s in self._specs.values() if s.tier is tier]

    def filter(
        self,
        *,
        capabilities: Iterable[str] | None = None,
        tier_at_most: Tier | None = None,
        workspace: str | None = None,
        loadable_only: bool = False,
    ) -> list[ToolSpec]:
        """Multi-criteria filter for tool_search.

        - capabilities: tools must declare at least one of these tags
          (OR semantics — searches "renders-pdf or renders-chart" return
          tools that do EITHER).
        - tier_at_most: tools strictly at-or-below this tier are kept.
          Tier order: SHADOW < CANARY < PRODUCTION < IMMUTABLE.
        - workspace: if set, tool's workspace_scope must include it
          (or include "*").
        - loadable_only: drop tools whose guard() currently returns False.
        """
        cap_set = set(capabilities) if capabilities else None
        tier_order = {Tier.SHADOW: 0, Tier.CANARY: 1, Tier.PRODUCTION: 2, Tier.IMMUTABLE: 3}
        max_rank = tier_order[tier_at_most] if tier_at_most else 99

        with self._lock:
            specs = list(self._specs.values())

        out: list[ToolSpec] = []
        for s in specs:
            if cap_set is not None and not (set(s.capabilities) & cap_set):
                continue
            if tier_order[s.tier] > max_rank:
                continue
            if workspace is not None and "*" not in s.workspace_scope and \
               workspace not in s.workspace_scope:
                continue
            if loadable_only and not s.is_loadable:
                continue
            out.append(s)
        return out

    # ── Instance creation (caches by lifecycle) ─────────────────

    def build_instance(self, name: str, *, agent_id: str = "default") -> object:
        """Construct (or fetch cached) tool instance for an agent.

        Caches according to lifecycle:
          - SINGLETON  → one per spec name, shared across agents
          - PER_AGENT  → one per (name, agent_id)
          - PER_CREW / PER_CALL → fresh every call (caller manages scope)

        Raises KeyError if the name is unknown, RuntimeError if the
        guard fails.
        """
        spec = self.get(name)
        if spec is None:
            raise KeyError(name)
        if not spec.is_loadable:
            raise RuntimeError(
                f"Tool {name!r} is registered but its guard() returned False — "
                "missing env config or unreachable dependency."
            )
        from app.tool_registry.types import Lifecycle  # local import

        if spec.lifecycle is Lifecycle.SINGLETON:
            with self._lock:
                cached = self._singleton_cache.get(name)
                if cached is not None:
                    return cached
                instance = self._call_factory(spec, agent_id=agent_id)
                self._singleton_cache[name] = instance
                return instance
        if spec.lifecycle is Lifecycle.PER_AGENT:
            key = (name, agent_id)
            with self._lock:
                cached = self._per_agent_cache.get(key)
                if cached is not None:
                    return cached
                instance = self._call_factory(spec, agent_id=agent_id)
                self._per_agent_cache[key] = instance
                return instance
        # PER_CREW / PER_CALL
        return self._call_factory(spec, agent_id=agent_id)

    def _call_factory(self, spec: ToolSpec, *, agent_id: str) -> object:
        """Call a factory that may or may not accept agent_id."""
        try:
            return spec.factory(agent_id=agent_id)  # type: ignore[call-arg]
        except TypeError:
            # Factory takes no kwargs (or a different signature) — call bare.
            return spec.factory()

    # ── Diagnostics ─────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Counts by tier and lifecycle, for /api/cp/tools."""
        with self._lock:
            specs = list(self._specs.values())
        out: dict[str, int] = {"total": len(specs)}
        for s in specs:
            out[f"tier:{s.tier.value}"] = out.get(f"tier:{s.tier.value}", 0) + 1
            out[f"lifecycle:{s.lifecycle.value}"] = (
                out.get(f"lifecycle:{s.lifecycle.value}", 0) + 1
            )
        # Capability coverage: how many tools per tag
        cap_count: dict[str, int] = {}
        for s in specs:
            for tag in s.capabilities:
                cap_count[tag] = cap_count.get(tag, 0) + 1
        out["unique_capabilities_used"] = len(cap_count)
        return out
