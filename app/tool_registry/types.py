"""Tier, Lifecycle, and ToolSpec — the registry's data model.

These types are deliberately narrow: a Tier and Lifecycle enum, plus
a frozen dataclass that holds everything we know about a single tool.
The decorator builds a ToolSpec; the registry stores it; persistence
and drift use it. No business logic lives here.
"""
from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable


class Tier(str, enum.Enum):
    """Registry tier — mirrors the auto_deployer tier hierarchy.

    Discovery filters by tier: a PRODUCTION-tier crew never sees
    SHADOW tools; SHADOW crews see all tiers (so they can validate
    promotions before they affect prod).
    """

    SHADOW = "shadow"          # Newly forged, not yet validated. Hidden by default.
    CANARY = "canary"          # Validated, in soak. Visible to canary crews.
    PRODUCTION = "production"  # Live and trusted.
    IMMUTABLE = "immutable"    # Pinned. Cannot be replaced or unloaded.


class Lifecycle(str, enum.Enum):
    """How tool instances are scoped.

    Most tools are SINGLETON — one instance per process, cached by
    the registry. Tools that need per-agent state (e.g. memory tools
    filtered by collection) are PER_AGENT — one instance per agent_id.
    PER_CREW is for transient state that should not bleed across
    crew runs. PER_CALL is rare; instantiate fresh on every invocation.
    """

    SINGLETON = "singleton"
    PER_AGENT = "per_agent"
    PER_CREW = "per_crew"
    PER_CALL = "per_call"


def _hash_description(description: str) -> str:
    """Stable 12-hex-char hash for drift detection."""
    return hashlib.sha256(description.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ToolSpec:
    """Everything the registry knows about one tool.

    The factory is the *only* way to obtain an instance — agents
    don't construct tools directly anymore. `args_schema` is the
    pydantic model that defines the tool's call signature; the
    description is what surfaces to the LLM after loading.

    `guard` is a deferred check: returns True iff the tool's runtime
    environment is configured (env vars set, services reachable, etc).
    Tools whose guard returns False are visible in the registry but
    cannot be loaded.

    `workspace_scope` is a tuple of workspace IDs the tool is allowed
    in, or ``("*",)`` for all workspaces. The discovery layer filters
    on this per request.
    """

    name: str
    capabilities: tuple[str, ...]
    tier: Tier
    lifecycle: Lifecycle
    description: str
    args_schema: type | None
    factory: Callable[..., Any]
    guard: Callable[[], bool]
    workspace_scope: tuple[str, ...]
    source_module: str

    # Computed at decoration time; used by drift detection.
    description_hash: str = field(default="")

    def __post_init__(self) -> None:
        # frozen dataclass — bypass to set the computed field.
        if not self.description_hash:
            object.__setattr__(self, "description_hash", _hash_description(self.description))

    @property
    def is_loadable(self) -> bool:
        """True iff guard() passes — tool can actually be instantiated."""
        try:
            return bool(self.guard())
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view for /api/cp/tools and persistence."""
        return {
            "name": self.name,
            "capabilities": list(self.capabilities),
            "tier": self.tier.value,
            "lifecycle": self.lifecycle.value,
            "description": self.description,
            "description_hash": self.description_hash,
            "workspace_scope": list(self.workspace_scope),
            "source_module": self.source_module,
            "is_loadable": self.is_loadable,
        }
