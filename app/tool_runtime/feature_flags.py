"""feature_flags.py — per-agent LoadableAgent rollout control.

Phase 4 introduces production agent migrations. Each migrated agent
ships with a per-agent env flag PLUS respects the global flag from
Phase 2:

  * ``LOADABLE_AGENT_EXPERIMENTAL=1``  — master switch; turns ON for
    all migrated agents that don't have an explicit per-agent flag.
  * ``LOADABLE_<AGENT>=1`` / ``=0``    — per-agent override. When set
    explicitly, it overrides the master (allowing operators to
    selectively enable / disable individual agents).

Resolution order (most specific wins):

  1. Per-agent env var ``LOADABLE_<AGENT_UPPER>`` set to ``"1"`` →
     loadable path.
  2. Per-agent env var set to ``"0"`` (or anything else) → legacy
     path, even if the master is on.
  3. No per-agent var → master flag decides.
  4. Both unset → legacy path (default).

This pattern lets operators:

  * Run an A/B comparison: ``LOADABLE_RESEARCHER=1`` while keeping
    coder/writer on stock.
  * Roll back one agent while keeping others migrated:
    ``LOADABLE_AGENT_EXPERIMENTAL=1 LOADABLE_RESEARCHER=0``.
  * Default-off everywhere: leave both unset (the production state
    until each migration's parity panel passes).

The helper is the single source of truth for this dispatch — every
migrated agent's factory calls ``is_loadable_for("researcher")``
(etc.) instead of duplicating env-var parsing.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


_MASTER_VAR = "LOADABLE_AGENT_EXPERIMENTAL"


def is_loadable_for(agent_name: str) -> bool:
    """Should ``agent_name`` use the LoadableAgent path right now?

    Args:
        agent_name: Lowercase agent name, e.g. "researcher", "coder",
            "introspector". Case-insensitive — the function uppercases
            internally to find the env var.

    Returns:
        True iff the agent should construct as LoadableAgent.
    """
    per_agent_var = f"LOADABLE_{agent_name.upper()}"
    explicit = os.environ.get(per_agent_var)
    if explicit is not None:
        # Operator set the per-agent flag explicitly — that wins.
        return explicit.strip() == "1"
    # Fall through to the master.
    return os.environ.get(_MASTER_VAR, "").strip() == "1"


def is_master_on() -> bool:
    """True iff the master flag is set (regardless of per-agent overrides)."""
    return os.environ.get(_MASTER_VAR, "").strip() == "1"


def explicit_flag_for(agent_name: str) -> str | None:
    """Return the operator-set per-agent flag value, or None if unset.

    Useful for diagnostics — the React /cp/tools page can show which
    agents have explicit overrides vs. inheriting the master.
    """
    return os.environ.get(f"LOADABLE_{agent_name.upper()}")
