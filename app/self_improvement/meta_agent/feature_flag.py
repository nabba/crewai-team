"""
app.self_improvement.meta_agent.feature_flag — opt-in dispatch control.

Three layers, each overriding the next:

    1. Per-crew env var ``META_AGENT_<CREW>=1 / =0``  (ops override)
    2. Master env var ``META_AGENT=1``                (ops master)
    3. Persisted settings JSON (dashboard Org Chart)  (everyday toggle)
    4. Default: OFF

The env layer mirrors ``app.tool_runtime.feature_flags`` and is reserved
for ops scenarios (emergency disable, A/B testing without a deploy).
The JSON layer mirrors ``app.crews.delegation_settings`` and is the
day-to-day surface — toggled from the React Org Chart.

Examples:
    META_AGENT_CODING=0  → coding crew is OFF regardless of dashboard
    META_AGENT=1         → all crews ON regardless of dashboard
    (env unset)          → dashboard JSON decides per-crew

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


_MASTER_VAR = "META_AGENT"


def is_meta_agent_enabled(crew_name: str) -> bool:
    """Should ``crew_name`` use the meta-agent recipe path?

    Walks the resolution order documented in the module docstring.
    Never raises; on any error reading the JSON layer it falls through
    to the default (OFF).

    Args:
        crew_name: Lowercase crew name, e.g. "coding", "research",
            "writing". Case-insensitive — the function uppercases
            internally for env-var lookup and lowercases for JSON.
    """
    crew_norm = crew_name.lower().strip()

    # 1. Per-crew env var (ops override)
    per_crew = f"META_AGENT_{crew_norm.upper()}"
    explicit = os.environ.get(per_crew)
    if explicit is not None:
        return explicit.strip() == "1"

    # 2. Master env var (ops master)
    if os.environ.get(_MASTER_VAR, "").strip() == "1":
        return True

    # 3. Persisted JSON (dashboard toggle)
    try:
        from app.self_improvement.meta_agent.meta_agent_settings import is_enabled
        return is_enabled(crew_norm)
    except Exception:
        logger.debug(
            "meta_agent.feature_flag: settings layer unavailable; "
            "defaulting to OFF",
            exc_info=True,
        )
        return False


def is_master_on() -> bool:
    """True iff the master META_AGENT env flag is set.

    Useful for diagnostics — the React Org Chart shows a banner when
    the master is on so the operator knows per-crew toggles are being
    overridden.
    """
    return os.environ.get(_MASTER_VAR, "").strip() == "1"


def explicit_flag_for(crew_name: str) -> str | None:
    """Return the operator-set per-crew env flag value, or None if unset."""
    return os.environ.get(f"META_AGENT_{crew_name.upper()}")
