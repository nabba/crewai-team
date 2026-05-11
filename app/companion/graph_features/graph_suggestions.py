"""L4.4 — Graph-driven suggestions (PRESCRIPTIVE).

PROGRAM §42 (2026-05-11) — Q4.2 Level 4.4.

System emits suggestions based on graph topology in 3 operator-
configurable categories:

  * ``cluster_dormancy``    — cluster shows ≥30d collective inactivity
  * ``bridge_maintenance``  — bridge edge hasn't seen recent activity
  * ``weak_tie_dormant``    — high-centrality cut-vertex with falling activity

CRITICAL: this sub-feature requires the SECOND typed-phrase gate
``ENABLE GRAPH-DRIVEN SUGGESTIONS`` distinct from the L4 master
typed-phrase. Two-step gating.

Rate limit: shares the L3 per-briefing cap (≤3 across L3 + L4.4).
The combined cap is enforced in app.companion.person_suggestions.

All suggestions phrased as QUESTIONS — never directives.

Per-category opt-in. Operator picks which of the 3 they want.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


_CLUSTER_DORMANCY_DAYS = 30
_BRIDGE_DORMANCY_DAYS = 45
_WEAK_TIE_DORMANCY_DAYS = 30
# Q4.2.1#3 — skip graph-derived nudges when the source-of-truth JSON
# is older than this. If the graph-features idle job crashed or got
# stuck, its output can rot for weeks; nudging against weeks-old
# topology would surface false structural roles. 72h is generous —
# the idle job runs every 12h.
_STRUCTURAL_FRESHNESS_HOURS = 72


def _is_fresh(data: dict, max_hours: int = _STRUCTURAL_FRESHNESS_HOURS) -> bool:
    """Check whether a graph-features JSON payload was generated
    recently. Returns False on missing/unparseable timestamp — caller
    treats stale-or-unknown as a skip."""
    ts_str = (data or {}).get("generated_at") or ""
    if not ts_str:
        return False
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return False
    age = datetime.now(timezone.utc) - ts
    return age < timedelta(hours=max_hours)


def _enabled() -> bool:
    try:
        from app.runtime_settings import (
            get_person_correlation_enabled,
            get_person_correlation_social_graph_enabled,
            get_graph_suggestions_enabled,
        )
        return (
            get_person_correlation_enabled()
            and get_person_correlation_social_graph_enabled()
            and get_graph_suggestions_enabled()
        )
    except Exception:
        return False


def _cluster_dormancy_enabled() -> bool:
    if not _enabled():
        return False
    try:
        from app.runtime_settings import get_graph_suggestions_cluster_dormancy_enabled
        return get_graph_suggestions_cluster_dormancy_enabled()
    except Exception:
        return False


def _bridge_maintenance_enabled() -> bool:
    if not _enabled():
        return False
    try:
        from app.runtime_settings import get_graph_suggestions_bridge_maintenance_enabled
        return get_graph_suggestions_bridge_maintenance_enabled()
    except Exception:
        return False


def _weak_tie_dormant_enabled() -> bool:
    if not _enabled():
        return False
    try:
        from app.runtime_settings import get_graph_suggestions_weak_tie_enabled
        return get_graph_suggestions_weak_tie_enabled()
    except Exception:
        return False


# ── Generators ───────────────────────────────────────────────────────────


def _generate_cluster_dormancy(profile_people: list[dict]) -> list:
    """When a whole community has gone quiet, surface one nudge for
    the cluster (not per-person)."""
    if not _cluster_dormancy_enabled():
        return []
    try:
        from app.companion.graph_features.communities import (
            _default_communities_path,
        )
        import json
        p = _default_communities_path()
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        clusters = data.get("clusters") or []
    except Exception:
        return []
    # Q4.2.1#3 — refuse to fire nudges from stale topology.
    if not _is_fresh(data):
        logger.debug("graph_suggestions: cluster_dormancy skipped — stale")
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_CLUSTER_DORMANCY_DAYS)

    # Index people by id for last_seen lookup.
    by_id = {p.get("person_id"): p for p in profile_people if p.get("person_id")}

    # Import the dataclass from person_suggestions
    from app.companion.person_suggestions import PersonSuggestion

    out = []
    for cluster in clusters:
        members = cluster.get("members") or []
        if not members:
            continue
        # Latest last_seen across cluster members.
        latest = None
        for m in members:
            pp = by_id.get(m)
            if not pp:
                continue
            ls = pp.get("last_seen") or ""
            if not ls:
                continue
            try:
                ls_dt = datetime.fromisoformat(ls.replace("Z", "+00:00"))
            except ValueError:
                continue
            if latest is None or ls_dt > latest:
                latest = ls_dt
        if latest is None or latest > cutoff:
            continue
        days = int((now - latest).total_seconds() / 86400.0)
        cluster_id = cluster.get("id", "?")
        size = cluster.get("size", len(members))
        # Use the FIRST member as the suggestion's person_id (so the
        # dedup-by-person logic in person_suggestions doesn't double-fire
        # for this cluster).
        anchor = members[0]
        out.append(PersonSuggestion(
            category="cluster_dormancy",
            person_id=anchor,
            display_name=f"cluster {cluster_id} ({size} people)",
            text=(
                f"Cluster {cluster_id} ({size} people) has gone quiet — "
                f"last activity {days}d ago. Touchpoint?"
            ),
            detected_at=now.isoformat(),
        ))
    return out


def _generate_bridge_maintenance(profile_people: list[dict]) -> list:
    """When a bridge edge or cut-vertex hasn't been touched in 45d+,
    surface a maintenance nudge."""
    if not _bridge_maintenance_enabled():
        return []
    try:
        from app.companion.graph_features.bridges import (
            _default_structural_path,
        )
        import json
        p = _default_structural_path()
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    # Q4.2.1#3 — stale structural data → no nudges.
    if not _is_fresh(data):
        logger.debug("graph_suggestions: bridge_maintenance skipped — stale")
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_BRIDGE_DORMANCY_DAYS)
    by_id = {p.get("person_id"): p for p in profile_people if p.get("person_id")}

    from app.companion.person_suggestions import PersonSuggestion

    out = []
    seen_pids: set[str] = set()
    cut_vertices = data.get("cut_vertices") or []
    for pid in cut_vertices:
        if pid in seen_pids:
            continue
        pp = by_id.get(pid)
        if not pp:
            continue
        ls = pp.get("last_seen") or ""
        try:
            ls_dt = datetime.fromisoformat(ls.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ls_dt > cutoff:
            continue
        days = int((now - ls_dt).total_seconds() / 86400.0)
        display = (pp.get("display_names") or [""])[0] or pid
        out.append(PersonSuggestion(
            category="bridge_maintenance",
            person_id=pid,
            display_name=display,
            text=(
                f"{display} is a bridge between graph clusters — last "
                f"seen {days}d ago. Reconnect?"
            ),
            detected_at=now.isoformat(),
        ))
        seen_pids.add(pid)
    return out


def _generate_weak_tie_dormant(profile_people: list[dict]) -> list:
    """When a high-centrality cut-vertex has falling recent activity,
    flag as 'weak tie dormant'. Distinct from bridge maintenance which
    triggers on absolute time-since-last-seen; this triggers on
    centrality drop."""
    if not _weak_tie_dormant_enabled():
        return []
    # First-ship heuristic: centrality-drop detection requires history
    # we don't yet collect. Use a simpler proxy: cut-vertex AND
    # last_seen 30d+ but <45d (so it's not yet a bridge_maintenance
    # nudge). This intentionally produces few results in v1.
    try:
        from app.companion.graph_features.bridges import (
            _default_structural_path,
        )
        import json
        p = _default_structural_path()
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    # Q4.2.1#3 — stale structural data → no nudges.
    if not _is_fresh(data):
        logger.debug("graph_suggestions: weak_tie_dormant skipped — stale")
        return []

    now = datetime.now(timezone.utc)
    by_id = {p.get("person_id"): p for p in profile_people if p.get("person_id")}

    from app.companion.person_suggestions import PersonSuggestion

    out = []
    for pid in data.get("cut_vertices") or []:
        pp = by_id.get(pid)
        if not pp:
            continue
        ls = pp.get("last_seen") or ""
        try:
            ls_dt = datetime.fromisoformat(ls.replace("Z", "+00:00"))
        except ValueError:
            continue
        days = int((now - ls_dt).total_seconds() / 86400.0)
        if not (_WEAK_TIE_DORMANCY_DAYS <= days < _BRIDGE_DORMANCY_DAYS):
            continue
        display = (pp.get("display_names") or [""])[0] or pid
        out.append(PersonSuggestion(
            category="weak_tie_dormant",
            person_id=pid,
            display_name=display,
            text=(
                f"{display} connects clusters but recent appearances "
                f"dropped ({days}d since last seen). Check in?"
            ),
            detected_at=now.isoformat(),
        ))
    return out


# ── Public ───────────────────────────────────────────────────────────────


def generate_graph_suggestions(profile_people: list[dict]) -> list:
    """Aggregate across all enabled categories. Returns
    list[PersonSuggestion] — rate-limited downstream in person_suggestions."""
    if not _enabled():
        return []
    out = []
    out.extend(_generate_cluster_dormancy(profile_people))
    out.extend(_generate_bridge_maintenance(profile_people))
    out.extend(_generate_weak_tie_dormant(profile_people))
    return out
