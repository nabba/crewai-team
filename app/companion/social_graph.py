"""Social graph — Level 4 of person-correlation.

PROGRAM §42 (2026-05-11) — Q4.2 Level 4 base layer.

Tracks co-appearance edges between people. Undirected weighted graph
stored at ``workspace/companion/social_graph.json``. Two people who
appear in the same calendar event / email thread / ticket comments
share an edge with weight = co-appearance count.

Master switch: ``person_correlation_social_graph_enabled`` (default
OFF). Enabling requires typed-phrase ``ENABLE SOCIAL GRAPH``.

Edge decay: 3-month half-life (4× faster than profile decay). Old
relationships fade unless reinforced.

The graph file is in the DR secret-denylist (substring "social_graph")
so it's NEVER exported by default. Stays on the host.

Graph features (shortest path, communities, bridges, suggestions)
live in ``app.companion.graph_features.*`` and gate independently
on their own sub-toggles. This module just maintains the data.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_EDGE_HALF_LIFE_DAYS = 90.0       # 3 months — faster decay than profiles
_RUN_CADENCE_S = 12 * 3600

_lock = threading.RLock()  # reentrant: composite ops may nest helpers


def _default_graph_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "social_graph.json"
    except Exception:
        return Path("/app/workspace/companion/social_graph.json")


def _default_pair_mutes_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "social_graph_pair_mutes.json"
    except Exception:
        return Path("/app/workspace/companion/social_graph_pair_mutes.json")


def _default_query_log() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "social_graph_query_log.jsonl"
    except Exception:
        return Path("/app/workspace/companion/social_graph_query_log.jsonl")


def _enabled() -> bool:
    try:
        from app.runtime_settings import (
            get_person_correlation_enabled,
            get_person_correlation_social_graph_enabled,
        )
        return (
            get_person_correlation_enabled()
            and get_person_correlation_social_graph_enabled()
        )
    except Exception:
        return False


# ── Pair mutes (operator can hide edges) ─────────────────────────────────


def _pair_key(a: str, b: str) -> str:
    """Canonical undirected pair key."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    return "||".join(sorted([a, b]))


def _load_pair_mutes() -> set[str]:
    p = _default_pair_mutes_path()
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(data.get("muted") or [])
    except (OSError, json.JSONDecodeError):
        return set()


def mute_pair(a: str, b: str) -> bool:
    key = _pair_key(a, b)
    p = _default_pair_mutes_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    muted = _load_pair_mutes()
    if key in muted:
        return False
    muted.add(key)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"muted": sorted(muted)}, indent=2), encoding="utf-8")
    tmp.replace(p)
    return True


# ── Path-eligibility opt-out (per-person, set by L4.1 commands) ──────────


def _default_path_opt_outs_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "social_graph_path_optouts.json"
    except Exception:
        return Path("/app/workspace/companion/social_graph_path_optouts.json")


def load_path_opt_outs() -> set[str]:
    """People who opted out of being intermediate hops in path queries."""
    p = _default_path_opt_outs_path()
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(str(x).lower() for x in (data.get("opted_out") or []))
    except (OSError, json.JSONDecodeError):
        return set()


def opt_out_of_paths(person_id: str) -> bool:
    pid = (person_id or "").strip().lower()
    if not pid:
        return False
    p = _default_path_opt_outs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    opted = load_path_opt_outs()
    if pid in opted:
        return False
    opted.add(pid)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"opted_out": sorted(opted)}, indent=2), encoding="utf-8")
    tmp.replace(p)
    return True


# ── Graph storage ────────────────────────────────────────────────────────


@dataclass
class GraphEdge:
    a: str            # canonical (sorted) endpoint
    b: str
    weight: float     # decayed co-appearance count
    last_touched: str  # ISO-8601

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_graph() -> dict[str, GraphEdge]:
    """Return ``{pair_key: GraphEdge}``."""
    p = _default_graph_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, GraphEdge] = {}
    for key, row in (data.get("edges") or {}).items():
        if isinstance(row, dict):
            try:
                out[key] = GraphEdge(
                    a=str(row.get("a") or ""),
                    b=str(row.get("b") or ""),
                    weight=float(row.get("weight") or 0.0),
                    last_touched=str(row.get("last_touched") or ""),
                )
            except (TypeError, ValueError):
                continue
    return out


def _save_graph(edges: dict[str, GraphEdge]) -> None:
    p = _default_graph_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "edges": {k: e.to_dict() for k, e in edges.items()},
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def forget_graph() -> int:
    """Delete the graph file. Returns edge count deleted."""
    with _lock:
        edges = _load_graph()
        n = len(edges)
        _save_graph({})
    return n


# ── Decay + recompute ────────────────────────────────────────────────────


def _decay_weight(weight: float, age_days: float) -> float:
    if age_days <= 0:
        return weight
    return weight * math.exp(-math.log(2) * age_days / _EDGE_HALF_LIFE_DAYS)


def _gather_calendar_pairs() -> list[set[str]]:
    """Each calendar event yields one set of attendee emails (co-appearance)."""
    try:
        from app.tools.gcal_tools import _list_events
    except Exception:
        return []
    try:
        events = _list_events(max_results=50)
    except Exception:
        return []
    out: list[set[str]] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        attendees = ev.get("attendees") or []
        emails = set()
        for a in attendees:
            if isinstance(a, dict):
                em = str(a.get("email") or "").strip().lower()
            else:
                em = str(a).strip().lower()
            if em:
                emails.add(em)
        if len(emails) >= 2:
            out.append(emails)
    return out


def compile_graph() -> dict[str, Any]:
    """Update edges from current data sources. Apply decay. Persist."""
    if not _enabled():
        return {"ok": False, "skipped": True, "reason": "social_graph_enabled=False"}

    with _lock:
        edges = _load_graph()
        pair_muted = _load_pair_mutes()
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # Step 1: ingest new co-appearances from calendar.
        new_pairs = 0
        for emails in _gather_calendar_pairs():
            for a, b in combinations(sorted(emails), 2):
                key = _pair_key(a, b)
                if key in pair_muted:
                    continue
                edge = edges.get(key)
                if edge is None:
                    edges[key] = GraphEdge(
                        a=a, b=b, weight=1.0, last_touched=now_iso,
                    )
                    new_pairs += 1
                else:
                    edge.weight += 1.0
                    edge.last_touched = now_iso

        # Step 2: apply decay to all edges.
        decayed_count = 0
        dropped = 0
        for key in list(edges.keys()):
            e = edges[key]
            try:
                touched = datetime.fromisoformat(
                    e.last_touched.replace("Z", "+00:00")
                )
            except ValueError:
                continue
            age_days = max(0.0, (now - touched).total_seconds() / 86400.0)
            new_weight = _decay_weight(e.weight, age_days)
            if new_weight < 0.1:
                del edges[key]
                dropped += 1
            else:
                e.weight = round(new_weight, 4)
                decayed_count += 1

        _save_graph(edges)

    return {
        "ok": True,
        "edges_total": len(edges),
        "new_pairs": new_pairs,
        "decayed": decayed_count,
        "dropped_low_weight": dropped,
    }


def current_graph() -> dict[str, Any]:
    """Read the graph for the React viz. Filters pair-muted edges."""
    if not _enabled():
        return {"edges": [], "enabled": False}
    edges = _load_graph()
    pair_muted = _load_pair_mutes()
    out_edges: list[dict[str, Any]] = []
    nodes: set[str] = set()
    for key, e in edges.items():
        if key in pair_muted:
            continue
        out_edges.append(e.to_dict())
        nodes.add(e.a)
        nodes.add(e.b)
    return {
        "edges": out_edges,
        "nodes": sorted(nodes),
        "enabled": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def adjacency() -> dict[str, dict[str, float]]:
    """Internal helper for graph_features: returns
    ``{node: {neighbor: weight}}`` excluding pair-muted edges."""
    if not _enabled():
        return {}
    edges = _load_graph()
    pair_muted = _load_pair_mutes()
    adj: dict[str, dict[str, float]] = {}
    for key, e in edges.items():
        if key in pair_muted:
            continue
        adj.setdefault(e.a, {})[e.b] = e.weight
        adj.setdefault(e.b, {})[e.a] = e.weight
    return adj


def log_query(query_type: str, payload: dict[str, Any]) -> None:
    """Append a query record to the operator-visible log. Used by
    shortest-path so the operator can review what they asked the
    graph to do."""
    try:
        from app.utils.jsonl_retention import append_with_cap
        path = _default_query_log()
        append_with_cap(
            path,
            json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": query_type,
                "payload": payload,
            }, sort_keys=True),
            max_lines=2000,
        )
    except Exception:
        logger.debug("social_graph: query log failed", exc_info=True)


# ── Idle-job entry ───────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One pass — cadence-guarded. Master-switch-gated."""
    if not _enabled():
        return {"ok": True, "skipped": True}
    try:
        from app.healing.handlers._common import read_state_json, write_state_json
    except Exception:
        return {"ok": False}
    import time as _time
    state = read_state_json("social_graph.json", {"last_run_at": 0.0})
    now = _time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return {"ok": True, "skipped": True, "reason": "cadence"}
    state["last_run_at"] = now
    summary = compile_graph()
    state["last_summary"] = summary
    write_state_json("social_graph.json", state)
    return summary
