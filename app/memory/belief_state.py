"""
belief_state.py — ProAgent-style belief state tracking.

Maintains explicit beliefs about each agent's current state, updated
after every observation.  This implements Theory of Mind / intention
inference from ProAgent (Zhang et al. AAAI 2024).

Beliefs are stored in the 'scope_beliefs' ChromaDB collection as JSON
documents with per-agent metadata for filtering.
"""

import json
import logging
from datetime import datetime, timezone
from app.memory.chromadb_manager import store, retrieve_with_metadata

logger = logging.getLogger(__name__)

BELIEFS_COLLECTION = "scope_beliefs"


def update_belief(
    agent_name: str,
    state: str,
    current_task: str = "",
    confidence: str = "medium",
    observations: list[str] = None,
    needs: list[str] = None,
    projected_next_task: str = "",
    intention_confidence: float = 0.0,
) -> None:
    """Store or update a belief about an agent's current state.

    Args:
        agent_name: The agent being observed (e.g. "researcher", "coder")
        state: One of "idle", "working", "blocked", "completed", "failed"
        current_task: What the agent is currently doing
        confidence: The agent's self-reported confidence level
        observations: Recent observations about this agent
        needs: What this agent needs from teammates
        projected_next_task: Predicted next task (Theory of Mind)
        intention_confidence: Confidence in the prediction (0.0-1.0)
    """
    belief = {
        "agent": agent_name,
        "state": state,
        "current_task": current_task[:300],
        "confidence": confidence,
        "observations": (observations or [])[-5:],  # Keep last 5
        "needs": (needs or [])[-5:],
        "projected_next_task": projected_next_task[:200],
        "intention_confidence": round(intention_confidence, 2),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    belief_text = json.dumps(belief)
    metadata = {
        "agent": agent_name,
        "state": state,
        "type": "belief_state",
        "ts": belief["last_updated"],
    }

    try:
        store(BELIEFS_COLLECTION, belief_text, metadata)
    except Exception:
        logger.warning(f"Failed to store belief for {agent_name}", exc_info=True)


def get_beliefs(agent_name: str = None) -> list[dict]:
    """Retrieve current beliefs about agent(s).

    If agent_name is provided, returns beliefs about that specific agent.
    Otherwise returns beliefs about all agents.
    """
    query = f"agent state {agent_name}" if agent_name else "agent state beliefs"
    items = retrieve_with_metadata(BELIEFS_COLLECTION, query, n=20)
    if not items:
        return []

    beliefs = []
    seen_agents = set()
    for item in items:
        try:
            belief = json.loads(item["document"])
            agent = belief.get("agent", "")

            # Filter by agent if specified
            if agent_name and agent != agent_name:
                continue

            # Only keep the most recent belief per agent
            if agent in seen_agents:
                continue
            seen_agents.add(agent)
            beliefs.append(belief)
        except (json.JSONDecodeError, KeyError):
            continue

    return beliefs


def get_team_state_summary() -> str:
    """Return a formatted summary of all agents' current states.

    Used by the Commander to understand team status before routing.
    """
    beliefs = get_beliefs()
    if not beliefs:
        return ""

    lines = ["TEAM STATE:"]
    for b in beliefs:
        agent = b.get("agent", "unknown")
        state = b.get("state", "unknown")
        task = b.get("current_task", "")
        confidence = b.get("confidence", "")
        needs = b.get("needs", [])

        line = f"  - {agent}: {state}"
        if task:
            line += f" | task: {task[:60]}"
        if confidence:
            line += f" | confidence: {confidence}"
        projected = b.get("projected_next_task", "")
        if projected:
            int_conf = b.get("intention_confidence", 0)
            line += f" | next: {projected[:40]} ({int_conf:.0%})"
        if needs:
            line += f" | needs: {', '.join(needs[:3])}"
        lines.append(line)

    return "\n".join(lines)


def infer_intentions(agent_name: str) -> dict:
    """Infer what an agent will likely do next based on task history patterns.

    Simple bigram model: common task transitions (research→coding, coding→writing).
    Returns {projected_next_task, intention_confidence}.
    """
    # Common task transition patterns (based on typical multi-agent workflows)
    TRANSITION_PATTERNS = {
        "research": {"coding": 0.4, "writing": 0.35, "research": 0.25},
        "coding": {"writing": 0.3, "research": 0.2, "coding": 0.5},
        "writing": {"research": 0.3, "writing": 0.5, "coding": 0.2},
        "media": {"writing": 0.4, "research": 0.3, "media": 0.3},
    }

    transitions = TRANSITION_PATTERNS.get(agent_name, {})
    if not transitions:
        return {"projected_next_task": "", "intention_confidence": 0.0}

    # Pick the most likely next task
    best_task = max(transitions, key=transitions.get)
    confidence = transitions[best_task]

    # Boost confidence if we have actual history
    try:
        from app.subia.self.agent_state import get_agent_stats
        stats = get_agent_stats(agent_name)
        if stats.get("tasks_completed", 0) > 10:
            confidence = min(1.0, confidence + 0.1)  # More data = more confident
    except Exception:
        pass

    return {"projected_next_task": best_task, "intention_confidence": confidence}


def revise_beliefs(observation: str, agent_name: str) -> None:
    """Store an observation that may change beliefs about an agent.

    This is called when one agent observes something about another
    agent's output or behavior.
    """
    metadata = {
        "type": "observation",
        "about": agent_name,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        store(BELIEFS_COLLECTION, observation[:500], metadata)
    except Exception:
        logger.warning(f"Failed to store observation about {agent_name}", exc_info=True)


# ── Stale "working" belief cleanup ─────────────────────────────────────
#
# Beliefs stored under state=="working" represent an active claim that a
# given agent is currently doing work.  When a crew crashes without the
# lifecycle envelope emitting its terminating transition, those claims
# become false and never self-correct — the dashboard's Theory-of-Mind
# view shows agents "working" for days when they've long since exited.
#
# Design constraint (explicit in the refactor brief): the cleanup MUST
# NOT mess up HOT-3's scoring semantics.  HOT-3 scores beliefs on two
# axes:
#
#   freshness   = fraction of beliefs whose ``last_updated`` is within
#                 the probe's cutoff (currently 1h).  This is supposed
#                 to measure "is the system's self-model actively
#                 maintained?" — so the cleanup MUST NOT write new
#                 ``last_updated`` timestamps, or it would inflate
#                 freshness without any real maintenance happening.
#
#   consistency = fraction of beliefs whose state is in the valid set
#                 (idle / working / completed / failed / blocked).
#                 Already 1.0 across the board today, so no concern.
#
# The only correct operation here is **delete** — removing a stale
# claim is honest (we no longer assert the agent is working).  We
# leave terminal-state beliefs ("completed", "failed", "idle",
# "blocked") alone even when they're old, because those accurately
# record the last known state of an agent and HOT-3's freshness factor
# naturally (correctly!) penalizes the whole fleet being idle.


def cleanup_stale_working_beliefs(max_age_hours: int = 6) -> int:
    """Delete 'working' beliefs whose ``last_updated`` is older than
    ``max_age_hours`` ago.

    Returns the number of deleted documents.  Non-fatal on error (the
    retrieval-side code already tolerates missing / partial data).

    Rationale for **delete** rather than **transition**:

    * Transitioning "working" → "idle" requires bumping ``last_updated``
      (otherwise the metadata is inconsistent with the new state).  The
      bump would artificially mark the belief as fresh, inflating
      HOT-3's freshness signal even though no agent actually did any
      work.  That would defeat the probe's purpose.

    * Deleting a stale "working" claim is semantically honest: we were
      wrong about the agent's state, and we no longer claim to know.
      Next time the agent runs, the lifecycle envelope will write a
      fresh, correct belief.

    The retention window defaults to 6 h — long enough that genuinely
    long-running crews (difficulty-9 research can take 15-30 min) are
    never caught, short enough that post-crash stale "working" state
    doesn't persist for days.
    """
    from datetime import timedelta
    from app.memory.chromadb_manager import _get_col

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

    try:
        col = _get_col(BELIEFS_COLLECTION)
        # ChromaDB's ``$lt`` operator on metadata requires a numeric
        # operand — it won't compare ISO-8601 strings directly
        # (``ValueError: Expected operand value to be an int or a float``).
        # So we do a two-step: fetch all rows with ``state='working'``,
        # parse the ``ts`` string in Python, and delete by id.  Cost is
        # small (the beliefs collection has tens, maybe hundreds of
        # entries, not millions) and the code stays portable across
        # Chroma versions.
        rows = col.get(where={"state": {"$eq": "working"}},
                       include=["metadatas"])
        ids = rows.get("ids") or []
        metas = rows.get("metadatas") or []
        stale_ids: list[str] = []
        for rid, meta in zip(ids, metas):
            meta = meta or {}
            ts = meta.get("ts") or ""
            # ``ts < cutoff`` in lexicographic order == chronological
            # order for ISO-8601 strings (which is how ``update_belief``
            # writes the metadata).
            if ts and ts < cutoff:
                stale_ids.append(rid)
        if not stale_ids:
            return 0
        col.delete(ids=stale_ids)
        logger.info(
            "belief_state: cleanup removed %d stale 'working' beliefs "
            "(older than %dh)", len(stale_ids), max_age_hours,
        )
        return len(stale_ids)
    except Exception:
        logger.debug("belief_state: stale-cleanup failed (non-fatal)",
                     exc_info=True)
        return 0


def _safe_count(col) -> int:
    """Best-effort row count for a ChromaDB collection.  Used purely
    for the cleanup log line — failures are non-fatal."""
    try:
        return col.count()
    except Exception:
        return 0
