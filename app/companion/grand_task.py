"""Grand-task synthesis — propose a higher-order workspace goal.

Per the user's spec answer #5 (12 h cadence): every twelve hours the
Companion looks at a workspace's accumulated polished ideas and the
implicit themes across them, then proposes a single higher-order
"grand task" that captures what the workspace is actually about.

Mirrors the Narrative-Self chapter consolidator pattern at workspace
scope. The proposal is appended to the event log as
``GRAND_TASK_PROPOSED`` and surfaced to the user; on accept,
``GRAND_TASK_ACCEPTED`` is emitted and the workspace's seed_prompt is
rotated to the new grand task. On reject, ``GRAND_TASK_REJECTED`` is
recorded so the next synthesis cycle knows to try a different angle.

Synthesis is ONE cheap-tier LLM call per workspace per cycle. With the
12 h cadence and the per-workspace activation gate (no proposal when
fewer than ``MIN_IDEAS_FOR_SYNTHESIS`` polished ideas exist), the cost
is bounded to ~$0.0002/workspace/day.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from app.companion import config as _config
from app.companion import events as _events
from app.companion import idea_store as _idea_store

logger = logging.getLogger(__name__)

# Cadence guard — at most one proposal per workspace per CADENCE_S.
CADENCE_HOURS = 12
CADENCE_S = CADENCE_HOURS * 3600

# How many polished ideas a workspace needs before synthesis kicks in.
MIN_IDEAS_FOR_SYNTHESIS = 3

# How many recent polished ideas inform one synthesis call.
MAX_IDEAS_IN_PROMPT = 12

# Truncate each idea body to this many characters in the prompt.
IDEA_SNIPPET_CHARS = 300


@dataclass
class GrandTaskProposal:
    """One proposed grand task for a workspace."""
    proposal_id: str
    workspace_id: str
    text: str
    rationale: str = ""
    superseded_seed: str | None = None  # the seed_prompt at proposal time
    ts: float = field(default_factory=time.time)


# ── Synthesis ──────────────────────────────────────────────────────────────

def synthesize(workspace_id: str) -> GrandTaskProposal | None:
    """Propose a grand task for one workspace; emit GRAND_TASK_PROPOSED.

    Returns None when synthesis is skipped (cadence not yet elapsed,
    not enough polished ideas, LLM unavailable, parse failed).
    """
    cfg = _config.load(workspace_id)
    if cfg is None or not cfg.enabled:
        return None
    if _too_recent(workspace_id):
        return None
    polished = _gather_polished_ideas(workspace_id)
    if len(polished) < MIN_IDEAS_FOR_SYNTHESIS:
        return None

    prompt = _compose_prompt(cfg.seed_prompt, polished)
    try:
        raw = _invoke_synthesizer(prompt)
    except Exception as exc:
        logger.debug("companion.grand_task: LLM failed for %s: %s",
                     workspace_id, exc)
        return None
    text, rationale = _parse_proposal(raw)
    if not text:
        return None

    proposal_id = f"gt_{uuid.uuid4().hex[:12]}"
    proposal = GrandTaskProposal(
        proposal_id=proposal_id,
        workspace_id=workspace_id,
        text=text,
        rationale=rationale,
        superseded_seed=cfg.seed_prompt,
    )
    _emit_proposed(proposal)
    return proposal


def run_synthesis_for_all_workspaces() -> int:
    """Idle-job entry — walks every CP project and runs synthesis where due.

    Returns the number of proposals emitted this run.
    """
    try:
        rows = _list_projects()
    except Exception as exc:
        logger.debug("companion.grand_task: list_projects failed: %s", exc)
        return 0
    n = 0
    for row in rows:
        pid = row.get("id")
        if not pid:
            continue
        cfg_raw = (row.get("config_json") or {}).get("companion") or {}
        if cfg_raw.get("enabled") is False:
            continue
        try:
            if synthesize(pid) is not None:
                n += 1
        except Exception as exc:
            logger.warning("companion.grand_task: failed for %s: %s",
                           pid, exc)
    return n


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """Idle-scheduler tuple — appended in ``loop.get_idle_jobs()``.

    Registered as MEDIUM (one cheap LLM call per workspace; cadence
    guard caps frequency at 2/day).
    """
    from app.idle_scheduler import JobWeight
    return [("companion-grand-task", run_synthesis_for_all_workspaces,
             JobWeight.MEDIUM)]


# ── Acceptance / rejection ─────────────────────────────────────────────────

def accept(workspace_id: str, proposal_id: str) -> bool:
    """Accept a proposal — rotate seed_prompt and emit GRAND_TASK_ACCEPTED.

    Returns False when the proposal isn't found or save fails.
    """
    proposal = find_proposal(workspace_id, proposal_id)
    if proposal is None:
        return False
    cfg = _config.load(workspace_id) or _config.CompanionConfig()
    cfg.seed_prompt = proposal.text
    if not _config.save(workspace_id, cfg):
        return False
    _events.append(_events.Event(
        workspace_id=workspace_id,
        idea_id=proposal_id,
        type=_events.EventType.GRAND_TASK_ACCEPTED,
        ts=time.time(),
        payload={"proposal_id": proposal_id, "new_seed": proposal.text},
    ))
    return True


def reject(workspace_id: str, proposal_id: str, *,
           reason: str = "") -> bool:
    """Reject a proposal — record the decision so synthesis avoids similar."""
    proposal = find_proposal(workspace_id, proposal_id)
    if proposal is None:
        return False
    _events.append(_events.Event(
        workspace_id=workspace_id,
        idea_id=proposal_id,
        type=_events.EventType.GRAND_TASK_REJECTED,
        ts=time.time(),
        payload={"proposal_id": proposal_id, "reason": (reason or "")[:500]},
    ))
    return True


# ── Reads ──────────────────────────────────────────────────────────────────

def list_proposals(workspace_id: str, *,
                    limit: int = 20) -> list[GrandTaskProposal]:
    """Return recent proposals for a workspace, newest first."""
    out: list[GrandTaskProposal] = []
    for ev in _events.read_all(workspace_id):
        if ev.type != _events.EventType.GRAND_TASK_PROPOSED:
            continue
        payload = ev.payload or {}
        out.append(GrandTaskProposal(
            proposal_id=payload.get("proposal_id", ev.idea_id),
            workspace_id=workspace_id,
            text=payload.get("text", ""),
            rationale=payload.get("rationale", ""),
            superseded_seed=payload.get("superseded_seed"),
            ts=ev.ts,
        ))
    out.sort(key=lambda p: p.ts, reverse=True)
    return out[:limit]


def find_proposal(workspace_id: str,
                   proposal_id: str) -> GrandTaskProposal | None:
    for p in list_proposals(workspace_id, limit=200):
        if p.proposal_id == proposal_id:
            return p
    return None


# ── Cadence + context gathering ────────────────────────────────────────────

def _too_recent(workspace_id: str, *, now: float | None = None) -> bool:
    cutoff = (now if now is not None else time.time()) - CADENCE_S
    for ev in _events.read_all(workspace_id):
        if ev.type == _events.EventType.GRAND_TASK_PROPOSED and ev.ts >= cutoff:
            return True
    return False


def _gather_polished_ideas(
        workspace_id: str) -> list[_idea_store.IdeaRecord]:
    """Collect recent polished ideas (DOCUMENTED preferred, fall back to
    SURFACED, then CONVERGED) up to ``MAX_IDEAS_IN_PROMPT``."""
    ideas = _idea_store.find_by_workspace(workspace_id,
                                           limit=10_000)
    polished: list[_idea_store.IdeaRecord] = []
    state_priority = {
        _idea_store.IdeaState.DOCUMENTED: 0,
        _idea_store.IdeaState.SURFACED: 1,
        _idea_store.IdeaState.CONVERGED: 2,
    }
    for r in ideas:
        cur = _idea_store.current_state(workspace_id, r.idea_id) or r.state
        if cur in state_priority:
            polished.append(r)
    polished.sort(
        key=lambda r: (state_priority.get(
            _idea_store.current_state(workspace_id, r.idea_id) or r.state, 9),
                       -r.created_at)
    )
    return polished[:MAX_IDEAS_IN_PROMPT]


# ── Prompt + parse ─────────────────────────────────────────────────────────

def _compose_prompt(seed_prompt: str | None,
                     ideas: list[_idea_store.IdeaRecord]) -> str:
    seed = (seed_prompt or "(no current seed)").strip() or "(no current seed)"
    bullets = []
    for i, r in enumerate(ideas, 1):
        body = (r.text or "").strip().replace("\n", " ")
        if len(body) > IDEA_SNIPPET_CHARS:
            body = body[:IDEA_SNIPPET_CHARS - 3] + "..."
        bullets.append(f"{i}. (panel {r.panel_score:.2f}, novelty "
                       f"{r.novelty:.2f}) {body}")
    bullet_block = "\n".join(bullets) if bullets else "(no ideas yet)"
    return _PROMPT_TEMPLATE.format(seed=seed, bullets=bullet_block)


_PROMPT_TEMPLATE = """\
A workspace has been generating ideas around the seed: "{seed}"

Recent polished ideas the user has either kept or seen:
{bullets}

Distill these into ONE higher-order GRAND TASK for the workspace — the
implicit larger goal across the ideas. The grand task should be more
ambitious than the original seed, drawing on what the workspace has
actually accumulated.

Output exactly two lines:
GRAND_TASK: <single sentence, ≤ 200 chars>
RATIONALE: <one short sentence on why this captures the implicit goal>
"""


def _parse_proposal(raw: str) -> tuple[str, str]:
    import re
    text = raw or ""
    task_m = re.search(
        r"GRAND_TASK\s*[:\-]\s*(.+?)(?:\n|$)",
        text, flags=re.IGNORECASE)
    rat_m = re.search(
        r"RATIONALE\s*[:\-]\s*(.+?)(?:\n\n|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL)
    grand = (task_m.group(1).strip() if task_m else "")[:240]
    rationale = re.sub(
        r"\s+", " ", (rat_m.group(1) if rat_m else "").strip())[:300]
    return grand, rationale


def _invoke_synthesizer(prompt: str) -> str:
    """Indirection over the cheap-tier LLM call."""
    from app.llm_factory import create_specialist_llm
    llm = create_specialist_llm(max_tokens=200, role="commander")
    return str(llm.call(prompt))


# ── Event emission + project listing ───────────────────────────────────────

def _emit_proposed(proposal: GrandTaskProposal) -> None:
    try:
        _events.append(_events.Event(
            workspace_id=proposal.workspace_id,
            idea_id=proposal.proposal_id,
            type=_events.EventType.GRAND_TASK_PROPOSED,
            ts=proposal.ts,
            payload={
                "proposal_id": proposal.proposal_id,
                "text": proposal.text,
                "rationale": proposal.rationale,
                "superseded_seed": proposal.superseded_seed,
            },
        ))
    except Exception as exc:
        logger.debug(
            "companion.grand_task: PROPOSED event append failed: %s", exc)


def _list_projects() -> list[dict]:
    """Indirection over CP project listing — same seam as scheduler/ingest."""
    from app.control_plane.projects import get_projects
    return get_projects().list_all() or []
