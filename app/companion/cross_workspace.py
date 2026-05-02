"""Cross-workspace transfer — abstract kernels propose to relevant peers.

Implements the user's hybrid model from spec answer #1: workspaces stay
focused on their own topics (Estonian forests doesn't contemplate
KaiCart e-commerce), but ABSTRACT/STRUCTURAL insights — feedback-loop
shape, attractor topology, governance pattern — can flow between
workspaces. Two gates protect focus discipline:

  1. **Sanitiser gate** — only kernels at ``TransferScope.GLOBAL_META``
     (no project nouns, no domain-specific markers) are propagated.
     Re-uses the existing ``app.transfer_memory.sanitizer.check`` so
     the cross-workspace path inherits the operator-reviewed denylists.

  2. **Relevance gate** — for each candidate target workspace, the
     kernel's embedding must clear ``CROSS_WORKSPACE_RELEVANCE_THRESHOLD``
     against the target's seed_prompt. KaiCart's "checkout UX" insight
     fails gate 2 against forests; a "delayed-reward feedback loop"
     insight passes both because it's structural AND embeds close to
     ecological dynamics.

Cross-pollinations are NEVER auto-injected. They land as
``CROSS_WORKSPACE_INBOX`` events on the target workspace's event log;
the user accepts (becomes context for the next N cycles) or dismisses.

Idle integration: ``run_propagation_for_all_workspaces`` walks every
active workspace daily as a LIGHT idle job. Per-source-idea cooldown
via ``CROSS_WORKSPACE_INBOX`` payload check prevents the same kernel
from being re-proposed to the same target on subsequent runs.
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

CROSS_WORKSPACE_RELEVANCE_THRESHOLD = 0.75
MAX_IDEAS_TO_SCAN = 30
MAX_PROPAGATIONS_PER_RUN = 5


@dataclass
class CrossWorkspaceProposal:
    """A kernel proposed to one target workspace from a source workspace."""
    kernel_id: str
    target_workspace_id: str
    source_workspace_id: str
    source_idea_id: str
    text: str
    relevance_score: float
    ts: float = field(default_factory=time.time)


# ── Propagation entry points ───────────────────────────────────────────────

def propagate_eligible(source_workspace_id: str) -> int:
    """Walk this workspace's polished ideas; propose to relevant peers.

    Returns the count of CROSS_WORKSPACE_INBOX events emitted this run
    (across all target workspaces). Bounded by ``MAX_PROPAGATIONS_PER_RUN``.
    """
    src_cfg = _config.load(source_workspace_id)
    if src_cfg is None or not src_cfg.enabled:
        return 0
    threshold = float(src_cfg.transferability_threshold)
    ideas = _idea_store.find_by_workspace(
        source_workspace_id, limit=MAX_IDEAS_TO_SCAN)
    eligible = [i for i in ideas
                if (i.text or "").strip()
                and float(i.transferability) >= threshold]
    if not eligible:
        return 0

    targets = _list_other_workspaces(source_workspace_id)
    if not targets:
        return 0

    proposals_made = 0
    for idea in eligible:
        if proposals_made >= MAX_PROPAGATIONS_PER_RUN:
            break
        if not _passes_sanitiser(idea.text):
            continue
        for target in targets:
            if proposals_made >= MAX_PROPAGATIONS_PER_RUN:
                break
            target_id = target.get("id")
            target_cfg_raw = (target.get("config_json") or {}).get(
                "companion") or {}
            target_seed = (target_cfg_raw.get("seed_prompt") or "").strip()
            if not target_id or not target_seed:
                continue
            if _already_proposed(target_id, idea.idea_id):
                continue
            relevance = _compute_relevance(idea.text, target_seed)
            if relevance < CROSS_WORKSPACE_RELEVANCE_THRESHOLD:
                continue
            _emit_inbox(
                target_workspace_id=target_id,
                source_workspace_id=source_workspace_id,
                source_idea=idea,
                relevance=relevance,
            )
            proposals_made += 1
    return proposals_made


def run_propagation_for_all_workspaces() -> int:
    """Idle-job entry — runs propagation for every active workspace."""
    try:
        rows = _list_projects()
    except Exception as exc:
        logger.debug("companion.cross_workspace: list_projects failed: %s",
                     exc)
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
            n += propagate_eligible(pid)
        except Exception as exc:
            logger.warning(
                "companion.cross_workspace: propagation failed for %s: %s",
                pid, exc)
    return n


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """Idle-scheduler tuple — appended in ``loop.get_idle_jobs()``."""
    from app.idle_scheduler import JobWeight
    return [("companion-xworkspace", run_propagation_for_all_workspaces,
             JobWeight.LIGHT)]


# ── Inbox + decisions ──────────────────────────────────────────────────────

def inbox(workspace_id: str) -> list[CrossWorkspaceProposal]:
    """List undecided cross-workspace proposals for one workspace."""
    decided: set[str] = set()
    proposals: list[tuple[float, CrossWorkspaceProposal]] = []
    for ev in _events.read_all(workspace_id):
        payload = ev.payload or {}
        kid = payload.get("kernel_id") or ev.idea_id
        if ev.type in (_events.EventType.CROSS_WORKSPACE_ACCEPTED,
                        _events.EventType.CROSS_WORKSPACE_DISMISSED):
            decided.add(kid)
        elif ev.type == _events.EventType.CROSS_WORKSPACE_INBOX:
            proposals.append((ev.ts, CrossWorkspaceProposal(
                kernel_id=kid,
                target_workspace_id=workspace_id,
                source_workspace_id=payload.get("source_workspace_id", ""),
                source_idea_id=payload.get("source_idea_id", ""),
                text=payload.get("text", ""),
                relevance_score=float(payload.get("relevance_score", 0.0)),
                ts=ev.ts,
            )))
    proposals.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in proposals if p.kernel_id not in decided]


def find_proposal(workspace_id: str,
                   kernel_id: str) -> CrossWorkspaceProposal | None:
    for p in inbox(workspace_id):
        if p.kernel_id == kernel_id:
            return p
    return None


def accept(workspace_id: str, kernel_id: str) -> bool:
    """Accept a kernel — record decision; the kernel becomes context for
    the workspace's next N cycles via the WorkspaceKB query path."""
    proposal = find_proposal(workspace_id, kernel_id)
    if proposal is None:
        return False
    _events.append(_events.Event(
        workspace_id=workspace_id,
        idea_id=kernel_id,
        type=_events.EventType.CROSS_WORKSPACE_ACCEPTED,
        ts=time.time(),
        payload={
            "kernel_id": kernel_id,
            "source_workspace_id": proposal.source_workspace_id,
            "source_idea_id": proposal.source_idea_id,
        },
    ))
    return True


def dismiss(workspace_id: str, kernel_id: str, *, reason: str = "") -> bool:
    """Dismiss a kernel — record decision so it's not re-proposed."""
    proposal = find_proposal(workspace_id, kernel_id)
    if proposal is None:
        return False
    _events.append(_events.Event(
        workspace_id=workspace_id,
        idea_id=kernel_id,
        type=_events.EventType.CROSS_WORKSPACE_DISMISSED,
        ts=time.time(),
        payload={
            "kernel_id": kernel_id,
            "source_workspace_id": proposal.source_workspace_id,
            "reason": (reason or "")[:500],
        },
    ))
    return True


# ── Internal: gates ────────────────────────────────────────────────────────

def _passes_sanitiser(text: str) -> bool:
    """True only when the existing transfer_memory sanitiser tags this as
    GLOBAL_META scope (no project nouns, no same-domain markers)."""
    try:
        from app.transfer_memory.sanitizer import check
        from app.transfer_memory.types import TransferScope
        verdict = check(text)
        if verdict.hard_rejected:
            return False
        return verdict.allowed_scope == TransferScope.GLOBAL_META
    except Exception as exc:
        logger.debug("companion.cross_workspace: sanitiser unavailable: %s",
                     exc)
        # Failing closed: if the sanitiser is unavailable we DO NOT
        # propagate. Cross-workspace risk is the wrong thing to risk on
        # a degraded import.
        return False


def _compute_relevance(kernel_text: str, target_seed: str) -> float:
    """Cosine similarity between kernel and target workspace's seed."""
    try:
        return _invoke_relevance(kernel_text, target_seed)
    except Exception as exc:
        logger.debug("companion.cross_workspace: relevance failed: %s", exc)
        return 0.0


def _already_proposed(target_workspace_id: str, source_idea_id: str) -> bool:
    """True if any prior INBOX event for this target references the same
    source idea — prevents re-proposing the same kernel on each idle run."""
    try:
        for ev in _events.read_all(target_workspace_id):
            if ev.type != _events.EventType.CROSS_WORKSPACE_INBOX:
                continue
            if (ev.payload or {}).get("source_idea_id") == source_idea_id:
                return True
    except Exception:
        return False
    return False


# ── Internal: indirections (for testability) ───────────────────────────────

def _invoke_relevance(kernel_text: str, seed_text: str) -> float:
    """Cosine similarity via the ChromaDB embedder. Returns [0, 1]."""
    from app.memory.chromadb_manager import embed
    a = embed(kernel_text)
    b = embed(seed_text)
    return _cosine(a, b)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))


def _list_projects() -> list[dict]:
    from app.control_plane.projects import get_projects
    return get_projects().list_all() or []


def _list_other_workspaces(exclude_id: str) -> list[dict]:
    return [r for r in _list_projects() if r.get("id") != exclude_id]


# ── Internal: event emission ───────────────────────────────────────────────

def _emit_inbox(*, target_workspace_id: str, source_workspace_id: str,
                 source_idea: _idea_store.IdeaRecord, relevance: float) -> None:
    """Append a CROSS_WORKSPACE_INBOX event to the TARGET workspace's log."""
    kernel_id = f"xw_{uuid.uuid4().hex[:12]}"
    try:
        _events.append(_events.Event(
            workspace_id=target_workspace_id,
            idea_id=kernel_id,
            type=_events.EventType.CROSS_WORKSPACE_INBOX,
            ts=time.time(),
            payload={
                "kernel_id": kernel_id,
                "source_workspace_id": source_workspace_id,
                "source_idea_id": source_idea.idea_id,
                "text": (source_idea.text or "")[:4000],
                "relevance_score": float(relevance),
                "transferability": float(source_idea.transferability),
                "panel_score": float(source_idea.panel_score),
            },
        ))
    except Exception as exc:
        logger.debug(
            "companion.cross_workspace: INBOX event append failed: %s", exc)
