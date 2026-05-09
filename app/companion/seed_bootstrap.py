"""Cold-start seed derivation from workspace activity.

When ``cycle.run_cycle`` picks a workspace whose ``seed_prompt`` is None,
this module reads the human signal that already exists — the project's
``control_plane.projects.mission`` and recent ``control_plane.tickets``
the operator has filed against the project — and synthesises a candidate
seed via one cheap-tier LLM call.

The result is persisted as the workspace's ``seed_prompt`` and surfaced
via a ``SEED_DERIVED`` event so the React Settings tab can flag the seed
as auto-derived. The user retains full override.

Phase 11.5 stops at mission + tickets. Phase 11.5b will add a third
source — recent conversation turns routed to this project — once
``conversation_store.add_message`` captures ``project_id``.

The ``default`` workspace is intentionally blocklisted: it's the
catch-all fallback project, with mixed-topic tickets that produce a
noisy seed unhelpful for contemplation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Activation gates (infrastructure-bounded — Self-Improver cannot widen).
MIN_MISSION_CHARS = 10
MIN_SIGNAL_TICKETS = 1     # need at least this many tickets OR a mission
MAX_TICKETS_IN_PROMPT = 15
TICKET_TITLE_MAX_CHARS = 200
SEED_MAX_CHARS = 250

# Workspaces that should NEVER be auto-bootstrapped.
DEFAULT_BLOCKLIST: tuple[str, ...] = ("default",)


@dataclass
class SeedDerivation:
    """One bootstrap result. Surfaced as a SEED_DERIVED event payload."""
    workspace_id: str
    text: str
    source_signal: str          # "mission+tickets" | "tickets_only" | "mission_only"
    ticket_count: int = 0
    has_mission: bool = False
    rationale: str = ""


def derive_seed(workspace_id: str, *,
                blocklist_names: tuple[str, ...] = DEFAULT_BLOCKLIST,
                ) -> SeedDerivation | None:
    """Read CP mission + recent tickets; synthesise a candidate seed.

    Returns None when:
      - workspace doesn't exist
      - workspace name is on the blocklist (e.g. "default")
      - signal too thin (no mission AND fewer than MIN_SIGNAL_TICKETS)
      - LLM call fails or response unparseable

    Cost: one cheap-tier LLM call (~$0.0002). Bootstrap fires at most
    once per workspace — subsequent cycles see ``seed_prompt`` already
    set and skip this path.
    """
    project = _get_project(workspace_id)
    if not project:
        return None
    name = (project.get("name") or "").strip()
    if name in blocklist_names:
        logger.debug("companion.seed_bootstrap: workspace %r is blocklisted",
                     name)
        return None

    mission = (project.get("mission") or "").strip()
    has_mission = len(mission) >= MIN_MISSION_CHARS
    tickets = _recent_tickets(workspace_id, limit=MAX_TICKETS_IN_PROMPT)

    if not has_mission and len(tickets) < MIN_SIGNAL_TICKETS:
        return None

    if has_mission and tickets:
        signal = "mission+tickets"
    elif tickets:
        signal = "tickets_only"
    else:
        signal = "mission_only"

    prompt = _PROMPT_TEMPLATE.format(
        project_name=name or workspace_id[:12],
        mission=mission or "(no mission set)",
        recent_tickets=("\n".join(f"- {t}" for t in tickets)
                        if tickets else "(no recent tickets)"),
    )
    try:
        raw = _invoke_synthesizer(prompt)
    except Exception as exc:
        logger.debug("companion.seed_bootstrap: LLM failed: %s", exc)
        return None

    text, rationale = _parse(raw)
    if not text:
        return None

    return SeedDerivation(
        workspace_id=workspace_id,
        text=text,
        source_signal=signal,
        ticket_count=len(tickets),
        has_mission=has_mission,
        rationale=rationale,
    )


# ── Prompt + parser ────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
A workspace named "{project_name}" has the mission:
"{mission}"

Recent tasks the operator has filed in this workspace:
{recent_tickets}

Write ONE seed prompt for ongoing background contemplation about this
workspace. The seed must:
  - Anchor on this workspace's actual focus (not a generic question)
  - Be ambitious and open-ended (months of contemplation possible)
  - Invite lateral thinking, analogies, conceptual blends
  - Stay on this workspace's topic — never wander into unrelated domains

Output exactly two lines:
SEED: <single seed prompt, ≤250 chars>
RATIONALE: <one short sentence on why this captures the workspace's heart>
"""


def _parse(raw: str) -> tuple[str, str]:
    text = raw or ""
    seed_m = re.search(r"SEED\s*[:\-]\s*(.+?)(?:\n|$)",
                        text, flags=re.IGNORECASE)
    rat_m = re.search(r"RATIONALE\s*[:\-]\s*(.+?)(?:\n\n|\Z)",
                       text, flags=re.IGNORECASE | re.DOTALL)
    seed = (seed_m.group(1).strip() if seed_m else "")[:SEED_MAX_CHARS]
    rationale = re.sub(
        r"\s+", " ",
        (rat_m.group(1) if rat_m else "").strip(),
    )[:300]
    return seed, rationale


# ── Indirections (testability) ─────────────────────────────────────────────

def _invoke_synthesizer(prompt: str) -> str:
    """Cheap-tier LLM call, role=commander matching grand_task."""
    from app.llm_factory import create_specialist_llm
    llm = create_specialist_llm(max_tokens=240, role="commander")
    return str(llm.call(prompt))


def _get_project(workspace_id: str) -> dict | None:
    """Indirection over CP project lookup."""
    try:
        from app.control_plane.projects import get_projects
        return get_projects().get_by_id(workspace_id)
    except Exception as exc:
        logger.debug(
            "companion.seed_bootstrap: get_project failed for %s: %s",
            workspace_id, exc,
        )
        return None


def _recent_tickets(workspace_id: str, *, limit: int) -> list[str]:
    """Last N ticket titles for this project, newest first.

    Returns [] on any DB failure or empty workspace.
    """
    try:
        from app.control_plane.db import execute
        rows = execute(
            "SELECT title FROM control_plane.tickets "
            "WHERE project_id = %s ORDER BY created_at DESC LIMIT %s",
            (workspace_id, int(limit)),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug(
            "companion.seed_bootstrap: ticket fetch failed for %s: %s",
            workspace_id, exc,
        )
        return []
    out: list[str] = []
    for r in rows:
        title = (r.get("title") or "").strip()
        if not title:
            continue
        out.append(title[:TICKET_TITLE_MAX_CHARS])
    return out
