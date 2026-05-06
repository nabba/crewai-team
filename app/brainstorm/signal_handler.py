"""Signal slash-command + active-session message handler.

Single entry point :func:`try_handle` returns a string reply if it claimed
the message, or ``None`` to fall through to normal task routing.

Routing rules:
    1. If ``text`` starts with ``/brainstorm``, parse it as a control command
       (start, list, status, next, skip, pause, resume, finish, cancel).
    2. Else, if the sender has an active brainstorm session, treat ``text``
       as a free-form answer to the current step.
    3. Else, return None.

Hooked into ``app.agents.commander.commands.try_command`` near the top so it
runs before any other slash-command parsing.

Team-mode invocation::

    /brainstorm scamper with 3 agents Improve onboarding flow
    /brainstorm six_hats with agents Should we ship feature X   # default 4
    /brainstorm scamper Improve onboarding flow                  # solo
"""

from __future__ import annotations

import logging
import re

from app.brainstorm import facilitator, store
from app.brainstorm.facilitator import FacilitatorError, StepDelivery
from app.brainstorm.multi_agent import DEFAULT_ROSTER, AgentResponse
from app.brainstorm.techniques import get as get_technique
from app.brainstorm.techniques import menu as technique_menu
from app.brainstorm.techniques import names as technique_names

logger = logging.getLogger(__name__)


_HELP = (
    "Brainstorm commands:\n"
    "  /brainstorm                                — show technique menu\n"
    "  /brainstorm <technique> <topic>            — start a solo session\n"
    "  /brainstorm <technique> with N agents <topic>\n"
    "                                             — start a team session\n"
    "                                               (N agents react with you;\n"
    "                                                up to 4: researcher,\n"
    "                                                writer, coder, critic)\n"
    "  /brainstorm <technique> with agents <topic>\n"
    "                                             — same; defaults N to 4\n"
    "  /brainstorm status                         — current session progress\n"
    "  /brainstorm skip                           — skip current step\n"
    "  /brainstorm pause                          — save and exit\n"
    "  /brainstorm resume [session_id]            — continue paused session\n"
    "  /brainstorm finish                         — generate the final report\n"
    "  /brainstorm cancel                         — discard active session\n"
    "  /brainstorm list                           — past sessions for this user\n"
    "  /brainstorm help                           — this message"
)


def try_handle(text: str, sender: str) -> str | None:
    """Route a Signal message into the brainstorm subsystem if applicable.

    Returns the reply string if handled, or ``None`` to fall through.
    """
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.lower().startswith("/brainstorm"):
        return _handle_command(stripped, sender)

    if store.get_active(sender) is not None:
        return _handle_response(stripped, sender)

    return None


# ── Command parser ───────────────────────────────────────────────────────


_WITH_AGENTS_RE = re.compile(
    r"\bwith\s+(?:(\d+)\s+)?agents?\b",
    re.IGNORECASE,
)


def _parse_start(rest: str) -> tuple[str, str, int | None] | None:
    """Parse '<technique> [with N agents] <topic>'.

    Returns (technique, topic, n_agents_or_None) or None if no technique found.
    """
    parts = rest.split(maxsplit=1)
    if not parts:
        return None
    technique = parts[0].lower()
    if technique not in technique_names():
        return None
    body = parts[1].strip() if len(parts) > 1 else ""

    n_agents: int | None = None
    match = _WITH_AGENTS_RE.search(body)
    if match:
        if match.group(1):
            try:
                n_agents = int(match.group(1))
            except ValueError:
                n_agents = None
        if n_agents is None:
            n_agents = len(DEFAULT_ROSTER)  # default 4 when 'with agents' alone
        body = (body[: match.start()] + body[match.end():]).strip()

    return technique, body, n_agents


def _handle_command(text: str, sender: str) -> str:
    rest = text[len("/brainstorm"):].strip()
    if not rest:
        return _show_menu(sender)

    parts = rest.split(maxsplit=1)
    verb = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if verb in ("help", "?"):
        return _HELP
    if verb == "list":
        return _show_list(sender)
    if verb == "status":
        return facilitator.status(sender)
    if verb == "skip":
        return _do_skip(sender)
    if verb == "pause":
        return _do_pause(sender)
    if verb == "resume":
        return _do_resume(sender, arg or None)
    if verb == "finish":
        return _do_finish(sender)
    if verb == "cancel":
        return _do_cancel(sender)

    parsed = _parse_start(rest)
    if parsed is None:
        return (
            f"Unknown technique or command '{verb}'. "
            f"Type '/brainstorm' for the menu or '/brainstorm help' for "
            f"command list."
        )
    technique, topic, n_agents = parsed
    if not topic:
        tech = get_technique(technique)
        return (
            f"Technique '{tech.title}' selected, but I need a topic. "
            f"Try: /brainstorm {technique} <your topic here>"
        )
    return _do_start(sender, technique, topic, n_agents)


# ── Action handlers ──────────────────────────────────────────────────────


def _show_menu(sender: str) -> str:
    active = store.get_active(sender)
    parts = [technique_menu()]
    parts.append("")
    parts.append("Solo:  /brainstorm <technique> <topic>")
    parts.append("Team:  /brainstorm <technique> with N agents <topic>   (N up to 4)")
    parts.append(
        "Example: /brainstorm six_hats with 3 agents Should we ship feature X"
    )
    parts.append("All commands: /brainstorm help")
    if active is not None:
        parts.append("")
        team = (
            f" (team: {', '.join(active.participants)})"
            if active.mode == "team"
            else ""
        )
        parts.append(
            f"(You currently have an active session: {active.session_id} — "
            f"{active.technique}{team}. Starting a new one will pause it.)"
        )
    return "\n".join(parts)


def _show_list(sender: str) -> str:
    sessions = store.list_sessions(sender=sender)
    if not sessions:
        return "No brainstorm sessions yet for this user."
    lines = ["Your brainstorm sessions (newest first):"]
    for s in sessions[:20]:
        flag = {"active": "●", "paused": "‖", "complete": "✓", "cancelled": "✗"}.get(s.status, "?")
        topic = s.topic[:60] + ("…" if len(s.topic) > 60 else "")
        team = f" [+{len(s.participants)}]" if s.mode == "team" else ""
        lines.append(f"  {flag} {s.session_id} — {s.technique}{team} — {topic}")
    if len(sessions) > 20:
        lines.append(f"  …and {len(sessions) - 20} more.")
    return "\n".join(lines)


def _do_start(sender: str, technique: str, topic: str, n_agents: int | None) -> str:
    try:
        session, delivery = facilitator.start(
            sender, technique, topic, with_agents=n_agents
        )
    except FacilitatorError as exc:
        return f"Couldn't start: {exc}"
    tech = get_technique(session.technique)
    if session.mode == "team":
        header = (
            f"Starting {tech.title} on: {session.topic}\n"
            f"Session: {session.session_id}\n"
            f"Team mode — {len(session.participants)} agent(s): "
            f"{', '.join(session.participants)}.\n"
            f"Each step has 3 phases: agents seed → you answer → agents react.\n"
            f"Use /brainstorm pause | skip | finish | cancel to control."
        )
    else:
        header = (
            f"Starting {tech.title} on: {session.topic}\n"
            f"Session: {session.session_id}\n"
            f"Reply with your answer to each step. "
            f"Use /brainstorm pause | skip | finish | cancel to control."
        )
    return _render_delivery(header, delivery, header_first=True)


def _do_skip(sender: str) -> str:
    try:
        session, delivery = facilitator.skip(sender)
    except FacilitatorError as exc:
        return f"{exc}"
    if delivery.prompt is None:
        return (
            "Last step skipped. All steps complete — reply /brainstorm finish "
            "to generate the report."
        )
    return _render_delivery("Skipped.", delivery, header_first=True)


def _do_pause(sender: str) -> str:
    try:
        session = facilitator.pause(sender)
    except FacilitatorError as exc:
        return f"{exc}"
    return (
        f"Session {session.session_id} paused. "
        f"Resume any time with /brainstorm resume."
    )


def _do_resume(sender: str, session_id: str | None) -> str:
    result = facilitator.resume(sender, session_id=session_id)
    if result is None:
        return "Nothing to resume — no paused sessions for this user."
    session, delivery = result
    header = (
        f"Resumed {session.session_id} ({session.technique}, topic: "
        f"{session.topic})."
    )
    return _render_delivery(header, delivery, header_first=True)


def _do_finish(sender: str) -> str:
    try:
        session = facilitator.finish(sender)
    except FacilitatorError as exc:
        return f"{exc}"
    if session.final_report_path:
        loc = session.final_report_path
        return (
            f"Session {session.session_id} finished. Report saved to:\n"
            f"  {loc}\n\n"
            f"--- Report preview ---\n"
            f"{(session.final_report or '')[:1500]}"
            + (
                "\n…(truncated — full report in the file above)"
                if session.final_report and len(session.final_report) > 1500
                else ""
            )
        )
    return f"Session {session.session_id} finished, but no report was produced."


def _do_cancel(sender: str) -> str:
    session = facilitator.cancel(sender)
    if session is None:
        return "No active brainstorm session to cancel."
    return f"Session {session.session_id} cancelled."


def _handle_response(text: str, sender: str) -> str:
    try:
        session, delivery, advanced = facilitator.respond(sender, text)
    except FacilitatorError as exc:
        return f"{exc}"
    if delivery.prompt is None and not delivery.react and advanced:
        return (
            "All steps complete. Reply /brainstorm finish to generate your "
            "report, or /brainstorm cancel to discard."
        )
    if not advanced:
        return f"(empty answer — please respond)\n\n{delivery.prompt or ''}"
    if delivery.prompt is None:
        # Last step: react may still be present
        body = _render_delivery("", delivery, header_first=False)
        return body + (
            "\n\n— All steps complete. Reply /brainstorm finish to generate "
            "your report."
        )
    return _render_delivery("", delivery, header_first=False)


# ── Rendering ────────────────────────────────────────────────────────────


def _render_agent_block(label: str, responses: list[AgentResponse]) -> str:
    """Render one round (seed or react) as a markdown-ish block."""
    blocks: list[str] = [f"=== {label} ==="]
    for r in responses:
        if r.error:
            blocks.append(f"— {r.role} (error: {r.error})")
            continue
        if not r.text.strip():
            blocks.append(f"— {r.role}: (no response)")
            continue
        blocks.append(f"— {r.role}\n{r.text.strip()}")
    return "\n\n".join(blocks)


def _render_delivery(
    header: str, delivery: StepDelivery, *, header_first: bool
) -> str:
    """Format a StepDelivery for Signal display.

    Order: header → react (if any) → prompt → seed (if any).
    React refers to the just-finished step; prompt is the new step; seed is
    the agents' opening for that new step.
    """
    chunks: list[str] = []
    if header_first and header:
        chunks.append(header)
    if delivery.react:
        chunks.append(_render_agent_block("AGENTS REACT", delivery.react))
    if delivery.prompt is not None:
        chunks.append(delivery.prompt)
    if delivery.seed:
        chunks.append(_render_agent_block("AGENTS SEED", delivery.seed))
    if not header_first and header:
        chunks.append(header)
    return "\n\n".join(c for c in chunks if c)
