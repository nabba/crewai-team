"""Facilitator — orchestrate one brainstorming session through Q/A.

Surface-agnostic core. Both the Signal slash-command handler and the CLI
REPL call into these functions; neither knows about the other.

In **solo** mode the flow is straightforward: prompt → user → next prompt.

In **team** (multi-agent) mode each step runs two extra rounds:

  1. *Seed* — when a prompt is first shown, the agent roster proposes
     initial answers in parallel. The user sees their seeds before typing.
  2. *React* — after the user answers, the agents react / extend / disagree
     in parallel. The user sees their reactions before the next step.

The facilitator gathers these rounds itself and returns them in a
:class:`StepDelivery` payload. Both surfaces render the same delivery — they
just format it differently.

The agent gatherer is injectable (``seed_gatherer``, ``react_gatherer``)
for tests and CLI dry-runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from app.brainstorm import store
from app.brainstorm.multi_agent import (
    AgentResponse,
    DEFAULT_ROSTER,
    gather_react as _default_gather_react,
    gather_seed as _default_gather_seed,
    resolve_roster,
)
from app.brainstorm.session import BrainstormSession
from app.brainstorm.techniques import get as get_technique
from app.brainstorm.techniques import names as technique_names

logger = logging.getLogger(__name__)


class FacilitatorError(Exception):
    """Operator-visible errors (unknown technique, no active session, etc.)."""


@dataclass
class StepDelivery:
    """What the facilitator hands back after a state transition.

    ``prompt`` — the next prompt to show the user (None when complete).
    ``seed`` — agents' opening answers for the upcoming prompt (team mode).
    ``react`` — agents' reactions to the just-given user answer (team mode).
    """

    prompt: str | None = None
    seed: list[AgentResponse] = field(default_factory=list)
    react: list[AgentResponse] = field(default_factory=list)


# ── Lifecycle ─────────────────────────────────────────────────────────────


def start(
    sender: str,
    technique_name: str,
    topic: str,
    *,
    with_agents: int | list[str] | None = None,
    seed_gatherer: Callable | None = None,
) -> tuple[BrainstormSession, StepDelivery]:
    """Open a fresh session and return the first prompt + optional seed round.

    Raises ``FacilitatorError`` if the technique is unknown or the topic is
    empty. If the sender has an active session, it is paused first.

    ``with_agents`` enables team mode. Accepts an int (first N from
    ``DEFAULT_ROSTER``) or a list of role names.
    """
    technique_name = (technique_name or "").strip().lower()
    topic = (topic or "").strip()
    if not topic:
        raise FacilitatorError("Topic must not be empty.")
    technique = get_technique(technique_name)
    if technique is None:
        raise FacilitatorError(
            f"Unknown technique '{technique_name}'. Try one of: "
            f"{', '.join(technique_names())}"
        )
    try:
        roster = resolve_roster(with_agents)
    except (ValueError, TypeError) as exc:
        raise FacilitatorError(str(exc)) from exc

    # Auto-pause any in-flight session so the new one becomes active.
    existing = store.get_active(sender)
    if existing is not None:
        existing.status = "paused"
        store.save(existing)
        store.clear_active(sender)

    session = BrainstormSession(
        session_id=BrainstormSession.new_id(),
        sender=sender,
        topic=topic,
        technique=technique.name,
        mode="team" if roster else "solo",
        participants=list(roster),
    )
    session.technique_state = technique.initial_state()
    first_prompt = technique.next_prompt(session.technique_state, topic)
    if first_prompt is None:
        raise FacilitatorError("Technique returned no first prompt — bug.")

    session.append_turn("system", f"Started {technique.title}: {topic}")
    if roster:
        session.append_turn(
            "system", f"Team mode — participants: user + {', '.join(roster)}"
        )
    session.append_turn("assistant", first_prompt)

    seed = _maybe_gather_seed(
        session, technique.title, first_prompt, seed_gatherer
    )

    store.save(session)
    store.set_active(sender, session.session_id)
    return session, StepDelivery(prompt=first_prompt, seed=seed)


def respond(
    sender: str,
    message: str,
    *,
    seed_gatherer: Callable | None = None,
    react_gatherer: Callable | None = None,
) -> tuple[BrainstormSession, StepDelivery, bool]:
    """Record a user response. Returns (session, delivery, advanced).

    ``advanced`` is True if the state machine moved forward. In team mode the
    delivery contains:
      - ``react`` — agents' reactions to the user's answer for the step that
        just completed
      - ``prompt`` — the next prompt (or None when the technique is done)
      - ``seed`` — agents' seeds for the upcoming prompt (empty when done)

    Empty messages are ignored — returns advanced=False, no agent rounds.
    """
    session = store.get_active(sender)
    if session is None:
        raise FacilitatorError(
            "No active brainstorm session. Start one with /brainstorm "
            "<technique> <topic>."
        )
    technique = get_technique(session.technique)
    if technique is None:
        raise FacilitatorError(
            f"Session references unknown technique '{session.technique}'."
        )

    text = (message or "").strip()
    if not text:
        current = technique.next_prompt(session.technique_state, session.topic)
        return session, StepDelivery(prompt=current), False

    current_prompt = technique.next_prompt(session.technique_state, session.topic)
    if current_prompt is None:
        return session, StepDelivery(prompt=None), False

    # Capture step_id BEFORE record_response advances the index.
    step_id = _current_step_id(session, technique)

    session.append_turn("user", text)
    technique.record_response(
        session.technique_state, text, prompt=current_prompt
    )

    react: list[AgentResponse] = []
    if session.mode == "team" and session.participants:
        react = _gather_with_fallback(
            "react",
            react_gatherer or _default_gather_react,
            technique_title=technique.title,
            topic=session.topic,
            step_prompt=current_prompt,
            roster=session.participants,
            user_answer=text,
            peer_seeds=_load_seed_responses(session, step_id),
        )
        _record_round(session, step_id=step_id, phase="react", responses=react)

    next_prompt = technique.next_prompt(session.technique_state, session.topic)
    seed: list[AgentResponse] = []
    if next_prompt is not None:
        session.append_turn("assistant", next_prompt)
        seed = _maybe_gather_seed(
            session, technique.title, next_prompt, seed_gatherer
        )
    else:
        session.append_turn(
            "system",
            f"All {technique.total_steps()} steps complete. Reply /brainstorm finish to generate the report.",
        )
    store.save(session)
    return session, StepDelivery(prompt=next_prompt, seed=seed, react=react), True


def skip(
    sender: str,
    *,
    seed_gatherer: Callable | None = None,
) -> tuple[BrainstormSession, StepDelivery]:
    """Skip the current step (records an empty/skipped response)."""
    session = store.get_active(sender)
    if session is None:
        raise FacilitatorError("No active brainstorm session.")
    technique = get_technique(session.technique)
    if technique is None:
        raise FacilitatorError(
            f"Session references unknown technique '{session.technique}'."
        )
    current = technique.next_prompt(session.technique_state, session.topic)
    if current is None:
        return session, StepDelivery(prompt=None)
    session.append_turn("user", "(skipped)")
    technique.record_response(
        session.technique_state, "(skipped)", prompt=current
    )
    next_prompt = technique.next_prompt(session.technique_state, session.topic)
    seed: list[AgentResponse] = []
    if next_prompt is not None:
        session.append_turn("assistant", next_prompt)
        seed = _maybe_gather_seed(
            session, technique.title, next_prompt, seed_gatherer
        )
    store.save(session)
    return session, StepDelivery(prompt=next_prompt, seed=seed)


def pause(sender: str) -> BrainstormSession:
    session = store.get_active(sender)
    if session is None:
        raise FacilitatorError("No active brainstorm session to pause.")
    session.status = "paused"
    session.append_turn("system", "Session paused.")
    store.save(session)
    store.clear_active(sender)
    return session


def resume(
    sender: str,
    session_id: str | None = None,
    *,
    seed_gatherer: Callable | None = None,
) -> tuple[BrainstormSession, StepDelivery] | None:
    """Resume the most-recent paused session (or a specific one by id).

    Returns (session, delivery) or None if there's nothing to resume. In
    team mode, a fresh seed round is gathered for the current prompt so the
    user re-enters with up-to-date agent input.
    """
    if session_id:
        sess = store.load(session_id)
        if sess is None or sess.sender != sender:
            return None
    else:
        paused = list(store.iter_paused(sender))
        if not paused:
            return None
        sess = paused[0]
    technique = get_technique(sess.technique)
    if technique is None:
        return None
    sess.status = "active"
    sess.append_turn("system", "Session resumed.")
    store.save(sess)
    store.set_active(sender, sess.session_id)
    prompt = technique.next_prompt(sess.technique_state, sess.topic)
    if prompt is None:
        return sess, StepDelivery(
            prompt="(this session is already complete — call /brainstorm finish to generate the report)"
        )
    seed = _maybe_gather_seed(sess, technique.title, prompt, seed_gatherer)
    if seed:
        store.save(sess)
    return sess, StepDelivery(prompt=prompt, seed=seed)


def cancel(sender: str) -> BrainstormSession | None:
    session = store.get_active(sender)
    if session is None:
        return None
    session.status = "cancelled"
    session.append_turn("system", "Session cancelled by user.")
    store.save(session)
    store.clear_active(sender)
    return session


def finish(
    sender: str,
    *,
    generate_report: bool = True,
    report_generator=None,
) -> BrainstormSession:
    """Close the session and optionally generate the final report.

    ``report_generator`` is injected for testability — defaults to
    :func:`app.brainstorm.report.generate_report`.
    """
    session = store.get_active(sender)
    if session is None:
        recent = store.list_sessions(sender=sender)
        if not recent:
            raise FacilitatorError("No brainstorm session to finish.")
        session = recent[0]

    if generate_report:
        if report_generator is None:
            from app.brainstorm.report import generate_report as report_generator
        try:
            report_text, report_path = report_generator(session)
            session.final_report = report_text
            session.final_report_path = report_path
        except Exception as exc:
            logger.exception("brainstorm.facilitator: report generation failed")
            session.append_turn(
                "system", f"(Report generation failed: {exc})"
            )

    session.status = "complete"
    session.append_turn("system", "Session finished.")
    store.save(session)
    store.clear_active(sender)
    return session


# ── Read-only views ───────────────────────────────────────────────────────


def status(sender: str) -> str:
    """Human-readable status line for the active session, or a default."""
    session = store.get_active(sender)
    if session is None:
        return "No active brainstorm session."
    technique = get_technique(session.technique)
    total = technique.total_steps() if technique else None
    done = session.technique_state.step_index
    progress = f"{done}/{total}" if total else f"step {done}"
    extra = (
        f" [team: {', '.join(session.participants)}]"
        if session.mode == "team"
        else ""
    )
    return (
        f"Active session {session.session_id}: {session.technique} "
        f"({progress}){extra} — topic: {session.topic[:80]}"
    )


# ── Internal helpers ──────────────────────────────────────────────────────


def _current_step_id(session: BrainstormSession, technique) -> str:
    """Return the step_id of the prompt about to be answered."""
    idx = session.technique_state.step_index
    steps = getattr(technique, "steps", [])
    if 0 <= idx < len(steps):
        return steps[idx].step_id
    return f"step_{idx}"


def _maybe_gather_seed(
    session: BrainstormSession,
    technique_title: str,
    prompt: str,
    seed_gatherer: Callable | None,
) -> list[AgentResponse]:
    """If the session is in team mode, gather a seed round and persist it."""
    if session.mode != "team" or not session.participants:
        return []
    technique = get_technique(session.technique)
    step_id = _current_step_id(session, technique)
    gatherer = seed_gatherer or _default_gather_seed
    seeds = _gather_with_fallback(
        "seed",
        gatherer,
        technique_title=technique_title,
        topic=session.topic,
        step_prompt=prompt,
        roster=session.participants,
    )
    _record_round(session, step_id=step_id, phase="seed", responses=seeds)
    return seeds


def _gather_with_fallback(
    phase: str, gatherer: Callable, **kwargs
) -> list[AgentResponse]:
    """Call the agent gatherer; on any unexpected error, return an empty list.

    Per-agent failures are already swallowed inside the gatherer — this
    catches catastrophic ones (import errors, executor death, etc.) so a
    failing team mode never breaks the user's session.
    """
    try:
        return gatherer(**kwargs)
    except Exception as exc:
        logger.warning(
            "brainstorm.facilitator: %s round crashed (%s) — continuing solo for this step",
            phase,
            exc,
        )
        return []


def _record_round(
    session: BrainstormSession,
    *,
    step_id: str,
    phase: str,
    responses: list[AgentResponse],
) -> None:
    if not responses:
        return
    session.record_agent_round(
        step_id=step_id,
        phase=phase,
        responses=[r.to_dict() for r in responses],
    )
    for r in responses:
        if r.error:
            session.append_turn(
                "system",
                f"({phase} round: {r.role} errored — {r.error})",
            )
        elif r.text:
            session.append_turn(
                "agent", r.text, participant=r.role, phase=phase
            )


def _load_seed_responses(
    session: BrainstormSession, step_id: str
) -> list[AgentResponse]:
    """Return the seed-phase AgentResponses for ``step_id`` if recorded."""
    for entry in session.agent_rounds:
        if entry.get("step_id") == step_id and entry.get("phase") == "seed":
            return [
                AgentResponse.from_dict(r)
                for r in entry.get("responses", [])
            ]
    return []
