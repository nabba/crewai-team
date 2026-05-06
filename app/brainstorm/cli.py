"""Standalone CLI: ``python -m app.brainstorm``.

Drives the same facilitator the Signal handler uses, just with a stdin/stdout
loop. Sender ID defaults to ``"cli:<username>"`` so CLI sessions don't
collide with Signal sessions.

Subcommands::

    python -m app.brainstorm                  # interactive: pick + run
    python -m app.brainstorm --with-agents 4  # team mode (4 = full roster)
    python -m app.brainstorm --resume         # resume most recent paused
    python -m app.brainstorm --resume <ID>    # resume specific session
    python -m app.brainstorm --list           # list past sessions
    python -m app.brainstorm --techniques     # show technique menu
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from app.brainstorm import facilitator, store
from app.brainstorm.facilitator import FacilitatorError, StepDelivery
from app.brainstorm.multi_agent import DEFAULT_ROSTER, AgentResponse
from app.brainstorm.techniques import get as get_technique
from app.brainstorm.techniques import menu as technique_menu
from app.brainstorm.techniques import names as technique_names


def _default_sender() -> str:
    user = os.environ.get("USER") or getpass.getuser() or "anon"
    return f"cli:{user}"


def _print(msg: str) -> None:
    sys.stdout.write(msg)
    if not msg.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _read(prompt: str = "> ") -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _pick_technique() -> str | None:
    _print(technique_menu())
    _print("")
    while True:
        choice = _read("Pick a technique (short name, or 'q' to quit): ").strip().lower()
        if choice in ("q", "quit", "exit", ""):
            return None
        if choice in technique_names():
            return choice
        _print(f"Unknown — try one of: {', '.join(technique_names())}")


def _print_agent_block(label: str, responses: list[AgentResponse]) -> None:
    if not responses:
        return
    _print("")
    _print(f"───── {label} ─────")
    for r in responses:
        if r.error:
            _print(f"  [{r.role}] (error: {r.error})")
            continue
        if not r.text.strip():
            _print(f"  [{r.role}] (no response)")
            continue
        _print(f"  [{r.role}]")
        for line in r.text.strip().splitlines():
            _print(f"    {line}")
    _print("")


def _render_delivery(delivery: StepDelivery) -> None:
    """Print a StepDelivery: react (prev step) → prompt → seed (new step)."""
    if delivery.react:
        _print_agent_block("AGENTS REACT", delivery.react)
    if delivery.prompt is not None:
        _print("")
        _print(delivery.prompt)
        _print("")
    if delivery.seed:
        _print_agent_block("AGENTS SEED", delivery.seed)


def _interactive_session(
    sender: str, technique: str, topic: str, n_agents: int | None
) -> int:
    try:
        session, delivery = facilitator.start(
            sender, technique, topic, with_agents=n_agents
        )
    except FacilitatorError as exc:
        _print(f"Could not start: {exc}")
        return 1
    tech = get_technique(session.technique)
    _print("")
    _print(f"=== {tech.title} — {session.session_id} ===")
    _print(f"Topic: {session.topic}")
    if session.mode == "team":
        _print(
            f"Team mode — {len(session.participants)} agent(s): "
            f"{', '.join(session.participants)}"
        )
    _print("")
    _print(
        "Commands during the session: 'skip', 'pause', 'finish', 'cancel', 'status'."
    )
    _print("Empty line repeats the current prompt. Ctrl-D to pause and exit.")
    _render_delivery(delivery)
    return _drive_loop(sender, delivery.prompt)


def _drive_loop(sender: str, current_prompt: str | None) -> int:
    while True:
        if current_prompt is None:
            _print(
                "All steps complete. Type 'finish' to generate the report, or "
                "'cancel' to discard."
            )
        line = _read("> ")
        if line == "" and current_prompt is None:
            continue
        verb = line.strip().lower()

        if verb == "":
            # Re-show same prompt
            if current_prompt is not None:
                _print("")
                _print(current_prompt)
                _print("")
            continue
        if verb == "status":
            _print(facilitator.status(sender))
            continue
        if verb == "skip":
            try:
                _, delivery = facilitator.skip(sender)
            except FacilitatorError as exc:
                _print(str(exc))
                continue
            current_prompt = delivery.prompt
            _render_delivery(delivery)
            continue
        if verb == "pause":
            try:
                facilitator.pause(sender)
                _print("Session paused. Resume with: python -m app.brainstorm --resume")
            except FacilitatorError as exc:
                _print(str(exc))
            return 0
        if verb == "cancel":
            facilitator.cancel(sender)
            _print("Session cancelled.")
            return 0
        if verb == "finish":
            return _finish_and_print(sender)

        # Free-form answer.
        try:
            _, delivery, advanced = facilitator.respond(sender, line)
        except FacilitatorError as exc:
            _print(str(exc))
            return 1
        if not advanced:
            _print("(empty answer — please respond)")
            continue
        current_prompt = delivery.prompt
        _render_delivery(delivery)


def _finish_and_print(sender: str) -> int:
    try:
        session = facilitator.finish(sender)
    except FacilitatorError as exc:
        _print(str(exc))
        return 1
    _print("")
    if session.final_report_path:
        _print(f"Report written to: {session.final_report_path}")
        _print("")
        _print("--- Report ---")
        _print(session.final_report or "(empty)")
    else:
        _print("Session finished, but no report was produced.")
    return 0


def _list_command(sender: str) -> int:
    sessions = store.list_sessions(sender=sender)
    if not sessions:
        _print(f"No sessions for {sender}.")
        return 0
    _print(f"Sessions for {sender} (newest first):")
    for s in sessions:
        topic = s.topic[:60] + ("…" if len(s.topic) > 60 else "")
        team = (
            f" [team: {','.join(s.participants)}]" if s.mode == "team" else ""
        )
        _print(f"  [{s.status:8}] {s.session_id} — {s.technique}{team} — {topic}")
    return 0


def _resume_command(sender: str, session_id: str | None) -> int:
    result = facilitator.resume(sender, session_id=session_id)
    if result is None:
        _print("Nothing to resume.")
        return 1
    session, delivery = result
    tech = get_technique(session.technique)
    _print(f"Resumed {session.session_id} ({tech.title if tech else session.technique}).")
    _print(f"Topic: {session.topic}")
    if session.mode == "team":
        _print(
            f"Team mode — {len(session.participants)} agent(s): "
            f"{', '.join(session.participants)}"
        )
    _render_delivery(delivery)
    return _drive_loop(sender, delivery.prompt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.brainstorm",
        description="Interactive brainstorming sessions backed by the BotArmy "
        "Writer agent. Optionally include 1-4 high-creativity agents as "
        "participants alongside you.",
    )
    parser.add_argument(
        "--sender",
        default=None,
        help="Sender ID (defaults to cli:$USER). Use to share sessions with "
        "Signal — set this to your Signal phone number.",
    )
    parser.add_argument(
        "--technique",
        choices=technique_names(),
        help="Technique short name; if omitted, prompts for one.",
    )
    parser.add_argument(
        "--topic",
        help="Topic to brainstorm. If omitted, prompts for one.",
    )
    parser.add_argument(
        "--with-agents",
        type=int,
        default=0,
        choices=list(range(0, len(DEFAULT_ROSTER) + 1)),
        help=(
            "Run in team mode with N agents (0 = solo, default). "
            f"N up to {len(DEFAULT_ROSTER)} drawn from "
            f"{', '.join(DEFAULT_ROSTER)}."
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__latest__",
        default=None,
        metavar="SESSION_ID",
        help="Resume the most recent paused session, or a specific one.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List past sessions and exit."
    )
    parser.add_argument(
        "--techniques",
        action="store_true",
        help="Show technique menu and exit.",
    )
    args = parser.parse_args(argv)

    sender = args.sender or _default_sender()

    if args.techniques:
        _print(technique_menu())
        return 0
    if args.list:
        return _list_command(sender)
    if args.resume is not None:
        sid = None if args.resume == "__latest__" else args.resume
        return _resume_command(sender, sid)

    technique = args.technique or _pick_technique()
    if technique is None:
        return 0
    topic = args.topic
    if not topic:
        topic = _read("Topic: ").strip()
        if not topic:
            _print("Aborted — no topic.")
            return 1
    n_agents = args.with_agents if args.with_agents > 0 else None
    return _interactive_session(sender, technique, topic, n_agents)


if __name__ == "__main__":
    raise SystemExit(main())
