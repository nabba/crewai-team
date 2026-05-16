"""Final-report generator — hands the brainstorm transcript to the Writer agent.

The report is a markdown document with:
    1. Header (technique, topic, timestamps)
    2. Executive summary (2-3 sentences)
    3. Per-step responses, formatted under technique-specific section headers
    4. Top ideas / promising directions
    5. Suggested next steps

If the Writer agent isn't reachable (no LLM configured, env error, …) we
fall back to a deterministic markdown rendering of the raw transcript so the
user always gets *something*.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.brainstorm.session import BrainstormSession
from app.brainstorm.techniques import get as get_technique

logger = logging.getLogger(__name__)


def _output_dir() -> Path:
    return Path(
        os.environ.get(
            "BRAINSTORM_OUTPUT_DIR", "workspace/output/brainstorm"
        )
    )


def _enriched_summary(session: BrainstormSession, base_summary: dict) -> dict:
    """Augment the technique summary with per-step agent contributions
    + per-idea novelty verdict (Q11.4) + per-idea aesthetic score (Q11.5).

    Verdicts and scores are advisory — the Writer prompt sees them
    and credits/critiques accordingly, but no idea is auto-dropped.
    """
    out = dict(base_summary)
    out["mode"] = session.mode
    out["participants"] = ["user"] + list(session.participants)
    # Q11.1 (PROGRAM §46.18) — pass analogues to the Writer so it
    # can cite cross-domain inspiration that landed in ideas.
    if session.analogues:
        out["analogues"] = list(session.analogues)

    by_step: dict[str, dict[str, list[dict]]] = {}
    for round_entry in session.agent_rounds:
        sid = round_entry.get("step_id", "?")
        phase = round_entry.get("phase", "?")
        by_step.setdefault(sid, {}).setdefault(phase, []).extend(
            round_entry.get("responses", [])
        )
    enriched_steps = []
    for step in out.get("steps", []):
        sid = step.get("step_id", "")
        rounds = by_step.get(sid, {})
        enriched_steps.append({
            **step,
            "agent_seed": _annotate_ideas(rounds.get("seed", [])),
            "agent_react": _annotate_ideas(rounds.get("react", [])),
            "user_response_annotation": _annotate_text(
                step.get("response") or ""
            ),
        })
    out["steps"] = enriched_steps
    return out


def _annotate_ideas(responses: list[dict]) -> list[dict]:
    """Run novelty + aesthetic against each agent response. Returns a
    NEW list of dicts with the annotations appended; never mutates."""
    out: list[dict] = []
    for r in responses or []:
        annot = _annotate_text(r.get("text") or "")
        out.append({**r, "annotation": annot})
    return out


def _annotate_text(text: str) -> dict:
    """Combined annotation for one idea text:

      * ``novelty`` — verdict (NOVEL / RECOMBINATION / RESTATED /
                      REJECTED_BEFORE) + match metadata
      * ``aesthetic_score`` — float 0..1 or None when store empty

    Failure-isolated — any error returns a sentinel dict so the
    Writer prompt always has a stable shape to render.
    """
    if not text or not text.strip():
        return {}
    annotation: dict[str, Any] = {}
    # Novelty (Q11.4)
    try:
        from app.creativity.novelty_wrap import assess_brainstorm_idea
        verdict = assess_brainstorm_idea(text)
        annotation["novelty"] = {
            "verdict": verdict.verdict.value,
            "primary_decision": verdict.primary_decision,
            "primary_distance": verdict.primary_distance,
            "primary_collection": verdict.primary_collection,
            "rejected_lesson_id": verdict.rejected_lesson_id,
            "rejected_score": verdict.rejected_score,
        }
    except Exception:
        annotation["novelty"] = None
    # Aesthetic (Q11.5)
    try:
        from app.creativity.aesthetic_score import score as _aes_score
        annotation["aesthetic_score"] = _aes_score(text)
    except Exception:
        annotation["aesthetic_score"] = None
    return annotation


def _writer_task_prompt(session: BrainstormSession, summary: dict) -> str:
    """Build the description text passed to the Writer agent."""
    technique = get_technique(session.technique)
    title = technique.title if technique else session.technique
    description = technique.description if technique else ""

    transcript_lines = []
    for turn in session.transcript:
        role = turn.get("role", "?")
        if role == "agent":
            who = turn.get("participant", "agent")
            phase = turn.get("phase", "")
            label = f"[agent:{who}/{phase}]" if phase else f"[agent:{who}]"
        else:
            label = f"[{role}]"
        content = turn.get("content", "")
        transcript_lines.append(f"{label} {content}")
    transcript_block = "\n".join(transcript_lines) or "(empty)"

    enriched = _enriched_summary(session, summary)
    summary_json = json.dumps(enriched, indent=2, ensure_ascii=False)

    team_note = ""
    if session.mode == "team" and session.participants:
        team_note = (
            f"\n# Team\n"
            f"This was a team session with {len(session.participants)} "
            f"agent participant(s) ({', '.join(session.participants)}) "
            f"alongside the human. Each step has agent SEED ideas (proposed "
            f"before the human spoke) and agent REACT responses (after). "
            f"The structured output below preserves both. Treat agent "
            f"contributions as input — credit or critique them by role when "
            f"useful, but the report's POV is the human user's.\n"
        )

    return (
        f"You are writing a final report for a brainstorming session.\n\n"
        f"# Technique\n"
        f"{title} — {description}\n\n"
        f"# Topic\n"
        f"{session.topic}\n"
        f"{team_note}\n"
        f"# Structured session output\n"
        f"```json\n{summary_json}\n```\n\n"
        f"# Raw transcript\n"
        f"```\n{transcript_block}\n```\n\n"
        f"# Your task\n"
        f"Produce a self-contained markdown report with these sections:\n"
        f"1. **Header** — technique name, topic, date. Note participants "
        f"if it was a team session.\n"
        f"2. **Executive summary** — 2-3 sentences capturing the most "
        f"important outcome of the session.\n"
        f"3. **Per-step output** — one subsection per step. Show the "
        f"human's response, then summarise the strongest agent "
        f"contributions (seed and react) attributing them by role. "
        f"Preserve the human's voice. Do not invent ideas no one proposed.\n"
        f"4. **Top ideas / promising directions** — extract 3-5 of the "
        f"strongest ideas across all participants, each with one sentence "
        f"on why it stands out and (if from an agent) which role suggested it.\n"
        f"5. **Suggested next steps** — 3-5 concrete actions, ordered by "
        f"leverage.\n\n"
        f"# Annotation legend (Q11.4 + Q11.5)\n"
        f"Each idea in the structured output carries an `annotation` "
        f"object:\n"
        f"  * `annotation.novelty.verdict` ∈ {{novel, recombination, "
        f"restated, rejected_before}}. RESTATED means the idea is "
        f"covered by the KBs; REJECTED_BEFORE means it matches a past "
        f"rejected proposal. Flag these in your top-ideas section.\n"
        f"  * `annotation.aesthetic_score` ∈ [0..1] or null. Higher = "
        f"closer to curated quality patterns. Use as a soft tiebreaker, "
        f"not a hard filter.\n"
        f"  * `analogues` (when present) are cross-domain patterns the "
        f"session was seeded with. Credit them when a session idea "
        f"obviously drew on one.\n\n"
        f"Tone: clear, direct, professional. No filler. No emojis. "
        f"Markdown only — no surrounding commentary or explanation of what "
        f"you've done."
    )


def _expected_output() -> str:
    return (
        "A markdown report with header / executive summary / per-step output / "
        "top ideas / next-steps sections. No surrounding prose, just the "
        "markdown."
    )


def _fallback_markdown(session: BrainstormSession) -> str:
    """Deterministic report when the Writer agent path is unavailable."""
    technique = get_technique(session.technique)
    title = technique.title if technique else session.technique
    base_summary = (
        technique.summarize(session.technique_state, session.topic)
        if technique
        else {"steps": []}
    )
    summary = _enriched_summary(session, base_summary)

    lines: list[str] = []
    lines.append(f"# Brainstorming Report — {title}")
    lines.append("")
    lines.append(f"**Topic:** {session.topic}")
    lines.append(
        f"**Technique:** {title}"
        + (f" — {technique.description}" if technique else "")
    )
    created = datetime.fromtimestamp(session.created_at, tz=timezone.utc)
    lines.append(f"**Started:** {created.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Session ID:** `{session.session_id}`")
    if session.mode == "team" and session.participants:
        lines.append(
            f"**Participants:** user + {', '.join(session.participants)}"
        )
    lines.append("")
    lines.append(
        "> Note: this report was generated without the Writer agent "
        "(LLM unavailable or disabled). The content below is the raw "
        "captured responses."
    )
    lines.append("")
    lines.append("## Per-step responses")
    lines.append("")
    for i, step in enumerate(summary.get("steps", []), start=1):
        sid = step.get("step_id", f"step_{i}")
        prompt = step.get("prompt", "")
        response = step.get("response", "") or "(no response)"
        lines.append(f"### {i}. {sid}")
        if prompt:
            lines.append("")
            lines.append(f"_Prompt:_ {prompt}")
        lines.append("")
        lines.append(f"**You:**")
        lines.append("")
        lines.append(response)
        lines.append("")
        for phase, label in (("seed", "Agent seeds (before)"), ("react", "Agent reactions (after)")):
            agent_resps = step.get(f"agent_{phase}", []) or []
            valid = [r for r in agent_resps if not r.get("error") and r.get("text", "").strip()]
            if not valid:
                continue
            lines.append(f"**{label}:**")
            lines.append("")
            for r in valid:
                lines.append(f"- _{r.get('role', '?')}_:")
                for ln in r.get("text", "").strip().splitlines():
                    lines.append(f"  {ln}")
            lines.append("")
    if not summary.get("steps"):
        lines.append("(No responses captured.)")
        lines.append("")
    return "\n".join(lines)


def _save_markdown(session: BrainstormSession, markdown: str) -> Path:
    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c for c in session.session_id if c.isalnum() or c in "-_")
    path = out_dir / f"{safe_id}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def generate_report(
    session: BrainstormSession,
    *,
    writer_factory=None,
    use_writer_agent: bool | None = None,
) -> tuple[str, str]:
    """Generate the final report. Returns ``(markdown, file_path)``.

    Tries the Writer agent first (CrewAI Task). If that fails or is disabled
    via ``BRAINSTORM_DISABLE_WRITER=1`` / ``use_writer_agent=False``, falls
    back to a deterministic markdown rendering. Either way the output is
    written to ``workspace/output/brainstorm/<session_id>.md``.

    ``writer_factory`` is injected for tests (it must return a CrewAI Agent).
    """
    technique = get_technique(session.technique)
    summary = (
        technique.summarize(session.technique_state, session.topic)
        if technique
        else {"steps": []}
    )

    use_writer = use_writer_agent
    if use_writer is None:
        use_writer = os.environ.get("BRAINSTORM_DISABLE_WRITER", "0") != "1"

    markdown: str | None = None
    if use_writer:
        try:
            markdown = _generate_via_writer(session, summary, writer_factory)
        except Exception as exc:
            logger.warning(
                "brainstorm.report: Writer-agent path failed (%s) — using fallback",
                exc,
                exc_info=True,
            )

    if markdown is None or not markdown.strip():
        markdown = _fallback_markdown(session)

    path = _save_markdown(session, markdown)
    return markdown, str(path)


def _generate_via_writer(
    session: BrainstormSession,
    summary: dict,
    writer_factory=None,
) -> str:
    """Run the CrewAI Writer agent over the session and return the markdown.

    Imports CrewAI lazily so test environments without the dependency can
    still import this module.
    """
    if writer_factory is None:
        from app.agents.writer import create_writer as writer_factory  # type: ignore
    from crewai import Crew, Process, Task  # type: ignore

    agent = writer_factory()
    description = _writer_task_prompt(session, summary)
    task = Task(
        description=description,
        expected_output=_expected_output(),
        agent=agent,
    )
    crew = Crew(
        agents=[agent], tasks=[task], process=Process.sequential, verbose=False
    )
    t0 = time.monotonic()
    result = crew.kickoff()
    logger.info(
        "brainstorm.report: writer kickoff took %.1fs", time.monotonic() - t0
    )
    return str(result).strip()
