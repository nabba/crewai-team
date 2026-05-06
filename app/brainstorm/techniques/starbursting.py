"""Starbursting — generate questions (not answers) using Who/What/When/Where/Why/How."""

from __future__ import annotations

from app.brainstorm.techniques.base import LinearTechnique, Step


class StarburstingTechnique(LinearTechnique):
    name = "starbursting"
    title = "Starbursting"
    description = (
        "Generate questions instead of answers. Six rays — Who, What, When, "
        "Where, Why, How — each prompts a cluster of probing questions. The "
        "output is a question map that exposes what you don't yet know."
    )

    steps = [
        Step(
            step_id="center",
            prompt=(
                "Centre. Topic: {topic}\n"
                "Restate the idea, project, or decision you're examining as the "
                "centre of the star. One sentence."
            ),
        ),
        Step(
            step_id="who",
            prompt=(
                "WHO. Topic: {topic}\n"
                "List 4–6 questions starting with 'Who'. Stakeholders, users, "
                "decision-makers, blockers, beneficiaries, opponents."
            ),
        ),
        Step(
            step_id="what",
            prompt=(
                "WHAT. Topic: {topic}\n"
                "List 4–6 'What' questions. Definitions, scope, components, "
                "deliverables, success criteria, dependencies."
            ),
        ),
        Step(
            step_id="when",
            prompt=(
                "WHEN. Topic: {topic}\n"
                "List 4–6 'When' questions. Timing, sequencing, deadlines, "
                "triggers, expiry conditions."
            ),
        ),
        Step(
            step_id="where",
            prompt=(
                "WHERE. Topic: {topic}\n"
                "List 4–6 'Where' questions. Channels, environments, locations, "
                "platforms, contexts of use."
            ),
        ),
        Step(
            step_id="why",
            prompt=(
                "WHY. Topic: {topic}\n"
                "List 4–6 'Why' questions. Motivation, justification, alternatives "
                "considered, root causes, what-happens-if-not."
            ),
        ),
        Step(
            step_id="how",
            prompt=(
                "HOW. Topic: {topic}\n"
                "List 4–6 'How' questions. Mechanisms, steps, measures of progress, "
                "fallbacks, edge cases."
            ),
        ),
        Step(
            step_id="prioritize",
            prompt=(
                "Prioritize. Topic: {topic}\n"
                "Of all the questions you generated, which 3 are the most "
                "load-bearing — answers we'd need before committing? Why those?"
            ),
        ),
    ]
