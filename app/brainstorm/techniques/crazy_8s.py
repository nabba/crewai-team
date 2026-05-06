"""Crazy-8s — generate 8 distinct ideas under tight time pressure (Design Sprint)."""

from __future__ import annotations

from app.brainstorm.techniques.base import LinearTechnique, Step


class CrazyEightsTechnique(LinearTechnique):
    name = "crazy_8s"
    title = "Crazy-8s"
    description = (
        "Generate 8 distinct ideas in 8 quick rounds, one minute each. Quantity "
        "over polish — the goal is to outrun your inner critic."
    )

    steps = [
        Step(
            step_id="frame",
            prompt=(
                "Frame. Topic: {topic}\n"
                "State the focus of this Crazy-8s round in one sentence — the "
                "specific question or feature you're generating ideas for. "
                "(Coming next: 8 timed rounds, one idea each.)"
            ),
        ),
        Step(
            step_id="idea_1",
            prompt=(
                "Idea 1 / 8. Topic: {topic}\n"
                "Spend ~1 minute. Sketch one rough idea in 1–3 sentences. Don't "
                "edit. Don't be clever. Just go."
            ),
        ),
        Step(
            step_id="idea_2",
            prompt=(
                "Idea 2 / 8. Topic: {topic}\n"
                "Different angle from idea 1. ~1 minute. Roughest form is fine."
            ),
        ),
        Step(
            step_id="idea_3",
            prompt=(
                "Idea 3 / 8. Topic: {topic}\n"
                "Try the silliest version that could work."
            ),
        ),
        Step(
            step_id="idea_4",
            prompt=(
                "Idea 4 / 8. Topic: {topic}\n"
                "What would the laziest possible solution look like?"
            ),
        ),
        Step(
            step_id="idea_5",
            prompt=(
                "Idea 5 / 8. Topic: {topic}\n"
                "What if you had 10× the budget or 10× the constraints?"
            ),
        ),
        Step(
            step_id="idea_6",
            prompt=(
                "Idea 6 / 8. Topic: {topic}\n"
                "Borrow shamelessly from a totally unrelated product or domain."
            ),
        ),
        Step(
            step_id="idea_7",
            prompt=(
                "Idea 7 / 8. Topic: {topic}\n"
                "What would your most opinionated user/colleague propose?"
            ),
        ),
        Step(
            step_id="idea_8",
            prompt=(
                "Idea 8 / 8. Topic: {topic}\n"
                "Last one. Whatever's left in your head — write it down."
            ),
        ),
        Step(
            step_id="star",
            prompt=(
                "Star. Topic: {topic}\n"
                "Look back at all 8. Pick your top 2. What makes them stand out — "
                "feasibility, novelty, fit?"
            ),
        ),
    ]
