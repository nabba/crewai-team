"""Reverse Brainstorming — provoke ideas by asking how to *cause* the problem."""

from __future__ import annotations

from app.brainstorm.techniques.base import LinearTechnique, Step


class ReverseBrainstormingTechnique(LinearTechnique):
    name = "reverse"
    title = "Reverse Brainstorming"
    description = (
        "Instead of solving the problem, ask how to cause or worsen it — then "
        "invert each failure mode into a prevention or solution."
    )

    steps = [
        Step(
            step_id="goal",
            prompt=(
                "Step 1 — Goal. Topic: {topic}\n"
                "Restate the goal you actually want to achieve, in one sentence."
            ),
        ),
        Step(
            step_id="reversal",
            prompt=(
                "Step 2 — Flip the goal. Topic: {topic}\n"
                "Now write the OPPOSITE: what would it look like to deliberately "
                "cause this problem, or to make the situation as bad as possible?"
            ),
        ),
        Step(
            step_id="cause",
            prompt=(
                "Step 3 — How to cause it. Topic: {topic}\n"
                "List 5–8 specific things someone could do to GUARANTEE the "
                "reversed (bad) outcome. Be concrete and a little playful."
            ),
        ),
        Step(
            step_id="worsen",
            prompt=(
                "Step 4 — How to worsen it. Topic: {topic}\n"
                "For an existing situation, list 3–5 things that would make it "
                "noticeably worse — failure-amplifying behaviours."
            ),
        ),
        Step(
            step_id="invert",
            prompt=(
                "Step 5 — Invert each. Topic: {topic}\n"
                "For each failure-cause you listed, write the inverse: a concrete "
                "action that would prevent or reverse it. These are your candidate "
                "solutions."
            ),
        ),
        Step(
            step_id="select",
            prompt=(
                "Step 6 — Select. Topic: {topic}\n"
                "Of the inverted solutions, which 2–3 feel most actionable or "
                "highest-leverage? Why?"
            ),
        ),
    ]
