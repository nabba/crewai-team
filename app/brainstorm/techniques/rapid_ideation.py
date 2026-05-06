"""Rapid Ideation — quantity bursts followed by clustering and selection."""

from __future__ import annotations

from app.brainstorm.techniques.base import LinearTechnique, Step


class RapidIdeationTechnique(LinearTechnique):
    name = "rapid_ideation"
    title = "Rapid Ideation"
    description = (
        "Three short bursts of quantity-first idea generation from different "
        "angles, then cluster and pick. Best when you don't yet know what good "
        "looks like."
    )

    steps = [
        Step(
            step_id="topic",
            prompt=(
                "Step 1 — Topic. Topic: {topic}\n"
                "Sharpen the topic into a single, concrete prompt you'll generate "
                "ideas against. The narrower the better."
            ),
        ),
        Step(
            step_id="constraints",
            prompt=(
                "Step 2 — Constraints. Topic: {topic}\n"
                "List any hard constraints — budget, time, audience, must-haves, "
                "must-NOT-haves. These bound the search space."
            ),
        ),
        Step(
            step_id="burst_1",
            prompt=(
                "Burst 1 / 3 — Obvious. Topic: {topic}\n"
                "List 10 ideas, fast. The obvious ones. Don't filter — get the "
                "clichés out so the deeper ideas can surface."
            ),
            expected_output="10 ideas",
        ),
        Step(
            step_id="burst_2",
            prompt=(
                "Burst 2 / 3 — Constraint-flipped. Topic: {topic}\n"
                "List 10 more ideas, but this time IGNORE one of the constraints "
                "(pick the one most likely to be hiding good options behind it)."
            ),
            expected_output="10 ideas",
        ),
        Step(
            step_id="burst_3",
            prompt=(
                "Burst 3 / 3 — Different lens. Topic: {topic}\n"
                "List 10 more ideas from a totally different POV — a child, an "
                "adversary, a future-you, a different field. Whatever shifts the "
                "frame."
            ),
            expected_output="10 ideas",
        ),
        Step(
            step_id="cluster",
            prompt=(
                "Cluster. Topic: {topic}\n"
                "Group the ~30 ideas into 3–6 themes. Name each theme."
            ),
        ),
        Step(
            step_id="select",
            prompt=(
                "Select. Topic: {topic}\n"
                "Pick the 3 ideas (across themes) you'd actually want to pursue. "
                "For each, one line: why this one?"
            ),
        ),
    ]
