"""How-Might-We — reframe a problem as opportunity questions, then expand."""

from __future__ import annotations

from app.brainstorm.techniques.base import LinearTechnique, Step


class HowMightWeTechnique(LinearTechnique):
    name = "how_might_we"
    title = "How-Might-We"
    description = (
        "Convert a problem into 'How might we…' opportunity questions, then "
        "broaden, narrow, and select the most generative phrasings before "
        "diverging on solutions."
    )

    steps = [
        Step(
            step_id="problem",
            prompt=(
                "Step 1 — Problem. Topic: {topic}\n"
                "Restate the problem in one sentence. Who is affected, and what's "
                "the specific friction or unmet need?"
            ),
        ),
        Step(
            step_id="user",
            prompt=(
                "Step 2 — Who. Topic: {topic}\n"
                "Who exactly are we solving for? Be concrete — a real person, role, "
                "or segment. What context are they in when this matters?"
            ),
        ),
        Step(
            step_id="insight",
            prompt=(
                "Step 3 — Insight. Topic: {topic}\n"
                "What's the surprising or non-obvious thing you've noticed about "
                "this problem? The insight that makes the HMW question generative."
            ),
        ),
        Step(
            step_id="hmw_seed",
            prompt=(
                "Step 4 — Seed HMW. Topic: {topic}\n"
                "Write your first 'How might we …' question based on the problem + "
                "user + insight. Keep it open enough that 5 different solutions could "
                "answer it, narrow enough that 100 couldn't."
            ),
        ),
        Step(
            step_id="amplify",
            prompt=(
                "Step 5 — Amplify the good. Topic: {topic}\n"
                "Take a positive aspect of the situation and ask: 'How might we make "
                "even more of [that good thing]?' Write 1–2 such HMWs."
            ),
        ),
        Step(
            step_id="remove",
            prompt=(
                "Step 6 — Remove the bad. Topic: {topic}\n"
                "Take the worst part of the situation: 'How might we eliminate / "
                "reduce / soften [that bad thing]?' Write 1–2 such HMWs."
            ),
        ),
        Step(
            step_id="opposite",
            prompt=(
                "Step 7 — Flip the assumption. Topic: {topic}\n"
                "What's the assumption everyone makes about this? Now write a HMW "
                "that takes the OPPOSITE assumption seriously."
            ),
        ),
        Step(
            step_id="analogy",
            prompt=(
                "Step 8 — Analogy. Topic: {topic}\n"
                "What's a totally different domain that solved a structurally similar "
                "problem? Write a HMW that borrows from that domain."
            ),
        ),
        Step(
            step_id="select",
            prompt=(
                "Step 9 — Select. Topic: {topic}\n"
                "Of all the HMW questions you wrote, pick the 2–3 most generative "
                "(broad enough to invite many ideas, narrow enough to be useful). "
                "List them."
            ),
        ),
        Step(
            step_id="solutions",
            prompt=(
                "Step 10 — First solutions. Topic: {topic}\n"
                "For each selected HMW, brainstorm 3–5 candidate solutions. Don't "
                "filter — quantity over quality at this stage."
            ),
        ),
    ]
