"""SCAMPER — apply 7 transformation lenses to an existing idea/object."""

from __future__ import annotations

from app.brainstorm.techniques.base import LinearTechnique, Step


class ScamperTechnique(LinearTechnique):
    name = "scamper"
    title = "SCAMPER"
    description = (
        "Apply 7 transformation lenses (Substitute, Combine, Adapt, Modify, "
        "Put-to-other-use, Eliminate, Reverse) to an existing idea or object."
    )

    steps = [
        Step(
            step_id="substitute",
            prompt=(
                "S — Substitute. About: {topic}\n"
                "What component, material, person, rule, or assumption could be "
                "swapped out? List 2–4 substitutions."
            ),
            expected_output="2-4 substitutions",
        ),
        Step(
            step_id="combine",
            prompt=(
                "C — Combine. About: {topic}\n"
                "What could you merge with this — another product, idea, audience, "
                "or process — to create something new? List 2–4 combinations."
            ),
            expected_output="2-4 combinations",
        ),
        Step(
            step_id="adapt",
            prompt=(
                "A — Adapt. About: {topic}\n"
                "What could you borrow from another field, era, or species that "
                "solves a similar problem? List 2–4 adaptations."
            ),
            expected_output="2-4 adaptations",
        ),
        Step(
            step_id="modify",
            prompt=(
                "M — Modify / Magnify. About: {topic}\n"
                "What could be made bigger, smaller, slower, faster, more frequent, "
                "louder, denser? List 2–4 modifications."
            ),
            expected_output="2-4 modifications",
        ),
        Step(
            step_id="put_to_other_use",
            prompt=(
                "P — Put to other use. About: {topic}\n"
                "Who else could use this, in what other context, for what other "
                "purpose? List 2–4 alternative uses."
            ),
            expected_output="2-4 alternative uses",
        ),
        Step(
            step_id="eliminate",
            prompt=(
                "E — Eliminate. About: {topic}\n"
                "What could be removed, simplified, or skipped without breaking the "
                "core value? List 2–4 things to cut."
            ),
            expected_output="2-4 cuts",
        ),
        Step(
            step_id="reverse",
            prompt=(
                "R — Reverse / Rearrange. About: {topic}\n"
                "What if the order, roles, or causality were flipped? List 2–4 "
                "reversals or rearrangements."
            ),
            expected_output="2-4 reversals",
        ),
    ]
