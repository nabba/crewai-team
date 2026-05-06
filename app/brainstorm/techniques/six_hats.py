"""Six Thinking Hats — examine an idea through 6 distinct lenses (de Bono)."""

from __future__ import annotations

from app.brainstorm.techniques.base import LinearTechnique, Step


class SixHatsTechnique(LinearTechnique):
    name = "six_hats"
    title = "Six Thinking Hats"
    description = (
        "Examine an idea through six distinct frames (white/red/black/yellow/"
        "green/blue) to surface facts, feelings, risks, benefits, alternatives, "
        "and a process verdict."
    )

    steps = [
        Step(
            step_id="blue_open",
            prompt=(
                "BLUE hat (process opener). Topic: {topic}\n"
                "State the question precisely. What decision are we trying to make, "
                "and what would 'good enough' look like?"
            ),
        ),
        Step(
            step_id="white",
            prompt=(
                "WHITE hat (facts). Topic: {topic}\n"
                "What do we actually know? What data, prior outcomes, constraints "
                "exist? What's missing — what would we need to know?"
            ),
        ),
        Step(
            step_id="red",
            prompt=(
                "RED hat (emotions, no justification). Topic: {topic}\n"
                "Gut feel. What attracts or repels you about this — fears, hopes, "
                "discomfort, excitement? Don't explain, just name the feelings."
            ),
        ),
        Step(
            step_id="black",
            prompt=(
                "BLACK hat (caution). Topic: {topic}\n"
                "What could go wrong? Risks, downsides, failure modes, hidden costs, "
                "fragile assumptions. Be specific."
            ),
        ),
        Step(
            step_id="yellow",
            prompt=(
                "YELLOW hat (benefits). Topic: {topic}\n"
                "What's the upside if it works? Who wins, what's the realistic best "
                "case, what cascading benefits follow?"
            ),
        ),
        Step(
            step_id="green",
            prompt=(
                "GREEN hat (creative alternatives). Topic: {topic}\n"
                "What other paths could we take? Variants, hybrids, third options "
                "we haven't considered. Aim for 3–5."
            ),
        ),
        Step(
            step_id="blue_close",
            prompt=(
                "BLUE hat (process close). Topic: {topic}\n"
                "Given what came up under the other hats — what's the verdict? "
                "Decision, defer, or experiment? What's the next concrete action?"
            ),
        ),
    ]
