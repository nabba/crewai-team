"""Concept-blend technique — Fauconnier-Turner 4-space blend.

PROGRAM §46.19 (Q11.2). The 8th brainstorm technique. Where SCAMPER
applies fixed lenses to an existing idea and How-Might-We reframes a
problem, concept-blend is a structured cross-domain creative
operator: take two unrelated mental spaces, find the generic
structure they share, then project selected elements into a blend
space that has emergent properties belonging to neither input.

The technique runs four short steps:

  1. Input A — operator names the first mental space.
  2. Input B — operator names the second (deliberately unrelated).
  3. Blend  — the system calls the Fauconnier-Turner operator at
              :func:`app.creativity.concept_blend.blend_concepts`
              and surfaces the structured result IN the prompt
              shown to the operator (and to the agent team in team
              mode).
  4. Select — operator picks 1-2 emergent properties to develop.

In team mode the multi-agent seed + react rounds run on each
step, so the four creative-crew agents weigh in on input choice +
which projections to keep.
"""
from __future__ import annotations

import logging

from app.brainstorm.techniques.base import (
    LinearTechnique,
    Step,
    TechniqueState,
)

logger = logging.getLogger(__name__)


class ConceptBlendTechnique(LinearTechnique):
    name = "concept-blend"
    title = "Concept Blend (Fauconnier-Turner)"
    description = (
        "Cross-domain creative operator. Pick two unrelated mental "
        "spaces, identify their generic structure, project selected "
        "elements into a blend with emergent properties. Best for "
        "questions where rearranging existing patterns won't work."
    )

    steps = [
        Step(
            step_id="input_a",
            prompt=(
                "Step 1 — Input space A. About: {topic}\n"
                "Name the FIRST mental space (a domain / system / process / "
                "concept) you'll blend. Be specific — describe it in 1-2 "
                "sentences with the structural elements that matter "
                "(agents, relationships, dynamics)."
            ),
            expected_output=(
                "One named mental space with 1-2 sentences of structure"
            ),
        ),
        Step(
            step_id="input_b",
            prompt=(
                "Step 2 — Input space B. About: {topic}\n"
                "Name the SECOND mental space — deliberately FAR from "
                "Input A (different domain, different scale, different "
                "purpose). 1-2 sentences with structural elements. The "
                "blend's value comes from input distance — pick boldly."
            ),
            expected_output=(
                "One named mental space, structurally distant from A"
            ),
        ),
        Step(
            step_id="generate_blend",
            prompt=(
                "Step 3 — Generate blend. About: {topic}\n"
                "The Fauconnier-Turner operator now combines the two "
                "spaces: identify the generic structure they share, "
                "project selected elements from each, and produce a "
                "blend space with emergent properties.\n\n"
                "Read the produced blend below. Note what's NEW in the "
                "blend that wasn't in either input."
            ),
            expected_output=(
                "Acknowledge or push back on the generated blend"
            ),
        ),
        Step(
            step_id="select_projections",
            prompt=(
                "Step 4 — Select projections. About: {topic}\n"
                "From the blend's emergent properties, pick 1-2 that "
                "actually answer the original topic question. For each "
                "selected: state the property and the concrete next "
                "step it suggests."
            ),
            expected_output=(
                "1-2 emergent properties + a concrete next step each"
            ),
        ),
    ]

    def next_prompt(self, state: TechniqueState, topic: str) -> str | None:
        """Override the base linear prompt for the ``generate_blend``
        step: invoke the Fauconnier-Turner operator with the two
        previously-supplied input spaces, then splice the blend
        result into the operator-facing prompt.

        Failure-isolated: if ``blend_concepts`` raises or returns an
        empty result, fall back to the static prompt so the
        technique still walks to completion.
        """
        if state.step_index >= len(self.steps):
            return None
        step = self.steps[state.step_index]
        if step.step_id != "generate_blend":
            return step.prompt.format(topic=topic)
        # We're at step 3 — compute the blend BEFORE rendering the
        # prompt. Inputs come from the two prior responses.
        responses = state.responses or []
        input_a = ""
        input_b = ""
        for r in responses:
            if r.get("step_id") == "input_a":
                input_a = (r.get("response") or "").strip()
            elif r.get("step_id") == "input_b":
                input_b = (r.get("response") or "").strip()
        if not input_a or not input_b:
            # Shouldn't happen in a normal flow, but degrade clean.
            return step.prompt.format(topic=topic)
        blend_block = _render_blend_block(topic, input_a, input_b)
        # Cache the rendered block on ``state.extras`` so the report
        # can reproduce it without re-calling the LLM at summary time.
        state.extras["blend_render"] = blend_block
        return (
            step.prompt.format(topic=topic)
            + "\n\n## Generated blend\n\n"
            + blend_block
        )


def _render_blend_block(topic: str, input_a: str, input_b: str) -> str:
    """Call the Fauconnier-Turner operator and render the result.
    Returns a markdown block ready to splice into the prompt; on
    any failure returns a brief degraded-mode notice so the
    operator knows the blend wasn't computed."""
    try:
        from app.creativity.concept_blend import blend_concepts
    except Exception:
        return "_(concept-blend operator unavailable; proceed manually)_"
    try:
        # Note: topic is implicit context — the blend operator focuses
        # on the structural blend, not the original question. The
        # technique's step 4 ("select_projections") asks the operator
        # to map the blend back to the topic.
        result = blend_concepts(input_a, input_b)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "concept_blend technique: blend_concepts raised: %s",
            exc, exc_info=True,
        )
        return f"_(blend operator failed: {exc})_"
    if result is None:
        return "_(blend operator returned no result)_"
    if getattr(result, "parse_failed", False):
        err = getattr(result, "parse_error", "parse failed")
        return f"_(blend parse failed: {err})_"
    # Defensive attribute access — the BlendResult dataclass shape
    # is documented in app.creativity.concept_blend.
    lines: list[str] = []
    label = getattr(result, "blend_label", "") or ""
    if label:
        lines.append(f"**{label}**")
        lines.append("")
    generic = getattr(result, "generic_structure", "") or ""
    if generic:
        lines.append("### Generic structure")
        lines.append(generic)
        lines.append("")
    description = getattr(result, "blend_description", "") or ""
    if description:
        lines.append("### Blend description")
        lines.append(description)
        lines.append("")
    projections = list(getattr(result, "selected_projections", None) or [])
    if projections:
        lines.append("### Selected projections")
        for p in projections[:6]:
            lines.append(f"- {p}")
        lines.append("")
    emergent = list(getattr(result, "emergent_structure", None) or [])
    if emergent:
        lines.append("### Emergent structure (what's NEW in the blend)")
        for e in emergent[:5]:
            lines.append(f"- {e}")
        lines.append("")
    follow = list(getattr(result, "follow_on_questions", None) or [])
    if follow:
        lines.append("### Follow-on questions")
        for q in follow[:4]:
            lines.append(f"- {q}")
    body = "\n".join(lines).strip()
    return body or "_(blend operator returned an empty result)_"
