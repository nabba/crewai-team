"""Fauconnier–Turner concept-blending operator.

Concept blending (Fauconnier & Turner, *The Way We Think*) is a
specific creativity operator. Two input mental spaces are projected
into a *blend space* — the blend inherits selected elements from
each input plus emergent structure neither input has alone.

Examples (canonical from the literature):

  "computer desktop"
    input A: physical office desktop (papers, folders, trash can)
    input B: GUI surface (windowing system primitives)
    generic structure: container with named addressable items
    blend: a virtual surface where users manipulate file-icons
           with the affordances of physical desktop work

  "complaint department" (Goffman analysis)
    input A: bureaucratic counter
    input B: ritualized public apology
    generic structure: triadic exchange (claimant / agent / authority)
    blend: a face-saving complaint resolution channel

This module wraps that operator as an injectable LLM call. The
prompt template forces the LLM to enumerate:

  1. Input space A: salient elements + relations
  2. Input space B: salient elements + relations
  3. Generic structure: what A and B share at the abstract level
  4. Blend: selectively-projected elements + emergent structure
  5. Optionally: 2-3 follow-on questions the blend opens up

Output is parsed into a :class:`BlendResult` for programmatic
consumption (brainstorm subsystem, reverie engine).

LLM call is injectable so the module is testable without
credentials. Production callers leave ``llm_call=None`` and the
module resolves a Tier-1 research model via the standard factory.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputSpace:
    """One of the two input mental spaces fed to the blend."""

    label: str
    salient_elements: list[str] = field(default_factory=list)
    salient_relations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BlendResult:
    """The product of one concept-blending operation."""

    input_a: InputSpace
    input_b: InputSpace
    generic_structure: str
    blend_label: str
    blend_description: str
    selected_projections: list[str] = field(default_factory=list)
    emergent_structure: list[str] = field(default_factory=list)
    follow_on_questions: list[str] = field(default_factory=list)
    raw_response: str = ""
    parse_failed: bool = False
    parse_error: str = ""

    def to_dict(self) -> dict:
        return {
            "input_a": asdict(self.input_a),
            "input_b": asdict(self.input_b),
            "generic_structure": self.generic_structure,
            "blend_label": self.blend_label,
            "blend_description": self.blend_description,
            "selected_projections": list(self.selected_projections),
            "emergent_structure": list(self.emergent_structure),
            "follow_on_questions": list(self.follow_on_questions),
            "parse_failed": self.parse_failed,
            "parse_error": self.parse_error,
        }


LlmCall = Callable[[str, str], str]
"""(system_prompt, user_prompt) -> raw response string."""


_SYSTEM_PROMPT = """\
You are performing a single Fauconnier–Turner concept blend. Two
input mental spaces are given; produce the blend.

Output STRICTLY in this JSON shape — no other text, no preamble:

{
  "input_a": {
    "label": "<short label>",
    "salient_elements": ["element1", "element2", ...],
    "salient_relations": ["relation1", "relation2", ...]
  },
  "input_b": { ...same shape... },
  "generic_structure": "<one paragraph naming what A and B share at the abstract level>",
  "blend_label": "<short label for the blend, 2-6 words>",
  "blend_description": "<two paragraphs describing the blend space>",
  "selected_projections": [
    "from A: <element/relation>",
    "from B: <element/relation>",
    ...
  ],
  "emergent_structure": [
    "<one structure that exists in the blend but in NEITHER input>",
    ...
  ],
  "follow_on_questions": [
    "<a generative question the blend opens up>",
    "<another>",
    "<another>"
  ]
}

Constraints:
  - Salience matters: name 3-7 elements per input, 2-5 relations.
  - The generic structure must be something A and B GENUINELY share,
    not a tautological "they're both things."
  - Selected projections are the elements that make it INTO the
    blend — not every element from each input is selected.
  - Emergent structure is the payoff: what the blend has that
    neither input alone has. List at least one.
  - Output ONLY the JSON. No code fences, no commentary.
"""


def _build_user_prompt(input_a_text: str, input_b_text: str) -> str:
    return (
        f"Input space A:\n{input_a_text}\n\n"
        f"Input space B:\n{input_b_text}\n\n"
        f"Produce the blend."
    )


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Defensive: some LLMs ignore the no-code-fence instruction."""
    text = text.strip()
    m = _JSON_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def _parse_response(raw: str) -> tuple[dict, bool, str]:
    """Returns (parsed_dict, failed, error)."""
    cleaned = _strip_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {}, True, f"JSON decode failed: {exc}"
    if not isinstance(parsed, dict):
        return {}, True, "top-level not a dict"
    return parsed, False, ""


def blend_concepts(
    input_a: str,
    input_b: str,
    *,
    llm_call: LlmCall | None = None,
) -> BlendResult:
    """Run one concept blend. Returns :class:`BlendResult`.

    On any failure (LLM exception, JSON parse failure, malformed
    structure) returns a result with ``parse_failed=True`` and
    a populated ``parse_error``. The brainstorm consumer can decide
    whether to retry or skip.
    """
    if not input_a.strip() or not input_b.strip():
        return BlendResult(
            input_a=InputSpace(label=""),
            input_b=InputSpace(label=""),
            generic_structure="",
            blend_label="",
            blend_description="",
            parse_failed=True,
            parse_error="empty input",
        )

    call = llm_call or _default_llm_call
    user_prompt = _build_user_prompt(input_a, input_b)
    try:
        raw = call(_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("concept_blend: LLM call raised: %s", exc)
        return BlendResult(
            input_a=InputSpace(label=""),
            input_b=InputSpace(label=""),
            generic_structure="",
            blend_label="",
            blend_description="",
            raw_response="",
            parse_failed=True,
            parse_error=f"LLM raised: {exc}",
        )

    parsed, failed, error = _parse_response(raw)
    if failed:
        return BlendResult(
            input_a=InputSpace(label=""),
            input_b=InputSpace(label=""),
            generic_structure="",
            blend_label="",
            blend_description="",
            raw_response=raw,
            parse_failed=True,
            parse_error=error,
        )

    def _input_space(d: dict) -> InputSpace:
        if not isinstance(d, dict):
            return InputSpace(label="")
        return InputSpace(
            label=str(d.get("label", "")),
            salient_elements=[str(x) for x in d.get("salient_elements", [])],
            salient_relations=[str(x) for x in d.get("salient_relations", [])],
        )

    return BlendResult(
        input_a=_input_space(parsed.get("input_a", {})),
        input_b=_input_space(parsed.get("input_b", {})),
        generic_structure=str(parsed.get("generic_structure", "")),
        blend_label=str(parsed.get("blend_label", "")),
        blend_description=str(parsed.get("blend_description", "")),
        selected_projections=[
            str(x) for x in parsed.get("selected_projections", [])
        ],
        emergent_structure=[
            str(x) for x in parsed.get("emergent_structure", [])
        ],
        follow_on_questions=[
            str(x) for x in parsed.get("follow_on_questions", [])
        ],
        raw_response=raw,
    )


def _default_llm_call(system: str, user: str) -> str:
    from app.llm_factory import create_specialist_llm
    llm = create_specialist_llm(role="research", max_tokens=2048)
    response = llm.call(messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    return str(response)
