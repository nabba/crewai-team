"""Tests for app.creativity.concept_blend."""

from __future__ import annotations

import json

from app.creativity.concept_blend import blend_concepts


_GOOD_RESPONSE = json.dumps({
    "input_a": {
        "label": "physical office desktop",
        "salient_elements": ["papers", "folders", "trash can"],
        "salient_relations": ["spatial proximity", "stacking"],
    },
    "input_b": {
        "label": "GUI surface",
        "salient_elements": ["windows", "icons", "menus"],
        "salient_relations": ["containment", "z-order"],
    },
    "generic_structure": "Container with named addressable items.",
    "blend_label": "computer desktop",
    "blend_description": "A virtual surface where files appear as icons and folders as containers.",
    "selected_projections": [
        "from A: trash can",
        "from B: windowing",
    ],
    "emergent_structure": [
        "double-clicking opens with a default app — neither A nor B has this",
    ],
    "follow_on_questions": [
        "What happens when the desktop is full?",
        "Can desktop metaphors scale to phones?",
    ],
})


def _llm_returning(text: str):
    def fake(_system: str, _user: str) -> str:
        return text
    return fake


def test_happy_path_parses_full_structure() -> None:
    out = blend_concepts(
        "physical office desktop",
        "GUI windowing system",
        llm_call=_llm_returning(_GOOD_RESPONSE),
    )
    assert not out.parse_failed
    assert out.input_a.label == "physical office desktop"
    assert out.input_b.label == "GUI surface"
    assert "papers" in out.input_a.salient_elements
    assert out.blend_label == "computer desktop"
    assert "Container with named addressable items" in out.generic_structure
    assert len(out.selected_projections) == 2
    assert len(out.emergent_structure) == 1
    assert len(out.follow_on_questions) == 2


def test_strips_markdown_code_fences() -> None:
    fenced = f"```json\n{_GOOD_RESPONSE}\n```"
    out = blend_concepts("a", "b", llm_call=_llm_returning(fenced))
    assert not out.parse_failed
    assert out.blend_label == "computer desktop"


def test_strips_unfenced_json_with_lang() -> None:
    fenced = f"```json\n{_GOOD_RESPONSE}\n```"
    out = blend_concepts("a", "b", llm_call=_llm_returning(fenced))
    assert not out.parse_failed


def test_malformed_json_returns_failed_parse() -> None:
    out = blend_concepts(
        "a", "b",
        llm_call=_llm_returning("this is not JSON"),
    )
    assert out.parse_failed
    assert "JSON" in out.parse_error


def test_top_level_not_dict_fails() -> None:
    out = blend_concepts(
        "a", "b",
        llm_call=_llm_returning(json.dumps([1, 2, 3])),
    )
    assert out.parse_failed
    assert "not a dict" in out.parse_error


def test_llm_exception_caught() -> None:
    def boom(s: str, u: str) -> str:
        raise RuntimeError("API rate limit")
    out = blend_concepts("a", "b", llm_call=boom)
    assert out.parse_failed
    assert "rate limit" in out.parse_error


def test_empty_input_short_circuits() -> None:
    out = blend_concepts("", "y", llm_call=_llm_returning(_GOOD_RESPONSE))
    assert out.parse_failed
    assert "empty input" in out.parse_error
    out = blend_concepts("x", "  ", llm_call=_llm_returning(_GOOD_RESPONSE))
    assert out.parse_failed


def test_partial_structure_does_not_crash() -> None:
    """Missing optional fields default to empty rather than failing."""
    minimal = json.dumps({
        "input_a": {"label": "A"},
        "input_b": {"label": "B"},
        "generic_structure": "shared structure",
        "blend_label": "blend",
        "blend_description": "desc",
    })
    out = blend_concepts("a", "b", llm_call=_llm_returning(minimal))
    assert not out.parse_failed
    assert out.input_a.label == "A"
    assert out.input_b.salient_elements == []
    assert out.emergent_structure == []
    assert out.follow_on_questions == []


def test_to_dict_round_trip() -> None:
    out = blend_concepts(
        "a", "b", llm_call=_llm_returning(_GOOD_RESPONSE),
    )
    d = out.to_dict()
    assert d["input_a"]["label"] == "physical office desktop"
    assert d["blend_label"] == "computer desktop"
    assert d["parse_failed"] is False
