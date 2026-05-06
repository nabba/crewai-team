"""Tests for the 7 brainstorming technique state machines."""

import pytest

from app.brainstorm.techniques import get, names, registry
from app.brainstorm.techniques.base import LinearTechnique, TechniqueState


ALL_NAMES = [
    "scamper",
    "six_hats",
    "how_might_we",
    "reverse",
    "crazy_8s",
    "rapid_ideation",
    "starbursting",
]


def test_registry_has_all_seven():
    assert sorted(names()) == sorted(ALL_NAMES)
    reg = registry()
    assert len(reg) == 7
    for n in ALL_NAMES:
        assert n in reg
        assert reg[n].name == n
        assert reg[n].title
        assert reg[n].description


def test_get_unknown_returns_none():
    assert get("not_a_technique") is None


@pytest.mark.parametrize("tech_name", ALL_NAMES)
def test_state_machine_walks_to_completion(tech_name):
    """Every technique should walk linearly through its steps and complete."""
    technique = get(tech_name)
    state = technique.initial_state()
    topic = "Testing topic"

    visited = []
    safety = 0
    while not technique.is_complete(state):
        prompt = technique.next_prompt(state, topic)
        assert prompt is not None
        assert isinstance(prompt, str)
        assert prompt  # non-empty
        # Topic should be substituted into the prompt
        assert topic in prompt or "{topic}" not in prompt
        visited.append(prompt)
        state = technique.record_response(state, f"answer-{len(visited)}", prompt=prompt)
        safety += 1
        if safety > 50:
            pytest.fail(f"{tech_name}: technique didn't terminate within 50 steps")

    # After completion, next_prompt returns None
    assert technique.next_prompt(state, topic) is None

    # Summary contains all responses
    summary = technique.summarize(state, topic)
    assert summary["technique"] == tech_name
    assert summary["topic"] == topic
    assert len(summary["steps"]) == len(visited)


@pytest.mark.parametrize("tech_name", ALL_NAMES)
def test_total_steps_matches_responses(tech_name):
    technique = get(tech_name)
    if not isinstance(technique, LinearTechnique):
        pytest.skip("only linear techniques have a fixed total_steps()")
    total = technique.total_steps()
    assert total > 0
    state = technique.initial_state()
    for _ in range(total):
        p = technique.next_prompt(state, "x")
        assert p is not None
        state = technique.record_response(state, "ans", prompt=p)
    assert technique.is_complete(state)
    assert technique.next_prompt(state, "x") is None


def test_state_round_trip_through_dict():
    """TechniqueState should serialize and deserialize cleanly."""
    state = TechniqueState(
        step_index=2,
        responses=[{"step_id": "a", "prompt": "p", "response": "r", "ts": 123.0}],
        extras={"foo": "bar"},
    )
    state2 = TechniqueState.from_dict(state.to_dict())
    assert state2.step_index == 2
    assert state2.responses == state.responses
    assert state2.extras == {"foo": "bar"}


def test_scamper_step_ids_match_letters():
    """SCAMPER should have a step for each of S/C/A/M/P/E/R."""
    tech = get("scamper")
    assert isinstance(tech, LinearTechnique)
    expected = ["substitute", "combine", "adapt", "modify", "put_to_other_use", "eliminate", "reverse"]
    assert [s.step_id for s in tech.steps] == expected


def test_six_hats_has_blue_open_and_close():
    tech = get("six_hats")
    ids = [s.step_id for s in tech.steps]
    assert ids[0] == "blue_open"
    assert ids[-1] == "blue_close"
    assert "white" in ids
    assert "red" in ids
    assert "black" in ids
    assert "yellow" in ids
    assert "green" in ids


def test_crazy_8s_has_eight_idea_steps():
    tech = get("crazy_8s")
    idea_ids = [s.step_id for s in tech.steps if s.step_id.startswith("idea_")]
    assert len(idea_ids) == 8


def test_starbursting_has_six_question_rays():
    tech = get("starbursting")
    ids = [s.step_id for s in tech.steps]
    for q in ("who", "what", "when", "where", "why", "how"):
        assert q in ids
