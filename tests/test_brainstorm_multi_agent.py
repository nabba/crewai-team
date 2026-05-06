"""Tests for multi-agent (team) brainstorming flow."""

import pytest

from app.brainstorm import facilitator, store
from app.brainstorm.facilitator import FacilitatorError, StepDelivery
from app.brainstorm.multi_agent import (
    DEFAULT_ROSTER,
    AgentResponse,
    resolve_roster,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINSTORM_DIR", str(tmp_path / "brainstorm"))
    monkeypatch.setenv("BRAINSTORM_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("BRAINSTORM_DISABLE_WRITER", "1")
    yield


# ── resolve_roster ────────────────────────────────────────────────────────


def test_resolve_roster_none_is_solo():
    assert resolve_roster(None) == []
    assert resolve_roster(0) == []


def test_resolve_roster_int_picks_first_n():
    assert resolve_roster(1) == DEFAULT_ROSTER[:1]
    assert resolve_roster(3) == DEFAULT_ROSTER[:3]
    assert resolve_roster(4) == DEFAULT_ROSTER


def test_resolve_roster_int_clamps():
    assert resolve_roster(99) == DEFAULT_ROSTER
    assert resolve_roster(-1) == []


def test_resolve_roster_list_of_names():
    assert resolve_roster(["writer", "critic"]) == ["writer", "critic"]


def test_resolve_roster_list_dedupes():
    assert resolve_roster(["writer", "writer", "critic"]) == ["writer", "critic"]


def test_resolve_roster_list_validates():
    with pytest.raises(ValueError):
        resolve_roster(["writer", "evil_clown"])


def test_resolve_roster_bad_type():
    with pytest.raises(TypeError):
        resolve_roster(3.14)


# ── Mock gatherers ────────────────────────────────────────────────────────


def _make_mock_gatherer(prefix: str):
    """Returns a gatherer that produces one deterministic response per role."""
    calls = []

    def gather(*, technique_title, topic, step_prompt, roster, **kwargs):
        calls.append({
            "technique_title": technique_title,
            "topic": topic,
            "step_prompt": step_prompt,
            "roster": list(roster),
            "extra": kwargs,
        })
        return [
            AgentResponse(
                role=r,
                text=f"{prefix}-{r}: idea about {topic}",
                duration_s=0.01,
            )
            for r in roster
        ]

    gather.calls = calls
    return gather


# ── Team-mode lifecycle ──────────────────────────────────────────────────


def test_start_team_mode_attaches_participants():
    seed_g = _make_mock_gatherer("seed")
    session, delivery = facilitator.start(
        "+1user",
        "scamper",
        "improve onboarding",
        with_agents=3,
        seed_gatherer=seed_g,
    )
    assert session.mode == "team"
    assert session.participants == DEFAULT_ROSTER[:3]
    assert isinstance(delivery, StepDelivery)
    assert delivery.prompt is not None
    assert len(delivery.seed) == 3
    # seed_gatherer was called once with the right context
    assert len(seed_g.calls) == 1
    assert seed_g.calls[0]["topic"] == "improve onboarding"
    assert seed_g.calls[0]["roster"] == DEFAULT_ROSTER[:3]


def test_start_solo_mode_does_not_call_seed():
    seed_g = _make_mock_gatherer("seed")
    session, delivery = facilitator.start(
        "+1user", "scamper", "topic", with_agents=None, seed_gatherer=seed_g
    )
    assert session.mode == "solo"
    assert delivery.seed == []
    assert seed_g.calls == []


def test_seed_round_persisted_on_session():
    seed_g = _make_mock_gatherer("seed")
    session, _ = facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=seed_g
    )
    reloaded = store.load(session.session_id)
    assert len(reloaded.agent_rounds) == 1
    assert reloaded.agent_rounds[0]["phase"] == "seed"
    assert reloaded.agent_rounds[0]["step_id"] == "substitute"
    assert len(reloaded.agent_rounds[0]["responses"]) == 2


def test_respond_team_mode_produces_react_and_next_seed():
    seed_g = _make_mock_gatherer("seed")
    react_g = _make_mock_gatherer("react")
    facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=seed_g
    )
    session, delivery, advanced = facilitator.respond(
        "+1user", "my answer", seed_gatherer=seed_g, react_gatherer=react_g
    )
    assert advanced
    assert isinstance(delivery, StepDelivery)
    # react happened for the JUST-completed step
    assert len(delivery.react) == 2
    assert all(r.text.startswith("react-") for r in delivery.react)
    # next_prompt is the next step → seed for it
    assert delivery.prompt is not None
    assert len(delivery.seed) == 2
    assert all(s.text.startswith("seed-") for s in delivery.seed)
    # gatherer call counts: 2 seed (start + after-respond), 1 react
    assert len(seed_g.calls) == 2
    assert len(react_g.calls) == 1


def test_react_gatherer_receives_user_answer_and_peer_seeds():
    seed_g = _make_mock_gatherer("seed")
    react_g = _make_mock_gatherer("react")
    facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=seed_g
    )
    facilitator.respond(
        "+1user", "the answer", seed_gatherer=seed_g, react_gatherer=react_g
    )
    react_call = react_g.calls[0]
    assert react_call["extra"]["user_answer"] == "the answer"
    peer_seeds = react_call["extra"]["peer_seeds"]
    assert len(peer_seeds) == 2
    assert all(isinstance(p, AgentResponse) for p in peer_seeds)


def test_team_session_walks_to_completion():
    seed_g = _make_mock_gatherer("seed")
    react_g = _make_mock_gatherer("react")
    session, delivery = facilitator.start(
        "+1user",
        "reverse",  # 6 steps, smallest team-friendly technique
        "topic",
        with_agents=2,
        seed_gatherer=seed_g,
    )
    safety = 0
    while delivery.prompt is not None:
        _, delivery, advanced = facilitator.respond(
            "+1user", f"answer {safety}",
            seed_gatherer=seed_g, react_gatherer=react_g,
        )
        assert advanced
        safety += 1
        assert safety < 20

    # Total seeds = 6 (one per step including step 0). Total reacts = 6.
    assert len(seed_g.calls) == 6
    assert len(react_g.calls) == 6

    reloaded = store.load(session.session_id)
    seeds = [r for r in reloaded.agent_rounds if r["phase"] == "seed"]
    reacts = [r for r in reloaded.agent_rounds if r["phase"] == "react"]
    assert len(seeds) == 6
    assert len(reacts) == 6


def test_skip_in_team_mode_seeds_next_step():
    seed_g = _make_mock_gatherer("seed")
    facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=seed_g
    )
    _, delivery = facilitator.skip("+1user", seed_gatherer=seed_g)
    assert delivery.prompt is not None
    assert len(delivery.seed) == 2
    # No react: user didn't answer, just skipped.
    assert delivery.react == []
    assert len(seed_g.calls) == 2  # one for start, one for new step


def test_resume_team_mode_regathers_seed():
    seed_g = _make_mock_gatherer("seed")
    facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=seed_g
    )
    facilitator.pause("+1user")
    seed_g.calls.clear()

    result = facilitator.resume("+1user", seed_gatherer=seed_g)
    assert result is not None
    _, delivery = result
    assert delivery.prompt is not None
    assert len(delivery.seed) == 2
    assert len(seed_g.calls) == 1


def test_seed_gatherer_failure_falls_back_to_solo_step():
    """If the seed gatherer crashes, the session continues without agent input."""
    def boom(**kwargs):
        raise RuntimeError("gatherer exploded")

    session, delivery = facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=boom
    )
    assert session.mode == "team"
    assert delivery.prompt is not None
    assert delivery.seed == []
    # Session is still active
    assert store.get_active("+1user") is not None


def test_react_gatherer_failure_falls_back_gracefully():
    seed_g = _make_mock_gatherer("seed")

    def boom(**kwargs):
        raise RuntimeError("react gatherer exploded")

    facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=seed_g
    )
    _, delivery, advanced = facilitator.respond(
        "+1user", "answer", seed_gatherer=seed_g, react_gatherer=boom
    )
    assert advanced
    assert delivery.react == []
    assert delivery.prompt is not None  # session continues


def test_per_agent_error_recorded_but_others_kept():
    """Individual AgentResponse errors should be persisted in the round."""
    def gather(*, roster, **kwargs):
        return [
            AgentResponse(role="researcher", text="ok", duration_s=0.0),
            AgentResponse(role="writer", text="", duration_s=0.0, error="LLM timeout"),
        ]

    session, delivery = facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=gather
    )
    assert len(delivery.seed) == 2
    assert delivery.seed[1].error == "LLM timeout"
    reloaded = store.load(session.session_id)
    rec = reloaded.agent_rounds[0]["responses"]
    assert rec[1]["error"] == "LLM timeout"


# ── Session round-trip with team data ────────────────────────────────────


def test_session_serializes_team_fields():
    seed_g = _make_mock_gatherer("seed")
    session, _ = facilitator.start(
        "+1user", "scamper", "topic", with_agents=2, seed_gatherer=seed_g
    )
    reloaded = store.load(session.session_id)
    assert reloaded.mode == "team"
    assert reloaded.participants == DEFAULT_ROSTER[:2]
    assert len(reloaded.agent_rounds) == 1
