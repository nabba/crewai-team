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


# ── Degenerate-response detector ──────────────────────────────────────────


class TestIsDegenerate:
    """Pure validation — no I/O. Catches the failure modes observed in
    real team-mode runs (CrewAI scaffolding echo, repetition loops,
    punctuation-heavy garbage)."""

    def _check(self, text):
        from app.brainstorm.multi_agent import _is_degenerate
        return _is_degenerate(text)

    def test_empty_is_not_degenerate(self):
        assert self._check("") == (False, "")
        assert self._check("   ") == (False, "")

    def test_short_response_passes_through(self):
        # Below the 30-char threshold we don't make a call either way.
        ok, _ = self._check("1. fast\n2. cheap")
        assert ok is False

    def test_normal_idea_list_is_clean(self):
        text = (
            "1. Substitute the manual review with an automated linter that\n"
            "   catches the top three failure patterns.\n"
            "2. Replace the email digest with a Slack thread anchored to\n"
            "   the original incident channel.\n"
            "3. Swap the JIRA ticket template for a one-line rotation note."
        )
        ok, reason = self._check(text)
        assert ok is False, f"got reason: {reason}"

    def test_scaffolding_echo_caught(self):
        # The exact failure mode from the user's screenshots.
        text = (
            "MUST return the actual content, not a summary. "
            "MUST return the actual content, not a summary. "
            "MUST return the actual content, not a summary. "
            "MUST return the actual content, not a summary."
        )
        ok, reason = self._check(text)
        assert ok is True
        assert "scaffolding" in reason.lower()

    def test_expected_criteria_echo_caught(self):
        text = (
            "This is the expected criteria for your final answer: "
            "This is the expected criteria for your final answer. "
            "This is the expected criteria for your final answer."
        )
        ok, reason = self._check(text)
        assert ok is True
        assert "scaffolding" in reason.lower()

    def test_repetition_loop_caught(self):
        text = "\n".join(["the same line"] * 8)
        ok, reason = self._check(text)
        assert ok is True
        assert "repetition" in reason.lower()

    def test_heavy_punctuation_caught(self):
        # >150 chars, ~zero alphanumeric — broken JSON / brace soup case.
        text = '")"")")")"")"")")")")"")"")")")"")"")")")"")")")"")"")")")"")"")")")"")"")")")"")"")")")"")"")")")"")"")")")"")"")")")"")"")")")"")"")")")"")"")")")"")")"")")"")")"")")"")")"")")")")"")"")"")"")"' * 2
        assert len(text) > 150
        ok, reason = self._check(text)
        assert ok is True, f"expected degenerate; got reason={reason!r}"
        assert "non-text" in reason.lower()

    def test_low_diversity_caught(self):
        # 50+ words, <15% unique
        text = " ".join(["foo bar baz"] * 50)  # 150 words, 3 unique = 2%
        ok, reason = self._check(text)
        assert ok is True
        assert "diversity" in reason.lower()

    def test_long_legitimate_response_not_caught(self):
        # ~600 chars of varied prose; legit long answer.
        text = (
            "1. The onboarding survey is currently five steps long but the "
            "drop-off concentrates between steps two and three, so a likely "
            "high-leverage substitution is replacing the multi-page wizard "
            "with a single scrolling form that reorders fields by inferred "
            "intent.\n"
            "2. A second angle: instead of asking the user to type their "
            "company name, fetch it from the email domain via Clearbit and "
            "let the user correct it in line. Reduces typing without losing "
            "fidelity for atypical addresses.\n"
            "3. Reframe step three's role-picker as an inferred default with "
            "an 'edit' affordance, the way Linear does for new-issue "
            "assignment."
        )
        ok, reason = self._check(text)
        assert ok is False, f"caught a clean response: {reason}"


def test_run_one_agent_demotes_degenerate_output(monkeypatch):
    """End-to-end: a degenerate kickoff result should produce an errored
    AgentResponse, not a populated text field."""
    import app.brainstorm.multi_agent as ma

    # Stub the agent build to avoid touching real LLM/factory code.
    monkeypatch.setattr(
        ma, "_build_creative_agent", lambda role, **_: object()
    )

    # Patch crewai.Crew.kickoff via a fake Crew class.
    class _FakeKickoff:
        def __init__(self, text):
            self._text = text

        def __str__(self):
            return self._text

    class _FakeCrew:
        def __init__(self, *, agents, tasks, process, verbose):
            pass

        def kickoff(self):
            # Return the exact failure mode from the screenshots.
            return _FakeKickoff(
                "MUST return the actual content, not a summary. "
                "MUST return the actual content, not a summary. "
                "MUST return the actual content, not a summary. "
                "MUST return the actual content, not a summary."
            )

    monkeypatch.setattr("crewai.Crew", _FakeCrew, raising=False)
    # Task and Process are imported by name inside the function; provide
    # benign stand-ins.
    import crewai

    monkeypatch.setattr(
        crewai, "Task", lambda **kw: object(), raising=False
    )

    resp = ma._run_one_agent(
        "researcher",
        "irrelevant",
        expected_output="ignored",
        phase="diverge",
    )
    assert resp.role == "researcher"
    assert resp.text == ""
    assert resp.error is not None
    assert "degenerate" in resp.error.lower()


def test_run_one_agent_keeps_clean_output(monkeypatch):
    """Counterpart: a normal kickoff result should pass through unchanged."""
    import app.brainstorm.multi_agent as ma

    monkeypatch.setattr(
        ma, "_build_creative_agent", lambda role, **_: object()
    )

    class _FakeKickoff:
        def __str__(self):
            return (
                "1. Replace the JIRA template with a one-line rotation note.\n"
                "2. Swap the email digest for a Slack thread on the incident "
                "channel.\n"
                "3. Combine the post-mortem and retro into one weekly meeting."
            )

    class _FakeCrew:
        def __init__(self, **kw):
            pass

        def kickoff(self):
            return _FakeKickoff()

    monkeypatch.setattr("crewai.Crew", _FakeCrew, raising=False)
    import crewai

    monkeypatch.setattr(crewai, "Task", lambda **kw: object(), raising=False)

    resp = ma._run_one_agent(
        "writer", "irrelevant", expected_output="x", phase="discuss"
    )
    assert resp.error is None
    assert "Replace the JIRA" in resp.text


# ── Tier configuration ────────────────────────────────────────────────────


def test_default_researcher_tier_is_budget():
    """Brainstorm overrides creative_crew's `local` for the researcher
    so degenerate Ollama output doesn't reach the user. See multi_agent's
    _TIER_BY_ROLE comment."""
    from app.brainstorm.multi_agent import _tier_for, _TIER_BY_ROLE
    assert _TIER_BY_ROLE["researcher"] == "budget"
    assert _tier_for("researcher") == "budget"


def test_tier_env_override(monkeypatch):
    """BRAINSTORM_TIER_<ROLE> rolls back individual tiers for ops cases
    where someone wants the original creative_crew behaviour."""
    from app.brainstorm.multi_agent import _tier_for

    monkeypatch.setenv("BRAINSTORM_TIER_RESEARCHER", "local")
    assert _tier_for("researcher") == "local"
    monkeypatch.setenv("BRAINSTORM_TIER_CRITIC", "mid")
    assert _tier_for("critic") == "mid"


def test_tier_env_override_unaffected_roles_use_default(monkeypatch):
    from app.brainstorm.multi_agent import _tier_for

    monkeypatch.setenv("BRAINSTORM_TIER_RESEARCHER", "local")
    # writer / coder / critic should NOT pick up the researcher override
    assert _tier_for("writer") == "mid"
    assert _tier_for("coder") == "budget"
    assert _tier_for("critic") == "premium"
