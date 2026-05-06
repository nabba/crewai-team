"""End-to-end facilitator tests with a mocked report generator."""

import pytest

from app.brainstorm import facilitator, store
from app.brainstorm.facilitator import FacilitatorError, StepDelivery


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINSTORM_DIR", str(tmp_path / "brainstorm"))
    yield


def _fake_report_generator(session):
    """Returns a deterministic report so we don't need the LLM."""
    return f"# Fake report for {session.session_id}", "/tmp/fake.md"


def test_start_returns_first_prompt():
    session, delivery = facilitator.start("+1user", "scamper", "improve onboarding")
    assert session.session_id
    assert session.technique == "scamper"
    assert session.topic == "improve onboarding"
    assert isinstance(delivery, StepDelivery)
    assert delivery.prompt
    assert "improve onboarding" in delivery.prompt
    assert session.mode == "solo"
    assert session.participants == []
    assert delivery.seed == []
    assert store.get_active("+1user").session_id == session.session_id


def test_start_unknown_technique_raises():
    with pytest.raises(FacilitatorError):
        facilitator.start("+1user", "not_a_thing", "topic")


def test_start_empty_topic_raises():
    with pytest.raises(FacilitatorError):
        facilitator.start("+1user", "scamper", "   ")


def test_start_pauses_existing_active_session():
    s1, _ = facilitator.start("+1user", "scamper", "topic A")
    s2, _ = facilitator.start("+1user", "six_hats", "topic B")
    reloaded_s1 = store.load(s1.session_id)
    assert reloaded_s1.status == "paused"
    assert store.get_active("+1user").session_id == s2.session_id


def test_full_session_walks_to_completion_and_finishes():
    session, delivery = facilitator.start("+1user", "reverse", "morning routine")
    assert delivery.prompt is not None

    safety = 0
    while True:
        next_session, next_delivery, advanced = facilitator.respond(
            "+1user", f"answer {safety}"
        )
        assert advanced
        if next_delivery.prompt is None:
            break
        safety += 1
        assert safety < 30, "loop did not terminate"

    finished = facilitator.finish(
        "+1user", report_generator=_fake_report_generator
    )
    assert finished.status == "complete"
    assert finished.final_report and "Fake report" in finished.final_report
    assert finished.final_report_path
    assert store.get_active("+1user") is None


def test_respond_with_no_active_session_raises():
    with pytest.raises(FacilitatorError):
        facilitator.respond("+1nobody", "answer")


def test_respond_empty_message_does_not_advance():
    session, _ = facilitator.start("+1user", "scamper", "topic")
    initial_idx = session.technique_state.step_index
    s2, delivery, advanced = facilitator.respond("+1user", "   ")
    assert advanced is False
    assert delivery.prompt is not None
    assert s2.technique_state.step_index == initial_idx


def test_skip_advances_state_with_marker():
    session, _ = facilitator.start("+1user", "scamper", "topic")
    s2, delivery = facilitator.skip("+1user")
    assert s2.technique_state.step_index == 1
    assert s2.technique_state.responses[0]["response"] == "(skipped)"
    assert delivery.prompt is not None


def test_pause_and_resume_round_trip():
    session, _ = facilitator.start("+1user", "scamper", "topic")
    facilitator.respond("+1user", "first response")

    paused = facilitator.pause("+1user")
    assert paused.status == "paused"
    assert store.get_active("+1user") is None

    result = facilitator.resume("+1user")
    assert result is not None
    resumed, delivery = result
    assert resumed.session_id == session.session_id
    assert resumed.status == "active"
    assert delivery.prompt
    assert resumed.technique_state.step_index == 1


def test_resume_with_no_paused_returns_none():
    assert facilitator.resume("+1nobody") is None


def test_resume_specific_session_id():
    s1, _ = facilitator.start("+1user", "scamper", "topic A")
    facilitator.pause("+1user")
    s2, _ = facilitator.start("+1user", "six_hats", "topic B")
    facilitator.pause("+1user")

    result = facilitator.resume("+1user", session_id=s1.session_id)
    assert result is not None
    resumed, _ = result
    assert resumed.session_id == s1.session_id


def test_resume_other_users_session_returns_none():
    s1, _ = facilitator.start("+1alice", "scamper", "topic")
    facilitator.pause("+1alice")
    assert facilitator.resume("+1mallory", session_id=s1.session_id) is None


def test_cancel_drops_active_session():
    session, _ = facilitator.start("+1user", "scamper", "topic")
    cancelled = facilitator.cancel("+1user")
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert store.get_active("+1user") is None


def test_cancel_with_no_active_returns_none():
    assert facilitator.cancel("+1nobody") is None


def test_status_string_describes_progress():
    facilitator.start("+1user", "six_hats", "should we ship X")
    s = facilitator.status("+1user")
    assert "six_hats" in s
    assert "0/" in s
    facilitator.respond("+1user", "answer 1")
    s2 = facilitator.status("+1user")
    assert "1/" in s2


def test_status_no_session():
    assert "No active" in facilitator.status("+1nobody")


def test_finish_without_active_falls_back_to_recent():
    session, _ = facilitator.start("+1user", "scamper", "topic")
    facilitator.cancel("+1user")
    out = facilitator.finish(
        "+1user",
        report_generator=_fake_report_generator,
    )
    assert out.session_id == session.session_id
    assert out.status == "complete"


def test_finish_with_no_sessions_at_all_raises():
    with pytest.raises(FacilitatorError):
        facilitator.finish("+1nobody", report_generator=_fake_report_generator)


def test_finish_handles_report_generator_failure_gracefully():
    facilitator.start("+1user", "scamper", "topic")

    def boom(_session):
        raise RuntimeError("LLM down")

    out = facilitator.finish("+1user", report_generator=boom)
    assert out.status == "complete"
    assert out.final_report is None
    assert any(
        "Report generation failed" in t.get("content", "")
        for t in out.transcript
    )
