"""Signal slash-command + active-session message routing tests."""

import pytest

from app.brainstorm import facilitator
from app.brainstorm.signal_handler import try_handle


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINSTORM_DIR", str(tmp_path / "brainstorm"))
    # Disable the Writer-agent path so finish() never tries to import CrewAI
    # / hit the LLM during these tests.
    monkeypatch.setenv("BRAINSTORM_DISABLE_WRITER", "1")
    monkeypatch.setenv("BRAINSTORM_OUTPUT_DIR", str(tmp_path / "output"))
    yield


def test_no_match_returns_none():
    assert try_handle("just a regular question", "+1user") is None
    assert try_handle("", "+1user") is None
    assert try_handle(None, "+1user") is None


def test_brainstorm_alone_shows_menu():
    out = try_handle("/brainstorm", "+1user")
    assert out is not None
    assert "scamper" in out.lower()
    assert "six_hats" in out.lower()
    assert "Available techniques" in out


def test_brainstorm_help():
    out = try_handle("/brainstorm help", "+1user")
    assert out is not None
    assert "Brainstorm commands" in out
    assert "/brainstorm" in out


def test_brainstorm_unknown_verb():
    out = try_handle("/brainstorm wat", "+1user")
    assert out is not None
    assert "Unknown" in out


def test_brainstorm_technique_no_topic_prompts_for_one():
    out = try_handle("/brainstorm six_hats", "+1user")
    assert out is not None
    assert "topic" in out.lower()


def test_start_with_technique_and_topic():
    out = try_handle("/brainstorm six_hats Should we ship feature X", "+1user")
    assert out is not None
    assert "Six Thinking Hats" in out
    assert "Should we ship feature X" in out
    # The first hat (BLUE) should appear in the prompt
    assert "BLUE" in out


def test_followup_message_routes_to_facilitator():
    """After /brainstorm starts a session, a plain message is treated as the answer."""
    try_handle("/brainstorm scamper improve onboarding", "+1user")
    out = try_handle("answer to substitute step", "+1user")
    assert out is not None
    # Should now show the next step (Combine = "C —")
    assert "C —" in out or "Combine" in out


def test_status_command():
    try_handle("/brainstorm scamper a topic", "+1user")
    out = try_handle("/brainstorm status", "+1user")
    assert out is not None
    assert "scamper" in out
    assert "0/" in out  # no answers yet


def test_pause_and_resume_via_signal():
    try_handle("/brainstorm scamper a topic", "+1user")
    try_handle("first answer", "+1user")
    paused = try_handle("/brainstorm pause", "+1user")
    assert paused is not None
    assert "paused" in paused.lower()

    # No active session — a plain message should NOT be claimed
    fall_through = try_handle("just chatting", "+1user")
    assert fall_through is None

    resumed = try_handle("/brainstorm resume", "+1user")
    assert resumed is not None
    assert "Resumed" in resumed


def test_skip_advances():
    try_handle("/brainstorm scamper a topic", "+1user")
    out = try_handle("/brainstorm skip", "+1user")
    assert out is not None
    assert "Skipped" in out or "Combine" in out or "C —" in out


def test_cancel_drops_session():
    try_handle("/brainstorm scamper a topic", "+1user")
    out = try_handle("/brainstorm cancel", "+1user")
    assert out is not None
    assert "cancelled" in out.lower()
    # Now plain messages should fall through
    assert try_handle("hi", "+1user") is None


def test_finish_produces_report_via_fallback_path():
    # Drive a quick reverse-brainstorm to completion (6 steps)
    try_handle("/brainstorm reverse a topic", "+1user")
    for i in range(6):
        try_handle(f"answer {i}", "+1user")
    out = try_handle("/brainstorm finish", "+1user")
    assert out is not None
    assert "Report saved to" in out or "no report" in out.lower()


def test_list_command_empty():
    out = try_handle("/brainstorm list", "+1user")
    assert out is not None
    assert "No brainstorm sessions" in out


def test_list_command_with_sessions():
    try_handle("/brainstorm scamper topic A", "+1user")
    facilitator.pause("+1user")
    out = try_handle("/brainstorm list", "+1user")
    assert out is not None
    assert "scamper" in out
    assert "topic A" in out


def test_sender_isolation():
    """Two senders should have independent active sessions."""
    try_handle("/brainstorm scamper alice topic", "+1alice")
    try_handle("/brainstorm six_hats bob topic", "+1bob")
    s_alice = facilitator.status("+1alice")
    s_bob = facilitator.status("+1bob")
    assert "scamper" in s_alice
    assert "six_hats" in s_bob


def test_empty_response_during_session_is_handled():
    try_handle("/brainstorm scamper a topic", "+1user")
    out = try_handle("   ", "+1user")
    # Whitespace-only message rejected at try_handle's strip() check → None.
    assert out is None


# ── Team-mode parser tests ───────────────────────────────────────────────


def test_parse_with_n_agents():
    from app.brainstorm.signal_handler import _parse_start

    technique, topic, n = _parse_start(
        "scamper with 3 agents Improve onboarding flow"
    )
    assert technique == "scamper"
    assert topic == "Improve onboarding flow"
    assert n == 3


def test_parse_with_agents_default_n():
    from app.brainstorm.signal_handler import _parse_start

    technique, topic, n = _parse_start(
        "six_hats with agents Should we ship feature X"
    )
    assert technique == "six_hats"
    assert topic == "Should we ship feature X"
    assert n == 4


def test_parse_solo():
    from app.brainstorm.signal_handler import _parse_start

    technique, topic, n = _parse_start("scamper Improve onboarding flow")
    assert technique == "scamper"
    assert topic == "Improve onboarding flow"
    assert n is None


def test_parse_unknown_technique_returns_none():
    from app.brainstorm.signal_handler import _parse_start

    assert _parse_start("not_a_technique with 3 agents Topic") is None


def test_parse_with_agents_anywhere_in_string():
    """The parser should handle 'with agents' appearing mid-topic too."""
    from app.brainstorm.signal_handler import _parse_start

    technique, topic, n = _parse_start(
        "scamper Should we add a feature with 2 agents next quarter"
    )
    assert technique == "scamper"
    # 'with 2 agents' is consumed; topic is what's left.
    assert "Should we add a feature" in topic
    assert "next quarter" in topic
    assert n == 2


def test_team_mode_command_starts_team_session(monkeypatch):
    """Verify the slash-command path actually invokes team mode."""
    # Stub the gatherers so we don't fire real LLM calls.
    fake_seeds = []
    fake_reacts = []

    def fake_seed_gather(*, roster, **kwargs):
        from app.brainstorm.multi_agent import AgentResponse
        out = [AgentResponse(role=r, text=f"seed-{r}", duration_s=0.0) for r in roster]
        fake_seeds.append(out)
        return out

    def fake_react_gather(*, roster, **kwargs):
        from app.brainstorm.multi_agent import AgentResponse
        out = [AgentResponse(role=r, text=f"react-{r}", duration_s=0.0) for r in roster]
        fake_reacts.append(out)
        return out

    monkeypatch.setattr("app.brainstorm.multi_agent.gather_seed", fake_seed_gather)
    monkeypatch.setattr("app.brainstorm.multi_agent.gather_react", fake_react_gather)
    # Re-import in facilitator's namespace too (it imports by name)
    import app.brainstorm.facilitator as fac
    monkeypatch.setattr(fac, "_default_gather_seed", fake_seed_gather)
    monkeypatch.setattr(fac, "_default_gather_react", fake_react_gather)

    out = try_handle("/brainstorm scamper with 2 agents improve docs", "+1team")
    assert out is not None
    assert "Team mode" in out
    assert "researcher" in out and "writer" in out
    assert "AGENTS SEED" in out
    assert "seed-researcher" in out
    assert len(fake_seeds) == 1


def test_team_mode_followup_includes_react_block(monkeypatch):
    from app.brainstorm.multi_agent import AgentResponse

    def fake_seed_gather(*, roster, **kwargs):
        return [AgentResponse(role=r, text=f"seed-{r}", duration_s=0.0) for r in roster]

    def fake_react_gather(*, roster, **kwargs):
        return [AgentResponse(role=r, text=f"react-{r}", duration_s=0.0) for r in roster]

    import app.brainstorm.facilitator as fac
    monkeypatch.setattr(fac, "_default_gather_seed", fake_seed_gather)
    monkeypatch.setattr(fac, "_default_gather_react", fake_react_gather)

    try_handle("/brainstorm scamper with 2 agents topic", "+1user")
    out = try_handle("my answer", "+1user")
    assert out is not None
    assert "AGENTS REACT" in out
    assert "react-researcher" in out
    assert "AGENTS SEED" in out  # seed for next step
    assert "seed-researcher" in out


def test_team_mode_status_string_shows_participants(monkeypatch):
    from app.brainstorm.multi_agent import AgentResponse

    def fake_seed(*, roster, **kwargs):
        return [AgentResponse(role=r, text="seed", duration_s=0.0) for r in roster]

    import app.brainstorm.facilitator as fac
    monkeypatch.setattr(fac, "_default_gather_seed", fake_seed)

    try_handle("/brainstorm scamper with 2 agents topic", "+1user")
    status = try_handle("/brainstorm status", "+1user")
    assert status is not None
    assert "team:" in status
    assert "researcher" in status


# ── Escape-command allowlist (regression for the workspace-switch hijack) ──
#
# Pre-fix, an active brainstorm session swallowed every subsequent message
# from that sender — including `switch workspace` commands. The user got
# `(empty answer — please respond)` repeatedly with no escape route except
# `/brainstorm cancel` (which they often forgot they had to type). Now
# try_handle returns None for a small allowlist of commander commands so
# they fall through to the normal command parser.


def _start_session(sender: str = "+1escape") -> None:
    """Helper — open an active brainstorm session for a sender."""
    out = try_handle("/brainstorm scamper escape-test topic", sender)
    assert out is not None  # session started


def test_escape_workspace_switch_falls_through():
    """The original bug. With an active session, 'switch to workspace X'
    must NOT be captured by brainstorm — it has to fall through so the
    commander's project switcher can handle it."""
    _start_session()
    for phrasing in (
        "switch to workspace eesti mets",
        "Switch to workspace eesti mets",
        "switch workspace to eesti mets",
        "switch project to PLG",
        "switch to project default",
        "workspace switch eesti mets",
        "project switch PLG",
    ):
        assert try_handle(phrasing, "+1escape") is None, (
            f"escape allowlist should let {phrasing!r} fall through"
        )


def test_escape_status_questions_fall_through():
    _start_session()
    for phrasing in (
        "workspace status",
        "project status",
        "where am I",
        "where am I?",
        "what is the current workspace",
        "what's the active project",
        "current project",
        "active workspace",
        "which project am I on",
        "what workspace am I working on",
    ):
        assert try_handle(phrasing, "+1escape") is None, (
            f"escape allowlist should let {phrasing!r} fall through"
        )


def test_escape_slash_commands_fall_through():
    """Any slash command (other than /brainstorm) must fall through.
    /brainstorm is handled separately at the top of try_handle."""
    _start_session()
    for cmd in (
        "/help", "/status", "/skill list", "/cancel", "/exit",
    ):
        assert try_handle(cmd, "+1escape") is None, (
            f"escape allowlist should let slash-command {cmd!r} fall through"
        )


def test_escape_single_token_commands_fall_through():
    _start_session()
    for cmd in ("help", "?", "status", "cancel", "exit", "stop", "workspaces"):
        assert try_handle(cmd, "+1escape") is None, (
            f"escape allowlist should let bare token {cmd!r} fall through"
        )


def test_brainstorm_response_still_captured_for_normal_text():
    """Sanity: a plain answer to a brainstorm prompt is still routed
    into the session — the allowlist only carves out commander commands."""
    _start_session()
    out = try_handle("Substitute the welcome flow with a guided tour", "+1escape")
    assert out is not None
    # Either the response advanced the session OR it was rejected as too short;
    # either way, brainstorm captured it (didn't return None).


def test_brainstorm_slash_commands_still_route_to_brainstorm():
    """`/brainstorm cancel`, `/brainstorm status`, etc. continue to go
    INTO brainstorm — they're not in the escape allowlist; they're
    matched by the earlier `/brainstorm`-prefix branch."""
    _start_session()
    out = try_handle("/brainstorm status", "+1escape")
    assert out is not None
    assert "scamper" in out.lower() or "step" in out.lower()


def test_no_session_means_no_special_routing():
    """Without an active session, the escape allowlist isn't consulted —
    everything falls through (handled by the no-session branch returning None)."""
    # No prior /brainstorm start
    for phrasing in (
        "switch to workspace eesti mets",
        "what is the current project",
        "/help",
        "regular user message",
    ):
        assert try_handle(phrasing, "+1nosession") is None


def test_looks_like_escape_command_unit():
    """Direct unit test on the predicate, without setting up a session."""
    from app.brainstorm.signal_handler import _looks_like_escape_command

    assert _looks_like_escape_command("switch to workspace eesti mets")
    assert _looks_like_escape_command("Switch to workspace eesti mets")
    assert _looks_like_escape_command("switch project to PLG")
    assert _looks_like_escape_command("/help")
    assert _looks_like_escape_command("/status")
    assert _looks_like_escape_command("workspace status")
    assert _looks_like_escape_command("where am I?")
    assert _looks_like_escape_command("help")
    assert _looks_like_escape_command("?")

    # Non-commands stay non-escape
    assert not _looks_like_escape_command("Improve onboarding flow")
    assert not _looks_like_escape_command("substitute the login screen")
    assert not _looks_like_escape_command("My idea is to add a button")
    assert not _looks_like_escape_command(
        "what about adding a wizard?",  # NOT a status/switch question
    )
