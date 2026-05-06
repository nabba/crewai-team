"""Integration test: `/brainstorm` is claimed by `try_command`."""

import pytest

from tests._v2_shim import install_settings_shim

install_settings_shim()


class _DummyCommander:
    last_crew_used = "dummy"

    def handle(self, text, sender, attachments):
        return "ok"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINSTORM_DIR", str(tmp_path / "brainstorm"))
    monkeypatch.setenv("BRAINSTORM_DISABLE_WRITER", "1")
    monkeypatch.setenv("BRAINSTORM_OUTPUT_DIR", str(tmp_path / "output"))

    import app.conversation_store as cs
    monkeypatch.setattr(cs, "DB_PATH", tmp_path / "conv.db")
    if hasattr(cs._local, "conn"):
        cs._local.conn = None

    from app.agents.commander import commands as cmd_mod
    monkeypatch.setattr(cmd_mod, "_NL_JOBS_FILE", tmp_path / "nl_jobs.json")
    yield


def test_try_command_routes_brainstorm_help():
    from app.agents.commander.commands import try_command

    out = try_command("/brainstorm help", "+15551112222", _DummyCommander())
    assert out is not None
    assert "Brainstorm commands" in out


def test_try_command_routes_brainstorm_menu():
    from app.agents.commander.commands import try_command

    out = try_command("/brainstorm", "+15551112222", _DummyCommander())
    assert out is not None
    assert "scamper" in out


def test_try_command_routes_followup_after_start():
    """After starting a session, the next plain message routes via brainstorm."""
    from app.agents.commander.commands import try_command

    sender = "+15551113333"
    started = try_command(
        "/brainstorm scamper testing topic", sender, _DummyCommander()
    )
    assert started is not None
    # Now a plain message should be claimed by the brainstorm subsystem.
    out = try_command("first response", sender, _DummyCommander())
    assert out is not None
    # ...and a sender with no session should fall through.
    other = try_command("hello world", "+15559999999", _DummyCommander())
    assert other is None


def test_try_command_does_not_intercept_unrelated_messages():
    """Without an active session, plain messages should fall through."""
    from app.agents.commander.commands import try_command

    out = try_command(
        "what's the weather like", "+15554441111", _DummyCommander()
    )
    assert out is None
