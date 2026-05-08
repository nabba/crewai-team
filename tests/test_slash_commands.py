"""
test_slash_commands — /help and /status mobile-surface commands.

Tests the dispatch logic in `try_command` and the helpers behind it. The
status helper queries other subsystems; we mock them out so the test runs
hermetically and doesn't depend on a live scheduler / structured log.
"""
from __future__ import annotations

import pytest


# Patch the brainstorm router (imported at the top of try_command) so it
# never claims our test inputs as a brainstorm message.
@pytest.fixture(autouse=True)
def _disable_brainstorm(monkeypatch):
    import app.brainstorm.signal_handler as bsh
    monkeypatch.setattr(bsh, "try_handle", lambda *a, **kw: None)


def test_help_command_returns_string():
    from app.agents.commander.commands import try_command
    out = try_command("/help", sender="+test", commander=None)
    assert isinstance(out, str)
    assert "/help" in out
    assert "/status" in out
    assert "voice mode" in out.lower() or "voice notes" in out.lower()


def test_help_aliases():
    from app.agents.commander.commands import try_command
    for alias in ("help", "?", "/help"):
        out = try_command(alias, sender="+test", commander=None)
        assert isinstance(out, str)
        assert "Signal commands" in out or "/help" in out


def test_status_returns_voice_mode():
    from app.agents.commander.commands import try_command
    out = try_command("/status", sender="+test", commander=None)
    assert isinstance(out, str)
    assert "voice:" in out


def test_status_does_not_claim_unrelated_input():
    from app.agents.commander.commands import try_command
    out = try_command("hello there", sender="+test", commander=None)
    assert out is None


def test_status_includes_push_when_configured(monkeypatch):
    """When VAPID keys are set, the status block surfaces device count."""
    from app.web_push import sender as wp_sender
    monkeypatch.setattr(wp_sender, "is_configured", lambda: True)
    monkeypatch.setattr(
        "app.web_push.list_subscriptions",
        lambda: [{"endpoint": "x"}],
    )
    from app.agents.commander.commands import try_command
    out = try_command("/status", sender="+test", commander=None)
    assert "push" in out.lower()
