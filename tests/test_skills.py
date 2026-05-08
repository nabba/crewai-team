"""
test_skills — registry + runner + slash command tests.

The runner uses a stub commander (`.handle(task, sender) -> str`) so we
exercise the dispatch path without touching the real CrewAI machinery.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    from app.skills import registry as r
    monkeypatch.setattr(r, "_STORE_PATH", tmp_path / "skills.json")
    yield


# Disable the brainstorm router so it never claims our test inputs.
@pytest.fixture(autouse=True)
def _disable_brainstorm(monkeypatch):
    import app.brainstorm.signal_handler as bsh
    monkeypatch.setattr(bsh, "try_handle", lambda *a, **kw: None)


# ── Registry ──────────────────────────────────────────────────────────────

def test_save_skill_round_trip():
    from app.skills import save_skill, get_skill, list_skills
    save_skill(
        name="weekly status",
        task_template="Summarize my Q{quarter} status with focus on {topic}.",
        description="Personal weekly summary.",
    )
    rows = list_skills()
    assert len(rows) == 1
    s = get_skill("weekly status")
    assert s is not None
    assert s.name == "weekly status"
    assert s.args_schema == ["quarter", "topic"]
    assert s.task_template.startswith("Summarize my Q{quarter}")


def test_save_skill_normalises_name():
    from app.skills import save_skill, get_skill
    save_skill(name="  Weekly  Status  ", task_template="Task.")
    s = get_skill("weekly status")
    assert s is not None
    s2 = get_skill("WEEKLY STATUS")
    assert s2 is not None and s2.name == "weekly status"


def test_save_skill_rejects_empty():
    from app.skills import save_skill
    with pytest.raises(ValueError):
        save_skill(name="", task_template="anything")
    with pytest.raises(ValueError):
        save_skill(name="x", task_template="  ")


def test_overwrite_preserves_run_counters():
    from app.skills import save_skill, record_run_result, get_skill
    save_skill(name="foo", task_template="t1")
    record_run_result("foo", success=True)
    record_run_result("foo", success=False)
    save_skill(name="foo", task_template="t2-rewrite")
    s = get_skill("foo")
    assert s.run_count == 2
    assert s.success_count == 1
    assert s.task_template == "t2-rewrite"


def test_delete_skill():
    from app.skills import save_skill, delete_skill, list_skills
    save_skill(name="foo", task_template="t")
    assert delete_skill("foo") is True
    assert delete_skill("foo") is False
    assert list_skills() == []


def test_extract_placeholders_dedup():
    from app.skills import extract_placeholders
    assert extract_placeholders("hello {name}, {name} again, {place}") == ["name", "place"]


# ── Runner ────────────────────────────────────────────────────────────────

class _StubCommander:
    def __init__(self, response: str = "ok"):
        self.calls: list[tuple[str, str]] = []
        self.response = response

    def handle(self, task: str, sender: str) -> str:
        self.calls.append((task, sender))
        return self.response


def test_expand_substitutes_placeholders():
    from app.skills import expand
    out = expand("Hi {name}, see Q{q}.", {"name": "Andrus", "q": "2"})
    assert out == "Hi Andrus, see Q2."


def test_expand_missing_arg_raises():
    from app.skills import expand
    with pytest.raises(ValueError) as exc:
        expand("Hi {missing}", {})
    assert "missing" in str(exc.value)


def test_run_skill_dispatches_and_records_success():
    from app.skills import save_skill, run_skill, get_skill
    save_skill(name="greet", task_template="Hello {who}!")
    cmd = _StubCommander(response="Done.")
    out = run_skill("greet", {"who": "world"}, "+test", cmd)
    assert out == "Done."
    assert cmd.calls == [("Hello world!", "+test")]
    s = get_skill("greet")
    assert s.run_count == 1
    assert s.success_count == 1


def test_run_skill_records_failure_on_error_response():
    from app.skills import save_skill, run_skill, get_skill
    save_skill(name="bad", task_template="x")
    cmd = _StubCommander(response="Sorry, I can't help with that.")
    run_skill("bad", {}, "+test", cmd)
    s = get_skill("bad")
    assert s.run_count == 1
    assert s.success_count == 0


def test_run_skill_unknown_raises_keyerror():
    from app.skills import run_skill
    with pytest.raises(KeyError):
        run_skill("nope", {}, "+test", _StubCommander())


# ── Signal slash commands ─────────────────────────────────────────────────

def test_skill_help_command():
    from app.agents.commander.commands import try_command
    out = try_command("/skill help", sender="+t", commander=None)
    assert isinstance(out, str)
    assert "/skill save" in out
    assert "/skill run" in out


def test_skill_save_inline():
    from app.agents.commander.commands import try_command
    out = try_command(
        "/skill save weekly: Summarize Q{quarter} status",
        sender="+t",
        commander=None,
    )
    assert "Saved skill" in out
    assert "weekly" in out
    assert "args: quarter" in out


def test_skill_run_substitutes_args():
    from app.agents.commander.commands import try_command
    try_command("/skill save weekly: Summarize Q{quarter} status", sender="+t", commander=None)
    cmd = _StubCommander(response="report ready")
    out = try_command("/skill run weekly quarter=2", sender="+t", commander=cmd)
    assert out == "report ready"
    assert cmd.calls == [("Summarize Q2 status", "+t")]


def test_skill_run_quoted_args():
    from app.agents.commander.commands import try_command
    try_command(
        '/skill save weekly: Q{q} focus on {topic}',
        sender="+t", commander=None,
    )
    cmd = _StubCommander(response="ok")
    out = try_command(
        '/skill run weekly q=2 topic="growth and ops"',
        sender="+t",
        commander=cmd,
    )
    assert cmd.calls[-1][0] == "Q2 focus on growth and ops"


def test_skill_list_lists_saved():
    from app.agents.commander.commands import try_command
    try_command("/skill save morning: Morning briefing", sender="+t", commander=None)
    try_command("/skill save weekly: Q{q} status", sender="+t", commander=None)
    out = try_command("/skill list", sender="+t", commander=None)
    assert "morning" in out
    assert "weekly" in out
    assert "args: q" in out


def test_skill_show_returns_template():
    from app.agents.commander.commands import try_command
    try_command("/skill save morning: Morning briefing", sender="+t", commander=None)
    out = try_command("/skill show morning", sender="+t", commander=None)
    assert "morning" in out
    assert "Morning briefing" in out


def test_skill_delete_removes():
    from app.agents.commander.commands import try_command
    try_command("/skill save foo: bar", sender="+t", commander=None)
    out = try_command("/skill delete foo", sender="+t", commander=None)
    assert "Deleted" in out
    out2 = try_command("/skill list", sender="+t", commander=None)
    assert "No skills saved" in out2


def test_skill_run_unknown_returns_friendly_error():
    from app.agents.commander.commands import try_command
    out = try_command("/skill run nope", sender="+t", commander=_StubCommander())
    assert "No skill named" in out
