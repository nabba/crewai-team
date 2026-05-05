"""Tests for the 7 coding_session_* agent tools — Phase 5.4-d.

Coverage:

  * **runtime singleton** — get_manager / set_manager_for_tests /
    backend factory (env-driven local vs bridge)
  * **capability vocabulary** — the 4 new tags exist + are in the
    code-development category + match what the tools declare
  * **tool factory** — create_coding_session_tools returns the 7
    tools with the right names
  * **per-tool _run** — happy path + error formatting for each of:
    start, read, write, run, diff, submit, discard

All tool tests inject a Manager wired to a FakeBackend (the same
shape used in test_coding_session_submit.py) so we don't need real
git or the bridge for unit coverage. The previous PRs already cover
the underlying machinery against real git.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ── Shared fakes (re-used from earlier test files) ──────────────────


@dataclass
class _FakeCR:
    id: str
    status: str = "pending"
    decision_reason: str | None = None


@dataclass
class FakePort:
    pending_paths: set[str] = field(default_factory=set)
    refused_paths: dict[str, str] = field(default_factory=dict)
    decision_reasons: dict[str, str] = field(default_factory=dict)
    create_calls: list[dict] = field(default_factory=list)
    send_ask_calls: list[str] = field(default_factory=list)

    def create_request(self, **kw: Any) -> _FakeCR:
        self.create_calls.append(dict(kw))
        path = kw["path"]
        if path in self.refused_paths:
            return _FakeCR(
                id=f"cr-{len(self.create_calls)}",
                status=self.refused_paths[path],
                decision_reason=self.decision_reasons.get(path),
            )
        return _FakeCR(id=f"cr-{len(self.create_calls)}", status="pending")

    def send_ask(self, request_id: str) -> int | None:
        self.send_ask_calls.append(request_id)
        return 1700000000


@dataclass
class FakeWorktreeBackend:
    """Production-shape FakeBackend with extra hooks for tool tests.

    Sessions all live in the actual filesystem under ``worktrees_root``
    so the write tool can verify on-disk state. ``create_worktree``
    just makes the directory.
    """

    base_files: dict[str, str] = field(default_factory=dict)
    changes: list[tuple[str, str]] = field(default_factory=list)
    fail_resolve: bool = False
    create_calls: list[dict] = field(default_factory=list)
    remove_calls: list[dict] = field(default_factory=list)

    def resolve_ref(self, ref: str) -> str:
        if self.fail_resolve:
            raise RuntimeError("fake: resolve_ref boom")
        return "a" * 40

    def create_worktree(self, *, worktree_path: str, base_sha: str) -> None:
        Path(worktree_path).parent.mkdir(parents=True, exist_ok=True)
        Path(worktree_path).mkdir(exist_ok=True)
        self.create_calls.append({"path": worktree_path, "sha": base_sha})

    def remove_worktree(self, *, worktree_path: str, force: bool = True) -> None:
        self.remove_calls.append({"path": worktree_path, "force": force})

    def list_changed_paths(self, *, worktree_path: str) -> list[tuple[str, str]]:
        return list(self.changes)

    def read_worktree_file(self, *, worktree_path: str, path: str) -> str:
        full = Path(worktree_path) / path
        if not full.exists():
            raise FileNotFoundError(path)
        return full.read_text(encoding="utf-8")

    def read_base_file(self, *, base_sha: str, path: str) -> str:
        if path not in self.base_files:
            raise FileNotFoundError(path)
        return self.base_files[path]


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app.coding_session import store

    monkeypatch.setattr(store, "_STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "store" / "audit.jsonl")
    store.reset_for_tests()
    return tmp_path


@pytest.fixture
def manager(store_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Manager wired to a real-disk FakeBackend, with the runtime
    singleton + worktree_root pointed at tmp_path."""
    from app.coding_session import Manager
    from app.coding_session import runtime

    backend = FakeWorktreeBackend()
    mgr = Manager(backend=backend)
    runtime.set_manager_for_tests(mgr)

    monkeypatch.setenv(
        "CODING_SESSION_WORKTREE_ROOT", str(tmp_path / "worktrees"),
    )

    yield mgr, backend

    runtime.set_manager_for_tests(None)


@pytest.fixture
def started_session(manager, tmp_path: Path):
    """Returns (session_id, manager, backend) with one ACTIVE session."""
    mgr, backend = manager
    cs = mgr.start(
        agent_id="coder", base="main", purpose="fix the bug",
        worktree_root=tmp_path / "worktrees",
    )
    return cs.id, mgr, backend


# ── Capability vocabulary ───────────────────────────────────────────


class TestCapabilityVocabulary:

    def test_four_tags_added(self) -> None:
        from app.tool_registry.capabilities import (
            CAPABILITIES, all_capability_tags, category_for, is_known,
        )

        for tag in (
            "reads-coding-session",
            "writes-coding-session",
            "runs-coding-session",
            "submits-coding-session",
        ):
            assert is_known(tag), f"capability {tag!r} should be known"
            assert tag in all_capability_tags()
            assert category_for(tag) == "code-development"

        # Category exists with all four
        assert "code-development" in CAPABILITIES
        assert len(CAPABILITIES["code-development"]) == 4

    def test_descriptions_nonempty(self) -> None:
        from app.tool_registry.capabilities import description_for

        for tag in (
            "reads-coding-session", "writes-coding-session",
            "runs-coding-session", "submits-coding-session",
        ):
            d = description_for(tag)
            assert d is not None
            assert len(d) > 20  # "real" description, not a placeholder


# ── Runtime singleton ───────────────────────────────────────────────


class TestRuntime:

    def test_set_and_get_manager(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.coding_session import Manager, runtime

        fake = FakeWorktreeBackend()
        mgr = Manager(backend=fake)
        runtime.set_manager_for_tests(mgr)
        try:
            got = runtime.get_manager()
            assert got is mgr
        finally:
            runtime.set_manager_for_tests(None)

    def test_lazy_build_uses_env_backend(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When CODING_SESSION_BACKEND=local, we get a Local backend."""
        from app.coding_session import LocalWorktreeBackend, runtime

        runtime.set_manager_for_tests(None)
        monkeypatch.setenv("CODING_SESSION_BACKEND", "local")
        monkeypatch.setenv("HOST_REPO_PATH", "/tmp/some-repo")
        try:
            mgr = runtime.get_manager()
            assert isinstance(mgr.backend, LocalWorktreeBackend)
            assert mgr.backend.repo_root == "/tmp/some-repo"
        finally:
            runtime.set_manager_for_tests(None)

    def test_lazy_build_default_is_bridge(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.coding_session import BridgeWorktreeBackend, runtime

        runtime.set_manager_for_tests(None)
        monkeypatch.delenv("CODING_SESSION_BACKEND", raising=False)
        try:
            mgr = runtime.get_manager()
            assert isinstance(mgr.backend, BridgeWorktreeBackend)
        finally:
            runtime.set_manager_for_tests(None)

    def test_unknown_backend_falls_back_to_bridge(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.coding_session import BridgeWorktreeBackend, runtime

        runtime.set_manager_for_tests(None)
        monkeypatch.setenv("CODING_SESSION_BACKEND", "rocketship")
        try:
            mgr = runtime.get_manager()
            assert isinstance(mgr.backend, BridgeWorktreeBackend)
        finally:
            runtime.set_manager_for_tests(None)

    def test_worktree_root_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.coding_session import runtime

        monkeypatch.setenv(
            "CODING_SESSION_WORKTREE_ROOT", "/var/agent-sessions",
        )
        assert runtime.worktree_root() == "/var/agent-sessions"

    def test_worktree_root_default(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.coding_session import runtime

        monkeypatch.delenv(
            "CODING_SESSION_WORKTREE_ROOT", raising=False,
        )
        assert runtime.worktree_root() == "/tmp/agent-sessions"


# ── Tool factory ────────────────────────────────────────────────────


class TestToolFactory:

    def test_seven_tools_returned(self) -> None:
        from app.tools.coding_session_tools import create_coding_session_tools

        tools = create_coding_session_tools()
        assert len(tools) == 7
        names = [t.name for t in tools]
        assert names == [
            "coding_session_start",
            "coding_session_read",
            "coding_session_write",
            "coding_session_run",
            "coding_session_diff",
            "coding_session_submit",
            "coding_session_discard",
        ]

    def test_tool_descriptions_nontrivial(self) -> None:
        from app.tools.coding_session_tools import create_coding_session_tools

        for tool in create_coding_session_tools():
            assert len(tool.description) > 50  # non-placeholder


# ── Tool: start ─────────────────────────────────────────────────────


def _tool(name: str):
    from app.tools.coding_session_tools import create_coding_session_tools

    for t in create_coding_session_tools():
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not in factory output")


class TestStartTool:

    def test_happy_path(self, manager, tmp_path: Path) -> None:
        out = _tool("coding_session_start")._run(
            base="main", purpose="fix the bug",
        )
        assert "Coding session" in out
        assert "session_id" in out
        assert "main" in out
        assert "ACTIVE" in out.upper() or "active" in out

    def test_missing_purpose_refused(self, manager) -> None:
        out = _tool("coding_session_start")._run(base="main", purpose="")
        assert out.startswith("REFUSED:")
        assert "purpose" in out.lower()

    def test_quota_exceeded_returns_quota_prefix(
        self, manager, tmp_path: Path,
    ) -> None:
        # Default per-agent cap is 3; spawn 3 then expect 4th to refuse
        for i in range(3):
            out = _tool("coding_session_start")._run(
                base="main", purpose=f"p{i}",
            )
            assert "started" in out.lower()
        out = _tool("coding_session_start")._run(
            base="main", purpose="overflow",
        )
        assert out.startswith("QUOTA_EXCEEDED:")

    def test_unknown_base_ref_refused(self, manager) -> None:
        mgr, backend = manager
        backend.fail_resolve = True
        out = _tool("coding_session_start")._run(
            base="main", purpose="x",
        )
        # Either REFUSED (ValueError translation) or ERROR (RuntimeError)
        # — both prefixes are fine. The point is the agent gets a
        # clean response.
        assert out.startswith(("REFUSED:", "ERROR:"))


# ── Tool: read ──────────────────────────────────────────────────────


class TestReadTool:

    def test_reads_existing_file(self, started_session) -> None:
        sid, mgr, backend = started_session
        # Drop a file into the worktree directly
        cs = mgr.get(sid)
        (Path(cs.worktree_path) / "x.py").write_text("print('hi')\n")

        out = _tool("coding_session_read")._run(
            session_id=sid, path="x.py",
        )
        assert out == "print('hi')\n"

    def test_missing_file_refused(self, started_session) -> None:
        sid, _, _ = started_session
        out = _tool("coding_session_read")._run(
            session_id=sid, path="nonexistent.py",
        )
        assert out.startswith("REFUSED:")

    def test_unknown_session(self, manager) -> None:
        out = _tool("coding_session_read")._run(
            session_id="nope", path="x.py",
        )
        assert out.startswith("ERROR:")
        assert "not found" in out

    def test_terminal_session_refused(self, started_session) -> None:
        sid, mgr, _ = started_session
        mgr.discard(sid, reason="testing")
        out = _tool("coding_session_read")._run(
            session_id=sid, path="x.py",
        )
        assert out.startswith("REFUSED:")
        assert "not ACTIVE" in out


# ── Tool: write ─────────────────────────────────────────────────────


class TestWriteTool:

    def test_write_lands_on_disk(self, started_session) -> None:
        sid, mgr, _ = started_session
        out = _tool("coding_session_write")._run(
            session_id=sid,
            path="app/foo.py",
            content="x = 1\n",
        )
        assert out.startswith("OK:")

        cs = mgr.get(sid)
        full = Path(cs.worktree_path) / "app/foo.py"
        assert full.read_text() == "x = 1\n"
        assert "app/foo.py" in cs.files_touched
        assert cs.bytes_written == len("x = 1\n".encode())

    def test_validator_refusal_on_outside_roots(self, started_session) -> None:
        """Path outside the change-request validator's allowed roots
        is refused at write time (fast-fail before submit)."""
        sid, _, _ = started_session
        out = _tool("coding_session_write")._run(
            session_id=sid, path="workspace/foo.py", content="x",
        )
        assert out.startswith("REFUSED:")
        assert "VALIDATOR" in out

    def test_tier_immutable_refusal_at_write_time(
        self, started_session,
    ) -> None:
        sid, _, _ = started_session
        out = _tool("coding_session_write")._run(
            session_id=sid,
            path="app/auto_deployer.py",
            content="x = 1",
        )
        assert out.startswith("REFUSED:")
        assert "TIER_IMMUTABLE" in out

    def test_unknown_session(self) -> None:
        out = _tool("coding_session_write")._run(
            session_id="nope", path="app/x.py", content="x",
        )
        assert out.startswith("ERROR:")


# ── Tool: run ───────────────────────────────────────────────────────


class TestRunTool:

    def test_runs_python(self, started_session) -> None:
        sid, _, _ = started_session
        out = _tool("coding_session_run")._run(
            session_id=sid,
            argv=["python", "-c", "print('hello from run')"],
            timeout_s=10,
        )
        # The output is JSON
        d = json.loads(out)
        assert d["exit_code"] == 0
        assert "hello from run" in d["stdout"]
        assert d["timed_out"] is False
        assert d["ok"] is True

    def test_disallowed_command_refused(self, started_session) -> None:
        sid, _, _ = started_session
        out = _tool("coding_session_run")._run(
            session_id=sid, argv=["bash", "-c", "echo x"], timeout_s=5,
        )
        d = json.loads(out)
        assert d.get("refused") is True
        assert "allowlist" in d.get("refusal_reason", "")

    def test_run_increments_count(self, started_session) -> None:
        sid, mgr, _ = started_session
        _tool("coding_session_run")._run(
            session_id=sid,
            argv=["python", "-c", "print(1)"],
            timeout_s=5,
        )
        cs_after = mgr.get(sid)
        assert cs_after.run_count == 1

    def test_terminal_session_refused(self, started_session) -> None:
        sid, mgr, _ = started_session
        mgr.discard(sid, reason="x")
        out = _tool("coding_session_run")._run(
            session_id=sid,
            argv=["python", "-c", "print(1)"],
            timeout_s=5,
        )
        assert out.startswith("REFUSED:")


# ── Tool: diff ──────────────────────────────────────────────────────


class TestDiffTool:

    def test_clean_worktree(self, started_session) -> None:
        sid, _, _ = started_session
        out = _tool("coding_session_diff")._run(session_id=sid)
        assert "clean worktree" in out

    def test_with_added_file_lists_it(self, started_session) -> None:
        """The diff tool surfaces untracked files even though
        `git diff HEAD` wouldn't show them."""
        sid, mgr, backend = started_session
        backend.changes = [("new.py", "A")]
        out = _tool("coding_session_diff")._run(session_id=sid)
        assert "new.py" in out


# ── Tool: submit ────────────────────────────────────────────────────


class TestSubmitTool:

    def test_clean_worktree_no_op(
        self, started_session, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sid, mgr, backend = started_session
        backend.changes = []

        # Wire a FakePort via the runtime — we don't have the real
        # change_requests in the unit test process, so monkeypatch
        # the DefaultChangeRequestPort on the submit module.
        from app.coding_session import submit as submit_mod

        port = FakePort()
        monkeypatch.setattr(
            submit_mod, "DefaultChangeRequestPort", lambda: port,
        )

        out = _tool("coding_session_submit")._run(
            session_id=sid, reason="no-op",
        )
        d = json.loads(out)
        assert d["submitted"] == 0
        assert d["refused"] == 0
        assert d["results"] == []

    def test_per_file_split(
        self, started_session, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sid, mgr, backend = started_session
        backend.base_files = {"a.py": "old"}
        backend.changes = [("a.py", "M"), ("b.py", "A")]

        from app.coding_session import submit as submit_mod

        port = FakePort()
        monkeypatch.setattr(
            submit_mod, "DefaultChangeRequestPort", lambda: port,
        )

        # Drop the worktree files so backend.read_worktree_file works
        cs = mgr.get(sid)
        (Path(cs.worktree_path) / "a.py").write_text("new\n")
        (Path(cs.worktree_path) / "b.py").write_text("brand new\n")

        out = _tool("coding_session_submit")._run(
            session_id=sid, reason="ran tests",
        )
        d = json.loads(out)
        assert d["submitted"] == 2
        assert len(port.create_calls) == 2

    def test_unknown_session(self, manager) -> None:
        out = _tool("coding_session_submit")._run(
            session_id="nope", reason="r",
        )
        assert out.startswith("ERROR:")


# ── Tool: discard ───────────────────────────────────────────────────


class TestDiscardTool:

    def test_discard_terminates(self, started_session) -> None:
        from app.coding_session import Status

        sid, mgr, _ = started_session
        out = _tool("coding_session_discard")._run(
            session_id=sid, reason="couldn't crack it",
        )
        assert "discarded" in out.lower()
        cs = mgr.get(sid)
        assert cs.status is Status.DISCARDED

    def test_unknown_session(self, manager) -> None:
        out = _tool("coding_session_discard")._run(
            session_id="nope", reason="x",
        )
        assert out.startswith("ERROR:")

    def test_discard_already_discarded_idempotent(
        self, started_session,
    ) -> None:
        sid, mgr, _ = started_session
        _tool("coding_session_discard")._run(session_id=sid, reason="x")
        out = _tool("coding_session_discard")._run(
            session_id=sid, reason="y",
        )
        # Second call returns OK shape (manager.discard is idempotent)
        assert "discarded" in out.lower()
