"""Tests for app.coding_session.submit — Phase 5.4-c.

Two layers of test:

  * **Unit** with a ``FakePort`` and an in-memory ``FakeBackend``
    that pre-records changed paths + content. Verifies the submit
    flow's logic without git or change_requests.
  * **Integration-light** using ``LocalWorktreeBackend`` against a
    real fixture repo + ``FakePort`` for the change-request side.
    Confirms the backend's read methods integrate correctly with
    the submit pipeline (real git status / git show paths).

The full integration against the real ``app.change_requests`` is
manual / smoke after both stacks land.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ── Fakes ───────────────────────────────────────────────────────────


@dataclass
class _FakeCR:
    """Stands in for ChangeRequest. Has .id and .status (string)."""

    id: str
    status: str = "pending"
    decision_reason: str | None = None


@dataclass
class FakePort:
    """In-memory ChangeRequestPort. Records every call; lets tests
    pre-program the response per path (TIER_IMMUTABLE refusal,
    validator failure, normal pending, etc.).
    """

    pending_paths: set[str] = field(default_factory=set)
    refused_paths: dict[str, str] = field(default_factory=dict)  # path → status
    decision_reasons: dict[str, str] = field(default_factory=dict)

    create_calls: list[dict] = field(default_factory=list)
    send_ask_calls: list[str] = field(default_factory=list)

    def create_request(self, **kw: Any) -> _FakeCR:
        self.create_calls.append(dict(kw))
        path = kw["path"]
        cr_id = f"cr-{path.replace('/', '_')}"
        if path in self.refused_paths:
            return _FakeCR(
                id=cr_id,
                status=self.refused_paths[path],
                decision_reason=self.decision_reasons.get(path),
            )
        return _FakeCR(id=cr_id, status="pending")

    def send_ask(self, request_id: str) -> int | None:
        self.send_ask_calls.append(request_id)
        return 1700000000  # any int — submit doesn't act on the value


@dataclass
class FakeWorktreeBackend:
    """In-memory backend covering the read + lifecycle methods that
    submit + manager need."""

    base_files: dict[str, str] = field(default_factory=dict)
    worktree_files: dict[str, str] = field(default_factory=dict)
    changes: list[tuple[str, str]] = field(default_factory=list)

    create_calls: list[dict] = field(default_factory=list)
    remove_calls: list[dict] = field(default_factory=list)

    fail_list_changed_paths: bool = False

    # Lifecycle (used by Manager.start)

    def resolve_ref(self, ref: str) -> str:
        return "a" * 40

    def create_worktree(self, *, worktree_path: str, base_sha: str) -> None:
        self.create_calls.append({"path": worktree_path, "sha": base_sha})

    def remove_worktree(self, *, worktree_path: str, force: bool = True) -> None:
        self.remove_calls.append({"path": worktree_path, "force": force})

    # Read (used by submit)

    def list_changed_paths(self, *, worktree_path: str) -> list[tuple[str, str]]:
        if self.fail_list_changed_paths:
            raise RuntimeError("fake: list_changed_paths failure")
        return list(self.changes)

    def read_worktree_file(self, *, worktree_path: str, path: str) -> str:
        if path not in self.worktree_files:
            raise FileNotFoundError(f"{path} not in fake worktree")
        return self.worktree_files[path]

    def read_base_file(self, *, base_sha: str, path: str) -> str:
        if path not in self.base_files:
            raise FileNotFoundError(f"{path} not in fake base {base_sha[:8]}")
        return self.base_files[path]


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app.coding_session import store

    monkeypatch.setattr(store, "_STORE_DIR", tmp_path)
    monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "audit.jsonl")
    store.reset_for_tests()
    return tmp_path


@pytest.fixture
def manager_with_fake(store_dir: Path):
    """Manager wired to the fake backend. Used by all unit tests."""
    from app.coding_session import Manager

    backend = FakeWorktreeBackend()
    return Manager(backend=backend), backend


# ── Unit: the per-file pipeline ─────────────────────────────────────


class TestSubmitOneFile:

    def _start(self, mgr, tmp_path: Path):
        return mgr.start(
            agent_id="coder",
            base="main",
            purpose="add the missing import",
            worktree_root=tmp_path,
        )

    def test_modify_existing_file_creates_pending_cr_and_sends_ask(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = self._start(mgr, tmp_path)
        backend.base_files["app/foo.py"] = "old\n"
        backend.worktree_files["app/foo.py"] = "old\nnew line\n"
        backend.changes = [("app/foo.py", "M")]

        port = FakePort()
        updated, results = submit_session(
            cs.id,
            submit_reason="ran pytest, all green",
            manager=mgr,
            port=port,
            cleanup_worktree=False,
        )

        # One CR created + one ASK sent
        assert len(port.create_calls) == 1
        assert len(port.send_ask_calls) == 1
        assert port.send_ask_calls[0] == port.create_calls[0]["path"].replace(
            "/", "_"
        ).join(["cr-", ""])
        # Result records change_request id + pending status
        assert len(results) == 1
        r = results[0]
        assert r.path == "app/foo.py"
        assert r.change_request_id is not None
        assert r.status == "pending"
        assert r.refusal_reason is None
        # Session is now SUBMITTED
        from app.coding_session import Status
        assert updated.status is Status.SUBMITTED
        assert updated.submit_results is not None and len(updated.submit_results) == 1

    def test_added_file_uses_empty_old_content(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = self._start(mgr, tmp_path)
        backend.worktree_files["app/new.py"] = "fresh\n"
        # base_files has no entry → read_base_file raises FileNotFoundError
        backend.changes = [("app/new.py", "A")]

        port = FakePort()
        submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
            cleanup_worktree=False,
        )

        assert len(port.create_calls) == 1
        # Old content should be empty for the added file
        assert port.create_calls[0]["old_content"] == ""
        assert port.create_calls[0]["new_content"] == "fresh\n"

    def test_deleted_file_refused(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = self._start(mgr, tmp_path)
        backend.changes = [("app/old.py", "D")]
        # No worktree_files entry needed — submit recognizes 'D' upfront

        port = FakePort()
        _, results = submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
            cleanup_worktree=False,
        )

        # No CR created — refused at the kind=D check
        assert port.create_calls == []
        assert len(results) == 1
        assert results[0].change_request_id is None
        assert results[0].status == "refused"
        assert results[0].refusal_reason is not None
        assert "delete-not-supported" in results[0].refusal_reason

    def test_unknown_kind_refused(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = self._start(mgr, tmp_path)
        backend.changes = [("app/weird.py", "?")]

        port = FakePort()
        _, results = submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
            cleanup_worktree=False,
        )

        assert port.create_calls == []
        assert results[0].status == "refused"


class TestSubmitMultipleFiles:

    def test_per_file_split_into_multiple_crs(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="multi-file fix",
            worktree_root=tmp_path,
        )
        backend.base_files = {"a.py": "a old\n", "b.py": "b old\n"}
        backend.worktree_files = {"a.py": "a new\n", "b.py": "b new\n"}
        backend.changes = [("a.py", "M"), ("b.py", "M")]

        port = FakePort()
        _, results = submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
            cleanup_worktree=False,
        )

        assert len(results) == 2
        assert {r.path for r in results} == {"a.py", "b.py"}
        # Two CRs + two ASKs
        assert len(port.create_calls) == 2
        assert len(port.send_ask_calls) == 2

    def test_tier_immutable_refused_per_file_does_not_block_others(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="mixed",
            worktree_root=tmp_path,
        )
        backend.base_files = {
            "app/foo.py": "ok\n",
            "app/auto_deployer.py": "tier3\n",
        }
        backend.worktree_files = {
            "app/foo.py": "ok modified\n",
            "app/auto_deployer.py": "tier3 modified\n",
        }
        backend.changes = [
            ("app/foo.py", "M"),
            ("app/auto_deployer.py", "M"),
        ]

        port = FakePort(
            refused_paths={
                "app/auto_deployer.py": "tier_immutable_refused",
            },
            decision_reasons={
                "app/auto_deployer.py": (
                    "TIER_IMMUTABLE: cannot be modified by agent path"
                ),
            },
        )
        _, results = submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
            cleanup_worktree=False,
        )

        # Both CRs were created — but one came back refused
        assert len(port.create_calls) == 2
        # send_ask is only sent for PENDING; refused path is skipped
        assert len(port.send_ask_calls) == 1
        # Refused result has the validator's reason
        by_path = {r.path: r for r in results}
        assert by_path["app/auto_deployer.py"].status == "tier_immutable_refused"
        assert (
            "TIER_IMMUTABLE"
            in (by_path["app/auto_deployer.py"].refusal_reason or "")
        )
        # The good file still landed PENDING
        assert by_path["app/foo.py"].status == "pending"
        assert by_path["app/foo.py"].refusal_reason is None

    def test_per_file_exception_does_not_kill_batch(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        """A genuine exception during per-file dispatch (e.g. a port
        error) records as 'error' in the SubmitResult; other files in
        the batch continue."""
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="x",
            worktree_root=tmp_path,
        )
        backend.base_files = {"a.py": "a\n", "b.py": "b\n"}
        backend.worktree_files = {"a.py": "a'\n", "b.py": "b'\n"}
        backend.changes = [("a.py", "M"), ("b.py", "M")]

        class ExplodingPort(FakePort):
            def create_request(self, **kw):
                if kw["path"] == "a.py":
                    raise RuntimeError("port boom")
                return super().create_request(**kw)

        port = ExplodingPort()
        _, results = submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
            cleanup_worktree=False,
        )

        by_path = {r.path: r for r in results}
        assert by_path["a.py"].status == "error"
        assert by_path["a.py"].refusal_reason is not None
        assert "port boom" in by_path["a.py"].refusal_reason
        # b.py still went through normally
        assert by_path["b.py"].status == "pending"


class TestSubmitLifecycle:

    def test_cleanup_worktree_default_true(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="x",
            worktree_root=tmp_path,
        )
        backend.changes = []
        port = FakePort()

        submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
        )
        # remove_worktree was called (default cleanup_worktree=True)
        assert len(backend.remove_calls) == 1

    def test_cleanup_worktree_false(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="x",
            worktree_root=tmp_path,
        )
        backend.changes = []
        port = FakePort()

        submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
            cleanup_worktree=False,
        )
        assert backend.remove_calls == []

    def test_clean_worktree_succeeds_with_no_results(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        """Submitting with no changes is allowed — submits the session
        with empty results. Could happen if the agent decided to
        no-op."""
        from app.coding_session import Status, submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="x",
            worktree_root=tmp_path,
        )
        backend.changes = []

        port = FakePort()
        updated, results = submit_session(
            cs.id, submit_reason="no-op", manager=mgr, port=port,
            cleanup_worktree=False,
        )
        assert results == []
        assert updated.status is Status.SUBMITTED
        assert port.create_calls == []

    def test_session_not_active_raises(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import IllegalTransition, submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="x",
            worktree_root=tmp_path,
        )
        mgr.discard(cs.id, reason="testing")
        backend.changes = []

        with pytest.raises(IllegalTransition):
            submit_session(
                cs.id, submit_reason="r", manager=mgr,
                port=FakePort(),
            )

    def test_unknown_session_raises(
        self, manager_with_fake,
    ) -> None:
        from app.coding_session import IllegalTransition, submit_session

        mgr, _ = manager_with_fake
        with pytest.raises(IllegalTransition, match="not found"):
            submit_session(
                "nope", submit_reason="r", manager=mgr,
                port=FakePort(),
            )

    def test_list_changed_paths_failure_marks_session_failed(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        """If git status itself fails, we transition to FAILED rather
        than SUBMITTED — leaves the worktree for forensics."""
        from app.coding_session import Status, submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="x",
            worktree_root=tmp_path,
        )
        backend.fail_list_changed_paths = True

        with pytest.raises(RuntimeError, match="fake: list_changed_paths"):
            submit_session(
                cs.id, submit_reason="r", manager=mgr,
                port=FakePort(),
            )

        cs_after = mgr.get(cs.id)
        assert cs_after is not None
        assert cs_after.status is Status.FAILED


class TestPortContract:
    """Verifies the port receives exactly the expected fields."""

    def test_create_request_kwargs(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="THE PURPOSE",
            worktree_root=tmp_path,
        )
        backend.base_files = {"app/foo.py": "old"}
        backend.worktree_files = {"app/foo.py": "new"}
        backend.changes = [("app/foo.py", "M")]

        port = FakePort()
        submit_session(
            cs.id, submit_reason="THE SUBMIT REASON", manager=mgr,
            port=port, cleanup_worktree=False,
        )

        call = port.create_calls[0]
        assert set(call.keys()) == {
            "requestor", "path", "new_content", "old_content", "reason",
        }
        assert call["requestor"] == "coder"
        assert call["path"] == "app/foo.py"
        assert call["old_content"] == "old"
        assert call["new_content"] == "new"
        # Reason includes session purpose, submit_reason, and a session ref
        assert "THE PURPOSE" in call["reason"]
        assert "THE SUBMIT REASON" in call["reason"]
        assert cs.id in call["reason"]

    def test_pending_triggers_send_ask_others_dont(
        self, manager_with_fake, tmp_path: Path,
    ) -> None:
        from app.coding_session import submit_session

        mgr, backend = manager_with_fake
        cs = mgr.start(
            agent_id="coder", base="main", purpose="x",
            worktree_root=tmp_path,
        )
        backend.base_files = {"a.py": "a", "b.py": "b"}
        backend.worktree_files = {"a.py": "a'", "b.py": "b'"}
        backend.changes = [("a.py", "M"), ("b.py", "M")]

        port = FakePort(
            refused_paths={"b.py": "rejected"},
        )
        submit_session(
            cs.id, submit_reason="r", manager=mgr, port=port,
            cleanup_worktree=False,
        )

        # Two CRs created
        assert len(port.create_calls) == 2
        # One ASK sent — only for the PENDING one
        assert len(port.send_ask_calls) == 1


# ── Integration-light: real LocalWorktreeBackend + FakePort ─────────


pytestmark_real_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git CLI not available",
)


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """Same shape as test_coding_session_backends.py — a tiny git
    repo with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*argv: str) -> None:
        subprocess.run(
            ["git", *argv],
            cwd=repo, check=True, capture_output=True, text=True,
        )

    _git("init", "--initial-branch=main")
    _git("config", "user.email", "test@example.com")
    _git("config", "user.name", "Test")
    _git("config", "commit.gpgsign", "false")
    (repo / "app.py").write_text("print('original')\n")
    _git("add", "app.py")
    _git("commit", "-m", "initial")
    return repo


@pytestmark_real_git
class TestSubmitWithRealLocalBackend:
    """End-to-end with real git for the read methods, FakePort for
    the change-request side. Catches integration bugs between the
    backend's read methods and the submit pipeline."""

    def test_modify_and_add_real_repo(
        self, fixture_repo: Path, tmp_path: Path, store_dir: Path,
    ) -> None:
        from app.coding_session import (
            LocalWorktreeBackend,
            Manager,
            submit_session,
        )

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        mgr = Manager(backend=backend)
        wt_root = tmp_path / "wts"

        cs = mgr.start(
            agent_id="coder",
            base="main",
            purpose="add a note + tweak app.py",
            worktree_root=wt_root,
        )

        # Agent edits — modify existing, add new
        wt = Path(cs.worktree_path)
        (wt / "app.py").write_text("print('agent-modified')\n")
        (wt / "NOTES.md").write_text("things the agent learned\n")

        port = FakePort()
        updated, results = submit_session(
            cs.id,
            submit_reason="modifications complete",
            manager=mgr,
            port=port,
            cleanup_worktree=True,   # exercise the cleanup path too
        )

        from app.coding_session import Status
        assert updated.status is Status.SUBMITTED
        assert {r.path for r in results} == {"app.py", "NOTES.md"}

        # The port saw correct old/new for app.py (modified)
        app_call = next(
            c for c in port.create_calls if c["path"] == "app.py"
        )
        assert app_call["old_content"] == "print('original')\n"
        assert app_call["new_content"] == "print('agent-modified')\n"

        # And empty old for NOTES.md (added)
        notes_call = next(
            c for c in port.create_calls if c["path"] == "NOTES.md"
        )
        assert notes_call["old_content"] == ""
        assert notes_call["new_content"] == "things the agent learned\n"

        # Worktree has been removed (cleanup_worktree=True)
        assert not wt.exists()
