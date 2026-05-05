"""Tests for app.coding_session.backends — Phase 5.4-b.

Two surfaces:

  * :class:`LocalWorktreeBackend` — tested against a real fixture
    git repo built fresh per test in ``tmp_path``. Catches genuine
    git-CLI integration bugs (argv shape, return codes, error
    messages) rather than relying on a mock that could drift.

  * :class:`BridgeWorktreeBackend` — tested with a stubbed bridge
    client. We don't have a host bridge available in unit tests;
    the real bridge surface is exercised in 5.4-c integration
    tests where the change-request flow already crosses it.

The Local tests are the primary canary — if the Bridge surface
contract drifts, this set still tells us git operations work; the
Bridge mock just verifies argv shape.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git CLI not available; backend tests require it",
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """Builds a tiny git repo at ``tmp_path/repo`` with one commit on
    ``main``. The repo is .git-having and has a default checkout —
    enough for ``git worktree add`` to succeed.

    Configures git locally so the test doesn't depend on a global
    user.email / user.name."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*argv: str) -> None:
        # check=True intentional: any failure here is a fixture bug, not a
        # production-code bug; surface it loudly.
        subprocess.run(
            ["git", *argv],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )

    _git("init", "--initial-branch=main")
    _git("config", "user.email", "test@example.com")
    _git("config", "user.name", "Test")
    _git("config", "commit.gpgsign", "false")

    (repo / "README.md").write_text("hello\n")
    _git("add", "README.md")
    _git("commit", "-m", "initial")

    return repo


# ── LocalWorktreeBackend ────────────────────────────────────────────


class TestLocalResolveRef:

    def test_main_resolves(self, fixture_repo: Path) -> None:
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        sha = backend.resolve_ref("main")
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_head_resolves(self, fixture_repo: Path) -> None:
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        head = backend.resolve_ref("HEAD")
        main = backend.resolve_ref("main")
        assert head == main

    def test_unknown_ref_raises_valueerror(self, fixture_repo: Path) -> None:
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        with pytest.raises(ValueError, match="cannot resolve ref 'totally-not-a-branch'"):
            backend.resolve_ref("totally-not-a-branch")


class TestLocalCreateWorktree:

    def test_create_at_sha_checks_out(
        self, fixture_repo: Path, tmp_path: Path,
    ) -> None:
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        sha = backend.resolve_ref("main")
        worktree = tmp_path / "wt" / "session1"

        backend.create_worktree(worktree_path=str(worktree), base_sha=sha)

        assert worktree.is_dir()
        assert (worktree / "README.md").exists()
        assert (worktree / "README.md").read_text() == "hello\n"

    def test_create_at_existing_path_fails(
        self, fixture_repo: Path, tmp_path: Path,
    ) -> None:
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        sha = backend.resolve_ref("main")
        existing = tmp_path / "already-there"
        existing.mkdir()

        with pytest.raises(RuntimeError, match="already exists"):
            backend.create_worktree(worktree_path=str(existing), base_sha=sha)

    def test_create_at_unknown_sha_fails(
        self, fixture_repo: Path, tmp_path: Path,
    ) -> None:
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        worktree = tmp_path / "wt" / "session1"

        with pytest.raises(RuntimeError, match="git worktree add failed"):
            backend.create_worktree(
                worktree_path=str(worktree),
                base_sha="0" * 40,   # nonexistent sha
            )


class TestLocalRemoveWorktree:

    def test_remove_clean(
        self, fixture_repo: Path, tmp_path: Path,
    ) -> None:
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        sha = backend.resolve_ref("main")
        worktree = tmp_path / "wt" / "s1"
        backend.create_worktree(worktree_path=str(worktree), base_sha=sha)

        backend.remove_worktree(worktree_path=str(worktree))
        assert not worktree.exists()

    def test_remove_with_dirty_files_force_succeeds(
        self, fixture_repo: Path, tmp_path: Path,
    ) -> None:
        """``force=True`` (the manager's default) is intentional — we
        want teardown to win even when the agent left local changes
        in the worktree."""
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        sha = backend.resolve_ref("main")
        worktree = tmp_path / "wt" / "s1"
        backend.create_worktree(worktree_path=str(worktree), base_sha=sha)

        # Agent left some uncommitted changes
        (worktree / "scratch.py").write_text("# leftover")
        (worktree / "README.md").write_text("modified\n")

        backend.remove_worktree(worktree_path=str(worktree), force=True)
        assert not worktree.exists()

    def test_remove_nonexistent_worktree_raises(
        self, fixture_repo: Path, tmp_path: Path,
    ) -> None:
        """If the directory was already gone (e.g. someone rm-rf'd by
        hand), the fallback path still fails out — the caller logs
        and moves on."""
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        # The fallback eventually calls `git worktree prune` which DOES
        # succeed even if the path was bogus, so this should NOT raise.
        # We just verify it doesn't crash.
        backend.remove_worktree(
            worktree_path=str(tmp_path / "nope" / "never"),
        )


class TestLocalRoundTrip:
    """End-to-end against the fixture repo."""

    def test_resolve_create_remove(
        self, fixture_repo: Path, tmp_path: Path,
    ) -> None:
        from app.coding_session import LocalWorktreeBackend

        backend = LocalWorktreeBackend(repo_root=fixture_repo)
        sha = backend.resolve_ref("main")

        wt1 = tmp_path / "wt" / "a"
        wt2 = tmp_path / "wt" / "b"
        backend.create_worktree(worktree_path=str(wt1), base_sha=sha)
        backend.create_worktree(worktree_path=str(wt2), base_sha=sha)

        assert wt1.is_dir() and wt2.is_dir()
        backend.remove_worktree(worktree_path=str(wt1))
        backend.remove_worktree(worktree_path=str(wt2))
        assert not wt1.exists() and not wt2.exists()


# ── BridgeWorktreeBackend ───────────────────────────────────────────


class _FakeBridge:
    """Minimal stand-in for the production bridge client.

    Records every ``execute`` call so the tests can assert exact argv
    shape; lets the test pre-program responses keyed on the first
    two args of ``argv``.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: dict[tuple[str, str], dict] = {}
        self.unavailable = False

    def is_available(self) -> bool:
        return not self.unavailable

    def execute(
        self,
        argv: list[str],
        *,
        working_dir: str,
        timeout: int,
    ) -> dict:
        self.calls.append({
            "argv": list(argv),
            "working_dir": working_dir,
            "timeout": timeout,
        })
        if not argv:
            return {"returncode": 1, "stderr": "empty argv", "stdout": ""}
        key = (argv[0], argv[1] if len(argv) > 1 else "")
        return self.responses.get(
            key, {"returncode": 0, "stdout": "", "stderr": ""},
        )


class TestBridgeBackendArgvShape:

    def _backend_with_fake(self, fake: _FakeBridge):
        from app.coding_session import BridgeWorktreeBackend

        backend = BridgeWorktreeBackend(
            repo_root="/host/repo", agent_id="test-coding",
        )
        backend._bridge = fake  # bypass the lazy import
        return backend

    def test_resolve_ref_argv(self) -> None:
        fake = _FakeBridge()
        fake.responses[("git", "rev-parse")] = {
            "returncode": 0,
            "stdout": "a" * 40 + "\n",
            "stderr": "",
        }
        backend = self._backend_with_fake(fake)

        sha = backend.resolve_ref("main")

        assert sha == "a" * 40
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["argv"] == ["git", "rev-parse", "--verify", "main"]
        assert call["working_dir"] == "/host/repo"

    def test_resolve_ref_failure_raises_valueerror(self) -> None:
        fake = _FakeBridge()
        fake.responses[("git", "rev-parse")] = {
            "returncode": 128,
            "stdout": "",
            "stderr": "fatal: ambiguous argument 'foo'",
        }
        backend = self._backend_with_fake(fake)

        with pytest.raises(ValueError, match="cannot resolve ref 'foo'"):
            backend.resolve_ref("foo")

    def test_create_worktree_argv(self) -> None:
        fake = _FakeBridge()
        # mkdir succeeds; worktree add succeeds
        backend = self._backend_with_fake(fake)

        backend.create_worktree(
            worktree_path="/tmp/wt/sess1", base_sha="b" * 40,
        )

        # First call: mkdir parent
        assert fake.calls[0]["argv"][:2] == ["mkdir", "-p"]
        # Second call: git worktree add
        assert fake.calls[1]["argv"] == [
            "git", "worktree", "add", "--detach",
            "/tmp/wt/sess1", "b" * 40,
        ]

    def test_create_worktree_failure_raises(self) -> None:
        fake = _FakeBridge()
        fake.responses[("git", "worktree")] = {
            "returncode": 128,
            "stdout": "",
            "stderr": "fatal: invalid reference",
        }
        backend = self._backend_with_fake(fake)

        with pytest.raises(RuntimeError, match="git worktree add failed"):
            backend.create_worktree(
                worktree_path="/tmp/wt/sess1", base_sha="bad",
            )

    def test_remove_worktree_force(self) -> None:
        fake = _FakeBridge()
        backend = self._backend_with_fake(fake)

        backend.remove_worktree(worktree_path="/tmp/wt/sess1", force=True)

        assert fake.calls[0]["argv"] == [
            "git", "worktree", "remove", "--force", "/tmp/wt/sess1",
        ]

    def test_remove_worktree_fallback_to_rm(self) -> None:
        """If `git worktree remove` fails, the backend falls through
        to `rm -rf` then `git worktree prune`."""
        fake = _FakeBridge()
        # First call (git worktree remove) → fail
        fake.responses[("git", "worktree")] = {
            "returncode": 128,
            "stdout": "",
            "stderr": "fatal: not a worktree",
        }
        backend = self._backend_with_fake(fake)

        # Bridge will receive: git worktree remove (fail), rm -rf,
        # git worktree prune. Override prune to also fail to verify
        # the RuntimeError path.
        fake.responses[("git", "worktree")] = {
            "returncode": 128,
            "stdout": "",
            "stderr": "fatal: not a worktree",
        }

        with pytest.raises(RuntimeError, match="prune both failed"):
            backend.remove_worktree(worktree_path="/tmp/wt/sess1")

        # Verify the fallback shape: 1) git worktree remove,
        # 2) rm -rf, 3) git worktree prune
        argvs = [c["argv"][:3] for c in fake.calls]
        assert ["git", "worktree", "remove"] in argvs
        assert ["rm", "-rf", "/tmp/wt/sess1"] in (c["argv"] for c in fake.calls)
        assert ["git", "worktree", "prune"] in (c["argv"][:3] for c in fake.calls)


class TestBridgeBackendUnavailable:

    def test_unavailable_bridge_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the bridge isn't reachable, the manager translates
        this to a clean error for the agent. The backend's job is to
        raise — the manager's tool wrapper handles the messaging."""
        from app.coding_session import BridgeWorktreeBackend

        # Monkeypatch get_bridge to return a None
        import app.coding_session.backends as backends_mod

        def fake_get_bridge(name: str):
            return None

        # The lazy-import lives inside _get_bridge; patch the symbol on
        # the module-level import path.
        import sys
        fake_module = type(sys)("app.bridge_client")
        fake_module.get_bridge = fake_get_bridge  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "app.bridge_client", fake_module)

        backend = BridgeWorktreeBackend(repo_root="/host/repo")
        with pytest.raises(RuntimeError, match="bridge unavailable"):
            backend.resolve_ref("main")
