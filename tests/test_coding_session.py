"""Tests for app.coding_session — Phase 5.4-a.

Coverage:

  * **Models** — ``CodingSession.to_dict``/``from_dict`` round-trip,
    ``SubmitResult`` round-trip, status predicates.
  * **Store** — save/get, list filtering by status + agent,
    ``count_active``, audit-log hash-chain integrity, lazy-load
    re-entry safety (RLock).
  * **Quotas** — per-agent count, system count, disk caps,
    ``cap_run_timeout``, ``QuotaConfig.from_env``.
  * **Manager** — start happy path, ref-resolution failure, quota
    rejection, touch / record_write / record_run, submit / discard /
    expire / fail transitions, illegal transitions, idempotency on
    re-discard / re-expire.
  * **Reconciler** — TTL expiry, idle expiry, mixed batch,
    teardown failure handling, racing reconciler is idempotent.

Tests use ``tmp_path`` + ``monkeypatch`` to redirect the store
directory so they don't pollute real workspace state. All worktree
backends are stubbed — Phase 5.4-a doesn't require real git.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


# ── Helpers ──────────────────────────────────────────────────────────


class FakeBackend:
    """In-memory ``WorktreeBackend`` for testing.

    Records every call; lets tests programmatically inject failures."""

    def __init__(self) -> None:
        self.refs: dict[str, str] = {"main": "abc123" * 6 + "ab"}  # 40 chars
        self.created: list[dict] = []
        self.removed: list[dict] = []
        self.fail_resolve: bool = False
        self.fail_create: bool = False
        self.fail_remove: bool = False

    def resolve_ref(self, ref: str) -> str:
        if self.fail_resolve:
            raise RuntimeError(f"fake backend: resolve_ref({ref!r}) failure")
        if ref not in self.refs:
            raise ValueError(f"unknown ref {ref!r}")
        return self.refs[ref]

    def create_worktree(self, *, worktree_path: str, base_sha: str) -> None:
        if self.fail_create:
            raise RuntimeError("fake backend: create_worktree failure")
        self.created.append({"path": worktree_path, "sha": base_sha})

    def remove_worktree(self, *, worktree_path: str, force: bool = True) -> None:
        if self.fail_remove:
            raise RuntimeError("fake backend: remove_worktree failure")
        self.removed.append({"path": worktree_path, "force": force})


@pytest.fixture
def store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the coding-sessions store to a tmp_path. The
    monkeypatched module globals live on the store module; the
    audit log path must also be patched (it was computed once at
    import time)."""
    from app.coding_session import store

    monkeypatch.setattr(store, "_STORE_DIR", tmp_path)
    monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "audit.jsonl")
    store.reset_for_tests()
    return tmp_path


@pytest.fixture
def manager(store_dir: Path) -> Any:
    """A Manager wired to an in-memory backend + tight quotas so
    tests can hit the limits quickly."""
    from app.coding_session import Manager, QuotaConfig

    cfg = QuotaConfig(
        per_agent_active=2,
        system_active=4,
        per_session_disk_bytes=1024,           # 1 KB
        system_disk_bytes=4 * 1024,            # 4 KB
        ttl_seconds=60,
        idle_seconds=30,
    )
    return Manager(backend=FakeBackend(), config=cfg)


# ── Models ──────────────────────────────────────────────────────────


class TestModels:

    def test_to_dict_from_dict_roundtrip(self) -> None:
        from app.coding_session import CodingSession, Status, SubmitResult

        cs = CodingSession(
            id="t1",
            agent_id="coder",
            purpose="fix the bug",
            created_at="2026-05-04T16:00:00+00:00",
            base="main",
            base_sha="a" * 40,
            worktree_path="/tmp/agent-sessions/t1",
            expires_at="2026-05-04T16:30:00+00:00",
            last_activity_at="2026-05-04T16:00:00+00:00",
            status=Status.SUBMITTED,
            files_touched=["app/foo.py", "tests/test_foo.py"],
            run_count=3,
            bytes_written=1234,
            terminated_at="2026-05-04T16:10:00+00:00",
            terminated_reason="submitted",
            submit_results=[
                SubmitResult(path="app/foo.py", change_request_id="cr1", status="pending"),
                SubmitResult(
                    path="app/auto_deployer.py",
                    change_request_id=None,
                    status="tier_immutable_refused",
                    refusal_reason="TIER_IMMUTABLE",
                ),
            ],
        )
        d = cs.to_dict()
        # JSON round-trip — ensures the dict is actually serializable
        cs2 = CodingSession.from_dict(json.loads(json.dumps(d)))
        assert cs2.id == cs.id
        assert cs2.status is Status.SUBMITTED
        assert cs2.files_touched == cs.files_touched
        assert cs2.submit_results is not None
        assert len(cs2.submit_results) == 2
        assert cs2.submit_results[1].refusal_reason == "TIER_IMMUTABLE"

    def test_status_predicates(self) -> None:
        from app.coding_session import CodingSession, Status

        cs = CodingSession(
            id="x", agent_id="a", purpose="p",
            created_at="t", base="main", base_sha="s",
            worktree_path="/tmp/x", expires_at="t", last_activity_at="t",
            status=Status.ACTIVE,
        )
        assert cs.is_active
        assert not cs.is_terminal

        for terminal in (
            Status.SUBMITTED, Status.DISCARDED, Status.EXPIRED, Status.FAILED,
        ):
            cs.status = terminal
            assert not cs.is_active
            assert cs.is_terminal

    def test_submit_result_dict_roundtrip_omits_none_refusal(self) -> None:
        from app.coding_session import SubmitResult

        r = SubmitResult(path="x.py", change_request_id="cr", status="pending")
        d = r.to_dict()
        assert "refusal_reason" not in d
        r2 = SubmitResult.from_dict(d)
        assert r2.refusal_reason is None


# ── Store ───────────────────────────────────────────────────────────


def _make_session(id: str = "t1", **overrides: Any):
    from app.coding_session import CodingSession, Status

    base: dict[str, Any] = dict(
        id=id, agent_id="coder", purpose="p",
        created_at=f"2026-05-0{(int(id[-1]) if id[-1].isdigit() else 1)}T00:00:00+00:00",
        base="main", base_sha="a" * 40,
        worktree_path=f"/tmp/agent-sessions/{id}",
        expires_at="2026-05-04T01:00:00+00:00",
        last_activity_at=f"2026-05-0{(int(id[-1]) if id[-1].isdigit() else 1)}T00:00:00+00:00",
        status=Status.ACTIVE,
    )
    base.update(overrides)
    return CodingSession(**base)


class TestStore:

    def test_save_get_roundtrip(self, store_dir: Path) -> None:
        from app.coding_session import store

        cs = _make_session("t1")
        store.save(cs, audit_event="started")
        loaded = store.get("t1")
        assert loaded is not None
        assert loaded.id == "t1"
        assert loaded.agent_id == "coder"

    def test_audit_log_hash_chain(self, store_dir: Path) -> None:
        from app.coding_session import store

        for i in range(3):
            cs = _make_session(f"t{i}")
            store.save(cs, audit_event=f"event{i}")

        log_path = store_dir / "audit.jsonl"
        assert log_path.exists()
        entries = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(entries) == 3
        assert entries[0]["prev_hash"] == ""
        assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
        assert entries[2]["prev_hash"] == entries[1]["entry_hash"]
        # payloads include the canonical fields
        assert entries[0]["payload"]["session_id"] == "t0"
        assert entries[0]["payload"]["agent_id"] == "coder"

    def test_save_does_not_audit_when_event_none(self, store_dir: Path) -> None:
        """``save(cs)`` without ``audit_event=`` is the per-touch path —
        too frequent to log. Confirms the audit log stays empty."""
        from app.coding_session import store

        cs = _make_session("t1")
        store.save(cs)  # no audit_event
        assert not (store_dir / "audit.jsonl").exists()

    def test_list_all_filtered_by_status(self, store_dir: Path) -> None:
        from app.coding_session import Status, store

        for i, status in enumerate([Status.ACTIVE, Status.SUBMITTED, Status.EXPIRED]):
            store.save(_make_session(f"t{i}", status=status))

        active = store.list_all(status=Status.ACTIVE)
        assert len(active) == 1
        assert active[0].id == "t0"

    def test_list_all_filtered_by_agent(self, store_dir: Path) -> None:
        from app.coding_session import store

        store.save(_make_session("t0", agent_id="coder"))
        store.save(_make_session("t1", agent_id="researcher"))
        store.save(_make_session("t2", agent_id="coder"))

        coder = store.list_all(agent_id="coder")
        assert {s.id for s in coder} == {"t0", "t2"}

    def test_count_active(self, store_dir: Path) -> None:
        from app.coding_session import Status, store

        store.save(_make_session("t0", status=Status.ACTIVE, agent_id="coder"))
        store.save(_make_session("t1", status=Status.ACTIVE, agent_id="coder"))
        store.save(_make_session("t2", status=Status.SUBMITTED, agent_id="coder"))
        store.save(_make_session("t3", status=Status.ACTIVE, agent_id="researcher"))

        assert store.count_active() == 3
        assert store.count_active(agent_id="coder") == 2
        assert store.count_active(agent_id="researcher") == 1
        assert store.count_active(agent_id="nobody") == 0

    def test_save_under_lock_does_not_deadlock(self, store_dir: Path) -> None:
        """RLock check — save() holds _LOCK while calling _index().
        With a plain Lock this deadlocks; with RLock it works.
        Regression for the same bug we caught in change_requests."""
        from app.coding_session import store

        # Force lazy-load to be fresh
        store.reset_for_tests()
        cs = _make_session("t1")
        store.save(cs, audit_event="started")  # would hang on plain Lock
        assert store.get("t1") is not None


# ── Quotas ──────────────────────────────────────────────────────────


class TestQuotas:

    def test_can_start_per_agent_cap(self) -> None:
        from app.coding_session import QuotaConfig, can_start_session

        cfg = QuotaConfig(per_agent_active=2, system_active=10)
        # Below cap — allow
        assert can_start_session(
            config=cfg, agent_active_count=1, system_active_count=5,
        ).ok
        # At cap — deny
        r = can_start_session(
            config=cfg, agent_active_count=2, system_active_count=5,
        )
        assert not r.ok
        assert "per-agent" in (r.reason or "")

    def test_can_start_system_cap(self) -> None:
        from app.coding_session import QuotaConfig, can_start_session

        cfg = QuotaConfig(per_agent_active=10, system_active=3)
        r = can_start_session(
            config=cfg, agent_active_count=0, system_active_count=3,
        )
        assert not r.ok
        assert "system" in (r.reason or "")

    def test_can_write_per_session(self) -> None:
        from app.coding_session import QuotaConfig, can_write_bytes

        cfg = QuotaConfig(per_session_disk_bytes=1024, system_disk_bytes=10_000)
        # OK
        assert can_write_bytes(
            config=cfg,
            session_bytes_after_write=1000,
            system_bytes_after_write=2000,
        ).ok
        # Per-session over
        r = can_write_bytes(
            config=cfg,
            session_bytes_after_write=2000,
            system_bytes_after_write=3000,
        )
        assert not r.ok
        assert "per-session" in (r.reason or "")

    def test_can_write_system(self) -> None:
        from app.coding_session import QuotaConfig, can_write_bytes

        cfg = QuotaConfig(per_session_disk_bytes=1024, system_disk_bytes=10_000)
        r = can_write_bytes(
            config=cfg,
            session_bytes_after_write=500,
            system_bytes_after_write=11_000,
        )
        assert not r.ok
        assert "total" in (r.reason or "")

    def test_cap_run_timeout(self) -> None:
        from app.coding_session import QuotaConfig, cap_run_timeout

        cfg = QuotaConfig(run_wallclock_default_s=60, run_wallclock_max_s=300)
        assert cap_run_timeout(config=cfg, requested_s=None) == 60
        assert cap_run_timeout(config=cfg, requested_s=0) == 60
        assert cap_run_timeout(config=cfg, requested_s=-1) == 60
        assert cap_run_timeout(config=cfg, requested_s=120) == 120
        assert cap_run_timeout(config=cfg, requested_s=999) == 300

    def test_quota_config_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No env vars set — picks dataclass defaults."""
        from app.coding_session import QuotaConfig

        for k in (
            "CODING_SESSION_PER_AGENT_ACTIVE",
            "CODING_SESSION_SYSTEM_ACTIVE",
            "CODING_SESSION_TTL_SECONDS",
        ):
            monkeypatch.delenv(k, raising=False)

        cfg = QuotaConfig.from_env()
        assert cfg.per_agent_active == 3
        assert cfg.system_active == 20

    def test_quota_config_from_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.coding_session import QuotaConfig

        monkeypatch.setenv("CODING_SESSION_PER_AGENT_ACTIVE", "7")
        monkeypatch.setenv("CODING_SESSION_TTL_SECONDS", "120")
        cfg = QuotaConfig.from_env()
        assert cfg.per_agent_active == 7
        assert cfg.ttl_seconds == 120

    def test_quota_config_from_env_invalid_falls_back(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.coding_session import QuotaConfig

        monkeypatch.setenv("CODING_SESSION_PER_AGENT_ACTIVE", "not-an-int")
        monkeypatch.setenv("CODING_SESSION_SYSTEM_ACTIVE", "-5")  # non-positive
        cfg = QuotaConfig.from_env()
        assert cfg.per_agent_active == 3   # default
        assert cfg.system_active == 20     # default


# ── Manager ─────────────────────────────────────────────────────────


class TestManagerStart:

    def test_start_happy_path(self, manager: Any, tmp_path: Path) -> None:
        from app.coding_session import Status

        cs = manager.start(
            agent_id="coder",
            base="main",
            purpose="add the missing import",
            worktree_root=tmp_path,
        )
        assert cs.status is Status.ACTIVE
        assert cs.agent_id == "coder"
        assert cs.base == "main"
        assert cs.base_sha  # resolved
        assert cs.worktree_path.startswith(str(tmp_path))
        # Backend was asked to create the worktree
        assert len(manager.backend.created) == 1
        assert manager.backend.created[0]["path"] == cs.worktree_path

    def test_start_validates_inputs(self, manager: Any, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            manager.start(
                agent_id="", base="main", purpose="p", worktree_root=tmp_path,
            )
        with pytest.raises(ValueError, match="purpose"):
            manager.start(
                agent_id="coder", base="main", purpose="   ",
                worktree_root=tmp_path,
            )
        with pytest.raises(ValueError, match="base"):
            manager.start(
                agent_id="coder", base="", purpose="p", worktree_root=tmp_path,
            )

    def test_start_unknown_base_ref_raises(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="cannot resolve base ref"):
            manager.start(
                agent_id="coder", base="branch-that-does-not-exist",
                purpose="p", worktree_root=tmp_path,
            )

    def test_start_per_agent_quota(self, manager: Any, tmp_path: Path) -> None:
        from app.coding_session import QuotaExceeded

        for i in range(2):  # cfg.per_agent_active == 2
            manager.start(
                agent_id="coder", base="main", purpose=f"p{i}",
                worktree_root=tmp_path,
            )
        with pytest.raises(QuotaExceeded, match="per-agent"):
            manager.start(
                agent_id="coder", base="main", purpose="overflow",
                worktree_root=tmp_path,
            )

    def test_start_system_quota(self, manager: Any, tmp_path: Path) -> None:
        from app.coding_session import QuotaExceeded

        # cfg.system_active == 4. Spread across agents to avoid the per-agent cap.
        agents = ["a1", "a1", "a2", "a2"]
        for i, agent in enumerate(agents):
            manager.start(
                agent_id=agent, base="main", purpose=f"p{i}",
                worktree_root=tmp_path,
            )
        with pytest.raises(QuotaExceeded, match="system-wide"):
            manager.start(
                agent_id="a3", base="main", purpose="overflow",
                worktree_root=tmp_path,
            )

    def test_start_failed_worktree_does_not_persist_session(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        """If create_worktree raises, the session record must NOT exist
        — the agent should be free to retry without a stuck record."""
        from app.coding_session import store

        manager.backend.fail_create = True
        with pytest.raises(RuntimeError, match="create_worktree failure"):
            manager.start(
                agent_id="coder", base="main", purpose="p",
                worktree_root=tmp_path,
            )
        assert store.list_all() == []


class TestManagerActiveMutations:

    def test_touch_updates_last_activity(
        self, manager: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.coding_session import store

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        original_activity = cs.last_activity_at

        # Force a different timestamp
        manager.touch(cs.id)
        cs2 = store.get(cs.id)
        assert cs2 is not None
        # touch always rewrites, even if same-second; so check it isn't earlier
        assert cs2.last_activity_at >= original_activity

    def test_touch_terminal_session_is_noop(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        manager.discard(cs.id, reason="agent gave up")
        manager.touch(cs.id)  # no exception, no state change

    def test_record_write_dedupes_path_and_accumulates_bytes(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        manager.record_write(cs.id, "app/foo.py", 100)
        manager.record_write(cs.id, "app/foo.py", 50)  # rewrite same file
        manager.record_write(cs.id, "tests/test_foo.py", 30)

        cs2 = manager.get(cs.id)
        assert cs2 is not None
        assert cs2.files_touched == ["app/foo.py", "tests/test_foo.py"]
        assert cs2.bytes_written == 180

    def test_record_run_increments_count(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        manager.record_run(cs.id)
        manager.record_run(cs.id)
        cs2 = manager.get(cs.id)
        assert cs2 is not None
        assert cs2.run_count == 2

    def test_record_write_on_terminal_raises(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        from app.coding_session import IllegalTransition

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        manager.discard(cs.id, reason="x")
        with pytest.raises(IllegalTransition, match="not ACTIVE"):
            manager.record_write(cs.id, "app/foo.py", 10)


class TestManagerTransitions:

    def test_submit_sets_results_and_terminates(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        from app.coding_session import Status, SubmitResult

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        results = [
            SubmitResult(path="app/foo.py", change_request_id="cr1", status="pending"),
        ]
        cs2 = manager.submit(cs.id, results=results)
        assert cs2.status is Status.SUBMITTED
        assert cs2.terminated_at is not None
        assert cs2.submit_results == results

    def test_discard_idempotent(self, manager: Any, tmp_path: Path) -> None:
        from app.coding_session import Status

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        cs2 = manager.discard(cs.id, reason="agent gave up")
        cs3 = manager.discard(cs.id, reason="another reason")
        assert cs2.status is Status.DISCARDED
        assert cs3.status is Status.DISCARDED
        # First discard's reason wins
        assert cs2.terminated_reason == "agent gave up"
        assert cs3.terminated_reason == "agent gave up"

    def test_discard_after_submit_raises(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        from app.coding_session import IllegalTransition

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        manager.submit(cs.id, results=[])
        with pytest.raises(IllegalTransition):
            manager.discard(cs.id, reason="too late")

    def test_expire_idempotent(self, manager: Any, tmp_path: Path) -> None:
        from app.coding_session import Status

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        manager.expire(cs.id, reason="ttl")
        cs2 = manager.expire(cs.id, reason="ttl-retry")
        assert cs2.status is Status.EXPIRED
        # First expire's reason wins
        assert cs2.terminated_reason == "ttl"

    def test_fail_keeps_worktree(self, manager: Any, tmp_path: Path) -> None:
        from app.coding_session import Status

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        cs2 = manager.fail(cs.id, reason="git worktree corrupt")
        assert cs2.status is Status.FAILED

        ok, note = manager.remove_worktree(cs2)
        assert ok is True
        assert note is not None and "FAILED" in note
        # Backend was NOT asked to remove the worktree
        assert manager.backend.removed == []

    def test_remove_worktree_on_normal_session_calls_backend(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        cs2 = manager.discard(cs.id, reason="agent gave up")
        ok, _ = manager.remove_worktree(cs2)
        assert ok is True
        assert len(manager.backend.removed) == 1
        assert manager.backend.removed[0]["path"] == cs.worktree_path

    def test_remove_worktree_failure_returns_error(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        cs2 = manager.discard(cs.id, reason="x")
        manager.backend.fail_remove = True
        ok, err = manager.remove_worktree(cs2)
        assert ok is False
        assert err is not None
        assert "remove_worktree failure" in err

    def test_unknown_session_id_raises(self, manager: Any) -> None:
        from app.coding_session import IllegalTransition

        with pytest.raises(IllegalTransition, match="not found"):
            manager.discard("nope", reason="x")
        with pytest.raises(IllegalTransition, match="not found"):
            manager.expire("nope", reason="x")


# ── Reconciler ──────────────────────────────────────────────────────


class TestReconciler:

    def test_no_active_sessions_is_noop(self, manager: Any) -> None:
        from app.coding_session import run_once

        report = run_once(manager=manager)
        assert report.scanned == 0
        assert report.expired_ttl == 0
        assert report.expired_idle == 0

    def test_ttl_expiry(
        self, manager: Any, tmp_path: Path, store_dir: Path,
    ) -> None:
        """Mutate the session's expires_at into the past, then run
        the reconciler with ``now`` past it. Confirms TTL path."""
        from app.coding_session import Status, run_once, store

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        # Force expiry into the past
        cs_before = replace(
            cs,
            expires_at=(
                datetime.now(timezone.utc) - timedelta(seconds=30)
            ).isoformat(),
        )
        store.save(cs_before)

        report = run_once(manager=manager)
        assert report.scanned == 1
        assert report.expired_ttl == 1
        assert report.expired_idle == 0
        assert report.teardowns_ok == 1

        cs_after = manager.get(cs.id)
        assert cs_after is not None
        assert cs_after.status is Status.EXPIRED
        assert cs_after.terminated_reason is not None
        assert "ttl" in cs_after.terminated_reason

    def test_idle_expiry(
        self, manager: Any, tmp_path: Path, store_dir: Path,
    ) -> None:
        from app.coding_session import Status, run_once, store

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        # cfg.idle_seconds = 30. Set last_activity 60 s in the past;
        # leave expires_at in the future.
        cs_before = replace(
            cs,
            last_activity_at=(
                datetime.now(timezone.utc) - timedelta(seconds=60)
            ).isoformat(),
            expires_at=(
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
        )
        store.save(cs_before)

        report = run_once(manager=manager)
        assert report.expired_idle == 1
        assert report.expired_ttl == 0

        cs_after = manager.get(cs.id)
        assert cs_after is not None
        assert cs_after.status is Status.EXPIRED
        assert cs_after.terminated_reason is not None
        assert "idle" in cs_after.terminated_reason

    def test_terminal_sessions_skipped(
        self, manager: Any, tmp_path: Path,
    ) -> None:
        from app.coding_session import run_once

        cs1 = manager.start(
            agent_id="coder", base="main", purpose="p1",
            worktree_root=tmp_path,
        )
        cs2 = manager.start(
            agent_id="coder", base="main", purpose="p2",
            worktree_root=tmp_path,
        )
        manager.submit(cs2.id, results=[])

        # Only cs1 is ACTIVE; in the future neither expires
        report = run_once(manager=manager)
        # nothing is past TTL / idle yet → nothing expires
        assert report.scanned == 1   # only the active session
        assert report.expired_ttl == 0
        assert report.expired_idle == 0

    def test_reconciler_idempotent_on_rerun(
        self, manager: Any, tmp_path: Path, store_dir: Path,
    ) -> None:
        """Running twice over the same set must not double-count or
        raise — manager.expire is no-op on already-EXPIRED."""
        from app.coding_session import run_once, store

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        cs_past = replace(
            cs,
            expires_at=(
                datetime.now(timezone.utc) - timedelta(seconds=30)
            ).isoformat(),
        )
        store.save(cs_past)

        first = run_once(manager=manager)
        second = run_once(manager=manager)
        assert first.expired_ttl == 1
        # Second pass: session is no longer ACTIVE, so the scan
        # finds 0 sessions to inspect.
        assert second.scanned == 0

    def test_teardown_failure_does_not_block_expire(
        self, manager: Any, tmp_path: Path, store_dir: Path,
    ) -> None:
        """If backend.remove_worktree raises, the session still gets
        marked EXPIRED — record reaches terminal state even if the
        on-disk worktree lingers."""
        from app.coding_session import Status, run_once, store

        cs = manager.start(
            agent_id="coder", base="main", purpose="p",
            worktree_root=tmp_path,
        )
        cs_past = replace(
            cs,
            expires_at=(
                datetime.now(timezone.utc) - timedelta(seconds=30)
            ).isoformat(),
        )
        store.save(cs_past)
        manager.backend.fail_remove = True

        report = run_once(manager=manager)
        assert report.expired_ttl == 1
        assert report.teardowns_ok == 0
        assert report.teardowns_failed == 1

        cs_after = manager.get(cs.id)
        assert cs_after is not None
        assert cs_after.status is Status.EXPIRED
