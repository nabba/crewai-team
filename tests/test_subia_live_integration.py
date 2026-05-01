"""
Phase 4: feature-flagged live-integration regression tests.

Verifies:
  - Flag off -> no registration, safe no-op
  - Flag on  -> two hooks registered at PRE_TASK + ON_COMPLETE
  - Env var SUBIA_FEATURE_FLAG_LIVE=1 triggers registration
  - Registration is idempotent (can call twice without doubling)
  - Host process survives broken registries (no raise)
  - Wrapper adapter round-trips ctx correctly
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub the heavy dependencies so tests run offline.
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.control_plane", "app.control_plane.db",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
sys.modules["app.control_plane.db"].execute = MagicMock(return_value=[])

import pytest

from app.subia.live_integration import (
    LiveIntegrationState,
    _AgentShim,
    _TaskShim,
    _wrap_post,
    _wrap_pre,
    enable_subia_hooks,
)


# ── Fake registry ─────────────────────────────────────────────────

class FakeRegistry:
    def __init__(self):
        self.hooks: dict[tuple[str, str], dict] = {}

    def register(self, name, hook_point, fn, priority=50,
                 immutable=False, description=""):
        self.hooks[(name, str(hook_point))] = {
            "fn": fn, "priority": priority,
            "immutable": immutable, "description": description,
        }

    def unregister(self, name, hook_point):
        key = (name, str(hook_point))
        return self.hooks.pop(key, None) is not None


# ── Flag handling ────────────────────────────────────────────────

class TestFlagHandling:
    def test_flag_false_does_nothing(self):
        reg = FakeRegistry()
        state = enable_subia_hooks(
            feature_flag=False, hook_registry=reg,
        )
        assert not state.registered
        assert state.reason == "feature_flag disabled"
        assert reg.hooks == {}

    def test_flag_true_registers_both_hooks(self, monkeypatch):
        # Prevent wiki kernel load from noising the test.
        monkeypatch.setattr(
            "app.subia.persistence.load_kernel_state",
            lambda path=None: __import__(
                "app.subia.kernel", fromlist=["SubjectivityKernel"],
            ).SubjectivityKernel(),
        )
        reg = FakeRegistry()
        # llm=MagicMock() so the predict path is an inert stub.
        llm_stub = MagicMock()
        llm_stub.call = MagicMock(return_value='{"confidence": 0.5}')

        state = enable_subia_hooks(
            feature_flag=True, hook_registry=reg, llm=llm_stub,
        )
        assert state.registered
        assert state.reason == "registered"
        # Both hook names present
        names = {name for (name, _hp) in reg.hooks}
        assert "subia_pre_task" in names
        assert "subia_post_task" in names

    def test_env_flag_is_honored(self, monkeypatch):
        monkeypatch.setattr(
            "app.subia.persistence.load_kernel_state",
            lambda path=None: __import__(
                "app.subia.kernel", fromlist=["SubjectivityKernel"],
            ).SubjectivityKernel(),
        )
        monkeypatch.setenv("SUBIA_FEATURE_FLAG_LIVE", "1")

        reg = FakeRegistry()
        llm_stub = MagicMock()
        llm_stub.call = MagicMock(return_value='{"confidence": 0.5}')
        state = enable_subia_hooks(
            hook_registry=reg, llm=llm_stub,
        )
        assert state.registered

    def test_env_flag_false_values_skip(self, monkeypatch):
        monkeypatch.setenv("SUBIA_FEATURE_FLAG_LIVE", "no")
        reg = FakeRegistry()
        state = enable_subia_hooks(hook_registry=reg)
        assert not state.registered

    def test_env_flag_missing_skips(self, monkeypatch):
        monkeypatch.delenv("SUBIA_FEATURE_FLAG_LIVE", raising=False)
        reg = FakeRegistry()
        state = enable_subia_hooks(hook_registry=reg)
        assert not state.registered


# ── Idempotent registration ──────────────────────────────────────

class TestIdempotent:
    def test_double_enable_does_not_duplicate(self, monkeypatch):
        monkeypatch.setattr(
            "app.subia.persistence.load_kernel_state",
            lambda path=None: __import__(
                "app.subia.kernel", fromlist=["SubjectivityKernel"],
            ).SubjectivityKernel(),
        )
        reg = FakeRegistry()
        llm_stub = MagicMock()
        llm_stub.call = MagicMock(return_value='{"confidence": 0.5}')
        enable_subia_hooks(feature_flag=True, hook_registry=reg, llm=llm_stub)
        enable_subia_hooks(feature_flag=True, hook_registry=reg, llm=llm_stub)
        # Two hooks, not four.
        assert len(reg.hooks) == 2


# ── Error tolerance ──────────────────────────────────────────────

class TestErrorTolerance:
    def test_broken_registry_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(
            "app.subia.persistence.load_kernel_state",
            lambda path=None: __import__(
                "app.subia.kernel", fromlist=["SubjectivityKernel"],
            ).SubjectivityKernel(),
        )

        class Broken:
            def register(self, **_kw):
                raise RuntimeError("cannot register")

            def unregister(self, *a, **kw):
                pass

        state = enable_subia_hooks(
            feature_flag=True, hook_registry=Broken(),
        )
        assert not state.registered
        assert "registration failed" in state.reason

    def test_missing_registry_reports_unavailable(self, monkeypatch):
        """If both hook_registry and the live registry are missing,
        return a descriptive state but don't raise.
        """
        monkeypatch.setattr(
            "app.subia.live_integration._get_live_registry",
            lambda: None,
        )
        state = enable_subia_hooks(
            feature_flag=True, hook_registry=None,
        )
        assert not state.registered
        assert "live registry unavailable" in state.reason


# ── Wrapper adapters ─────────────────────────────────────────────

class TestWrapperAdapters:
    def test_wrap_pre_sets_injection_on_ctx(self):
        class FakeCtx:
            def __init__(self, agent_id="r", desc="task"):
                self.agent_id = agent_id
                self.task_description = desc
                self._set_calls = []

            def set(self, k, v):
                self._set_calls.append((k, v))

            def get(self, k, d=None):
                return d

        class FakeHooks:
            def pre_task(self, agent, task):
                assert agent.role == "researcher"
                assert task.description == "do a thing"
                return "--- SubIA Context ---\n(test)\n---"

            def post_task(self, agent, task, result):
                pass

        pre = _wrap_pre(FakeHooks())
        ctx = FakeCtx(agent_id="researcher", desc="do a thing")
        pre(ctx)
        assert ctx._set_calls
        assert ctx._set_calls[0][0] == "subia_context_injection"
        assert "SubIA Context" in ctx._set_calls[0][1]

    def test_wrap_pre_swallows_errors(self):
        class FakeCtx:
            agent_id = "r"
            task_description = "x"

            def set(self, k, v):
                pass

            def get(self, k, d=None):
                return d

        class RaisingHooks:
            def pre_task(self, *a, **kw):
                raise RuntimeError("boom")

            def post_task(self, *a, **kw):
                pass

        pre = _wrap_pre(RaisingHooks())
        # Must not raise
        pre(FakeCtx())

    def test_wrap_post_passes_result(self):
        class FakeCtx:
            agent_id = "r"
            task_description = "x"

            def __init__(self):
                self.get_calls = []

            def get(self, k, d=None):
                self.get_calls.append(k)
                return {"summary": "result-stub"} if k == "result" else d

        captured = {}

        class FakeHooks:
            def pre_task(self, *a, **kw):
                pass

            def post_task(self, agent, task, result):
                captured["result"] = result

        post = _wrap_post(FakeHooks())
        post(FakeCtx())
        assert captured["result"] == {"summary": "result-stub"}


# ── Real lifecycle_hooks registry smoke test ────────────────────

class TestAgainstRealRegistry:
    def test_registers_against_real_get_registry(self, monkeypatch):
        """Smoke test: the real HookRegistry from app.lifecycle_hooks
        accepts our registrations.
        """
        monkeypatch.setattr(
            "app.subia.persistence.load_kernel_state",
            lambda path=None: __import__(
                "app.subia.kernel", fromlist=["SubjectivityKernel"],
            ).SubjectivityKernel(),
        )
        from app.lifecycle_hooks import HookPoint, HookRegistry

        fresh = HookRegistry()
        llm_stub = MagicMock()
        llm_stub.call = MagicMock(return_value='{"confidence": 0.5}')
        state = enable_subia_hooks(
            feature_flag=True, hook_registry=fresh, llm=llm_stub,
        )
        assert state.registered
        # Confirm by listing hooks
        pre_hooks = fresh.list_hooks(HookPoint.PRE_TASK)
        assert any(h["name"] == "subia_pre_task" for h in pre_hooks)
        complete_hooks = fresh.list_hooks(HookPoint.ON_COMPLETE)
        assert any(h["name"] == "subia_post_task" for h in complete_hooks)
