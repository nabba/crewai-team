"""
Phase 4: SubIA lifecycle hooks regression tests.

Verifies the hook surface that plugs the CIL loop into the
crewai-amendments lifecycle system:

  - pre_task returns a string injection that contains the scene
  - post_task passes the task result through to the loop
  - Operation classification maps descriptions to the right path
  - register/unregister are idempotent against a duck-typed registry
  - Non-string task_result and weird agent shapes don't crash
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

import pytest

from app.subia.hooks import (
    SubIALifecycleHooks,
    _HOOK_NAME_POST,
    _HOOK_NAME_PRE,
    register,
    unregister,
)
from app.subia.kernel import Prediction, SubjectivityKernel
from app.subia.loop import SubIALoop
from app.subia.scene.buffer import CompetitiveGate


# ── Fixtures ─────────────────────────────────────────────────────

@dataclass
class FakeAgent:
    role: str = "researcher"


@dataclass
class FakeTask:
    description: str = "task_execute default"
    id: str = "t-1"


class FakeRegistry:
    """Duck-typed hooks registry for tests."""
    def __init__(self):
        self.hooks = {}

    def register(self, name, when, fn, priority=0):
        self.hooks[name] = {
            "when": when, "fn": fn, "priority": priority,
        }

    def unregister(self, name):
        self.hooks.pop(name, None)


def _make_hooks() -> SubIALifecycleHooks:
    def predict(ctx):
        return Prediction(
            id="p", operation="o",
            predicted_outcome={}, predicted_self_change={},
            predicted_homeostatic_effect={},
            confidence=0.7, created_at="",
        )

    loop = SubIALoop(
        kernel=SubjectivityKernel(),
        scene_gate=CompetitiveGate(capacity=5),
        predict_fn=predict,
    )
    return SubIALifecycleHooks(loop=loop)


# ── pre_task ─────────────────────────────────────────────────────

class TestPreTask:
    def test_returns_subia_context_block(self):
        hooks = _make_hooks()
        out = hooks.pre_task(FakeAgent(), FakeTask())
        assert "--- SubIA Context ---" in out
        assert "--- End SubIA Context ---" in out

    def test_block_contains_loop_type(self):
        hooks = _make_hooks()
        out = hooks.pre_task(FakeAgent(), FakeTask())
        assert "loop: full" in out

    def test_prediction_confidence_rendered(self):
        hooks = _make_hooks()
        out = hooks.pre_task(FakeAgent(), FakeTask())
        assert "prediction: conf=0.70" in out

    def test_compressed_when_wiki_read(self):
        hooks = _make_hooks()
        out = hooks.pre_task(
            FakeAgent(),
            FakeTask(description="please wiki_read /archibal/landscape"),
        )
        assert "loop: compressed" in out

    def test_classification_table(self):
        hooks = _make_hooks()
        cases = [
            ("ingest new source from firecrawl", "ingest"),
            ("lint the evolution archive", "lint"),
            ("wiki_read /x/y.md", "wiki_read"),
            ("wiki_search for 'protocol'", "wiki_search"),
            ("a routine query", "routine_query"),
            ("draft Q2 plan", "task_execute"),
        ]
        for desc, expected in cases:
            assert hooks._classify_operation(desc) == expected, desc


# ── post_task ────────────────────────────────────────────────────

class TestPostTask:
    def test_records_agency(self):
        hooks = _make_hooks()
        hooks.post_task(
            FakeAgent(), FakeTask(),
            task_result={"success": True, "summary": "shipped feature"},
        )
        log = hooks.loop.kernel.self_state.agency_log
        assert len(log) == 1
        assert log[0]["summary"] == "shipped feature"

    def test_accepts_string_result(self):
        hooks = _make_hooks()
        hooks.post_task(FakeAgent(), FakeTask(), task_result="plain string")
        log = hooks.loop.kernel.self_state.agency_log
        assert len(log) == 1
        assert "plain string" in log[0]["summary"]

    def test_pre_and_post_round_trip(self):
        hooks = _make_hooks()
        agent, task = FakeAgent(), FakeTask()
        out = hooks.pre_task(agent, task)
        assert "--- SubIA Context ---" in out
        hooks.post_task(agent, task, task_result={"summary": "done"})
        # _last_pre cache should have drained for this task
        assert str(task.id) not in hooks._last_pre

    def test_post_task_without_pre_task_still_safe(self):
        hooks = _make_hooks()
        hooks.post_task(FakeAgent(), FakeTask(),
                        task_result={"summary": "orphan"})
        # Should not raise, kernel loop_count should advance.
        assert hooks.loop.kernel.loop_count == 1


# ── Weird inputs ─────────────────────────────────────────────────

class TestWeirdInputs:
    def test_agent_without_role(self):
        hooks = _make_hooks()

        class NoRoleAgent:
            pass

        out = hooks.pre_task(NoRoleAgent(), FakeTask())
        assert "--- SubIA Context ---" in out

    def test_task_without_description(self):
        hooks = _make_hooks()

        class NoDescTask:
            pass

        out = hooks.pre_task(FakeAgent(), NoDescTask())
        assert "--- SubIA Context ---" in out

    def test_string_task(self):
        hooks = _make_hooks()
        out = hooks.pre_task(FakeAgent(), "just a string task")
        assert "--- SubIA Context ---" in out


# ── Registration ─────────────────────────────────────────────────

class TestRegistration:
    def test_register_adds_both_hooks(self):
        hooks = _make_hooks()
        reg = FakeRegistry()
        register(reg, hooks)
        assert _HOOK_NAME_PRE in reg.hooks
        assert _HOOK_NAME_POST in reg.hooks
        assert reg.hooks[_HOOK_NAME_PRE]["when"] == "pre_task"
        assert reg.hooks[_HOOK_NAME_POST]["when"] == "post_task"

    def test_register_is_idempotent(self):
        hooks = _make_hooks()
        reg = FakeRegistry()
        register(reg, hooks)
        register(reg, hooks)  # second call must not double-up
        assert len(reg.hooks) == 2

    def test_unregister_removes_both(self):
        hooks = _make_hooks()
        reg = FakeRegistry()
        register(reg, hooks)
        unregister(reg)
        assert reg.hooks == {}

    def test_registry_without_register_raises_type_error(self):
        hooks = _make_hooks()

        class Broken:
            pass

        with pytest.raises(TypeError):
            register(Broken(), hooks)

    def test_registry_without_priority_kwarg_falls_back(self):
        hooks = _make_hooks()

        class NoPriority:
            def __init__(self):
                self.hooks = {}

            def register(self, name, when, fn):
                self.hooks[name] = fn

            def unregister(self, name):
                self.hooks.pop(name, None)

        reg = NoPriority()
        register(reg, hooks)
        assert _HOOK_NAME_PRE in reg.hooks
        assert _HOOK_NAME_POST in reg.hooks


# ── Behavioural wiring ───────────────────────────────────────────

class TestBehaviour:
    def test_block_suppressed_by_default_on_allow(self):
        """Default dispatch is ALLOW — so no dispatch line in the
        injection block (avoid noise).
        """
        hooks = _make_hooks()
        out = hooks.pre_task(FakeAgent(), FakeTask())
        assert "dispatch:" not in out

    def test_block_surfaces_in_injection_on_block_verdict(self):
        """If a BLOCK occurs, the injection must name it so the
        agent cannot silently proceed.
        """
        def blocking_decider(consulted_beliefs, suspended_candidates,
                             task_description, crew_name):
            from app.subia.belief.dispatch_gate import DispatchDecision
            return DispatchDecision(
                verdict="BLOCK",
                reason="synthetic block for test",
            )

        loop = SubIALoop(
            kernel=SubjectivityKernel(),
            scene_gate=CompetitiveGate(capacity=5),
            predict_fn=lambda ctx: Prediction(
                id="p", operation="o", predicted_outcome={},
                predicted_self_change={}, predicted_homeostatic_effect={},
                confidence=0.7, created_at="",
            ),
            consult_fn=lambda **_: [],
            dispatch_decider=blocking_decider,
        )
        hooks = SubIALifecycleHooks(loop=loop)
        out = hooks.pre_task(FakeAgent(), FakeTask())
        assert "dispatch: BLOCK" in out
        assert "synthetic block for test" in out
