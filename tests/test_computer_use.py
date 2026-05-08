"""
test_computer_use — budget guard, audit logger, runner loop.

The runner uses an injected fake Anthropic client + fake backend so we
exercise the full step loop without launching Chromium or hitting the API.
"""
from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """Redirect budget + audit + steps files to a tmp dir."""
    from app.computer_use import budget as bm
    from app.computer_use import audit as am
    monkeypatch.setattr(bm, "_STORE_PATH", tmp_path / "spend.json")
    monkeypatch.setattr(am, "_STEPS_PATH", tmp_path / "steps.jsonl")
    yield


@pytest.fixture(autouse=True)
def _force_cap(monkeypatch):
    """Pin the monthly cap to a known value for deterministic checks."""
    from app.computer_use import budget as bm
    monkeypatch.setattr(bm, "get_monthly_cap_usd", lambda: 10.0)
    yield


# ── Budget tracker ────────────────────────────────────────────────────────

def test_estimate_cost_haiku_pricing():
    from app.computer_use.budget import estimate_cost_usd
    cost = estimate_cost_usd({
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    })
    # 1M input @ $1 + 1M output @ $5
    assert cost == pytest.approx(6.0)


def test_estimate_cost_with_prompt_cache():
    from app.computer_use.budget import estimate_cost_usd
    cost = estimate_cost_usd({
        "input_tokens": 0,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
    })
    # 1M cache write @ $1.25 + 1M cache read @ $0.10 + 1M output @ $5
    assert cost == pytest.approx(6.35)


def test_check_can_start_passes_at_zero():
    from app.computer_use.budget import check_can_start
    check_can_start()  # should not raise


def test_check_can_start_blocks_over_cap():
    from app.computer_use.budget import (
        BudgetExceeded, check_can_start, record_task_cost,
    )
    record_task_cost("seed", cost_usd=11.0, steps=1, success=True)
    with pytest.raises(BudgetExceeded) as exc:
        check_can_start()
    assert exc.value.scope == "monthly"


def test_per_task_cap_blocks_runaway():
    from app.computer_use.budget import (
        BudgetExceeded, MAX_USD_PER_TASK, check_step_within_budget,
    )
    with pytest.raises(BudgetExceeded) as exc:
        check_step_within_budget(MAX_USD_PER_TASK + 0.01)
    assert exc.value.scope == "per-task"


def test_record_task_cost_accumulates(tmp_path):
    from app.computer_use.budget import record_task_cost, snapshot
    record_task_cost("first", cost_usd=0.10, steps=3, success=True)
    record_task_cost("second", cost_usd=0.05, steps=2, success=False)
    snap = snapshot()
    assert snap["spent_usd"] == pytest.approx(0.15)
    assert snap["task_count"] == 2
    # Newest first in recent_tasks.
    assert snap["recent_tasks"][0]["summary"] == "second"


def test_snapshot_remaining_clamped_at_zero(monkeypatch):
    from app.computer_use.budget import record_task_cost, snapshot
    record_task_cost("blow", cost_usd=15.0, steps=1, success=True)
    snap = snapshot()
    assert snap["remaining_usd"] == 0.0


# ── Audit logger ──────────────────────────────────────────────────────────

def test_audit_log_step_writes_jsonl():
    import json as _json
    from app.computer_use.audit import log_step, recent_steps
    log_step(1, "left_click", payload={"coordinate": [100, 200]},
             result="left_click(100,200)", screenshot_kb=42, cost_usd=0.001)
    log_step(2, "type", payload={"text": "hello"}, result="type(5 chars)")
    rows = recent_steps()
    assert len(rows) == 2
    assert rows[0]["action"] == "left_click"
    assert rows[1]["payload"]["text"] == "hello"


# ── Tool factory ──────────────────────────────────────────────────────────

def test_factory_returns_empty_when_disabled(monkeypatch):
    from app.tools.computer_use_tool import create_computer_use_tools
    monkeypatch.setattr("app.runtime_settings.get_vision_cu_enabled", lambda: False)
    assert create_computer_use_tools() == []


def test_factory_returns_empty_when_anthropic_missing(monkeypatch):
    """The factory must gracefully degrade if the Anthropic SDK can't import."""
    monkeypatch.setattr("app.runtime_settings.get_vision_cu_enabled", lambda: True)
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "anthropic":
            raise ImportError("not for this test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from app.tools.computer_use_tool import create_computer_use_tools
    assert create_computer_use_tools() == []


# ── Runner loop ───────────────────────────────────────────────────────────

class _FakeBackend:
    """Recorder backend the runner can drive without a real browser."""

    def __init__(self):
        self.actions: list[str] = []
        self._closed = False

    def start(self, *, start_url: str = "about:blank"):
        self.actions.append(f"start({start_url})")

    def close(self):
        self._closed = True

    def screenshot(self) -> bytes:
        return b"\x89PNG_FAKE"

    def perform(self, action):
        kind = action.get("action", "?")
        self.actions.append(kind)
        return f"did_{kind}"


class _FakeContent:
    def __init__(self, kind: str, **kw):
        self.type = kind
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeUsage:
    input_tokens = 0
    output_tokens = 100
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage = _FakeUsage()


def _make_fake_client(scripted: list[list[Any]]):
    """A fake Anthropic client that returns scripted responses turn by turn."""
    turns = list(scripted)
    client = MagicMock()

    def fake_create(**kwargs):
        if not turns:
            raise AssertionError("scripted responses exhausted")
        return _FakeResponse(turns.pop(0))

    client.messages.create.side_effect = fake_create
    return client


def test_runner_executes_one_tool_call_then_finalises():
    from app.computer_use.runner import run_task
    backend = _FakeBackend()
    scripted = [
        # Turn 1: click + screenshot tool call
        [
            _FakeContent("text", text="I will click the button."),
            _FakeContent("tool_use", id="tu_1", name="computer",
                         input={"action": "left_click", "coordinate": [100, 200]}),
        ],
        # Turn 2: model finalises
        [_FakeContent("text", text="Done — clicked the target.")],
    ]
    client = _make_fake_client(scripted)
    result = run_task(
        "click the button at 100,200",
        backend=backend,
        client_factory=lambda: client,
    )
    assert result.success is True
    assert "Done" in result.text
    assert result.steps == 2
    # Backend got the click.
    assert "left_click" in backend.actions


def test_runner_caps_at_max_steps(monkeypatch):
    """If the model never finalises, the loop must stop at MAX_STEPS_PER_TASK."""
    from app.computer_use.runner import run_task
    backend = _FakeBackend()
    looping_response = [
        _FakeContent("tool_use", id="tu_loop", name="computer",
                     input={"action": "screenshot"}),
    ]
    # 3 turns max so the test runs fast.
    scripted = [list(looping_response) for _ in range(3)]
    client = _make_fake_client(scripted)
    result = run_task(
        "loop forever",
        backend=backend,
        client_factory=lambda: client,
        max_steps=3,
    )
    # The model never sent a text-only finalisation.
    assert result.success is False
    assert result.steps == 3


def test_runner_refuses_when_monthly_cap_reached():
    from app.computer_use.budget import record_task_cost
    from app.computer_use.runner import run_task
    record_task_cost("seed_overflow", cost_usd=15.0, steps=1, success=True)
    result = run_task(
        "do anything",
        backend=_FakeBackend(),
        client_factory=lambda: MagicMock(),
    )
    assert result.success is False
    assert "monthly" in result.refused_reason.lower()


def test_runner_records_task_in_budget_ledger():
    from app.computer_use.runner import run_task
    from app.computer_use.budget import snapshot
    backend = _FakeBackend()
    scripted = [[_FakeContent("text", text="trivial answer")]]
    client = _make_fake_client(scripted)
    run_task(
        "trivial",
        backend=backend,
        client_factory=lambda: client,
    )
    snap = snapshot()
    assert snap["task_count"] == 1
    assert snap["recent_tasks"][0]["summary"] == "trivial"
