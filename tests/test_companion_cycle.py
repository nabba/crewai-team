"""Tests for app.companion.cycle — Creative MAS wiring + cost capture."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.companion import cycle as _cycle
from app.companion import workspace_kb
from app.companion.config import CompanionConfig


def _make_creative_result(*, p1=3, p2=2, final="The idea", cost=0.05,
                          aborted=None, scores=None):
    return SimpleNamespace(
        final_output=final,
        phase_1_outputs=[SimpleNamespace(role=f"r{i}", text="x", duration_s=1.0)
                         for i in range(p1)],
        phase_2_outputs=[SimpleNamespace(role=f"r{i}", text="y", duration_s=1.0)
                         for i in range(p2)],
        cost_usd=cost,
        aborted_reason=aborted,
        scores=scores,
    )


def test_no_seed_returns_aborted():
    cfg = CompanionConfig(seed_prompt=None).clamp()
    result = _cycle.run_cycle("ws-1", cfg)
    assert result.aborted_reason == "no_seed_prompt"
    assert result.cost_usd == 0.0
    assert result.phase_1_count == 0


def test_blank_seed_returns_aborted():
    cfg = CompanionConfig(seed_prompt="   ").clamp()
    result = _cycle.run_cycle("ws-1", cfg)
    assert result.aborted_reason == "no_seed_prompt"


def test_run_cycle_happy_path():
    cfg = CompanionConfig(seed_prompt="Estonian forests").clamp()
    fake = _make_creative_result(p1=4, p2=3, final="abc", cost=0.07,
                                  scores={"originality": 0.8})

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake):
        result = _cycle.run_cycle("ws-1", cfg)

    assert result.aborted_reason is None
    assert result.phase_1_count == 4
    assert result.phase_2_count == 3
    assert result.final_output == "abc"
    assert result.final_output_chars == 3
    assert result.cost_usd == pytest.approx(0.07)
    assert result.creative_scores == {"originality": 0.8}


def test_run_cycle_passes_seed_into_prompt():
    """The seed and any KB snippets must reach Creative MAS via task_description."""
    cfg = CompanionConfig(seed_prompt="forests of Estonia").clamp()
    snippets = [
        workspace_kb.KBSnippet(text="ecology fact", score=0.9, source="episteme"),
    ]
    captured = {}

    def _capture(task_description):
        captured["task"] = task_description
        return _make_creative_result()

    with patch("app.companion.workspace_kb.compose", lambda **kw: snippets), \
         patch("app.companion.cycle._invoke_creative_crew", _capture):
        _cycle.run_cycle("ws-1", cfg)

    assert "forests of Estonia" in captured["task"]
    assert "ecology fact" in captured["task"]
    assert "episteme" in captured["task"]


def test_run_cycle_handles_creative_crew_failure():
    cfg = CompanionConfig(seed_prompt="forests").clamp()

    def _broken(*a, **kw):
        raise RuntimeError("LLM down")

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew", _broken):
        result = _cycle.run_cycle("ws-1", cfg)

    assert result.aborted_reason is not None
    assert "creative_crew_failed" in result.aborted_reason
    assert result.cost_usd == 0.0


def test_run_cycle_propagates_creative_aborted_reason():
    cfg = CompanionConfig(seed_prompt="x").clamp()
    fake = _make_creative_result(aborted="budget exceeded")

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew", lambda *a, **kw: fake):
        result = _cycle.run_cycle("ws-1", cfg)

    assert result.aborted_reason == "budget exceeded"


def test_compose_prompt_skips_context_when_no_snippets():
    prompt = _cycle._compose_prompt("forests", [])
    assert "## Context" not in prompt
    assert "forests" in prompt


def test_compose_prompt_includes_context_when_snippets_present():
    snippets = [
        workspace_kb.KBSnippet(text="a", score=0.5, source="episteme"),
    ]
    prompt = _cycle._compose_prompt("seed", snippets)
    assert "## Context" in prompt
    assert "## Workspace seed\nseed" in prompt
    assert "## Task" in prompt


def test_compose_prompt_drops_empty_text_snippets():
    snippets = [
        workspace_kb.KBSnippet(text="", score=0, source="temporal_context"),
        workspace_kb.KBSnippet(text="real content", score=0.5, source="episteme"),
    ]
    prompt = _cycle._compose_prompt("seed", snippets)
    assert "real content" in prompt
    # The body of the temporal_context snippet was empty; nothing for it
    # to render. Only the episteme line shows up under ## Context.
    assert prompt.count("[temporal_context") == 0
