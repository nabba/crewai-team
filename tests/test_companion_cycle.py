"""Tests for app.companion.cycle — Creative MAS wiring + cost capture."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.companion import cycle as _cycle
from app.companion import events as _events
from app.companion import workspace_kb
from app.companion.config import CompanionConfig


@pytest.fixture(autouse=True)
def _isolate_events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each cycle test gets its own events log so cooldown checks are clean."""
    monkeypatch.setattr(_events, "_EVENTS_DIR", tmp_path / "events")


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


# ── Phase 3: persistence + scoring ─────────────────────────────────────────

def test_run_cycle_persists_converged_idea_with_scores():
    cfg = CompanionConfig(seed_prompt="forests").clamp()
    fake = _make_creative_result(p1=2, p2=2, final="converged synthesis",
                                  cost=0.05)
    persisted: list = []

    def _fake_persist(rec):
        persisted.append(rec)
        return rec.idea_id

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake), \
         patch("app.companion.idea_store.persist", _fake_persist), \
         patch("app.companion.scoring.compute_novelty", lambda *a, **kw: 0.8), \
         patch("app.companion.scoring.compute_quality", lambda t: 0.7), \
         patch("app.companion.scoring.compute_transferability", lambda t: 0.6):
        result = _cycle.run_cycle("ws-1", cfg)

    # 2 fragments + 2 developed + 1 converged = 5 records.
    assert len(persisted) == 5
    assert result.converged_idea_id is not None
    assert result.novelty == pytest.approx(0.8)
    assert result.quality == pytest.approx(0.7)
    assert result.transferability == pytest.approx(0.6)
    # Cycle id propagates to all records.
    assert all(r.cycle_id == result.cycle_id for r in persisted)


def test_run_cycle_persists_lineage_correctly():
    """Fragments → developed (parents=fragments) → converged (parents=developed)."""
    cfg = CompanionConfig(seed_prompt="forests").clamp()
    fake = _make_creative_result(p1=3, p2=2, final="final")
    persisted: list = []

    def _fake_persist(rec):
        persisted.append(rec)
        return rec.idea_id

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake), \
         patch("app.companion.idea_store.persist", _fake_persist), \
         patch("app.companion.scoring.compute_novelty", lambda *a, **kw: 0.0), \
         patch("app.companion.scoring.compute_quality", lambda t: 0.0), \
         patch("app.companion.scoring.compute_transferability", lambda t: 0.0):
        result = _cycle.run_cycle("ws-1", cfg)

    from app.companion.idea_store import IdeaState
    fragments = [r for r in persisted if r.state == IdeaState.FRAGMENT]
    developed = [r for r in persisted if r.state == IdeaState.DEVELOPED]
    converged = [r for r in persisted if r.state == IdeaState.CONVERGED]

    assert len(fragments) == 3
    assert len(developed) == 2
    assert len(converged) == 1

    # Fragments have no parents.
    assert all(r.lineage_parents == [] for r in fragments)
    # Developed parents = ALL fragment ids.
    fragment_ids = [r.idea_id for r in fragments]
    for r in developed:
        assert sorted(r.lineage_parents) == sorted(fragment_ids)
    # Converged parents = developed ids.
    developed_ids = [r.idea_id for r in developed]
    assert sorted(converged[0].lineage_parents) == sorted(developed_ids)

    # CycleResult exposes the same ids in the right slots.
    assert sorted(result.fragment_ids) == sorted(fragment_ids)
    assert sorted(result.developed_ids) == sorted(developed_ids)
    assert result.converged_idea_id == converged[0].idea_id


def test_run_cycle_skips_persistence_when_aborted():
    cfg = CompanionConfig(seed_prompt="x").clamp()
    fake = _make_creative_result(aborted="budget exceeded")
    persisted: list = []

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake), \
         patch("app.companion.idea_store.persist",
               lambda r: persisted.append(r) or r.idea_id):
        result = _cycle.run_cycle("ws-1", cfg)

    assert persisted == []
    assert result.converged_idea_id is None


def test_run_cycle_skips_persistence_when_empty_final():
    cfg = CompanionConfig(seed_prompt="x").clamp()
    fake = _make_creative_result(final="   ")
    persisted: list = []

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake), \
         patch("app.companion.idea_store.persist",
               lambda r: persisted.append(r) or r.idea_id):
        result = _cycle.run_cycle("ws-1", cfg)

    assert persisted == []
    assert result.converged_idea_id is None


def test_run_cycle_emits_cycle_id_even_on_abort():
    cfg = CompanionConfig(seed_prompt=None).clamp()
    result = _cycle.run_cycle("ws-1", cfg)
    assert result.cycle_id.startswith("cyc_")
    assert result.aborted_reason == "no_seed_prompt"


# ── Phase 4: surfacing ─────────────────────────────────────────────────────

def test_run_cycle_surfaces_when_eligible():
    cfg = CompanionConfig(seed_prompt="forests",
                           novelty_threshold=0.5,
                           surface_threshold=0.5).clamp()
    fake = _make_creative_result(p1=2, p2=1, final="A solid idea body")

    surface_calls: list = []

    def _surface(idea, config):
        surface_calls.append(idea.idea_id)
        return True

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake), \
         patch("app.companion.idea_store.persist",
               lambda r: r.idea_id), \
         patch("app.companion.scoring.compute_novelty",
               lambda *a, **kw: 0.9), \
         patch("app.companion.scoring.compute_quality", lambda t: 0.9), \
         patch("app.companion.scoring.compute_transferability",
               lambda t: 0.5), \
         patch("app.companion.surfacing.surface", _surface):
        result = _cycle.run_cycle("ws-1", cfg)

    assert result.surfaced is True
    assert result.surface_reason == "ok"
    assert len(surface_calls) == 1


def test_run_cycle_does_not_surface_below_threshold():
    cfg = CompanionConfig(seed_prompt="forests",
                           novelty_threshold=0.7,
                           surface_threshold=0.7).clamp()
    fake = _make_creative_result(p1=1, p2=1, final="lukewarm idea")
    surface_calls: list = []

    def _surface(idea, config):
        surface_calls.append(idea.idea_id)
        return True

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake), \
         patch("app.companion.idea_store.persist",
               lambda r: r.idea_id), \
         patch("app.companion.scoring.compute_novelty",
               lambda *a, **kw: 0.4), \
         patch("app.companion.scoring.compute_quality", lambda t: 0.5), \
         patch("app.companion.scoring.compute_transferability",
               lambda t: 0.5), \
         patch("app.companion.surfacing.surface", _surface):
        result = _cycle.run_cycle("ws-1", cfg)

    assert result.surfaced is False
    assert result.surface_reason == "below_novelty"
    assert surface_calls == []


def test_run_cycle_skips_surfacing_when_aborted():
    cfg = CompanionConfig(seed_prompt="x").clamp()
    fake = _make_creative_result(aborted="budget exceeded")
    surface_calls: list = []

    def _surface(idea, config):
        surface_calls.append(idea.idea_id)
        return True

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake), \
         patch("app.companion.surfacing.surface", _surface):
        result = _cycle.run_cycle("ws-1", cfg)

    assert result.surfaced is False
    assert result.surface_reason == "not_attempted"
    assert surface_calls == []


def test_run_cycle_handles_surfacing_exception():
    cfg = CompanionConfig(seed_prompt="x",
                           novelty_threshold=0.0,
                           surface_threshold=0.0).clamp()
    fake = _make_creative_result(p1=1, p2=1, final="ok body")

    def _broken(idea, config):
        raise RuntimeError("send blew up")

    with patch("app.companion.workspace_kb.compose", lambda **kw: []), \
         patch("app.companion.cycle._invoke_creative_crew",
               lambda *a, **kw: fake), \
         patch("app.companion.idea_store.persist",
               lambda r: r.idea_id), \
         patch("app.companion.scoring.compute_novelty",
               lambda *a, **kw: 0.9), \
         patch("app.companion.scoring.compute_quality", lambda t: 0.9), \
         patch("app.companion.scoring.compute_transferability",
               lambda t: 0.5), \
         patch("app.companion.surfacing.surface", _broken):
        result = _cycle.run_cycle("ws-1", cfg)

    assert result.surfaced is False
    assert "surface_failed" in result.surface_reason
