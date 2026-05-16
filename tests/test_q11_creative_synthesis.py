"""PROGRAM §46.18-§46.22 — Q11 creative synthesis tests.

Covers:

  §46.18 Q11.1 — analogy_populator (HEAVY weekly LLM extraction) +
                  brainstorm consumer wiring
  §46.19 Q11.2 — concept_blend as 8th brainstorm technique with
                  Fauconnier-Turner LLM operator integration
  §46.20 Q11.4 — novelty verdict folded into brainstorm report
  §46.21 Q11.5 — aesthetic score folded into brainstorm report
  §46.22 Q11.3 — population-based idea_evolution standalone module
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# ─────────────────────────────────────────────────────────────────────
#   §46.18 — analogy_populator
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_workspace(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


def test_analogy_populator_parses_strict_json() -> None:
    from app.creativity.analogy_populator import _parse_llm_output
    raw = (
        '{"structure_signature": "feedback_loop_with_delay",'
        ' "structure_description": "A system where output influences'
        ' input after a delay creating oscillation.",'
        ' "domain_examples": [{"domain": "ecology", "title": "predator-prey",'
        ' "summary": "Lynx-hare cycles."},'
        ' {"domain": "economics", "title": "stock-flow misperception",'
        ' "summary": "Bathtub dynamics in inventory."},'
        ' {"domain": "control_theory", "title": "PI controller overshoot",'
        ' "summary": "Integral term + lag = overshoot."}]}'
    )
    data = _parse_llm_output(raw)
    assert data is not None
    assert data["structure_signature"] == "feedback_loop_with_delay"
    assert len(data["domain_examples"]) == 3


def test_analogy_populator_parses_fenced_json() -> None:
    """LLM wraps in ```json``` fence → still parses."""
    from app.creativity.analogy_populator import _parse_llm_output
    raw = (
        "```json\n"
        '{"structure_signature": "x", "structure_description": "y",'
        ' "domain_examples": []}\n'
        "```"
    )
    data = _parse_llm_output(raw)
    assert data is not None
    assert data["structure_signature"] == "x"


def test_analogy_populator_refuses_when_too_few_examples(
    tmp_workspace: Path, monkeypatch,
) -> None:
    """≥2 cross-domain examples required to land an entry."""
    from app.creativity.analogy_populator import _extract_entry_from_text

    def fake_llm(system: str, user: str) -> str:
        return json.dumps({
            "structure_signature": "x",
            "structure_description": "abstract pattern",
            "domain_examples": [
                {"domain": "physics", "title": "t1", "summary": "s1"},
                # Only 1 example → should refuse
            ],
        })

    entry = _extract_entry_from_text("test:src", "some text", llm_call=fake_llm)
    assert entry is None


def test_analogy_populator_writes_entry_on_success(
    tmp_workspace: Path, monkeypatch,
) -> None:
    """Successful LLM output → entry appears in the analogy_index JSONL."""
    monkeypatch.setenv("ANALOGY_INDEX_ENABLED", "true")
    monkeypatch.setenv("ANALOGY_INDEX_POPULATOR_ENABLED", "true")
    # Redirect the index to tmp
    idx_path = tmp_workspace / "creativity" / "analogy_index.jsonl"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    from app.creativity import analogy_index as ai
    ai._reset_for_tests(idx_path)
    # Create a wiki text the populator will pick up
    wiki = tmp_workspace / "wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    (wiki / "note.md").write_text(
        "# Test note\n\nThis is a long enough body to clear the "
        "MIN_TEXT_CHARS gate of the populator pipeline so it is "
        "considered as a source candidate during the pass.\n\n"
        + ("padding " * 50),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANALOGY_POPULATOR_WIKI_ROOT", str(wiki))

    def fake_llm(system: str, user: str) -> str:
        return json.dumps({
            "structure_signature": "test_pattern",
            "structure_description": "an abstract pattern for testing",
            "domain_examples": [
                {"domain": "biology", "title": "t1", "summary": "s1"},
                {"domain": "physics", "title": "t2", "summary": "s2"},
                {"domain": "music", "title": "t3", "summary": "s3"},
            ],
        })

    from app.creativity.analogy_populator import run_one_pass
    result = run_one_pass(force=True, llm_call=fake_llm, max_new=2)
    assert result.status == "ok"
    assert result.new_entries >= 1
    # Verify the entry actually landed
    from app.creativity.analogy_index import list_all
    entries = list_all(path=idx_path)
    assert len(entries) >= 1
    assert any(e.structure_signature == "test_pattern" for e in entries)
    # Cleanup
    ai._reset_for_tests(None)


def test_analogy_populator_skipped_when_disabled(monkeypatch) -> None:
    """Source-level + behavioural check that the master switch flips
    OFF the populator. _enabled() reads runtime_settings first, env
    fallback — we patch _enabled directly for a clean per-test
    contract."""
    import app.creativity.analogy_populator as ap
    monkeypatch.setattr(ap, "_enabled", lambda: False)
    result = ap.run_one_pass()
    assert result.status == "skipped_disabled"


def test_brainstorm_facilitator_queries_analogy_index() -> None:
    """Source-level: facilitator.start calls query_analogies and
    stores results on session.analogues."""
    src = Path("app/brainstorm/facilitator.py").read_text(encoding="utf-8")
    assert "from app.creativity.analogy_index import query_analogies" in src
    assert "session.analogues" in src


def test_brainstorm_session_carries_analogues_field() -> None:
    src = Path("app/brainstorm/session.py").read_text(encoding="utf-8")
    assert "analogues" in src
    # to_dict / from_dict round-trip the field
    assert '"analogues": list(self.analogues)' in src
    assert 'analogues=list(data.get("analogues"' in src


def test_brainstorm_facilitator_injects_analogues_into_seed_prompt() -> None:
    src = Path("app/brainstorm/facilitator.py").read_text(encoding="utf-8")
    assert "_inject_analogues_into_prompt" in src
    assert "Cross-domain analogues" in src


# ─────────────────────────────────────────────────────────────────────
#   §46.19 — concept_blend technique
# ─────────────────────────────────────────────────────────────────────


def test_concept_blend_technique_registered() -> None:
    from app.brainstorm.techniques import get, names
    assert "concept-blend" in names()
    technique = get("concept-blend")
    assert technique is not None
    assert technique.name == "concept-blend"
    assert "Fauconnier" in technique.title or "Concept Blend" in technique.title


def test_concept_blend_has_four_steps() -> None:
    from app.brainstorm.techniques import get
    technique = get("concept-blend")
    assert len(technique.steps) == 4
    step_ids = [s.step_id for s in technique.steps]
    assert step_ids == ["input_a", "input_b", "generate_blend", "select_projections"]


def test_concept_blend_renders_blend_at_step_3(monkeypatch) -> None:
    """Step 3 (generate_blend) prompt includes the blend operator's
    output spliced in. Uses a fake blend result to avoid LLM calls."""
    from app.brainstorm.techniques import get
    from app.brainstorm.techniques.base import TechniqueState
    # Inject a fake blend_concepts that returns a deterministic result
    import app.brainstorm.techniques.concept_blend as cb

    class _FakeBlend:
        parse_failed = False
        blend_label = "test_blend_label"
        generic_structure = "abstract structure shared by A and B"
        blend_description = "two-paragraph description"
        selected_projections = ["from A: x", "from B: y"]
        emergent_structure = ["new emergent property z"]
        follow_on_questions = ["what about q?"]

    def fake_blend_concepts(input_a, input_b, **_):
        return _FakeBlend()

    monkeypatch.setattr(
        "app.creativity.concept_blend.blend_concepts",
        fake_blend_concepts,
        raising=False,
    )
    technique = get("concept-blend")
    state = technique.initial_state()
    # Walk to step 3
    technique.record_response(state, "input space A description")
    technique.record_response(state, "input space B description")
    prompt = technique.next_prompt(state, topic="example topic")
    assert "Generated blend" in prompt
    assert "test_blend_label" in prompt
    assert "new emergent property z" in prompt


def test_concept_blend_falls_back_on_blend_failure(monkeypatch) -> None:
    """When blend_concepts raises, prompt still contains a degraded
    notice and the technique walks on."""
    from app.brainstorm.techniques import get

    def boom(input_a, input_b, **_):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(
        "app.creativity.concept_blend.blend_concepts",
        boom, raising=False,
    )
    technique = get("concept-blend")
    state = technique.initial_state()
    technique.record_response(state, "A description")
    technique.record_response(state, "B description")
    prompt = technique.next_prompt(state, topic="topic")
    assert prompt is not None
    # Either the degraded notice OR the fallback static prompt
    assert "blend" in prompt.lower()


# ─────────────────────────────────────────────────────────────────────
#   §46.20 — novelty in report
# ─────────────────────────────────────────────────────────────────────


def test_annotate_text_returns_novelty_and_aesthetic_keys() -> None:
    from app.brainstorm.report import _annotate_text
    out = _annotate_text("a test idea about feedback loops")
    # Both keys present (values may be None when stores are empty)
    assert "novelty" in out
    assert "aesthetic_score" in out


def test_annotate_text_empty_returns_empty_dict() -> None:
    from app.brainstorm.report import _annotate_text
    assert _annotate_text("") == {}
    assert _annotate_text("   ") == {}


def test_annotate_ideas_walks_response_list() -> None:
    from app.brainstorm.report import _annotate_ideas
    responses = [
        {"role": "researcher", "text": "an interesting idea"},
        {"role": "writer", "text": "another distinct idea"},
    ]
    annotated = _annotate_ideas(responses)
    assert len(annotated) == 2
    for a in annotated:
        assert "annotation" in a
        assert a["text"] != ""


def test_writer_prompt_explains_annotation_legend() -> None:
    src = Path("app/brainstorm/report.py").read_text(encoding="utf-8")
    assert "Annotation legend" in src
    assert "annotation.novelty.verdict" in src
    assert "annotation.aesthetic_score" in src


# ─────────────────────────────────────────────────────────────────────
#   §46.21 — aesthetic_score
# ─────────────────────────────────────────────────────────────────────


def test_aesthetic_score_returns_none_on_empty_text() -> None:
    from app.creativity.aesthetic_score import score
    assert score("") is None
    assert score("   ") is None


def test_aesthetic_score_handles_store_error_gracefully(monkeypatch) -> None:
    """When the aesthetics store raises, score returns None — never
    propagates the exception to the brainstorm pipeline. The
    aesthetics package import itself may fail in the test env
    (chromadb deps); that's also a None outcome by design."""
    from app.creativity import aesthetic_score as asc
    # Force the import branch to fail
    import sys
    monkeypatch.setitem(
        sys.modules, "app.aesthetics.vectorstore", None,
    )
    assert asc.score("some text") is None


def test_aesthetic_score_normalises_quality_score() -> None:
    """quality_score stored as 0..10 in metadata is normalised to 0..1.
    Verified at the source-level for clarity."""
    src = Path("app/creativity/aesthetic_score.py").read_text(encoding="utf-8")
    # The normalisation branch must be present
    assert "q_val / 10.0" in src
    assert "q_val > 1.0" in src


# ─────────────────────────────────────────────────────────────────────
#   §46.22 — idea_evolution
# ─────────────────────────────────────────────────────────────────────


def test_idea_evolution_disabled_returns_empty_result(monkeypatch) -> None:
    monkeypatch.setenv("IDEA_EVOLUTION_ENABLED", "false")
    from app.brainstorm.idea_evolution import evolve_ideas
    result = evolve_ideas(
        task="X", seed_ideas=["one"], generations=1,
    )
    assert result.truncated_reason == "disabled"
    assert result.population == []


def test_idea_evolution_requires_seed_ideas() -> None:
    from app.brainstorm.idea_evolution import evolve_ideas
    result = evolve_ideas(task="X", seed_ideas=[])
    assert result.truncated_reason == "no_seed_ideas"


def test_idea_evolution_runs_with_injected_hooks() -> None:
    """Inject deterministic mutator + judge; verify the loop runs and
    produces a scored population."""
    from app.brainstorm.idea_evolution import evolve_ideas

    counter = {"n": 0}

    def fake_mutator(task, parent, neighbours):
        counter["n"] += 1
        return f"{parent} +variant{counter['n']}"

    def fake_judge(task, idea, constraints):
        # Score by length (silly but deterministic)
        return min(1.0, len(idea) / 100.0), f"len={len(idea)}"

    result = evolve_ideas(
        task="design a thing",
        seed_ideas=["seed A", "seed B", "seed C"],
        generations=2,
        population_size=4,
        budget_usd=2.0,
        mutator_fn=fake_mutator,
        judge_fn=fake_judge,
    )
    assert result.generations_run >= 1
    assert len(result.population) > 0
    assert all(0.0 <= m.score <= 1.0 for m in result.population)
    # At least one mutate call happened (we asked for population_size=4
    # with only 3 seeds, so one mutate at minimum to pad)
    assert result.mutate_calls >= 1
    # Top ideas roundtrip
    tops = result.top_ideas(n=3)
    assert len(tops) >= 1


def test_idea_evolution_caps_generations() -> None:
    """generations=999 is clamped to MAX_GENERATIONS."""
    from app.brainstorm.idea_evolution import evolve_ideas, MAX_GENERATIONS

    def fake_mutator(task, parent, neighbours):
        return parent + "_m"

    def fake_judge(task, idea, constraints):
        return 0.5, "x"

    result = evolve_ideas(
        task="t",
        seed_ideas=["s1", "s2", "s3"],
        generations=999,
        population_size=3,
        budget_usd=2.0,
        mutator_fn=fake_mutator,
        judge_fn=fake_judge,
    )
    assert result.generations_run <= MAX_GENERATIONS


def test_idea_evolution_budget_exhaustion() -> None:
    """A miserly budget with large population + many generations
    terminates early with the right reason. At the hard-coded
    per-call cost estimates, max-config produces ~$0.30-0.50 of
    spend, so a $0.05 budget definitely trips the guard."""
    from app.brainstorm.idea_evolution import (
        MAX_GENERATIONS, MAX_POPULATION, evolve_ideas,
    )

    def fake_mutator(task, parent, neighbours):
        return parent + "_m"

    def fake_judge(task, idea, constraints):
        return 0.5, "x"

    result = evolve_ideas(
        task="t",
        seed_ideas=["s1", "s2"],
        generations=MAX_GENERATIONS,
        population_size=MAX_POPULATION,
        budget_usd=0.05,  # min — runs out quickly under max config
        mutator_fn=fake_mutator,
        judge_fn=fake_judge,
    )
    assert "budget" in result.truncated_reason


def test_idea_member_to_dict_roundtrip() -> None:
    from app.brainstorm.idea_evolution import IdeaMember
    m = IdeaMember(id="abc", text="idea", score=0.7, generation=2)
    d = m.to_dict()
    assert d["id"] == "abc"
    assert d["score"] == 0.7


# ─────────────────────────────────────────────────────────────────────
#   React + runtime_settings wiring
# ─────────────────────────────────────────────────────────────────────


def test_runtime_settings_defines_analogy_populator_key() -> None:
    src = Path("app/runtime_settings.py").read_text(encoding="utf-8")
    assert '"analogy_index_populator_enabled": True' in src
    assert "def get_analogy_index_populator_enabled" in src
    assert "def set_analogy_index_populator_enabled" in src


def test_config_api_handles_analogy_populator_key() -> None:
    src = Path("app/api/config_api.py").read_text(encoding="utf-8")
    assert "set_analogy_index_populator_enabled" in src
    assert '"analogy_index_populator_enabled" in payload' in src


def test_idle_scheduler_registers_analogy_populator_job() -> None:
    src = Path("app/idle_scheduler.py").read_text(encoding="utf-8")
    assert "analogy-populator" in src
    assert "JobWeight.HEAVY" in src
    # Cross-reference: the registration is right after meta-evolution
    meta_idx = src.find("meta-evolution")
    analogy_idx = src.find("analogy-populator")
    assert analogy_idx > meta_idx


def test_react_card_imported_and_mounted() -> None:
    settings = Path(
        "dashboard-react/src/components/SettingsPage.tsx",
    ).read_text(encoding="utf-8")
    assert "import { AnalogyIndexCard }" in settings
    assert "<AnalogyIndexCard" in settings
    card = Path(
        "dashboard-react/src/components/AnalogyIndexCard.tsx",
    ).read_text(encoding="utf-8")
    assert "analogy_index_populator_enabled" in card


def test_queries_type_carries_analogy_field() -> None:
    src = Path("dashboard-react/src/api/queries.ts").read_text(encoding="utf-8")
    assert "analogy_index_populator_enabled" in src
