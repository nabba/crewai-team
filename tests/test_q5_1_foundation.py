"""PROGRAM §43.1 — Q5.1 foundation tests.

Covers:
  * philosophy.dialectics.consult_panel — return shape, cache, gating
  * panel_bridge.file_unresolved_tensions — duck-typed, dedup, fail-open
  * identity.relevant_history.classify_path — full taxonomy coverage
  * identity.relevant_history.relevant_history_by_kind — outcome buckets
  * The three wires (Tier-3 amendment, identity ratification, calibration)
    pass panel + governor results through their evidence channels
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_runtime_settings_or_skip():
    """Skip tests that require app.runtime_settings on dev environments
    missing pydantic_settings. CI has the full dep set."""
    try:
        import app.runtime_settings  # noqa: F401
    except Exception as exc:
        pytest.skip(f"app.runtime_settings unavailable: {exc}")


# ─────────────────────────────────────────────────────────────────────────
#   dialectics.consult_panel
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def dialectics():
    return _load_isolated(
        "dialectics_q51",
        "app/philosophy/dialectics.py",
    )


def test_consult_panel_returns_skipped_when_disabled(dialectics, monkeypatch):
    """Master switch off → empty perspectives + skipped_reason='disabled'."""
    monkeypatch.setattr(dialectics, "_panel_enabled", lambda: False)
    result = dialectics.consult_panel("Should we ship Q5?")
    assert result.skipped_reason == "disabled"
    assert result.perspectives == []
    assert result.unresolved_tensions == []


def test_consult_panel_returns_skipped_when_kb_empty(dialectics, monkeypatch):
    """Neo4j returns no chains → skipped_reason='kb_empty'."""
    monkeypatch.setattr(dialectics, "_panel_enabled", lambda: True)
    class _StubGraph:
        def find_dialectical_chain(self, q, n=5):
            return []
    monkeypatch.setattr(dialectics, "get_graph", lambda: _StubGraph())
    result = dialectics.consult_panel("foo", use_cache=False)
    assert result.skipped_reason == "kb_empty"
    assert result.coverage == 0.0


def test_consult_panel_handles_empty_question(dialectics):
    """Empty question → skipped with empty_question reason."""
    result = dialectics.consult_panel("")
    assert result.skipped_reason == "empty_question"


def test_consult_panel_builds_perspectives_per_tradition(dialectics, monkeypatch):
    """Each requested tradition produces a perspective when KB has a chain
    in that tradition."""
    monkeypatch.setattr(dialectics, "_panel_enabled", lambda: True)
    class _StubGraph:
        def find_dialectical_chain(self, q, n=5):
            return [
                {
                    "claim": "Endure with reason",
                    "claim_tradition": "Stoicism",
                    "counter_claim": "Maximize utility",
                    "counter_tradition": "Utilitarianism",
                    "synthesis": "Endure when utility-positive",
                },
                {
                    "claim": "Cultivate excellence",
                    "claim_tradition": "Virtue ethics",
                    "counter_claim": "Categorical duty",
                    "counter_tradition": "Kantian",
                    "synthesis": None,  # unresolved
                },
            ]
    monkeypatch.setattr(dialectics, "get_graph", lambda: _StubGraph())
    result = dialectics.consult_panel(
        "Should I act?",
        traditions=["Stoicism", "Utilitarianism", "Virtue ethics", "Kantian"],
        use_cache=False,
    )
    assert result.skipped_reason is None
    assert len(result.perspectives) >= 2
    # Stoicism perspective with synthesis.
    stoic = next((p for p in result.perspectives if p.tradition == "Stoicism"), None)
    assert stoic is not None
    assert stoic.synthesis is not None
    # Virtue ethics perspective without synthesis → unresolved entry.
    virtue = next((p for p in result.perspectives if p.tradition == "Virtue ethics"), None)
    assert virtue is not None
    assert virtue.synthesis is None
    # At least one unresolved tension recorded (the Virtue/Kantian pair).
    assert any("Virtue ethics" in u or "Kantian" in u for u in result.unresolved_tensions)


def test_consult_panel_cache_round_trip(dialectics, monkeypatch, tmp_path):
    """Second consult with same question returns cache_hit=True."""
    monkeypatch.setattr(dialectics, "_panel_enabled", lambda: True)
    monkeypatch.setattr(
        dialectics, "_panel_cache_path",
        lambda: tmp_path / "cache.jsonl",
    )
    class _StubGraph:
        calls = 0
        def find_dialectical_chain(self, q, n=5):
            _StubGraph.calls += 1
            return [{
                "claim": "X", "claim_tradition": "Stoicism",
                "counter_claim": "Y", "counter_tradition": "Utilitarianism",
                "synthesis": "Z",
            }]
    monkeypatch.setattr(dialectics, "get_graph", lambda: _StubGraph())
    first = dialectics.consult_panel("test cache", traditions=["Stoicism"])
    second = dialectics.consult_panel("test cache", traditions=["Stoicism"])
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert _StubGraph.calls == 1  # Neo4j only called once
    # No-cache path bypasses.
    third = dialectics.consult_panel(
        "test cache", traditions=["Stoicism"], use_cache=False,
    )
    assert third.cache_hit is False


def test_panel_result_to_dict_round_trip(dialectics):
    """to_dict produces the operator-surface shape we promised."""
    panel = dialectics.PanelResult(
        question="q?",
        perspectives=[dialectics.PerspectiveTension(
            tradition="Stoicism", claim="x", counter_claim=None,
            synthesis=None, source="", confidence=1.0,
        )],
        unresolved_tensions=["foo"],
        coverage=0.5,
        consulted_at="2026-05-13T00:00:00+00:00",
    )
    d = panel.to_dict()
    assert d["question"] == "q?"
    assert d["coverage"] == 0.5
    assert len(d["perspectives"]) == 1
    assert d["unresolved_tensions"] == ["foo"]


def test_format_panel_for_operator_empty_when_skipped(dialectics):
    panel = dialectics.PanelResult(
        question="q", skipped_reason="disabled", consulted_at="x",
    )
    assert dialectics.format_panel_for_operator(panel) == ""


def test_format_panel_for_operator_renders_perspectives(dialectics):
    panel = dialectics.PanelResult(
        question="q?",
        perspectives=[dialectics.PerspectiveTension(
            tradition="Stoicism", claim="endure", counter_claim="x",
            synthesis="endure with reason", source="", confidence=1.0,
        )],
        coverage=1.0,
        consulted_at="2026-05-13T00:00:00+00:00",
    )
    rendered = dialectics.format_panel_for_operator(panel)
    assert "Stoicism" in rendered
    assert "endure" in rendered
    assert "100%" in rendered  # coverage


# ─────────────────────────────────────────────────────────────────────────
#   panel_bridge
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def panel_bridge():
    return _load_isolated(
        "panel_bridge_q51",
        "app/sentience_experiments/panel_bridge.py",
    )


def test_panel_bridge_files_tensions_via_duck_type(panel_bridge, tmp_path, monkeypatch):
    """Duck-typed panel input. file_unresolved_tensions returns IDs."""
    # Use a dict-shaped "panel" — bridge should accept it.
    panel = {
        "question": "Should we relax SAFETY_FLOOR?",
        "unresolved_tensions": [
            "Stoicism: no stance found in KB for 'Should we relax SAFETY_FLOOR?'",
            "Virtue ethics: claim present but no synthesis with counter — unresolved tension",
        ],
    }
    # Patch the tensions module path to use tmp_path.
    import app.companion.tensions as t_mod
    monkeypatch.setattr(t_mod, "_default_tensions_dir", lambda: tmp_path)
    ids = panel_bridge.file_unresolved_tensions(
        panel,
        source_kind="tier3_amendment",
        source_ref="app/foo.py",
    )
    assert len(ids) >= 1
    # Filed tensions should exist on disk.
    filed = list(tmp_path.glob("*.json"))
    assert len(filed) >= 1


def test_panel_bridge_returns_empty_when_no_unresolved(panel_bridge):
    panel = {"question": "q", "unresolved_tensions": []}
    assert panel_bridge.file_unresolved_tensions(
        panel, source_kind="x", source_ref="y",
    ) == []


def test_panel_bridge_returns_empty_when_none_panel(panel_bridge):
    assert panel_bridge.file_unresolved_tensions(
        None, source_kind="x", source_ref="y",
    ) == []


def test_panel_bridge_dedupes_against_existing_open_questions(
    panel_bridge, tmp_path, monkeypatch,
):
    """When the same question already has an open tension, don't refile."""
    import app.companion.tensions as t_mod
    monkeypatch.setattr(t_mod, "_default_tensions_dir", lambda: tmp_path)
    panel = {
        "question": "Same question?",
        "unresolved_tensions": ["Stoicism: no stance found"],
    }
    # First call files one.
    ids_a = panel_bridge.file_unresolved_tensions(
        panel, source_kind="tier3_amendment", source_ref="x",
    )
    # Second call should dedupe.
    ids_b = panel_bridge.file_unresolved_tensions(
        panel, source_kind="tier3_amendment", source_ref="x",
    )
    assert len(ids_a) >= 1
    assert len(ids_b) == 0


def test_panel_bridge_max_three_per_consult(panel_bridge, tmp_path, monkeypatch):
    """Even with N unresolved, bridge files at most _MAX_TENSIONS_PER_CONSULT."""
    import app.companion.tensions as t_mod
    monkeypatch.setattr(t_mod, "_default_tensions_dir", lambda: tmp_path)
    panel = {
        "question": "Long-tail panel question?",
        "unresolved_tensions": [
            f"Tradition_{i}: no stance found" for i in range(10)
        ],
    }
    ids = panel_bridge.file_unresolved_tensions(
        panel, source_kind="x", source_ref="y",
    )
    assert len(ids) <= 3


# ─────────────────────────────────────────────────────────────────────────
#   classify_path + relevant_history_by_kind
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def rh():
    return _load_isolated(
        "rh_q51",
        "app/identity/relevant_history.py",
    )


@pytest.mark.parametrize("path,expected_kind", [
    ("app/souls/coder.md", "soul_edit"),
    ("wiki/governance/constitution.md", "governance_constitution"),
    ("wiki/governance/policies.md", "governance_doc"),
    ("app/governance_amendment/protocol.py", "amendment_protocol"),
    ("app/governance_ratchet/protocol.py", "governance_ratchet"),
    ("app/affect/welfare.py", "welfare_envelope"),
    ("app/goodhart_guard.py", "goodhart_gate"),
    ("app/safety_guardian.py", "safety_core"),
    ("app/subia/scene/buffer.py", "kernel"),
    ("app/subia/integrity.py", "integrity_manifest"),
    ("app/agents/commander/foo.py", "agent_definition"),
    ("app/tools/web_search.py", "tool_implementation"),
    ("app/tool_registry/foo.py", "tool_registry"),
    ("app/memory/foo.py", "memory_store"),
    ("app/identity/continuity_ledger.py", "identity_layer"),
    ("app/affect/narrative.py", "affect_layer"),
    ("app/companion/tensions.py", "companion"),
    ("app/life_companion/daily_briefing.py", "life_companion"),
    ("app/healing/runbooks.py", "healing"),
    ("wiki/self/2026.md", "wiki_self"),
    ("docs/SOMETHING.md", "docs"),
    ("tests/test_x.py", "tests"),
    ("random/unmatched/path.py", "other"),
    ("", "other"),
    ("./app/souls/x.md", "soul_edit"),
    ("/app/souls/x.md", "soul_edit"),
])
def test_classify_path_covers_taxonomy(rh, path, expected_kind):
    assert rh.classify_path(path) == expected_kind


def test_relevant_history_by_kind_empty_when_disabled(rh, monkeypatch):
    """Master switch off → empty result with correct kind."""
    _ensure_runtime_settings_or_skip()
    monkeypatch.setattr(
        "app.runtime_settings.get_ledger_governor_enabled", lambda: False,
    )
    result = rh.relevant_history_by_kind("app/souls/coder.md")
    assert result["file_kind"] == "soul_edit"
    assert result["counts_by_outcome"] == {
        "applied": 0, "rolled_back": 0, "rejected": 0,
        "in_flight": 0, "amended": 0, "ratcheted": 0,
    }


def test_relevant_history_by_kind_aggregates_ledger_events(rh, tmp_path, monkeypatch):
    """Two ledger events under same kind → counts aggregate by outcome."""
    _ensure_runtime_settings_or_skip()
    monkeypatch.setattr(
        "app.runtime_settings.get_ledger_governor_enabled", lambda: True,
    )

    # Stub continuity ledger to return two soul_edit events.
    class _Event:
        def __init__(self, ts, kind, actor, summary, detail):
            self.ts, self.kind, self.actor = ts, kind, actor
            self.summary, self.detail = summary, detail

    now_iso = datetime.now(timezone.utc).isoformat()
    events = [
        _Event(now_iso, "soul_edit", "operator", "edit 1",
               {"path": "app/souls/coder.md"}),
        _Event(now_iso, "tier3_amendment", "operator", "applied amend",
               {"path": "app/souls/writer.md"}),
    ]
    import app.identity.continuity_ledger as cl
    monkeypatch.setattr(cl, "list_events", lambda **kwargs: events)
    # Empty CR audit.
    monkeypatch.setattr(rh, "_cr_audit_path", lambda: tmp_path / "nope.jsonl")

    result = rh.relevant_history_by_kind("app/souls/anything.md")
    assert result["file_kind"] == "soul_edit"
    assert result["counts_by_outcome"]["amended"] == 1
    assert result["counts_by_outcome"]["applied"] == 1
    assert result["success_rate"] == 1.0  # 1 applied / (1 applied + 0 rollbacks)


def test_relevant_history_by_kind_aggregates_cr_events(rh, tmp_path, monkeypatch):
    """CR audit entries classify by path → kind."""
    _ensure_runtime_settings_or_skip()
    monkeypatch.setattr(
        "app.runtime_settings.get_ledger_governor_enabled", lambda: True,
    )
    import app.identity.continuity_ledger as cl
    monkeypatch.setattr(cl, "list_events", lambda **kwargs: [])
    cr_log = tmp_path / "audit.jsonl"
    now_iso = datetime.now(timezone.utc).isoformat()
    cr_log.write_text("\n".join([
        json.dumps({"ts": now_iso, "payload": {
            "event": "applied", "path": "app/tools/web_search.py",
            "request_id": "cr1", "status": "applied",
        }}),
        json.dumps({"ts": now_iso, "payload": {
            "event": "rolled_back", "path": "app/tools/another.py",
            "request_id": "cr2", "status": "rolled_back",
        }}),
        json.dumps({"ts": now_iso, "payload": {
            "event": "rejected", "path": "app/souls/foo.md",
            "request_id": "cr3", "status": "rejected",
        }}),
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(rh, "_cr_audit_path", lambda: cr_log)
    # Tool-implementation kind → 1 applied + 1 rolled_back.
    result = rh.relevant_history_by_kind("app/tools/anything.py")
    assert result["file_kind"] == "tool_implementation"
    assert result["counts_by_outcome"]["applied"] == 1
    assert result["counts_by_outcome"]["rolled_back"] == 1
    assert result["counts_by_outcome"]["rejected"] == 0  # different kind
    assert result["success_rate"] == 0.5


def test_format_by_kind_for_operator_empty_on_no_activity(rh):
    """Empty history → empty string (don't clutter operator surface)."""
    empty = rh._empty_by_kind("tests", 365)
    assert rh.format_by_kind_for_operator(empty) == ""


def test_format_by_kind_for_operator_renders(rh):
    history = rh._empty_by_kind("soul_edit", 365)
    history["counts_by_outcome"]["applied"] = 3
    history["counts_by_outcome"]["rolled_back"] = 1
    history["success_rate"] = 0.75
    history["summary_line"] = "3 applied, 1 rolled back"
    rendered = rh.format_by_kind_for_operator(history)
    assert "soul_edit" in rendered
    assert "75%" in rendered
    assert "3 applied" in rendered


# ─────────────────────────────────────────────────────────────────────────
#   Wiring: Tier-3 amendment proposal includes panel + by_kind history
# ─────────────────────────────────────────────────────────────────────────


def test_tier3_amendment_tool_passes_panel_and_by_kind():
    """Source-level: the tier-3 amendment tool wires consult_panel +
    relevant_history_by_kind via extra_evidence. We verify by reading
    the source for the import + the merge into the protocol call."""
    src = Path("app/tools/request_tier3_amendment.py").read_text()
    assert "from app.philosophy.dialectics import consult_panel" in src
    assert "relevant_history_by_kind" in src
    assert "philosophy_panel" in src
    assert "panel_bridge" in src


# ─────────────────────────────────────────────────────────────────────────
#   Wiring: Identity-claim ratification consults panel
# ─────────────────────────────────────────────────────────────────────────


def test_narrative_ratification_consults_panel():
    """Source-level: _ratify_identity_claims includes the panel check."""
    src = Path("app/affect/narrative.py").read_text()
    assert "from app.philosophy.dialectics import consult_panel" in src
    assert "file_unresolved_tensions" in src
    assert "contested" in src
    # The contested set must be honored in the ratification loop.
    assert "if text in contested:" in src


# ─────────────────────────────────────────────────────────────────────────
#   Wiring: Calibration shifts append panel to report
# ─────────────────────────────────────────────────────────────────────────


def test_calibration_proposals_consults_panel_after_apply():
    """Source-level: evaluate_and_apply appends philosophy_panel to the
    report only after the apply branch — never before guardrails."""
    src = Path("app/affect/calibration_proposals.py").read_text()
    assert 'report["philosophy_panel"]' in src
    # Panel call must appear AFTER the apply branch sets status.
    apply_idx = src.find('report["status"] = "applied"')
    panel_idx = src.find('report["philosophy_panel"]')
    assert apply_idx > 0
    assert panel_idx > apply_idx, \
        "panel consult must come after the apply branch sets status"
