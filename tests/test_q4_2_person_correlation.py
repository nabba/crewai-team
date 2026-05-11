"""PROGRAM §42 — Q4.2 Person correlation stack regression tests.

Covers all four levels:
  * L1 — Presence (person_model)
  * L2 — Centrality (3 formulas)
  * L3 — Suggestions (dormancy + responsiveness)
  * L4 — Social graph + sub-features (L4.1 path / L4.2 communities /
         L4.3 bridges / L4.4 graph suggestions)

Plus: master-switch gating, mute/forget semantics, typed-phrase gates,
DR exclusion, integration with arbiter/briefing.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
#   L1 — Person model
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def person_model():
    return _load_isolated("person_model_q42", "app/companion/person_model.py")


def test_mute_unmute_roundtrip(person_model, tmp_path):
    mute_path = tmp_path / "mutes.json"
    assert person_model.mute("a@example.com", path=mute_path) is True
    assert person_model.mute("a@example.com", path=mute_path) is False  # idempotent
    assert "a@example.com" in person_model._load_mutes(mute_path)
    assert person_model.unmute("a@example.com", path=mute_path) is True
    assert person_model.unmute("a@example.com", path=mute_path) is False


def test_normalize_email_extracts_canonical(person_model):
    assert person_model._normalize_email("Maria <maria@example.com>") == "maria@example.com"
    assert person_model._normalize_email("MARIA@EXAMPLE.COM") == "maria@example.com"
    assert person_model._normalize_email("") == ""


def test_extract_display_name(person_model):
    assert person_model._extract_display_name('Maria Smith <m@x.com>') == "Maria Smith"
    assert person_model._extract_display_name('m@x.com') == ""


def test_master_switch_off_no_op(person_model, monkeypatch):
    """When master switch is OFF, compile_profile must early-out."""
    monkeypatch.setattr(person_model, "_enabled", lambda: False)
    result = person_model.compile_profile()
    assert result.get("skipped") is True


def test_current_profile_filters_muted(person_model, monkeypatch, tmp_path):
    monkeypatch.setattr(person_model, "_enabled", lambda: True)
    profile_path = tmp_path / "profile.json"
    mute_path = tmp_path / "mutes.json"
    monkeypatch.setattr(person_model, "_default_profile_path", lambda: profile_path)
    monkeypatch.setattr(person_model, "_default_mutes_path", lambda: mute_path)
    # Seed two people, mute one.
    now = datetime.now(timezone.utc).isoformat()
    profile = {
        "alice@x.com": person_model.PersonProfile(
            person_id="alice@x.com", first_seen=now, last_seen=now,
            occurrences_per_modality={"emails": 3},
        ),
        "bob@x.com": person_model.PersonProfile(
            person_id="bob@x.com", first_seen=now, last_seen=now,
            occurrences_per_modality={"emails": 5},
        ),
    }
    person_model._save_profile(profile, path=profile_path)
    person_model.mute("alice@x.com", path=mute_path)
    out = person_model.current_profile()
    person_ids = [p["person_id"] for p in out.get("people", [])]
    assert "alice@x.com" not in person_ids
    assert "bob@x.com" in person_ids


def test_forget_removes_and_unmutes(person_model, monkeypatch, tmp_path):
    monkeypatch.setattr(person_model, "_enabled", lambda: True)
    profile_path = tmp_path / "profile.json"
    mute_path = tmp_path / "mutes.json"
    monkeypatch.setattr(person_model, "_default_profile_path", lambda: profile_path)
    monkeypatch.setattr(person_model, "_default_mutes_path", lambda: mute_path)
    now = datetime.now(timezone.utc).isoformat()
    profile = {
        "x@y.com": person_model.PersonProfile(
            person_id="x@y.com", first_seen=now, last_seen=now,
        ),
    }
    person_model._save_profile(profile, path=profile_path)
    person_model.mute("x@y.com", path=mute_path)
    assert person_model.forget("x@y.com", path=profile_path) is True
    # Profile gone
    assert "x@y.com" not in person_model._load_profile(path=profile_path)
    # Mute also cleared
    assert "x@y.com" not in person_model._load_mutes(mute_path)


# ─────────────────────────────────────────────────────────────────────────
#   L2 — Centrality
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def centrality():
    return _load_isolated("centrality_q42", "app/companion/person_centrality.py")


def test_centrality_formula_validation(centrality):
    assert centrality.FORMULA_FREQUENCY in centrality.VALID_FORMULAS
    assert centrality.FORMULA_RECENCY in centrality.VALID_FORMULAS
    assert centrality.FORMULA_CROSS_MODAL in centrality.VALID_FORMULAS


def test_cross_modal_formula_saturates(centrality):
    """Cross-modal score should saturate at 4 modalities + 19 hits."""
    p = {"modality_count": 4, "total_occurrences": 19}
    s = centrality._score_cross_modal(p)
    assert s >= 0.95
    # More modalities don't push above 1.0
    p2 = {"modality_count": 10, "total_occurrences": 100}
    s2 = centrality._score_cross_modal(p2)
    assert s2 <= 1.0


def test_frequency_formula_normalizes(centrality):
    """Frequency formula returns 1.0 for the max-total person."""
    p1 = {"total_occurrences": 50}
    assert centrality._score_frequency(p1, max_total=50) == 1.0
    assert centrality._score_frequency(p1, max_total=100) == 0.5


def test_recency_decay_with_age(centrality):
    """Recent appearance scores higher than old appearance."""
    now = datetime.now(timezone.utc)
    p_recent = {
        "last_seen": (now - timedelta(days=1)).isoformat(),
        "total_occurrences": 10,
    }
    p_old = {
        "last_seen": (now - timedelta(days=90)).isoformat(),
        "total_occurrences": 10,
    }
    assert centrality._score_recency_weighted(p_recent, now) > centrality._score_recency_weighted(p_old, now)


def test_master_switch_off_returns_disabled(centrality, monkeypatch):
    monkeypatch.setattr(centrality, "_enabled", lambda: False)
    result = centrality.compute_centrality()
    assert result.get("enabled") is False


# ─────────────────────────────────────────────────────────────────────────
#   L4 — Social graph base
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def social_graph():
    return _load_isolated("social_graph_q42", "app/companion/social_graph.py")


def test_pair_key_canonical_order(social_graph):
    assert social_graph._pair_key("b@x.com", "a@x.com") == \
           social_graph._pair_key("a@x.com", "b@x.com")


def test_decay_weight_reduces_with_age(social_graph):
    fresh = social_graph._decay_weight(10.0, age_days=0)
    aged = social_graph._decay_weight(10.0, age_days=90)
    # 90d = 1 half-life
    assert abs(aged - 5.0) < 0.01
    assert fresh == 10.0


def test_path_opt_out_persists(social_graph, monkeypatch, tmp_path):
    monkeypatch.setattr(social_graph, "_default_path_opt_outs_path",
                        lambda: tmp_path / "optouts.json")
    monkeypatch.setattr(social_graph, "_enabled", lambda: True)
    assert social_graph.opt_out_of_paths("x@y.com") is True
    assert social_graph.opt_out_of_paths("x@y.com") is False  # idempotent
    assert "x@y.com" in social_graph.load_path_opt_outs()


# ─────────────────────────────────────────────────────────────────────────
#   L4.1 — Shortest path
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def shortest_path():
    return _load_isolated("shortest_path_q42", "app/companion/graph_features/shortest_path.py")


def test_path_disabled_returns_error(shortest_path, monkeypatch):
    monkeypatch.setattr(shortest_path, "_enabled", lambda: False)
    r = shortest_path.find_path("a", "b")
    assert r["ok"] is False
    assert "disabled" in r["error"]


def test_path_finds_two_hops(shortest_path, monkeypatch):
    """Graph: a — b — c. find_path(a, c) should return [a, b, c]."""
    monkeypatch.setattr(shortest_path, "_enabled", lambda: True)
    # Stub the adjacency
    import app.companion.social_graph as sg
    monkeypatch.setattr(sg, "adjacency", lambda: {
        "a": {"b": 1.0},
        "b": {"a": 1.0, "c": 1.0},
        "c": {"b": 1.0},
    })
    monkeypatch.setattr(sg, "load_path_opt_outs", lambda: set())
    monkeypatch.setattr(sg, "log_query", lambda *args, **kwargs: None)
    r = shortest_path.find_path("a", "c")
    assert r["ok"] is True
    assert r["path"] == ["a", "b", "c"]
    assert r["hops"] == 2


def test_path_excludes_opt_outs_as_intermediate(shortest_path, monkeypatch):
    """If B opts out, find_path(A, C) must NOT go through B."""
    monkeypatch.setattr(shortest_path, "_enabled", lambda: True)
    import app.companion.social_graph as sg
    monkeypatch.setattr(sg, "adjacency", lambda: {
        "a": {"b": 1.0, "d": 1.0},
        "b": {"a": 1.0, "c": 1.0},
        "d": {"a": 1.0, "c": 1.0},
        "c": {"b": 1.0, "d": 1.0},
    })
    monkeypatch.setattr(sg, "load_path_opt_outs", lambda: {"b"})
    monkeypatch.setattr(sg, "log_query", lambda *args, **kwargs: None)
    r = shortest_path.find_path("a", "c")
    assert r["ok"] is True
    assert "b" not in r["path"]  # B excluded; should go via D
    assert "d" in r["path"]


# ─────────────────────────────────────────────────────────────────────────
#   L4.2 — Communities
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def communities():
    return _load_isolated("communities_q42", "app/companion/graph_features/communities.py")


def test_label_propagation_finds_two_clusters(communities):
    """Two dense groups connected by a single weak edge → 2 clusters."""
    adj = {
        # Cluster 1: a-b-c densely connected
        "a": {"b": 3.0, "c": 3.0},
        "b": {"a": 3.0, "c": 3.0},
        "c": {"a": 3.0, "b": 3.0, "d": 0.5},
        # Cluster 2: d-e-f densely connected
        "d": {"e": 3.0, "f": 3.0, "c": 0.5},
        "e": {"d": 3.0, "f": 3.0},
        "f": {"d": 3.0, "e": 3.0},
    }
    labels = communities._label_propagation(adj, seed=42)
    # a/b/c should share a label; d/e/f should share a different label.
    assert labels["a"] == labels["b"] == labels["c"]
    assert labels["d"] == labels["e"] == labels["f"]
    assert labels["a"] != labels["d"]


def test_modularity_positive_for_clustered(communities):
    """Well-clustered graph has positive modularity."""
    adj = {
        "a": {"b": 3.0, "c": 3.0},
        "b": {"a": 3.0, "c": 3.0},
        "c": {"a": 3.0, "b": 3.0},
        "d": {"e": 3.0},
        "e": {"d": 3.0},
    }
    labels = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1}
    q = communities._compute_modularity(adj, labels)
    assert q > 0.2


# ─────────────────────────────────────────────────────────────────────────
#   L4.3 — Bridges
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def bridges():
    return _load_isolated("bridges_q42", "app/companion/graph_features/bridges.py")


def test_tarjan_finds_bridge_in_chain(bridges):
    """Graph: a-b-c. The edge a-b is a bridge; so is b-c. b is a cut-vertex."""
    adj = {
        "a": {"b": 1.0},
        "b": {"a": 1.0, "c": 1.0},
        "c": {"b": 1.0},
    }
    bridge_list, articulations = bridges._find_bridges_and_articulations(adj)
    assert len(bridge_list) == 2
    assert "b" in articulations


def test_tarjan_no_bridge_in_triangle(bridges):
    """Triangle has no bridges, no cut-vertices."""
    adj = {
        "a": {"b": 1.0, "c": 1.0},
        "b": {"a": 1.0, "c": 1.0},
        "c": {"a": 1.0, "b": 1.0},
    }
    bridge_list, articulations = bridges._find_bridges_and_articulations(adj)
    assert bridge_list == []
    assert articulations == set()


# ─────────────────────────────────────────────────────────────────────────
#   Runtime settings — gating
# ─────────────────────────────────────────────────────────────────────────


def test_all_runtime_flags_default_off():
    """Source-level: all 11+ new runtime_settings flags default to False
    (except formula which is a string + decay_months which is an int)."""
    src = Path("app/runtime_settings.py").read_text()
    flags_default_off = [
        "person_correlation_enabled",
        "person_centrality_enabled",
        "person_suggestions_enabled",
        "person_suggestions_dormancy_enabled",
        "person_suggestions_responsiveness_enabled",
        "person_correlation_social_graph_enabled",
        "graph_shortest_path_enabled",
        "graph_communities_enabled",
        "graph_bridges_enabled",
        "graph_suggestions_enabled",
        "graph_suggestions_cluster_dormancy_enabled",
        "graph_suggestions_bridge_maintenance_enabled",
        "graph_suggestions_weak_tie_enabled",
    ]
    for f in flags_default_off:
        assert f'"{f}": False' in src, f"flag {f} missing default-False"


# ─────────────────────────────────────────────────────────────────────────
#   Config-api typed-phrase gates
# ─────────────────────────────────────────────────────────────────────────


def test_config_api_enforces_social_graph_typed_phrase():
    """Source-level: enabling L4 requires the typed phrase."""
    src = Path("app/api/config_api.py").read_text()
    assert "ENABLE SOCIAL GRAPH" in src
    assert "social_graph_confirm_phrase" in src


def test_config_api_enforces_graph_suggestions_typed_phrase():
    """Source-level: enabling L4.4 requires the SECOND typed phrase."""
    src = Path("app/api/config_api.py").read_text()
    assert "ENABLE GRAPH-DRIVEN SUGGESTIONS" in src
    assert "graph_suggestions_confirm_phrase" in src


# ─────────────────────────────────────────────────────────────────────────
#   DR exclusion
# ─────────────────────────────────────────────────────────────────────────


def test_dr_denylist_excludes_social_graph():
    """social_graph substring catches the graph file + all derived
    analysis files (communities, structural, query log)."""
    mod = _load_isolated("dr_export_q42", "app/dr/export_kbs.py")
    assert mod._is_secret_path("workspace/companion/social_graph.json")
    assert mod._is_secret_path("workspace/companion/social_graph_communities.json")
    assert mod._is_secret_path("workspace/companion/social_graph_structural.json")
    assert mod._is_secret_path("workspace/companion/social_graph_query_log.jsonl")
    # Innocent paths still pass.
    assert not mod._is_secret_path("workspace/companion/person_profile.json")


# ─────────────────────────────────────────────────────────────────────────
#   Arbiter integration
# ─────────────────────────────────────────────────────────────────────────


def test_arbiter_centrality_helper_caps_contribution():
    """Source-level: centrality boost is capped at 0.15."""
    src = Path("app/notify/arbiter.py").read_text()
    assert "_person_centrality_boost" in src
    # The cap value 0.15 appears in the function body.
    fn_start = src.find("def _person_centrality_boost")
    fn_end = src.find("\ndef ", fn_start + 1)
    body = src[fn_start:fn_end]
    assert "0.15" in body


def test_arbiter_bridge_helper_caps_contribution():
    src = Path("app/notify/arbiter.py").read_text()
    assert "_bridge_boost" in src
    fn_start = src.find("def _bridge_boost")
    fn_end = src.find("\ndef ", fn_start + 1)
    body = src[fn_start:fn_end]
    assert "0.10" in body


# ─────────────────────────────────────────────────────────────────────────
#   Idle-job registration
# ─────────────────────────────────────────────────────────────────────────


def test_companion_loop_registers_person_jobs():
    src = Path("app/companion/loop.py").read_text()
    assert "person-model" in src
    assert "social-graph" in src
    assert "graph-features" in src


# ─────────────────────────────────────────────────────────────────────────
#   Briefing integration
# ─────────────────────────────────────────────────────────────────────────


def test_briefing_includes_people_and_suggestions_sections():
    src = Path("app/life_companion/daily_briefing.py").read_text()
    assert "🧑 People showing up" in src
    assert "💬 Suggestions" in src
    assert "_gather_people_insights" in src
    assert "_gather_person_suggestions" in src
