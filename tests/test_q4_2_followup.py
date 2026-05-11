"""PROGRAM §42.1 + §42.2 — Q4.2 follow-up tests.

Covers the 6 bug fixes (Q4.2.1) + 6 missing wires (Q4.2.2) flagged by
the deep audit of the initial Q4.2 ship.
"""
from __future__ import annotations

import importlib.util
import json
import sys
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
#   Q4.2.1#1 — L4.4 mute-suggestions filter
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def suggestions():
    return _load_isolated(
        "person_suggestions_q421",
        "app/companion/person_suggestions.py",
    )


def test_mute_suggestions_filters_l44_graph_nudges(suggestions, monkeypatch, tmp_path):
    """Q4.2.1#1 — a person muted via mute_suggestions_for must NOT receive
    bridge_maintenance or weak_tie_dormant nudges from L4.4."""
    # Force master switches ON.
    monkeypatch.setattr(suggestions, "_enabled", lambda: True)
    monkeypatch.setattr(suggestions, "_default_sug_mutes",
                        lambda: tmp_path / "mutes.json")
    monkeypatch.setattr(suggestions, "_default_emitted_log",
                        lambda: tmp_path / "emitted.jsonl")

    suggestions.mute_suggestions_for("maria@x.com")

    # Stub current_profile to provide Maria as a candidate.
    now = datetime.now(timezone.utc)
    long_ago = (now - timedelta(days=120)).isoformat()
    monkeypatch.setattr(
        "app.companion.person_model.current_profile",
        lambda: {
            "enabled": True,
            "people": [{
                "person_id": "maria@x.com",
                "display_names": ["Maria"],
                "last_seen": long_ago,
                "modality_count": 4,
                "total_occurrences": 30,
                "occurrences_per_modality": {"emails": 10, "calendar": 10, "convs": 10},
            }],
        },
    )
    monkeypatch.setattr(suggestions, "_dormancy_enabled", lambda: True)
    # Stub the L4.4 graph_suggestions to return a forced bridge nudge.
    fake_sug = suggestions.PersonSuggestion(
        category="bridge_maintenance",
        person_id="maria@x.com",
        display_name="Maria",
        text="Maria is a bridge — reconnect?",
        detected_at=now.isoformat(),
    )
    import app.companion.graph_features.graph_suggestions as gs_mod
    monkeypatch.setattr(gs_mod, "generate_graph_suggestions", lambda people: [fake_sug])
    # Disable welfare so we don't get suppression by mistake.
    monkeypatch.setattr("app.notify.arbiter.welfare_breaching", lambda: False)

    out = suggestions.generate_suggestions()
    assert all(s["person_id"] != "maria@x.com" for s in out), \
        "mute-suggestions should also filter L4.4 graph nudges"


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.1#2 — recent_emitted gated on master switch
# ─────────────────────────────────────────────────────────────────────────


def test_recent_emitted_empty_when_master_off(suggestions, monkeypatch, tmp_path):
    """Q4.2.1#2 — disabling L3 must hide historical emitted nudges."""
    monkeypatch.setattr(suggestions, "_default_emitted_log",
                        lambda: tmp_path / "emitted.jsonl")
    # Seed the log with a historical entry.
    log = tmp_path / "emitted.jsonl"
    log.write_text(json.dumps({
        "category": "dormancy_nudge",
        "person_id": "x@y.com",
        "display_name": "X",
        "text": "old nudge",
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }) + "\n", encoding="utf-8")
    # Master OFF.
    monkeypatch.setattr(suggestions, "_enabled", lambda: False)
    assert suggestions.recent_emitted() == []
    # Master ON returns the entry.
    monkeypatch.setattr(suggestions, "_enabled", lambda: True)
    rows = suggestions.recent_emitted()
    assert len(rows) == 1
    assert rows[0]["person_id"] == "x@y.com"


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.1#3 — stale structural freshness check
# ─────────────────────────────────────────────────────────────────────────


def test_graph_suggestions_skips_stale_topology(monkeypatch, tmp_path):
    """Q4.2.1#3 — bridge_maintenance / weak_tie_dormant must skip
    when social_graph_structural.json is too old."""
    gs = _load_isolated(
        "gs_q421_stale",
        "app/companion/graph_features/graph_suggestions.py",
    )
    # All gates ON.
    monkeypatch.setattr(gs, "_enabled", lambda: True)
    monkeypatch.setattr(gs, "_bridge_maintenance_enabled", lambda: True)
    monkeypatch.setattr(gs, "_weak_tie_dormant_enabled", lambda: True)
    # Stale structural file (200h old).
    stale_path = tmp_path / "structural.json"
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
    stale_path.write_text(json.dumps({
        "generated_at": old_ts,
        "bridges": [], "cut_vertices": ["alice@x.com"],
    }), encoding="utf-8")
    import app.companion.graph_features.bridges as br
    monkeypatch.setattr(br, "_default_structural_path", lambda: stale_path)
    now = datetime.now(timezone.utc)
    people = [{
        "person_id": "alice@x.com",
        "display_names": ["Alice"],
        "last_seen": (now - timedelta(days=60)).isoformat(),
    }]
    assert gs._generate_bridge_maintenance(people) == []
    assert gs._generate_weak_tie_dormant(people) == []
    # Fresh structural file → emits.
    fresh_ts = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    stale_path.write_text(json.dumps({
        "generated_at": fresh_ts,
        "bridges": [], "cut_vertices": ["alice@x.com"],
    }), encoding="utf-8")
    out = gs._generate_bridge_maintenance(people)
    assert len(out) == 1


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.1#4 — file naming consistency
# ─────────────────────────────────────────────────────────────────────────


def test_dissolved_clusters_filename_matches_docs():
    """Q4.2.1#4 — code path matches the name documented in
    docs/PERSON_CORRELATION.md + PROGRAM §42."""
    mod = _load_isolated(
        "communities_q421",
        "app/companion/graph_features/communities.py",
    )
    p = mod._default_dissolved_path()
    assert p.name == "social_graph_dissolved_clusters.json"


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.1#5 — conversation_store time-bound
# ─────────────────────────────────────────────────────────────────────────


def test_gather_conversation_participants_is_time_bounded():
    """Q4.2.1#5 — _gather_conversation_participants must include a
    ts >= cutoff predicate, falling back to unbounded only on schema
    failure."""
    src = Path("app/companion/person_model.py").read_text()
    assert "AND ts >= ?" in src, "expected time-bounded sender_id query"
    # Fall-through is intentional, document it stays in place.
    assert "fall back to" in src.lower()


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.1#6 — per-briefing dedup against prior 24h
# ─────────────────────────────────────────────────────────────────────────


def test_reemit_cooldown_dedupes_within_24h(suggestions, monkeypatch, tmp_path):
    """Q4.2.1#6 — the same (category, person_id) emitted <24h ago must
    not re-fire."""
    monkeypatch.setattr(suggestions, "_default_emitted_log",
                        lambda: tmp_path / "emitted.jsonl")
    # Seed an emission 2h ago for Maria.
    recent_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    log = tmp_path / "emitted.jsonl"
    log.write_text(json.dumps({
        "category": "dormancy_nudge",
        "person_id": "maria@x.com",
        "display_name": "Maria",
        "text": "old",
        "detected_at": recent_ts,
    }) + "\n", encoding="utf-8")
    keys = suggestions._recent_emission_keys(24)
    assert ("dormancy_nudge", "maria@x.com") in keys
    # Aged out at >24h.
    log.write_text(json.dumps({
        "category": "dormancy_nudge",
        "person_id": "maria@x.com",
        "display_name": "Maria",
        "text": "old",
        "detected_at": (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(),
    }) + "\n", encoding="utf-8")
    keys2 = suggestions._recent_emission_keys(24)
    assert ("dormancy_nudge", "maria@x.com") not in keys2


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.2#1 — identity continuity ledger emission
# ─────────────────────────────────────────────────────────────────────────


def test_continuity_ledger_accepts_person_correlation_kind():
    """Q4.2.2#1 — new event kind person_correlation_policy is in the
    accepted set."""
    mod = _load_isolated(
        "continuity_q422",
        "app/identity/continuity_ledger.py",
    )
    assert "person_correlation_policy" in mod.IDENTITY_EVENT_KINDS


def test_continuity_ledger_round_trip(tmp_path):
    """Q4.2.2#1 — record_event for person_correlation_policy round-trips
    through list_events."""
    mod = _load_isolated(
        "continuity_q422_rt",
        "app/identity/continuity_ledger.py",
    )
    target = tmp_path / "ledger.jsonl"
    assert mod.record_event(
        kind="person_correlation_policy",
        actor="operator",
        summary="L4 enabled",
        detail={"level": "L4", "enabled": True},
        path=target,
    ) is True
    rows = mod.list_events(path=target, kinds={"person_correlation_policy"})
    assert len(rows) == 1
    assert rows[0].detail["level"] == "L4"


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.2#2 — welfare_breaching public alias
# ─────────────────────────────────────────────────────────────────────────


def test_arbiter_exports_public_welfare_breaching():
    """Q4.2.2#2 — public alias must exist."""
    from app.notify import arbiter
    assert hasattr(arbiter, "welfare_breaching")
    assert callable(arbiter.welfare_breaching)


def test_l3_suggestions_suppressed_under_welfare_breach(suggestions, monkeypatch, tmp_path):
    """Q4.2.2#2 — generate_suggestions must short-circuit when
    welfare_breaching() returns True."""
    monkeypatch.setattr(suggestions, "_enabled", lambda: True)
    monkeypatch.setattr(suggestions, "_default_emitted_log",
                        lambda: tmp_path / "emitted.jsonl")
    monkeypatch.setattr(suggestions, "_default_sug_mutes",
                        lambda: tmp_path / "mutes.json")
    # Force welfare breach.
    monkeypatch.setattr("app.notify.arbiter.welfare_breaching", lambda: True)
    # Stub profile with a dormancy candidate.
    monkeypatch.setattr(
        "app.companion.person_model.current_profile",
        lambda: {
            "enabled": True,
            "people": [{
                "person_id": "alice@x.com",
                "display_names": ["Alice"],
                "last_seen": (datetime.now(timezone.utc) - timedelta(days=120)).isoformat(),
                "modality_count": 4,
                "total_occurrences": 20,
                "occurrences_per_modality": {"emails": 20},
            }],
        },
    )
    monkeypatch.setattr(suggestions, "_dormancy_enabled", lambda: True)
    out = suggestions.generate_suggestions()
    assert out == [], "suggestions must be suppressed under welfare breach"


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.2#3 — cross-modal patterns over people
# ─────────────────────────────────────────────────────────────────────────


def test_detect_person_patterns_emits_when_thresholds_met(monkeypatch):
    """Q4.2.2#3 — a person crossing 3+ modalities at high volume should
    produce a Pattern(kind='person')."""
    cmp = _load_isolated(
        "cmp_q422",
        "app/companion/cross_modal_patterns.py",
    )
    # Stub current_profile to provide a strong-convergence person.
    monkeypatch.setattr(
        "app.companion.person_model.current_profile",
        lambda: {
            "enabled": True,
            "people": [{
                "person_id": "maria@x.com",
                "display_names": ["Maria"],
                "modality_count": 4,
                "total_occurrences": 20,
                "occurrences_per_modality": {
                    "emails": 5, "calendar": 5, "convs": 5, "tickets": 5,
                },
            }],
        },
    )
    # Stub the tension boost so we don't hit the real store.
    monkeypatch.setattr(
        cmp, "_boost_matching_tensions_for_person",
        lambda pid, names: 0,
    )
    patterns = cmp.detect_person_patterns()
    assert len(patterns) == 1
    assert patterns[0].kind == "person"
    assert patterns[0].topic == "Maria"
    assert patterns[0].strength >= 0.7


def test_detect_person_patterns_noop_when_disabled(monkeypatch):
    """Q4.2.2#3 — when person-correlation master is OFF, current_profile
    returns enabled=False; detect_person_patterns must return []."""
    cmp = _load_isolated(
        "cmp_q422_off",
        "app/companion/cross_modal_patterns.py",
    )
    monkeypatch.setattr(
        "app.companion.person_model.current_profile",
        lambda: {"enabled": False, "people": []},
    )
    assert cmp.detect_person_patterns() == []


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.2#4 — tension ↔ person cross-link
# ─────────────────────────────────────────────────────────────────────────


def test_boost_freshness_for_person_matches_display_name(tmp_path):
    """Q4.2.2#4 — a tension whose question mentions a display name should
    get its freshness bumped when that person re-appears."""
    tensions = _load_isolated(
        "tensions_q422",
        "app/companion/tensions.py",
    )
    # Create an open tension mentioning Maria.
    t = tensions.create_tension(
        question="Should I follow up with Maria about the X collab?",
        sources=[],
        base=tmp_path,
    )
    assert t is not None
    before = t.last_touched_at
    # Boost.
    count = tensions.boost_freshness_for_person(
        "maria@example.com", ["Maria"], base=tmp_path,
    )
    assert count == 1
    # The tension's last_touched_at should have bumped.
    refreshed = tensions.list_tensions(
        status=tensions.STATUS_OPEN, base=tmp_path,
    )
    assert any(r.id == t.id and r.last_touched_at >= before for r in refreshed)


def test_boost_freshness_for_person_skips_short_names(tmp_path):
    """Q4.2.2#4 — names <3 chars must not produce matches (would spuriously
    hit short tokens)."""
    tensions = _load_isolated(
        "tensions_q422_short",
        "app/companion/tensions.py",
    )
    tensions.create_tension(
        question="Should I follow up with Maria?",
        sources=[],
        base=tmp_path,
    )
    # Empty display names + email local-part "ma" is 2 chars → no match.
    count = tensions.boost_freshness_for_person(
        "ma@example.com", [], base=tmp_path,
    )
    assert count == 0


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.2#5 — GW publish opaque counts
# ─────────────────────────────────────────────────────────────────────────


def test_person_model_publishes_opaque_counts_to_gw(monkeypatch):
    """Q4.2.2#5 — compile_profile must publish summary to GW with NO
    person_ids in the content."""
    pm = _load_isolated(
        "pm_q422_gw",
        "app/companion/person_model.py",
    )
    monkeypatch.setattr(pm, "_enabled", lambda: True)
    # Force a single sighting through stubbed collectors.
    monkeypatch.setattr(pm, "_gather_email_senders",
                        lambda d: [("alice@x.com", "Alice", 1.0)])
    monkeypatch.setattr(pm, "_gather_calendar_attendees", lambda d: [])
    monkeypatch.setattr(pm, "_gather_conversation_participants", lambda d: [])
    monkeypatch.setattr(pm, "_default_profile_path",
                        lambda: Path("/tmp/q422_gw_profile.json"))
    monkeypatch.setattr(pm, "_default_history_path",
                        lambda: Path("/tmp/q422_gw_history.jsonl"))
    monkeypatch.setattr(pm, "_default_mutes_path",
                        lambda: Path("/tmp/q422_gw_mutes.json"))
    # Clean prior state.
    for p in [Path("/tmp/q422_gw_profile.json"),
              Path("/tmp/q422_gw_history.jsonl"),
              Path("/tmp/q422_gw_mutes.json")]:
        if p.exists():
            p.unlink()
    captured: list[dict] = []
    def fake_publish(**kwargs):
        captured.append(kwargs)
    import app.workspace_publish as wp
    monkeypatch.setattr(wp, "publish_to_workspace", fake_publish)
    out = pm.compile_profile(lookback_days=7)
    assert out["ok"] is True
    assert out["new_sightings"] == 1
    # GW was called.
    assert len(captured) == 1
    content = captured[0]["content"]
    # Must NOT contain person_ids or names.
    assert "alice" not in content.lower()
    assert "@" not in content
    # Must contain the opaque count.
    assert "1 new" in content


# ─────────────────────────────────────────────────────────────────────────
#   Q4.2.2#6 — /cp/monitor person-correlation probe
# ─────────────────────────────────────────────────────────────────────────


def test_dashboard_api_has_person_correlation_probe():
    """Q4.2.2#6 — system_status must register a person-correlation row."""
    src = Path("app/control_plane/dashboard_api.py").read_text()
    assert "_person_correlation" in src
    assert '"Person correlation", "Internal"' in src
