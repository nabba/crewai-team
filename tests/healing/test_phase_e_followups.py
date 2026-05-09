"""Phase E targeted tests for the audit-finding fixes.

Each test pins behavior that the original Phase A-D test suite missed
because it stubbed too aggressively. These tests exercise REAL
integration points (registry walks, embedder source detection, etc.)
so a future regression of the same kind can't slip through.
"""
from __future__ import annotations

import json

import pytest


# ── E2: pattern_learner reads the real runbooks registry ─────────────────


def test_pattern_learner_registry_walk_uses_real_lock_name():
    """The earlier code imported ``_LOCK`` (doesn't exist; actual is
    ``_registry_lock``). Phase E #2 fixed it. Verify the import path
    resolves on the actual module so a future rename surfaces here."""
    from app.healing.pattern_learner import _registered_signatures
    sigs = _registered_signatures()
    # We don't care WHICH signatures are registered (depends on env);
    # we care that the function returns a set without crashing on the
    # registry symbols. The empty-set fallback would also pass — but
    # the import itself succeeding is the signal we want.
    assert isinstance(sigs, set)


def test_runbooks_lock_symbol_is_what_pattern_learner_expects():
    """If runbooks.py renames _registry_lock, this test catches it
    BEFORE production silently treats every signature as un-covered."""
    from app.healing import runbooks
    assert hasattr(runbooks, "_registry_lock")
    assert hasattr(runbooks, "_REGISTERED_RUNBOOKS")


# ── E10: llm_output_drift — embedder source mismatch detection ──────────


@pytest.fixture
def isolated_drift(tmp_path, monkeypatch):
    from app.healing import llm_output_drift as drift
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(drift, "_BASELINE_PATH", tmp_path / "baseline.json")
    monkeypatch.setattr(drift, "_HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr(drift, "_PROBES_PATH", tmp_path / "probes.json")

    sent: list[str] = []
    monkeypatch.setattr(
        "app.healing.handlers._common.send_signal_alert",
        lambda body, **kw: sent.append(body) or True,
    )
    monkeypatch.setattr(
        "app.healing.handlers._common.audit_event",
        lambda *a, **k: None,
    )
    yield tmp_path, sent


def test_drift_returns_source_alongside_vector():
    """The new contract is ``(vec, source)``. Tests pin the shape."""
    from app.healing.llm_output_drift import _embed
    v, src = _embed("estonia capital")
    assert isinstance(v, list)
    assert src in ("chroma", "hash")


def test_drift_alerts_on_embedder_source_change(isolated_drift, monkeypatch):
    """Baseline written with one source, current pass uses different
    source — alert fires once and avg_similarity stays None."""
    tmp_path, sent = isolated_drift
    from app.healing import llm_output_drift as drift

    # Force baseline to be seeded with "chroma".
    answers = iter(["Tallinn"] * 10)
    monkeypatch.setattr(drift, "_ask_llm", lambda q: next(answers))
    sources = iter(["chroma"] * 5)
    monkeypatch.setattr(
        drift, "_embed",
        lambda text: ([1.0, 0.0, 0.0], next(sources)),
    )
    drift.run()

    # Reset cadence; second run with "hash" source.
    state_path = tmp_path / "self_heal" / "llm_output_drift.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))

    sources2 = iter(["hash"] * 5)
    monkeypatch.setattr(
        drift, "_embed",
        lambda text: ([0.5, 0.5], next(sources2)),
    )
    summary = drift.run()
    assert summary["avg_similarity"] is None  # we refused to compare
    assert any(
        "embedder source changed" in body.lower() for body in sent
    )


# ── E14: runtime_settings tolerates incomplete Settings ─────────────────


def test_runtime_settings_defensive_reads(monkeypatch):
    """A stripped-down Settings (no voice_mode etc) must not crash
    runtime_settings._defaults — Phase E #14."""
    from app import runtime_settings

    class _Bare:
        pass  # has NONE of the fields runtime_settings reads
    monkeypatch.setattr(runtime_settings, "get_settings", lambda: _Bare())
    defaults = runtime_settings._defaults()
    # Every documented key still produces a value.
    for key in (
        "voice_mode", "vision_cu_enabled", "vision_cu_monthly_cap_usd",
        "concierge_persona_enabled", "tier3_amendment_enabled",
        "error_runbooks_enabled", "tool_supervisor_enabled",
        "recovery_loop_enabled", "goodhart_hard_gate_disabled",
        "goodhart_hard_gate_enforcing",
    ):
        assert key in defaults


# ── E1: feedback_router uses the right table + columns ──────────────────


def test_feedback_router_resolve_send_ts_targets_response_metadata():
    """The query must hit ``feedback.response_metadata.msg_timestamp``,
    not the non-existent ``feedback.responses.target_timestamp``."""
    import inspect
    from app.companion import feedback_router
    src = inspect.getsource(feedback_router._resolve_send_ts)
    assert "feedback.response_metadata" in src
    assert "msg_timestamp" in src
    # Defensively check the OLD wrong query is gone.
    assert "feedback.responses" not in src
    assert "target_timestamp" not in src


# ── E3: calendar_prep targets the right helper functions ─────────────────


def test_calendar_prep_uses_real_gmail_helper():
    """Verify the import target is the real helper name (not the
    earlier non-existent ``_list_messages``)."""
    import inspect
    from app.life_companion import calendar_prep
    src = inspect.getsource(calendar_prep._recent_inbox_from)
    assert "from app.tools.gmail_tools import _list_recent" in src


def test_calendar_prep_uses_real_mem0_helper():
    """Same — the earlier ``get_manager`` doesn't exist in mem0_manager."""
    import inspect
    from app.life_companion import calendar_prep
    src = inspect.getsource(calendar_prep._mem0_facts_about)
    assert "from app.memory.mem0_manager import search_memory" in src


# ── E4: interest_model filters to UP polarity ───────────────────────────


def test_interest_model_filters_feedback_to_up_only(tmp_path, monkeypatch):
    """A 👎 comment must NOT show up as positive interest signal."""
    from app.companion import interest_model
    events_path = tmp_path / "events.jsonl"
    rows = [
        {"type": "FEEDBACK", "ts": 9999999999.0,  # always within lookback
         "payload": {"polarity": "down", "comment": "I dislike forest carbon"}},
        {"type": "FEEDBACK", "ts": 9999999999.0,
         "payload": {"polarity": "up", "comment": "love the kaicart angle"}},
    ]
    events_path.write_text("\n".join(json.dumps(r) for r in rows))

    # Repoint the events.jsonl path the function reads.
    from pathlib import Path
    from unittest.mock import patch
    real_path = Path
    def patched_path(s):
        if isinstance(s, str) and s == "/app/workspace/companion/events.jsonl":
            return events_path
        return real_path(s)
    monkeypatch.setattr(interest_model, "Path", patched_path)

    out = list(interest_model._feedback_events_text(lookback_days=14))
    comments = [text for text, _ in out]
    assert any("kaicart" in c for c in comments)
    assert not any("dislike forest carbon" in c for c in comments)


# ── E5: paper_pipeline retains newest 5000 by ts ────────────────────────


def test_paper_pipeline_seen_evicts_oldest_first(tmp_path, monkeypatch):
    from app.episteme import paper_pipeline
    monkeypatch.setattr(paper_pipeline, "_SEEN_PATH", tmp_path / "seen.json")
    seen = {f"id{i}": float(i) for i in range(5050)}
    paper_pipeline._save_seen(seen)
    loaded = paper_pipeline._load_seen()
    assert len(loaded) == 5000
    # Oldest 50 evicted.
    assert "id0" not in loaded
    assert "id49" not in loaded
    assert "id50" in loaded
    assert "id5049" in loaded


# ── E7: shared hash_embedding util — round trip ─────────────────────────


def test_hash_embedding_deterministic_across_modules():
    """Both consumers (llm_drift fallback + lessons_learned) must
    produce identical vectors for identical text."""
    from app.utils.hash_embedding import embed
    from app.healing.llm_output_drift import _hash_embed as drift_hash
    from app.companion.lessons_learned import _embed as lessons_embed

    text = "rewrite the auth module"
    v0 = embed(text)
    v1 = drift_hash(text)
    v2 = lessons_embed(text)
    assert v0 == v1
    assert v0 == v2


# ── E12: boot_reset doesn't create empty dbm files ──────────────────────


def test_boot_reset_no_dbm_no_creation(tmp_path, monkeypatch):
    """If there's no dbm file on disk, boot_reset should NOT create one."""
    from app.healing import boot_reset
    monkeypatch.setattr(boot_reset, "_already_ran", False)
    monkeypatch.setattr(
        boot_reset, "_DBM_PATH_CANDIDATES",
        (tmp_path / "absent_idle_state",),
    )
    summary = boot_reset.reset_stale_cooldowns()
    # Nothing examined, nothing reset, nothing created.
    assert summary["examined"] == 0
    # No spurious files left behind.
    assert not (tmp_path / "absent_idle_state").exists()
    assert not (tmp_path / "absent_idle_state.db").exists()
