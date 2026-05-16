"""Q17 — Multi-year resilience (PROGRAM §52).

Eight observational subsystems in one batch. All additive,
observational, default-conservative.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolated_module(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_notify(monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    fake = type(sys)("app.notify")
    fake.notify = lambda **kw: captured.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake)
    return captured


def _stub_ledger(monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    fake = type(sys)("app.identity.continuity_ledger")
    fake.record_event = lambda **kw: (captured.append(kw) or True)
    monkeypatch.setitem(sys.modules, "app.identity.continuity_ledger", fake)
    return captured


def _stub_runtime_settings(monkeypatch, **flags: bool) -> None:
    fake = type(sys)("app.runtime_settings")
    for name, value in flags.items():
        setattr(fake, name, lambda v=value: v)
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake)


# ── Q17.1 warm_spare ────────────────────────────────────────────────────


def _load_manifest(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/warm_spare/manifest.py", "_q17_warm_spare_manifest")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def test_warm_spare_manifest_walks_critical_paths(monkeypatch, tmp_path) -> None:
    mod = _load_manifest(monkeypatch, tmp_path)
    (tmp_path / "identity").mkdir()
    (tmp_path / "identity" / "continuity_ledger.jsonl").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "page.md").write_text("hello", encoding="utf-8")
    m = mod.build_manifest()
    assert m["n_files"] >= 2
    paths = {e["path"] for e in m["entries"]}
    assert "identity/continuity_ledger.jsonl" in paths
    assert "wiki/page.md" in paths


def test_warm_spare_manifest_excludes_pycache(monkeypatch, tmp_path) -> None:
    mod = _load_manifest(monkeypatch, tmp_path)
    (tmp_path / "wiki" / "__pycache__").mkdir(parents=True)
    (tmp_path / "wiki" / "__pycache__" / "x.pyc").write_text("x", encoding="utf-8")
    (tmp_path / "wiki" / "page.md").write_text("hello", encoding="utf-8")
    m = mod.build_manifest()
    paths = {e["path"] for e in m["entries"]}
    assert not any(".pyc" in p for p in paths)
    assert not any("__pycache__" in p for p in paths)


def test_warm_spare_manifest_save_load_roundtrip(monkeypatch, tmp_path) -> None:
    mod = _load_manifest(monkeypatch, tmp_path)
    (tmp_path / "identity").mkdir()
    (tmp_path / "identity" / "continuity_ledger.jsonl").write_text("x\n", encoding="utf-8")
    m = mod.build_manifest()
    mod.save_manifest(m)
    m2 = mod.load_manifest()
    assert m2["n_files"] == m["n_files"]


def _load_failover(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/warm_spare/failover.py", "_q17_warm_spare_failover")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def test_warm_spare_claim_rejects_wrong_phrase(monkeypatch, tmp_path) -> None:
    mod = _load_failover(monkeypatch, tmp_path)
    out = mod.claim_canonical("wrong words")
    assert out["accepted"] is False


def test_warm_spare_claim_rejects_recent_heartbeat(monkeypatch, tmp_path) -> None:
    mod = _load_failover(monkeypatch, tmp_path)
    mod.record_heartbeat()
    out = mod.claim_canonical("CLAIM CANONICAL", min_silence_minutes=15)
    assert out["accepted"] is False
    assert out["silence_minutes"] is not None and out["silence_minutes"] < 15


def test_warm_spare_claim_accepts_stale_heartbeat(monkeypatch, tmp_path) -> None:
    mod = _load_failover(monkeypatch, tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    p = tmp_path / "warm_spare"
    p.mkdir()
    (p / "canonical_heartbeat.json").write_text(json.dumps({"ts": old, "hostname": "h"}))
    out = mod.claim_canonical("CLAIM CANONICAL", min_silence_minutes=15)
    assert out["accepted"] is True
    assert out["state"] == mod.FailoverState.CLAIMING.value


# ── Q17.2 local_only drill ──────────────────────────────────────────────


def test_local_only_module_present_and_registered() -> None:
    p = REPO_ROOT / "app" / "resilience_drills" / "drills" / "local_only.py"
    assert p.is_file()
    src = p.read_text()
    assert 'name="local_only"' in src
    assert "DrillRisk.LOW" in src
    assert "drill_local_only_enabled" in src
    init = (REPO_ROOT / "app" / "resilience_drills" / "drills" / "__init__.py").read_text()
    assert "local_only" in init


def _load_local_only_drill():
    return _isolated_module("app/resilience_drills/drills/local_only.py", "_q17_local_only")


def test_local_only_ollama_probe_handles_unreachable(monkeypatch) -> None:
    mod = _load_local_only_drill()
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    out = mod._probe_ollama()
    assert out["provider"] == "ollama"
    assert out["socket_ok"] is False
    assert out["ready"] is False


def test_local_only_key_probe_recognises_format(monkeypatch) -> None:
    mod = _load_local_only_drill()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "a" * 40)
    out = mod._probe_key("groq")
    assert out["env_set"] is True
    assert out["format_ok"] is True
    assert out["ready"] is True


def test_local_only_key_probe_rejects_wrong_format(monkeypatch) -> None:
    mod = _load_local_only_drill()
    monkeypatch.setenv("GROQ_API_KEY", "wrong-prefix-abc")
    out = mod._probe_key("groq")
    assert out["env_set"] is True
    assert out["format_ok"] is False


# ── Q17.3 bit_rot_scan ──────────────────────────────────────────────────


def _load_bit_rot(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/healing/monitors/bit_rot_scan.py", "_q17_bit_rot")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def test_bit_rot_first_run_records_baseline(monkeypatch, tmp_path) -> None:
    mod = _load_bit_rot(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_bit_rot_scan_enabled=True)
    _stub_notify(monkeypatch)
    _stub_ledger(monkeypatch)
    (tmp_path / "identity").mkdir()
    (tmp_path / "identity" / "continuity_ledger.jsonl").write_text("a\nb\n", encoding="utf-8")
    out = mod.run()
    assert out["checked"] is True
    assert out["n_new"] >= 1
    assert out["alerts"] == []


def test_bit_rot_detects_inplace_mutation(monkeypatch, tmp_path) -> None:
    mod = _load_bit_rot(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_bit_rot_scan_enabled=True)
    _stub_notify(monkeypatch)
    ledger = _stub_ledger(monkeypatch)
    (tmp_path / "identity").mkdir()
    p = tmp_path / "identity" / "continuity_ledger.jsonl"
    p.write_text("a\nb\n", encoding="utf-8")
    mod.run()
    state_p = tmp_path / "healing" / "bit_rot_state.json"
    s = json.loads(state_p.read_text())
    s["last_run"] = 0
    state_p.write_text(json.dumps(s))
    p.write_text("c\nb\n", encoding="utf-8")
    out = mod.run()
    assert out["checked"] is True
    assert any(a["change"] == "inplace_mutated" for a in out["alerts"])
    assert any(e["kind"] == "q17_landmark" for e in ledger)


def test_bit_rot_detects_shrinkage(monkeypatch, tmp_path) -> None:
    mod = _load_bit_rot(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_bit_rot_scan_enabled=True)
    _stub_notify(monkeypatch)
    _stub_ledger(monkeypatch)
    (tmp_path / "identity").mkdir()
    p = tmp_path / "identity" / "continuity_ledger.jsonl"
    p.write_text("aaaa\nbbbb\ncccc\n", encoding="utf-8")
    mod.run()
    state_p = tmp_path / "healing" / "bit_rot_state.json"
    s = json.loads(state_p.read_text())
    s["last_run"] = 0
    state_p.write_text(json.dumps(s))
    p.write_text("aaaa\n", encoding="utf-8")
    out = mod.run()
    assert any(a["change"] == "shrunk" for a in out["alerts"])


def test_bit_rot_append_is_not_an_alert(monkeypatch, tmp_path) -> None:
    mod = _load_bit_rot(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_bit_rot_scan_enabled=True)
    _stub_notify(monkeypatch)
    _stub_ledger(monkeypatch)
    (tmp_path / "identity").mkdir()
    p = tmp_path / "identity" / "continuity_ledger.jsonl"
    p.write_text("a\nb\n", encoding="utf-8")
    mod.run()
    state_p = tmp_path / "healing" / "bit_rot_state.json"
    s = json.loads(state_p.read_text())
    s["last_run"] = 0
    state_p.write_text(json.dumps(s))
    p.write_text("a\nb\nc\nd\n", encoding="utf-8")
    out = mod.run()
    assert out["n_append_ok"] >= 1
    assert out["alerts"] == []


def test_bit_rot_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_bit_rot(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_bit_rot_scan_enabled=False)
    out = mod.run()
    assert out.get("skipped") is True


# ── Q17.4 operator_transition ──────────────────────────────────────────


def _load_op_state(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/operator_transition/state.py", "_q17_op_state")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def test_operator_transition_phase_active(monkeypatch, tmp_path) -> None:
    mod = _load_op_state(monkeypatch, tmp_path)
    _stub_notify(monkeypatch)
    _stub_ledger(monkeypatch)
    audit = tmp_path / "audit.log"
    row = {"ts": datetime.now(timezone.utc).isoformat(), "kind": "request_received", "sender_id": "signal:+358"}
    audit.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cur = mod.current_phase()
    assert cur["phase"] == mod.OperatorPhase.ACTIVE.value


def test_operator_transition_phase_absent_30d(monkeypatch, tmp_path) -> None:
    mod = _load_op_state(monkeypatch, tmp_path)
    _stub_notify(monkeypatch)
    _stub_ledger(monkeypatch)
    audit = tmp_path / "audit.log"
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    row = {"ts": old, "kind": "request_received", "sender_id": "signal:+358"}
    audit.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cur = mod.current_phase()
    assert cur["phase"] == mod.OperatorPhase.ABSENT_30D.value


def test_operator_transition_phase_read_mostly(monkeypatch, tmp_path) -> None:
    mod = _load_op_state(monkeypatch, tmp_path)
    _stub_notify(monkeypatch)
    _stub_ledger(monkeypatch)
    audit = tmp_path / "audit.log"
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    row = {"ts": old, "kind": "request_received", "sender_id": "signal:+358"}
    audit.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cur = mod.current_phase()
    assert cur["phase"] == mod.OperatorPhase.READ_MOSTLY.value


def _load_successor(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/operator_transition/successor.py", "_q17_succ")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def test_successor_declare_save_load_roundtrip(monkeypatch, tmp_path) -> None:
    mod = _load_successor(monkeypatch, tmp_path)
    _stub_ledger(monkeypatch)
    decl = mod.declare_successor("Asta R.", signal_id="signal:+372...", instructions="Honour the ritual.")
    out = mod.load_successor()
    assert out is not None
    assert out.successor_name == "Asta R."


# ── Q17.5 agreement_ledger ──────────────────────────────────────────────


def _load_agreement(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/self_model/agreement_ledger.py", "_q17_agreement")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def test_agreement_ledger_record_and_respond(monkeypatch, tmp_path) -> None:
    mod = _load_agreement(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_agreement_ledger_enabled=True)
    sid = mod.record_suggestion(category="library_radar", source_module="x", summary="adopt rich")
    mod.record_response(sid, mod.AgreementResponse.ACCEPTED)
    rate = mod.rolling_rate("library_radar")
    assert rate["n"] == 1
    assert rate["by_response"]["accepted"] == 1


def test_agreement_ledger_rolling_rate_buckets(monkeypatch, tmp_path) -> None:
    mod = _load_agreement(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_agreement_ledger_enabled=True)
    for i in range(5):
        sid = mod.record_suggestion(category="proactive_briefing", summary=f"s{i}")
        mod.record_response(sid, mod.AgreementResponse.ACCEPTED)
    for i in range(3):
        sid = mod.record_suggestion(category="proactive_briefing", summary=f"s_r_{i}")
        mod.record_response(sid, mod.AgreementResponse.REJECTED)
    rate = mod.rolling_rate("proactive_briefing")
    assert rate["n"] == 8
    assert rate["rates"]["accepted"] == 0.625
    assert rate["rates"]["rejected"] == 0.375


def test_agreement_ledger_briefing_section(monkeypatch, tmp_path) -> None:
    mod = _load_agreement(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_agreement_ledger_enabled=True)
    sid = mod.record_suggestion(category="paper_pipeline")
    mod.record_response(sid, mod.AgreementResponse.ACCEPTED)
    text = mod.briefing_section()
    assert "paper_pipeline" in text


def test_agreement_ledger_unknown_category_buckets_to_other(monkeypatch, tmp_path) -> None:
    mod = _load_agreement(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_agreement_ledger_enabled=True)
    sid = mod.record_suggestion(category="not_a_real_category")
    mod.record_response(sid, mod.AgreementResponse.IGNORED)
    rows = mod._read_all()
    cats = [r.get("category") for r in rows if "category" in r]
    assert "other" in cats


# ── Q17.6 kb_contradiction ─────────────────────────────────────────────


def _load_kb_contradict(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/healing/monitors/kb_contradiction.py", "_q17_kb_contradict")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def test_kb_contradiction_detects_simple_negation(monkeypatch, tmp_path) -> None:
    mod = _load_kb_contradict(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_kb_contradiction_monitor_enabled=True)
    _stub_notify(monkeypatch)
    _stub_ledger(monkeypatch)
    (tmp_path / "epistemic").mkdir()
    rows = [
        {"id": "a", "subject": "espresso", "text": "Espresso is best brewed at 93 degrees"},
        {"id": "b", "subject": "espresso", "text": "Espresso isn't best brewed at 93 degrees"},
    ]
    p = tmp_path / "epistemic" / "claims.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = mod.run(rng_seed=0)
    assert out["n_contradictions"] >= 1


def test_kb_contradiction_skips_without_shared_tokens(monkeypatch, tmp_path) -> None:
    mod = _load_kb_contradict(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_kb_contradiction_monitor_enabled=True)
    _stub_notify(monkeypatch)
    _stub_ledger(monkeypatch)
    (tmp_path / "epistemic").mkdir()
    rows = [
        {"id": "a", "subject": "alpha", "text": "Spring is warm."},
        {"id": "b", "subject": "alpha", "text": "Winter isn't cold."},
    ]
    p = tmp_path / "epistemic" / "claims.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = mod.run(rng_seed=0)
    assert out["n_contradictions"] == 0


def test_kb_contradiction_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_kb_contradict(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_kb_contradiction_monitor_enabled=False)
    out = mod.run()
    assert out.get("skipped") is True


# ── Q17.7 synthesis_pass ───────────────────────────────────────────────


def _load_synthesis(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/creativity/synthesis_pass.py", "_q17_synthesis")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def test_synthesis_pass_picks_distinct_subsystems(monkeypatch, tmp_path) -> None:
    mod = _load_synthesis(monkeypatch, tmp_path)
    import random
    pairs = mod._pick_pairs(random.Random(42), 5)
    for i, j in pairs:
        assert i != j


def test_synthesis_pass_produces_candidate_with_stub_llm(monkeypatch, tmp_path) -> None:
    mod = _load_synthesis(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_synthesis_pass_enabled=True)
    _stub_ledger(monkeypatch)
    from dataclasses import dataclass, field as df

    @dataclass
    class FakeBlend:
        blend_label: str = "test blend"
        emergent_structure: list = df(default_factory=lambda: ["e1", "e2"])
        follow_on_questions: list = df(default_factory=list)

    def fake_blend_concepts(a, b, llm_call=None):
        return FakeBlend()

    cb_mod = sys.modules.get("app.creativity.concept_blend")
    if cb_mod is None:
        cb_mod = type(sys)("app.creativity.concept_blend")
        sys.modules["app.creativity.concept_blend"] = cb_mod

    @dataclass
    class FakeInput:
        label: str
        salient_elements: list = df(default_factory=list)
        salient_relations: list = df(default_factory=list)

    cb_mod.InputSpace = FakeInput
    cb_mod.blend_concepts = fake_blend_concepts

    out = mod.run_one_pass(rng_seed=0)
    assert out["n_persisted"] >= 1
    assert out["candidates"]
    assert out["candidates"][0]["subsystem_a"] != out["candidates"][0]["subsystem_b"]


def test_synthesis_pass_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_synthesis(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_synthesis_pass_enabled=False)
    out = mod.run_one_pass()
    assert out.get("skipped") is True


# ── Q17.8 conversation_memory ──────────────────────────────────────────


def _load_index(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/conversation_memory/temporal_index.py", "_q17_temporal_index")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def _load_retrieval(monkeypatch, tmp_path: Path):
    mod = _isolated_module("app/conversation_memory/retrieval.py", "_q17_retrieval")
    monkeypatch.setattr(mod, "_workspace_root", lambda: tmp_path)
    return mod


def _seed_audit(tmp_path: Path, rows: list[dict]) -> None:
    p = tmp_path / "audit.log"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_conversation_memory_scan_builds_index(monkeypatch, tmp_path) -> None:
    mod = _load_index(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_conversation_memory_enabled=True)
    rows = [
        {"ts": "2026-01-15T10:00:00+00:00", "kind": "request_received", "message": "how to brew espresso properly"},
        {"ts": "2026-02-20T11:00:00+00:00", "kind": "response_sent", "message": "you need a 30s extraction time"},
    ]
    _seed_audit(tmp_path, rows)
    summary = mod.scan_audit_log()
    assert summary["n_indexed"] == 2


def test_conversation_memory_redacts_email(monkeypatch, tmp_path) -> None:
    mod = _load_index(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_conversation_memory_enabled=True)
    rows = [{"ts": "2026-01-15T10:00:00+00:00", "kind": "request_received",
             "message": "email me at user@example.com tomorrow"}]
    _seed_audit(tmp_path, rows)
    mod.scan_audit_log()
    idx = tmp_path / "conversation_memory" / "index.jsonl"
    blob = idx.read_text()
    assert "user@example.com" not in blob
    assert "<email>" in blob


def test_conversation_memory_recall_finds_token_overlap(monkeypatch, tmp_path) -> None:
    idx_mod = _load_index(monkeypatch, tmp_path)
    ret_mod = _load_retrieval(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_conversation_memory_enabled=True)
    rows = [
        {"ts": "2026-01-15T10:00:00+00:00", "kind": "request_received",
         "message": "should we adopt the rich logging library"},
        {"ts": "2026-02-20T11:00:00+00:00", "kind": "request_received",
         "message": "weather is great today"},
    ]
    _seed_audit(tmp_path, rows)
    idx_mod.scan_audit_log()
    refs = ret_mod.recall("rich logging library", window_months=24, top_k=5)
    assert len(refs) >= 1
    assert "rich" in refs[0].tokens_matched or "logging" in refs[0].tokens_matched


def test_conversation_memory_recall_disabled(monkeypatch, tmp_path) -> None:
    ret_mod = _load_retrieval(monkeypatch, tmp_path)
    _stub_runtime_settings(monkeypatch, get_conversation_memory_enabled=False)
    assert ret_mod.recall("anything") == []


# ── cross-cutting wiring ───────────────────────────────────────────────


def test_q17_event_kind_in_continuity_ledger() -> None:
    src = (REPO_ROOT / "app" / "identity" / "continuity_ledger.py").read_text()
    assert "q17_landmark" in src
    assert "PROGRAM §52" in src


def test_q17_master_switches_present_in_runtime_settings() -> None:
    src = (REPO_ROOT / "app" / "runtime_settings.py").read_text()
    for key in (
        "warm_spare_enabled",
        "warm_spare_partner_target",
        "drill_local_only_enabled",
        "bit_rot_scan_enabled",
        "operator_transition_enabled",
        "agreement_ledger_enabled",
        "kb_contradiction_monitor_enabled",
        "synthesis_pass_enabled",
        "conversation_memory_enabled",
    ):
        assert key in src, f"missing master switch {key}"


def test_q17_monitors_registered_in_driver_loop() -> None:
    src = (REPO_ROOT / "app" / "healing" / "monitors" / "__init__.py").read_text()
    assert "bit_rot_scan" in src
    assert "kb_contradiction" in src
    assert '"bit_rot_scan":' in src
    assert '"kb_contradiction":' in src


def test_q17_synthesis_pass_anchored_in_healing_init() -> None:
    src = (REPO_ROOT / "app" / "healing" / "__init__.py").read_text()
    assert "synthesis_pass" in src
    assert "local_only" in src


def test_q17_local_only_drill_in_drills_init() -> None:
    src = (REPO_ROOT / "app" / "resilience_drills" / "drills" / "__init__.py").read_text()
    assert "local_only" in src
