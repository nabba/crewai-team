"""Tests for ``app.coding_session.audit_verify`` + the daily monitor."""
from __future__ import annotations

import hashlib
import json
import time

import pytest


def _line(prev: str, payload: dict) -> dict:
    """Build one chain entry the way ``store._append_audit`` would."""
    body = json.dumps(payload, sort_keys=True, default=str)
    h = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()[:16]
    return {
        "ts": "2026-05-09T00:00:00+00:00",
        "prev_hash": prev,
        "entry_hash": h,
        "payload": payload,
    }


def _write_chain(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, default=str) + "\n")


# ── verify_chain ──────────────────────────────────────────────────────────


def test_intact_chain_passes(tmp_path):
    from app.coding_session.audit_verify import verify_chain

    audit = tmp_path / "audit.jsonl"
    a = _line("", {"event": "started", "id": "1"})
    b = _line(a["entry_hash"], {"event": "submitted", "id": "1"})
    c = _line(b["entry_hash"], {"event": "applied", "id": "1"})
    _write_chain(audit, [a, b, c])

    ok, broken = verify_chain(audit_path=audit)
    assert ok
    assert broken == []


def test_missing_chain_returns_ok(tmp_path):
    from app.coding_session.audit_verify import verify_chain
    ok, broken = verify_chain(audit_path=tmp_path / "nope.jsonl")
    assert ok
    assert broken == []


def test_empty_chain_returns_ok(tmp_path):
    from app.coding_session.audit_verify import verify_chain
    audit = tmp_path / "audit.jsonl"
    audit.write_text("")
    ok, broken = verify_chain(audit_path=audit)
    assert ok


def test_tampered_payload_detected(tmp_path):
    from app.coding_session.audit_verify import verify_chain

    audit = tmp_path / "audit.jsonl"
    a = _line("", {"event": "started", "id": "1"})
    b = _line(a["entry_hash"], {"event": "submitted", "id": "1"})
    # Mutate the payload of b without recomputing the hash.
    b_tampered = dict(b)
    b_tampered["payload"] = {"event": "submitted", "id": "ATTACKER"}
    _write_chain(audit, [a, b_tampered])

    ok, broken = verify_chain(audit_path=audit)
    assert not ok
    reasons = [b["reason"] for b in broken]
    assert "entry_hash_mismatch" in reasons


def test_prev_hash_break_detected(tmp_path):
    from app.coding_session.audit_verify import verify_chain

    audit = tmp_path / "audit.jsonl"
    a = _line("", {"event": "started", "id": "1"})
    # Forge a "b" whose prev_hash doesn't match a.entry_hash.
    b = _line("deadbeefdeadbeef", {"event": "submitted", "id": "1"})
    _write_chain(audit, [a, b])

    ok, broken = verify_chain(audit_path=audit)
    assert not ok
    reasons = [item["reason"] for item in broken]
    assert "prev_hash_mismatch" in reasons


def test_malformed_json_reported(tmp_path):
    from app.coding_session.audit_verify import verify_chain

    audit = tmp_path / "audit.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    a = _line("", {"event": "started"})
    audit.write_text(json.dumps(a) + "\n{not json}\n")

    ok, broken = verify_chain(audit_path=audit)
    assert not ok
    assert any(b["reason"] == "invalid_json" for b in broken)


def test_one_break_does_not_cascade(tmp_path):
    """A single break should report ONCE, not flag every subsequent line."""
    from app.coding_session.audit_verify import verify_chain

    audit = tmp_path / "audit.jsonl"
    a = _line("", {"event": "1"})
    b = _line(a["entry_hash"], {"event": "2"})
    # Tamper b's payload only; subsequent c uses b's claimed hash so
    # the chain "looks" intact downstream.
    b_tampered = dict(b)
    b_tampered["payload"] = {"event": "2-tampered"}
    c = _line(b["entry_hash"], {"event": "3"})
    _write_chain(audit, [a, b_tampered, c])

    ok, broken = verify_chain(audit_path=audit)
    assert not ok
    # Only ONE entry should be flagged (the tampered b), not c.
    assert len(broken) == 1
    assert broken[0]["line_no"] == 2


def test_chain_summary_reports_counts(tmp_path):
    from app.coding_session.audit_verify import chain_summary

    audit = tmp_path / "audit.jsonl"
    a = _line("", {"event": "1"})
    b = _line(a["entry_hash"], {"event": "2"})
    _write_chain(audit, [a, b])

    summ = chain_summary(audit_path=audit)
    assert summ["exists"]
    assert summ["lines"] == 2
    assert summ["ok"]
    assert summ["broken_count"] == 0
    assert summ["last_entry_hash"] == b["entry_hash"]


# ── Monitor cadence + alert behaviour ─────────────────────────────────────


def test_monitor_alerts_on_first_break(tmp_path, monkeypatch):
    from app.life_companion import _common
    from app.healing.monitors import audit_chain_check

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    sent: list[str] = []
    monkeypatch.setattr(audit_chain_check, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(audit_chain_check, "audit_event",
                        lambda *a, **k: None)
    monkeypatch.setattr(audit_chain_check, "background_enabled", lambda: True)

    audit = tmp_path / "audit.jsonl"
    a = _line("", {"event": "1"})
    b = _line(a["entry_hash"], {"event": "2"})
    bad = dict(b); bad["payload"] = {"event": "tampered"}
    _write_chain(audit, [a, bad])

    monkeypatch.setattr(
        "app.coding_session.audit_verify._AUDIT_LOG", audit,
    )

    audit_chain_check.run()
    assert any("audit chain integrity check failed" in s for s in sent)


def test_monitor_cadence_skips_under_window(tmp_path, monkeypatch):
    from app.life_companion import _common
    from app.healing.monitors import audit_chain_check

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(audit_chain_check, "background_enabled", lambda: True)
    sent: list[str] = []
    monkeypatch.setattr(audit_chain_check, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)

    # Pre-write state with a recent last_run_at so cadence guard skips.
    from app.life_companion._common import write_state_json
    write_state_json("audit_chain_check.json", {"last_run_at": time.time()})

    audit_chain_check.run()
    # No work happened — no alert.
    assert sent == []
