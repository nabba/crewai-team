"""Tests for ``app.healing.pattern_learner`` (Phase C #4)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing import pattern_learner
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(pattern_learner, "_ERRORS_LOG", tmp_path / "errors.jsonl")
    monkeypatch.setattr(pattern_learner, "_PROPOSED_DIR", tmp_path / "proposed_runbooks")

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(pattern_learner, "send_signal_alert",
                        lambda body, tag=None, **kw: sent.append((body, tag or "")))
    monkeypatch.setattr(pattern_learner, "audit_event", lambda *a, **k: None)

    # No registered runbooks by default in this fixture — exercise the
    # uncovered path. Tests can override via monkeypatch on
    # _registered_signatures.
    monkeypatch.setattr(pattern_learner, "_registered_signatures", lambda: set())

    yield tmp_path, sent


def _seed_errors(path, n: int, message: str, level: str = "ERROR") -> None:
    rows = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        rows.append({
            "ts": (now - timedelta(hours=i % 48)).isoformat(),
            "level": level,
            "logger": "app.foo",
            "module": "foo", "lineno": 1,
            "message": message,
        })
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_no_errors_no_proposals(isolated):
    tmp_path, sent = isolated
    from app.healing import pattern_learner
    summary = pattern_learner.run()
    assert summary["ran"] is True
    assert summary["new_proposals"] == 0
    assert sent == []


def test_below_threshold_skipped(isolated):
    tmp_path, sent = isolated
    from app.healing import pattern_learner
    _seed_errors(tmp_path / "errors.jsonl", n=5, message="rare bug X")
    summary = pattern_learner.run()
    # min occurrences is 10
    assert summary["new_proposals"] == 0
    assert sent == []


def test_above_threshold_proposed(isolated):
    tmp_path, sent = isolated
    from app.healing import pattern_learner
    _seed_errors(tmp_path / "errors.jsonl", n=15,
                 message="common pattern Y broken")
    summary = pattern_learner.run()
    assert summary["new_proposals"] == 1
    files = list((tmp_path / "proposed_runbooks").glob("*.md"))
    assert len(files) == 1
    body = files[0].read_text()
    assert "common pattern Y broken" in body
    assert "## Suggested next steps" in body
    assert any("pattern_learner" in tag for _, tag in sent)


def test_skips_already_covered_signature(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing import pattern_learner

    _seed_errors(tmp_path / "errors.jsonl", n=15, message="known pattern X")
    # Compute the signature the learner would produce.
    from app.healing.handlers._common import compute_signature
    sig = compute_signature("app.foo", "known pattern X")
    monkeypatch.setattr(pattern_learner, "_registered_signatures",
                        lambda: {sig})
    summary = pattern_learner.run()
    assert summary["new_proposals"] == 0


def test_dedup_within_window(isolated):
    tmp_path, sent = isolated
    from app.healing import pattern_learner
    _seed_errors(tmp_path / "errors.jsonl", n=15, message="dedup pattern Z")
    pattern_learner.run()
    initial_count = sum(1 for _, _ in sent)

    # Reset cadence; rerun.
    state_path = tmp_path / "self_heal" / "pattern_learner.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    pattern_learner.run()

    # Same pattern should not re-alert (14-day dedup).
    assert sum(1 for _, _ in sent) == initial_count


def test_only_warn_error_critical_count(isolated):
    tmp_path, sent = isolated
    from app.healing import pattern_learner
    _seed_errors(tmp_path / "errors.jsonl", n=15,
                 message="info-level chatter", level="INFO")
    summary = pattern_learner.run()
    # INFO entries should be filtered out, no proposals.
    assert summary["new_proposals"] == 0


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("HEALING_PATTERN_LEARNER_ENABLED", "0")
    from app.healing import pattern_learner
    summary = pattern_learner.run()
    assert summary["ran"] is False
