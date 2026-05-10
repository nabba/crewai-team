"""Tests for app.healing.monitors.version_upgrade_drill (§2.5)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.healing.monitors import version_upgrade_drill as vud


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Each test gets a fresh state dir + alert capture."""
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "state")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        vud, "send_signal_alert",
        lambda body, tag=None, **kw: sent.append((body, tag or "")),
    )
    monkeypatch.setattr(vud, "audit_event", lambda *a, **k: None)
    yield tmp_path, sent


def _write_manifest(path: Path, **fields) -> None:
    base = {
        "runs": [],
        "last_drill_at": fields.get("last_drill_at"),
        "last_drill_ok": fields.get("last_drill_ok", True),
        "last_target_versions": fields.get("last_target_versions") or {
            "postgres": "pgvector/pgvector:0.8.0-pg17",
            "neo4j": "neo4j:5.21",
            "chromadb": "chromadb/chroma:1.0",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base))


def _iso_days_ago(d: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=d)).isoformat()


# ── core paths ─────────────────────────────────────────────────────────


def test_no_manifest_alerts_never_run(isolated_state) -> None:
    tmp, sent = isolated_state
    out = vud.run(manifest_path=tmp / "missing.json", now=time.time())
    assert out["ran"] is True
    assert out["manifest_present"] is False
    assert out["alert_fired"] is True
    assert out["alert_tag"] == "version_upgrade_drill:never_run"
    assert any("never been tested" in body for body, _ in sent)


def test_recent_ok_drill_no_alert(isolated_state) -> None:
    tmp, sent = isolated_state
    mp = tmp / "manifest.json"
    _write_manifest(mp, last_drill_at=_iso_days_ago(30), last_drill_ok=True)

    out = vud.run(manifest_path=mp, now=time.time())
    assert out["ran"] is True
    assert out["alert_fired"] is False
    assert sent == []
    assert out["last_drill_ok"] is True
    assert out["last_drill_age_days"] is not None
    assert 29 <= out["last_drill_age_days"] <= 31


def test_stale_drill_alerts(isolated_state) -> None:
    tmp, sent = isolated_state
    mp = tmp / "manifest.json"
    _write_manifest(mp, last_drill_at=_iso_days_ago(120), last_drill_ok=True)

    out = vud.run(manifest_path=mp, now=time.time())
    assert out["alert_fired"] is True
    assert out["alert_tag"] == "version_upgrade_drill:stale"
    assert any("stale" in body.lower() for body, _ in sent)


def test_failed_drill_alerts_with_target_versions(isolated_state) -> None:
    tmp, sent = isolated_state
    mp = tmp / "manifest.json"
    _write_manifest(
        mp,
        last_drill_at=_iso_days_ago(7),
        last_drill_ok=False,
        last_target_versions={
            "postgres": "pgvector/pgvector:0.8.0-pg17",
            "neo4j": "neo4j:5.21",
            "chromadb": "chromadb/chroma:1.0",
        },
    )

    out = vud.run(manifest_path=mp, now=time.time())
    assert out["alert_fired"] is True
    assert out["alert_tag"] == "version_upgrade_drill:failed"
    body = sent[0][0]
    assert "FAILED" in body
    assert "postgres=pgvector/pgvector:0.8.0-pg17" in body


def test_unparseable_last_drill_at_alerts_stale(isolated_state) -> None:
    tmp, sent = isolated_state
    mp = tmp / "manifest.json"
    _write_manifest(mp, last_drill_at="not-a-timestamp", last_drill_ok=True)

    out = vud.run(manifest_path=mp, now=time.time())
    assert out["alert_fired"] is True
    assert out["alert_tag"] == "version_upgrade_drill:stale"


# ── cadence + dedup ────────────────────────────────────────────────────


def test_cadence_guard_skips_within_24h(isolated_state) -> None:
    tmp, sent = isolated_state
    mp = tmp / "manifest.json"
    _write_manifest(mp, last_drill_at=_iso_days_ago(120), last_drill_ok=True)

    # First run fires alert.
    out1 = vud.run(manifest_path=mp, now=time.time())
    assert out1["ran"] is True

    # Second run within cadence window — skip + no alert.
    out2 = vud.run(manifest_path=mp, now=time.time())
    assert out2["ran"] is False


def test_alert_dedup_per_tag_within_window(isolated_state) -> None:
    tmp, sent = isolated_state
    mp = tmp / "manifest.json"
    _write_manifest(mp, last_drill_at=_iso_days_ago(120), last_drill_ok=True)

    # Run, fire, advance time past cadence but within dedup window.
    base = time.time()
    vud.run(manifest_path=mp, now=base)
    vud.run(manifest_path=mp, now=base + 26 * 3600)  # cadence cleared
    # Two probes; only first one fired the stale alert (dedup).
    stale_alerts = [body for body, tag in sent if tag == "version_upgrade_drill:stale"]
    assert len(stale_alerts) == 1


def test_alert_re_fires_after_dedup_window(isolated_state) -> None:
    tmp, sent = isolated_state
    mp = tmp / "manifest.json"
    _write_manifest(mp, last_drill_at=_iso_days_ago(120), last_drill_ok=True)

    base = time.time()
    vud.run(manifest_path=mp, now=base)
    # 15 days later — past the 14-day dedup window.
    vud.run(manifest_path=mp, now=base + 15 * 86400)
    stale_alerts = [body for body, tag in sent if tag == "version_upgrade_drill:stale"]
    assert len(stale_alerts) == 2


# ── disabled + threshold env ───────────────────────────────────────────


def test_disabled_short_circuits(isolated_state, monkeypatch) -> None:
    tmp, sent = isolated_state
    monkeypatch.setenv("VERSION_UPGRADE_DRILL_MONITOR_ENABLED", "false")
    out = vud.run(manifest_path=tmp / "missing.json", now=time.time())
    assert out["ran"] is False
    assert sent == []


def test_stale_threshold_env_override(isolated_state, monkeypatch) -> None:
    tmp, sent = isolated_state
    monkeypatch.setenv("VERSION_UPGRADE_DRILL_STALE_DAYS", "30")
    mp = tmp / "manifest.json"
    # 45 days old — passes default 100d threshold but exceeds the 30d override.
    _write_manifest(mp, last_drill_at=_iso_days_ago(45), last_drill_ok=True)
    out = vud.run(manifest_path=mp, now=time.time())
    assert out["alert_fired"] is True
    assert out["alert_tag"] == "version_upgrade_drill:stale"


def test_threshold_floor_is_seven_days(monkeypatch) -> None:
    """A nonsensical low threshold is clamped to 7 days minimum."""
    monkeypatch.setenv("VERSION_UPGRADE_DRILL_STALE_DAYS", "1")
    assert vud._stale_days() == 7
    monkeypatch.setenv("VERSION_UPGRADE_DRILL_STALE_DAYS", "garbage")
    assert vud._stale_days() == 100  # default


# ── format helpers ─────────────────────────────────────────────────────


def test_format_targets_with_versions() -> None:
    out = vud._format_targets({
        "last_target_versions": {
            "postgres": "pgvector/pgvector:0.8.0-pg17",
            "neo4j": "neo4j:5.21",
            "chromadb": "chromadb/chroma:1.0",
        },
    })
    assert "postgres=pgvector/pgvector:0.8.0-pg17" in out
    assert "neo4j=neo4j:5.21" in out


def test_format_targets_unknown_when_missing() -> None:
    assert vud._format_targets({}) == "(unknown)"
    assert vud._format_targets({"last_target_versions": {}}) == "(unknown)"
