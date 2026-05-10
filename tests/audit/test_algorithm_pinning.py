"""Tests for app.audit.algorithm_pinning + crypto_rotation_drill monitor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.audit.algorithm_pinning import (
    KNOWN_ARTIFACT_CLASSES,
    list_pins,
    missing_artifact_classes,
    pin_algorithm,
    run_rotation_drill,
    stale_pins,
)


# ── pin_algorithm ─────────────────────────────────────────────────────


def test_pin_then_list_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    pin_algorithm(
        "subia_integrity_manifest",
        "sha256",
        rationale="canonical SubIA pin",
        path=p,
    )
    pins = list_pins(path=p)
    assert len(pins) == 1
    assert pins[0].artifact_class == "subia_integrity_manifest"
    assert pins[0].algorithm == "sha256"
    assert pins[0].rationale == "canonical SubIA pin"


def test_repin_replaces_prior(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    pin_algorithm("rolled_audit_log", "sha256", path=p)
    pin_algorithm("rolled_audit_log", "sha3_256", path=p)
    pins = list_pins(path=p)
    assert len(pins) == 1
    assert pins[0].algorithm == "sha3_256"


def test_pin_unknown_algorithm_raises(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        pin_algorithm("rolled_audit_log", "fake_algo_42", path=p)


def test_pin_empty_artifact_class_raises(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        pin_algorithm("   ", "sha256", path=p)


def test_pin_disabled_short_circuits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALGORITHM_PINNING_ENABLED", "false")
    with pytest.raises(RuntimeError):
        pin_algorithm("x", "sha256", path=tmp_path / "m.json")


# ── stale_pins ───────────────────────────────────────────────────────


def test_stale_pins_flags_old(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    now = datetime(2026, 5, 15, tzinfo=timezone.utc)
    old = now - timedelta(days=800)
    pin_algorithm(
        "subia_integrity_manifest", "sha256", path=p, now=old,
    )
    stales = stale_pins(interval_days=730, path=p, now=now)
    assert len(stales) == 1
    assert stales[0].artifact_class == "subia_integrity_manifest"


def test_fresh_pins_not_stale(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    now = datetime(2026, 5, 15, tzinfo=timezone.utc)
    recent = now - timedelta(days=30)
    pin_algorithm("rolled_audit_log", "sha256", path=p, now=recent)
    assert stale_pins(interval_days=730, path=p, now=now) == []


def test_unparseable_timestamp_treated_as_ancient(tmp_path: Path) -> None:
    """A pin with a malformed pinned_at is flagged stale so the operator
    can fix the manifest."""
    p = tmp_path / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps({"pins": [{
        "artifact_class": "x", "algorithm": "sha256",
        "pinned_at": "not-a-date",
    }]}))
    stales = stale_pins(path=p)
    assert len(stales) == 1


# ── missing_artifact_classes ────────────────────────────────────────


def test_missing_artifact_classes_lists_unpinned(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    pin_algorithm("subia_integrity_manifest", "sha256", path=p)
    missing = set(missing_artifact_classes(path=p))
    expected = KNOWN_ARTIFACT_CLASSES - {"subia_integrity_manifest"}
    assert missing == expected


def test_no_manifest_means_all_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.json"
    missing = set(missing_artifact_classes(path=p))
    assert missing == KNOWN_ARTIFACT_CLASSES


# ── run_rotation_drill ────────────────────────────────────────────────


def test_drill_passes_for_real_algorithms() -> None:
    out = run_rotation_drill(
        "rolled_audit_log",
        legacy_algorithm="sha256",
        target_algorithm="sha3_256",
    )
    assert out.ok
    assert out.error == ""
    assert len(out.legacy_chain_root) == 64  # sha256 hex
    assert len(out.target_chain_root) == 64  # sha3_256 hex
    assert out.legacy_chain_root != out.target_chain_root


def test_drill_with_unknown_algorithm_fails_gracefully() -> None:
    out = run_rotation_drill(
        "rolled_audit_log",
        target_algorithm="bogus_algo_42",
    )
    assert not out.ok
    assert "not available" in out.error


def test_drill_is_deterministic_across_runs() -> None:
    out_a = run_rotation_drill("x")
    out_b = run_rotation_drill("x")
    assert out_a.legacy_chain_root == out_b.legacy_chain_root
    assert out_a.target_chain_root == out_b.target_chain_root


def test_drill_with_custom_samples() -> None:
    samples = [b"a", b"b", b"c", b"d"]
    out = run_rotation_drill(
        "rolled_audit_log",
        sample_entries=samples,
    )
    assert out.ok
    assert out.n_entries == 4


# ── crypto_rotation_drill monitor ────────────────────────────────────


@pytest.fixture
def monitor_isolated(tmp_path, monkeypatch):
    from app.healing.handlers import _common as _h_common
    from app.healing.monitors import crypto_rotation_drill as crd

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "state")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        crd, "send_signal_alert",
        lambda body, tag=None, **kw: sent.append((body, tag or "")),
    )
    monkeypatch.setattr(crd, "audit_event", lambda *a, **k: None)
    yield tmp_path, sent


def test_monitor_alerts_when_no_pins_exist(monitor_isolated) -> None:
    tmp, sent = monitor_isolated
    from app.healing.monitors import crypto_rotation_drill as crd
    out = crd.run(manifest_path=tmp / "missing.json")
    assert out["ran"] is True
    assert set(out["missing"]) == KNOWN_ARTIFACT_CLASSES
    assert any("missing_pins" in tag for _body, tag in sent)


def test_monitor_silent_when_pins_fresh(monitor_isolated) -> None:
    tmp, sent = monitor_isolated
    from app.healing.monitors import crypto_rotation_drill as crd

    p = tmp / "manifest.json"
    for cls in KNOWN_ARTIFACT_CLASSES:
        pin_algorithm(cls, "sha256", path=p)

    out = crd.run(manifest_path=p)
    assert out["ran"] is True
    assert out["missing"] == []
    assert out["stale"] == []
    assert out["drill_ok"] is True
    # No missing/stale alerts; drill_ok so no drill_failed alert.
    tags = {tag for _body, tag in sent}
    assert "crypto_rotation:missing_pins" not in tags
    assert "crypto_rotation:drill_failed" not in tags


def test_monitor_alerts_on_stale_pins(monitor_isolated) -> None:
    tmp, sent = monitor_isolated
    from app.healing.monitors import crypto_rotation_drill as crd

    p = tmp / "manifest.json"
    old = datetime.now(timezone.utc) - timedelta(days=800)
    for cls in KNOWN_ARTIFACT_CLASSES:
        pin_algorithm(cls, "sha256", path=p, now=old)

    out = crd.run(manifest_path=p)
    assert any("stale_pins" in tag for _body, tag in sent)
    assert len(out["stale"]) == len(KNOWN_ARTIFACT_CLASSES)


def test_monitor_drill_failure_alerts(monitor_isolated, monkeypatch) -> None:
    tmp, sent = monitor_isolated
    from app.healing.monitors import crypto_rotation_drill as crd

    # Force the drill to fail by pointing target at a bogus algorithm.
    monkeypatch.setenv("CRYPTO_ROTATION_TARGET_ALGORITHM", "bogus_algo_42")

    p = tmp / "manifest.json"
    for cls in KNOWN_ARTIFACT_CLASSES:
        pin_algorithm(cls, "sha256", path=p)

    out = crd.run(manifest_path=p)
    assert out["drill_ok"] is False
    assert any("drill_failed" in tag for _body, tag in sent)


def test_monitor_disabled_short_circuits(monitor_isolated, monkeypatch) -> None:
    tmp, sent = monitor_isolated
    monkeypatch.setenv("CRYPTO_ROTATION_DRILL_MONITOR_ENABLED", "false")
    from app.healing.monitors import crypto_rotation_drill as crd
    out = crd.run(manifest_path=tmp / "missing.json")
    assert out["ran"] is False
    assert sent == []
