"""Tests for ``app.training.adapter_lifecycle``."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect adapter dirs + state dir to tmp."""
    adapters = tmp_path / "training_adapters"
    models = tmp_path / "trained_models"
    adapters.mkdir()
    models.mkdir()

    from app.training import adapter_lifecycle
    from app.life_companion import _common

    monkeypatch.setattr(adapter_lifecycle, "_ADAPTERS_DIR", adapters)
    monkeypatch.setattr(adapter_lifecycle, "_MODELS_DIR", models)
    monkeypatch.setattr(
        adapter_lifecycle, "_REGISTRY_PATH", adapters / "registry.json",
    )
    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "state")

    monkeypatch.setattr(adapter_lifecycle, "background_enabled", lambda: True)

    sent: list[str] = []
    monkeypatch.setattr(adapter_lifecycle, "send_signal_alert",
                         lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(adapter_lifecycle, "audit_event", lambda *a, **k: None)

    yield tmp_path, adapters, models, sent


def _write_registry(adapters_dir: Path, registry: dict) -> None:
    (adapters_dir / "registry.json").write_text(json.dumps(registry))


def _make_old_dir(p: Path, age_days: int) -> None:
    p.mkdir(parents=True, exist_ok=True)
    (p / "weights.bin").write_text("x" * 100)
    old = time.time() - age_days * 24 * 3600
    os.utime(p / "weights.bin", (old, old))
    os.utime(p, (old, old))


# ── Orphan cleanup ────────────────────────────────────────────────────────


def test_orphan_old_enough_is_deleted(isolated):
    tmp, adapters, models, _ = isolated
    from app.training import adapter_lifecycle

    # Active adapter referenced by registry.
    active = adapters / "general_specialist"
    _make_old_dir(active, age_days=60)
    _write_registry(adapters, {
        "general_specialist": {
            "name": "general_specialist",
            "adapter_path": str(active),
            "training_run_id": "run-1",
            "eval_score": 0.82,
            "promoted": True,
            "created_at": "2026-01-01T00:00:00Z",
        },
    })

    # Orphan in the same root, 60 days old, NOT in registry.
    orphan = adapters / "stale_orphan"
    _make_old_dir(orphan, age_days=60)

    summary = adapter_lifecycle._do_one_pass()
    assert summary["orphans_examined"] == 1
    assert summary["orphans_deleted"] == 1
    assert active.exists()
    assert not orphan.exists()


def test_orphan_too_young_is_spared(isolated):
    tmp, adapters, models, _ = isolated
    from app.training import adapter_lifecycle

    _write_registry(adapters, {})

    # Young orphan — 5 days old, under the 30-day floor.
    orphan = adapters / "fresh_orphan"
    _make_old_dir(orphan, age_days=5)

    summary = adapter_lifecycle._do_one_pass()
    assert summary["orphans_too_young"] == 1
    assert summary["orphans_deleted"] == 0
    assert orphan.exists()


def test_models_dir_orphans_cleaned(isolated):
    """The same orphan logic applies to MODELS_DIR (fused models)."""
    tmp, adapters, models, _ = isolated
    from app.training import adapter_lifecycle

    _write_registry(adapters, {})

    orphan = models / "old_fused"
    _make_old_dir(orphan, age_days=60)
    summary = adapter_lifecycle._do_one_pass()
    assert summary["orphans_deleted"] == 1
    assert not orphan.exists()


# ── Dead-pointer detection ────────────────────────────────────────────────


def test_dead_pointer_surfaced(isolated):
    tmp, adapters, models, _ = isolated
    from app.training import adapter_lifecycle

    # Registry references a path that doesn't exist.
    _write_registry(adapters, {
        "ghost": {
            "name": "ghost",
            "adapter_path": str(adapters / "missing_path"),
            "training_run_id": "run-2",
            "eval_score": 0.7,
            "promoted": True,
            "created_at": "",
        },
    })

    summary = adapter_lifecycle._do_one_pass()
    assert summary["dead_pointers"] == [{
        "adapter_name": "ghost",
        "missing_path": str(adapters / "missing_path"),
        "training_run_id": "run-2",
    }]


def test_dead_pointer_alerts_via_signal(isolated):
    tmp, adapters, models, sent = isolated
    from app.training import adapter_lifecycle

    _write_registry(adapters, {
        "ghost": {
            "name": "ghost",
            "adapter_path": str(adapters / "missing"),
            "training_run_id": "r",
        },
    })

    # Cadence-bypass: clear last_run_at.
    from app.life_companion._common import write_state_json
    write_state_json(adapter_lifecycle._STATE_FILE, {"last_run_at": 0})

    adapter_lifecycle.run()
    assert any("dead pointers" in s for s in sent)


# ── Bloat detection ───────────────────────────────────────────────────────


def test_bloat_alert_above_threshold(isolated, monkeypatch):
    tmp, adapters, models, sent = isolated
    from app.training import adapter_lifecycle

    # Tiny threshold so the test files trigger it.
    # 0.000001 GB = 1073.74 bytes — write more than that.
    monkeypatch.setattr(adapter_lifecycle, "_BLOAT_THRESHOLD_GB", 0.000001)
    _write_registry(adapters, {})
    big = adapters / "big_orphan"
    big.mkdir()
    (big / "weights.bin").write_text("x" * 4096)  # 4 KB > 1073 byte threshold

    summary = adapter_lifecycle._do_one_pass()
    assert summary["bloat_alert"]
    assert summary["total_bytes"] >= 4096


# ── History trail ─────────────────────────────────────────────────────────


def test_history_snapshot_appended(isolated):
    tmp, adapters, models, _ = isolated
    from app.training import adapter_lifecycle
    from app.life_companion._common import write_state_json, state_path

    _write_registry(adapters, {
        "alpha": {
            "name": "alpha", "adapter_path": str(adapters / "alpha"),
            "training_run_id": "r1", "eval_score": 0.81, "promoted": True,
            "created_at": "2026-04-01T00:00:00Z",
        },
    })
    write_state_json(adapter_lifecycle._STATE_FILE, {"last_run_at": 0})
    adapter_lifecycle.run()

    history_path = state_path(adapter_lifecycle._HISTORY_FILE)
    assert history_path.exists()
    lines = [json.loads(l) for l in history_path.read_text().strip().split("\n") if l]
    assert lines
    assert lines[-1]["registry"]["alpha"]["eval_score"] == 0.81


# ── Cadence + degraded modes ──────────────────────────────────────────────


def test_cadence_skips_under_window(isolated):
    tmp, adapters, models, _ = isolated
    from app.training import adapter_lifecycle
    from app.life_companion._common import write_state_json, state_path

    _write_registry(adapters, {})
    # Set last_run_at to "very recent" — should skip.
    write_state_json(adapter_lifecycle._STATE_FILE,
                     {"last_run_at": time.time()})

    adapter_lifecycle.run()
    # No history written = no real pass happened.
    history = state_path(adapter_lifecycle._HISTORY_FILE)
    assert not history.exists()


def test_missing_registry_is_no_op(isolated):
    tmp, adapters, models, sent = isolated
    from app.training import adapter_lifecycle

    # Registry doesn't exist — pass should run cleanly with empty results.
    summary = adapter_lifecycle._do_one_pass()
    assert summary["registry_size"] == 0
    assert summary["orphans_deleted"] == 0
    assert summary["dead_pointers"] == []


def test_malformed_registry_is_no_op(isolated):
    tmp, adapters, models, _ = isolated
    from app.training import adapter_lifecycle

    (adapters / "registry.json").write_text("{not json")
    summary = adapter_lifecycle._do_one_pass()
    assert summary["registry_size"] == 0
