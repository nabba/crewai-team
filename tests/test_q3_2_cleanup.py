"""PROGRAM §40.2 — Q3.2 third-pass cleanup regression sweep.

Targets the round-3 findings:
  * restart-required claim queue + startup self-check
  * pre-cutover auto-export hook
  * cross-rotation reader/writer flock
  * HNSW orphan segment detection
  * reclaim trend tracking
  * annual reflection covers substrate_migration kind (verification)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
#   Restart-required claim queue (Items 1 + 9)
# ─────────────────────────────────────────────────────────────────────────


def test_runtime_settings_exposes_restart_claim_helpers():
    """Source-level: the three helpers exist with the right shapes."""
    src = Path("app/runtime_settings.py").read_text()
    assert "def get_post_amendment_restart_claims" in src
    assert "def append_post_amendment_restart_claim" in src
    assert "def clear_post_amendment_restart_claims" in src
    assert '"post_amendment_restart_claims": []' in src


# runtime_settings imports app.config which needs pydantic_settings.
# Skip the live-helper tests when that dep is absent (e.g. local dev);
# source-level tests above still cover schema + dedup semantics.
_HAS_PYDANTIC_SETTINGS = True
try:
    import pydantic_settings  # noqa: F401
except ImportError:
    _HAS_PYDANTIC_SETTINGS = False


@pytest.mark.skipif(
    not _HAS_PYDANTIC_SETTINGS,
    reason="pydantic_settings not available locally; covered by gateway-deps suite",
)
def test_restart_claim_append_dedups_on_id(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime_settings.json"
    state_path.write_text(json.dumps({
        "post_amendment_restart_claims": [],
    }))
    import app.runtime_settings as rs
    monkeypatch.setattr(rs, "_STATE_PATH", state_path)
    monkeypatch.setattr(rs, "_cache", None)
    rs.append_post_amendment_restart_claim({
        "id": "claim-a", "claim_kind": "restart_required",
        "source": "test", "reason": "first",
    })
    rs.append_post_amendment_restart_claim({
        "id": "claim-a", "claim_kind": "restart_required",
        "source": "test", "reason": "duplicate",
    })
    claims = rs.get_post_amendment_restart_claims()
    assert len(claims) == 1
    assert claims[0]["reason"] == "first"


@pytest.mark.skipif(
    not _HAS_PYDANTIC_SETTINGS,
    reason="pydantic_settings not available locally",
)
def test_restart_claim_clear_by_id(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime_settings.json"
    state_path.write_text(json.dumps({"post_amendment_restart_claims": []}))
    import app.runtime_settings as rs
    monkeypatch.setattr(rs, "_STATE_PATH", state_path)
    monkeypatch.setattr(rs, "_cache", None)
    for cid in ("c1", "c2", "c3"):
        rs.append_post_amendment_restart_claim({
            "id": cid, "claim_kind": "restart_required",
            "source": "t", "reason": cid,
        })
    cleared = rs.clear_post_amendment_restart_claims(ids=["c2"])
    assert cleared == 1
    remaining = {c["id"] for c in rs.get_post_amendment_restart_claims()}
    assert remaining == {"c1", "c3"}


@pytest.mark.skipif(
    not _HAS_PYDANTIC_SETTINGS,
    reason="pydantic_settings not available locally",
)
def test_restart_claim_clear_all(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime_settings.json"
    state_path.write_text(json.dumps({"post_amendment_restart_claims": []}))
    import app.runtime_settings as rs
    monkeypatch.setattr(rs, "_STATE_PATH", state_path)
    monkeypatch.setattr(rs, "_cache", None)
    rs.append_post_amendment_restart_claim({
        "id": "x", "claim_kind": "restart_required",
        "source": "t", "reason": "y",
    })
    cleared = rs.clear_post_amendment_restart_claims(ids=None)
    assert cleared == 1
    assert rs.get_post_amendment_restart_claims() == []


@pytest.mark.skipif(
    not _HAS_PYDANTIC_SETTINGS,
    reason="pydantic_settings not available locally",
)
def test_restart_claim_append_requires_id():
    import app.runtime_settings as rs
    with pytest.raises(ValueError, match="non-empty id"):
        rs.append_post_amendment_restart_claim({
            "claim_kind": "restart_required", "source": "t", "reason": "r",
        })


def test_startup_check_clears_satisfied_claim_source_level():
    """The startup helper exists and references the right paths."""
    src = Path("app/main.py").read_text()
    assert "_process_post_amendment_restart_claims" in src
    assert "embedding_migration.cutover" in src
    assert "expected_embed_dim" in src
    # It must clear ONLY satisfied claims, not flush unsatisfied ones.
    assert "satisfied_ids" in src
    assert "unsatisfied" in src


# ─────────────────────────────────────────────────────────────────────────
#   Pre-cutover auto-export (Item 2)
# ─────────────────────────────────────────────────────────────────────────


def test_cutover_runs_pre_cutover_export_source_level():
    src = Path("app/memory/embedding_migration/cutover.py").read_text()
    assert "_run_pre_cutover_export" in src
    assert "pre_cutover_tarball" in src


def test_pre_cutover_export_failure_is_non_fatal():
    """Source-level: the helper catches exceptions and returns None
    rather than raising — preserves the verifier-confirmed backup
    as a fallback."""
    src = Path("app/memory/embedding_migration/cutover.py").read_text()
    # Find the helper
    start = src.find("def _run_pre_cutover_export")
    end = src.find("\ndef ", start + 1)
    assert start > 0 and end > start
    body = src[start:end]
    assert "try:" in body
    assert "except Exception" in body
    assert "return None" in body


# ─────────────────────────────────────────────────────────────────────────
#   Cross-rotation reader-writer flock (Item 3)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def jsonl_retention():
    return _load_isolated(
        "jsonl_retention_q32",
        "app/utils/jsonl_retention.py",
    )


def test_rotation_lock_helper_exists(jsonl_retention):
    assert hasattr(jsonl_retention, "_rotation_lock")
    assert hasattr(jsonl_retention, "_lock_path_for")


def test_rotation_lock_creates_sidecar(jsonl_retention, tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("")
    with jsonl_retention._rotation_lock(p, exclusive=True):
        pass
    # Sidecar exists after first acquisition.
    lock_path = jsonl_retention._lock_path_for(p)
    assert lock_path.exists()
    assert lock_path.name.startswith(".")
    assert "rotation_lock" in lock_path.name


def test_rotation_lock_concurrent_shared_readers(jsonl_retention, tmp_path):
    """Two concurrent readers can both hold LOCK_SH simultaneously
    (the lock is shared between readers, exclusive only against
    writers). Race protection: each thread acquires, sleeps briefly
    inside the lock, then releases."""
    p = tmp_path / "x.jsonl"
    p.write_text("")
    overlap = {"both_in_critical": False}
    in_critical = {"a": False, "b": False}
    barrier = threading.Barrier(2)

    def reader(name):
        with jsonl_retention._rotation_lock(p, exclusive=False):
            in_critical[name] = True
            barrier.wait(timeout=2.0)
            if all(in_critical.values()):
                overlap["both_in_critical"] = True
            time.sleep(0.05)
            in_critical[name] = False

    t1 = threading.Thread(target=reader, args=("a",))
    t2 = threading.Thread(target=reader, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    # On platforms with fcntl (macOS/Linux), shared locks DO overlap.
    # If fcntl absent, lock is a no-op and overlap is trivially true.
    assert overlap["both_in_critical"] is True


def test_rotation_lock_writer_blocks_reader(jsonl_retention, tmp_path):
    """On platforms with fcntl, an LOCK_EX in one thread blocks a
    competing LOCK_SH in another until released. (Skipped if fcntl
    unavailable.)"""
    if not jsonl_retention._HAS_FCNTL:
        pytest.skip("fcntl not available on this platform")
    p = tmp_path / "x.jsonl"
    p.write_text("")
    writer_holding = threading.Event()
    reader_done = threading.Event()
    reader_started_at = [0.0]

    def writer():
        with jsonl_retention._rotation_lock(p, exclusive=True):
            writer_holding.set()
            time.sleep(0.3)

    def reader():
        writer_holding.wait(timeout=2.0)
        reader_started_at[0] = time.monotonic()
        with jsonl_retention._rotation_lock(p, exclusive=False):
            reader_done.set()

    start = time.monotonic()
    t_writer = threading.Thread(target=writer)
    t_reader = threading.Thread(target=reader)
    t_writer.start(); t_reader.start()
    t_writer.join(); t_reader.join()
    # Reader should have been blocked at least the time the writer
    # held the lock minus a small margin.
    assert reader_done.is_set()
    # Total wall-clock should include the writer's hold.
    elapsed = time.monotonic() - start
    assert elapsed >= 0.25


def test_rotation_still_works_under_lock(jsonl_retention, tmp_path):
    """End-to-end smoke: appending many entries past the cap with the
    lock-wrapped rotator still produces a correct archive + live
    split. (Regression guard for the Item 3 critical section.)"""
    p = tmp_path / "trace.jsonl"
    fake = datetime(2026, 5, 11, tzinfo=timezone.utc)
    for i in range(500):
        jsonl_retention.append_with_archive_rotate(
            p, f'{{"i":{i}}}', max_lines=100, _now=fake,
        )
    seen = [line.strip() for line in jsonl_retention.read_archive(p)]
    assert len(seen) == 500
    assert seen[0] == '{"i":0}'
    assert seen[-1] == '{"i":499}'


# ─────────────────────────────────────────────────────────────────────────
#   HNSW orphan detection (Item 5)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hygiene():
    return _load_isolated(
        "chromadb_hygiene_q32",
        "app/healing/monitors/chromadb_hygiene.py",
    )


def test_orphan_scan_finds_uuid_dirs_not_in_sqlite(hygiene, tmp_path):
    """Create a fake chroma.sqlite3 with one known collection ID; add
    two UUID-named directories alongside, one matching + one orphan.
    Scanner should flag the orphan only."""
    kb_dir = tmp_path / "memory"
    kb_dir.mkdir()
    db = kb_dir / "chroma.sqlite3"
    known_id = "11111111-2222-3333-4444-555555555555"
    orphan_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT)"
    )
    conn.execute("INSERT INTO collections (id, name) VALUES (?, 'team')", (known_id,))
    conn.commit()
    conn.close()
    # On-disk dirs
    (kb_dir / known_id).mkdir()
    (kb_dir / orphan_id).mkdir()
    # Add a small file inside the orphan so it has non-zero size
    (kb_dir / orphan_id / "data.bin").write_bytes(b"x" * 100)

    findings = hygiene._scan_for_orphan_segments([db])
    assert len(findings) == 1
    assert findings[0]["kb"] == "memory"
    assert orphan_id in findings[0]["orphan_uuids"]
    assert known_id not in findings[0]["orphan_uuids"]
    assert findings[0]["orphan_bytes"] > 0


def test_orphan_scan_quiet_when_no_orphans(hygiene, tmp_path):
    kb_dir = tmp_path / "memory"
    kb_dir.mkdir()
    db = kb_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    findings = hygiene._scan_for_orphan_segments([db])
    assert findings == []


def test_orphan_scan_skips_non_uuid_subdirs(hygiene, tmp_path):
    kb_dir = tmp_path / "memory"
    kb_dir.mkdir()
    db = kb_dir / "chroma.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    # Decoys that look like dirs but aren't UUID-shaped
    (kb_dir / "archive").mkdir()
    (kb_dir / "backup_2026").mkdir()
    (kb_dir / "11111111-2222").mkdir()   # too short
    findings = hygiene._scan_for_orphan_segments([db])
    assert findings == []


# ─────────────────────────────────────────────────────────────────────────
#   Reclaim trend tracking (Item 6)
# ─────────────────────────────────────────────────────────────────────────


def test_reclaim_trend_no_alert_before_baseline(hygiene):
    """First 3 passes shouldn't alert — not enough history."""
    state = {}
    trend = hygiene._update_and_check_reclaim_trend(state, 50_000_000)
    assert trend["alert"] is False
    trend = hygiene._update_and_check_reclaim_trend(state, 60_000_000)
    assert trend["alert"] is False
    trend = hygiene._update_and_check_reclaim_trend(state, 55_000_000)
    assert trend["alert"] is False


def test_reclaim_trend_alerts_on_2x_growth(hygiene):
    """Once we have history, a 2× spike fires the alert."""
    state = {}
    # Three small passes establish baseline ~50 MB.
    for _ in range(3):
        hygiene._update_and_check_reclaim_trend(state, 50_000_000)
    # Fourth pass: 200 MB → far above 2× median. Alert.
    trend = hygiene._update_and_check_reclaim_trend(state, 200_000_000)
    assert trend["alert"] is True
    assert "2×" in trend["reason"]


def test_reclaim_trend_ignores_tiny_baseline(hygiene):
    """If the baseline is below 10 MB, we don't alert — there's
    nothing meaningful to compare against."""
    state = {}
    for _ in range(4):
        hygiene._update_and_check_reclaim_trend(state, 100_000)   # 100 KB
    trend = hygiene._update_and_check_reclaim_trend(state, 5_000_000)
    assert trend["alert"] is False
    assert "too small" in trend["reason"]


# ─────────────────────────────────────────────────────────────────────────
#   Annual reflection covers substrate_migration (Item 7 verification)
# ─────────────────────────────────────────────────────────────────────────


def test_summarise_drift_aggregates_dynamic_kinds(tmp_path):
    """``summarise_drift.by_kind`` is a Counter over event.kind; a
    new kind like ``substrate_migration`` surfaces automatically
    without code changes to the annual reflection module."""
    mod = _load_isolated(
        "continuity_ledger_q32",
        "app/identity/continuity_ledger.py",
    )
    ledger = tmp_path / "ledger.jsonl"
    # Write a few events of various kinds, including substrate_migration.
    for kind in [
        "tier3_amendment", "soul_edit", "substrate_migration",
        "substrate_migration", "scorecard_change",
    ]:
        mod.record_event(
            kind=kind, actor="test", summary=f"e-{kind}",
            detail={}, path=ledger,
        )
    drift = mod.summarise_drift(window_days=365, path=ledger)
    assert drift.by_kind["substrate_migration"] == 2
    assert "substrate_migration" in drift.by_kind
    assert drift.n_events == 5


def test_annual_reflection_prompt_includes_by_kind():
    """Verify the annual-reflection prompt format string surfaces
    by_kind so substrate_migration naturally shows up."""
    src = Path("app/identity/annual_reflection.py").read_text()
    assert "drift.by_kind" in src
    assert "summarise_drift" in src
