"""PROGRAM §40 — Q3 Multi-year hygiene test sweep.

Items covered:
  10. ChromaDB hygiene (SQLite VACUUM monitor + rebuild helper)
  11. JSONL caps + archive rotation
  12. Embedding migration framework (plan, state, verify)
  13. DR drill (export/import roundtrip)
  14. Cost trend math (OLS forecast + anomaly detection)

Tests run pure-Python where possible. Anything requiring chromadb /
psycopg2 / FastAPI is gated behind import-availability skips so the
suite still runs in a stripped local environment.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ── Helper: import a module by file path without going through its
#    package __init__. Used to bypass missing local deps.
def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
#   Item 11 — JSONL retention (archive rotation)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def jsonl_retention():
    return _load_isolated(
        "jsonl_retention_q3",
        "app/utils/jsonl_retention.py",
    )


def test_archive_rotate_preserves_all_lines(jsonl_retention, tmp_path):
    """Many appends past the cap end up split between live + archive,
    but every original line is recoverable in chronological order."""
    p = tmp_path / "trace.jsonl"
    fake_now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    for i in range(500):
        jsonl_retention.append_with_archive_rotate(
            p, f'{{"i":{i}}}', max_lines=100, _now=fake_now,
        )
    live = p.read_text().splitlines()
    assert len(live) <= 100, f"live file kept {len(live)} > 100"
    assert (tmp_path / "archive").exists()
    seen = [r.strip() for r in jsonl_retention.read_archive(p)]
    assert len(seen) == 500
    assert seen[0] == '{"i":0}'
    assert seen[-1] == '{"i":499}'


def test_archive_rotate_uses_monthly_filenames(jsonl_retention, tmp_path):
    p = tmp_path / "salience.jsonl"
    may = datetime(2026, 5, 11, tzinfo=timezone.utc)
    jun = datetime(2026, 6, 1, tzinfo=timezone.utc)
    # 200 lines in May, 200 lines in June, cap=100 each → multiple
    # rotations, but only TWO monthly archive files.
    for i in range(200):
        jsonl_retention.append_with_archive_rotate(
            p, f'{{"i":{i},"m":"may"}}', max_lines=100, _now=may,
        )
    for i in range(200, 400):
        jsonl_retention.append_with_archive_rotate(
            p, f'{{"i":{i},"m":"jun"}}', max_lines=100, _now=jun,
        )
    archive_dir = tmp_path / "archive"
    archives = sorted(archive_dir.iterdir())
    assert len(archives) == 2
    assert archives[0].name.startswith("2026-05_")
    assert archives[1].name.startswith("2026-06_")


def test_archive_stats_summary(jsonl_retention, tmp_path):
    p = tmp_path / "x.jsonl"
    fake = datetime(2026, 5, 11, tzinfo=timezone.utc)
    for i in range(150):
        jsonl_retention.append_with_archive_rotate(
            p, f'{{"i":{i}}}', max_lines=50, _now=fake,
        )
    stats = jsonl_retention.archive_stats(p)
    assert stats["archive_files"] >= 1
    assert stats["oldest_month"] == "2026-05"
    assert stats["archived_lines"] > 0


def test_cap_jsonl_is_idempotent(jsonl_retention, tmp_path):
    p = tmp_path / "ops.jsonl"
    for i in range(20):
        jsonl_retention.append_with_cap(
            p, f'{{"i":{i}}}', max_lines=10,
        )
    keep1 = p.read_text().splitlines()
    # Second pass should be a no-op.
    dropped = jsonl_retention.cap_jsonl(p, max_lines=10)
    assert dropped == 0
    keep2 = p.read_text().splitlines()
    assert keep1 == keep2


# ─────────────────────────────────────────────────────────────────────────
#   Item 14 — Cost trend math
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cost_trends():
    return _load_isolated(
        "cost_trends_q3",
        "app/control_plane/cost_trends.py",
    )


def test_ols_perfect_linear_fit_has_zero_sigma(cost_trends):
    slope, intercept, sigma = cost_trends._ols_fit(
        [0.0, 1, 2, 3, 4], [10.0, 20, 30, 40, 50],
    )
    assert abs(slope - 10.0) < 1e-9
    assert abs(intercept - 10.0) < 1e-9
    assert sigma < 1e-9


def test_ols_noisy_fit_has_positive_sigma(cost_trends):
    slope, intercept, sigma = cost_trends._ols_fit(
        [0.0, 1, 2, 3, 4, 5], [10.0, 21, 29, 41, 49, 60],
    )
    assert 9.5 < slope < 10.5
    assert sigma > 0.0


def test_month_label_rollover(cost_trends):
    assert cost_trends._next_month_label("2026-05") == "2026-06"
    assert cost_trends._next_month_label("2026-12") == "2027-01"
    assert cost_trends._next_month_label("2099-11") == "2099-12"


def test_forecast_perfect_linear_collapses_ci(cost_trends):
    monthly = [
        {"month": f"2026-{i+1:02d}", "total_cost_usd": 10*i + 10,
         "total_tokens": 0, "call_count": 0}
        for i in range(5)
    ]
    fc = cost_trends._build_forecast(monthly, horizon=6)
    assert len(fc) == 6
    assert abs(fc[0]["projected_usd"] - 60.0) < 1e-6
    assert abs(fc[5]["projected_usd"] - 110.0) < 1e-6
    # Perfect fit → CI is zero-width
    assert fc[0]["ci_low"] == fc[0]["ci_high"] == 60.0


def test_forecast_clamps_ci_to_zero(cost_trends):
    """Forecast is clamped to ≥0 — costs can't go negative even if the
    projected line crosses zero."""
    monthly = [
        {"month": "2026-01", "total_cost_usd": 10.0, "total_tokens": 0, "call_count": 0},
        {"month": "2026-02", "total_cost_usd": 5.0, "total_tokens": 0, "call_count": 0},
        {"month": "2026-03", "total_cost_usd": 0.5, "total_tokens": 0, "call_count": 0},
    ]
    fc = cost_trends._build_forecast(monthly, horizon=6)
    for entry in fc:
        assert entry["projected_usd"] >= 0.0
        assert entry["ci_low"] >= 0.0


def test_anomaly_detection_flags_spike(cost_trends):
    import random
    random.seed(0)
    daily = [
        {"day": f"2026-04-{i+1:02d}", "total_cost_usd": 0.10 + random.uniform(0, 0.02)}
        for i in range(30)
    ]
    daily.append({"day": "2026-05-01", "total_cost_usd": 5.0})
    flagged = cost_trends._detect_anomalies(
        daily, window=30, z_threshold=3.0,
    )
    assert len(flagged) == 1
    assert flagged[0]["kind"] == "spike"
    assert flagged[0]["z_score"] > 3.0


def test_anomaly_detection_handles_zero_variance(cost_trends):
    """30 identical days followed by another identical day — stdev=0
    means we don't z-score, so no anomaly is flagged."""
    daily = [
        {"day": f"2026-04-{i+1:02d}", "total_cost_usd": 1.0}
        for i in range(30)
    ]
    daily.append({"day": "2026-05-01", "total_cost_usd": 1.0})
    flagged = cost_trends._detect_anomalies(daily, window=30)
    assert flagged == []


def test_summary_reports_growth_rate(cost_trends):
    monthly = [
        {"month": f"2026-{i+1:02d}", "total_cost_usd": 10 * (1.10 ** i),
         "total_tokens": 0, "call_count": 0}
        for i in range(6)
    ]
    fc = cost_trends._build_forecast(monthly, horizon=6)
    s = cost_trends._build_summary(monthly, fc)
    assert s["history_months_observed"] == 6
    assert s["trend_pct_per_month"] is not None
    # 10% compounded → trend close to +10%/month
    assert 5.0 < s["trend_pct_per_month"] < 15.0


# ─────────────────────────────────────────────────────────────────────────
#   Item 10 — ChromaDB hygiene + rebuild
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def chromadb_hygiene():
    return _load_isolated(
        "chromadb_hygiene_q3",
        "app/healing/monitors/chromadb_hygiene.py",
    )


def test_find_chroma_sqlites_skips_corrupt_dir(chromadb_hygiene, tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "chroma.sqlite3").touch()
    (tmp_path / "philosophy").mkdir()
    (tmp_path / "philosophy" / "chroma.sqlite3").touch()
    # Should be skipped:
    (tmp_path / "memory.corrupt_2026").mkdir()
    (tmp_path / "memory.corrupt_2026" / "chroma.sqlite3").touch()
    (tmp_path / "memory.bak_old").mkdir()
    (tmp_path / "memory.bak_old" / "chroma.sqlite3").touch()
    found = chromadb_hygiene._find_chroma_sqlites(tmp_path)
    names = sorted(p.parent.name for p in found)
    assert names == ["memory", "philosophy"]


def test_vacuum_one_returns_summary_on_real_sqlite(chromadb_hygiene, tmp_path):
    """Run VACUUM on a freshly-written SQLite file; we don't care about
    free bytes, just that the helper returns a sane summary dict."""
    import sqlite3
    db = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, v BLOB); "
        "INSERT INTO x (v) VALUES (randomblob(1024)); "
        "DELETE FROM x;"
    )
    conn.commit()
    conn.close()
    summary = chromadb_hygiene._vacuum_one(db)
    assert summary["ok"] is True
    assert summary["bytes_before"] >= 0
    assert summary["bytes_after"] >= 0
    assert "duration_s" in summary


# ─────────────────────────────────────────────────────────────────────────
#   Item 13 — DR portable export + import
# ─────────────────────────────────────────────────────────────────────────


def test_dr_export_secret_path_filter():
    """The denylist regex catches every tarball-poisoning candidate."""
    mod = _load_isolated(
        "dr_export_q3",
        "app/dr/export_kbs.py",
    )
    assert mod._is_secret_path(".env") is True
    assert mod._is_secret_path(".env.production") is True
    assert mod._is_secret_path("secrets/foo.json") is True
    assert mod._is_secret_path("workspace/secrets/x") is True
    assert mod._is_secret_path("google_token.json") is True
    assert mod._is_secret_path("vapid_private.pem") is True
    assert mod._is_secret_path("a/b/credentials.txt") is True
    # Allowed:
    assert mod._is_secret_path("affect/trace.jsonl") is False
    assert mod._is_secret_path("identity/continuity_ledger.jsonl") is False


# ─────────────────────────────────────────────────────────────────────────
#   Item 12 — Embedding-migration plan + state machine
# ─────────────────────────────────────────────────────────────────────────


def test_migration_plan_roundtrip(tmp_path, monkeypatch):
    plan_mod = _load_isolated(
        "embedding_migration_plan_q3",
        "app/memory/embedding_migration/plan.py",
    )
    # Q3.1: plan paths are lazy-resolved via _default_plan_file(); patch
    # the resolver so save_plan + load_plan use tmp_path.
    monkeypatch.setattr(
        plan_mod, "_default_plan_file",
        lambda: tmp_path / "plan.json",
    )
    plan = plan_mod.MigrationPlan(
        plan_id="test-plan-1",
        source=plan_mod.EmbeddingModel(provider="ollama", name="nomic", dim=768),
        target=plan_mod.EmbeddingModel(provider="ollama", name="mxbai", dim=1024),
        targets=[
            plan_mod.MigrationTarget(
                kind="chromadb", kb="memory", collection="team_shared",
            ),
        ],
    )
    plan_mod.save_plan(plan)
    loaded = plan_mod.load_plan()
    assert loaded is not None
    assert loaded.plan_id == "test-plan-1"
    assert loaded.source.dim == 768
    assert loaded.target.dim == 1024
    assert loaded.targets[0].collection == "team_shared"


def test_migration_state_invalid_transition_rejected():
    """An IDLE → CUTOVER jump must raise."""
    state_mod = _load_isolated(
        "embedding_migration_state_q3",
        "app/memory/embedding_migration/state.py",
    )
    # Stub out runtime_settings reads/writes so we don't touch the live
    # workspace.
    blob = {}

    def _read():
        return dict(blob)

    def _write(d):
        blob.clear()
        blob.update(d)

    state_mod._read_raw = _read
    state_mod._write_raw = _write
    # Reset to IDLE.
    blob.update({"phase": state_mod.PHASE_IDLE})

    with pytest.raises(state_mod.MigrationStateError):
        state_mod.transition(state_mod.PHASE_CUTOVER, reason="test")


def test_migration_state_valid_progression():
    state_mod = _load_isolated(
        "embedding_migration_state_q3b",
        "app/memory/embedding_migration/state.py",
    )
    blob = {}

    def _read():
        return dict(blob)

    def _write(d):
        blob.clear()
        blob.update(d)

    state_mod._read_raw = _read
    state_mod._write_raw = _write

    state_mod.adopt_plan("plan-x")
    assert state_mod.get_state().phase == state_mod.PHASE_PLANNED
    state_mod.transition(state_mod.PHASE_DUAL_WRITE, "advance")
    state_mod.transition(state_mod.PHASE_BACKFILLING, "advance")
    state_mod.transition(state_mod.PHASE_SHADOW_READ, "advance")
    state_mod.transition(state_mod.PHASE_READY, "advance")
    assert state_mod.get_state().phase == state_mod.PHASE_READY
    # Abort from any phase.
    state_mod.abort("done")
    assert state_mod.get_state().phase == state_mod.PHASE_ABORTED


def test_migration_counters_increment_independently():
    state_mod = _load_isolated(
        "embedding_migration_state_q3c",
        "app/memory/embedding_migration/state.py",
    )
    blob = {}

    def _read():
        return dict(blob)

    def _write(d):
        blob.clear()
        blob.update(d)

    state_mod._read_raw = _read
    state_mod._write_raw = _write

    state_mod.increment_shadow_write(5)
    state_mod.increment_backfill(10)
    state_mod.record_shadow_query(0.97, window_size=42)
    s = state_mod.get_state()
    assert s.counters.shadow_writes == 5
    assert s.counters.backfill_rows == 10
    assert s.counters.shadow_query_count == 1
    assert s.counters.last_ndcg_at_10 == pytest.approx(0.97)
    assert s.counters.last_ndcg_window_size == 42


def test_ndcg_at_10_perfect_match():
    """Identical top-10 → NDCG = 1.0."""
    sr_mod = _load_isolated(
        "embedding_migration_shadow_read_q3",
        "app/memory/embedding_migration/shadow_read.py",
    )
    ids = [f"d{i}" for i in range(10)]
    assert sr_mod._ndcg_at_10(ids, ids) == pytest.approx(1.0)


def test_ndcg_at_10_zero_overlap():
    sr_mod = _load_isolated(
        "embedding_migration_shadow_read_q3b",
        "app/memory/embedding_migration/shadow_read.py",
    )
    ideal = [f"d{i}" for i in range(10)]
    observed = [f"e{i}" for i in range(10)]
    assert sr_mod._ndcg_at_10(ideal, observed) == 0.0


def test_ndcg_at_10_partial_overlap():
    """Half overlap with relevant items at the bottom of observed →
    NDCG strictly < 1 (the relevant items lose DCG weight by being
    ranked late)."""
    sr_mod = _load_isolated(
        "embedding_migration_shadow_read_q3c",
        "app/memory/embedding_migration/shadow_read.py",
    )
    # Ideal: top-5 are d0..d4. Observed has 5 distractors first, then d0..d4.
    ideal = [f"d{i}" for i in range(5)]
    observed = [f"e{i}" for i in range(5)] + ideal
    score = sr_mod._ndcg_at_10(ideal, observed)
    # Some relevant items are present but ranked late → score in (0, 1).
    assert 0.0 < score < 1.0
