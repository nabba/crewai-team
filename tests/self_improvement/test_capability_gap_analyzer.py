"""Tests for app.self_improvement.capability_gap_analyzer."""

from __future__ import annotations

import logging
from pathlib import Path

from app.self_improvement import capability_gap_analyzer as analyzer
from app.self_improvement.types import GapSource, GapStatus, LearningGap


def _gap(
    description: str,
    *,
    source: GapSource = GapSource.RETRIEVAL_MISS,
    detected_at: str = "2026-05-10T00:00:00+00:00",
) -> LearningGap:
    return LearningGap(
        id=f"gap-{abs(hash(description)) % 10**8}",
        source=source,
        description=description,
        detected_at=detected_at,
        status=GapStatus.OPEN,
    )


# ── pure logic ───────────────────────────────────────────────────────────


def test_cluster_groups_similar_descriptions(tmp_path: Path) -> None:
    gaps = [
        _gap("query about Estonian forest cover percentages"),
        _gap("Estonian forest cover percentage by year"),
        _gap("forest cover Estonia time series query"),
        _gap("how to compute compound interest in Python"),
    ]
    clusters = analyzer._cluster_gaps(gaps)
    # The 3 forest gaps should cluster; the python one is a singleton
    # below MIN_CLUSTER_SIZE so it doesn't surface.
    assert len(clusters) == 1
    assert clusters[0].size == 3


def test_singleton_clusters_filtered_below_min_size(tmp_path: Path) -> None:
    # Lexically disjoint phrases — hash-trick gives near-zero cosine,
    # so they don't cluster, and even if they did the size-2 floor
    # filters them out.
    gaps = [
        _gap("Estonian boreal forest aerial imagery"),
        _gap("compound interest formula derivation Python"),
    ]
    clusters = analyzer._cluster_gaps(gaps)
    assert clusters == []


def test_signature_is_stable_for_same_text() -> None:
    s1 = analyzer._signature_for("hello world")
    s2 = analyzer._signature_for("hello world")
    s3 = analyzer._signature_for("hello universe")
    assert s1 == s2
    assert s1 != s3
    assert len(s1) == 12


def test_slug_extracts_clean_path_component() -> None:
    assert analyzer._slug_from_label("Estonia: forest!! data") == "estonia_forest_data"
    assert analyzer._slug_from_label("") == "capability"
    assert len(analyzer._slug_from_label("a" * 100)) <= 30


# ── render ───────────────────────────────────────────────────────────────


def test_render_draft_has_architecture_request_skeleton() -> None:
    cluster = analyzer.CapabilityCluster(
        signature="abc123",
        label="forest cover Estonia",
        size=4,
        sources={"retrieval_miss": 3, "user_correction": 1},
        samples=[
            "Estonian forest cover by year",
            "forest cover Estonia time series",
            "Estonia forest area trend",
        ],
        first_seen="2026-04-01T00:00:00+00:00",
        last_seen="2026-05-10T00:00:00+00:00",
    )
    draft = analyzer._render_draft(cluster)
    assert "Capability gap draft — forest cover Estonia" in draft
    assert "abc123" in draft
    assert "4 evidence item(s)" in draft
    assert "3 retrieval_miss" in draft
    assert '"package_path"' in draft
    assert "POST to `/api/cp/architecture-requests`" in draft


# ── run_one_pass orchestration ───────────────────────────────────────────


def test_run_one_pass_writes_draft_for_clusters(tmp_path: Path) -> None:
    gaps = [
        _gap("Estonian forest cover by year"),
        _gap("forest cover Estonia time series"),
        _gap("Estonia forest area trend"),
    ]
    out = analyzer.run_one_pass(proposed_dir=tmp_path, gaps=gaps)
    assert out["status"] == "ok"
    assert out["drafts_written"] == 1
    assert out["drafts_skipped_dedup"] == 0
    drafts = list(tmp_path.glob("*.md"))
    assert len(drafts) == 1
    text = drafts[0].read_text()
    assert "Capability gap draft" in text


def test_run_one_pass_dedups_by_signature(tmp_path: Path) -> None:
    gaps = [
        _gap("Estonian forest cover by year"),
        _gap("forest cover Estonia time series"),
        _gap("Estonia forest area trend"),
    ]
    first = analyzer.run_one_pass(proposed_dir=tmp_path, gaps=gaps)
    assert first["drafts_written"] == 1
    second = analyzer.run_one_pass(proposed_dir=tmp_path, gaps=gaps)
    # Same cluster signature → re-emit suppressed.
    assert second["drafts_written"] == 0
    assert second["drafts_skipped_dedup"] == 1


def test_run_one_pass_no_evidence(tmp_path: Path) -> None:
    out = analyzer.run_one_pass(proposed_dir=tmp_path, gaps=[])
    assert out["status"] == "no_evidence"
    assert out["drafts_written"] == 0


def test_run_one_pass_no_clusters_below_min(tmp_path: Path) -> None:
    # Three lexically disjoint singletons — none cluster together,
    # all stay below MIN_CLUSTER_SIZE=3.
    gaps = [
        _gap("Tallinn ferry departure schedule"),
        _gap("compound interest derivation formula Python"),
        _gap("kanban board column synchronization webhook"),
    ]
    out = analyzer.run_one_pass(proposed_dir=tmp_path, gaps=gaps)
    assert out["status"] == "no_clusters"
    assert out["drafts_written"] == 0


def test_run_one_pass_disabled_short_circuits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CAPABILITY_GAP_ANALYZER_ENABLED", "false")
    out = analyzer.run_one_pass(proposed_dir=tmp_path, gaps=[_gap("x")])
    assert out["status"] == "disabled"
    assert out["drafts_written"] == 0


def test_run_one_pass_load_failed_when_store_raises(monkeypatch, tmp_path: Path) -> None:
    def boom(**_):
        raise RuntimeError("ChromaDB unavailable")

    monkeypatch.setattr(analyzer, "list_open_gaps", boom)
    out = analyzer.run_one_pass(proposed_dir=tmp_path)
    assert out["status"] == "load_failed"
    assert "ChromaDB" in out["error"]
    assert out["drafts_written"] == 0


# ── multi-source clustering ──────────────────────────────────────────────


def test_clusters_collapse_across_source_kinds(tmp_path: Path) -> None:
    gaps = [
        _gap("query about Tallinn ferry schedule",
             source=GapSource.RETRIEVAL_MISS),
        _gap("Tallinn ferry departure times needed",
             source=GapSource.REFLEXION_FAILURE),
        _gap("ferry Tallinn-Helsinki timetable",
             source=GapSource.USER_CORRECTION),
    ]
    out = analyzer.run_one_pass(proposed_dir=tmp_path, gaps=gaps)
    assert out["drafts_written"] == 1
    drafts = list(tmp_path.glob("*.md"))
    assert len(drafts) == 1
    text = drafts[0].read_text()
    # All three source kinds should be visible in the draft.
    assert "retrieval_miss" in text
    assert "reflexion_failure" in text
    assert "user_correction" in text


# ── daemon discipline ────────────────────────────────────────────────────


def test_disabled_short_circuits_start(monkeypatch, caplog) -> None:
    monkeypatch.setenv("CAPABILITY_GAP_ANALYZER_ENABLED", "false")
    assert analyzer._enabled() is False
    with caplog.at_level(logging.INFO, logger="app.self_improvement.capability_gap_analyzer"):
        analyzer.start()
    assert any("disabled via" in r.message for r in caplog.records)


def test_stop_sets_event() -> None:
    analyzer.stop()
    assert analyzer._stop_event.is_set()


def test_start_is_idempotent(monkeypatch) -> None:
    monkeypatch.setenv("CAPABILITY_GAP_ANALYZER_ENABLED", "true")
    analyzer.stop()
    analyzer._driver_started = False
    analyzer.start()
    n_first = sum(
        1 for t in analyzer.threading.enumerate()
        if t.name == analyzer._DAEMON_THREAD_NAME
    )
    analyzer.start()
    n_second = sum(
        1 for t in analyzer.threading.enumerate()
        if t.name == analyzer._DAEMON_THREAD_NAME
    )
    assert n_first == n_second
    analyzer.stop()
