"""Tests for app.companion.ingest — daily source fetcher + indexer."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import ingest as _ing
from app.companion import sources as _s


@pytest.fixture
def tmp_sources_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_s, "_SOURCES_DIR", tmp_path)
    monkeypatch.setattr(_ing, "_index_chromadb",
                         lambda iid, t, m: True)
    return tmp_path


def _make_search_results(*urls):
    return [
        {"title": f"Title {u}", "description": f"Desc {u}", "url": u}
        for u in urls
    ]


def test_ingest_for_workspace_calls_search(tmp_sources_dir):
    _s.add_source("ws-1", "web_search", {"query": "forests"})
    captured = []

    def _fake_search(query, count):
        captured.append((query, count))
        return _make_search_results("https://a", "https://b")

    indexed: list = []
    with patch("app.companion.ingest._invoke_search", _fake_search), \
         patch("app.companion.ingest._index_chromadb",
               lambda iid, t, m: indexed.append((iid, m["url"])) or True):
        n = _ing.ingest_for_workspace("ws-1")

    assert n == 2
    assert captured == [("forests", _ing.WEB_SEARCH_COUNT)]
    urls = sorted([u for _, u in indexed])
    assert urls == ["https://a", "https://b"]


def test_ingest_skips_disabled_source(tmp_sources_dir):
    src = _s.add_source("ws-1", "web_search", {"query": "x"})
    _s.set_enabled("ws-1", src.source_id, False)

    with patch("app.companion.ingest._invoke_search",
               lambda *a, **k: pytest.fail("should not call search")):
        n = _ing.ingest_for_workspace("ws-1")
    assert n == 0


def test_ingest_skips_recently_ingested(tmp_sources_dir):
    src = _s.add_source("ws-1", "web_search", {"query": "x"})
    _s.update_ingest_status("ws-1", src.source_id, ts=time.time())

    with patch("app.companion.ingest._invoke_search",
               lambda *a, **k: pytest.fail("should not call search")):
        n = _ing.ingest_for_workspace("ws-1")
    assert n == 0


def test_ingest_runs_after_cooldown(tmp_sources_dir):
    src = _s.add_source("ws-1", "web_search", {"query": "x"})
    _s.update_ingest_status(
        "ws-1", src.source_id,
        ts=time.time() - _ing.MIN_REINGEST_S - 1000,
    )

    with patch("app.companion.ingest._invoke_search",
               lambda *a, **k: _make_search_results("https://x")):
        n = _ing.ingest_for_workspace("ws-1")
    assert n == 1


def test_fetch_failure_marks_status_error(tmp_sources_dir):
    src = _s.add_source("ws-1", "web_search", {"query": "x"})

    def _broken(*a, **k):
        raise RuntimeError("brave down")

    with patch("app.companion.ingest._invoke_search", _broken):
        _ing.ingest_for_workspace("ws-1")

    listed = _s.list_sources("ws-1")[0]
    assert "error: RuntimeError" in listed.last_ingest_status


def test_index_skips_empty_text(tmp_sources_dir):
    _s.add_source("ws-1", "web_search", {"query": "x"})
    items = [{"title": "", "description": "", "url": "https://empty"}]

    with patch("app.companion.ingest._invoke_search", lambda *a, **k: items):
        n = _ing.ingest_for_workspace("ws-1")
    assert n == 0


def test_unknown_source_type_returns_no_items(tmp_sources_dir):
    src = _s.Source(type="rss", config={"feed": "x"})
    _s._save_all("ws-1", [src])

    n = _ing.ingest_for_workspace("ws-1")
    assert n == 0


def test_run_ingest_handles_project_listing_failure(tmp_sources_dir):
    def _broken():
        raise RuntimeError("DB down")

    with patch("app.companion.ingest._list_projects", _broken):
        _ing.run_ingest()


def test_run_ingest_skips_disabled_workspaces(tmp_sources_dir):
    rows = [
        {"id": "a", "config_json": {"companion": {"enabled": False}}},
        {"id": "b", "config_json": {"companion": {"enabled": True}}},
    ]
    visited = []

    def _fake_for_workspace(ws, **kw):
        visited.append(ws)

    with patch("app.companion.ingest._list_projects", lambda: rows), \
         patch("app.companion.ingest.ingest_for_workspace",
               _fake_for_workspace):
        _ing.run_ingest()

    assert visited == ["b"]


def test_get_idle_jobs_returns_companion_ingest():
    jobs = _ing.get_idle_jobs()
    assert len(jobs) == 1
    name, fn, weight = jobs[0]
    assert name == "companion-ingest"
    assert callable(fn)

    from app.idle_scheduler import JobWeight
    assert weight == JobWeight.LIGHT


def test_stable_id_deterministic():
    a = _ing._stable_id("ws-1", "src-1", "https://x")
    b = _ing._stable_id("ws-1", "src-1", "https://x")
    c = _ing._stable_id("ws-1", "src-1", "https://y")
    assert a == b
    assert a != c
    assert a.startswith("src_")
