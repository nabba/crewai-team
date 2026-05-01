"""Tests for app.companion.sources — JSON sidecar persistence."""

from pathlib import Path

import pytest

from app.companion import sources as _s


@pytest.fixture
def tmp_sources_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_s, "_SOURCES_DIR", tmp_path)
    return tmp_path


def test_list_empty_workspace(tmp_sources_dir):
    assert _s.list_sources("never") == []


def test_add_source_round_trip(tmp_sources_dir):
    src = _s.add_source("ws-1", "web_search", {"query": "Estonian forests"})
    assert src is not None
    assert src.source_id.startswith("src_")

    listed = _s.list_sources("ws-1")
    assert len(listed) == 1
    assert listed[0].config == {"query": "Estonian forests"}
    assert listed[0].enabled is True


def test_add_source_rejects_bad_type(tmp_sources_dir):
    src = _s.add_source("ws-1", "telepathy", {"channel": 7})
    assert src is None
    assert _s.list_sources("ws-1") == []


def test_remove_source(tmp_sources_dir):
    a = _s.add_source("ws-1", "web_search", {"query": "x"})
    b = _s.add_source("ws-1", "web_search", {"query": "y"})

    assert _s.remove_source("ws-1", a.source_id) is True
    remaining = _s.list_sources("ws-1")
    assert len(remaining) == 1
    assert remaining[0].source_id == b.source_id


def test_remove_unknown_source_returns_false(tmp_sources_dir):
    assert _s.remove_source("ws-1", "no-such-id") is False


def test_set_enabled_toggles(tmp_sources_dir):
    src = _s.add_source("ws-1", "web_search", {"query": "x"})
    assert _s.set_enabled("ws-1", src.source_id, False) is True
    assert _s.list_sources("ws-1")[0].enabled is False
    assert _s.set_enabled("ws-1", src.source_id, True) is True
    assert _s.list_sources("ws-1")[0].enabled is True


def test_update_ingest_status(tmp_sources_dir):
    src = _s.add_source("ws-1", "web_search", {"query": "x"})
    _s.update_ingest_status("ws-1", src.source_id, ts=12345.0, status="ok")
    listed = _s.list_sources("ws-1")[0]
    assert listed.last_ingested_at == 12345.0
    assert listed.last_ingest_status == "ok"


def test_path_sanitises_workspace_id(tmp_sources_dir):
    _s.add_source("../../etc/passwd", "web_search", {"query": "x"})
    files = list(tmp_sources_dir.iterdir())
    assert len(files) == 1
    assert files[0].parent == tmp_sources_dir


def test_workspaces_isolated(tmp_sources_dir):
    _s.add_source("ws-1", "web_search", {"query": "x"})
    _s.add_source("ws-2", "web_search", {"query": "y"})
    assert len(_s.list_sources("ws-1")) == 1
    assert len(_s.list_sources("ws-2")) == 1
    assert _s.list_sources("ws-1")[0].config == {"query": "x"}
