"""Tests for app.control_plane.proposals_api."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from app.control_plane import auth_dep
    monkeypatch.setattr(auth_dep, "require_gateway_auth", lambda: True)

    cap_dir = tmp_path / "capabilities"
    lib_dir = tmp_path / "libraries"
    recipe_path = tmp_path / "recipe_proposals.jsonl"
    cap_dir.mkdir()
    lib_dir.mkdir()

    import app.control_plane.proposals_api as mod
    monkeypatch.setattr(mod, "_CAPABILITY_DIR", cap_dir)
    monkeypatch.setattr(mod, "_LIBRARY_DIR", lib_dir)
    monkeypatch.setattr(mod, "_RECIPE_PATH", recipe_path)

    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[auth_dep.require_gateway_auth] = lambda: True

    yield TestClient(app), cap_dir, lib_dir, recipe_path


def _seed_md(d: Path, name: str, title: str) -> None:
    (d / name).write_text(f"# {title}\n\nbody here\n")


def test_empty_listing(client) -> None:
    c, *_ = client
    body = c.get("/api/cp/proposals").json()
    assert body["count"] == 0
    assert body["kinds"] == ["capability", "library", "recipe"]


def test_lists_all_three_kinds(client) -> None:
    c, cap, lib, recipe = client
    _seed_md(cap, "abc123.md", "forest data gap")
    _seed_md(lib, "xyz789-langgraph.md", "LangGraph 0.5")
    recipe.write_text(json.dumps({
        "recipe_id": "r-1", "crew_name": "coder", "health": 0.18,
        "reason": "low winrate", "proposed_at": "2026-05-15T00:00:00+00:00",
    }) + "\n")

    body = c.get("/api/cp/proposals").json()
    assert body["count"] == 3
    kinds = {p["kind"] for p in body["proposals"]}
    assert kinds == {"capability", "library", "recipe"}


def test_filter_by_kind(client) -> None:
    c, cap, lib, recipe = client
    _seed_md(cap, "a.md", "A")
    _seed_md(lib, "b.md", "B")
    recipe.write_text(json.dumps({
        "recipe_id": "r1", "crew_name": "x", "health": 0.1,
        "reason": "x", "proposed_at": "2026-05-15T00:00:00+00:00",
    }) + "\n")

    cap_only = c.get("/api/cp/proposals?kind=capability").json()
    assert {p["kind"] for p in cap_only["proposals"]} == {"capability"}

    lib_only = c.get("/api/cp/proposals?kind=library").json()
    assert {p["kind"] for p in lib_only["proposals"]} == {"library"}

    recipe_only = c.get("/api/cp/proposals?kind=recipe").json()
    assert {p["kind"] for p in recipe_only["proposals"]} == {"recipe"}


def test_invalid_kind_returns_400(client) -> None:
    c, *_ = client
    assert c.get("/api/cp/proposals?kind=bogus").status_code == 400


def test_get_capability_detail(client) -> None:
    c, cap, *_ = client
    _seed_md(cap, "abc.md", "forest gap")
    body = c.get("/api/cp/proposals/capability/abc.md").json()
    assert body["kind"] == "capability"
    assert body["title"] == "forest gap"
    assert "body here" in body["body"]


def test_get_library_detail(client) -> None:
    c, _, lib, _ = client
    _seed_md(lib, "xyz-lg.md", "LangGraph")
    body = c.get("/api/cp/proposals/library/xyz-lg.md").json()
    assert body["kind"] == "library"
    assert body["title"] == "LangGraph"


def test_get_recipe_detail(client) -> None:
    c, _, _, recipe = client
    recipe.write_text(json.dumps({
        "recipe_id": "r-coder-1", "crew_name": "coder",
        "health": 0.18, "reason": "low winrate",
        "proposed_at": "2026-05-15T00:00:00+00:00",
    }) + "\n")
    body = c.get("/api/cp/proposals/recipe/r-coder-1").json()
    assert body["kind"] == "recipe"
    assert body["row"]["health"] == 0.18


def test_unknown_recipe_returns_404(client) -> None:
    c, *_ = client
    assert c.get("/api/cp/proposals/recipe/nonexistent").status_code == 404


def test_unknown_capability_returns_404(client) -> None:
    c, *_ = client
    assert c.get("/api/cp/proposals/capability/missing.md").status_code == 404


def test_traversal_safely_refused(client) -> None:
    c, *_ = client
    for attempt in (
        "/api/cp/proposals/capability/..%2Fescape.md",
        "/api/cp/proposals/library/sub%2Fdir.md",
    ):
        r = c.get(attempt)
        assert r.status_code in (400, 404)


def test_recipe_dedup_keeps_latest_per_id(client) -> None:
    """Multiple JSONL rows for the same recipe — only the most recent
    proposed_at survives in the listing."""
    c, _, _, recipe = client
    recipe.write_text(
        json.dumps({
            "recipe_id": "r-1", "crew_name": "x", "health": 0.30,
            "reason": "first", "proposed_at": "2026-04-01T00:00:00+00:00",
        }) + "\n"
        + json.dumps({
            "recipe_id": "r-1", "crew_name": "x", "health": 0.20,
            "reason": "dropped further", "proposed_at": "2026-05-15T00:00:00+00:00",
        }) + "\n"
    )
    body = c.get("/api/cp/proposals?kind=recipe").json()
    assert body["count"] == 1
    assert body["proposals"][0]["row"]["health"] == 0.20
