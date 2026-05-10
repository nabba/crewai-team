"""Tests for app.control_plane.threads_api."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.threads import reset_for_tests


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from app.control_plane import auth_dep
    monkeypatch.setattr(auth_dep, "require_gateway_auth", lambda: True)
    reset_for_tests(tmp_path / "threads")

    from app.control_plane.threads_api import router
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_dep.require_gateway_auth] = lambda: True

    yield TestClient(app)
    reset_for_tests(None)


def test_create_thread(client) -> None:
    r = client.post("/api/cp/threads", json={
        "title": "Investigate forest data gap",
        "description": "ESA dataset missing for 2024",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["thread"]["title"] == "Investigate forest data gap"
    assert body["thread"]["status"] == "open"


def test_create_with_empty_title_returns_422(client) -> None:
    r = client.post("/api/cp/threads", json={"title": ""})
    assert r.status_code == 422  # pydantic min_length


def test_list_threads_default_open_only(client) -> None:
    a = client.post("/api/cp/threads", json={"title": "A"}).json()["thread"]
    b = client.post("/api/cp/threads", json={"title": "B"}).json()["thread"]
    client.post(f"/api/cp/threads/{a['id']}/transition",
                json={"transition": "resolved"})

    open_only = client.get("/api/cp/threads").json()
    everything = client.get("/api/cp/threads?open_only=false").json()
    assert {t["id"] for t in open_only["threads"]} == {b["id"]}
    assert {t["id"] for t in everything["threads"]} == {a["id"], b["id"]}


def test_get_thread_serializes_helpers(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    body = client.get(f"/api/cp/threads/{t['id']}").json()
    assert body["is_terminal"] is False
    assert body["open_subquestion_count"] == 0
    assert body["resolved_subquestion_count"] == 0


def test_unknown_thread_returns_404(client) -> None:
    assert client.get("/api/cp/threads/missing").status_code == 404


def test_add_and_resolve_subquestion(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    after = client.post(
        f"/api/cp/threads/{t['id']}/sub-question",
        json={"text": "what is the threshold?"},
    ).json()["thread"]
    assert after["open_subquestion_count"] == 1
    sq_id = after["sub_questions"][0]["id"]

    resolved = client.post(
        f"/api/cp/threads/{t['id']}/resolve-sq",
        json={"subquestion_id": sq_id, "resolution": "0.40 cosine"},
    ).json()["thread"]
    assert resolved["resolved_subquestion_count"] == 1
    assert resolved["sub_questions"][0]["resolution"] == "0.40 cosine"


def test_resolve_unknown_subquestion_returns_404(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    r = client.post(
        f"/api/cp/threads/{t['id']}/resolve-sq",
        json={"subquestion_id": "nonexistent", "resolution": ""},
    )
    assert r.status_code == 404


def test_blocker_lifecycle(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    after = client.post(
        f"/api/cp/threads/{t['id']}/blocker",
        json={"text": "ESA contact unresponsive"},
    ).json()["thread"]
    assert after["blockers"] == ["ESA contact unresponsive"]

    cleared = client.post(
        f"/api/cp/threads/{t['id']}/clear-blockers",
    ).json()["thread"]
    assert cleared["blockers"] == []


def test_note_appends(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    client.post(f"/api/cp/threads/{t['id']}/note", json={"text": "first"})
    after = client.post(
        f"/api/cp/threads/{t['id']}/note", json={"text": "second"},
    ).json()["thread"]
    assert after["notes"] == ["first", "second"]


def test_link_cr_and_inquiry(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    client.post(
        f"/api/cp/threads/{t['id']}/link-cr",
        json={"crew_task_id": "task-abc"},
    )
    client.post(
        f"/api/cp/threads/{t['id']}/link-inquiry",
        json={"inquiry_slug": "are-the-goals-coherent"},
    )
    body = client.get(f"/api/cp/threads/{t['id']}").json()
    assert body["related_crew_task_ids"] == ["task-abc"]
    assert body["related_inquiry_slugs"] == ["are-the-goals-coherent"]


def test_transition_blocked_then_unblocked(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    after = client.post(
        f"/api/cp/threads/{t['id']}/transition",
        json={"transition": "blocked", "blocker": "waiting"},
    ).json()["thread"]
    assert after["status"] == "blocked"

    after = client.post(
        f"/api/cp/threads/{t['id']}/transition",
        json={"transition": "in_progress"},
    ).json()["thread"]
    assert after["status"] == "in_progress"


def test_transition_resolve(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    after = client.post(
        f"/api/cp/threads/{t['id']}/transition",
        json={"transition": "resolved", "summary": "done"},
    ).json()["thread"]
    assert after["status"] == "resolved"
    assert any("[resolution] done" in n for n in after["notes"])


def test_transition_abandon_requires_reason(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    r = client.post(
        f"/api/cp/threads/{t['id']}/transition",
        json={"transition": "abandoned"},
    )
    assert r.status_code == 400


def test_transition_invalid_value_returns_400(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    r = client.post(
        f"/api/cp/threads/{t['id']}/transition",
        json={"transition": "bogus"},
    )
    assert r.status_code == 400


def test_cannot_modify_terminal_thread(client) -> None:
    t = client.post("/api/cp/threads", json={"title": "x"}).json()["thread"]
    client.post(
        f"/api/cp/threads/{t['id']}/transition",
        json={"transition": "resolved", "summary": "done"},
    )
    r = client.post(
        f"/api/cp/threads/{t['id']}/note", json={"text": "late note"},
    )
    assert r.status_code == 409
