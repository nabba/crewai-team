"""Tests for app.control_plane.inquiries_api.

Lives under tests/architecture_requests/ for now since it shares the
auth-bypass fixture pattern; the surface itself is in
app/control_plane/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from app.control_plane import auth_dep
    monkeypatch.setattr(auth_dep, "require_gateway_auth", lambda: True)

    inquiries_dir = tmp_path / "wiki" / "self" / "inquiries"
    inquiries_dir.mkdir(parents=True)
    questions_file = tmp_path / "wiki" / "self" / "inquiry_questions.md"
    questions_file.write_text(
        "# Inquiry questions\n\n"
        "## Are the goals coherent?\n\nFraming.\n\n"
        "## Substrate substitution\n\nFraming.\n",
    )

    import app.control_plane.inquiries_api as mod
    monkeypatch.setattr(mod, "_INQUIRIES_DIR", inquiries_dir)
    monkeypatch.setattr(mod, "_QUESTIONS_FILE", questions_file)

    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[auth_dep.require_gateway_auth] = lambda: True

    yield TestClient(app), inquiries_dir


def test_list_empty(client) -> None:
    c, _ = client
    r = c.get("/api/cp/inquiries")
    assert r.status_code == 200
    assert r.json() == {"count": 0, "inquiries": []}


def test_list_returns_summaries_newest_first(client) -> None:
    c, inquiries = client
    (inquiries / "2026-05-08-old.md").write_text(
        "---\nquestion: Old?\n---\n\n# Old?\n\nbody one.\n",
    )
    (inquiries / "2026-05-15-new.md").write_text(
        "---\nquestion: New?\n---\n\n# New?\n\nbody two.\n",
    )
    r = c.get("/api/cp/inquiries")
    body = r.json()
    assert body["count"] == 2
    # Sorted newest first by filename (which starts with date).
    assert body["inquiries"][0]["filename"] == "2026-05-15-new.md"
    assert body["inquiries"][0]["question_text"] == "New?"


def test_questions_endpoint_with_answer_state(client) -> None:
    c, inquiries = client
    (inquiries / "2026-05-15-are-the-goals-coherent.md").write_text(
        "---\nquestion: Are the goals coherent?\n---\n\nbody.\n",
    )
    r = c.get("/api/cp/inquiries/questions")
    body = r.json()
    assert body["count"] == 2
    by_slug = {q["slug"]: q for q in body["questions"]}
    assert by_slug["are-the-goals-coherent"]["most_recent_answer_date"] == "2026-05-15"
    assert by_slug["substrate-substitution"]["most_recent_answer_date"] is None


def test_get_inquiry_detail(client) -> None:
    c, inquiries = client
    (inquiries / "2026-05-15-x.md").write_text(
        "---\nquestion: X?\n---\n\n# X?\n\nbody text here.\n",
    )
    r = c.get("/api/cp/inquiries/2026-05-15-x.md")
    assert r.status_code == 200
    body = r.json()
    assert body["question_text"] == "X?"
    assert "body text here." in body["body"]


def test_get_inquiry_404_unknown(client) -> None:
    c, _ = client
    assert c.get("/api/cp/inquiries/nope.md").status_code == 404


def test_get_inquiry_traversal_safely_refused(client) -> None:
    """Traversal attempts via URL-encoded slashes should never reach a file
    on disk. Whether FastAPI 404s the request before our handler sees it
    or our handler 400s, both are safe rejections."""
    c, _ = client
    for attempt in (
        "/api/cp/inquiries/..%2Fescape",
        "/api/cp/inquiries/sub%2Fdir.md",
    ):
        r = c.get(attempt)
        assert r.status_code in (400, 404), f"{attempt!r}: {r.status_code}"
