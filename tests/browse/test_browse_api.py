"""Tests for app.control_plane.browse_api."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(_reset_browse_state: Path, monkeypatch):
    from app.control_plane import auth_dep, browse_api
    monkeypatch.setattr(auth_dep, "require_gateway_auth", lambda: True)
    app = FastAPI()
    app.include_router(browse_api.router)
    app.dependency_overrides[auth_dep.require_gateway_auth] = lambda: True
    return TestClient(app), _reset_browse_state


def _seed_event(base: Path, *, domain: str, title: str,
                 ts: str = "2026-05-15T10:00:00+00:00") -> None:
    """Append one event directly to the store via the public API."""
    from app.browse import store
    from app.browse.models import BrowseEvent
    store.append_events([BrowseEvent(
        visit_ts=ts, domain=domain, path="/",
        title=title, browser="chrome", profile=None,
    )])


def _seed_topic_file(base: Path, day: date, topics: list[tuple[str, int]]) -> None:
    out_dir = base / "topics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{day.isoformat()}.json").write_text(json.dumps({
        "day": day.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "test",
        "topics": [
            {"label": label, "title_count": count, "sample_titles": ["sample"]}
            for label, count in topics
        ],
    }), encoding="utf-8")


# ── /state ────────────────────────────────────────────────────────────


def test_state_reports_enabled(client) -> None:
    c, base = client
    _seed_event(base, domain="github.com", title="GH")
    r = c.get("/api/cp/browse/state")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["stats"]["total"] == 1


def test_state_reports_disabled(
    client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    c, _ = client
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "false")
    body = c.get("/api/cp/browse/state").json()
    assert body["enabled"] is False


# ── /recent ───────────────────────────────────────────────────────────


def test_recent_returns_events_newest_first(client) -> None:
    c, base = client
    _seed_event(base, domain="old.com", title="Old", ts="2026-05-14T08:00:00+00:00")
    _seed_event(base, domain="new.com", title="New", ts="2026-05-15T08:00:00+00:00")
    body = c.get("/api/cp/browse/recent?days=3").json()
    assert body["count"] == 2
    assert body["events"][0]["domain"] == "new.com"


def test_recent_respects_limit(client) -> None:
    c, base = client
    for i in range(10):
        _seed_event(base, domain=f"d{i}.com", title=f"T{i}")
    body = c.get("/api/cp/browse/recent?days=3&limit=5").json()
    assert body["count"] == 10
    assert len(body["events"]) == 5
    assert body["truncated"] is True


# ── /categories ───────────────────────────────────────────────────────


def test_categories_aggregates_across_days(client) -> None:
    c, base = client
    today = datetime.now(timezone.utc).date()
    _seed_topic_file(base, today, [("claude code", 5), ("misc", 1)])
    _seed_topic_file(base, today.replace(day=today.day),
                     [("claude code", 3), ("finnish nature", 2)])
    body = c.get("/api/cp/browse/categories?days=7").json()
    labels = [r["label"] for r in body["categories"]]
    assert "claude code" in labels


def test_categories_empty_when_no_files(client) -> None:
    c, _ = client
    body = c.get("/api/cp/browse/categories").json()
    assert body["categories"] == []
    assert body["days_with_data"] == []


# ── /blocklist + /mute ────────────────────────────────────────────────


def test_blocklist_lists_seeded_and_operator(client) -> None:
    c, _ = client
    body = c.get("/api/cp/browse/blocklist").json()
    assert "paypal.com" in body["seeded"]
    assert body["operator"] == []


def test_mute_adds_domain(client) -> None:
    c, _ = client
    r = c.post("/api/cp/browse/mute", json={"domain": "custom.example"})
    assert r.status_code == 200
    assert r.json()["added"] is True
    body = c.get("/api/cp/browse/blocklist").json()
    assert "custom.example" in body["operator"]


def test_mute_idempotent(client) -> None:
    c, _ = client
    c.post("/api/cp/browse/mute", json={"domain": "x.example"})
    r = c.post("/api/cp/browse/mute", json={"domain": "x.example"})
    assert r.json()["added"] is False


# ── /forget ───────────────────────────────────────────────────────────


def test_forget_all(client) -> None:
    c, base = client
    _seed_event(base, domain="x.com", title="X")
    r = c.post("/api/cp/browse/forget", json={"scope": "all"})
    assert r.status_code == 200
    body = c.get("/api/cp/browse/state").json()
    assert body["stats"]["total"] == 0


def test_forget_domain(client) -> None:
    c, base = client
    _seed_event(base, domain="a.com", title="A")
    _seed_event(base, domain="b.com", title="B")
    r = c.post(
        "/api/cp/browse/forget",
        json={"scope": "domain", "domain": "a.com"},
    )
    assert r.status_code == 200
    assert r.json()["rows_removed"] == 1
    body = c.get("/api/cp/browse/state").json()
    assert body["stats"]["total"] == 1


def test_forget_day(client) -> None:
    c, base = client
    _seed_event(base, domain="x.com", title="X",
                ts="2026-05-15T08:00:00+00:00")
    r = c.post(
        "/api/cp/browse/forget",
        json={"scope": "day", "day": "2026-05-15"},
    )
    assert r.status_code == 200
    assert r.json()["removed"] is True


def test_forget_unknown_scope_rejected(client) -> None:
    c, _ = client
    r = c.post(
        "/api/cp/browse/forget",
        json={"scope": "everything"},
    )
    assert r.status_code in (400, 422)
