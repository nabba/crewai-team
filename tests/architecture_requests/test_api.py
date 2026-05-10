"""Tests for app.control_plane.architecture_requests_api."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.architecture_requests import lifecycle, store
from app.architecture_requests.models import FileSpec, IntegrationPoint


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    # Bypass auth for the test client.
    from app.control_plane import auth_dep
    monkeypatch.setattr(auth_dep, "require_gateway_auth", lambda: True)

    store.reset_for_tests(tmp_path / "arch")

    from app.control_plane.architecture_requests_api import router
    app = FastAPI()
    app.include_router(router)
    # Re-bind the auth dependency in this app instance.
    app.dependency_overrides[auth_dep.require_gateway_auth] = lambda: True

    yield TestClient(app), tmp_path
    store.reset_for_tests(None)


def _create() -> str:
    req = lifecycle.create_request(
        requestor="self_improver",
        intent="Add inquiry pass",
        motivation="The system needs a place for explicit philosophical thinking.",
        package_path="app/inquiry/",
        file_layout=[
            FileSpec(path="app/inquiry/__init__.py", purpose="surface"),
            FileSpec(path="app/inquiry/composer.py", purpose="runs the inquiry"),
        ],
        integration_points=[
            IntegrationPoint(
                kind="idle_job_registration",
                target_module="app/idle_scheduler.py",
                detail={"name": "philosophical-inquiry", "weight": "HEAVY"},
            ),
        ],
        env_switches={"INQUIRY_PASS_ENABLED": "true"},
        test_plan="composer rejects phenomenal language; path confinement.",
    )
    return req.id


def test_list_returns_created_request(client) -> None:
    c, _ = client
    rid = _create()
    r = c.get("/api/cp/architecture-requests")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["architecture_requests"][0]["id"] == rid


def test_list_filters_by_status(client) -> None:
    c, _ = client
    a = _create()
    b = _create()
    lifecycle.approve(a, source=lifecycle.DecisionSource.REACT_APPROVE)
    r1 = c.get("/api/cp/architecture-requests?status=approved")
    r2 = c.get("/api/cp/architecture-requests?status=proposed")
    assert {x["id"] for x in r1.json()["architecture_requests"]} == {a}
    assert {x["id"] for x in r2.json()["architecture_requests"]} == {b}


def test_get_detail_serialises_helpers(client) -> None:
    c, _ = client
    rid = _create()
    body = c.get(f"/api/cp/architecture-requests/{rid}").json()
    assert body["id"] == rid
    assert body["is_terminal"] is False
    assert body["is_decided"] is False
    assert body["package_is_protected"] is False


def test_get_unknown_404(client) -> None:
    c, _ = client
    assert c.get("/api/cp/architecture-requests/missing").status_code == 404


def test_approve_transitions(client) -> None:
    c, _ = client
    rid = _create()
    r = c.post(f"/api/cp/architecture-requests/{rid}/approve", json={"operator": "op"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["architecture_request"]["status"] == "approved"


def test_approve_idempotent_returns_already_approved_note(client) -> None:
    c, _ = client
    rid = _create()
    c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    r2 = c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    assert r2.status_code == 200
    assert "already approved" in r2.json()["note"]


def test_reject_transitions(client) -> None:
    c, _ = client
    rid = _create()
    r = c.post(
        f"/api/cp/architecture-requests/{rid}/reject",
        json={"operator": "op", "reason": "duplicates X"},
    )
    assert r.status_code == 200
    assert r.json()["architecture_request"]["status"] == "rejected"


def test_cannot_reject_after_approval(client) -> None:
    c, _ = client
    rid = _create()
    c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    r = c.post(f"/api/cp/architecture-requests/{rid}/reject", json={})
    assert r.status_code == 409


def test_scaffold_runs_and_writes_manifest(client) -> None:
    c, tmp_path = client
    rid = _create()
    c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    r = c.post(f"/api/cp/architecture-requests/{rid}/scaffold")
    assert r.status_code == 200
    body = r.json()
    assert body["architecture_request"]["status"] == "scaffolded"
    scaffold_dir = Path(body["scaffold_dir"])
    assert scaffold_dir.exists()
    assert (scaffold_dir / "MANIFEST.md").exists()


def test_scaffold_idempotent(client) -> None:
    c, _ = client
    rid = _create()
    c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    c.post(f"/api/cp/architecture-requests/{rid}/scaffold")
    r2 = c.post(f"/api/cp/architecture-requests/{rid}/scaffold")
    assert r2.status_code == 200
    assert "already scaffolded" in r2.json()["note"]


def test_scaffold_refuses_pre_approval(client) -> None:
    c, _ = client
    rid = _create()
    r = c.post(f"/api/cp/architecture-requests/{rid}/scaffold")
    assert r.status_code == 409


def test_get_manifest_after_scaffold(client) -> None:
    c, _ = client
    rid = _create()
    c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    c.post(f"/api/cp/architecture-requests/{rid}/scaffold")
    r = c.get(f"/api/cp/architecture-requests/{rid}/scaffold/manifest")
    assert r.status_code == 200
    body = r.json()
    assert "Architecture-request manifest" in body["manifest_text"]
    assert "Add inquiry pass" in body["manifest_text"]


def test_get_manifest_pre_scaffold_returns_409(client) -> None:
    c, _ = client
    rid = _create()
    r = c.get(f"/api/cp/architecture-requests/{rid}/scaffold/manifest")
    assert r.status_code == 409


def test_record_child_cr_then_mark_complete(client) -> None:
    c, _ = client
    rid = _create()
    c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    c.post(f"/api/cp/architecture-requests/{rid}/scaffold")
    r1 = c.post(
        f"/api/cp/architecture-requests/{rid}/record-child-cr",
        json={"child_change_request_id": "cr-001"},
    )
    assert r1.status_code == 200
    assert r1.json()["architecture_request"]["status"] == "implementing"
    r2 = c.post(f"/api/cp/architecture-requests/{rid}/mark-complete")
    assert r2.status_code == 200
    assert r2.json()["architecture_request"]["status"] == "completed"


def test_audit_endpoint_returns_request_specific_entries(client) -> None:
    c, _ = client
    rid = _create()
    c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    r = c.get(f"/api/cp/architecture-requests/{rid}/audit")
    assert r.status_code == 200
    events = [e["event"] for e in r.json()["entries"]]
    assert events == ["created", "approved"]


def test_abandon_after_scaffold(client) -> None:
    c, _ = client
    rid = _create()
    c.post(f"/api/cp/architecture-requests/{rid}/approve", json={})
    c.post(f"/api/cp/architecture-requests/{rid}/scaffold")
    r = c.post(
        f"/api/cp/architecture-requests/{rid}/abandon",
        json={"reason": "duplicates a different proposal"},
    )
    assert r.status_code == 200
    assert r.json()["architecture_request"]["status"] == "abandoned"


def test_invalid_status_filter_returns_400(client) -> None:
    c, _ = client
    r = c.get("/api/cp/architecture-requests?status=bogus")
    assert r.status_code == 400
