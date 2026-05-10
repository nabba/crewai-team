"""Tests for app.control_plane.action_requests_api."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.action_requests import (
    ActionStatus,
    ActionType,
    DecisionSource,
    create_request,
    reset_for_tests,
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from app.control_plane import auth_dep
    monkeypatch.setattr(auth_dep, "require_gateway_auth", lambda: True)
    reset_for_tests(tmp_path / "action_requests")

    from app.control_plane.action_requests_api import router
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_dep.require_gateway_auth] = lambda: True

    yield TestClient(app)
    reset_for_tests(None)


def _good_email_request():
    return create_request(
        requestor="companion",
        action_type=ActionType.EMAIL_DRAFT,
        summary="weekly status email",
        data={
            "to": "operator@example.com",
            "subject": "weekly digest",
            "body": "All systems nominal.",
        },
        reason="Friday digest to operator",
    )


def test_get_types(client) -> None:
    r = client.get("/api/cp/action-requests/types")
    assert r.status_code == 200
    body = r.json()
    assert "email_draft" in body["types"]


def test_list_returns_created(client) -> None:
    req = _good_email_request()
    r = client.get("/api/cp/action-requests")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["action_requests"][0]["id"] == req.id


def test_list_filters_by_status(client) -> None:
    req = _good_email_request()
    r1 = client.get("/api/cp/action-requests?status=pending")
    r2 = client.get("/api/cp/action-requests?status=approved")
    assert {x["id"] for x in r1.json()["action_requests"]} == {req.id}
    assert r2.json()["count"] == 0


def test_list_filters_by_action_type(client) -> None:
    _good_email_request()
    r = client.get("/api/cp/action-requests?action_type=email_draft")
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_invalid_status_returns_400(client) -> None:
    assert client.get("/api/cp/action-requests?status=bogus").status_code == 400


def test_invalid_action_type_returns_400(client) -> None:
    assert client.get("/api/cp/action-requests?action_type=bogus").status_code == 400


def test_get_detail_serializes_helpers(client) -> None:
    req = _good_email_request()
    body = client.get(f"/api/cp/action-requests/{req.id}").json()
    assert body["id"] == req.id
    assert body["is_terminal"] is False
    assert body["is_decided"] is False


def test_unknown_id_returns_404(client) -> None:
    assert client.get("/api/cp/action-requests/missing").status_code == 404


def test_approve_path_applies(client, monkeypatch) -> None:
    """Approve via REST should trigger apply via the email handler."""
    sent = []

    def fake_send(**kwargs):
        sent.append(kwargs)
        return {"ok": True, "message_id": "<abc@host>"}

    import app.delivery.email_send as email_send_mod
    monkeypatch.setattr(email_send_mod, "send_via_email", fake_send)

    req = _good_email_request()
    r = client.post(
        f"/api/cp/action-requests/{req.id}/approve",
        json={"operator": "op", "reason": "looks good"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["action_request"]["status"] == "applied"
    assert sent[0]["subject"] == "weekly digest"


def test_invalid_request_cannot_be_approved(client) -> None:
    """An INVALID action-request (validator rejected) returns 403 on approve."""
    req = create_request(
        requestor="companion",
        action_type=ActionType.EMAIL_DRAFT,
        summary="bad email",
        data={"to": "not-an-email", "subject": "x", "body": "y"},
        reason="r",
    )
    assert req.status is ActionStatus.INVALID

    r = client.post(
        f"/api/cp/action-requests/{req.id}/approve",
        json={"operator": "op"},
    )
    assert r.status_code == 403
    assert "INVALID" in r.json()["detail"]


def test_reject_path(client) -> None:
    req = _good_email_request()
    r = client.post(
        f"/api/cp/action-requests/{req.id}/reject",
        json={"operator": "op", "reason": "content not ready"},
    )
    assert r.status_code == 200
    assert r.json()["action_request"]["status"] == "rejected"


def test_cannot_reject_after_approval(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.delivery.email_send.send_via_email",
        lambda **_: {"ok": True},
    )
    req = _good_email_request()
    client.post(
        f"/api/cp/action-requests/{req.id}/approve", json={},
    )
    r = client.post(
        f"/api/cp/action-requests/{req.id}/reject", json={},
    )
    assert r.status_code == 409


def test_retry_apply_after_failure(client, monkeypatch) -> None:
    """After APPLY_FAILED, retry-apply transitions back through approve+apply."""
    calls = {"n": 0}

    def maybe_fail(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {"ok": True, "message_id": "<x>"}

    import app.delivery.email_send as email_send_mod
    monkeypatch.setattr(email_send_mod, "send_via_email", maybe_fail)

    req = _good_email_request()
    client.post(
        f"/api/cp/action-requests/{req.id}/approve",
        json={"operator": "op"},
    )
    # First apply fails — status now APPLY_FAILED.
    detail = client.get(f"/api/cp/action-requests/{req.id}").json()
    assert detail["status"] == "apply_failed"

    r = client.post(f"/api/cp/action-requests/{req.id}/retry-apply")
    assert r.status_code == 200
    assert r.json()["action_request"]["status"] == "applied"


def test_retry_apply_only_for_apply_failed(client) -> None:
    req = _good_email_request()
    r = client.post(f"/api/cp/action-requests/{req.id}/retry-apply")
    assert r.status_code == 409
