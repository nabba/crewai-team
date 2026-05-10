"""Tests for app.action_requests.* — non-code action gate primitive."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.action_requests import (
    ActionRequest,
    ActionStatus,
    ActionType,
    ApplyResult,
    DecisionSource,
    InvalidActionTransition,
    apply,
    approve,
    create_request,
    expire,
    find_by_signal_ts,
    get,
    get_handler,
    is_action_type_supported,
    list_action_types,
    list_all,
    reject,
    reset_for_tests,
    validate,
)
from app.action_requests.handlers.base import ActionHandler
from app.action_requests.handlers import register_handler
from app.action_requests.lifecycle import attach_signal_ts


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path):
    reset_for_tests(tmp_path)
    yield
    reset_for_tests(None)


def _email_data(**overrides) -> dict[str, Any]:
    base = {
        "to": "operator@example.com",
        "subject": "weekly summary",
        "body": "All systems nominal.",
    }
    base.update(overrides)
    return base


def _create_email(**overrides) -> ActionRequest:
    data = _email_data(**overrides.pop("data", {}))
    return create_request(
        requestor=overrides.get("requestor", "companion"),
        action_type=ActionType.EMAIL_DRAFT,
        summary=overrides.get("summary", "weekly status email"),
        data=data,
        reason=overrides.get("reason", "Friday digest to operator"),
    )


# ── models + serialization ────────────────────────────────────────────


def test_round_trip_serialization() -> None:
    req = _create_email()
    raw = req.to_dict()
    restored = ActionRequest.from_dict(raw)
    assert restored.id == req.id
    assert restored.action_type is ActionType.EMAIL_DRAFT
    assert restored.data["subject"] == "weekly summary"
    assert restored.status is ActionStatus.PENDING


def test_is_terminal_property() -> None:
    req = _create_email()
    assert not req.is_terminal
    req.status = ActionStatus.APPLIED
    assert req.is_terminal


def test_is_decided_property() -> None:
    req = _create_email()
    assert not req.is_decided
    req.status = ActionStatus.APPROVED
    assert req.is_decided


# ── validator ─────────────────────────────────────────────────────────


def test_email_draft_validate_success() -> None:
    req = _create_email()
    out = validate(req)
    assert out.ok


def test_validate_rejects_empty_summary() -> None:
    req = _create_email(summary="")
    out = validate(req)
    assert not out.ok
    assert "summary" in out.reason


def test_validate_rejects_invalid_email_address() -> None:
    req = _create_email(data={"to": "not-an-email"})
    out = validate(req)
    assert not out.ok
    assert "invalid email" in out.reason.lower()


def test_validate_rejects_missing_subject() -> None:
    req = _create_email(data={"subject": ""})
    out = validate(req)
    assert not out.ok
    assert "subject" in out.reason


def test_validate_rejects_oversize_body() -> None:
    req = _create_email(data={"body": "x" * (200_000)})
    out = validate(req)
    assert not out.ok
    assert "body" in out.reason


def test_email_draft_supports_recipient_list() -> None:
    req = _create_email(data={"to": ["a@example.com", "b@example.com"]})
    out = validate(req)
    assert out.ok


def test_is_action_type_supported() -> None:
    assert is_action_type_supported(ActionType.EMAIL_DRAFT)


def test_list_action_types_contains_email_draft() -> None:
    assert ActionType.EMAIL_DRAFT in list_action_types()


# ── lifecycle happy path ──────────────────────────────────────────────


def test_create_invalid_email_lands_in_invalid_status() -> None:
    req = create_request(
        requestor="companion",
        action_type=ActionType.EMAIL_DRAFT,
        summary="bad email",
        data={"to": "not-an-email", "subject": "x", "body": "y"},
        reason="r",
    )
    assert req.status is ActionStatus.INVALID
    assert "invalid email" in req.invalid_reason.lower()


def test_full_happy_path_to_applied(monkeypatch) -> None:
    sent_payloads: list[dict] = []

    def fake_send(**kwargs):
        sent_payloads.append(kwargs)
        return {"ok": True, "message_id": "<abc@host>"}

    import app.delivery.email_send as email_send_mod
    monkeypatch.setattr(email_send_mod, "send_via_email", fake_send)

    req = _create_email()
    assert req.status is ActionStatus.PENDING

    req = approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    assert req.status is ActionStatus.APPROVED

    req = apply(req.id)
    assert req.status is ActionStatus.APPLIED
    assert req.apply_artifact.get("message_id") == "<abc@host>"
    assert sent_payloads[0]["subject"] == "weekly summary"


def test_reject_path() -> None:
    req = _create_email()
    req = reject(
        req.id,
        source=DecisionSource.SIGNAL_THUMBS_DOWN,
        decision_reason="content not ready",
    )
    assert req.status is ActionStatus.REJECTED
    assert req.decision_reason == "content not ready"


def test_apply_failure_path(monkeypatch) -> None:
    def boom(**_):
        raise RuntimeError("smtp unreachable")

    import app.delivery.email_send as email_send_mod
    monkeypatch.setattr(email_send_mod, "send_via_email", boom)

    req = _create_email()
    approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    req = apply(req.id)
    assert req.status is ActionStatus.APPLY_FAILED
    assert "smtp unreachable" in req.apply_error


def test_apply_failure_reapproval_retries(monkeypatch) -> None:
    """APPLY_FAILED → re-approve resets to APPROVED → apply tries again.
    Useful for transient failures (network blip, etc)."""
    calls = {"n": 0}

    def maybe_fail(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {"ok": True, "message_id": "<retry@host>"}

    import app.delivery.email_send as email_send_mod
    monkeypatch.setattr(email_send_mod, "send_via_email", maybe_fail)

    req = _create_email()
    approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    req = apply(req.id)
    assert req.status is ActionStatus.APPLY_FAILED

    # Retry: re-approve + re-apply.
    req = approve(req.id, source=DecisionSource.REACT_APPROVE,
                  decision_reason="retry after transient failure")
    assert req.status is ActionStatus.APPROVED
    req = apply(req.id)
    assert req.status is ActionStatus.APPLIED


def test_apply_handler_returning_false_is_apply_failed(monkeypatch) -> None:
    def fake(**_):
        return {"ok": False, "error": "spam filter blocked"}

    import app.delivery.email_send as email_send_mod
    monkeypatch.setattr(email_send_mod, "send_via_email", fake)

    req = _create_email()
    approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    req = apply(req.id)
    assert req.status is ActionStatus.APPLY_FAILED
    assert "spam filter" in req.apply_error


def test_expire_only_from_pending() -> None:
    req = _create_email()
    req = expire(req.id)
    assert req.status is ActionStatus.TIMEOUT


def test_illegal_transitions() -> None:
    req = _create_email()
    # Cannot apply before approval.
    with pytest.raises(InvalidActionTransition):
        apply(req.id)
    # Cannot reject after approval.
    approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    with pytest.raises(InvalidActionTransition):
        reject(req.id, source=DecisionSource.SIGNAL_THUMBS_DOWN)


def test_unknown_request_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        approve("nonexistent-id", source=DecisionSource.SIGNAL_THUMBS_UP)


# ── store + list ──────────────────────────────────────────────────────


def test_list_all_filters_by_status() -> None:
    a = _create_email(summary="A")
    b = _create_email(summary="B")
    approve(a.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    pendings = list_all(status=ActionStatus.PENDING)
    approveds = list_all(status=ActionStatus.APPROVED)
    assert {r.id for r in pendings} == {b.id}
    assert {r.id for r in approveds} == {a.id}


def test_signal_ts_correlation() -> None:
    req = _create_email()
    attach_signal_ts(req.id, 1747900000)
    assert find_by_signal_ts(1747900000) == req.id
    assert find_by_signal_ts(0) is None
    assert find_by_signal_ts(99999) is None


def test_persistence_survives_index_reset(tmp_path: Path) -> None:
    req = _create_email()
    approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    reset_for_tests(tmp_path)
    reloaded = get(req.id)
    assert reloaded is not None
    assert reloaded.status is ActionStatus.APPROVED


# ── handler render_summary ─────────────────────────────────────────────


def test_email_handler_render_summary_single_recipient() -> None:
    handler = get_handler(ActionType.EMAIL_DRAFT)
    out = handler.render_summary({
        "to": "operator@example.com",
        "subject": "weekly digest",
        "body": "ok",
    })
    assert "operator@example.com" in out
    assert "weekly digest" in out


def test_email_handler_render_summary_multi_recipient() -> None:
    handler = get_handler(ActionType.EMAIL_DRAFT)
    out = handler.render_summary({
        "to": ["a@example.com", "b@example.com", "c@example.com"],
        "subject": "broadcast",
        "body": "ok",
    })
    assert "+2 more" in out


# ── handler registry ───────────────────────────────────────────────────


def test_custom_handler_can_register_and_handle_apply() -> None:
    """Anyone can plug in a new action type with a 3-method handler."""

    class _DummyHandler(ActionHandler):
        @property
        def action_type(self):
            return ActionType.EMAIL_DRAFT  # piggyback on existing enum for the test

        def validate(self, data):
            return True, None

        def apply(self, data):
            return ApplyResult(ok=True, artifact={"echoed": data.get("body", "")})

        def render_summary(self, data):
            return "dummy"

    register_handler(_DummyHandler())  # overwrites email_draft for this test
    try:
        req = _create_email()
        approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
        req = apply(req.id)
        assert req.status is ActionStatus.APPLIED
        assert req.apply_artifact["echoed"] == "All systems nominal."
    finally:
        # Restore the real handler.
        from app.action_requests.handlers.email_draft import EmailDraftHandler
        register_handler(EmailDraftHandler())
