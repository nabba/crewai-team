"""Tests for app.architecture_requests.lifecycle."""

from __future__ import annotations

import pytest

from app.architecture_requests import lifecycle, store
from app.architecture_requests.lifecycle import InvalidTransition
from app.architecture_requests.models import (
    ArchStatus,
    DecisionSource,
    FileSpec,
    IntegrationPoint,
)


def _proposal_kwargs(**overrides):
    base = {
        "requestor": "self_improver",
        "intent": "Add weekly philosophical inquiry pass",
        "motivation": (
            "The system needs a place for explicit philosophical thinking "
            "separate from any task pressure."
        ),
        "package_path": "app/inquiry/",
        "file_layout": [
            FileSpec(path="app/inquiry/__init__.py", purpose="surface"),
            FileSpec(path="app/inquiry/composer.py", purpose="runs the inquiry"),
        ],
        "integration_points": [
            IntegrationPoint(
                kind="idle_job_registration",
                target_module="app/idle_scheduler.py",
                detail={"name": "philosophical-inquiry", "weight": "HEAVY"},
            ),
        ],
        "env_switches": {"INQUIRY_ENABLED": "true"},
        "test_plan": "Composer rejects phenomenal language; path confinement.",
    }
    base.update(overrides)
    return base


def test_create_request_happy_path() -> None:
    req = lifecycle.create_request(**_proposal_kwargs())
    assert req.status is ArchStatus.PROPOSED
    assert store.get(req.id) is not None


def test_create_request_with_tier_immutable_path() -> None:
    req = lifecycle.create_request(
        **_proposal_kwargs(
            file_layout=[
                FileSpec(path="app/inquiry/__init__.py", purpose="ok"),
                FileSpec(path="app/safety_guardian.py", purpose="hijack"),
            ],
        )
    )
    assert req.status is ArchStatus.TIER_IMMUTABLE_REFUSED
    assert req.is_terminal
    assert "TIER_IMMUTABLE" in (req.decision_reason or "")


def test_create_request_with_invalid_fields_rejects_at_validate_time() -> None:
    req = lifecycle.create_request(**_proposal_kwargs(intent=""))
    assert req.status is ArchStatus.REJECTED
    assert req.is_terminal


def test_full_happy_path_through_completed() -> None:
    req = lifecycle.create_request(**_proposal_kwargs())
    assert req.status is ArchStatus.PROPOSED

    req = lifecycle.approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    assert req.status is ArchStatus.APPROVED
    assert req.decided_by is DecisionSource.SIGNAL_THUMBS_UP

    req = lifecycle.scaffold(req.id, scaffold_dir="/tmp/scaffold/x")
    assert req.status is ArchStatus.SCAFFOLDED
    assert req.scaffold_dir == "/tmp/scaffold/x"

    req = lifecycle.record_child_change_request(req.id, "cr-001")
    assert req.status is ArchStatus.IMPLEMENTING
    assert "cr-001" in req.child_change_request_ids

    req = lifecycle.record_child_change_request(req.id, "cr-002")
    assert req.status is ArchStatus.IMPLEMENTING
    assert req.child_change_request_ids == ["cr-001", "cr-002"]

    req = lifecycle.mark_complete(req.id)
    assert req.status is ArchStatus.COMPLETED
    assert req.is_terminal


def test_reject_path() -> None:
    req = lifecycle.create_request(**_proposal_kwargs())
    req = lifecycle.reject(
        req.id,
        source=DecisionSource.SIGNAL_THUMBS_DOWN,
        decision_reason="duplicates an existing system",
    )
    assert req.status is ArchStatus.REJECTED
    assert req.decision_reason == "duplicates an existing system"
    assert req.is_terminal


def test_approve_is_idempotent() -> None:
    req = lifecycle.create_request(**_proposal_kwargs())
    a = lifecycle.approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    b = lifecycle.approve(req.id, source=DecisionSource.REACT_APPROVE)
    # Idempotent: second call doesn't transition or overwrite source.
    assert a.status is b.status is ArchStatus.APPROVED
    assert b.decided_by is DecisionSource.SIGNAL_THUMBS_UP


def test_illegal_transitions_raise() -> None:
    req = lifecycle.create_request(**_proposal_kwargs())
    # Cannot scaffold directly from PROPOSED.
    with pytest.raises(InvalidTransition):
        lifecycle.scaffold(req.id, scaffold_dir="/tmp/x")
    # Cannot mark_complete from APPROVED.
    lifecycle.approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    with pytest.raises(InvalidTransition):
        lifecycle.mark_complete(req.id)


def test_abandon_from_scaffolded_or_implementing() -> None:
    req = lifecycle.create_request(**_proposal_kwargs())
    lifecycle.approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    lifecycle.scaffold(req.id, scaffold_dir="/tmp/x")
    req = lifecycle.abandon(req.id, "operator changed their mind")
    assert req.status is ArchStatus.ABANDONED
    assert req.abandon_reason == "operator changed their mind"
    assert req.is_terminal


def test_expire_only_from_proposed() -> None:
    req = lifecycle.create_request(**_proposal_kwargs())
    lifecycle.approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    with pytest.raises(InvalidTransition):
        lifecycle.expire(req.id)


def test_audit_chain_records_every_transition() -> None:
    req = lifecycle.create_request(**_proposal_kwargs())
    lifecycle.approve(req.id, source=DecisionSource.SIGNAL_THUMBS_UP)
    lifecycle.scaffold(req.id, scaffold_dir="/tmp/x")
    lifecycle.record_child_change_request(req.id, "cr-1")
    lifecycle.mark_complete(req.id)

    events = [p["event"] for p in store.iter_audit_entries()]
    assert events == [
        "created", "approved", "scaffolded", "implementing_started", "completed",
    ]


def test_unknown_request_raises() -> None:
    with pytest.raises(KeyError):
        lifecycle.approve("nonexistent-id", source=DecisionSource.SIGNAL_THUMBS_UP)
