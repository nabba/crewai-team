"""Tests for app.architecture_requests.models."""

from __future__ import annotations

from app.architecture_requests.models import (
    ArchitectureRequest,
    ArchStatus,
    DecisionSource,
    FileSpec,
    IntegrationPoint,
)
from .conftest import make_request


def test_round_trip_serialization() -> None:
    req = make_request()
    req.signal_message_ts = 1747900000
    req.decided_by = DecisionSource.SIGNAL_THUMBS_UP
    req.decided_at = "2026-05-10T03:00:00+00:00"
    req.decision_reason = "looks good"

    data = req.to_dict()
    restored = ArchitectureRequest.from_dict(data)

    assert restored.id == req.id
    assert restored.intent == req.intent
    assert restored.package_path == req.package_path
    assert [f.path for f in restored.file_layout] == [f.path for f in req.file_layout]
    assert restored.env_switches == req.env_switches
    assert restored.signal_message_ts == 1747900000
    assert restored.decided_by is DecisionSource.SIGNAL_THUMBS_UP
    assert restored.status is req.status


def test_is_terminal_property() -> None:
    req = make_request(status=ArchStatus.PROPOSED)
    assert not req.is_terminal

    req.status = ArchStatus.APPROVED
    assert not req.is_terminal

    for s in (
        ArchStatus.REJECTED,
        ArchStatus.TIER_IMMUTABLE_REFUSED,
        ArchStatus.TIMEOUT,
        ArchStatus.COMPLETED,
        ArchStatus.ABANDONED,
    ):
        req.status = s
        assert req.is_terminal, f"expected {s} to be terminal"


def test_is_decided_property() -> None:
    req = make_request()
    assert not req.is_decided
    req.status = ArchStatus.APPROVED
    assert req.is_decided
    req.status = ArchStatus.REJECTED
    assert req.is_decided


def test_filespec_round_trip() -> None:
    fs = FileSpec(path="app/inquiry/composer.py", purpose="runs the inquiry")
    assert FileSpec.from_dict(fs.to_dict()) == fs

    fs2 = FileSpec(path="app/inquiry/__init__.py", purpose="surface", initial_stub="x")
    assert FileSpec.from_dict(fs2.to_dict()).initial_stub == "x"


def test_integration_point_round_trip() -> None:
    ip = IntegrationPoint(
        kind="idle_job_registration",
        target_module="app/idle_scheduler.py",
        detail={"name": "philosophical-inquiry", "weight": "HEAVY"},
    )
    restored = IntegrationPoint.from_dict(ip.to_dict())
    assert restored == ip


def test_status_values_match_str_enum() -> None:
    # Sanity: serialised status must be a string.
    assert ArchStatus.PROPOSED.value == "proposed"
    assert ArchStatus.SCAFFOLDED.value == "scaffolded"
    assert ArchStatus.TIER_IMMUTABLE_REFUSED.value == "tier_immutable_refused"
