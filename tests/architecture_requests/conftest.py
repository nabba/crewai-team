"""Shared fixtures for architecture-request tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.architecture_requests import store
from app.architecture_requests.models import (
    ArchitectureRequest,
    ArchStatus,
    FileSpec,
    IntegrationPoint,
)


@pytest.fixture(autouse=True)
def isolate_store(tmp_path: Path):
    """Each test gets a fresh store rooted at tmp_path."""
    store.reset_for_tests(tmp_path / "architecture_requests")
    yield
    store.reset_for_tests(None)


def make_request(
    *,
    intent: str = "Add weekly philosophical inquiry pass",
    package_path: str = "app/inquiry/",
    file_layout: list[FileSpec] | None = None,
    integration_points: list[IntegrationPoint] | None = None,
    env_switches: dict[str, str] | None = None,
    requestor: str = "self_improver",
    test_plan: str = "Test composer; test no current_goals write; test path confinement.",
    status: ArchStatus = ArchStatus.PROPOSED,
) -> ArchitectureRequest:
    """Construct a valid-by-default ArchitectureRequest for tests."""
    if file_layout is None:
        file_layout = [
            FileSpec(path="app/inquiry/__init__.py", purpose="public surface"),
            FileSpec(path="app/inquiry/composer.py", purpose="runs the inquiry"),
        ]
    if integration_points is None:
        integration_points = [
            IntegrationPoint(
                kind="idle_job_registration",
                target_module="app/idle_scheduler.py",
                detail={"name": "philosophical-inquiry", "weight": "HEAVY"},
            ),
        ]
    if env_switches is None:
        env_switches = {"INQUIRY_PASS_ENABLED": "true"}
    return ArchitectureRequest(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        requestor=requestor,
        intent=intent,
        motivation=(
            "The system has phronesis and shadow but no place for explicit "
            "philosophical inquiry. A weekly observational pass writes one "
            "essay-length entry under wiki/self/inquiries/."
        ),
        package_path=package_path,
        file_layout=list(file_layout),
        integration_points=list(integration_points),
        env_switches=dict(env_switches),
        test_plan=test_plan,
        status=status,
    )
