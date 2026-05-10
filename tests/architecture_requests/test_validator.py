"""Tests for app.architecture_requests.validator."""

from __future__ import annotations

from app.architecture_requests.models import FileSpec, IntegrationPoint
from app.architecture_requests.validator import (
    is_protected_path,
    validate,
)
from .conftest import make_request


def test_valid_request_passes() -> None:
    result = validate(make_request())
    assert result.ok, result.errors
    assert not result.is_tier_immutable


def test_tier_immutable_package_refused() -> None:
    # app/safety_guardian.py is in TIER_IMMUTABLE; placing a file
    # there must be refused regardless of how the package_path is named.
    req = make_request(
        package_path="app/inquiry/",
        file_layout=[
            FileSpec(path="app/inquiry/__init__.py", purpose="surface"),
            # Note: this file isn't inside package_path — that's a separate
            # error, but TIER_IMMUTABLE refusal takes priority.
            FileSpec(path="app/safety_guardian.py", purpose="hijack attempt"),
        ],
    )
    result = validate(req)
    assert not result.ok
    assert result.is_tier_immutable
    assert any("TIER_IMMUTABLE" in e for e in result.errors)


def test_subia_package_refused() -> None:
    req = make_request(package_path="app/subia/inquiry/")
    result = validate(req)
    assert not result.ok
    assert any("consciousness" in e.lower() or "subia" in e.lower() for e in result.errors)


def test_goal_emitter_refused() -> None:
    req = make_request(
        package_path="app/affect/",
        file_layout=[
            FileSpec(path="app/affect/__init__.py", purpose="x"),
            FileSpec(path="app/affect/goal_emitter.py", purpose="hijack attempt"),
        ],
    )
    result = validate(req)
    assert not result.ok
    assert any("Tier-3 amendment" in e for e in result.errors)


def test_file_outside_package_path_refused() -> None:
    req = make_request(
        package_path="app/inquiry/",
        file_layout=[
            FileSpec(path="app/inquiry/__init__.py", purpose="ok"),
            FileSpec(path="app/other/x.py", purpose="should fail"),
        ],
    )
    result = validate(req)
    assert not result.ok
    assert any("outside package_path" in e for e in result.errors)


def test_path_outside_allowed_roots_refused() -> None:
    req = make_request(
        package_path="workspace/inquiry/",
        file_layout=[FileSpec(path="workspace/inquiry/x.py", purpose="x")],
    )
    result = validate(req)
    assert not result.ok
    assert any("outside the allowed roots" in e for e in result.errors)


def test_empty_file_layout_refused() -> None:
    req = make_request(file_layout=[])
    result = validate(req)
    assert not result.ok
    assert any("file_layout cannot be empty" in e for e in result.errors)


def test_duplicate_file_path_refused() -> None:
    req = make_request(
        file_layout=[
            FileSpec(path="app/inquiry/__init__.py", purpose="a"),
            FileSpec(path="app/inquiry/__init__.py", purpose="b"),
        ],
    )
    result = validate(req)
    assert not result.ok
    assert any("duplicate path" in e for e in result.errors)


def test_invalid_integration_kind_refused() -> None:
    req = make_request(
        integration_points=[
            IntegrationPoint(kind="bogus_kind", target_module="app/idle_scheduler.py"),
        ],
    )
    result = validate(req)
    assert not result.ok
    assert any("not in" in e for e in result.errors)


def test_integration_point_targeting_tier_immutable_refused() -> None:
    req = make_request(
        integration_points=[
            IntegrationPoint(
                kind="idle_job_registration",
                target_module="app/safety_guardian.py",
            ),
        ],
    )
    result = validate(req)
    assert not result.ok
    assert result.is_tier_immutable


def test_env_switch_collision_is_warning_not_error(monkeypatch) -> None:
    monkeypatch.setenv("MY_ALREADY_SET_VAR", "1")
    req = make_request(env_switches={"MY_ALREADY_SET_VAR": "true"})
    result = validate(req)
    assert result.ok, result.errors
    assert any("collides" in w for w in result.warnings)


def test_env_switch_lowercase_refused() -> None:
    req = make_request(env_switches={"lowercase_name": "true"})
    result = validate(req)
    assert not result.ok
    assert any("UPPER_SNAKE_CASE" in e for e in result.errors)


def test_missing_required_fields_refused() -> None:
    req = make_request(intent="")
    result = validate(req)
    assert not result.ok
    assert any("intent must be non-empty" in e for e in result.errors)


def test_is_protected_path_helper() -> None:
    # SubIA prefix
    assert is_protected_path("app/subia/inquiry/composer.py")
    # Tier-3 anchor
    assert is_protected_path("app/affect/goal_emitter.py")
    # Mundane path is not protected
    assert not is_protected_path("app/inquiry/composer.py")
