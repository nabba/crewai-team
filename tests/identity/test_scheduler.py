"""Tests for app.identity.scheduler — the idle-job entry points."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.identity import scheduler


def test_get_idle_jobs_returns_two_jobs() -> None:
    jobs = scheduler.get_idle_jobs()
    assert len(jobs) == 2
    names = [j[0] for j in jobs]
    assert "identity-annual-reflection" in names
    assert "identity-legacy-essay" in names


def test_run_annual_reflection_skips_when_llm_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """When the LLM factory raises, the job logs and returns without raising."""
    monkeypatch.setattr(scheduler, "_resolve_llm_call", lambda: None)
    # Should not raise.
    scheduler.run_annual_reflection()


def test_run_annual_reflection_calls_run_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LLM is available, the scheduler calls run_one_pass with it."""
    captured: list[tuple] = []

    def fake_llm_call(system: str, user: str) -> str:
        return "(mock essay)"

    monkeypatch.setattr(scheduler, "_resolve_llm_call", lambda: fake_llm_call)

    def fake_run_one_pass(*, llm_call):
        captured.append(("annual", llm_call is fake_llm_call))
        from app.identity.annual_reflection import ReflectionResult
        return ReflectionResult(status="skipped_recent", year=2026)

    with patch(
        "app.identity.annual_reflection.run_one_pass",
        side_effect=fake_run_one_pass,
    ):
        scheduler.run_annual_reflection()

    assert captured == [("annual", True)]


def test_run_legacy_essay_calls_run_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy job mirrors the annual job's behaviour."""
    captured: list[tuple] = []

    def fake_llm_call(system: str, user: str) -> str:
        return "(mock essay)"

    monkeypatch.setattr(scheduler, "_resolve_llm_call", lambda: fake_llm_call)

    def fake_run_one_pass(*, llm_call):
        captured.append(("legacy", llm_call is fake_llm_call))
        from app.identity.legacy_essay import LegacyResult
        return LegacyResult(status="skipped_recent", year=2026)

    with patch(
        "app.identity.legacy_essay.run_one_pass",
        side_effect=fake_run_one_pass,
    ):
        scheduler.run_legacy_essay()

    assert captured == [("legacy", True)]


def test_run_annual_reflection_swallows_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If run_one_pass raises, the scheduler must not propagate."""
    monkeypatch.setattr(
        scheduler, "_resolve_llm_call", lambda: lambda s, u: "x",
    )
    with patch(
        "app.identity.annual_reflection.run_one_pass",
        side_effect=RuntimeError("boom"),
    ):
        # Must not raise.
        scheduler.run_annual_reflection()


def test_run_legacy_essay_swallows_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scheduler, "_resolve_llm_call", lambda: lambda s, u: "x",
    )
    with patch(
        "app.identity.legacy_essay.run_one_pass",
        side_effect=RuntimeError("boom"),
    ):
        scheduler.run_legacy_essay()


def test_resolve_llm_call_returns_callable_or_none() -> None:
    """_resolve_llm_call returns either a callable or None — never raises."""
    # In the test environment the factory may or may not be wired; either
    # way the resolver returns cleanly.
    result = scheduler._resolve_llm_call()
    assert result is None or callable(result)


def test_companion_loop_picks_up_identity_jobs() -> None:
    """The companion loop's get_idle_jobs() includes the identity jobs."""
    from app.companion.loop import get_idle_jobs as companion_jobs
    names = [name for name, _, _ in companion_jobs()]
    assert "identity-annual-reflection" in names
    assert "identity-legacy-essay" in names
