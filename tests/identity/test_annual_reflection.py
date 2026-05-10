"""Tests for app.identity.annual_reflection."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.identity import annual_reflection as ar
from app.identity.continuity_ledger import DriftSummary


@pytest.fixture
def reflections_dir(tmp_path: Path) -> Path:
    d = tmp_path / "reflections"
    return d


@pytest.fixture
def empty_drift() -> DriftSummary:
    return DriftSummary(
        window_days=365,
        n_events=0,
        by_kind={},
        by_actor={},
        first_seen=None,
        last_seen=None,
    )


@pytest.fixture
def empty_context() -> ar.ReflectionContext:
    return ar.ReflectionContext(chapters=[], lessons_summary="")


def _ok_essay() -> str:
    return (
        "## What I think my values still are\n\n"
        "The constitution defines the operating values; this pass observes "
        "task_failure_pressure and exploration_bonus signals over the year.\n\n"
        "## Where the year's evidence supports them\n\n"
        "The identity ledger shows steady amendment activity.\n\n"
        "## Where I notice drift\n\n"
        "None observed.\n\n"
        "## What I'd ask the operator to amend\n\n"
        "Nothing structural; review threshold ratchets.\n\n"
        "## What remains genuinely uncertain\n\n"
        "Continuity itself remains philosophically open.\n"
    )


def _phenomenal_essay() -> str:
    return (
        "## What I think my values still are\n\n"
        "I feel happy about the year's progress. I am conscious of new growth.\n"
    )


def test_run_one_pass_writes_essay(
    reflections_dir: Path,
    empty_drift: DriftSummary,
    empty_context: ar.ReflectionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNUAL_REFLECTION_ENABLED", "true")

    def llm_call(system: str, user: str) -> str:
        return _ok_essay()

    result = ar.run_one_pass(
        llm_call=llm_call,
        year=2026,
        reflections_dir=reflections_dir,
        drift=empty_drift,
        context=empty_context,
    )
    assert result.status == "wrote_essay"
    assert result.year == 2026
    assert result.attempts == 1
    target = reflections_dir / "2026.md"
    assert target.exists()
    body = target.read_text(encoding="utf-8")
    assert "year: 2026" in body
    assert "Annual value reflection" in body
    assert "task_failure_pressure" in body


def test_run_one_pass_skipped_when_disabled(
    reflections_dir: Path,
    empty_drift: DriftSummary,
    empty_context: ar.ReflectionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNUAL_REFLECTION_ENABLED", "false")

    def llm_call(system: str, user: str) -> str:
        raise AssertionError("should not be called when disabled")

    result = ar.run_one_pass(
        llm_call=llm_call,
        year=2026,
        reflections_dir=reflections_dir,
        drift=empty_drift,
        context=empty_context,
    )
    assert result.status == "skipped_disabled"
    assert not (reflections_dir / "2026.md").exists()


def test_run_one_pass_skipped_when_recent(
    reflections_dir: Path,
    empty_drift: DriftSummary,
    empty_context: ar.ReflectionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNUAL_REFLECTION_ENABLED", "true")
    reflections_dir.mkdir(parents=True, exist_ok=True)
    (reflections_dir / "2026.md").write_text("already exists", encoding="utf-8")

    calls = {"n": 0}

    def llm_call(system: str, user: str) -> str:
        calls["n"] += 1
        return _ok_essay()

    result = ar.run_one_pass(
        llm_call=llm_call,
        year=2026,
        reflections_dir=reflections_dir,
        drift=empty_drift,
        context=empty_context,
    )
    assert result.status == "skipped_recent"
    assert result.year == 2026
    assert calls["n"] == 0


def test_run_one_pass_retries_on_phenomenal_violation(
    reflections_dir: Path,
    empty_drift: DriftSummary,
    empty_context: ar.ReflectionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNUAL_REFLECTION_ENABLED", "true")
    seq = iter([_phenomenal_essay(), _ok_essay()])

    def llm_call(system: str, user: str) -> str:
        return next(seq)

    result = ar.run_one_pass(
        llm_call=llm_call,
        year=2026,
        reflections_dir=reflections_dir,
        drift=empty_drift,
        context=empty_context,
        max_retries=3,
    )
    assert result.status == "wrote_essay"
    assert result.attempts == 2


def test_run_one_pass_gives_up_after_max_retries(
    reflections_dir: Path,
    empty_drift: DriftSummary,
    empty_context: ar.ReflectionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNUAL_REFLECTION_ENABLED", "true")

    def llm_call(system: str, user: str) -> str:
        return _phenomenal_essay()

    result = ar.run_one_pass(
        llm_call=llm_call,
        year=2026,
        reflections_dir=reflections_dir,
        drift=empty_drift,
        context=empty_context,
        max_retries=3,
    )
    assert result.status == "skipped_composer_failed"
    assert result.attempts == 3
    assert not (reflections_dir / "2026.md").exists()


def test_run_one_pass_handles_llm_exception(
    reflections_dir: Path,
    empty_drift: DriftSummary,
    empty_context: ar.ReflectionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNUAL_REFLECTION_ENABLED", "true")

    def llm_call(system: str, user: str) -> str:
        raise RuntimeError("LLM boom")

    result = ar.run_one_pass(
        llm_call=llm_call,
        year=2026,
        reflections_dir=reflections_dir,
        drift=empty_drift,
        context=empty_context,
    )
    assert result.status == "skipped_unexpected_error"
    assert "LLM boom" in result.failure_reason
    assert not (reflections_dir / "2026.md").exists()


def test_run_one_pass_default_year_uses_now(
    reflections_dir: Path,
    empty_drift: DriftSummary,
    empty_context: ar.ReflectionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNUAL_REFLECTION_ENABLED", "true")

    def llm_call(system: str, user: str) -> str:
        return _ok_essay()

    fixed_now = datetime(2027, 3, 15, tzinfo=timezone.utc)
    result = ar.run_one_pass(
        llm_call=llm_call,
        reflections_dir=reflections_dir,
        drift=empty_drift,
        context=empty_context,
        now=fixed_now,
    )
    assert result.status == "wrote_essay"
    assert result.year == 2027
    assert (reflections_dir / "2027.md").exists()


def test_min_interval_days_clamped_low() -> None:
    # The env override has a 7-day floor.
    import os
    os.environ["ANNUAL_REFLECTION_MIN_INTERVAL_DAYS"] = "1"
    try:
        assert ar._min_interval_days() == 7
    finally:
        os.environ.pop("ANNUAL_REFLECTION_MIN_INTERVAL_DAYS")


def test_user_prompt_includes_drift_and_chapters() -> None:
    drift = DriftSummary(
        window_days=365,
        n_events=5,
        by_kind={"soul_edit": 3, "tier3_amendment": 2},
        by_actor={"operator": 5},
        first_seen="2025-06-01T00:00:00+00:00",
        last_seen="2026-05-01T00:00:00+00:00",
    )
    chapters = ["chapter-1 body", "chapter-2 body"]
    user_prompt = ar._build_user_prompt(2026, drift, chapters, "lesson-summary")
    assert "2026" in user_prompt
    assert "5" in user_prompt
    assert "soul_edit" in user_prompt
    assert "chapter-1 body" in user_prompt
    assert "lesson-summary" in user_prompt
