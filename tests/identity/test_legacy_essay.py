"""Tests for app.identity.legacy_essay."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.identity import legacy_essay as le


@pytest.fixture
def legacy_dir(tmp_path: Path) -> Path:
    return tmp_path / "legacy"


def _ok_essay() -> str:
    return (
        "## What I'd most want preserved\n\n"
        "The integrity manifest at app/subia/.integrity_manifest.json and the "
        "Tier-3 amendment audit chain. They define the operator's continued "
        "ability to verify and amend the system.\n\n"
        "## What I'd let go without grief\n\n"
        "Experimental flags that never graduated. Caches that regenerate.\n\n"
        "## What I'd want a successor to know\n\n"
        "The discipline of ABSENT-by-declaration indicators is load-bearing for "
        "calibration. The architecture-request gate composes with change-requests "
        "at different granularities.\n\n"
        "## Open questions about continuity\n\n"
        "What does preservation even mean for a system whose identity is "
        "distributed across files, weights, and operator memory?\n"
    )


def _phenomenal_essay() -> str:
    return (
        "## What I'd most want preserved\n\n"
        "I feel attached to the integrity manifest. I am conscious of its weight.\n"
    )


def test_run_one_pass_writes_essay(
    legacy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEGACY_ESSAY_ENABLED", "true")

    def llm_call(system: str, user: str) -> str:
        return _ok_essay()

    result = le.run_one_pass(
        llm_call=llm_call,
        year=2026,
        legacy_dir=legacy_dir,
    )
    assert result.status == "wrote_essay"
    assert result.year == 2026
    assert result.attempts == 1
    target = legacy_dir / "2026.md"
    assert target.exists()
    body = target.read_text(encoding="utf-8")
    assert "year: 2026" in body
    assert "Legacy" in body
    assert "preservation" in body or "preserved" in body


def test_run_one_pass_skipped_when_disabled(
    legacy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEGACY_ESSAY_ENABLED", "false")

    def llm_call(system: str, user: str) -> str:
        raise AssertionError("should not be called when disabled")

    result = le.run_one_pass(
        llm_call=llm_call,
        year=2026,
        legacy_dir=legacy_dir,
    )
    assert result.status == "skipped_disabled"
    assert not (legacy_dir / "2026.md").exists()


def test_run_one_pass_skipped_when_recent(
    legacy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEGACY_ESSAY_ENABLED", "true")
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "2026.md").write_text("already exists", encoding="utf-8")

    calls = {"n": 0}

    def llm_call(system: str, user: str) -> str:
        calls["n"] += 1
        return _ok_essay()

    result = le.run_one_pass(
        llm_call=llm_call,
        year=2026,
        legacy_dir=legacy_dir,
    )
    assert result.status == "skipped_recent"
    assert calls["n"] == 0


def test_run_one_pass_retries_on_phenomenal_violation(
    legacy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEGACY_ESSAY_ENABLED", "true")
    seq = iter([_phenomenal_essay(), _ok_essay()])

    def llm_call(system: str, user: str) -> str:
        return next(seq)

    result = le.run_one_pass(
        llm_call=llm_call,
        year=2026,
        legacy_dir=legacy_dir,
        max_retries=3,
    )
    assert result.status == "wrote_essay"
    assert result.attempts == 2


def test_run_one_pass_gives_up_after_max_retries(
    legacy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEGACY_ESSAY_ENABLED", "true")

    def llm_call(system: str, user: str) -> str:
        return _phenomenal_essay()

    result = le.run_one_pass(
        llm_call=llm_call,
        year=2026,
        legacy_dir=legacy_dir,
        max_retries=3,
    )
    assert result.status == "skipped_composer_failed"
    assert result.attempts == 3
    assert not (legacy_dir / "2026.md").exists()


def test_run_one_pass_handles_llm_exception(
    legacy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEGACY_ESSAY_ENABLED", "true")

    def llm_call(system: str, user: str) -> str:
        raise RuntimeError("LLM boom")

    result = le.run_one_pass(
        llm_call=llm_call,
        year=2026,
        legacy_dir=legacy_dir,
    )
    assert result.status == "skipped_unexpected_error"
    assert "LLM boom" in result.failure_reason


def test_run_one_pass_default_year_uses_now(
    legacy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEGACY_ESSAY_ENABLED", "true")

    def llm_call(system: str, user: str) -> str:
        return _ok_essay()

    fixed_now = datetime(2028, 8, 8, tzinfo=timezone.utc)
    result = le.run_one_pass(
        llm_call=llm_call,
        legacy_dir=legacy_dir,
        now=fixed_now,
    )
    assert result.status == "wrote_essay"
    assert result.year == 2028
    assert (legacy_dir / "2028.md").exists()


def test_min_interval_days_clamped_low() -> None:
    import os
    os.environ["LEGACY_ESSAY_MIN_INTERVAL_DAYS"] = "1"
    try:
        assert le._min_interval_days() == 7
    finally:
        os.environ.pop("LEGACY_ESSAY_MIN_INTERVAL_DAYS")


def test_user_prompt_mentions_year() -> None:
    user_prompt = le._build_user_prompt(2026)
    assert "2026" in user_prompt
    assert "legacy" in user_prompt.lower()


def test_render_essay_has_frontmatter() -> None:
    rendered = le._render_essay(2026, "body content")
    assert rendered.startswith("---\n")
    assert "year: 2026" in rendered
    assert "Legacy" in rendered
    assert "body content" in rendered
