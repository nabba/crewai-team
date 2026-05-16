"""Tests for the /browse Signal slash command handler."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("crewai")
from app.agents.commander.commands import _handle_browse_command  # noqa: E402
from app.browse import store  # noqa: E402
from app.browse.models import BrowseEvent  # noqa: E402


def _ev(domain: str, *, title: str | None = "T",
        ts: str = "2026-05-15T10:00:00+00:00") -> BrowseEvent:
    return BrowseEvent(
        visit_ts=ts, domain=domain, path="/", title=title,
        browser="chrome", profile=None,
    )


def test_handler_returns_none_for_unrelated_input(_reset_browse_state: Path) -> None:
    """The handler must return None when the input isn't a /browse
    command, so the dispatcher in try_command falls through."""
    assert _handle_browse_command("hello") is None
    assert _handle_browse_command("/person list") is None


def test_help_subcommand(_reset_browse_state: Path) -> None:
    out = _handle_browse_command("/browse help")
    assert out is not None
    assert "/browse" in out
    assert "categories" in out


def test_bare_command_when_disabled(
    _reset_browse_state: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "false")
    out = _handle_browse_command("/browse")
    assert out is not None
    assert "OFF" in out


def test_bare_command_with_no_events(_reset_browse_state: Path) -> None:
    out = _handle_browse_command("/browse")
    assert out is not None
    assert "no events" in out.lower()


def test_bare_command_with_events(_reset_browse_state: Path) -> None:
    store.append_events([_ev("github.com"), _ev("github.com"), _ev("wikipedia.org")])
    out = _handle_browse_command("/browse")
    assert out is not None
    assert "events" in out.lower()
    assert "chrome: 3" in out


def test_domains_subcommand(_reset_browse_state: Path) -> None:
    store.append_events([_ev("github.com"), _ev("github.com"), _ev("wiki.org")])
    out = _handle_browse_command("/browse domains")
    assert out is not None
    assert "github.com" in out
    assert "wiki.org" in out


def test_categories_subcommand_empty(_reset_browse_state: Path) -> None:
    out = _handle_browse_command("/browse categories")
    assert out is not None
    assert "no topic clusters" in out.lower()


def test_categories_subcommand_with_data(_reset_browse_state: Path) -> None:
    today = datetime.now(timezone.utc).date()
    out_dir = _reset_browse_state / "topics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{today.isoformat()}.json").write_text(json.dumps({
        "day": today.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "test",
        "topics": [
            {"label": "claude code", "title_count": 5, "sample_titles": []},
            {"label": "miscellaneous", "title_count": 1, "sample_titles": []},
        ],
    }), encoding="utf-8")
    out = _handle_browse_command("/browse categories")
    assert out is not None
    assert "claude code" in out
    assert "miscellaneous" not in out  # filtered


def test_mute_subcommand(_reset_browse_state: Path) -> None:
    out = _handle_browse_command("/browse mute custom.test")
    assert out is not None
    assert "✅" in out or "Muted" in out
    # Second call is idempotent.
    out2 = _handle_browse_command("/browse mute custom.test")
    assert out2 is not None
    assert "Already" in out2


def test_forget_domain_subcommand(_reset_browse_state: Path) -> None:
    store.append_events([_ev("trash.com"), _ev("keep.com")])
    out = _handle_browse_command("/browse forget trash.com")
    assert out is not None
    assert "1" in out
    assert store.event_counts()["total"] == 1


def test_forget_day_subcommand(_reset_browse_state: Path) -> None:
    store.append_events([_ev("x.com", ts="2026-05-15T08:00:00+00:00")])
    out = _handle_browse_command("/browse forget-day 2026-05-15")
    assert out is not None
    assert "Cleared" in out
    assert store.event_counts()["total"] == 0


def test_forget_day_bad_date(_reset_browse_state: Path) -> None:
    out = _handle_browse_command("/browse forget-day not-a-date")
    assert out is not None
    assert "YYYY-MM-DD" in out


def test_forget_all_subcommand(_reset_browse_state: Path) -> None:
    store.append_events([_ev("a.com"), _ev("b.com")])
    out = _handle_browse_command("/browse forget-all")
    assert out is not None
    assert "Removed" in out
    assert store.event_counts()["total"] == 0


def test_unknown_subcommand_returns_help(_reset_browse_state: Path) -> None:
    out = _handle_browse_command("/browse banana")
    assert out is not None
    assert "/browse" in out
