"""Tests for ``app.life_companion.personalized_digest`` (Phase D #5)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.life_companion import personalized_digest as pd
    from app.life_companion import _common

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(pd, "background_enabled", lambda: True)
    monkeypatch.setattr(pd, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(pd, "_FEEDS_PATH", tmp_path / "personalized_feeds.json")

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pd, "send_signal_alert",
        lambda body, tag=None, **kw: sent.append((body, tag or "")),
    )
    yield tmp_path, sent


# ── Pure parser tests ────────────────────────────────────────────────────


def test_parse_rss():
    from app.life_companion.personalized_digest import _parse_feed
    xml = """
    <rss version="2.0"><channel>
      <item><title>One</title><link>https://example.com/1</link>
            <description>summary one</description></item>
      <item><title>Two</title><link>https://example.com/2</link>
            <description>summary two</description></item>
    </channel></rss>
    """
    items = _parse_feed(xml, max_items=10)
    assert len(items) == 2
    assert items[0]["title"] == "One"
    assert items[0]["link"].startswith("https://example.com/1")


def test_parse_atom():
    from app.life_companion.personalized_digest import _parse_feed
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Atom Entry</title>
        <link href="https://arxiv.org/abs/2505.12345"/>
        <summary>Abstract.</summary>
        <published>2026-05-09T12:00:00Z</published>
      </entry>
    </feed>
    """
    items = _parse_feed(xml, max_items=5)
    assert len(items) == 1
    assert "arxiv" in items[0]["link"]


def test_parse_handles_garbage():
    from app.life_companion.personalized_digest import _parse_feed
    assert _parse_feed("not xml at all", max_items=5) == []
    assert _parse_feed("", max_items=5) == []


# ── ISO-week + cadence ───────────────────────────────────────────────────


def test_iso_week_str():
    from app.life_companion.personalized_digest import _iso_week_str
    dt = datetime(2026, 5, 9)
    assert _iso_week_str(dt).startswith("2026-W")


def test_run_skips_outside_target_window(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import personalized_digest as pd

    # Force "wrong day" — Tuesday-not-Friday.
    class _Tue:
        @classmethod
        def __call__(cls):
            return cls()
        weekday = lambda self=None: 1   # Tuesday
        hour = 10

    fake_now = datetime(2026, 5, 5, 10, 0).astimezone()
    monkeypatch.setattr(pd, "_now_local", lambda: fake_now)
    summary = pd.run()
    assert summary["sent"] is False


def test_run_in_window_sends_digest(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import personalized_digest as pd

    # Force "right day" — Friday at 10:00.
    fake_now = datetime(2026, 5, 8, 10, 0).astimezone()
    assert fake_now.weekday() == 4  # Friday
    monkeypatch.setattr(pd, "_now_local", lambda: fake_now)

    monkeypatch.setattr(pd, "_load_feeds_config", lambda: {
        "rss": [], "github_username": "", "arxiv_authors": [], "ventures": [],
    })
    monkeypatch.setattr(pd, "_gather_rss",
                        lambda feeds, seen: [
                            {"source": "rss", "title": "RSS Item",
                             "link": "https://x.com/1", "summary": "sum",
                             "published": ""},
                        ])
    monkeypatch.setattr(pd, "_gather_github_user_events",
                        lambda u, s: [])
    monkeypatch.setattr(pd, "_gather_arxiv_by_author",
                        lambda authors, s: [])
    monkeypatch.setattr(pd, "_gather_venture_news",
                        lambda v, s: [])

    summary = pd.run()
    assert summary["sent"] is True
    assert any("Weekly personalized digest" in body for body, _ in sent)


def test_dedup_by_iso_week(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import personalized_digest as pd

    fake_now = datetime(2026, 5, 8, 10, 0).astimezone()
    monkeypatch.setattr(pd, "_now_local", lambda: fake_now)
    monkeypatch.setattr(pd, "_load_feeds_config", lambda: {})
    monkeypatch.setattr(pd, "_gather_rss",
                        lambda feeds, seen: [{"source": "rss", "title": "x",
                                              "link": "https://y/1", "summary": "",
                                              "published": ""}])
    monkeypatch.setattr(pd, "_gather_github_user_events",
                        lambda u, s: [])
    monkeypatch.setattr(pd, "_gather_arxiv_by_author",
                        lambda a, s: [])
    monkeypatch.setattr(pd, "_gather_venture_news",
                        lambda v, s: [])

    pd.run()
    initial = len(sent)

    # Reset cadence; same ISO week — should NOT send again.
    state_path = tmp_path / "lc" / "personalized_digest.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    pd.run()
    assert len(sent) == initial


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("PERSONALIZED_DIGEST_ENABLED", "0")
    from app.life_companion import personalized_digest as pd
    summary = pd.run()
    assert summary["ran"] is False
