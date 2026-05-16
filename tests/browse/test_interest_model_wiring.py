"""Test that the browse topic collector yields topics into the
interest_model pipeline correctly.

PROGRAM §50 Q15.2. The new ``_browse_topics_text(lookback_days)`` is
the integration boundary between the browse subsystem and the rest of
the companion. Both sides own half of the contract:

  * topic_extraction writes structured topics/<day>.json files
  * interest_model reads them and emits (text, age_days) tuples

We test the boundary in isolation here. The end-to-end
``compile_interest_profile()`` path is covered by
``tests/companion/test_interest_model*.py``.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest


def _write_topic_file(
    base: Path, day: date, topics: list[tuple[str, int]],
) -> None:
    """Synthesise a topic_extraction output file."""
    out_dir = base / "topics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{day.isoformat()}.json"
    out_path.write_text(json.dumps({
        "day": day.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "test",
        "topics": [
            {"label": label, "title_count": count, "sample_titles": []}
            for label, count in topics
        ],
    }), encoding="utf-8")


def test_collector_yields_topics_when_files_exist(
    _reset_browse_state: Path,
) -> None:
    """When per-day topic files exist, the collector yields each
    label `min(title_count, 10)` times so frequency-weighting reflects
    visit volume."""
    from app.companion.interest_model import _browse_topics_text
    today = datetime.now(timezone.utc).date()
    _write_topic_file(_reset_browse_state, today, [
        ("claude code", 5),
        ("finnish nature", 2),
    ])
    items = list(_browse_topics_text(lookback_days=3))
    labels = [t for t, _age in items]
    # Title-count clamp: 5 yields 5, 2 yields 2.
    assert labels.count("claude code") == 5
    assert labels.count("finnish nature") == 2


def test_collector_silent_when_disabled(
    _reset_browse_state: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRIVACY PIN: master switch off → no yields, even if files exist."""
    from app.companion.interest_model import _browse_topics_text
    today = datetime.now(timezone.utc).date()
    _write_topic_file(_reset_browse_state, today, [("a topic", 3)])
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "false")
    assert list(_browse_topics_text(lookback_days=3)) == []


def test_collector_silent_when_no_files(_reset_browse_state: Path) -> None:
    """First-day-after-opt-in: no topic files yet → empty yield."""
    from app.companion.interest_model import _browse_topics_text
    assert list(_browse_topics_text(lookback_days=3)) == []


def test_collector_clamps_runaway_titles(_reset_browse_state: Path) -> None:
    """A 9999-count topic shouldn't dominate the unigram counter —
    the per-day-per-topic cap of 10 prevents one runaway from
    saturating the score."""
    from app.companion.interest_model import _browse_topics_text
    today = datetime.now(timezone.utc).date()
    _write_topic_file(_reset_browse_state, today, [("runaway", 9999)])
    items = list(_browse_topics_text(lookback_days=1))
    assert len(items) == 10


def test_collector_ages_by_day(_reset_browse_state: Path) -> None:
    from app.companion.interest_model import _browse_topics_text
    today = datetime.now(timezone.utc).date()
    _write_topic_file(_reset_browse_state, today, [("today-topic", 1)])
    _write_topic_file(_reset_browse_state, today - timedelta(days=2), [
        ("two-day-old", 1),
    ])
    items = list(_browse_topics_text(lookback_days=5))
    ages = {label: age for label, age in items}
    assert ages["today-topic"] == 0.0
    assert ages["two-day-old"] == 2.0


def test_collector_skips_empty_labels(_reset_browse_state: Path) -> None:
    from app.companion.interest_model import _browse_topics_text
    today = datetime.now(timezone.utc).date()
    _write_topic_file(_reset_browse_state, today, [
        ("", 5),
        ("real", 2),
    ])
    items = list(_browse_topics_text(lookback_days=1))
    assert {t for t, _ in items} == {"real"}
