"""Test the daily briefing's browse-themes section."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path


def _write_topic_file(
    base: Path, day: date, topics: list[tuple[str, int]],
) -> None:
    out_dir = base / "topics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{day.isoformat()}.json").write_text(json.dumps({
        "day": day.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "test",
        "topics": [
            {"label": label, "title_count": count, "sample_titles": []}
            for label, count in topics
        ],
    }), encoding="utf-8")


def test_section_empty_when_disabled(
    _reset_browse_state: Path, monkeypatch,
) -> None:
    """PRIVACY PIN: master switch off → briefing section is empty."""
    from app.life_companion.daily_briefing import _gather_browse_themes
    today = datetime.now(timezone.utc).date()
    _write_topic_file(_reset_browse_state, today, [("a topic", 1)])
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "false")
    assert _gather_browse_themes() == []


def test_section_empty_when_no_topic_files(_reset_browse_state: Path) -> None:
    from app.life_companion.daily_briefing import _gather_browse_themes
    assert _gather_browse_themes() == []


def test_section_aggregates_top_topics(_reset_browse_state: Path) -> None:
    from app.life_companion.daily_briefing import _gather_browse_themes
    today = datetime.now(timezone.utc).date()
    _write_topic_file(_reset_browse_state, today, [
        ("claude code", 10),
        ("finnish nature", 4),
        ("miscellaneous", 99),  # should be filtered
    ])
    out = _gather_browse_themes(n=5)
    # miscellaneous filtered, others sorted by count desc.
    assert out == [
        "  • claude code (10)",
        "  • finnish nature (4)",
    ]


def test_section_caps_to_n(_reset_browse_state: Path) -> None:
    from app.life_companion.daily_briefing import _gather_browse_themes
    today = datetime.now(timezone.utc).date()
    many = [(f"topic-{i}", 10 - i) for i in range(10)]
    _write_topic_file(_reset_browse_state, today, many)
    out = _gather_browse_themes(n=3)
    assert len(out) == 3
