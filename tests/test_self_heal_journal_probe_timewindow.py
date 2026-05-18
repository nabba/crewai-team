"""Regression: Self-heal journal probe must time-window its count.

Pre-fix shape (the operator-reported false-WARN, 2026-05-10):

  Internal  WARN  Self-heal journal  50 recent errors, top: coding:BadRequestError×16

  The probe called ``get_recent_errors(50)`` which returns the last
  50 entries regardless of timestamp.  On the affected install, those
  50 entries spanned 2026-04-02 → 2026-04-28 — NO errors had been
  recorded in the past 24 hours.  But the journal-FIFO is sticky, so
  any install with historical errors stayed in WARN forever.

Post-fix:
  • Pull up to 200 entries (generous slice for busy days)
  • Filter by ``ts`` against a 24 h window
  • Status logic:
      - 0 entries in 24h, 0 historical    → OK "no recent errors"
      - 0 entries in 24h, N historical    → OK "no errors in last 24h
                                            (N historical in journal)"
      - N entries in 24h                  → WARN "N errors in last 24h,
                                            top: <pattern>×<count>"
  • Top pattern is computed over the WINDOW, not all-time, so the
    operator sees what's actually firing now.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD_API = (
    _REPO_ROOT / "app" / "control_plane" / "dashboard_api.py"
)


@pytest.fixture(scope="module")
def src() -> str:
    return _DASHBOARD_API.read_text(encoding="utf-8")


# ── Source-grep contracts ──────────────────────────────────────────


class TestProbeUsesTimeWindow:

    def test_probe_imports_datetime_for_filtering(self, src: str) -> None:
        import re
        m = re.search(
            r"def _self_heal\(\):.*?checks\.append\(_probe\(\"Self-heal journal\"",
            src, re.DOTALL,
        )
        assert m is not None, "could not slice _self_heal()"
        body = m.group(0)
        assert "from datetime import" in body, (
            "probe must do time-window filtering — needs datetime import"
        )
        assert "timedelta" in body, (
            "probe must use a timedelta to define the window"
        )

    def test_probe_pulls_more_than_50_entries(self, src: str) -> None:
        """Pulling 50 risks truncating the time window on busy days.
        At least 100 to give the cut some headroom."""
        import re
        m = re.search(
            r"def _self_heal\(\):.*?checks\.append\(_probe\(\"Self-heal journal\"",
            src, re.DOTALL,
        )
        body = m.group(0) if m else ""
        # Strip docstrings — the docstring intentionally mentions the
        # old `get_recent_errors(50)` for context; only LIVE code matters.
        body_no_doc = re.sub(r'"""[\s\S]*?"""', "", body)
        m2 = re.search(r"get_recent_errors\((\d+)\)", body_no_doc)
        assert m2 is not None, "must call get_recent_errors with a count"
        n = int(m2.group(1))
        assert n >= 100, (
            f"must pull ≥100 entries to give time-window cut headroom; "
            f"got {n}"
        )


# ── Functional tests ──────────────────────────────────────────────


def _make_entry(hours_ago: float, crew: str = "research", err: str = "RuntimeError") -> dict:
    """Build a journal entry with ts set N hours in the past."""
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {
        "ts": ts.isoformat(),
        "crew": crew,
        "error_type": err,
        "error_msg": "test error",
    }


@pytest.fixture
def patched_journal(monkeypatch: pytest.MonkeyPatch):
    """Returns a setter; tests inject the entries they want the probe
    to see."""
    entries: list[dict] = []

    def _set(new_entries: list[dict]) -> None:
        entries.clear()
        entries.extend(new_entries)

    monkeypatch.setattr(
        "app.healing.error_diagnosis.get_recent_errors",
        lambda n=10: entries[-n:] if n else list(entries),
    )
    return _set


class TestEmptyJournal:

    def test_empty_journal_returns_ok(self, patched_journal) -> None:
        from app.control_plane.dashboard_routes_ops_misc import system_status

        patched_journal([])
        result = system_status()
        check = next(
            c for c in result["checks"] if c["name"] == "Self-heal journal"
        )
        assert check["status"] == "ok"
        assert "no recent errors" in check["message"]


class TestHistoricalOnlyJournal:
    """The original false-WARN scenario: journal has entries, but none
    are recent."""

    def test_historical_only_returns_ok_with_count(
        self, patched_journal,
    ) -> None:
        from app.control_plane.dashboard_routes_ops_misc import system_status

        # All entries 30+ days old.
        patched_journal([
            _make_entry(hours_ago=30 * 24, err=f"E{i}")
            for i in range(50)
        ])
        result = system_status()
        check = next(
            c for c in result["checks"] if c["name"] == "Self-heal journal"
        )
        assert check["status"] == "ok", (
            f"historical-only journal must be OK, not WARN; got {check}"
        )
        assert "last 24h" in check["message"]
        assert "50 historical" in check["message"]


class TestRecentEntriesTriggerWarn:

    def test_5_recent_entries_warns(self, patched_journal) -> None:
        from app.control_plane.dashboard_routes_ops_misc import system_status

        # 5 fresh + 50 historical → only the fresh count.
        patched_journal(
            [_make_entry(hours_ago=2.0, crew="coding", err="BadRequestError")
             for _ in range(5)]
            + [_make_entry(hours_ago=30 * 24, err=f"E{i}") for i in range(50)]
        )
        result = system_status()
        check = next(
            c for c in result["checks"] if c["name"] == "Self-heal journal"
        )
        assert check["status"] == "warn"
        assert "5 errors in last 24h" in check["message"]


class TestTopPatternIsWindowedNotAllTime:
    """The top pattern in the message must be computed over the
    time-windowed slice, not the whole journal."""

    def test_top_reflects_recent_only(self, patched_journal) -> None:
        from app.control_plane.dashboard_routes_ops_misc import system_status

        # 100 historical of TYPE_OLD, 3 fresh of TYPE_NEW.
        patched_journal(
            [_make_entry(hours_ago=30 * 24, crew="old", err="TYPE_OLD")
             for _ in range(100)]
            + [_make_entry(hours_ago=2.0, crew="new", err="TYPE_NEW")
               for _ in range(3)]
        )
        result = system_status()
        check = next(
            c for c in result["checks"] if c["name"] == "Self-heal journal"
        )
        assert check["status"] == "warn"
        # Must show the fresh pattern, not the dominant historical one.
        assert "new:TYPE_NEW" in check["message"], (
            f"top pattern must be windowed; got {check}"
        )
        assert "TYPE_OLD" not in check["message"]


class TestMalformedTsHandled:
    """Entries with broken/missing timestamps must not crash the
    probe — they should be excluded from the time window silently."""

    def test_no_ts_entry_excluded_from_window(self, patched_journal) -> None:
        from app.control_plane.dashboard_routes_ops_misc import system_status

        bad_entry = {"crew": "x", "error_type": "Y"}  # no ts
        patched_journal([bad_entry])
        result = system_status()
        check = next(
            c for c in result["checks"] if c["name"] == "Self-heal journal"
        )
        # Bad entry not in window → counts toward "historical" total.
        assert check["status"] == "ok"
        assert "1 historical" in check["message"]
