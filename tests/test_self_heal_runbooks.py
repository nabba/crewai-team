"""Tests for app.healing.runbooks — the anomaly-driven runbook
dispatcher added in the May 2026 self-healing pass (track B).

Reconstructed from orphan .pyc bytecode + PROGRAM.md §14 spec on
2026-05-06. Originals were never committed; this file ships with the
runbooks source so the safety claim in §14 is reproducible.

Coverage shape mirrors the original (per .pyc test-name extraction):
  - TestEnvGate / TestSeverityGate / TestPatternMatch /
    TestRunbookEnabledFlag / TestRecurrenceGate / TestSuccessRateGate /
    TestConcurrencyCap — one per gate of the dispatcher
  - TestHappyPath        end-to-end with a registered handler
  - TestLogOnly          reference handler is auto-registered
  - TestStatsHelpers     recent-history + success-rate helpers
  - TestRegistration     register + unregister + pattern compile
"""
from __future__ import annotations

import re
import time
from unittest.mock import patch

import pytest

from app.healing import runbooks


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_runbooks(tmp_path, monkeypatch):
    """Reset module state and redirect stats/settings paths to tmp.

    log_only is auto-registered at import; we keep it but reset state
    so each test starts fresh.
    """
    settings = tmp_path / "runbook_settings.json"
    stats = tmp_path / "runbook_stats.json"
    monkeypatch.setattr(runbooks, "_SETTINGS_PATH", settings)
    monkeypatch.setattr(runbooks, "_STATS_PATH", stats)
    monkeypatch.setattr(runbooks, "_RUNBOOK_DIR", tmp_path)

    # Snapshot + restore registry around each test (preserve log_only).
    saved = dict(runbooks._REGISTERED_RUNBOOKS)
    runbooks._active_runbooks.clear()
    yield
    runbooks._REGISTERED_RUNBOOKS.clear()
    runbooks._REGISTERED_RUNBOOKS.update(saved)
    runbooks._active_runbooks.clear()


@pytest.fixture
def silenced_audit(monkeypatch):
    events = []

    def _audit(action, **detail):
        events.append((action, detail))

    monkeypatch.setattr(runbooks, "_runbook_audit", _audit)
    return events


@pytest.fixture
def mock_recurrence(monkeypatch):
    def _set(value):
        monkeypatch.setattr(
            runbooks, "_signature_recurrence", lambda _sig: value,
        )

    return _set


# ── Gate 1: env flag ───────────────────────────────────────────────────────


class TestEnvGate:
    def test_disabled_returns_none(self, isolated_runbooks, monkeypatch):
        monkeypatch.delenv("ERROR_RUNBOOKS_ENABLED", raising=False)
        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "x", "severity": "warning"},
        )
        assert result is None

    def test_runbooks_enabled_function(self, monkeypatch):
        monkeypatch.delenv("ERROR_RUNBOOKS_ENABLED", raising=False)
        assert runbooks.runbooks_enabled() is False
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        assert runbooks.runbooks_enabled() is True


# ── Gate 2: severity ───────────────────────────────────────────────────────


class TestSeverityGate:
    def test_severity_info_skipped(
        self, isolated_runbooks, monkeypatch, silenced_audit,
    ):
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "x", "severity": "info"},
        )
        assert result is None
        skipped = [e for e in silenced_audit if e[0] == "dispatch.skipped"]
        assert any(d.get("reason") == "severity_info" for _, d in skipped)


# ── Gate 3: pattern match ──────────────────────────────────────────────────


class TestPatternMatch:
    def test_no_match_when_no_runbook_registered(
        self, isolated_runbooks, monkeypatch, silenced_audit,
    ):
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        # Clear registry — log_only is removed for this test.
        runbooks._REGISTERED_RUNBOOKS.clear()
        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "anything", "severity": "warning"},
        )
        assert result is None
        skipped = [e for e in silenced_audit if e[0] == "dispatch.skipped"]
        assert any(d.get("reason") == "no_pattern_match" for _, d in skipped)

    def test_match_first_registered_wins(self, isolated_runbooks):
        runbooks._REGISTERED_RUNBOOKS.clear()

        runbooks.register_runbook(
            "first", r"target",
            lambda a: runbooks.RunbookResult(name="first", success=True),
        )
        runbooks.register_runbook(
            "second", r"target",
            lambda a: runbooks.RunbookResult(name="second", success=True),
        )

        entry = runbooks._match_runbook("target pattern fired")
        assert entry is not None
        assert entry.name == "first"


# ── Gate 4: per-runbook enabled ────────────────────────────────────────────


class TestRunbookEnabledFlag:
    def test_disabled_runbook_skipped(
        self, isolated_runbooks, monkeypatch, silenced_audit, mock_recurrence,
    ):
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        runbooks._REGISTERED_RUNBOOKS.clear()
        runbooks.register_runbook(
            "rb1", r".*",
            lambda a: runbooks.RunbookResult(name="rb1", success=True),
        )
        # Settings says rb1 is disabled.
        runbooks._SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        runbooks._SETTINGS_PATH.write_text(
            '{"runbooks": {"rb1": {"enabled": false, "min_recurrence": 1}}}',
        )
        mock_recurrence(99)

        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "x", "severity": "warning"},
        )
        assert result is None
        skipped = [e for e in silenced_audit if e[0] == "dispatch.skipped"]
        assert any(d.get("reason") == "runbook_disabled" for _, d in skipped)

    def test_runbook_missing_from_settings_defaults_off(
        self, isolated_runbooks, monkeypatch, silenced_audit, mock_recurrence,
    ):
        """Unknown runbook name in settings → enabled defaults to False."""
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        runbooks._REGISTERED_RUNBOOKS.clear()
        runbooks.register_runbook(
            "rb1", r".*",
            lambda a: runbooks.RunbookResult(name="rb1", success=True),
        )
        # No settings file written — every runbook unknown.
        mock_recurrence(99)

        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "x", "severity": "warning"},
        )
        assert result is None


# ── Gate 5: recurrence ─────────────────────────────────────────────────────


class TestRecurrenceGate:
    def _setup(self, monkeypatch):
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        runbooks._REGISTERED_RUNBOOKS.clear()
        runbooks.register_runbook(
            "rb1", r".*",
            lambda a: runbooks.RunbookResult(name="rb1", success=True),
        )
        runbooks._SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        runbooks._SETTINGS_PATH.write_text(
            '{"runbooks": {"rb1": {"enabled": true, "min_recurrence": 5}}}',
        )

    def test_below_threshold_skipped(
        self, isolated_runbooks, monkeypatch, silenced_audit, mock_recurrence,
    ):
        self._setup(monkeypatch)
        mock_recurrence(2)
        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "x", "severity": "warning"},
        )
        assert result is None
        skipped = [e for e in silenced_audit if e[0] == "dispatch.skipped"]
        assert any(
            d.get("reason") == "below_recurrence_threshold" for _, d in skipped
        )

    def test_at_threshold_proceeds(
        self, isolated_runbooks, monkeypatch, silenced_audit, mock_recurrence,
    ):
        self._setup(monkeypatch)
        mock_recurrence(5)
        # Block actual thread spawn so the test stays deterministic.
        with patch.object(runbooks, "threading") as fake_threading:
            result = runbooks.maybe_run_runbook(
                {"pattern_signature": "x", "severity": "warning"},
            )
            assert fake_threading.Thread.called
        assert result is not None
        assert result.name == "rb1"


# ── Gate 6: success rate ───────────────────────────────────────────────────


class TestSuccessRateGate:
    def test_low_success_rate_skipped(
        self, isolated_runbooks, monkeypatch, silenced_audit, mock_recurrence,
    ):
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        runbooks._REGISTERED_RUNBOOKS.clear()
        runbooks.register_runbook(
            "rb1", r".*",
            lambda a: runbooks.RunbookResult(name="rb1", success=True),
        )
        runbooks._SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        runbooks._SETTINGS_PATH.write_text(
            '{"runbooks": {"rb1": {"enabled": true, "min_recurrence": 1}}}',
        )
        # Seed stats with 8/10 failures → success rate 0.2 < 0.5.
        runbooks._STATS_PATH.write_text(
            '{"runbooks": {"rb1": {"recent": '
            + '[{"success": true}, {"success": true}, '
            + '{"success": false}, {"success": false}, '
            + '{"success": false}, {"success": false}, '
            + '{"success": false}, {"success": false}, '
            + '{"success": false}, {"success": false}]}}}',
        )
        mock_recurrence(99)

        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "x", "severity": "warning"},
        )
        assert result is None
        skipped = [e for e in silenced_audit if e[0] == "dispatch.skipped"]
        assert any(
            d.get("reason") == "recent_success_rate_low" for _, d in skipped
        )

    def test_no_history_treated_as_passing(self, isolated_runbooks):
        # First run gets a chance — empty stats means 'allow'.
        assert runbooks._runbook_success_rate("nonexistent") == 1.0


# ── Gate 7: concurrency ────────────────────────────────────────────────────


class TestConcurrencyCap:
    def test_cap_blocks_second_dispatch(
        self, isolated_runbooks, monkeypatch, silenced_audit, mock_recurrence,
    ):
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        runbooks._REGISTERED_RUNBOOKS.clear()
        runbooks.register_runbook(
            "rb1", r".*",
            lambda a: runbooks.RunbookResult(name="rb1", success=True),
        )
        runbooks._SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        runbooks._SETTINGS_PATH.write_text(
            '{"runbooks": {"rb1": {"enabled": true, "min_recurrence": 1}}}',
        )
        mock_recurrence(99)

        # Pre-fill _active_runbooks to simulate one already in flight.
        runbooks._active_runbooks.add("other")

        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "x", "severity": "warning"},
        )
        assert result is None
        skipped = [e for e in silenced_audit if e[0] == "dispatch.skipped"]
        assert any(d.get("reason") == "concurrency_cap" for _, d in skipped)


# ── Happy path ─────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_handler_executes_and_records_outcome(
        self, isolated_runbooks, monkeypatch, mock_recurrence,
    ):
        monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")
        runbooks._REGISTERED_RUNBOOKS.clear()

        called = {"n": 0}

        def handler(anomaly):
            called["n"] += 1
            return runbooks.RunbookResult(
                name="rb1", success=True, detail="ran",
            )

        runbooks.register_runbook("rb1", r".*", handler)
        runbooks._SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        runbooks._SETTINGS_PATH.write_text(
            '{"runbooks": {"rb1": {"enabled": true, "min_recurrence": 1}}}',
        )
        mock_recurrence(5)

        result = runbooks.maybe_run_runbook(
            {"pattern_signature": "x", "severity": "warning"},
        )
        assert result is not None
        assert result.name == "rb1"

        # Wait for the daemon thread to fully exit (handler + outcome
        # write + audit + _active_runbooks.discard) so the fixture
        # teardown doesn't undo monkeypatch mid-write.
        deadline = time.monotonic() + 3.0
        while ("rb1" in runbooks._active_runbooks
               and time.monotonic() < deadline):
            time.sleep(0.01)
        assert called["n"] == 1
        assert "rb1" not in runbooks._active_runbooks


# ── log_only reference handler ─────────────────────────────────────────────


class TestLogOnly:
    def test_log_only_returns_success(self):
        result = runbooks._runbook_log_only(
            {"pattern_signature": "x", "severity": "warning"},
        )
        assert result.name == "log_only"
        assert result.success is True

    def test_log_only_registered_at_import(self):
        # log_only is wired by import — match should hit it for any sample.
        # (No env required for this — purely registry inspection.)
        assert "log_only" in runbooks._REGISTERED_RUNBOOKS


# ── Stats helpers ──────────────────────────────────────────────────────────


class TestStatsHelpers:
    def test_record_outcome_appends_to_recent(self, isolated_runbooks):
        runbooks._record_runbook_outcome("rb1", True)
        runbooks._record_runbook_outcome("rb1", False)
        stats = runbooks._load_runbook_stats()
        recent = stats["rb1"]["recent"]
        assert len(recent) == 2
        assert recent[0]["success"] is True
        assert recent[1]["success"] is False

    def test_success_rate_empty_history(self, isolated_runbooks):
        assert runbooks._runbook_success_rate("never_run") == 1.0

    def test_success_rate_mixed(self, isolated_runbooks):
        runbooks._record_runbook_outcome("rb1", True)
        runbooks._record_runbook_outcome("rb1", True)
        runbooks._record_runbook_outcome("rb1", False)
        rate = runbooks._runbook_success_rate("rb1")
        assert abs(rate - 2 / 3) < 1e-6


# ── Registration ───────────────────────────────────────────────────────────


class TestRegistration:
    def test_register_and_unregister(self, isolated_runbooks):
        runbooks._REGISTERED_RUNBOOKS.clear()

        def h(a):
            return runbooks.RunbookResult(name="rb1", success=True)

        runbooks.register_runbook("rb1", r"foo", h)
        assert "rb1" in runbooks._REGISTERED_RUNBOOKS
        runbooks.unregister_runbook("rb1")
        assert "rb1" not in runbooks._REGISTERED_RUNBOOKS

    def test_pattern_string_compiles(self, isolated_runbooks):
        runbooks._REGISTERED_RUNBOOKS.clear()
        runbooks.register_runbook(
            "rb1", r"foo.*bar",
            lambda a: runbooks.RunbookResult(name="rb1", success=True),
        )
        entry = runbooks._REGISTERED_RUNBOOKS["rb1"]
        assert isinstance(entry.pattern, re.Pattern)

    def test_pattern_compiled_regex_accepted(self, isolated_runbooks):
        runbooks._REGISTERED_RUNBOOKS.clear()
        compiled = re.compile(r"baz")
        runbooks.register_runbook(
            "rb1", compiled,
            lambda a: runbooks.RunbookResult(name="rb1", success=True),
        )
        entry = runbooks._REGISTERED_RUNBOOKS["rb1"]
        assert entry.pattern is compiled
