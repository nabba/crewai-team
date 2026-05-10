"""Q1 — Unclog the CR pipeline. Tests for items 1, 4, 5.

Item 1 — db_pool_reset min_recurrence: pure config, single
assertion that the runbook_settings.json value is 1.

Item 4 — Tier-3 amendment producer wiring + governance notifier:
  * runtime_settings flag flipped
  * governance_notifier loads + has the right state-table shape
  * notify_proposal_created sends a Signal alert + GW publish
  * run_one_pass detects state transitions

Item 5 — Goodhart advisory report:
  * report() handles missing data file (no crash)
  * report() aggregates severities from a synthetic signal log
  * "would have blocked" projection matches high-severity count
  * runtime_settings setter emits a continuity-ledger event on flip
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Item 1 — db_pool_reset min_recurrence ───────────────────────────


class TestItem1RunbookSettings:

    def test_db_pool_reset_min_recurrence_is_one(self):
        """Pool exhaustion at instance #1 is already actionable;
        previous threshold of 5 meant the runbook never triggered
        in normal operation."""
        # Read the runbook_settings.json directly to avoid pulling
        # in app.config (and therefore pydantic_settings) which is
        # a heavy dependency the gateway has but local CI may not.
        settings_path = Path(
            "/Users/andrus/BotArmy/crewai-team/workspace/self_heal/"
            "runbook_settings.json"
        )
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
        db_pool = settings["runbooks"]["db_pool_reset"]
        assert db_pool["enabled"] is True
        assert db_pool["min_recurrence"] == 1, (
            "db_pool_reset.min_recurrence must be 1 for Q1 — see "
            "docs/SELF_HEAL_V3.md and the Q1 plan"
        )


# ── Item 4 — Tier-3 amendment producer wiring ──────────────────────


class TestItem4TierFlagFlip:

    def test_runtime_settings_has_tier3_enabled_true(self):
        """The flag flip enables propose_amendment to actually run.
        Read the JSON directly to avoid pulling in app.config /
        pydantic_settings which the local CI may lack."""
        settings_path = Path(
            "/Users/andrus/BotArmy/crewai-team/workspace/runtime_settings.json"
        )
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
        assert settings.get("tier3_amendment_enabled") is True


_GATEWAY_DEPS_AVAILABLE = True
try:
    import pydantic_settings  # noqa: F401
except ImportError:
    _GATEWAY_DEPS_AVAILABLE = False


@pytest.mark.skipif(
    not _GATEWAY_DEPS_AVAILABLE,
    reason="needs gateway deps (pydantic_settings); runs in CI/docker",
)
class TestItem4GovernanceNotifier:

    @pytest.fixture(autouse=True)
    def isolated_snapshot(self, tmp_path, monkeypatch):
        """Redirect the notifier snapshot to a tmp dir so tests don't
        cross-contaminate."""
        monkeypatch.setenv("CHANGE_REQUESTS_DIR", str(tmp_path))
        from app import governance_notifier
        # Reset the daemon state so start() is a clean call if any
        # test ever exercises it.
        governance_notifier.stop()
        governance_notifier._driver_started = False
        yield

    def test_notify_states_covers_operator_relevant(self):
        """Make sure the notify table covers the states that need
        operator attention. Other states (PROPOSED, REJECTED,
        COOLDOWN_FAILED) are intentionally silent."""
        from app.governance_notifier import _NOTIFY_STATES
        for state in (
            "staged", "eligibility_failed", "cooldown_ok",
            "approved", "applied", "stable", "reverted",
        ):
            assert state in _NOTIFY_STATES, (
                f"state {state!r} must be in notify table"
            )

    def test_notify_states_salience_in_range(self):
        """Salience values must be in [0, 1] per the workspace_publish
        contract."""
        from app.governance_notifier import _NOTIFY_STATES
        for state, (salience, template) in _NOTIFY_STATES.items():
            assert 0.0 <= salience <= 1.0, (
                f"salience for {state} out of range: {salience}"
            )
            # Templates must reference id + target — those are the
            # operator-relevant fields.
            assert "{id}" in template
            assert "{target}" in template

    def test_notify_proposal_created_alerts_and_snapshots(self, tmp_path):
        from app import governance_notifier

        class _FakeState:
            def __init__(self, value):
                self.value = value

        class _FakeProposal:
            id = "p123"
            state = _FakeState("staged")
            target_path = "app/forge/audit/__init__.py"
            proposer = "self_improver"
            citation = "Test citation that meets the 30-char minimum length here."
            eligibility_failures: list[str] = []

        with patch.object(governance_notifier, "_send_signal_alert") as alert, \
             patch.object(governance_notifier, "_publish_to_workspace") as gw:
            governance_notifier.notify_proposal_created(_FakeProposal())

        alert.assert_called_once()
        gw.assert_called_once()
        # The snapshot file should now record the observed state.
        snap = json.loads(
            (tmp_path / "tier3_amendment_observed_states.json").read_text(),
        )
        assert snap == {"p123": "staged"}

    def test_run_one_pass_detects_transition(self, tmp_path):
        """Walk between two states — first pass records, second pass
        detects + alerts the change."""
        from app import governance_notifier

        class _S:
            def __init__(self, value):
                self.value = value

        class _P:
            def __init__(self, state):
                self.id = "px"
                self.state = _S(state)
                self.target_path = "app/forge/audit/__init__.py"
                self.proposer = "self_improver"
                self.citation = "x" * 50
                self.eligibility_failures: list[str] = []

        with patch.object(governance_notifier, "_list_proposals", return_value=[_P("staged")]), \
             patch.object(governance_notifier, "_try_advance_cooldown", return_value=None), \
             patch.object(governance_notifier, "_send_signal_alert") as alert1, \
             patch.object(governance_notifier, "_publish_to_workspace"):
            counters_first = governance_notifier.run_one_pass()
        # First pass: snapshot empty → state=='staged' is a "transition" to staged.
        assert counters_first["transitions_alerted"] == 1
        alert1.assert_called_once()

        # Second pass: state hasn't changed; no alert.
        with patch.object(governance_notifier, "_list_proposals", return_value=[_P("staged")]), \
             patch.object(governance_notifier, "_try_advance_cooldown", return_value=None), \
             patch.object(governance_notifier, "_send_signal_alert") as alert2, \
             patch.object(governance_notifier, "_publish_to_workspace"):
            counters_second = governance_notifier.run_one_pass()
        assert counters_second["transitions_alerted"] == 0
        alert2.assert_not_called()

        # Third pass: state moves to cooldown_ok → alert fires.
        with patch.object(governance_notifier, "_list_proposals", return_value=[_P("cooldown_ok")]), \
             patch.object(governance_notifier, "_try_advance_cooldown", return_value=None), \
             patch.object(governance_notifier, "_send_signal_alert") as alert3, \
             patch.object(governance_notifier, "_publish_to_workspace"):
            counters_third = governance_notifier.run_one_pass()
        assert counters_third["transitions_alerted"] == 1
        alert3.assert_called_once()

    def test_disabled_flag_short_circuits(self, monkeypatch):
        from app import governance_notifier
        monkeypatch.setenv("TIER3_GOVERNANCE_NOTIFIER_ENABLED", "false")
        counters = governance_notifier.run_one_pass()
        # Returned struct exists but no work done.
        assert counters["seen"] == 0
        assert counters["transitions_alerted"] == 0


# ── Item 5 — Goodhart advisory report ──────────────────────────────


@pytest.mark.skipif(
    not _GATEWAY_DEPS_AVAILABLE,
    reason="needs gateway deps (pydantic_settings); runs in CI/docker",
)
class TestItem5GoodhartReport:

    def test_report_handles_missing_signal_log(self, tmp_path, monkeypatch):
        """No data file → data_status='no_data_file', zero counts,
        never raises."""
        from app.observability import goodhart_advisory_report
        # Point at a non-existent file.
        monkeypatch.setattr(
            "app.goodhart_guard.GAMING_REPORT_PATH",
            tmp_path / "absent.json",
        )
        out = goodhart_advisory_report.report(window_days=30)
        assert out["data_status"] == "no_data_file"
        assert out["n_signals"] == 0
        assert out["would_have_blocked_in_enforcing"] == 0

    def test_report_aggregates_signals(self, tmp_path, monkeypatch):
        """Synthetic signal log → counts match input."""
        from app.observability import goodhart_advisory_report
        log_path = tmp_path / "goodhart_reports.json"
        now = time.time()
        signals = [
            {"signal_type": "trend", "severity": "high", "description": "h1",
             "metric_value": 0.9, "threshold": 0.5, "detected_at": now - 3600},
            {"signal_type": "trend", "severity": "high", "description": "h2",
             "metric_value": 0.85, "threshold": 0.5, "detected_at": now - 7200},
            {"signal_type": "spike", "severity": "medium", "description": "m1",
             "metric_value": 0.6, "threshold": 0.5, "detected_at": now - 86400},
            {"signal_type": "drift", "severity": "low", "description": "l1",
             "metric_value": 0.55, "threshold": 0.5, "detected_at": now - 100000},
            # Outside the 30-day window — must be excluded.
            {"signal_type": "trend", "severity": "high", "description": "old",
             "metric_value": 0.99, "threshold": 0.5, "detected_at": now - 60 * 86400},
        ]
        log_path.write_text(json.dumps(signals))

        monkeypatch.setattr(
            "app.goodhart_guard.GAMING_REPORT_PATH", log_path,
        )

        out = goodhart_advisory_report.report(window_days=30)
        assert out["data_status"] == "ok"
        assert out["n_signals"] == 4  # the OLD entry is excluded
        assert out["counts_by_severity"] == {"low": 1, "medium": 1, "high": 2}
        assert out["n_high_severity"] == 2
        # "Would have blocked" tracks high severity exactly.
        assert out["would_have_blocked_in_enforcing"] == 2
        # Sample list contains both high entries (descriptions trimmed).
        descriptions = [s["description"] for s in out["samples_high"]]
        assert set(descriptions) == {"h1", "h2"}

    def test_report_effective_mode_label(self):
        """Effective mode reads from runtime_settings."""
        from app.observability import goodhart_advisory_report
        out = goodhart_advisory_report.report(window_days=30)
        # In CI/dev: hard_gate_disabled=False, hard_gate_enforcing=False
        # → advisory.
        assert out["effective_mode"] in ("advisory", "enforcing", "disabled")


@pytest.mark.skipif(
    not _GATEWAY_DEPS_AVAILABLE,
    reason="needs gateway deps (pydantic_settings); runs in CI/docker",
)
class TestItem5LedgerEmission:

    def test_setter_records_continuity_event_on_flip(self, monkeypatch):
        """Flipping ``goodhart_hard_gate_enforcing`` emits a
        ``governance_ratchet`` continuity-ledger event so annual
        reflection picks up the drift."""
        from app import runtime_settings

        # Force a known prior state.
        # _emit isn't called when prior == new, so ensure a real flip.
        original = runtime_settings.get_goodhart_hard_gate_enforcing()
        try:
            recorded = []

            def _fake_record(**kwargs):
                recorded.append(kwargs)

            with patch(
                "app.identity.continuity_ledger.record_event",
                side_effect=_fake_record,
            ), patch(
                "app.workspace_publish.publish_to_workspace",
            ):
                runtime_settings.set_goodhart_hard_gate_enforcing(not original)

            assert len(recorded) == 1, (
                "exactly one continuity-ledger event per flip"
            )
            event = recorded[0]
            assert event["kind"] == "governance_ratchet"
            assert "goodhart" in event["summary"].lower()
            assert event["detail"]["setting"] == "goodhart_hard_gate_enforcing"
            assert event["detail"]["prior"] is bool(original)
            assert event["detail"]["new"] is bool(not original)
            assert event["detail"]["effective_mode"] in (
                "advisory", "enforcing", "disabled",
            )
        finally:
            runtime_settings.set_goodhart_hard_gate_enforcing(original)

    def test_setter_no_event_on_idempotent_set(self):
        """Setting to the SAME value should NOT emit a ledger event —
        the continuity ledger only records actual transitions."""
        from app import runtime_settings

        recorded = []

        def _fake_record(**kwargs):
            recorded.append(kwargs)

        original = runtime_settings.get_goodhart_hard_gate_enforcing()
        with patch(
            "app.identity.continuity_ledger.record_event",
            side_effect=_fake_record,
        ), patch(
            "app.workspace_publish.publish_to_workspace",
        ):
            runtime_settings.set_goodhart_hard_gate_enforcing(original)

        assert recorded == []


# ── Cross-item smoke ────────────────────────────────────────────────


class TestQ1Wiring:

    def test_governance_notifier_has_anchor_in_healing(self):
        """The healing/__init__.py boot-anchor pattern is what makes
        the daemon actually start. Verify the import line is there."""
        healing_init = Path(
            "/Users/andrus/BotArmy/crewai-team/app/healing/__init__.py"
        )
        text = healing_init.read_text()
        assert "from app import governance_notifier" in text
        assert "from app.change_requests import auto_revert" in text
        assert "from app import proposal_bridge" in text
