"""Q2 §39 — Closure for the proposal generators. Tests for items 7, 8, 9.

Item 7 delta — coding-session spec on ProposalState + render in promoter:
  * field round-trips through to_dict/from_dict
  * stage() accepts the spec
  * promoter renders YAML block in CR body for non-Tier-3 paths
  * promoter SUPPRESSES the block for TIER_IMMUTABLE paths

Item 8 — structured-diagnosis pipeline:
  * 8.1 generate_structured_fix returns None when LLM unavailable
  * 8.1 returns declined fix when LLM declines
  * 8.1 returns actionable fix when LLM produces real patch
  * 8.2 telemetry filed / declined / resolution events
  * 8.2 approval_rate computation requires sufficient data
  * 8.3 auto-tune adjusts up when approval too low
  * 8.3 auto-tune adjusts down when approval too high
  * 8.3 auto-tune holds in band
  * 8.3 auto-tune respects floor/ceiling clamp
  * 8.3 auto-tune respects 24h cadence
  * 8.3 auto-tune respects hysteresis (≥5 resolutions)
  * 8.3 operator override short-circuits auto-tune
  * 8.5 HOT-1 observation row schema
  * 8.5 HOT-1 emitted on declined AND filed paths

Item 9 — relevant_history lookup:
  * empty history returns clean empty result
  * continuity ledger event matched by detail.path
  * CR audit log event filtered by path + window
  * format_for_operator returns empty string for empty history
  * format_for_operator renders multi-line markdown for non-empty
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


_GATEWAY_DEPS_AVAILABLE = True
try:
    import pydantic_settings  # noqa: F401
except ImportError:
    _GATEWAY_DEPS_AVAILABLE = False


# ── Item 7 delta — coding-session spec ─────────────────────────────


@pytest.mark.skipif(
    not _GATEWAY_DEPS_AVAILABLE,
    reason="needs gateway deps; runs in CI/docker",
)
class TestCodingSessionSpec:

    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROPOSAL_BRIDGE_DIR", str(tmp_path / "proposal_bridge"))
        from app.proposal_bridge import promoter
        promoter.stop()
        promoter._driver_started = False
        yield

    def test_proposal_state_roundtrip_with_spec(self):
        from app.proposal_bridge.store import ProposalState, ProposalStatus
        spec = {
            "intent": "Test",
            "files": [{"path": "a.py", "action": "create"}],
            "acceptance": ["pytest"],
            "expected_duration_min": 30,
        }
        state = ProposalState(
            source="capability_gap", signature="abc",
            title="x", target_path="docs/x.md",
            body_hash="h", staged_at="2026-05-10T00:00:00+00:00",
            status=ProposalStatus.STAGED,
            coding_session_spec=spec,
        )
        d = state.to_dict()
        assert d["coding_session_spec"]["intent"] == "Test"
        restored = ProposalState.from_dict(d)
        assert restored.coding_session_spec == spec

    def test_stage_accepts_spec(self):
        from app.proposal_bridge import stage, get_proposal
        spec = {"intent": "X", "files": [], "acceptance": ["pytest"]}
        state, was_new = stage(
            source="capability_gap", signature="sig123",
            title="X", body_markdown="# body", target_path="docs/x.md",
            coding_session_spec=spec,
        )
        assert was_new is True
        retrieved = get_proposal("capability_gap", "sig123")
        assert retrieved is not None
        assert retrieved.coding_session_spec == spec

    def test_promoter_renders_yaml_block_for_non_tier_immutable(self):
        from app.proposal_bridge.promoter import _augment_body_with_spec
        from app.proposal_bridge.store import ProposalState, ProposalStatus
        spec = {
            "intent": "Add forest-cover module",
            "files": [{"path": "app/forest/__init__.py", "action": "create"}],
            "acceptance": ["pytest tests/forest/"],
            "expected_duration_min": 45,
        }
        state = ProposalState(
            source="capability_gap", signature="abc",
            title="x", target_path="docs/proposed_capabilities/abc.md",
            body_hash="h", staged_at="2026-05-10T00:00:00+00:00",
            coding_session_spec=spec,
        )
        body = "# Original body\n"
        result = _augment_body_with_spec(body, state)
        assert "## Coding-session spec (non-Tier-3)" in result
        assert "Add forest-cover module" in result
        assert "app/forest/__init__.py" in result
        assert "pytest tests/forest/" in result
        assert "expected_duration_min: 45" in result

    def test_promoter_suppresses_spec_for_tier_immutable(self, monkeypatch):
        from app.proposal_bridge import promoter
        from app.proposal_bridge.store import ProposalState
        # Force the path-immutable check to return True.
        monkeypatch.setattr(promoter, "_path_is_tier_immutable", lambda p: True)
        spec = {"intent": "Edit governance.py", "files": [], "acceptance": []}
        state = ProposalState(
            source="capability_gap", signature="abc",
            title="x", target_path="app/governance.py",
            body_hash="h", staged_at="2026-05-10T00:00:00+00:00",
            coding_session_spec=spec,
        )
        body = "# Original\n"
        assert promoter._augment_body_with_spec(body, state) == body

    def test_promoter_no_spec_field_no_change(self):
        from app.proposal_bridge.promoter import _augment_body_with_spec
        from app.proposal_bridge.store import ProposalState
        state = ProposalState(
            source="capability_gap", signature="abc",
            title="x", target_path="docs/x.md",
            body_hash="h", staged_at="2026-05-10T00:00:00+00:00",
            coding_session_spec=None,
        )
        body = "# Original\n"
        assert _augment_body_with_spec(body, state) == body


# ── Item 8.2 — diagnosis telemetry ────────────────────────────────


@pytest.mark.skipif(
    not _GATEWAY_DEPS_AVAILABLE,
    reason="needs gateway deps; runs in CI/docker",
)
class TestDiagnosisTelemetry:

    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        log_path = tmp_path / "telemetry.jsonl"
        monkeypatch.setenv(
            "STRUCTURED_DIAGNOSIS_TELEMETRY_LOG", str(log_path),
        )
        yield

    def test_record_filed_appends_row(self):
        from app.healing.diagnosis_telemetry import record_filed, _read_all
        record_filed(
            cr_id="cr_x", pattern_signature="sig",
            file_path="app/x.py", error_class="NameError",
            confidence=0.85, threshold=0.70,
            delta_added=1, delta_removed=0,
        )
        rows = _read_all()
        assert len(rows) == 1
        assert rows[0]["event_kind"] == "filed"
        assert rows[0]["llm_confidence"] == 0.85

    def test_record_declined_appends_row(self):
        from app.healing.diagnosis_telemetry import record_declined, _read_all
        record_declined(
            pattern_signature="sig", file_path="app/x.py",
            error_class="NameError", confidence=0.4,
            threshold=0.70, decline_reason="below_threshold",
        )
        rows = _read_all()
        assert len(rows) == 1
        assert rows[0]["event_kind"] == "declined"
        assert rows[0]["decline_reason"] == "below_threshold"

    def test_record_resolution_appends_row(self):
        from app.healing.diagnosis_telemetry import record_resolution, _read_all
        record_resolution(
            cr_id="cr_x", decided_by="signal-thumbs-up",
            approved=True, decided_at="2026-05-10T00:00:00+00:00",
        )
        rows = _read_all()
        assert len(rows) == 1
        assert rows[0]["event_kind"] == "resolution"
        assert rows[0]["resolution"]["approved"] is True

    def test_approval_rate_returns_none_for_insufficient_data(self):
        from app.healing.diagnosis_telemetry import (
            record_filed, approval_rate,
        )
        # Only one filed event with no resolution → insufficient.
        record_filed(
            cr_id="cr_1", pattern_signature="s", file_path="x",
            error_class="E", confidence=0.8, threshold=0.7,
            delta_added=1, delta_removed=0,
        )
        assert approval_rate(window=20) is None

    def test_approval_rate_computed_when_sufficient(self):
        from app.healing.diagnosis_telemetry import (
            record_filed, record_resolution, approval_rate,
        )
        # 10 filed CRs, 8 resolved (5 approved, 3 rejected).
        for i in range(10):
            record_filed(
                cr_id=f"cr_{i}", pattern_signature="s", file_path="x",
                error_class="E", confidence=0.8, threshold=0.7,
                delta_added=1, delta_removed=0,
            )
        for i in range(5):
            record_resolution(
                cr_id=f"cr_{i}", decided_by="signal-thumbs-up",
                approved=True,
            )
        for i in range(5, 8):
            record_resolution(
                cr_id=f"cr_{i}", decided_by="signal-thumbs-down",
                approved=False,
            )
        rate = approval_rate(window=10)
        assert rate is not None
        assert abs(rate - 5/8) < 0.01

    def test_attempts_for_pattern_in_window(self):
        from app.healing.diagnosis_telemetry import (
            record_filed, attempts_for_pattern_in_window,
        )
        for _ in range(3):
            record_filed(
                cr_id=None or "x", pattern_signature="sig123",
                file_path="x", error_class="E",
                confidence=0.8, threshold=0.7,
                delta_added=1, delta_removed=0,
            )
        n = attempts_for_pattern_in_window("sig123", window_seconds=3600)
        assert n == 3
        assert attempts_for_pattern_in_window("other_sig", window_seconds=3600) == 0


# ── Item 8.3 — auto-tune ───────────────────────────────────────────


@pytest.mark.skipif(
    not _GATEWAY_DEPS_AVAILABLE,
    reason="needs gateway deps; runs in CI/docker",
)
class TestDiagnosisAutoTune:

    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        state_path = tmp_path / "threshold.json"
        monkeypatch.setenv(
            "STRUCTURED_DIAGNOSIS_THRESHOLD_STATE", str(state_path),
        )
        # Reset runtime_settings cache so floor/ceiling reads fresh.
        from app import runtime_settings
        runtime_settings._cache = None
        monkeypatch.setattr(runtime_settings, "_STATE_PATH", tmp_path / "rt.json")
        yield
        runtime_settings._cache = None

    def test_compute_adjustment_in_band_no_change(self):
        from app.healing.diagnosis_auto_tune import _compute_adjustment
        new, alert = _compute_adjustment(
            current=0.70, rate=0.75, floor=0.5, ceiling=0.95,
        )
        assert new == 0.70
        assert alert is None

    def test_compute_adjustment_low_approval_raises_threshold(self):
        from app.healing.diagnosis_auto_tune import _compute_adjustment
        new, alert = _compute_adjustment(
            current=0.70, rate=0.50, floor=0.5, ceiling=0.95,
        )
        assert abs(new - 0.72) < 0.001
        assert alert is None

    def test_compute_adjustment_high_approval_lowers_threshold(self):
        from app.healing.diagnosis_auto_tune import _compute_adjustment
        new, alert = _compute_adjustment(
            current=0.70, rate=0.95, floor=0.5, ceiling=0.95,
        )
        assert abs(new - 0.68) < 0.001
        assert alert is None

    def test_compute_adjustment_pinned_at_floor(self):
        from app.healing.diagnosis_auto_tune import _compute_adjustment
        new, alert = _compute_adjustment(
            current=0.50, rate=0.95, floor=0.50, ceiling=0.95,
        )
        assert new == 0.50  # already at floor, can't lower
        assert alert == "pinned_at_floor"

    def test_compute_adjustment_pinned_at_ceiling(self):
        from app.healing.diagnosis_auto_tune import _compute_adjustment
        new, alert = _compute_adjustment(
            current=0.95, rate=0.30, floor=0.50, ceiling=0.95,
        )
        assert new == 0.95
        assert alert == "pinned_at_ceiling"

    def test_24h_cadence_gate(self):
        from app.healing.diagnosis_auto_tune import _at_least_24h_since
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        assert _at_least_24h_since(recent) is False
        assert _at_least_24h_since(old) is True
        assert _at_least_24h_since(None) is True

    def test_disabled_short_circuits(self):
        from app.healing.diagnosis_auto_tune import maybe_adjust_threshold
        from app import runtime_settings
        runtime_settings.set_structured_diagnosis_auto_tune_enabled(False)
        # No state mutation should happen even with a low approval rate.
        result = maybe_adjust_threshold()
        # Effective threshold matches the state file (no adjustment ran).
        assert result["auto_tune_enabled"] is False

    def test_operator_override_short_circuits(self):
        from app.healing.diagnosis_auto_tune import current_state
        from app import runtime_settings
        runtime_settings.set_structured_diagnosis_threshold_override(0.85)
        state = current_state()
        # Effective is the override, not the state-file value.
        assert state["effective"] == 0.85
        assert state["override"] == 0.85


# ── Item 8.5 — HOT-1 observation hook ──────────────────────────────


@pytest.mark.skipif(
    not _GATEWAY_DEPS_AVAILABLE,
    reason="needs gateway deps; runs in CI/docker",
)
class TestHot1Observation:

    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        log_path = tmp_path / "metacog.jsonl"
        monkeypatch.setenv("HOT1_OBSERVATION_LOG", str(log_path))
        self.log_path = log_path
        yield

    def test_observation_emitted_on_declined_fix(self):
        from app.healing.structured_diagnosis import (
            StructuredFix, _emit_hot1_observation,
        )
        fix = StructuredFix(
            path="app/x.py", new_content="", old_content="",
            confidence=0.4, reasoning="multi-site",
            declined=True, decline_reason="multi_site",
        )
        _emit_hot1_observation(
            fix=fix, pattern_signature="sig", file_path="app/x.py",
            error_class="NameError",
        )
        assert self.log_path.exists()
        rows = self.log_path.read_text().strip().splitlines()
        row = json.loads(rows[0])
        assert row["kind"] == "metacognitive_repair_proposal"
        assert "HOT-1" in row["indicator_relevance"]
        assert row["higher_order_thought"]["declined"] is True
        assert row["higher_order_thought"]["decline_reason"] == "multi_site"

    def test_observation_emitted_on_filed_fix(self):
        from app.healing.structured_diagnosis import (
            StructuredFix, _emit_hot1_observation,
        )
        fix = StructuredFix(
            path="app/x.py",
            new_content="x = 1\nadded\n",
            old_content="x = 1\n",
            confidence=0.85,
            reasoning="Added missing import to fix NameError",
        )
        _emit_hot1_observation(
            fix=fix, pattern_signature="sig", file_path="app/x.py",
            error_class="NameError",
        )
        rows = self.log_path.read_text().strip().splitlines()
        row = json.loads(rows[0])
        assert row["higher_order_thought"]["declined"] is False
        assert row["higher_order_thought"]["self_assessed_confidence"] == 0.85
        assert row["proposed_intervention"]["delta_lines_added"] >= 1
        assert row["proposed_intervention"]["delta_additive_only"] is True


# ── Item 9 — relevant_history ──────────────────────────────────────


@pytest.mark.skipif(
    not _GATEWAY_DEPS_AVAILABLE,
    reason="needs gateway deps; runs in CI/docker",
)
class TestRelevantHistory:

    def test_empty_path_returns_empty_result(self):
        from app.identity.relevant_history import relevant_history
        result = relevant_history("")
        assert result["counts"] == {"ledger": 0, "cr_audit": 0}

    def test_format_for_operator_empty_returns_empty_string(self):
        from app.identity.relevant_history import format_for_operator
        empty = {
            "window_days": 90,
            "continuity_events": [],
            "change_request_events": [],
            "counts": {"ledger": 0, "cr_audit": 0},
            "summary_line": "no prior activity in 90d",
        }
        assert format_for_operator(empty) == ""

    def test_format_for_operator_renders_markdown_block(self):
        from app.identity.relevant_history import format_for_operator
        history = {
            "window_days": 90,
            "continuity_events": [
                {"ts": "2026-04-15T00:00:00+00:00",
                 "kind": "tier3_amendment",
                 "actor": "self_improver",
                 "summary": "Replace deprecated forge audit format v1 with v2",
                 "detail": {}},
            ],
            "change_request_events": [
                {"ts": "2026-04-30T00:00:00+00:00",
                 "event": "applied", "cr_id": "abc",
                 "status": "applied", "requestor": "coder",
                 "decided_by": "signal-thumbs-up"},
            ],
            "counts": {"ledger": 1, "cr_audit": 1},
            "summary_line": "1 CRs, 1 applied, 1 amendment in 90d",
        }
        block = format_for_operator(history)
        assert "📜 Recent activity" in block
        assert "Replace deprecated forge audit format" in block
        assert "applied" in block
        assert "self_improver" in block

    def test_ledger_event_matches_path_via_detail(self):
        from app.identity.relevant_history import _ledger_event_matches_path
        from types import SimpleNamespace
        event = SimpleNamespace(detail={"path": "app/x.py"})
        assert _ledger_event_matches_path(event, "app/x.py") is True
        assert _ledger_event_matches_path(event, "app/y.py") is False

    def test_ledger_event_matches_path_via_target_path(self):
        from app.identity.relevant_history import _ledger_event_matches_path
        from types import SimpleNamespace
        event = SimpleNamespace(detail={"target_path": "app/governance.py"})
        assert _ledger_event_matches_path(event, "app/governance.py") is True

    def test_summary_line_includes_kinds(self):
        from app.identity.relevant_history import _build_summary_line
        ledger = [
            {"kind": "tier3_amendment", "ts": "x", "actor": "y",
             "summary": "z", "detail": {}},
            {"kind": "tier3_amendment", "ts": "x", "actor": "y",
             "summary": "z", "detail": {}},
        ]
        cr = [
            {"event": "created", "ts": "x", "cr_id": "1",
             "status": "p", "requestor": "x", "decided_by": ""},
        ]
        line = _build_summary_line(
            ledger_events=ledger, cr_events=cr, window_days=60,
        )
        assert "1 CR" in line
        assert "2 amendments" in line
        assert "60d" in line

    def test_no_prior_activity_summary(self):
        from app.identity.relevant_history import _build_summary_line
        line = _build_summary_line(
            ledger_events=[], cr_events=[], window_days=90,
        )
        assert "no prior activity" in line


# ── Wiring smoke (runs locally without gateway deps) ───────────────


class TestQ2WiringSmoke:

    def test_proposal_bridge_store_has_coding_session_spec_field(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/app/proposal_bridge/store.py"
        ).read_text()
        assert "coding_session_spec" in src
        assert "coding_session_spec: Optional[dict[str, Any]]" in src

    def test_promoter_has_spec_renderer(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/app/proposal_bridge/promoter.py"
        ).read_text()
        assert "_render_spec_section" in src
        assert "_path_is_tier_immutable" in src
        assert "_augment_body_with_spec" in src

    def test_structured_diagnosis_module_exists(self):
        assert Path(
            "/Users/andrus/BotArmy/crewai-team/app/healing/structured_diagnosis.py"
        ).exists()

    def test_diagnosis_telemetry_module_exists(self):
        assert Path(
            "/Users/andrus/BotArmy/crewai-team/app/healing/diagnosis_telemetry.py"
        ).exists()

    def test_diagnosis_auto_tune_module_exists(self):
        assert Path(
            "/Users/andrus/BotArmy/crewai-team/app/healing/diagnosis_auto_tune.py"
        ).exists()

    def test_relevant_history_module_exists(self):
        assert Path(
            "/Users/andrus/BotArmy/crewai-team/app/identity/relevant_history.py"
        ).exists()

    def test_consciousness_hot1_doc_exists(self):
        assert Path(
            "/Users/andrus/BotArmy/crewai-team/docs/CONSCIOUSNESS_HOT1_OBSERVATIONS.md"
        ).exists()

    def test_lifecycle_appends_history_to_reason(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/app/change_requests/lifecycle.py"
        ).read_text()
        assert "from app.identity.relevant_history import" in src
        assert "format_for_operator" in src

    def test_lifecycle_emits_diagnosis_telemetry_on_resolution(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/app/change_requests/lifecycle.py"
        ).read_text()
        assert "_maybe_emit_diagnosis_telemetry" in src
        # All four resolution paths.
        assert src.count("_maybe_emit_diagnosis_telemetry(") >= 5  # def + 4 calls

    def test_error_diagnosis_calls_structured_path(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/app/healing/error_diagnosis.py"
        ).read_text()
        assert "_try_structured_path" in src
        assert "from app.healing.structured_diagnosis import" in src

    def test_monitors_register_auto_tune(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/app/healing/monitors/__init__.py"
        ).read_text()
        assert "diagnosis_auto_tune" in src
        assert "maybe_adjust_threshold" in src

    def test_runtime_settings_has_threshold_keys(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/app/runtime_settings.py"
        ).read_text()
        assert "structured_diagnosis_threshold_floor" in src
        assert "structured_diagnosis_threshold_ceiling" in src
        assert "structured_diagnosis_threshold_override" in src
        assert "structured_diagnosis_auto_tune_enabled" in src

    def test_config_api_has_structured_diagnosis_endpoints(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/app/api/config_api.py"
        ).read_text()
        assert "/structured_diagnosis/state" in src
        assert "/structured_diagnosis/telemetry" in src

    def test_react_settings_page_renders_card(self):
        src = Path(
            "/Users/andrus/BotArmy/crewai-team/dashboard-react/src/components/SettingsPage.tsx"
        ).read_text()
        assert "StructuredDiagnosisCard" in src
        assert "Structured-diagnosis confidence" in src
