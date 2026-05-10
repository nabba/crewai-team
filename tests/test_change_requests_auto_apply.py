"""Tests for the AUTO_APPLY change-request capability.

Covers:
  * RiskClass enum + ChangeRequest.risk_class field round-trip
  * validate_auto_apply: requestor/path allowlists, forbidden
    prefixes, line cap, additive-only constraint
  * create_request gracefully downgrades AUTO_APPLY → STANDARD when
    the strict validator fails (no raise)
  * auto_approve: refuses non-AUTO_APPLY, rate limits, happy path
  * auto_revert watcher: register/expire/rollback semantics
  * Disabled flag short-circuits the watcher

Tests use ``tmp_path`` + ``monkeypatch`` to redirect the store dir
and the auto_revert state file. The validator allowlists are
empty by default — tests opt in by monkeypatching the module-level
constants.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Redirect the change_requests store + auto_revert state to tmp."""
    from app.change_requests import store
    from app.change_requests import auto_revert

    store_dir = tmp_path / "change_requests"
    store_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store, "_STORE_DIR", store_dir)
    monkeypatch.setattr(store, "_AUDIT_LOG", store_dir / "audit.jsonl")
    store.reset_for_tests()

    # auto_revert reads CHANGE_REQUESTS_DIR via env var.
    monkeypatch.setenv("CHANGE_REQUESTS_DIR", str(store_dir))
    auto_revert.stop()
    auto_revert._driver_started = False

    yield store_dir
    store.reset_for_tests()


@pytest.fixture
def auto_apply_allowlists(monkeypatch):
    """Open the validator's allowlists for the duration of the test
    (default state is empty, dormant)."""
    from app.change_requests import validator
    monkeypatch.setattr(
        validator, "_AUTO_APPLY_ALLOWED_REQUESTORS",
        frozenset({"test_handler"}),
    )
    monkeypatch.setattr(
        validator, "_AUTO_APPLY_ALLOWED_PATHS",
        ("docs/auto_apply_test/",),
    )


# ── Models ─────────────────────────────────────────────────────────


class TestModels:

    def test_risk_class_enum_values(self):
        from app.change_requests import RiskClass
        assert RiskClass.STANDARD.value == "standard"
        assert RiskClass.AUTO_APPLY.value == "auto_apply"

    def test_decision_source_self_heal_auto_apply(self):
        from app.change_requests import DecisionSource
        assert DecisionSource.SELF_HEAL_AUTO_APPLY.value == "self-heal-auto-apply"

    def test_change_request_default_risk_class(self):
        from app.change_requests import ChangeRequest, RiskClass, Status
        cr = ChangeRequest(
            id="x", created_at="2026-05-10T00:00:00+00:00",
            requestor="r", path="docs/x.md", new_content="a",
            old_content="", reason="r", diff="",
        )
        assert cr.risk_class == RiskClass.STANDARD
        assert cr.origin_pattern_signature is None
        assert cr.status == Status.PENDING

    def test_to_dict_from_dict_preserves_auto_apply_fields(self):
        from app.change_requests import ChangeRequest, RiskClass
        cr = ChangeRequest(
            id="x", created_at="2026-05-10T00:00:00+00:00",
            requestor="r", path="docs/x.md", new_content="a",
            old_content="", reason="r", diff="",
            risk_class=RiskClass.AUTO_APPLY,
            origin_pattern_signature="sigabc",
        )
        d = cr.to_dict()
        assert d["risk_class"] == "auto_apply"
        assert d["origin_pattern_signature"] == "sigabc"
        cr2 = ChangeRequest.from_dict(d)
        assert cr2.risk_class == RiskClass.AUTO_APPLY
        assert cr2.origin_pattern_signature == "sigabc"

    def test_from_dict_back_compat_missing_risk_class(self):
        """Records persisted before the field existed load as STANDARD."""
        from app.change_requests import ChangeRequest, RiskClass
        legacy = {
            "id": "x", "created_at": "2026-05-10T00:00:00+00:00",
            "requestor": "r", "path": "docs/x.md",
            "new_content": "a", "old_content": "", "reason": "r",
            "diff": "", "status": "pending",
        }
        cr = ChangeRequest.from_dict(legacy)
        assert cr.risk_class == RiskClass.STANDARD
        assert cr.origin_pattern_signature is None


# ── validate_auto_apply ────────────────────────────────────────────


class TestValidateAutoApply:

    def test_standard_failure_passes_through(self):
        """Outside-roots failure surfaces as such — auto_apply layers
        AFTER standard, so standard failures dominate."""
        from app.change_requests import validate_auto_apply
        result = validate_auto_apply(
            path="workspace/foo.py", new_content="x", old_content="",
            requestor="anyone",
        )
        assert not result.ok
        assert "outside the repo's allowed roots" in result.reason

    def test_empty_requestor_allowlist_default(self):
        """Default state: nobody is in the requestor allowlist —
        capability is dormant."""
        from app.change_requests import validate_auto_apply
        result = validate_auto_apply(
            path="docs/x.md", new_content="x", old_content="",
            requestor="self_heal_handler",
        )
        assert not result.ok
        assert "auto-apply allowlist" in result.reason

    def test_allowlisted_requestor_passes(self, auto_apply_allowlists):
        from app.change_requests import validate_auto_apply
        result = validate_auto_apply(
            path="docs/auto_apply_test/x.md",
            new_content="line1\n", old_content="",
            requestor="test_handler",
        )
        assert result.ok

    def test_non_allowlist_path_rejected(self, auto_apply_allowlists):
        from app.change_requests import validate_auto_apply
        result = validate_auto_apply(
            path="docs/somewhere_else/x.md",
            new_content="x", old_content="",
            requestor="test_handler",
        )
        assert not result.ok
        assert "auto-apply path allowlist" in result.reason

    def test_forbidden_prefix_memory(self, auto_apply_allowlists, monkeypatch):
        """Even allowlisted callers can't auto-apply changes to
        memory/ paths — embedding-dim invariants are too risky."""
        # Add the memory path to the allowlist; the forbidden-prefix
        # check must STILL refuse.
        from app.change_requests import validator, validate_auto_apply
        monkeypatch.setattr(
            validator, "_AUTO_APPLY_ALLOWED_PATHS",
            ("app/memory/",),
        )
        result = validate_auto_apply(
            path="app/memory/foo.py", new_content="x", old_content="",
            requestor="test_handler",
        )
        assert not result.ok
        assert "auto-apply is categorically forbidden" in result.reason
        assert "memory/" in result.reason

    def test_forbidden_prefix_souls(self, auto_apply_allowlists, monkeypatch):
        from app.change_requests import validator, validate_auto_apply
        monkeypatch.setattr(
            validator, "_AUTO_APPLY_ALLOWED_PATHS",
            ("app/souls/",),
        )
        result = validate_auto_apply(
            path="app/souls/concierge.md", new_content="x", old_content="",
            requestor="test_handler",
        )
        assert not result.ok
        assert "souls/" in result.reason

    def test_line_cap_exceeded(self, auto_apply_allowlists):
        """Patches over the line cap fail — need operator review."""
        from app.change_requests import validate_auto_apply
        body = "\n".join(f"line{i}" for i in range(25)) + "\n"
        result = validate_auto_apply(
            path="docs/auto_apply_test/big.md",
            new_content=body, old_content="",
            requestor="test_handler",
        )
        assert not result.ok
        assert "exceeds auto-apply cap" in result.reason

    def test_deletions_violate_additive_only(self, auto_apply_allowlists):
        """Removing lines requires operator review — auto-apply is
        for defensively additive patches only."""
        from app.change_requests import validate_auto_apply
        result = validate_auto_apply(
            path="docs/auto_apply_test/x.md",
            new_content="line1\nline3\n",
            old_content="line1\nline2\nline3\n",
            requestor="test_handler",
        )
        assert not result.ok
        assert "additive-only" in result.reason

    def test_additive_under_cap_passes(self, auto_apply_allowlists):
        from app.change_requests import validate_auto_apply
        result = validate_auto_apply(
            path="docs/auto_apply_test/x.md",
            new_content="existing\nadded\n",
            old_content="existing\n",
            requestor="test_handler",
        )
        assert result.ok

    def test_new_file_is_additive(self, auto_apply_allowlists):
        """A new file (old_content='') with content under the cap
        is additive-only by definition."""
        from app.change_requests import validate_auto_apply
        result = validate_auto_apply(
            path="docs/auto_apply_test/new.md",
            new_content="new file body\n",
            old_content="",
            requestor="test_handler",
        )
        assert result.ok


# ── create_request with risk_class ─────────────────────────────────


class TestCreateRequestRiskClass:

    def test_auto_apply_downgrades_when_validator_fails(self, isolated_store):
        """AUTO_APPLY with no allowlist match → CR persists as
        STANDARD (graceful downgrade, no raise)."""
        from app.change_requests import create_request, RiskClass
        cr = create_request(
            requestor="self_heal_handler",
            path="docs/proposed_capabilities/x.md",
            new_content="body\n", old_content="",
            reason="test", risk_class=RiskClass.AUTO_APPLY,
            origin_pattern_signature="sig123",
        )
        assert cr.risk_class == RiskClass.STANDARD
        # origin_pattern_signature is cleared when downgraded
        assert cr.origin_pattern_signature is None

    def test_auto_apply_passes_validator_persists_with_risk_class(
        self, isolated_store, auto_apply_allowlists,
    ):
        from app.change_requests import create_request, RiskClass, Status
        cr = create_request(
            requestor="test_handler",
            path="docs/auto_apply_test/x.md",
            new_content="line1\n", old_content="",
            reason="test", risk_class=RiskClass.AUTO_APPLY,
            origin_pattern_signature="sig999",
        )
        assert cr.risk_class == RiskClass.AUTO_APPLY
        assert cr.origin_pattern_signature == "sig999"
        assert cr.status == Status.PENDING  # auto_approve transitions next

    def test_standard_ignores_origin_pattern_signature(self, isolated_store):
        from app.change_requests import create_request, RiskClass
        cr = create_request(
            requestor="coder", path="docs/x.md", new_content="body\n",
            old_content="", reason="r", risk_class=RiskClass.STANDARD,
            origin_pattern_signature="sigxxx",  # ignored for STANDARD
        )
        assert cr.risk_class == RiskClass.STANDARD
        assert cr.origin_pattern_signature is None

    def test_default_risk_class_is_standard(self, isolated_store):
        from app.change_requests import create_request, RiskClass
        cr = create_request(
            requestor="coder", path="docs/x.md", new_content="x",
            old_content="", reason="r",
        )
        assert cr.risk_class == RiskClass.STANDARD


# ── auto_approve ───────────────────────────────────────────────────


class TestAutoApprove:

    def _stage_auto_apply_cr(self, requestor="test_handler",
                              signature="sig123"):
        from app.change_requests import create_request, RiskClass
        return create_request(
            requestor=requestor,
            path="docs/auto_apply_test/x.md",
            new_content="line1\n", old_content="",
            reason="test", risk_class=RiskClass.AUTO_APPLY,
            origin_pattern_signature=signature,
        )

    def test_refuses_non_auto_apply_cr(self, isolated_store):
        from app.change_requests import auto_approve, create_request
        cr = create_request(
            requestor="coder", path="docs/x.md", new_content="x",
            old_content="", reason="r",
        )
        with pytest.raises(ValueError, match="non-AUTO_APPLY"):
            auto_approve(cr.id)

    def test_refuses_cr_not_in_pending(
        self, isolated_store, auto_apply_allowlists,
    ):
        from app.change_requests import auto_approve, store, Status
        cr = self._stage_auto_apply_cr()
        cr.status = Status.REJECTED
        store.save(cr)
        with pytest.raises(ValueError, match="cannot auto-approve"):
            auto_approve(cr.id)

    def test_happy_path_transitions_to_applied(
        self, isolated_store, auto_apply_allowlists,
    ):
        """auto_approve transitions PENDING → APPROVED → APPLIED with
        the right DecisionSource. apply_change is mocked because we
        don't have a host bridge in tests."""
        from app.change_requests import auto_approve, DecisionSource, Status
        from app.change_requests.apply import ApplyResult

        cr = self._stage_auto_apply_cr()

        # Mock apply_change to simulate a successful apply.
        def _fake_apply(request_id):
            from app.change_requests import lifecycle
            lifecycle.mark_applied(
                request_id, git_branch="auto/change_xxx",
                git_commit_sha="deadbeef", pr_url="https://x/pr/1",
            )
            return ApplyResult(ok=True, git_commit_sha="deadbeef")

        with patch("app.change_requests.apply.apply_change", side_effect=_fake_apply), \
             patch("app.change_requests.lifecycle._send_auto_apply_alert"), \
             patch("app.change_requests.auto_revert.register"), \
             patch("app.change_requests.lifecycle._publish_auto_apply_event"):
            result = auto_approve(cr.id)

        assert result.status == Status.APPLIED
        assert result.decided_by == DecisionSource.SELF_HEAL_AUTO_APPLY

    def test_rate_limit_per_pattern_blocks(
        self, isolated_store, auto_apply_allowlists, monkeypatch,
    ):
        """Once a pattern hits the per-pattern cap, further
        auto-applies for that signature stay PENDING."""
        from app.change_requests import lifecycle, store, Status, RiskClass

        # Seed the store with N=3 already-applied auto-applies for
        # this pattern, dated today.
        today_iso = datetime.now(timezone.utc).isoformat()
        for i in range(3):
            cr = self._stage_auto_apply_cr(signature="sigPATTERN")
            cr.status = Status.APPLIED
            cr.decided_at = today_iso
            cr.decided_by = lifecycle.DecisionSource.SELF_HEAL_AUTO_APPLY
            store.save(cr)

        # Now stage a 4th — it should be rate-limited.
        cr4 = self._stage_auto_apply_cr(signature="sigPATTERN")
        with patch("app.change_requests.lifecycle._send_auto_apply_alert"):
            result = lifecycle.auto_approve(cr4.id)

        assert result.status == Status.PENDING
        assert "rate-limited" in (result.decision_reason or "")
        assert "per-pattern rate limit" in (result.decision_reason or "")


# ── auto_revert watcher ────────────────────────────────────────────


class TestAutoRevertWatcher:

    def test_register_persists_watch(self, isolated_store):
        from app.change_requests import auto_revert
        with patch.object(auto_revert, "_signature_count", return_value=2):
            auto_revert.register(
                cr_id="cr_x", origin_pattern_signature="sigA",
                applied_at_iso=datetime.now(timezone.utc).isoformat(),
            )
        watches = auto_revert.list_active_watches()
        assert len(watches) == 1
        assert watches[0]["cr_id"] == "cr_x"
        assert watches[0]["origin_pattern_signature"] == "sigA"
        assert watches[0]["baseline_count"] == 2

    def test_register_idempotent_replaces(self, isolated_store):
        from app.change_requests import auto_revert
        with patch.object(auto_revert, "_signature_count", return_value=0):
            auto_revert.register(
                cr_id="cr_x", origin_pattern_signature="sigA",
                applied_at_iso=datetime.now(timezone.utc).isoformat(),
            )
            auto_revert.register(
                cr_id="cr_x", origin_pattern_signature="sigB",
                applied_at_iso=datetime.now(timezone.utc).isoformat(),
            )
        watches = auto_revert.list_active_watches()
        assert len(watches) == 1
        assert watches[0]["origin_pattern_signature"] == "sigB"

    def test_run_one_pass_expires_watches_past_window(self, isolated_store):
        from app.change_requests import auto_revert

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with patch.object(auto_revert, "_signature_count", return_value=0):
            auto_revert.register(
                cr_id="cr_old", origin_pattern_signature="sigA",
                applied_at_iso=old_ts,
            )
        with patch.object(auto_revert, "_signature_count", return_value=0):
            counters = auto_revert.run_one_pass()
        assert counters["expired"] == 1
        assert auto_revert.list_active_watches() == []

    def test_run_one_pass_triggers_rollback_on_recurrence(self, isolated_store):
        """Pattern recurrence within the watch window → rollback."""
        from app.change_requests import auto_revert
        from app.change_requests.apply import ApplyResult

        with patch.object(auto_revert, "_signature_count", return_value=2):
            auto_revert.register(
                cr_id="cr_x", origin_pattern_signature="sigA",
                applied_at_iso=datetime.now(timezone.utc).isoformat(),
            )

        with patch.object(auto_revert, "_signature_count", return_value=5), \
             patch(
                 "app.change_requests.apply.rollback_change",
                 return_value=ApplyResult(ok=True, git_commit_sha="r1"),
             ) as mock_rollback, \
             patch.object(auto_revert, "_alert_revert_success"), \
             patch.object(auto_revert, "_publish_auto_revert"):
            counters = auto_revert.run_one_pass()

        assert counters["reverted"] == 1
        mock_rollback.assert_called_once_with(
            "cr_x", operator="auto_revert_watcher",
        )
        # Watch entry removed after successful revert.
        assert auto_revert.list_active_watches() == []

    def test_run_one_pass_keeps_watch_when_pattern_stable(self, isolated_store):
        """No recurrence → watch stays active."""
        from app.change_requests import auto_revert

        with patch.object(auto_revert, "_signature_count", return_value=2):
            auto_revert.register(
                cr_id="cr_x", origin_pattern_signature="sigA",
                applied_at_iso=datetime.now(timezone.utc).isoformat(),
            )

        with patch.object(auto_revert, "_signature_count", return_value=2):
            counters = auto_revert.run_one_pass()
        assert counters["reverted"] == 0
        assert counters["expired"] == 0
        assert len(auto_revert.list_active_watches()) == 1

    def test_disabled_flag_short_circuits(self, isolated_store, monkeypatch):
        from app.change_requests import auto_revert
        monkeypatch.setenv("CHANGE_REQUESTS_AUTO_REVERT_ENABLED", "false")
        with patch.object(auto_revert, "_signature_count", return_value=99):
            auto_revert.register(
                cr_id="cr_x", origin_pattern_signature="sigA",
                applied_at_iso=datetime.now(timezone.utc).isoformat(),
            )
            # Even a huge recurrence can't trigger rollback when disabled.
            counters = auto_revert.run_one_pass()
        assert counters["reverted"] == 0


# ── End-to-end smoke ───────────────────────────────────────────────


class TestEndToEnd:

    def test_dormant_capability_by_default(self, isolated_store):
        """The whole pipeline is dormant by default — even a
        well-formed AUTO_APPLY request from a 'plausible' caller
        downgrades to STANDARD because the allowlist is empty.
        This is the central safety property of 'future-facing
        capability'."""
        from app.change_requests import create_request, RiskClass
        cr = create_request(
            requestor="self_heal_handler",  # NOT in allowlist by default
            path="docs/auto_apply_test/x.md",
            new_content="add\n", old_content="",
            reason="test", risk_class=RiskClass.AUTO_APPLY,
            origin_pattern_signature="sig",
        )
        assert cr.risk_class == RiskClass.STANDARD, (
            "AUTO_APPLY must downgrade to STANDARD when allowlist is empty"
        )
