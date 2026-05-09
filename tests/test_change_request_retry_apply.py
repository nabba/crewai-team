"""Lifecycle regression: retry-apply must work from APPLY_FAILED.

Pre-fix shape (the operator-reported bug):

    POST /api/cp/changes/{id}/retry-apply
    → calls approve(id, source=REACT_APPROVE) per changes_api.py:264
    → lifecycle.approve raised ValueError because status=APPLY_FAILED
      wasn't in the accepted set.
    → endpoint 500'd, retry never reached apply_change()

Post-fix: approve() accepts APPLY_FAILED → APPROVED as the retry
path.  audit_event is ``re-approved-for-retry`` (vs ``approved`` for
the normal first-approve path) so forensics can distinguish them.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the change_requests JSON store to a tmp_path so the
    test doesn't touch the real workspace state."""
    from app.change_requests import store

    monkeypatch.setattr(store, "_STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(
        store, "_AUDIT_LOG", tmp_path / "store" / "audit.jsonl",
    )
    store.reset_for_tests()
    yield


def _seed_request(status_value: str = "pending"):
    """Insert a synthetic ChangeRequest with the requested status."""
    from app.change_requests import store as cr_store
    from app.change_requests.models import (
        ChangeRequest, Status,
    )

    cr = ChangeRequest(
        id="abc123abc123",
        created_at="2026-05-09T00:00:00+00:00",
        requestor="test",
        path="docs/example.md",
        new_content="hello\n",
        old_content="",
        reason="test",
        diff="",
        status=Status(status_value),
    )
    cr_store.save(cr)
    return cr


# ── Approve transition table ────────────────────────────────────────


class TestApproveAcceptedStatuses:
    """approve() accepts PENDING (first-time) and APPLY_FAILED (retry).
    Already-APPROVED is idempotent.  Everything else raises ValueError."""

    def test_pending_to_approved(self, isolated_store) -> None:
        from app.change_requests import (
            DecisionSource, Status, approve, get,
        )

        _seed_request("pending")
        approve(
            "abc123abc123",
            source=DecisionSource.REACT_APPROVE,
            decision_reason="user clicked approve",
        )
        cr = get("abc123abc123")
        assert cr.status is Status.APPROVED
        assert cr.decision_reason == "user clicked approve"

    def test_apply_failed_to_approved_for_retry(
        self, isolated_store,
    ) -> None:
        """The pre-fix bug: this raised ValueError.  Now it succeeds
        and the audit log records 're-approved-for-retry'."""
        from app.change_requests import (
            DecisionSource, Status, approve, get,
        )

        _seed_request("apply_failed")
        approve(
            "abc123abc123",
            source=DecisionSource.REACT_APPROVE,
            decision_reason="retry after apply failure",
        )
        cr = get("abc123abc123")
        assert cr.status is Status.APPROVED
        assert cr.decision_reason == "retry after apply failure"

    def test_already_approved_is_idempotent(self, isolated_store) -> None:
        from app.change_requests import (
            DecisionSource, Status, approve, get,
        )

        _seed_request("approved")
        # No exception, returns the same record.
        approve(
            "abc123abc123",
            source=DecisionSource.REACT_APPROVE,
        )
        cr = get("abc123abc123")
        assert cr.status is Status.APPROVED

    def test_rejected_cannot_be_approved(self, isolated_store) -> None:
        from app.change_requests import DecisionSource, approve

        _seed_request("rejected")
        with pytest.raises(ValueError, match="cannot approve"):
            approve("abc123abc123", source=DecisionSource.REACT_APPROVE)

    def test_applied_cannot_be_re_approved(self, isolated_store) -> None:
        """APPLIED is terminal-success; re-approving doesn't make sense."""
        from app.change_requests import DecisionSource, approve

        _seed_request("applied")
        with pytest.raises(ValueError, match="cannot approve"):
            approve("abc123abc123", source=DecisionSource.REACT_APPROVE)

    def test_rolled_back_cannot_be_approved(self, isolated_store) -> None:
        from app.change_requests import DecisionSource, approve

        _seed_request("rolled_back")
        with pytest.raises(ValueError, match="cannot approve"):
            approve("abc123abc123", source=DecisionSource.REACT_APPROVE)

    def test_tier_immutable_refused_cannot_be_approved(
        self, isolated_store,
    ) -> None:
        """TIER_IMMUTABLE is the absolute backstop — re-approving
        does nothing useful."""
        from app.change_requests import DecisionSource, approve

        _seed_request("tier_immutable_refused")
        with pytest.raises(ValueError, match="cannot approve"):
            approve("abc123abc123", source=DecisionSource.REACT_APPROVE)

    def test_unknown_id_raises_keyerror(self, isolated_store) -> None:
        from app.change_requests import DecisionSource, approve

        with pytest.raises(KeyError, match="not found"):
            approve(
                "nope-no-such-id",
                source=DecisionSource.REACT_APPROVE,
            )


# ── Audit-event distinction ─────────────────────────────────────────


class TestAuditEventTagsByPriorStatus:
    """First-time approval is logged as ``approved``; retry is logged
    as ``re-approved-for-retry``.  Forensics can separate them."""

    def test_first_approve_audit_event(self, isolated_store) -> None:
        from app.change_requests import DecisionSource, approve
        from app.change_requests import store as cr_store

        _seed_request("pending")
        approve("abc123abc123", source=DecisionSource.REACT_APPROVE)

        # Read the audit log
        with cr_store._AUDIT_LOG.open() as f:
            lines = f.readlines()
        # The most recent entry is our approval
        import json
        last = json.loads(lines[-1])
        assert last["payload"]["event"] == "approved"

    def test_retry_approve_audit_event(self, isolated_store) -> None:
        from app.change_requests import DecisionSource, approve
        from app.change_requests import store as cr_store

        _seed_request("apply_failed")
        approve("abc123abc123", source=DecisionSource.REACT_APPROVE)

        with cr_store._AUDIT_LOG.open() as f:
            lines = f.readlines()
        import json
        last = json.loads(lines[-1])
        assert last["payload"]["event"] == "re-approved-for-retry"
