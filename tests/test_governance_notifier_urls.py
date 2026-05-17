"""Tier-3 amendment Signal messages include clickable iPhone + Mac links.

The operator's decision (2026-05-17): keep Tier-3 amendments React-only
(diff + eligibility + monitoring window are best reviewed on a full
screen, not approved by tapping a phone reaction) — but every alert
must include both an iPhone (Funnel HTTPS) and a Mac (Tailnet :3100)
link so the operator can tap whichever device is at hand. This test
pins the contract.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.governance_notifier import (  # noqa: E402
    _NOTIFY_STATES,
    _dashboard_url,
    _format_message,
)


class TestBackwardsCompatHelper:
    """The old _dashboard_url alias still resolves (no in-tree callers
    but external scripts may have it)."""

    def test_returns_iphone_url(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://example.test")
        assert _dashboard_url("/cp/amendments/abc") == "https://example.test/cp/amendments/abc"


class TestEveryNotifyStateIncludesBothLinks:
    """Every Tier-3 amendment Signal message must include BOTH an
    iPhone and a Mac link. Tier-3 is React-only by design (no 👍/👎
    wiring) so the operator's only path from alert to action is via
    the dashboard — make sure they can get there from either device.
    """

    def test_every_template_references_links(self):
        for state, (_, template) in _NOTIFY_STATES.items():
            assert "{links}" in template, (
                f"State {state!r} alert is missing the {{links}} placeholder "
                f"— Tier-3 amendments must include clickable links."
            )

    def test_cooldown_ok_has_both_links(self, monkeypatch):
        """COOLDOWN_OK is the actual decision point — both devices
        must be represented."""
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://phone.test")
        monkeypatch.setenv("DASHBOARD_MAC_URL", "http://mac.test:3100")
        proposal = SimpleNamespace(
            id="amend-abc-123",
            target_path="app/foo.py",
            proposer="self_improver",
            citation="42",
            evidence={},
        )
        salience, template = _NOTIFY_STATES["cooldown_ok"]
        body = _format_message(template, proposal)
        assert "📱" in body and "💻" in body
        assert "https://phone.test/cp/amendments/amend-abc-123" in body
        assert "http://mac.test:3100/cp/amendments/amend-abc-123" in body
        # The CTA wording should make the action obvious.
        assert "approve" in body.lower()

    def test_reverted_has_both_links(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://phone.test")
        monkeypatch.setenv("DASHBOARD_MAC_URL", "http://mac.test:3100")
        proposal = SimpleNamespace(
            id="amend-rev-1",
            target_path="app/bar.py",
            proposer="evolution",
            citation="",
            eligibility_failures=[],
            evidence={},
        )
        salience, template = _NOTIFY_STATES["reverted"]
        body = _format_message(template, proposal)
        assert "https://phone.test/cp/amendments/amend-rev-1" in body
        assert "http://mac.test:3100/cp/amendments/amend-rev-1" in body

    def test_eligibility_failed_has_both_links(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://phone.test")
        monkeypatch.setenv("DASHBOARD_MAC_URL", "http://mac.test:3100")
        proposal = SimpleNamespace(
            id="amend-fail-9",
            target_path="app/baz.py",
            proposer="agent_x",
            eligibility_failures=["promotions<200", "rollback>5%"],
            citation="",
            evidence={},
        )
        salience, template = _NOTIFY_STATES["eligibility_failed"]
        body = _format_message(template, proposal)
        assert "https://phone.test/cp/amendments/amend-fail-9" in body
        assert "http://mac.test:3100/cp/amendments/amend-fail-9" in body
        assert "promotions<200" in body or "promotions" in body

    def test_every_state_renders_both_links_with_id(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://phone.test")
        monkeypatch.setenv("DASHBOARD_MAC_URL", "http://mac.test:3100")
        proposal = SimpleNamespace(
            id="distinctive-id-99",
            target_path="app/zzz.py",
            proposer="x",
            citation="",
            evidence={},
        )
        for state, (_, template) in _NOTIFY_STATES.items():
            if state == "eligibility_failed":
                proposal.eligibility_failures = []
            body = _format_message(template, proposal)
            assert "distinctive-id-99" in body, f"{state}: id missing"
            assert "https://phone.test/cp/amendments/distinctive-id-99" in body, (
                f"{state}: iPhone URL missing or malformed"
            )
            assert "http://mac.test:3100/cp/amendments/distinctive-id-99" in body, (
                f"{state}: Mac URL missing or malformed"
            )
