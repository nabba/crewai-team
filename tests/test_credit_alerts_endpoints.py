"""Tests for the 2026-04-28 credit-alerts dashboard plumbing.

User asked: "please make a special section in react app budget section
where any top-up request will pop up when credits are depleted
(openrouter, Anthropic, etc.)"

Backend exposes the active alerts via /api/cp/credit-alerts; the
React dashboard polls the endpoint and renders a per-provider card
on the Budgets page with a deep-link to the provider's billing page.

These tests pin:
  * The two new endpoints exist + return the expected shape.
  * The openai SDK patch reports alerts via _check_credit_error so
    402s on the CrewAI providers path don't disappear silently.
  * The frontend has the matching endpoint constants + component.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════
# Backend endpoints
# ══════════════════════════════════════════════════════════════════════

class TestEndpointsRegistered:

    def test_get_endpoint_present(self):
        text = (REPO / "app" / "control_plane" / "dashboard_api.py").read_text()
        assert '@router.get("/credit-alerts")' in text

    def test_dismiss_endpoint_present(self):
        text = (REPO / "app" / "control_plane" / "dashboard_api.py").read_text()
        assert '@router.post("/credit-alerts/dismiss")' in text

    def test_get_response_shape_includes_alerts_and_count(self):
        """Static check that the GET handler returns the agreed shape
        ({alerts: {provider: alert}, count: int}) so the React side's
        type expectations don't drift."""
        text = (REPO / "app" / "control_plane" / "dashboard_api.py").read_text()
        # The body must reference both keys
        assert '"alerts": dict(_active_alerts)' in text
        assert '"count": len(_active_alerts)' in text

    def test_dismiss_uses_existing_resolve_helper(self):
        """The dismiss endpoint should call into firebase.publish's
        existing resolve_credit_alert — not reinvent the dict mutation."""
        text = (REPO / "app" / "control_plane" / "dashboard_api.py").read_text()
        assert "resolve_credit_alert(body.provider)" in text


# ══════════════════════════════════════════════════════════════════════
# Behavioral — endpoint returns live data from the alerts dict
# ══════════════════════════════════════════════════════════════════════

class TestEndpointBehavior:

    def setup_method(self):
        # Reset alerts so each test starts clean
        try:
            from app.firebase.publish import _active_alerts
            _active_alerts.clear()
        except Exception:
            pass

    def test_no_alerts_returns_empty(self):
        from app.control_plane.dashboard_routes_budgets_costs import credit_alerts
        out = credit_alerts()
        assert out == {"alerts": {}, "count": 0}

    def test_active_alerts_returned(self):
        from app.firebase.publish import _active_alerts
        _active_alerts["openrouter"] = {
            "provider": "openrouter",
            "error": "Error code: 402 — requires more credits",
            "url": "https://openrouter.ai/settings/credits",
            "ts": "2026-04-28T19:42:18+00:00",
            "resolved": False,
        }
        from app.control_plane.dashboard_routes_budgets_costs import credit_alerts
        out = credit_alerts()
        assert out["count"] == 1
        assert "openrouter" in out["alerts"]
        assert out["alerts"]["openrouter"]["url"].startswith("https://openrouter.ai")

    def test_dismiss_clears_active_alert(self):
        from app.firebase.publish import _active_alerts
        _active_alerts["openrouter"] = {
            "provider": "openrouter", "error": "x", "url": "y",
            "ts": "z", "resolved": False,
        }
        from app.control_plane.dashboard_routes_budgets_costs import (
            dismiss_credit_alert, CreditAlertDismiss,
        )
        out = dismiss_credit_alert(CreditAlertDismiss(provider="openrouter"))
        assert out["status"] == "dismissed"
        assert "openrouter" not in _active_alerts

    def test_dismiss_unknown_provider_returns_not_found(self):
        from app.control_plane.dashboard_routes_budgets_costs import (
            dismiss_credit_alert, CreditAlertDismiss,
        )
        out = dismiss_credit_alert(CreditAlertDismiss(provider="not_a_provider"))
        assert out["status"] == "not_found"


# ══════════════════════════════════════════════════════════════════════
# openai SDK patch — must also fire alerts on 402
# ══════════════════════════════════════════════════════════════════════

class TestOpenaiPatchTriggersAlerts:
    """The openai SDK patch added earlier today reuses the litellm
    failover machinery but ALSO needs to fire the credit alert so
    the dashboard shows the top-up card."""

    def test_patch_calls_check_credit_error_in_sync_path(self):
        text = (REPO / "app" / "rate_throttle.py").read_text()
        # Find the _patched_create body — it must call _check_credit_error
        # before invoking the failover helper.
        sync_block = re.search(
            r"def _patched_create.*?def _patched_acreate",
            text, re.DOTALL,
        )
        assert sync_block, "couldn't isolate _patched_create body"
        body = sync_block.group(0)
        assert "_check_credit_error(exc, provider)" in body, (
            "openai SDK sync patch must call _check_credit_error so the "
            "Budgets page's CreditAlertsPanel sees the alert."
        )

    def test_patch_calls_check_credit_error_in_async_path(self):
        text = (REPO / "app" / "rate_throttle.py").read_text()
        async_block = re.search(
            r"async def _patched_acreate.*?Completions\.create",
            text, re.DOTALL,
        )
        assert async_block, "couldn't isolate _patched_acreate body"
        body = async_block.group(0)
        assert "_check_credit_error(exc, provider)" in body


# ══════════════════════════════════════════════════════════════════════
# Frontend — endpoint constants + component
# ══════════════════════════════════════════════════════════════════════

class TestFrontendPlumbing:

    def test_endpoint_constants_exist(self):
        text = (REPO / "dashboard-react" / "src" / "api" / "endpoints.ts").read_text()
        assert "creditAlerts:" in text
        assert "creditAlertDismiss:" in text
        assert "/credit-alerts" in text
        assert "/credit-alerts/dismiss" in text

    def test_component_file_exists(self):
        path = (
            REPO / "dashboard-react" / "src" / "components"
            / "CreditAlertsPanel.tsx"
        )
        assert path.exists(), (
            "CreditAlertsPanel.tsx must exist for the Budgets page to import."
        )

    def test_component_renders_top_up_link_per_provider(self):
        text = (
            REPO / "dashboard-react" / "src" / "components"
            / "CreditAlertsPanel.tsx"
        ).read_text()
        # Provider URLs should be rendered as anchors with target=_blank
        assert "target=\"_blank\"" in text
        assert "Add credits" in text

    def test_component_uses_polling_query(self):
        text = (
            REPO / "dashboard-react" / "src" / "components"
            / "CreditAlertsPanel.tsx"
        ).read_text()
        # Must auto-refresh so the user sees alerts appear without
        # navigating away/back.
        assert "refetchInterval" in text

    def test_component_used_by_budget_dashboard(self):
        text = (
            REPO / "dashboard-react" / "src" / "components"
            / "BudgetDashboard.tsx"
        ).read_text()
        assert "import { CreditAlertsPanel }" in text
        assert "<CreditAlertsPanel />" in text

    def test_component_has_dismiss_action(self):
        text = (
            REPO / "dashboard-react" / "src" / "components"
            / "CreditAlertsPanel.tsx"
        ).read_text()
        assert "Dismiss" in text
        assert "creditAlertDismiss" in text

    def test_component_renders_nothing_when_no_alerts(self):
        """Static check that an empty alerts response returns null,
        keeping the Budgets page tidy during normal operation."""
        text = (
            REPO / "dashboard-react" / "src" / "components"
            / "CreditAlertsPanel.tsx"
        ).read_text()
        assert re.search(r"providers\.length\s*===\s*0", text), (
            "Component must early-return null when there are no providers "
            "with active alerts — otherwise it eats vertical space "
            "during normal operation."
        )

    def test_provider_meta_covers_known_providers(self):
        """OpenRouter, Anthropic, OpenAI, Google should all have icon +
        friendly name + fallback URL so the card looks right even if the
        backend payload is missing the URL field."""
        text = (
            REPO / "dashboard-react" / "src" / "components"
            / "CreditAlertsPanel.tsx"
        ).read_text()
        for provider in ("openrouter", "anthropic", "openai", "google"):
            assert f"{provider}:" in text, f"PROVIDER_META missing {provider}"
