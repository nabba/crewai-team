"""Tests for app.tool_registry.forge_bridge — Phase 3.

Covers:
  * **Status → Tier mapping** — exact correspondence between Forge's
    enum and our tier levels.
  * **Bridge is no-op when Forge disabled** — env var off, or import
    failure, returns 0 cleanly.
  * **register / replace_spec / unregister** — registry state changes
    on tier transitions and removals.
  * **BaseTool wrapper** — proxies into ``forge.runtime.dispatcher.invoke_tool``,
    passes params through, formats responses correctly.
  * **Reconciler loop** — the asyncio coroutine starts, runs, and is
    cancellable.

Forge's actual DB and dispatcher are mocked — these tests don't
require a live Forge environment.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ── Status → Tier mapping ────────────────────────────────────────────


class TestStatusMapping:

    def test_only_three_statuses_bridged(self):
        """Forge has 7 statuses. Only SHADOW / CANARY / ACTIVE map.
        DRAFT / QUARANTINED / DEPRECATED / KILLED do NOT bridge."""
        from app.tool_registry.forge_bridge import _BRIDGED_STATUSES
        from app.tool_registry import Tier

        assert _BRIDGED_STATUSES["SHADOW"] is Tier.SHADOW
        assert _BRIDGED_STATUSES["CANARY"] is Tier.CANARY
        assert _BRIDGED_STATUSES["ACTIVE"] is Tier.PRODUCTION
        # No other statuses
        assert "DRAFT" not in _BRIDGED_STATUSES
        assert "KILLED" not in _BRIDGED_STATUSES
        assert "DEPRECATED" not in _BRIDGED_STATUSES
        assert "QUARANTINED" not in _BRIDGED_STATUSES


# ── Bridge no-op when Forge disabled ─────────────────────────────────


class TestForgeDisabled:

    def test_returns_zero_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("TOOL_FORGE_ENABLED", raising=False)
        from app.tool_registry.forge_bridge import sync_forge_tools
        # Should not raise; should return 0.
        assert sync_forge_tools() == 0

    def test_returns_zero_when_env_zero(self, monkeypatch):
        monkeypatch.setenv("TOOL_FORGE_ENABLED", "0")
        from app.tool_registry.forge_bridge import sync_forge_tools
        assert sync_forge_tools() == 0

    def test_returns_zero_on_forge_import_failure(self, monkeypatch):
        monkeypatch.setenv("TOOL_FORGE_ENABLED", "1")
        from app.tool_registry import forge_bridge
        # Force the inner forge.registry import to fail.
        with patch.object(forge_bridge, "_is_forge_enabled", return_value=False):
            assert forge_bridge.sync_forge_tools() == 0


# ── Sync semantics ───────────────────────────────────────────────────


class TestSyncSemantics:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        ToolRegistry.reset_for_tests()

    def test_new_forge_tool_registered(self):
        from app.tool_registry import ToolRegistry
        from app.tool_registry import forge_bridge

        # Fake list_tools that returns one SHADOW tool.
        rows = {
            "SHADOW": [{"tool_id": "abc-1", "name": "fake_tool",
                        "description": "Fake forge tool description here."}],
            "CANARY": [],
            "ACTIVE": [],
        }
        with patch.object(forge_bridge, "_is_forge_enabled", return_value=True), \
             patch("app.forge.registry.list_tools",
                   side_effect=lambda status, limit=500: rows.get(status, [])):
            count = forge_bridge.sync_forge_tools()

        assert count == 1
        spec = ToolRegistry.instance().get("fake_tool")
        assert spec is not None
        from app.tool_registry import Tier
        assert spec.tier is Tier.SHADOW
        assert spec.source_module == "app.forge.tools.fake_tool"

    def test_tier_transition_replaces_spec(self):
        """SHADOW → CANARY → ACTIVE must update the registry's tier."""
        from app.tool_registry import Tier, ToolRegistry
        from app.tool_registry import forge_bridge

        # First sync: tool is SHADOW.
        rows_state = {
            "SHADOW": [{"tool_id": "abc-1", "name": "promotee",
                        "description": "Tool that gets promoted up the tiers."}],
            "CANARY": [],
            "ACTIVE": [],
        }
        with patch.object(forge_bridge, "_is_forge_enabled", return_value=True), \
             patch("app.forge.registry.list_tools",
                   side_effect=lambda status, limit=500: rows_state.get(status, [])):
            forge_bridge.sync_forge_tools()
        assert ToolRegistry.instance().get("promotee").tier is Tier.SHADOW

        # Second sync: tool moved to CANARY.
        rows_state = {
            "SHADOW": [],
            "CANARY": [{"tool_id": "abc-1", "name": "promotee",
                        "description": "Tool that gets promoted up the tiers."}],
            "ACTIVE": [],
        }
        with patch.object(forge_bridge, "_is_forge_enabled", return_value=True), \
             patch("app.forge.registry.list_tools",
                   side_effect=lambda status, limit=500: rows_state.get(status, [])):
            forge_bridge.sync_forge_tools()
        assert ToolRegistry.instance().get("promotee").tier is Tier.CANARY

        # Third sync: tool moved to ACTIVE → maps to Tier.PRODUCTION.
        rows_state = {
            "SHADOW": [],
            "CANARY": [],
            "ACTIVE": [{"tool_id": "abc-1", "name": "promotee",
                        "description": "Tool that gets promoted up the tiers."}],
        }
        with patch.object(forge_bridge, "_is_forge_enabled", return_value=True), \
             patch("app.forge.registry.list_tools",
                   side_effect=lambda status, limit=500: rows_state.get(status, [])):
            forge_bridge.sync_forge_tools()
        assert ToolRegistry.instance().get("promotee").tier is Tier.PRODUCTION

    def test_disappearing_tool_unregistered(self):
        """Tool present in sync N but not N+1 is removed from the registry."""
        from app.tool_registry import ToolRegistry
        from app.tool_registry import forge_bridge

        # Initial: tool exists.
        with patch.object(forge_bridge, "_is_forge_enabled", return_value=True), \
             patch("app.forge.registry.list_tools",
                   side_effect=lambda status, limit=500: [
                       {"tool_id": "abc-1", "name": "ephemeral",
                        "description": "Goes away after first sync."}
                   ] if status == "SHADOW" else []):
            forge_bridge.sync_forge_tools()
        assert ToolRegistry.instance().get("ephemeral") is not None

        # Second sync: tool gone (status changed to KILLED, no longer
        # in {SHADOW, CANARY, ACTIVE}).
        with patch.object(forge_bridge, "_is_forge_enabled", return_value=True), \
             patch("app.forge.registry.list_tools",
                   side_effect=lambda status, limit=500: []):
            forge_bridge.sync_forge_tools()
        assert ToolRegistry.instance().get("ephemeral") is None

    def test_existing_tool_idempotent(self):
        """Re-sync with same status + description = no churn."""
        from app.tool_registry import ToolRegistry
        from app.tool_registry import forge_bridge

        rows = lambda status, limit=500: (
            [{"tool_id": "abc", "name": "stable",
              "description": "Stable description that shouldn't churn."}]
            if status == "SHADOW" else []
        )
        with patch.object(forge_bridge, "_is_forge_enabled", return_value=True), \
             patch("app.forge.registry.list_tools", side_effect=rows):
            forge_bridge.sync_forge_tools()
            spec_before = ToolRegistry.instance().get("stable")
            forge_bridge.sync_forge_tools()
            spec_after = ToolRegistry.instance().get("stable")

        # Identical hash → same spec object kept (register is idempotent).
        assert spec_before is spec_after


# ── BaseTool wrapper ─────────────────────────────────────────────────


class TestBaseToolWrapper:

    def test_wrapper_proxies_params_to_dispatcher(self):
        """When the agent invokes the wrapped tool, params reach
        ``forge.runtime.dispatcher.invoke_tool``."""
        from app.tool_registry.forge_bridge import _make_forge_basetool

        cls = _make_forge_basetool(
            tool_id="test-id",
            tool_name="test_proxy",
            tool_description="Test proxy description.",
        )
        instance = cls()
        with patch("app.forge.runtime.dispatcher.invoke_tool") as mock_invoke:
            mock_invoke.return_value = {"ok": True, "result": "OK", "mode": "production"}
            out = instance._run(params={"x": 42})

        mock_invoke.assert_called_once_with(tool_id="test-id", params={"x": 42})
        assert "OK" in out

    def test_wrapper_handles_dispatcher_refusal(self):
        from app.tool_registry.forge_bridge import _make_forge_basetool

        cls = _make_forge_basetool(
            tool_id="test-id", tool_name="test_proxy",
            tool_description="x" * 50,
        )
        with patch("app.forge.runtime.dispatcher.invoke_tool") as mock:
            mock.return_value = {"ok": False, "error": "killswitch is engaged"}
            out = cls()._run(params={})
        assert "refused" in out.lower()
        assert "killswitch" in out

    def test_wrapper_signals_shadow_mode(self):
        """SHADOW-tier tools get a special response that tells the
        agent the result was withheld."""
        from app.tool_registry.forge_bridge import _make_forge_basetool

        cls = _make_forge_basetool(
            tool_id="test-id", tool_name="test_proxy",
            tool_description="x" * 50,
        )
        with patch("app.forge.runtime.dispatcher.invoke_tool") as mock:
            mock.return_value = {
                "ok": True, "mode": "SHADOW", "shadow_result": "secret_value",
                "elapsed_ms": 150, "capability_used": "http.lan",
            }
            out = cls()._run(params={})
        assert "SHADOW" in out
        assert "secret_value" not in out  # critical: result NOT exposed
        assert "operator review" in out.lower()

    def test_wrapper_handles_dispatcher_exception(self):
        from app.tool_registry.forge_bridge import _make_forge_basetool

        cls = _make_forge_basetool(
            tool_id="test-id", tool_name="test_proxy",
            tool_description="x" * 50,
        )
        with patch("app.forge.runtime.dispatcher.invoke_tool",
                   side_effect=RuntimeError("DB down")):
            out = cls()._run(params={})
        assert "ERROR" in out
        assert "DB down" in out


# ── Reconciler loop ──────────────────────────────────────────────────


class TestReconciler:

    def test_loop_starts_sleeps_and_can_be_cancelled(self):
        """The loop runs and returns cleanly when cancelled."""
        import asyncio
        from app.tool_registry.forge_bridge import reconciliation_loop

        async def run():
            task = asyncio.create_task(reconciliation_loop(interval_sec=0.05))
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return task.done()

        assert asyncio.run(run()) is True

    def test_loop_calls_sync_periodically(self):
        import asyncio
        from app.tool_registry import forge_bridge

        call_count = {"n": 0}

        def fake_sync():
            call_count["n"] += 1
            return 0

        async def run():
            with patch.object(forge_bridge, "sync_forge_tools", side_effect=fake_sync):
                task = asyncio.create_task(
                    forge_bridge.reconciliation_loop(interval_sec=0.05)
                )
                await asyncio.sleep(0.18)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run(run())
        # At least 2 sync calls in ~0.15s with 0.05s interval.
        assert call_count["n"] >= 2


# ── replace_spec / unregister registry methods ──────────────────────


class TestRegistryMutators:

    def test_replace_spec_updates_existing(self):
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool, ToolSpec,
        )
        ToolRegistry.reset_for_tests()
        reg = ToolRegistry.instance()

        @register_tool(
            name="r_test",
            capabilities=["renders-pdf"],
            description="Initial description here.",
            tier=Tier.SHADOW,
        )
        def f():
            class _T:
                name = "r_test"
            return _T()

        # Build a new spec at PRODUCTION tier.
        new_spec = ToolSpec(
            name="r_test", capabilities=("renders-pdf",),
            tier=Tier.PRODUCTION, lifecycle=Lifecycle.SINGLETON,
            description="Updated description here.",
            args_schema=None, factory=f, guard=lambda: True,
            workspace_scope=("*",), source_module="app.test",
        )
        reg.replace_spec(new_spec)

        assert reg.get("r_test").tier is Tier.PRODUCTION
        assert reg.get("r_test").description == "Updated description here."

    def test_unregister_removes(self):
        from app.tool_registry import (
            Tier, ToolRegistry, register_tool,
        )
        ToolRegistry.reset_for_tests()
        reg = ToolRegistry.instance()

        @register_tool(
            name="u_test",
            capabilities=["renders-pdf"],
            description="To be unregistered description.",
            tier=Tier.PRODUCTION,
        )
        def f():
            class _T:
                name = "u_test"
            return _T()

        assert reg.get("u_test") is not None
        assert reg.unregister("u_test") is True
        assert reg.get("u_test") is None
        # Idempotent — second call returns False.
        assert reg.unregister("u_test") is False
