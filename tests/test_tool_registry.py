"""Tests for app.tool_registry — Phase 1a foundation.

Three layers covered:

  * **Capability vocabulary** — bounded, no surprises, deprecation
    path works.
  * **Decorator + registry** — registration is idempotent, unknown
    capabilities are rejected, lifecycle caching works correctly.
  * **Boot integration** — production tools annotated in Phase 1a
    are reachable via the registry; descriptions hash stably; drift
    detection produces sensible output.

Tests use ``ToolRegistry.reset_for_tests()`` to start each test with
a clean singleton, then re-import to repopulate. This is the same
pattern memory_tool tests use.
"""
from __future__ import annotations

import pytest


# ── Capability vocabulary ────────────────────────────────────────────


class TestCapabilities:

    def test_no_duplicates_across_categories(self):
        """A capability tag must live in exactly one category."""
        from app.tool_registry.capabilities import CAPABILITIES
        seen: dict[str, str] = {}
        for cat_name, cat in CAPABILITIES.items():
            for tag in cat:
                assert tag not in seen, (
                    f"capability tag {tag!r} appears in both "
                    f"{seen[tag]!r} and {cat_name!r}"
                )
                seen[tag] = cat_name

    def test_all_tags_kebab_case(self):
        """Tags follow the kebab-case verb-or-noun-verb convention."""
        from app.tool_registry.capabilities import all_capability_tags
        for tag in all_capability_tags():
            assert tag.islower(), f"tag {tag!r} should be lowercase"
            assert " " not in tag, f"tag {tag!r} has whitespace"
            assert "_" not in tag, f"tag {tag!r} should be kebab-case (-) not snake_case (_)"

    def test_known_tags_have_descriptions(self):
        """Every registered tag has a non-empty human description."""
        from app.tool_registry.capabilities import (
            all_capability_tags, description_for,
        )
        for tag in all_capability_tags():
            desc = description_for(tag)
            assert desc, f"tag {tag!r} has no description"
            assert len(desc) > 10, f"tag {tag!r} description too short"

    def test_category_for_unknown(self):
        from app.tool_registry.capabilities import category_for
        assert category_for("totally-fake-tag") is None

    def test_is_known(self):
        from app.tool_registry.capabilities import is_known
        assert is_known("renders-pdf")
        assert not is_known("totally-fake-tag")


# ── Decorator + registry ─────────────────────────────────────────────


class TestRegistryDecorator:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        ToolRegistry.reset_for_tests()

    def test_register_simple_tool(self):
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool,
        )

        @register_tool(
            name="t_simple",
            capabilities=["renders-pdf"],
            description="Test tool — renders a PDF (description must be >10 chars).",
            tier=Tier.PRODUCTION,
            lifecycle=Lifecycle.SINGLETON,
        )
        def factory():
            class _T:
                name = "t_simple"
            return _T()

        reg = ToolRegistry.instance()
        spec = reg.get("t_simple")
        assert spec is not None
        assert spec.name == "t_simple"
        assert spec.capabilities == ("renders-pdf",)
        assert spec.tier is Tier.PRODUCTION

    def test_unknown_capability_rejected_at_decoration(self):
        from app.tool_registry import register_tool

        with pytest.raises(ValueError, match="unknown capability"):
            @register_tool(
                name="t_bad",
                capabilities=["totally-fake-tag"],
                description="x" * 50,
            )
            def f():
                pass

    def test_no_capabilities_rejected(self):
        from app.tool_registry import register_tool
        with pytest.raises(ValueError, match="at least one capability"):
            @register_tool(
                name="t_nocaps",
                capabilities=[],
                description="x" * 50,
            )
            def f():
                pass

    def test_idempotent_re_registration(self):
        """Same name + same description = no-op, no warning."""
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool,
        )

        @register_tool(
            name="t_idem",
            capabilities=["renders-pdf"],
            description="Test idempotent tool description here.",
        )
        def factory_v1():
            class _T:
                name = "t_idem"
            return _T()

        @register_tool(
            name="t_idem",
            capabilities=["renders-pdf"],
            description="Test idempotent tool description here.",
        )
        def factory_v2():  # different fn, same spec → silent no-op
            class _T2:
                name = "t_idem"
            return _T2()

        reg = ToolRegistry.instance()
        assert len([s for s in reg.all() if s.name == "t_idem"]) == 1

    def test_singleton_lifecycle_caches(self):
        from app.tool_registry import (
            Lifecycle, ToolRegistry, register_tool,
        )

        instances_built = []

        @register_tool(
            name="t_singleton",
            capabilities=["renders-pdf"],
            description="Singleton lifecycle test tool.",
            lifecycle=Lifecycle.SINGLETON,
        )
        def factory():
            class _T:
                name = "t_singleton"
            inst = _T()
            instances_built.append(inst)
            return inst

        reg = ToolRegistry.instance()
        a = reg.build_instance("t_singleton")
        b = reg.build_instance("t_singleton")
        assert a is b
        assert len(instances_built) == 1

    def test_per_call_lifecycle_does_not_cache(self):
        from app.tool_registry import (
            Lifecycle, ToolRegistry, register_tool,
        )

        instances_built = []

        @register_tool(
            name="t_percall",
            capabilities=["renders-pdf"],
            description="PER_CALL lifecycle test tool description.",
            lifecycle=Lifecycle.PER_CALL,
        )
        def factory():
            class _T:
                name = "t_percall"
            inst = _T()
            instances_built.append(inst)
            return inst

        reg = ToolRegistry.instance()
        a = reg.build_instance("t_percall")
        b = reg.build_instance("t_percall")
        assert a is not b
        assert len(instances_built) == 2

    def test_guard_blocks_loading(self):
        from app.tool_registry import ToolRegistry, register_tool

        @register_tool(
            name="t_guarded",
            capabilities=["renders-pdf"],
            description="Tool whose guard returns False — never loadable.",
            guard=lambda: False,
        )
        def factory():
            class _T:
                name = "t_guarded"
            return _T()

        reg = ToolRegistry.instance()
        spec = reg.get("t_guarded")
        assert spec is not None
        assert not spec.is_loadable
        with pytest.raises(RuntimeError, match="guard"):
            reg.build_instance("t_guarded")


# ── Filter API ────────────────────────────────────────────────────────


class TestRegistryFilter:

    def setup_method(self) -> None:
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool,
        )
        ToolRegistry.reset_for_tests()

        # Three tools at different tiers + workspaces
        @register_tool(
            name="f_prod",
            capabilities=["renders-pdf"],
            description="Production tool description.",
            tier=Tier.PRODUCTION,
            workspace_scope=("*",),
        )
        def fp():
            class _T:
                name = "f_prod"
            return _T()

        @register_tool(
            name="f_shadow",
            capabilities=["renders-pdf"],
            description="Shadow tier tool description.",
            tier=Tier.SHADOW,
            workspace_scope=("*",),
        )
        def fs():
            class _T:
                name = "f_shadow"
            return _T()

        @register_tool(
            name="f_workspace_eesti",
            capabilities=["renders-chart"],
            description="Workspace-scoped tool description.",
            tier=Tier.PRODUCTION,
            workspace_scope=("eesti-mets",),
        )
        def fw():
            class _T:
                name = "f_workspace_eesti"
            return _T()

    def test_capability_filter(self):
        from app.tool_registry import ToolRegistry
        reg = ToolRegistry.instance()
        results = reg.filter(capabilities=["renders-pdf"])
        names = sorted(s.name for s in results)
        assert names == ["f_prod", "f_shadow"]

    def test_tier_at_most(self):
        from app.tool_registry import ToolRegistry, Tier
        reg = ToolRegistry.instance()
        results = reg.filter(tier_at_most=Tier.PRODUCTION)
        names = sorted(s.name for s in results)
        # SHADOW + PRODUCTION are below-or-at PRODUCTION
        assert "f_prod" in names
        assert "f_shadow" in names
        assert "f_workspace_eesti" in names

    def test_workspace_filter(self):
        from app.tool_registry import ToolRegistry
        reg = ToolRegistry.instance()
        # Workspace 'plg' should NOT see the eesti-only tool
        results = reg.filter(workspace="plg")
        names = sorted(s.name for s in results)
        assert "f_workspace_eesti" not in names
        assert "f_prod" in names  # *-scoped — visible everywhere

        # Workspace 'eesti-mets' SHOULD see it
        results = reg.filter(workspace="eesti-mets")
        names = sorted(s.name for s in results)
        assert "f_workspace_eesti" in names


# ── Boot integration: real tools registered ──────────────────────────


class TestBootIntegration:
    """Verifies the production tools annotated in Phase 1a actually
    end up in the registry after boot_registry()."""

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        ToolRegistry.reset_for_tests()
        # Re-import the modules so their decorators fire on the fresh registry.
        # boot_registry does this for us via walk_packages.
        from app.tool_registry.boot import boot_registry
        boot_registry(snapshot_to_postgres=False)

    def test_pdf_compose_registered(self):
        from app.tool_registry import ToolRegistry
        spec = ToolRegistry.instance().get("pdf_compose")
        assert spec is not None
        assert "renders-pdf" in spec.capabilities

    def test_signal_attachment_registered(self):
        from app.tool_registry import ToolRegistry
        spec = ToolRegistry.instance().get("signal_send_attachment")
        assert spec is not None
        assert "sends-signal" in spec.capabilities

    def test_core_tools_registered(self):
        from app.tool_registry import ToolRegistry
        reg = ToolRegistry.instance()
        for name in ["file_manager", "web_search", "read_attachment", "execute_code"]:
            assert reg.get(name) is not None, f"{name} should be in registry"

    def test_all_registered_tools_have_capabilities(self):
        from app.tool_registry import ToolRegistry
        reg = ToolRegistry.instance()
        for spec in reg.all():
            assert len(spec.capabilities) >= 1, (
                f"tool {spec.name!r} has no capabilities — would never be "
                f"discoverable"
            )

    def test_no_name_collisions(self):
        from app.tool_registry import ToolRegistry
        reg = ToolRegistry.instance()
        names = [s.name for s in reg.all()]
        assert len(names) == len(set(names)), "duplicate tool names in registry"

    def test_description_hash_stable_across_boots(self):
        """Re-booting against the same code should produce identical hashes."""
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        before = {s.name: s.description_hash for s in ToolRegistry.instance().all()}

        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False)
        after = {s.name: s.description_hash for s in ToolRegistry.instance().all()}

        assert before == after, "description hashes drifted across boots"


# ── Capability governance: TIER_IMMUTABLE inclusion ──────────────────


class TestGovernance:

    def test_capabilities_file_is_immutable(self):
        """capabilities.py must be in TIER_IMMUTABLE so the Self-Improver
        can't grow the vocabulary on its own."""
        from app.auto_deployer import TIER_IMMUTABLE
        assert "app/tool_registry/capabilities.py" in TIER_IMMUTABLE
