"""
Tests for hierarchical workspace isolation.

Architecture: project-local workspaces + global meta-workspace.
  - "generic" workspace (cap=5): default for non-project tasks
  - Project workspaces (cap=3): PLG, Archibal, KaiCart, dynamic
  - "__meta__" workspace (cap=7): aggregates top items from all projects

Total: ~30 tests
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "chromadb", "chromadb.config", "chromadb.utils",
             "chromadb.utils.embedding_functions",
             "app.control_plane", "app.control_plane.db",
             "app.memory", "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
sys.modules["app.control_plane.db"].execute = MagicMock(return_value=[])

import pytest


def _reset_workspace_state():
    """Reset module-level state between tests."""
    import app.consciousness.workspace_buffer as wb
    wb._gates.clear()
    wb._scorer = None


class TestPerProjectWorkspaces:
    """Each project should get its own isolated workspace."""

    def setup_method(self):
        _reset_workspace_state()

    def test_generic_workspace_default(self):
        from app.consciousness.workspace_buffer import get_workspace_gate, GENERIC_WORKSPACE
        gate = get_workspace_gate()
        assert gate is get_workspace_gate(GENERIC_WORKSPACE)

    def test_generic_capacity(self):
        from app.consciousness.workspace_buffer import get_workspace_gate
        gate = get_workspace_gate("generic")
        # Generic uses config default (usually 5)
        assert gate.capacity >= 3

    def test_project_workspace_isolated(self):
        from app.consciousness.workspace_buffer import get_workspace_gate
        plg = get_workspace_gate("plg")
        archibal = get_workspace_gate("archibal")
        assert plg is not archibal

    def test_project_workspace_capacity(self):
        from app.consciousness.workspace_buffer import get_workspace_gate
        plg = get_workspace_gate("plg")
        assert plg.capacity == 3  # Default project capacity

    def test_meta_workspace_larger(self):
        from app.consciousness.workspace_buffer import get_workspace_gate, META_WORKSPACE
        meta = get_workspace_gate(META_WORKSPACE)
        assert meta.capacity == 7

    def test_same_project_returns_same_gate(self):
        from app.consciousness.workspace_buffer import get_workspace_gate
        g1 = get_workspace_gate("plg")
        g2 = get_workspace_gate("plg")
        assert g1 is g2

    def test_none_project_returns_generic(self):
        from app.consciousness.workspace_buffer import get_workspace_gate
        g1 = get_workspace_gate(None)
        g2 = get_workspace_gate("generic")
        assert g1 is g2


class TestDynamicWorkspaceCreation:
    """New workspaces should be creatable on demand."""

    def setup_method(self):
        _reset_workspace_state()

    def test_create_workspace(self):
        from app.consciousness.workspace_buffer import create_workspace
        gate = create_workspace("new_venture", capacity=4)
        assert gate.capacity == 4

    def test_create_workspace_idempotent(self):
        from app.consciousness.workspace_buffer import create_workspace
        g1 = create_workspace("venture_x", capacity=4)
        g2 = create_workspace("venture_x", capacity=6)  # Should return existing
        assert g1 is g2
        assert g1.capacity == 4  # First creation wins

    def test_create_workspace_bounded(self):
        from app.consciousness.workspace_buffer import create_workspace
        gate = create_workspace("extreme", capacity=20)
        assert gate.capacity == 9  # Clamped

    def test_list_workspaces(self):
        from app.consciousness.workspace_buffer import (
            get_workspace_gate, create_workspace, list_workspaces,
        )
        get_workspace_gate("generic")
        create_workspace("plg")
        create_workspace("archibal")
        ws = list_workspaces()
        assert "generic" in ws
        assert "plg" in ws
        assert "archibal" in ws


class TestMetaWorkspacePromotion:
    """Top items from project workspaces should promote to meta-workspace."""

    def setup_method(self):
        _reset_workspace_state()

    def test_promote_from_empty_project(self):
        from app.consciousness.meta_workspace import MetaWorkspace
        from app.consciousness.workspace_buffer import get_workspace_gate
        get_workspace_gate("plg")  # Create empty project workspace
        meta = MetaWorkspace()
        assert meta.promote_from_project("plg") is False

    def test_promote_from_project_with_items(self):
        from app.consciousness.meta_workspace import MetaWorkspace
        from app.consciousness.workspace_buffer import (
            get_workspace_gate, WorkspaceItem, META_WORKSPACE,
        )
        # Add item to project workspace
        plg_gate = get_workspace_gate("plg")
        item = WorkspaceItem(content="PLG ticket analysis", salience_score=0.8)
        plg_gate.evaluate(item)

        meta = MetaWorkspace()
        promoted = meta.promote_from_project("plg")
        assert promoted is True

        # Check meta-workspace has the item
        meta_gate = get_workspace_gate(META_WORKSPACE)
        assert len(meta_gate.active_items) == 1
        assert "[plg]" in meta_gate.active_items[0].content

    def test_promote_all(self):
        from app.consciousness.meta_workspace import MetaWorkspace
        from app.consciousness.workspace_buffer import (
            get_workspace_gate, WorkspaceItem,
        )
        # Items in two projects
        for pid in ("plg", "archibal"):
            gate = get_workspace_gate(pid)
            gate.evaluate(WorkspaceItem(content=f"{pid} task", salience_score=0.7))

        meta = MetaWorkspace()
        results = meta.promote_all()
        assert results.get("plg") is True
        assert results.get("archibal") is True

    def test_promote_does_not_remove_from_project(self):
        from app.consciousness.meta_workspace import MetaWorkspace
        from app.consciousness.workspace_buffer import (
            get_workspace_gate, WorkspaceItem,
        )
        plg_gate = get_workspace_gate("plg")
        plg_gate.evaluate(WorkspaceItem(content="Keep me", salience_score=0.9))

        MetaWorkspace().promote_from_project("plg")

        # Original still in project workspace
        assert len(plg_gate.active_items) == 1

    def test_cross_project_snapshot(self):
        from app.consciousness.meta_workspace import MetaWorkspace
        from app.consciousness.workspace_buffer import (
            get_workspace_gate, WorkspaceItem,
        )
        for pid in ("generic", "plg"):
            gate = get_workspace_gate(pid)
            gate.evaluate(WorkspaceItem(content=f"{pid} item", salience_score=0.6))

        meta = MetaWorkspace()
        meta.promote_all()
        snapshot = meta.get_cross_project_snapshot()
        assert "by_project" in snapshot
        assert snapshot["project_count"] >= 2

    def test_meta_does_not_promote_into_itself(self):
        from app.consciousness.meta_workspace import MetaWorkspace
        from app.consciousness.workspace_buffer import (
            get_workspace_gate, WorkspaceItem, META_WORKSPACE,
        )
        meta_gate = get_workspace_gate(META_WORKSPACE)
        meta_gate.evaluate(WorkspaceItem(content="meta item", salience_score=0.5))

        meta = MetaWorkspace()
        results = meta.promote_all()
        assert META_WORKSPACE not in results


class TestBroadcastIsolation:
    """Each project should have its own broadcast engine."""

    def test_project_engines_isolated(self):
        import app.consciousness.global_broadcast as gb
        gb._engines.clear()
        e1 = gb.get_broadcast_engine("plg")
        e2 = gb.get_broadcast_engine("archibal")
        assert e1 is not e2
        gb._engines.clear()

    def test_none_returns_generic(self):
        import app.consciousness.global_broadcast as gb
        gb._engines.clear()
        e1 = gb.get_broadcast_engine(None)
        e2 = gb.get_broadcast_engine("generic")
        assert e1 is e2
        gb._engines.clear()

    def test_engines_have_default_listeners(self):
        import app.consciousness.global_broadcast as gb
        gb._engines.clear()
        engine = gb.get_broadcast_engine("plg")
        assert len(engine._listeners) == 4  # researcher, coder, writer, media_analyst
        gb._engines.clear()


class TestOrchestratorWiring:
    """Verify orchestrator passes project_id to consciousness modules."""

    def test_orchestrator_uses_project_scoped_gate(self):
        src = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        assert "get_workspace_gate(_project_id)" in src

    def test_orchestrator_uses_project_scoped_broadcast(self):
        src = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        assert "get_broadcast_engine(_project_id)" in src

    def test_orchestrator_promotes_to_meta(self):
        src = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        assert "promote_from_project" in src

    def test_fallback_to_generic(self):
        src = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        assert '_project_id = "generic"' in src

    def test_meta_promotion_in_idle_scheduler(self):
        src = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "meta-workspace-promotion" in src
        assert "promote_all" in src

    def test_workspace_api_registered(self):
        src = (Path(__file__).parent.parent / "app" / "main.py").read_text()
        assert "workspace_router" in src
