"""Tests for app.system_inventory — the auto-catalogue that closes the
CLAUDE.md / actual-capability drift gap.

Verifies the AST scan extracts module shape correctly, the persistence
round-trips, queries filter as advertised, and the summary line is
sensibly formed.
"""
from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture
def fake_app(tmp_path: Path) -> Path:
    """Build a small ``app/`` tree we can scan deterministically."""
    app_root = tmp_path / "app"
    app_root.mkdir()
    (app_root / "__init__.py").write_text('"""Root package."""\n')

    (app_root / "alpha.py").write_text(dedent('''
        """Alpha module — does the alpha thing.

        Second paragraph; should not appear in summary.
        """

        def public_one(x: int) -> int:
            """Return x doubled."""
            return x * 2

        async def public_two(y: str) -> str:
            return y.upper()

        def _private_helper():
            pass

        class PublicClass:
            """A public class."""
            pass
    ''').lstrip())

    (app_root / "beta.py").write_text(dedent('''
        """Beta uses register_tool capabilities."""
        from app.tool_registry import register_tool

        @register_tool(name="beta_tool", capabilities=["does-beta", "reads-x"])
        def _factory():
            return None

        @register_tool(name="beta_other", capability="legacy-singular")
        def _factory2():
            return None
    ''').lstrip())

    pkg = app_root / "sub"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""Sub-package."""\n')

    # Add a malformed file — scan must keep going.
    (app_root / "broken.py").write_text("def oops(:\n    pass\n")

    return app_root


def test_build_snapshot_walks_app_tree(fake_app: Path):
    from app.system_inventory import build_snapshot

    snap = build_snapshot(app_root=fake_app)

    paths = {m.path for m in snap.modules}
    assert any(p.endswith("app/alpha.py") for p in paths)
    assert any(p.endswith("app/beta.py") for p in paths)
    assert any(p.endswith("app/sub/__init__.py") for p in paths)
    # broken.py is skipped — SyntaxError is failure-isolated.
    assert not any(p.endswith("broken.py") for p in paths)
    assert snap.n_modules >= 3
    assert snap.n_packages >= 2


def test_summary_extracts_first_docstring_line_only(fake_app: Path):
    from app.system_inventory import build_snapshot

    snap = build_snapshot(app_root=fake_app)
    alpha = next(m for m in snap.modules if m.path.endswith("alpha.py"))
    assert alpha.summary == "Alpha module — does the alpha thing."
    assert "Second paragraph" not in alpha.summary


def test_public_symbols_excludes_underscore_prefix(fake_app: Path):
    from app.system_inventory import build_snapshot

    snap = build_snapshot(app_root=fake_app)
    alpha = next(m for m in snap.modules if m.path.endswith("alpha.py"))
    assert "public_one" in alpha.public_symbols
    assert "public_two" in alpha.public_symbols
    assert "PublicClass" in alpha.public_symbols
    assert "_private_helper" not in alpha.public_symbols


def test_capabilities_extracted_from_list_and_singular_kwargs(fake_app: Path):
    from app.system_inventory import build_snapshot

    snap = build_snapshot(app_root=fake_app)
    beta = next(m for m in snap.modules if m.path.endswith("beta.py"))
    assert "does-beta" in beta.capabilities
    assert "reads-x" in beta.capabilities
    assert "legacy-singular" in beta.capabilities


def test_kind_is_package_for_dunder_init(fake_app: Path):
    from app.system_inventory import build_snapshot

    snap = build_snapshot(app_root=fake_app)
    init = next(m for m in snap.modules if m.path.endswith("app/__init__.py"))
    assert init.kind == "package"
    alpha = next(m for m in snap.modules if m.path.endswith("alpha.py"))
    assert alpha.kind == "module"


def test_persist_and_load_round_trip(fake_app: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "ws"))
    # Re-import so the lazy WORKSPACE_ROOT helper picks up the new env.
    import importlib
    from app.system_inventory import store as store_mod
    importlib.reload(store_mod)

    from app.system_inventory.scanner import build_snapshot

    snap = build_snapshot(app_root=fake_app)
    store_mod.persist_snapshot(snap)

    snap_path = tmp_path / "ws" / "system_inventory" / "snapshot.json"
    assert snap_path.exists()

    reloaded = store_mod.get_snapshot(rebuild_if_missing=False)
    assert reloaded is not None
    assert reloaded.n_modules == snap.n_modules
    # Round-trip preserves capability extraction.
    beta = next(m for m in reloaded.modules if m.path.endswith("beta.py"))
    assert "does-beta" in beta.capabilities


def test_query_filters_by_keyword_and_kind(fake_app: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "ws"))
    import importlib
    from app.system_inventory import store as store_mod
    importlib.reload(store_mod)

    from app.system_inventory.scanner import build_snapshot

    snap = build_snapshot(app_root=fake_app)
    store_mod.persist_snapshot(snap)

    # keyword: should find alpha by path substring
    hits = store_mod.query_inventory(keyword="alpha")
    assert any(h.path.endswith("alpha.py") for h in hits)

    # capability filter
    hits = store_mod.query_inventory(capability="does-beta")
    assert any(h.path.endswith("beta.py") for h in hits)
    assert all("does-beta" in h.capabilities for h in hits)

    # kind filter
    pkgs = store_mod.query_inventory(kind="package")
    assert all(p.kind == "package" for p in pkgs)
    mods = store_mod.query_inventory(kind="module")
    assert all(m.kind == "module" for m in mods)


def test_inventory_summary_is_one_line(fake_app: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "ws"))
    import importlib
    from app.system_inventory import store as store_mod
    importlib.reload(store_mod)

    from app.system_inventory.scanner import build_snapshot

    store_mod.persist_snapshot(build_snapshot(app_root=fake_app))
    line = store_mod.inventory_summary()
    assert "system_inventory@" in line
    assert "modules" in line
    assert "\n" not in line  # truly one line
