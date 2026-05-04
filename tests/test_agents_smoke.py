"""Smoke test: every agent factory in app/agents/* must construct
without a NameError.

Motivation
----------
2026-05-04 — PIM crew failed in production with::

    Crew pim failed: name 'optional_tool_group' is not defined

Latent bug introduced 2026-05-01 (commit 5401d3ea, "phase E
observability") which added ``optional_tool_group(...)`` wrappers
inside 4 agent factories without importing the helper. The bug sat
for 3 days; nobody hit it until a calendar question routed to PIM.

This test catches the **whole class** of bug — any agent factory
that uses an undefined symbol in its body fails here, not in
production. It runs at the cost of constructing every factory once
(~5-10s in the container).

What the test asserts
---------------------
For every ``def create_<name>(...)`` exported by an ``app.agents.*``
module:

  * Calling the factory with sensible defaults must NOT raise
    ``NameError``.
  * ANY OTHER exception is allowed — factories may legitimately fail
    when env config is missing (e.g. PIM without email service,
    GEE without credentials). Those failures surface elsewhere.

The asymmetry is deliberate: ``NameError`` always indicates a code
bug, while other exceptions can be operational. The test enforces
the former, ignores the latter.
"""
from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from typing import Any

import pytest


def _agent_factories() -> list[tuple[str, Any]]:
    """Discover every public ``create_<name>`` in ``app.agents.*``.

    Returns ``[(qualified_name, callable), ...]`` sorted for stable
    test ordering. Skips factories defined as private (leading
    underscore) and skips the ``app.agents.commander`` package
    (Commander is a class-based orchestrator, not an Agent factory —
    see Phase 4d doc).
    """
    import app.agents as agents_pkg

    out: list[tuple[str, Any]] = []
    for mod_info in pkgutil.walk_packages(agents_pkg.__path__, prefix="app.agents."):
        # Skip the commander subpackage and its modules — orchestrator
        # is class-based, not a CrewAI Agent factory.
        if mod_info.name.startswith("app.agents.commander"):
            continue
        try:
            mod = importlib.import_module(mod_info.name)
        except Exception as exc:
            # If the module itself fails to import, surface that as a
            # named test failure rather than swallowing.
            out.append((f"{mod_info.name}::__import__", _make_import_failer(mod_info.name, exc)))
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("create_"):
                continue
            if name.startswith("_"):
                continue
            # Stay within the agents package — don't pick up
            # re-exports of helpers from other modules.
            if obj.__module__ != mod_info.name:
                continue
            out.append((f"{mod_info.name}.{name}", obj))
    return sorted(out, key=lambda x: x[0])


def _make_import_failer(mod_name: str, exc: Exception):
    """Return a synthetic 'factory' that just re-raises the import
    error, so test discovery shows it as a per-module failure rather
    than vanishing the whole module."""
    def _failer():
        raise exc
    _failer.__name__ = f"{mod_name}_import"
    _failer.__doc__ = f"synthetic re-raise for failed import of {mod_name}"
    return _failer


# Materialize once at module-load — pytest IDs come from this.
_FACTORIES = _agent_factories()


@pytest.fixture(autouse=True)
def _silence_crewai_telemetry(monkeypatch):
    monkeypatch.setenv("CREWAI_TELEMETRY_OPT_OUT", "true")


@pytest.mark.parametrize("name,factory", _FACTORIES, ids=[n for n, _ in _FACTORIES])
def test_agent_factory_constructs_without_nameerror(name: str, factory: Any) -> None:
    """Every agent factory must NOT raise NameError when called.

    Other exceptions are allowed — env-config-driven failures are
    operational, not bugs. A NameError specifically means an agent
    file references a symbol it never imported; the 2026-05-04 PIM
    incident is the historical case this test guards against.
    """
    try:
        factory()
    except NameError as exc:
        # The thing we're guarding against. Surface with the agent
        # name so failure output is actionable.
        pytest.fail(
            f"{name}() raised NameError — agent file is missing an "
            f"import. Symbol: {exc}. Add the missing import at the "
            f"top of the file. (See 2026-05-04 PIM incident in "
            f"PROGRAM.md for the historical case this test exists "
            f"to prevent.)"
        )
    except Exception:
        # Operational failure (missing env, unreachable service,
        # etc.) — not what this test guards. Pass.
        pass


def test_factory_discovery_finds_known_agents() -> None:
    """Sanity: the discovery walker found the factories we know
    exist. If this regresses to 0, the walker is broken."""
    factory_names = {n.split(".")[-1] for n, _ in _FACTORIES}
    # Anchor on a few stable names — these have been around since
    # before the 2026-05 work.
    expected = {
        "create_coder", "create_writer", "create_researcher",
        "create_pim_agent", "create_introspector",
    }
    missing = expected - factory_names
    assert not missing, (
        f"Factory discovery missed expected names: {missing}. "
        "Either the walker is broken or these factories were renamed."
    )
