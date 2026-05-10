"""Tests for the coding_session_evolve_solution tool registration.

Smoke tests focused on the tool's wiring + delegation to the bridge.
The bridge itself is covered in test_evolution_bridge.py.
"""

from __future__ import annotations

import json

from app.tools.coding_session_tools import create_coding_session_tools


def test_factory_returns_eight_tools() -> None:
    tools = create_coding_session_tools()
    names = [t.name for t in tools]
    assert "coding_session_evolve_solution" in names
    assert len(tools) == 8


def test_evolve_tool_has_expected_args_schema() -> None:
    tools = create_coding_session_tools()
    evolve = next(t for t in tools if t.name == "coding_session_evolve_solution")
    fields = evolve.args_schema.model_fields
    assert "session_id" in fields
    assert "initial_path" in fields
    assert "evaluate_path" in fields
    assert "num_generations" in fields
    assert "num_islands" in fields
    assert "max_cost_usd" in fields


def test_evolve_tool_refuses_unknown_session() -> None:
    tools = create_coding_session_tools()
    evolve = next(t for t in tools if t.name == "coding_session_evolve_solution")
    out = evolve._run(
        session_id="nonexistent",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
    )
    # The bridge surfaces refused via REFUSED: prefix.
    assert "REFUSED:" in out or "ERROR:" in out


def test_evolve_tool_describes_caps_and_safety() -> None:
    tools = create_coding_session_tools()
    evolve = next(t for t in tools if t.name == "coding_session_evolve_solution")
    desc = evolve.description.lower()
    assert "tier_immutable" in desc
    assert "subia" in desc
    assert "20 generation" in desc or "20 generations" in desc


def test_evolve_tool_returns_json_for_non_refusal_paths(monkeypatch, tmp_path) -> None:
    """Even for an unknown session, only refusal returns the prefix string;
    other branches return JSON. Verify by patching evolve_in_session to
    return an "improved" result."""
    from app.coding_session.evolution_bridge import EvolutionResult

    def fake_evolve(**_):
        return EvolutionResult(
            status="improved",
            baseline_score=0.5,
            best_score=0.9,
            delta=0.4,
            diff="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
            generations_run=5,
            variants_evaluated=20,
            duration_seconds=12.3,
        )

    import app.tools.coding_session_tools as cst
    # The tool imports evolve_in_session inside _run via a local import.
    # Patch the bridge module's symbol.
    import app.coding_session.evolution_bridge as bridge
    monkeypatch.setattr(bridge, "evolve_in_session", fake_evolve)

    tools = create_coding_session_tools()
    evolve = next(t for t in tools if t.name == "coding_session_evolve_solution")
    out = evolve._run(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
    )
    payload = json.loads(out)
    assert payload["status"] == "improved"
    assert payload["delta"] == 0.4
    assert "x = 2" in payload["diff"]
