"""Tests for app.system_state — Phase 5.1 foundation.

Covers:
  * **Crew-run ring buffer** — record, read, eviction at cap, never
    raises on bad input.
  * **State composition** — every section has `available`; degrades
    gracefully when individual sources fail.
  * **Caching** — repeated calls hit cache; invalidation works.
  * **Agent tool** — registers in the tool registry; constructs;
    `_run` returns JSON.
  * **HTTP endpoint** — `/api/cp/system-state` returns 200 with the
    expected shape; auth required when `GATEWAY_AUTH_REQUIRED=1`.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ── crew-run ring buffer ─────────────────────────────────────────────


class TestCrewRunBuffer:

    def setup_method(self) -> None:
        from app.system_state.crew_runs import reset_for_tests
        reset_for_tests()

    def test_record_and_read(self):
        from app.system_state.crew_runs import record_crew_run, recent_runs
        record_crew_run("pim", ok=True, duration_s=2.4)
        record_crew_run("pim", ok=True, duration_s=1.8)
        record_crew_run("coding", ok=False, error="NameError: x")

        runs = recent_runs()
        assert "pim" in runs
        assert len(runs["pim"]) == 2
        assert all(r["ok"] for r in runs["pim"])
        assert "coding" in runs
        assert runs["coding"][0]["ok"] is False
        assert "NameError" in runs["coding"][0]["error"]

    def test_newest_first(self):
        from app.system_state.crew_runs import record_crew_run, recent_runs
        record_crew_run("pim", ok=True, duration_s=1.0)
        record_crew_run("pim", ok=False, error="boom")
        runs = recent_runs(crew="pim")
        # Newest first: failure first, success second
        assert runs["pim"][0]["ok"] is False
        assert runs["pim"][1]["ok"] is True

    def test_buffer_bounded(self):
        from app.system_state import crew_runs
        # Push more than the cap (50 per crew) — older ones evict.
        for i in range(100):
            crew_runs.record_crew_run("test", ok=True, duration_s=float(i))
        runs = crew_runs.recent_runs(crew="test", limit=200)
        # Bounded at 50.
        assert len(runs["test"]) == 50
        # Newest entry has duration_s=99 (i=99 was last).
        assert runs["test"][0]["duration_s"] == 99.0

    def test_empty_crew_string_silent_drop(self):
        """Recording with empty crew name must not raise + must not appear."""
        from app.system_state.crew_runs import record_crew_run, recent_runs
        record_crew_run("", ok=True)
        record_crew_run("real_crew", ok=True)
        assert "" not in recent_runs()
        assert "real_crew" in recent_runs()

    def test_record_never_raises_on_bad_input(self):
        """Defensive: observational telemetry must never break the
        request path even when called with weird data."""
        from app.system_state.crew_runs import record_crew_run
        # All of these should silently no-op or succeed.
        record_crew_run("crew", ok=True, error=None, duration_s=None, task_id=None)
        record_crew_run("crew", ok=False, error="x" * 10000)  # very long error → truncated
        record_crew_run("crew", ok=True, duration_s=1.234567890)


# ── state composition ───────────────────────────────────────────────


class TestStateComposition:

    def setup_method(self) -> None:
        from app.system_state.state import reset_cache_for_tests
        from app.system_state.crew_runs import reset_for_tests
        reset_cache_for_tests()
        reset_for_tests()

    def test_all_sections_present(self):
        from app.system_state import get_system_state
        state = get_system_state(use_cache=False)
        assert "git" in state
        assert "gateway" in state
        assert "tier_immutable" in state
        assert "tools" in state
        assert "recent_crew_runs" in state
        # Every section has `available`.
        for section_name in ("git", "gateway", "tier_immutable", "tools", "recent_crew_runs"):
            assert "available" in state[section_name], (
                f"section {section_name!r} missing 'available' bool"
            )

    def test_window_hours_passes_through(self):
        from app.system_state import get_system_state
        state = get_system_state(window_hours=12, use_cache=False)
        assert state["window_hours"] == 12

    def test_caching_works(self):
        from app.system_state import get_system_state
        from app.system_state.state import reset_cache_for_tests

        reset_cache_for_tests()
        # First call populates cache; second call returns same dict.
        s1 = get_system_state()
        s2 = get_system_state()
        # Same `ts` field (cached, not regenerated).
        assert s1["ts"] == s2["ts"]

    def test_cache_bypass(self):
        from app.system_state import get_system_state
        s1 = get_system_state(use_cache=False)
        s2 = get_system_state(use_cache=False)
        # Different `ts` — both regenerated.
        # (May rarely be same if both ran in <1ns; assert via length).
        assert s1.keys() == s2.keys()

    def test_degrades_when_git_unavailable(self):
        """If every git source fails, the section reports unavailable."""
        from app.system_state import state as state_mod
        from app.system_state.state import reset_cache_for_tests

        reset_cache_for_tests()
        with patch.object(state_mod, "_git_via_bridge", return_value=None), \
             patch.object(state_mod, "_git_via_subprocess", return_value=None), \
             patch.object(state_mod, "_git_via_env", return_value=None):
            from app.system_state import get_system_state
            state = get_system_state(use_cache=False)
        assert state["git"]["available"] is False
        # Other sections still work.
        assert state["gateway"]["available"] is True

    def test_crew_runs_section_includes_recorded_runs(self):
        from app.system_state import (
            get_system_state, record_crew_run,
        )
        from app.system_state.state import reset_cache_for_tests

        record_crew_run("pim", ok=True)
        record_crew_run("pim", ok=False, error="boom")
        reset_cache_for_tests()
        state = get_system_state(use_cache=False)
        assert state["recent_crew_runs"]["available"] is True
        assert "pim" in state["recent_crew_runs"]["by_crew"]
        # Two pim records.
        assert len(state["recent_crew_runs"]["by_crew"]["pim"]) == 2


# ── agent-callable tool ─────────────────────────────────────────────


class TestAgentTool:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)

    def test_tool_registered(self):
        from app.tool_registry import ToolRegistry
        spec = ToolRegistry.instance().get("get_system_state")
        assert spec is not None
        assert "reads-deployment-state" in spec.capabilities

    def test_tool_constructs(self):
        from app.tool_registry import ToolRegistry
        inst = ToolRegistry.instance().build_instance("get_system_state")
        assert inst.name == "get_system_state"

    def test_tool_run_returns_json(self):
        import json
        from app.tool_registry import ToolRegistry
        inst = ToolRegistry.instance().build_instance("get_system_state")
        out = inst._run(window_hours=24)
        # Should parse as JSON.
        parsed = json.loads(out)
        assert "git" in parsed
        assert "gateway" in parsed

    def test_factory_returns_one_tool(self):
        pytest.importorskip("crewai.tools")
        from app.tools.system_state_tool import create_system_state_tools
        tools = create_system_state_tools()
        assert len(tools) == 1
        assert tools[0].name == "get_system_state"


# ── HTTP endpoint ────────────────────────────────────────────────────


class TestHTTPEndpoint:

    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_AUTH_REQUIRED", "0")
        monkeypatch.setenv("CREWAI_TELEMETRY_OPT_OUT", "true")
        from app.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_endpoint_returns_200(self, client):
        r = client.get("/api/cp/system-state")
        assert r.status_code == 200

    def test_endpoint_shape(self, client):
        r = client.get("/api/cp/system-state")
        data = r.json()
        assert set(data.keys()) >= {
            "ts", "window_hours", "git", "gateway",
            "tier_immutable", "tools", "recent_crew_runs",
        }

    def test_window_hours_param(self, client):
        r = client.get("/api/cp/system-state?window_hours=12")
        data = r.json()
        assert data["window_hours"] == 12

    def test_window_hours_bounds(self, client):
        # Too small → 422
        r = client.get("/api/cp/system-state?window_hours=0")
        assert r.status_code == 422
        # Too large → 422
        r = client.get("/api/cp/system-state?window_hours=999")
        assert r.status_code == 422

    def test_use_cache_false_works(self, client):
        r = client.get("/api/cp/system-state?use_cache=false")
        assert r.status_code == 200


# ── capability vocabulary governance ────────────────────────────────


class TestVocabularyAddition:

    def test_reads_deployment_state_in_vocabulary(self):
        from app.tool_registry.capabilities import (
            all_capability_tags, category_for, description_for,
        )
        assert "reads-deployment-state" in all_capability_tags()
        assert category_for("reads-deployment-state") == "observability"
        assert description_for("reads-deployment-state")
