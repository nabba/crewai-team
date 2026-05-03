"""Tests for Phase 4d — /api/cp/tools/flags diagnostics endpoint.

Phase 4d ships the deferred flags endpoint that lets operators see
at a glance which migrated agents are running on which path. The
endpoint reads the same env vars that the agent factories
themselves consult, so what it shows IS what the agents do.

Covers:
  * **default state** — both vars unset, every agent on legacy.
  * **master flag** — sets default-on for unflagged agents.
  * **per-agent override** — `=1` enables despite master off;
    `=0` disables despite master on.
  * **shape** — response includes count_loadable / count_legacy
    summary for the React UI.

Auth is bypassed (GATEWAY_AUTH_REQUIRED unset) for these tests; the
production auth gate is tested separately.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Start a TestClient with a clean env (no flags set)."""
    monkeypatch.setenv("GATEWAY_AUTH_REQUIRED", "0")
    for var in (
        "LOADABLE_AGENT_EXPERIMENTAL",
        "LOADABLE_INTROSPECTOR",
        "LOADABLE_RESEARCHER",
        "LOADABLE_WRITER",
        "LOADABLE_CODER",
    ):
        monkeypatch.delenv(var, raising=False)
    from app.main import app
    return TestClient(app)


class TestFlagsEndpoint:

    def test_default_state_all_legacy(self, client):
        """Default: both master and per-agent vars unset → all
        migrated agents on legacy path."""
        r = client.get("/api/cp/tools/flags")
        assert r.status_code == 200
        data = r.json()
        assert data["master_flag"] is False
        assert data["count_loadable"] == 0
        assert data["count_legacy"] == 4
        for row in data["migrated_agents"]:
            assert row["loadable"] is False
            assert row["source"] == "default"
            assert row["explicit_flag"] is None

    def test_master_flag_enables_all_unflagged(self, client, monkeypatch):
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        r = client.get("/api/cp/tools/flags")
        data = r.json()
        assert data["master_flag"] is True
        assert data["count_loadable"] == 4
        for row in data["migrated_agents"]:
            assert row["loadable"] is True
            assert row["source"] == "master flag"
            assert row["explicit_flag"] is None

    def test_per_agent_override_off_with_master_on(self, client, monkeypatch):
        """Master ON + LOADABLE_RESEARCHER=0 → researcher legacy,
        others loadable."""
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.setenv("LOADABLE_RESEARCHER", "0")
        r = client.get("/api/cp/tools/flags")
        data = r.json()
        assert data["master_flag"] is True
        # 3 loadable, 1 legacy (researcher)
        assert data["count_loadable"] == 3
        assert data["count_legacy"] == 1

        researcher_row = next(
            row for row in data["migrated_agents"] if row["agent"] == "researcher"
        )
        assert researcher_row["loadable"] is False
        assert researcher_row["source"] == "per-agent override"
        assert researcher_row["explicit_flag"] == "0"

    def test_per_agent_override_on_with_master_off(self, client, monkeypatch):
        """Master OFF + LOADABLE_CODER=1 → coder loadable, others legacy."""
        monkeypatch.setenv("LOADABLE_CODER", "1")
        r = client.get("/api/cp/tools/flags")
        data = r.json()
        assert data["master_flag"] is False
        assert data["count_loadable"] == 1
        assert data["count_legacy"] == 3

        coder_row = next(
            row for row in data["migrated_agents"] if row["agent"] == "coder"
        )
        assert coder_row["loadable"] is True
        assert coder_row["source"] == "per-agent override"
        assert coder_row["explicit_flag"] == "1"

    def test_response_shape(self, client):
        """Top-level keys for React rendering."""
        r = client.get("/api/cp/tools/flags")
        data = r.json()
        assert set(data.keys()) == {
            "master_flag", "migrated_agents",
            "count_loadable", "count_legacy",
        }
        assert isinstance(data["migrated_agents"], list)
        assert len(data["migrated_agents"]) >= 4
        for row in data["migrated_agents"]:
            assert set(row.keys()) >= {
                "agent", "loadable", "source", "explicit_flag",
            }
            assert row["source"] in ("default", "master flag", "per-agent override")

    def test_all_phase_4_agents_listed(self, client):
        """Every Phase 4 migration must appear in the listing."""
        r = client.get("/api/cp/tools/flags")
        agents = {row["agent"] for row in r.json()["migrated_agents"]}
        # Phase 2 + 4a + 4b + 4c
        assert "introspector" in agents
        assert "researcher" in agents
        assert "writer" in agents
        assert "coder" in agents
