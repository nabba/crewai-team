"""REST endpoints for the Q18 drill v2 surface (PROGRAM §57).

We test by invoking the route handlers directly with synthetic
fixtures — full FastAPI TestClient with auth would require pulling
in the entire app, which several of these tests don't need.

The dashboard_routes_sentience_drills module transitively imports
psycopg2 (via the control_plane.db chain). On hosts that don't have
psycopg2 installed (typical CI / dev laptop), the entire test file
is skipped — the route handlers still work in production because
the gateway image bundles psycopg2.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

# Skip the whole module when psycopg2 isn't available — the route
# handlers under test pull in control_plane.db which requires it.
psycopg2 = pytest.importorskip("psycopg2")


@pytest.fixture(autouse=True)
def isolated_dirs(monkeypatch, tmp_path):
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    # Also point CR store at temp.
    from app.change_requests import store
    monkeypatch.setattr(store, "_STORE_DIR", tmp_path / "change_requests")
    monkeypatch.setattr(store, "_AUDIT_LOG",
                         tmp_path / "change_requests" / "audit.jsonl")
    store.reset_for_tests()
    yield
    store.reset_for_tests()


@pytest.fixture
def fresh_registry():
    """Drop registered drills before each test.

    Eagerly imports the drills package FIRST so the module-level
    ``register(SPEC, run)`` side-effects run BEFORE our clear.
    Otherwise any subsequent test code that triggers a drills
    package import (route handlers, scheduler hooks) would
    cascade-register all 9 production drills and inflate
    registry queries the test makes."""
    try:
        import app.resilience_drills.drills  # noqa: F401 — register side-effect
    except Exception:
        pass
    from app.resilience_drills.protocol import get_registry
    reg = get_registry()
    reg.clear_for_tests()
    yield reg
    reg.clear_for_tests()


def _register_passing(reg, name="p"):
    from app.resilience_drills.protocol import (
        DrillResult, DrillSpec, DrillStatus, register,
    )
    spec = DrillSpec(name=name, cadence_days=90, warmup_days=0)
    def _r(*, dry_run=True):
        return DrillResult(
            drill_name=name, status=DrillStatus.PASS,
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0.01, dry_run=dry_run,
            observation={"k": 1},
        )
    register(spec, _r)
    return spec


def test_registry_endpoint_returns_v2_fields(fresh_registry):
    _register_passing(fresh_registry, "alpha")
    from app.control_plane.dashboard_routes_sentience_drills import drills_registry
    out = drills_registry()
    assert "drills" in out
    drills = out["drills"]
    assert len(drills) == 1
    d = drills[0]
    # Legacy §44 fields
    assert d["name"] == "alpha"
    assert "cadence_days" in d
    assert "risk" in d
    # Q18 fields
    assert "state" in d
    assert "consecutive_failures" in d
    assert "warmup_days" in d
    assert "has_baseline" in d
    assert "is_runnable_now" in d


def test_drill_detail_endpoint_returns_state_and_baseline(fresh_registry):
    _register_passing(fresh_registry, "alpha")
    from app.control_plane.dashboard_routes_sentience_drills import drill_detail
    out = drill_detail("alpha")
    assert out["spec"]["name"] == "alpha"
    assert "state" in out
    assert out["state"]["drill_name"] == "alpha"
    assert "baseline" in out
    assert "recent_results" in out
    assert "recent_observations" in out


def test_drill_detail_404_on_unknown(fresh_registry):
    from app.control_plane.dashboard_routes_sentience_drills import drill_detail
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        drill_detail("nonexistent")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_ratify_baseline_with_observation(fresh_registry):
    """Operator ratifies the most recent observation as baseline."""
    _register_passing(fresh_registry, "alpha")
    # Invoke the drill once to produce an observation
    from app.resilience_drills.runner import invoke_drill_by_name
    invoke_drill_by_name("alpha")

    from app.control_plane.dashboard_routes_sentience_drills import drills_ratify_baseline
    # Mock Request.json()
    req = AsyncMock()
    req.json = AsyncMock(return_value={
        "operator": "test",
        "tolerances": {"k": {"rule": "min", "value": 1}},
        "notes": "k≥1 is fine",
    })
    out = await drills_ratify_baseline("alpha", req)
    assert out["ok"] is True
    assert out["baseline"]["measurements"]["k"] == 1


@pytest.mark.asyncio
async def test_ratify_baseline_400_without_observation(fresh_registry):
    """Cannot ratify if no observation exists yet."""
    _register_passing(fresh_registry, "alpha")
    from app.control_plane.dashboard_routes_sentience_drills import drills_ratify_baseline
    from fastapi import HTTPException
    req = AsyncMock()
    req.json = AsyncMock(return_value={})
    with pytest.raises(HTTPException) as exc_info:
        await drills_ratify_baseline("alpha", req)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_unquarantine_endpoint(fresh_registry):
    from app.resilience_drills import state as st
    # Set up a quarantined drill
    rec = st.load_or_initialize("alpha", warmup_days=0)
    rec.state = st.DrillState.QUARANTINED
    rec.quarantined_at = "2026-05-18T00:00:00Z"
    st.save(rec)

    from app.control_plane.dashboard_routes_sentience_drills import drills_unquarantine
    req = AsyncMock()
    req.json = AsyncMock(return_value={"operator": "andrus", "reason": "fixed"})
    out = await drills_unquarantine("alpha", req)
    assert out["ok"] is True
    assert out["state"]["state"] == st.DrillState.WATCH.value


@pytest.mark.asyncio
async def test_unquarantine_404_unknown_drill():
    from app.control_plane.dashboard_routes_sentience_drills import drills_unquarantine
    from fastapi import HTTPException
    req = AsyncMock()
    req.json = AsyncMock(return_value={})
    with pytest.raises(HTTPException) as exc_info:
        await drills_unquarantine("nonexistent", req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_mute_then_unmute(fresh_registry):
    _register_passing(fresh_registry, "alpha")
    from app.control_plane.dashboard_routes_sentience_drills import (
        drills_mute, drills_unmute,
    )
    from app.resilience_drills import state as st

    req = AsyncMock()
    req.json = AsyncMock(return_value={"operator": "andrus", "reason": "too loud"})
    out = await drills_mute("alpha", req)
    assert out["ok"] is True
    rec = st.load("alpha")
    assert rec.state == st.DrillState.MUTED

    out = await drills_unmute("alpha", req)
    assert out["ok"] is True
    rec = st.load("alpha")
    assert rec.state == st.DrillState.HEALTHY


def test_drills_run_endpoint_uses_orchestrator(fresh_registry, monkeypatch):
    """The legacy /drills/run/{name} endpoint must go through the
    Q18 orchestrator so state transitions are recorded."""
    spec = _register_passing(fresh_registry, "beta")
    monkeypatch.setattr(
        "app.resilience_drills.protocol.drill_enabled",
        lambda s: True,
    )
    from app.control_plane.dashboard_routes_sentience_drills import drills_run
    out = drills_run("beta", None)
    assert out["status"] == "pass"
    # State was threaded
    from app.resilience_drills import state as st
    rec = st.load("beta")
    assert rec.last_success_at is not None
