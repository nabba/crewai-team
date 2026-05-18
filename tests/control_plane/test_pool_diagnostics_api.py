"""Tests for the /api/cp/pool/diagnostics REST endpoint.

PR 2 (2026-05-16). Pins the surface shape so dashboards built against
it don't break when the underlying counter dict gains new keys.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_diagnostics():
    from app.control_plane import db
    db.reset_pool_diagnostics()
    yield
    db.reset_pool_diagnostics()


def test_endpoint_returns_zero_counters_initially():
    from app.control_plane.dashboard_routes_ops_misc import pool_diagnostics

    result = pool_diagnostics()

    assert "counters" in result
    assert "maxconn" in result
    assert "utilisation" in result
    assert "peak_utilisation" in result

    counters = result["counters"]
    assert counters["acquires_total"] == 0
    assert counters["current_borrows"] == 0
    assert counters["peak_borrows"] == 0
    assert counters["failures_pool_exhausted"] == 0


def test_endpoint_reflects_counter_updates(monkeypatch):
    """Diagnostics endpoint reads the same shared counters as production code."""
    from app.control_plane import db
    from app.control_plane.dashboard_routes_ops_misc import pool_diagnostics

    db._diag_record_failure("pool_exhausted")
    db._diag_record_failure("pool_exhausted")
    db._diag_record_failure("connection_error")

    result = pool_diagnostics()
    counters = result["counters"]
    assert counters["failures_pool_exhausted"] == 2
    assert counters["failures_connection_error"] == 1


def test_endpoint_computes_utilisation(monkeypatch):
    """Utilisation = current_borrows / maxconn, rounded to 3 decimals."""
    from app.control_plane import db
    from app.control_plane.dashboard_routes_ops_misc import pool_diagnostics

    monkeypatch.setenv("CONTROL_PLANE_POOL_MAX", "10")
    # Simulate 3 concurrent borrows
    db._diag_record_acquire(0.01)
    db._diag_record_acquire(0.01)
    db._diag_record_acquire(0.01)

    result = pool_diagnostics()
    assert result["maxconn"] == 10
    assert result["utilisation"] == 0.3
    assert result["peak_utilisation"] == 0.3


def test_endpoint_handles_diagnostics_failure(monkeypatch):
    """If the underlying counter read fails, the endpoint returns
    a structured error instead of 500."""
    from app.control_plane import dashboard_api

    def boom():
        raise RuntimeError("counter access failed")

    monkeypatch.setattr(
        "app.control_plane.db.get_pool_diagnostics", boom,
    )

    result = dashboard_api.pool_diagnostics()
    assert "error" in result
    assert result["counters"] is None
