"""Tests for the operational runbook handlers shipped 2026-05-09.

Covers:
  * ``compute_signature`` agrees with ``error_monitor._signature``
    so handlers actually fire on the hashes the monitor produces.
  * ``db_pool_reset`` calls ``_reset_pool`` exactly once per allowed
    window; rate-limited subsequent calls return without acting.
  * Specific-hash handlers refuse on sample-substring mismatch (defends
    against rare hash collisions).
  * ``multi_router`` dispatches to the correct sub-handler based on
    sample text alone.
"""
from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect ``workspace/self_heal/`` to a tmp dir so handlers don't
    touch the real workspace state files during tests.
    """
    from app.healing.handlers import _common
    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path)
    yield tmp_path


# ── compute_signature parity ──────────────────────────────────────────────


def test_compute_signature_matches_monitor_formula():
    """The hash ``compute_signature`` produces must equal what
    ``error_monitor._signature`` produces for the same record. Otherwise
    handlers register against ghost hashes that anomalies never carry.
    """
    from app.healing.handlers._common import compute_signature
    from app.observability.error_monitor import _signature

    cases = [
        (
            "app.control_plane.db",
            "control_plane SQL error: connection pool exhausted",
        ),
        (
            "root",
            "Anthropic API call failed: 'str' object has no attribute 'content'",
        ),
        (
            "apscheduler.executors.default",
            'Run time of job "lifespan.<locals>._heartbeat_tick (trigger: '
            'interval[0:01:00], next run at: 2026-05-02 06:58:17 UTC)" was '
            'missed by 0:00:05.387825',
        ),
    ]
    for module, message in cases:
        record = {"logger": module, "message": message}
        sig_from_monitor, _sample = _signature(record)
        sig_from_helper = compute_signature(module, message)
        assert sig_from_monitor == sig_from_helper, (
            f"sig mismatch for {module!r}: "
            f"monitor={sig_from_monitor!r} helper={sig_from_helper!r}"
        )


# ── db_pool_reset ─────────────────────────────────────────────────────────


def test_db_pool_reset_calls_reset_once_then_rate_limits(isolated_state):
    from app.healing.handlers import db_pool

    db_pool._LIMITER.reset()  # Fresh window for the test.

    fake_db = type("M", (), {"_reset_pool": lambda: None})()
    fake_db._reset_pool = lambda: setattr(fake_db, "_calls",
                                          getattr(fake_db, "_calls", 0) + 1)

    anomaly = {
        "pattern_signature": db_pool._SIGNATURE,
        "pattern_sample": (
            "control_plane SQL error: connection pool exhausted"
        ),
        "severity": "warning",
        "anomaly_type": "rate_spike",
    }

    with patch.object(db_pool, "audit_event", lambda *a, **k: None), \
         patch.object(db_pool, "send_signal_alert", lambda *a, **k: True), \
         patch("app.control_plane.db._reset_pool",
               new=fake_db._reset_pool):
        r1 = db_pool._handle_db_pool_reset(anomaly)
        r2 = db_pool._handle_db_pool_reset(anomaly)

    assert r1.success is True
    assert r1.detail.startswith("pool reset")
    assert r2.success is False
    assert r2.error == "rate_limited"
    assert getattr(fake_db, "_calls", 0) == 1


def test_db_pool_reset_refuses_on_sample_mismatch(isolated_state):
    from app.healing.handlers import db_pool
    db_pool._LIMITER.reset()
    anomaly = {
        "pattern_signature": db_pool._SIGNATURE,
        "pattern_sample": "completely different error text",
        "severity": "warning",
    }
    r = db_pool._handle_db_pool_reset(anomaly)
    assert r.success is False
    assert r.error == "sample_mismatch"


# ── multi_router substring dispatch ───────────────────────────────────────


def test_multi_router_dispatches_to_embed(isolated_state):
    from app.healing.handlers import multi_router

    captured: dict = {}

    def fake_embed(anomaly):
        from app.healing.runbooks import RunbookResult
        captured["embed"] = anomaly
        return RunbookResult(name="embed_model_misroute_alert", success=True,
                             detail="ok")

    with patch.object(multi_router, "_SUB_HANDLERS",
                      [(("does not support chat",), fake_embed)]):
        result = multi_router._handle({
            "pattern_sample": (
                'OpenAI API call failed: "nomic-embed-text:latest" does not '
                'support chat'
            ),
            "severity": "warning",
        })

    assert result.success is True
    assert result.name == "self_heal_router"
    assert result.extra.get("routed_to") == "embed_model_misroute_alert"
    assert "embed" in captured


def test_multi_router_handles_sub_handler_exception(isolated_state):
    from app.healing.handlers import multi_router

    def boom(anomaly):
        raise RuntimeError("kaboom")

    with patch.object(multi_router, "_SUB_HANDLERS",
                      [(("token X",), boom)]):
        result = multi_router._handle({
            "pattern_sample": "lol token X here",
        })

    assert result.success is False
    assert result.name == "self_heal_router"
    assert "RuntimeError" in (result.error or "")


def test_multi_router_returns_no_match_for_unknown(isolated_state):
    from app.healing.handlers import multi_router
    result = multi_router._handle({
        "pattern_sample": "totally unrelated message that nothing claims",
    })
    assert result.success is True
    assert "no sub-handler" in result.detail


# ── Registration order invariant ──────────────────────────────────────────


def test_registration_specific_hashes_before_catch_all():
    """The dispatcher returns the FIRST pattern.search match. Specific
    hashes MUST be registered before the catch-all router, otherwise the
    router shadows them.
    """
    from app.healing import runbooks
    # Trigger registration if not already done in this process.
    from app.healing import handlers  # noqa: F401  — side-effect import
    handlers.install()

    names = list(runbooks._REGISTERED_RUNBOOKS.keys())
    # ``self_heal_router`` should appear AFTER db_pool_reset in insertion order.
    if "db_pool_reset" in names and "self_heal_router" in names:
        assert names.index("db_pool_reset") < names.index("self_heal_router")


# ── Schema-drift CR filing path ───────────────────────────────────────────


def test_numeric_overflow_dedups_by_precision_scale(isolated_state, monkeypatch):
    """A repeat (precision, scale) overflow should NOT file a second CR."""
    from app.healing.handlers import schema_drift

    cr_calls: list = []

    def fake_cr(**kwargs):
        cr_calls.append(kwargs)
        return f"cr-{len(cr_calls)}"

    monkeypatch.setattr(schema_drift, "file_change_request", fake_cr)
    monkeypatch.setattr(schema_drift, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(schema_drift, "send_signal_alert", lambda *a, **k: True)

    anomaly = {
        "pattern_signature": schema_drift._E_SIGNATURE,
        "pattern_sample": (
            "control_plane SQL error: numeric field overflow\n"
            "DETAIL:  A field with precision 10, scale 6 must round to an "
            "absolute value less than 10^4."
        ),
        "severity": "warning",
    }

    r1 = schema_drift._handle_numeric_overflow(anomaly)
    r2 = schema_drift._handle_numeric_overflow(anomaly)

    assert r1.success is True
    assert r2.success is True
    # Second call detected a duplicate and didn't file another CR.
    assert len(cr_calls) == 1
    assert "already proposed" in r2.detail


# ── F now files a CR ──────────────────────────────────────────────────────


def test_missing_column_files_cr_first_time(isolated_state, monkeypatch):
    """First sighting of a missing column should file a CR with the
    placeholder migration file. Subsequent sightings dedup to the same CR.
    """
    from app.healing.handlers import schema_drift

    cr_calls: list = []

    def fake_cr(**kwargs):
        cr_calls.append(kwargs)
        return f"cr-{len(cr_calls)}"

    monkeypatch.setattr(schema_drift, "file_change_request", fake_cr)
    monkeypatch.setattr(schema_drift, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(schema_drift, "send_signal_alert", lambda *a, **k: True)

    anomaly = {
        "pattern_signature": "x" * 16,
        "pattern_sample": (
            'control_plane SQL error: column "cost_mode" does not exist'
        ),
        "severity": "warning",
    }

    r1 = schema_drift._handle_missing_column(anomaly)
    r2 = schema_drift._handle_missing_column(anomaly)

    assert r1.success is True
    assert r1.name == "schema_missing_column_cr"
    assert "filed CR" in r1.detail
    assert len(cr_calls) == 1
    # Path follows the migrations/<ts>_missing_column_<col>.sql pattern.
    assert cr_calls[0]["path"].startswith("migrations/")
    assert "cost_mode" in cr_calls[0]["path"]
    # Content is the placeholder marker (refs the column + alembic upgrade).
    assert "cost_mode" in cr_calls[0]["new_content"]
    assert "alembic upgrade head" in cr_calls[0]["new_content"]
    # Dedup on second occurrence — no second CR filed.
    assert "already proposed" in r2.detail


# ── G now files a CR ──────────────────────────────────────────────────────


def test_anthropic_str_content_files_cr_first_time(isolated_state, monkeypatch):
    """G now files a CR for the response-guard module, not just an alert."""
    from app.healing.handlers import code_drift

    cr_calls: list = []

    def fake_cr(**kwargs):
        cr_calls.append(kwargs)
        return f"cr-{len(cr_calls)}"

    monkeypatch.setattr(code_drift, "file_change_request", fake_cr)
    monkeypatch.setattr(code_drift, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(code_drift, "send_signal_alert", lambda *a, **k: True)

    anomaly = {
        "pattern_signature": code_drift._G_SIGNATURE,
        "pattern_sample": (
            "Anthropic API call failed: 'str' object has no attribute 'content'"
        ),
        "severity": "warning",
    }

    r1 = code_drift._handle_anthropic_str_content(anomaly)
    r2 = code_drift._handle_anthropic_str_content(anomaly)

    assert r1.success is True
    assert r1.name == "anthropic_str_content_cr"
    assert "filed CR" in r1.detail
    assert len(cr_calls) == 1
    # CR proposes the new guard module path.
    assert cr_calls[0]["path"] == "app/llms/anthropic_response_guard.py"
    assert "coerce_response" in cr_calls[0]["new_content"]
    # Dedup — second call doesn't refile.
    assert "already proposed" in r2.detail
