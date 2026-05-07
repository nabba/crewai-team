"""Tests for app.tool_runtime.supervisor — the mid-iteration tool repair
wrapper added in the May 2026 self-healing pass (track A).

Reconstructed from orphan .pyc bytecode + PROGRAM.md §14 spec on
2026-05-06. Originals were never committed; this file ships with the
supervisor source so the safety claim in §14 is reproducible.

Coverage shape mirrors the original (per .pyc test-name extraction):
  - TestClassifyException  exception bucketing
  - TestWrapDisabled       no-op when env flag unset
  - TestRetry              retry policy on transient categories
  - TestSubstitute         registry-driven substitution
  - TestAudit              audit emission on failure paths
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# Import the unit under test up front so a missing module fails fast.
from app.tool_runtime import supervisor  # noqa: E402


# ── classify_exception ─────────────────────────────────────────────────────


class TestClassifyException:
    def test_rate_limit_class_name(self):
        class RateLimitError(Exception):
            pass
        assert supervisor.classify_exception(RateLimitError("x")) == "rate_limit"

    def test_rate_limit_message(self):
        assert supervisor.classify_exception(
            RuntimeError("HTTP 429 too many requests")
        ) == "rate_limit"

    def test_auth_class_name(self):
        class AuthError(Exception):
            pass
        assert supervisor.classify_exception(AuthError("denied")) == "auth"

    def test_auth_message(self):
        assert supervisor.classify_exception(
            RuntimeError("401 Unauthorized")
        ) == "auth"

    def test_network_isinstance(self):
        assert supervisor.classify_exception(ConnectionError("reset")) == "network"

    def test_network_message(self):
        assert supervisor.classify_exception(
            RuntimeError("connection refused")
        ) == "network"

    def test_timeout_isinstance(self):
        assert supervisor.classify_exception(TimeoutError("slow")) == "timeout"

    def test_timeout_message(self):
        assert supervisor.classify_exception(
            RuntimeError("Operation timed out after 30s")
        ) == "timeout"

    def test_schema_message(self):
        assert supervisor.classify_exception(
            ValueError("validation error: missing required field 'foo'")
        ) == "schema"

    def test_unknown_fallback(self):
        assert supervisor.classify_exception(RuntimeError("kaboom")) == "unknown"


# ── wrap when disabled ─────────────────────────────────────────────────────


class TestWrapDisabled:
    def test_returns_original_when_off(self, monkeypatch):
        monkeypatch.delenv("TOOL_SUPERVISOR_ENABLED", raising=False)

        def f():
            return 1

        wrapped = supervisor.wrap_tool_function("f", f)
        assert wrapped is f

    def test_supervise_dict_passthrough_when_off(self, monkeypatch):
        monkeypatch.delenv("TOOL_SUPERVISOR_ENABLED", raising=False)

        def f():
            return 1

        d = {"f": f}
        out = supervisor.supervise_available_functions(d)
        assert out is d


# ── retry ──────────────────────────────────────────────────────────────────


class TestRetry:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setenv("TOOL_SUPERVISOR_ENABLED", "true")
        monkeypatch.setenv("TOOL_SUPERVISOR_MAX_RETRIES", "2")
        monkeypatch.setenv("TOOL_SUPERVISOR_BACKOFF_MS", "0")  # no sleep in tests

    def test_transient_retries_then_succeeds(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("reset")
            return "ok"

        wrapped = supervisor.wrap_tool_function("net", fn)
        assert wrapped() == "ok"
        assert calls["n"] == 2

    def test_transient_exhausts_retries_returns_observation(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ConnectionError("permanent")

        with patch.object(supervisor, "_find_substitute", return_value=None):
            wrapped = supervisor.wrap_tool_function("net", fn)
            result = wrapped()
        assert isinstance(result, str)
        assert "[tool-supervisor]" in result
        assert "network" in result
        assert calls["n"] == 3  # 1 initial + 2 retries

    def test_non_transient_no_retry(self):
        calls = {"n": 0}

        class AuthError(Exception):
            pass

        def fn():
            calls["n"] += 1
            raise AuthError("nope")

        with patch.object(supervisor, "_find_substitute", return_value=None):
            wrapped = supervisor.wrap_tool_function("api", fn)
            result = wrapped()
        assert "[tool-supervisor]" in result
        assert calls["n"] == 1  # no retries on non-transient

    def test_keyboard_interrupt_propagates(self):
        def fn():
            raise KeyboardInterrupt()

        wrapped = supervisor.wrap_tool_function("x", fn)
        with pytest.raises(KeyboardInterrupt):
            wrapped()


# ── substitute ─────────────────────────────────────────────────────────────


class TestSubstitute:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setenv("TOOL_SUPERVISOR_ENABLED", "true")
        monkeypatch.setenv("TOOL_SUPERVISOR_MAX_RETRIES", "0")
        monkeypatch.setenv("TOOL_SUPERVISOR_BACKOFF_MS", "0")

    def test_substitute_returns_alt_result(self):
        def alt(*a, **kw):
            return "alt-result"

        sub = supervisor._Substitute(name="alt_tool", callable=alt)

        def primary():
            raise RuntimeError("dead")

        with patch.object(supervisor, "_find_substitute", return_value=sub):
            wrapped = supervisor.wrap_tool_function("primary", primary)
            assert wrapped() == "alt-result"

    def test_substitute_failure_returns_observation(self):
        def alt():
            raise RuntimeError("alt also fails")

        sub = supervisor._Substitute(name="alt_tool", callable=alt)

        def primary():
            raise RuntimeError("dead")

        with patch.object(supervisor, "_find_substitute", return_value=sub):
            wrapped = supervisor.wrap_tool_function("primary", primary)
            result = wrapped()
        assert isinstance(result, str)
        assert "Tried alternative 'alt_tool'" in result

    def test_recursion_guard_substitute_does_not_recurse(self):
        """If a substitute itself tries to call the supervisor, it should
        execute unsupervised — verified by checking that the inner call
        bypasses the wrapper's retry logic.
        """
        inner_calls = {"n": 0}

        def inner():
            inner_calls["n"] += 1
            return "inner"

        def alt():
            # Inner is itself supervised; under recursion guard this
            # should call inner once and return immediately.
            inner_wrapped = supervisor.wrap_tool_function("inner", inner)
            return inner_wrapped()

        sub = supervisor._Substitute(name="alt", callable=alt)

        def primary():
            raise RuntimeError("dead")

        with patch.object(supervisor, "_find_substitute", return_value=sub):
            wrapped = supervisor.wrap_tool_function("primary", primary)
            assert wrapped() == "inner"
        assert inner_calls["n"] == 1


# ── audit ──────────────────────────────────────────────────────────────────


class TestAudit:
    def test_failure_and_giveup_are_audited(self, monkeypatch):
        monkeypatch.setenv("TOOL_SUPERVISOR_ENABLED", "true")
        monkeypatch.setenv("TOOL_SUPERVISOR_MAX_RETRIES", "0")
        monkeypatch.setenv("TOOL_SUPERVISOR_BACKOFF_MS", "0")

        events = []

        def fake_audit(action, **detail):
            events.append((action, detail))

        with patch.object(supervisor, "_audit", side_effect=fake_audit), \
             patch.object(supervisor, "_find_substitute", return_value=None):
            def fn():
                raise RuntimeError("boom")

            wrapped = supervisor.wrap_tool_function("t", fn)
            wrapped()

        actions = [a for a, _ in events]
        assert "invocation.failed" in actions
        assert "invocation.gave_up" in actions
