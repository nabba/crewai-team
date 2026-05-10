"""Regression: System Monitor probes for Signal-cli and Host bridge
must reflect the actual runtime, not obsolete enable-signals.

Pre-fix shape (2026-05-10):

  * **Signal-cli probe** hit ``GET /v1/about`` — the path used by the
    *separate* signal-cli-rest-api Docker wrapper, NOT by vanilla
    signal-cli.  Vanilla signal-cli's ``--http`` mode is JSON-RPC at
    ``POST /api/v1/rpc``.  Result: false "Endpoint not found" ERROR
    in the Monitor for every deployment using upstream signal-cli.

  * **Host-bridge probe** gated on ``settings.bridge_enabled`` (env
    ``BRIDGE_ENABLED=1``).  Since §27.3 the actual wiring uses
    per-agent ``BRIDGE_TOKEN_<AGENT>`` tokens — ``BRIDGE_ENABLED``
    is no longer the source of truth.  Result: false "host bridge
    disabled" WARN while the bridge was fully functional.

Post-fix:
  * Signal-cli probe POSTs JSON-RPC ``version`` and parses the
    response; OK with version string when the daemon answers.
  * Host-bridge probe calls ``bridge_client.get_bridge('change_requests')
    .is_available()`` first; falls back to the legacy enable+probe
    path when no per-agent tokens are configured.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD_API = (
    _REPO_ROOT / "app" / "control_plane" / "dashboard_api.py"
)


@pytest.fixture(scope="module")
def src() -> str:
    return _DASHBOARD_API.read_text(encoding="utf-8")


# ── Source-grep contracts ──────────────────────────────────────────


class TestSignalProbeUsesJsonRpc:

    def test_signal_probe_posts_to_api_v1_rpc(self, src: str) -> None:
        # Slice from the _signal() definition through the probe registration.
        m = re.search(
            r"def _signal\(\):.*?checks\.append\(_probe\(\"Signal-cli daemon\"",
            src, re.DOTALL,
        )
        assert m is not None, "could not slice _signal()"
        body = m.group(0)
        assert "/api/v1/rpc" in body, (
            "Signal-cli probe must hit POST /api/v1/rpc (JSON-RPC), "
            "not GET /v1/about"
        )
        assert 'method="POST"' in body, (
            "JSON-RPC requires POST"
        )
        assert '"jsonrpc"' in body and '"version"' in body, (
            "must send a {jsonrpc: 2.0, method: 'version'} payload"
        )

    def test_signal_probe_does_not_use_v1_about(self, src: str) -> None:
        m = re.search(
            r"def _signal\(\):.*?checks\.append\(_probe\(\"Signal-cli daemon\"",
            src, re.DOTALL,
        )
        body = m.group(0) if m else ""
        # Strip docstrings + comments before checking — explanatory
        # text about "the OLD probe used /v1/about" is fine; we only
        # care that no live code path references it.
        # Drop triple-quoted docstring.
        body_no_doc = re.sub(r'"""[\s\S]*?"""', "", body)
        # Drop line comments.
        body_code = "\n".join(
            line for line in body_no_doc.splitlines()
            if not line.strip().startswith("#")
        )
        assert "/v1/about" not in body_code, (
            "the legacy /v1/about path (signal-cli-rest-api wrapper) "
            "must NOT be used — vanilla signal-cli returns 404"
        )


class TestBridgeProbeChecksPerAgentToken:

    def test_bridge_probe_calls_get_bridge_first(self, src: str) -> None:
        m = re.search(
            r"def _bridge\(\):.*?checks\.append\(_probe\(\"Host bridge\"",
            src, re.DOTALL,
        )
        assert m is not None, "could not slice _bridge()"
        body = m.group(0)
        # Modern path must be tried first — find positions.
        idx_get_bridge = body.find("get_bridge(")
        idx_legacy_check = body.find("s.bridge_enabled")
        assert idx_get_bridge >= 0, (
            "must call bridge_client.get_bridge(...) — that's the "
            "actual code path the rest of the system uses"
        )
        assert idx_legacy_check >= 0, (
            "must keep the legacy enable check as a fallback for "
            "laptop-dev setups without per-agent tokens"
        )
        assert idx_get_bridge < idx_legacy_check, (
            "modern per-agent path must be tried BEFORE the legacy "
            "fallback (otherwise functional bridges still report "
            "as warn/disabled)"
        )

    def test_bridge_probe_uses_change_requests_agent(
        self, src: str,
    ) -> None:
        m = re.search(
            r"def _bridge\(\):.*?checks\.append\(_probe\(\"Host bridge\"",
            src, re.DOTALL,
        )
        body = m.group(0) if m else ""
        assert 'get_bridge("change_requests")' in body, (
            "use 'change_requests' as the canonical probe agent — "
            "it's what the React /cp/changes surface uses"
        )


# ── Functional simulation ──────────────────────────────────────────


class TestSignalProbeFunctional:

    def test_signal_ok_when_jsonrpc_returns_version(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When signal-cli answers JSON-RPC version, probe returns OK
        with the version string in the message."""
        from app.control_plane import dashboard_api as api

        # Stub urllib so we don't actually hit signal-cli.
        class _FakeResp:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def read(self) -> bytes:
                return self._payload

        captured: dict = {}

        def _fake_urlopen(req, timeout=None):
            captured["url"] = (
                req.full_url if hasattr(req, "full_url") else str(req)
            )
            captured["method"] = (
                req.get_method() if hasattr(req, "get_method") else "?"
            )
            captured["data"] = (
                req.data if hasattr(req, "data") else None
            )
            return _FakeResp(json.dumps({
                "jsonrpc": "2.0",
                "result": {"version": "0.14.1"},
                "id": 1,
            }).encode("utf-8"))

        monkeypatch.setattr(
            "urllib.request.urlopen", _fake_urlopen,
        )

        # Run the probe collection — it builds the full checks list.
        result = api.system_status()
        signal_check = next(
            (c for c in result["checks"] if c["name"] == "Signal-cli daemon"),
            None,
        )
        assert signal_check is not None
        assert signal_check["status"] == "ok", (
            f"expected OK, got {signal_check}"
        )
        assert "0.14.1" in signal_check["message"]

        # Confirm the right URL+method+payload were used.
        assert "/api/v1/rpc" in captured["url"]
        assert captured["method"] == "POST"
        body = json.loads(captured["data"])
        assert body["method"] == "version"
        assert body["jsonrpc"] == "2.0"


class TestBridgeProbeFunctional:

    def test_bridge_ok_when_per_agent_token_works(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With a working per-agent client, the probe must short-
        circuit at OK without touching the legacy enable check."""
        from app.control_plane import dashboard_api as api

        class _FakeBridge:
            def is_available(self) -> bool:
                return True

        monkeypatch.setattr(
            "app.bridge_client.get_bridge",
            lambda agent_id: _FakeBridge() if agent_id == "change_requests" else None,
        )

        result = api.system_status()
        bridge_check = next(
            (c for c in result["checks"] if c["name"] == "Host bridge"),
            None,
        )
        assert bridge_check is not None
        assert bridge_check["status"] == "ok", (
            f"expected OK with per-agent token; got {bridge_check}"
        )
        assert "per-agent" in bridge_check["message"]

    def test_bridge_warn_when_no_token_and_legacy_disabled(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No per-agent token AND BRIDGE_ENABLED=0 → warn (laptop dev
        with bridge intentionally off)."""
        from app.control_plane import dashboard_api as api

        # No per-agent token.
        monkeypatch.setattr(
            "app.bridge_client.get_bridge", lambda agent_id: None,
        )

        # Legacy disabled.
        from app.config import get_settings
        original = get_settings()

        class _FakeSettings:
            bridge_enabled = False
            bridge_host = original.bridge_host
            bridge_port = original.bridge_port
            signal_http_url = original.signal_http_url
            signal_owner_number = original.signal_owner_number
            workspace_backup_repo = original.workspace_backup_repo

        monkeypatch.setattr(
            "app.config.get_settings", lambda: _FakeSettings(),
        )

        result = api.system_status()
        bridge_check = next(
            (c for c in result["checks"] if c["name"] == "Host bridge"),
            None,
        )
        assert bridge_check is not None
        assert bridge_check["status"] == "warn"
        # Updated message mentions both signals so operators understand
        # what's needed to enable it.
        assert "BRIDGE_TOKEN_*" in bridge_check["message"] or (
            "BRIDGE_ENABLED" in bridge_check["message"]
        )
