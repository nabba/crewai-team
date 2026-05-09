"""
mcp/client.py — MCP client for consuming tools from external MCP servers.

Uses shared transports from mcp.transports. Handles lifecycle:
  initialize → tools/list → tools/call → shutdown

Thread-safe: all public methods safe for concurrent crew access.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.mcp.transports import (
    StdioTransport, SSETransport, StreamableHTTPTransport,
    jsonrpc_request, jsonrpc_notification,
)

logger = logging.getLogger(__name__)


# Auth-failure substrings.  When any appears in a connect()/init()
# error message, we treat it as an operator-action failure (wrong
# token / expired credential) and trip a per-server circuit breaker
# instead of retrying every connect() call.  Pattern_learner reported
# 'STUzhy/py_execute_mcp' init failed: HTTP 401: {"error":"invalid_token"}
# at a high enough volume to require this fix.
_MCP_AUTH_ERROR_FRAGMENTS = (
    "HTTP 401",
    "HTTP 403",
    "invalid_token",
    "Unauthorized",
    "Forbidden",
)


def _is_mcp_auth_error(exc_text: str) -> bool:
    """Return True if the error message looks like an auth failure."""
    return any(frag in exc_text for frag in _MCP_AUTH_ERROR_FRAGMENTS)


def _mcp_breaker_name(server_name: str) -> str:
    """Per-server breaker key.  Auth failures on one MCP server must
    not block connections to other servers."""
    return f"mcp_auth:{server_name}"


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    timeout: float = 30.0
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)  # auth headers for remote servers

    @classmethod
    def from_dict(cls, d: dict) -> "MCPServerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MCPToolSchema:
    server_name: str
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)


class MCPClient:
    """Client for a single MCP server."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.tools: list[MCPToolSchema] = []
        self._transport: StdioTransport | SSETransport | StreamableHTTPTransport
        # Build auth headers for remote transports
        self._headers = dict(config.headers) if config.headers else {}
        # Auto-inject Smithery API key for Smithery-hosted servers
        if "smithery.ai" in (config.url or "") and "Authorization" not in self._headers:
            import os
            _sk = os.environ.get("SMITHERY_API_KEY", "")
            if _sk:
                self._headers["Authorization"] = f"Bearer {_sk}"

        if config.transport in ("http", "streamable-http"):
            self._transport = StreamableHTTPTransport(config.url, config.timeout, self._headers)
        elif config.transport == "sse":
            # Try SSE first; if it fails, fall back to Streamable HTTP
            self._transport = SSETransport(config.url, config.timeout)
        else:
            self._transport = StdioTransport(config.command, config.args, config.env)
        self._initialized = False

    def _record_failure_log(self, exc_or_text, action: str) -> None:
        """Log a connect/init failure at the right level.

        Auth failures (401/403/invalid_token) trip the per-server
        breaker and log INFO (the breaker's own CLOSED→OPEN log
        provides the operator-visible WARN once).  Transient/other
        failures log WARN as before so connectivity issues stay
        visible.
        """
        text = str(exc_or_text)
        if _is_mcp_auth_error(text):
            try:
                from app import circuit_breaker
                circuit_breaker.record_failure(
                    _mcp_breaker_name(self.config.name)
                )
            except Exception:
                pass
            logger.info(
                f"mcp_client: '{self.config.name}' auth-{action} — "
                f"breaker tripped, deferred until operator rotates "
                f"credential: {text}"
            )
        else:
            logger.warning(
                f"mcp_client: '{self.config.name}' {action}: {text}"
            )

    def connect(self) -> bool:
        # Per-server circuit breaker for auth failures.  Created lazily
        # with operator-action shape (1 failure → 1 h cooldown).  When
        # OPEN, we silently skip; the breaker logs once per
        # CLOSED→OPEN transition (operator alert) and INFO on
        # subsequent re-trips.
        try:
            from app import circuit_breaker
            circuit_breaker.ensure_breaker(
                _mcp_breaker_name(self.config.name),
                failure_threshold=1, cooldown_seconds=3600,
            )
            if not circuit_breaker.is_available(
                _mcp_breaker_name(self.config.name)
            ):
                logger.debug(
                    f"mcp_client: '{self.config.name}' auth breaker OPEN; "
                    f"skipping connect"
                )
                return False
        except Exception:
            # Best-effort; never let breaker import break the client.
            pass

        try:
            self._transport.start()
        except Exception as exc:
            # SSE → Streamable HTTP fallback for remote servers
            if self.config.transport == "sse" and self.config.url:
                logger.info(
                    f"mcp_client: '{self.config.name}' SSE failed, "
                    f"trying Streamable HTTP: {exc}"
                )
                self._transport = StreamableHTTPTransport(
                    self.config.url, self.config.timeout, self._headers,
                )
                try:
                    self._transport.start()
                except Exception as exc2:
                    self._record_failure_log(exc2, "HTTP also failed")
                    return False
            else:
                self._record_failure_log(exc, "start failed")
                return False

        # Initialize handshake
        try:
            resp = self._transport.send_receive(jsonrpc_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "AndrusAI", "version": "1.0"},
            }))
            if "error" in resp:
                # JSON-RPC error envelope — check for auth in the
                # error code/message.
                err_text = json.dumps(resp.get("error", {}))
                self._record_failure_log(err_text, "init error")
                return False
        except Exception as exc:
            self._record_failure_log(exc, "init failed")
            return False

        self._transport.send_notification(jsonrpc_notification("notifications/initialized"))

        # Discover tools
        try:
            tools_resp = self._transport.send_receive(jsonrpc_request("tools/list"))
            raw_tools = tools_resp.get("result", {}).get("tools", [])
            self.tools = [
                MCPToolSchema(
                    server_name=self.config.name,
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                for t in raw_tools
            ]
            logger.info(f"mcp_client: '{self.config.name}' — {len(self.tools)} tools")
        except Exception as exc:
            logger.warning(f"mcp_client: '{self.config.name}' tool discovery failed: {exc}")

        self._initialized = True
        return True

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self._initialized:
            raise ConnectionError(f"MCP '{self.config.name}' not initialized")
        resp = self._transport.send_receive(jsonrpc_request("tools/call", {
            "name": tool_name, "arguments": arguments,
        }))
        if "error" in resp:
            return f"MCP error: {resp['error'].get('message', str(resp['error']))}"
        content = resp.get("result", {}).get("content", [])
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts) if parts else json.dumps(resp.get("result", {}))

    def disconnect(self) -> None:
        self._transport.stop()
        self._initialized = False

    @property
    def is_connected(self) -> bool:
        return self._initialized and self._transport.is_alive
