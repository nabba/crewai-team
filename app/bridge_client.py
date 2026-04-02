"""
bridge_client.py — Client for Host Bridge Service.

Used by agents inside the Docker container to access host resources
via the capability-gated bridge at host.docker.internal:9100.

Each agent gets a BridgeClient initialized with its capability token.
The client handles authentication, error responses, and provides
typed methods for each bridge endpoint.

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

BRIDGE_URL = "http://{}:{}".format(
    os.getenv("BRIDGE_HOST", "host.docker.internal"),
    os.getenv("BRIDGE_PORT", "9100"),
)


class BridgeError(Exception):
    """Error from the host bridge."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class BridgeClient:
    """Client for the Host Bridge Service. Each agent gets one with its token."""

    def __init__(self, agent_id: str, token: str):
        self.agent_id = agent_id
        self.token = token
        self._base = BRIDGE_URL

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make a request to the bridge with capability token."""
        import httpx
        try:
            response = httpx.request(
                method, f"{self._base}{path}",
                headers={"X-Capability-Token": self.token},
                timeout=60,
                **kwargs,
            )
            if response.status_code == 403:
                detail = response.json().get("detail", "Permission denied")
                logger.warning(f"bridge_client: {self.agent_id} denied for {path}: {detail}")
                return {"error": "permission_denied", "detail": detail}
            if response.status_code == 429:
                return {"error": "rate_limited", "detail": "Too many requests"}
            if response.status_code == 503:
                return {"error": "kill_switch", "detail": "System halted by operator"}
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"bridge_client: request to {path} failed: {e}")
            return {"error": "connection_error", "detail": str(e)[:200]}

    # ── Filesystem ────────────────────────────────────────────────

    def read_file(self, path: str, max_bytes: int = 1_000_000) -> dict:
        return self._request("POST", "/filesystem/read", json={
            "path": path, "max_bytes": max_bytes,
        })

    def write_file(self, path: str, content: str, create_dirs: bool = False) -> dict:
        return self._request("POST", "/filesystem/write", json={
            "path": path, "content": content, "create_dirs": create_dirs,
        })

    def list_files(self, path: str, pattern: str = "*", recursive: bool = False) -> dict:
        return self._request("POST", "/filesystem/list", json={
            "path": path, "pattern": pattern, "recursive": recursive,
        })

    # ── Network ───────────────────────────────────────────────────

    def http_request(self, url: str, method: str = "GET",
                     headers: dict = None, body: str = None) -> dict:
        return self._request("POST", "/network/http", json={
            "method": method, "url": url,
            "headers": headers or {}, "body": body,
        })

    def scan_network(self, subnet: str = "192.168.1.0/24",
                     ports: list[int] = None) -> dict:
        return self._request("POST", "/network/scan", json={
            "subnet": subnet, "ports": ports or [80, 443, 8080, 22],
        })

    # ── Execution ─────────────────────────────────────────────────

    def execute(self, command: list[str], working_dir: str = "/tmp",
                timeout: int = 30, env: dict = None) -> dict:
        return self._request("POST", "/execute", json={
            "command": command, "working_dir": working_dir,
            "timeout": timeout, "env": env or {},
        })

    # ── GPU / Ollama ──────────────────────────────────────────────

    def inference(self, prompt: str, model: str = "qwen3:30b-a3b",
                  system: str = "", temperature: float = 0.7) -> dict:
        return self._request("POST", "/gpu/inference", json={
            "model": model, "prompt": prompt,
            "system": system, "temperature": temperature,
        })

    # ── Status ────────────────────────────────────────────────────

    def health(self) -> dict:
        return self._request("GET", "/health")

    def status(self) -> dict:
        return self._request("GET", "/status")

    # ── Convenience ───────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the bridge is reachable."""
        try:
            result = self.health()
            return result.get("status") == "ok"
        except Exception:
            return False


# ── Agent token loading ───────────────────────────────────────────────────────

_agent_tokens: dict[str, str] = {}


def _load_tokens():
    """Load agent tokens from AGENT_TOKENS env var."""
    global _agent_tokens
    raw = os.getenv("AGENT_TOKENS", "{}")
    try:
        _agent_tokens = json.loads(raw)
    except json.JSONDecodeError:
        _agent_tokens = {}


def get_bridge(agent_id: str) -> Optional[BridgeClient]:
    """Get a BridgeClient for an agent. Returns None if no token configured."""
    if not _agent_tokens:
        _load_tokens()
    token = _agent_tokens.get(agent_id)
    if not token:
        return None
    return BridgeClient(agent_id=agent_id, token=token)
