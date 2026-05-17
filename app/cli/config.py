"""Endpoint + bearer resolution for the operator CLI.

Resolution order (highest priority first):

1. CLI flag (``--endpoint``, ``--bearer``)
2. Environment (``AAI_ENDPOINT``, ``AAI_BEARER`` — or ``GATEWAY_SECRET``)
3. ``~/.config/andrusai/config.toml`` (``[endpoints]`` and ``[auth]``)
4. Built-in defaults

Defaults: endpoint = ``http://localhost:3100``. No bearer.

Named endpoints (``local``, ``tailnet``, ``funnel``) are resolved from the
config file's ``[endpoints]`` section if present, otherwise from
``DASHBOARD_PUBLIC_URL`` / ``DASHBOARD_MAC_URL`` env vars, otherwise
from sensible built-in defaults that match the rest of the system.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# tomllib is stdlib on 3.11+; fall back to tomli for 3.10 if anyone runs there
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]

_BUILTIN_NAMED = {
    "local": "http://localhost:3100",
    # These two are best-effort defaults; users with different tailnet names
    # should set DASHBOARD_PUBLIC_URL / DASHBOARD_MAC_URL or override in TOML.
    "tailnet": "http://andrus-macbook-pro-16.tail5b289b.ts.net:3100",
    "funnel": "https://andrus-macbook-pro-16.tail5b289b.ts.net",
}


@dataclass(frozen=True)
class CLIConfig:
    endpoint: str
    bearer: str | None

    def auth_header(self) -> dict[str, str]:
        if self.bearer:
            return {"Authorization": f"Bearer {self.bearer}"}
        return {}


def _load_toml() -> dict:
    if tomllib is None:
        return {}
    path = Path.home() / ".config" / "andrusai" / "config.toml"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, ValueError):
        # Corrupt config: warn loudly on stderr but don't crash — the CLI
        # might be the only thing the operator can still use.
        print(f"warning: failed to parse {path}", file=sys.stderr)
        return {}


def _resolve_named(name: str, toml: dict) -> str:
    endpoints = toml.get("endpoints") or {}
    if name in endpoints:
        return str(endpoints[name])
    if name == "funnel":
        env = os.environ.get("DASHBOARD_PUBLIC_URL")
        if env:
            return env
    if name == "tailnet":
        env = os.environ.get("DASHBOARD_MAC_URL")
        if env:
            return env
    return _BUILTIN_NAMED.get(name, name)


def resolve(*, endpoint: str | None = None, bearer: str | None = None) -> CLIConfig:
    """Resolve CLI config from flags / env / TOML / defaults.

    ``endpoint`` may be a named alias (``local`` / ``tailnet`` / ``funnel``)
    or an absolute URL. URLs are passed through verbatim; bare names hit the
    resolver.
    """
    toml = _load_toml()

    if endpoint is None:
        endpoint = os.environ.get("AAI_ENDPOINT")
    if endpoint is None:
        endpoint = (toml.get("default") or {}).get("endpoint")
    if endpoint is None:
        endpoint = "local"

    if "://" not in endpoint:
        endpoint = _resolve_named(endpoint, toml)

    endpoint = endpoint.rstrip("/")

    if bearer is None:
        bearer = os.environ.get("AAI_BEARER") or os.environ.get("GATEWAY_SECRET")
    if not bearer:
        bearer = (toml.get("auth") or {}).get("bearer")

    return CLIConfig(endpoint=endpoint, bearer=bearer or None)
