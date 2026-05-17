"""HTTP transport for the operator CLI.

Stdlib-only (``urllib``). Deliberately not ``httpx`` / ``requests``: this CLI
must work when the gateway venv is sick. Adding deps would couple the recovery
surface to the thing it might be needed to recover.

Exit codes mapped to ``TransportError`` subclasses:

* :class:`AuthError` → 2
* :class:`NetworkError` → 2
* :class:`GatewayError` → 3
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app.cli.config import CLIConfig


class TransportError(Exception):
    exit_code: int = 2


class AuthError(TransportError):
    exit_code = 2


class NetworkError(TransportError):
    exit_code = 2


class GatewayError(TransportError):
    exit_code = 3


def _request(
    cfg: CLIConfig,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> Any:
    url = cfg.endpoint + path
    if params:
        flat = {k: ("true" if v is True else "false" if v is False else str(v))
                for k, v in params.items() if v is not None}
        if flat:
            url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(flat)

    headers = {"Accept": "application/json", **cfg.auth_header()}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if exc.code == 401 or exc.code == 403:
            raise AuthError(f"auth rejected ({exc.code}): {body_text or exc.reason}") from exc
        raise GatewayError(f"gateway returned {exc.code}: {body_text or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise NetworkError(f"cannot reach {cfg.endpoint}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise NetworkError(f"timeout against {cfg.endpoint}") from exc

    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Some endpoints return plain text — preserve verbatim
        return payload.decode("utf-8", errors="replace")


def get(cfg: CLIConfig, path: str, **kwargs: Any) -> Any:
    return _request(cfg, "GET", path, **kwargs)


def post(cfg: CLIConfig, path: str, **kwargs: Any) -> Any:
    return _request(cfg, "POST", path, **kwargs)
