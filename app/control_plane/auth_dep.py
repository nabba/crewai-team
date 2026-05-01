"""
control_plane.auth_dep — FastAPI dependency for gateway-secret auth.

Phase B1 of the Phase-A-through-G remediation plan.

Two-mode dependency:
    * Dev mode (default, GATEWAY_AUTH_REQUIRED unset/false):
      Pass-through — preserves the laptop developer experience where
      the dashboard reaches localhost without ceremony.
    * Enforced mode (GATEWAY_AUTH_REQUIRED=1, set by Helm in K8s):
      Validate ``Authorization: Bearer <gateway-secret>`` via constant-time
      HMAC compare. Reject with 401 if missing or wrong.

Why a separate module (and not import from main.py):
    main.py is TIER_IMMUTABLE and already imports the dashboard router
    at startup; pulling its private ``_verify_gateway_secret`` back here
    creates a circular import. The function is small enough that
    duplication beats the gymnastics.

Why an env var rather than a Settings field:
    Mirrors the pattern used by :mod:`app.epistemic.is_enabled` and
    :mod:`app.recovery.loop.is_enabled` — additive cross-cutting toggles
    that need to read identically from tests, scripts, and the live app
    without going through the full Pydantic Settings round-trip.

Usage::

    from app.control_plane.auth_dep import require_gateway_auth

    router = APIRouter(
        prefix="/api/cp",
        dependencies=[Depends(require_gateway_auth)],
    )

Internal Python callers (e.g. ``record_override`` invoked from
``orchestrator_hook``) DO NOT pass through this dependency. The
dependency only fires on HTTP entry. That is intentional — the auth
boundary is the gateway, not the function call.
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


def _enforcement_enabled() -> bool:
    """Return True when GATEWAY_AUTH_REQUIRED is set to a truthy value."""
    val = os.environ.get("GATEWAY_AUTH_REQUIRED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth[len("Bearer "):]


def require_gateway_auth(request: Request) -> None:
    """FastAPI dependency: enforce Bearer-token auth when configured.

    Raises HTTPException(401) when enforcement is on AND the token is
    missing or does not match the gateway secret. Returns None on success
    or when enforcement is off (dev mode).
    """
    if not _enforcement_enabled():
        return  # dev / laptop mode — pass through

    # Lazy import: app.config is IMMUTABLE infrastructure but readable;
    # import inside the function so importing this module never fails
    # in test contexts that haven't fully constructed Settings.
    from app.config import get_gateway_secret

    expected = get_gateway_secret() or ""
    received = _extract_bearer_token(request) or ""

    if not expected:
        # Enforcement is on but no secret configured — fail loud rather
        # than silently allow. Operator misconfiguration, not a request bug.
        logger.error(
            "GATEWAY_AUTH_REQUIRED=1 but gateway secret is empty; rejecting request"
        )
        raise HTTPException(status_code=503, detail="auth misconfigured")

    if not received:
        raise HTTPException(status_code=401, detail="missing bearer token")

    if not hmac.compare_digest(received, expected):
        # Constant-time compare prevents timing side channels.
        raise HTTPException(status_code=401, detail="invalid bearer token")

    return  # authorized
