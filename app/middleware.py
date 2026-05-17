"""
middleware.py — Security headers and CORS middleware.

Extracted from main.py to reduce gravity-well coupling.

PR 1 (2026-05-16): CORS configuration is now the single source of
truth for the gateway. The previous duplicate block in ``main.py``
has been removed; ``allow_origins`` here is the union of both prior
configs (dashboard dev port 3100 across localhost / loopback /
tailscale, plus the same-origin gateway port and the Firebase-hosted
assets). Stack order: ``CORSMiddleware`` then ``SecurityHeadersMiddleware``
— security headers wrap the response after CORS has decided whether
to short-circuit a preflight.
"""

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware


def _dashboard_origins(gateway_port: int) -> list[str]:
    """Build the explicit allow-origin list.

    Explicit origins (no wildcards) are required because we use
    ``allow_credentials=True`` for the dashboard — browsers reject
    ``Access-Control-Allow-Origin: *`` with credentials.

    Extra origins can be appended via ``CORS_EXTRA_ORIGINS`` env
    (comma-separated). Useful for adding new tailscale hostnames or
    reverse-proxy URLs without editing source.
    """
    base = [
        # Dashboard dev server (Vite at port 3100)
        "http://localhost:3100",
        "http://127.0.0.1:3100",
        "http://100.85.195.121:3100",
        "http://plgs-macbook-pro---andrus:3100",
        "http://plgs-macbook-pro---andrus.tail5b289b.ts.net:3100",
        # Same-origin gateway (mainly for the bundled /cp/ React app)
        f"http://127.0.0.1:{gateway_port}",
        f"http://localhost:{gateway_port}",
        # Firebase-hosted assets calling back into the gateway
        "https://botarmy-ba0c9.web.app",
        "https://botarmy-ba0c9.firebaseapp.com",
    ]
    extra = os.environ.get("CORS_EXTRA_ORIGINS", "")
    if extra:
        base.extend(
            o.strip() for o in extra.split(",") if o.strip()
        )
    return base


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response.

    The /cp/ dashboard needs scripts, styles, images, and API connections,
    so it gets a permissive CSP. All other routes (API, webhooks) keep the
    strict default-src 'none' policy.
    """

    # CSP for the React dashboard: allow self-hosted assets + API calls
    _DASHBOARD_CSP = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "frame-ancestors 'none'"
    )
    # Strict CSP for API routes: block everything
    _API_CSP = "default-src 'none'; frame-ancestors 'none'"

    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        path = request.url.path
        if path.startswith("/cp"):
            response.headers["Content-Security-Policy"] = self._DASHBOARD_CSP
            # Allow browser caching for static assets
            if "/assets/" in path:
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "no-cache"
        else:
            response.headers["Cache-Control"] = "no-store"
            response.headers["Content-Security-Policy"] = self._API_CSP
        return response


def add_middleware(app, settings):
    """Configure all middleware on the FastAPI app.

    Order matters: ``add_middleware`` in FastAPI is LIFO — the last-added
    middleware is the OUTERMOST (closest to the request). We add CORS
    first so it can short-circuit preflight, then SecurityHeaders so the
    headers wrap the final response.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_dashboard_origins(settings.gateway_port),
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
        max_age=3600,
    )
    app.add_middleware(SecurityHeadersMiddleware)
