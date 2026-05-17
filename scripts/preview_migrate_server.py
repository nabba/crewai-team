"""Minimal FastAPI server for preview-verification of the migrate wizard.

Mounts:
  * static React dashboard (built artifacts) at /cp/*
  * /api/cp/migrate/* — the wizard's main endpoints
  * /config/runtime_settings — the runtime-settings GET/POST the
    useRuntimeSettingsQuery + useUpdateRuntimeSettings hooks call
  * /api/cp/settings — the alias that does the same thing
  * stub /api/cp/health — so Layout's useHealthQuery returns clean

NOT a production gateway. No auth, no Docker dependencies, no daemons.
Just enough surface for Step-4 toggle verification in the browser.

Run: .venv/bin/python scripts/preview_migrate_server.py
Then open http://127.0.0.1:8000/cp/migrate
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root + workspace
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("WORKSPACE_ROOT", str(REPO / "workspace"))
os.environ.setdefault("IDENTITY_LEDGER_ENABLED", "true")
# Auth pass-through (dev mode)
os.environ.pop("GATEWAY_AUTH_REQUIRED", None)

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="migrate-preview")

# ── Preview override: make cloud_doctor return OK so the wizard
# advances past step 2 in this isolated demo. Real cloud_doctor is
# always invoked in production. ──────────────────────────────────────
from app.substrate import cloud_doctor as _cd  # noqa: E402

_OK_PROBES = [
    {"name": "terraform", "status": "OK", "detail": "Terraform v1.15.0", "required": True},
    {"name": "kubectl", "status": "OK", "detail": "Client v1.28", "required": True},
    {"name": "helm", "status": "OK", "detail": "v3.13.0", "required": True},
    {"name": "docker", "status": "OK", "detail": "Docker 29.4", "required": True},
    {"name": "gcloud", "status": "OK", "detail": "Cloud SDK 561", "required": True},
    {"name": "gcloud auth", "status": "OK", "detail": "andrus@raudsalu.com", "required": True},
    {"name": "gcloud project", "status": "OK", "detail": "botarmy-495107", "required": True},
    {"name": "ADC", "status": "OK", "detail": "present", "required": False},
    {"name": "gcloud account type", "status": "OK", "detail": "user account", "required": False},
    {"name": "gcloud project access", "status": "OK", "detail": "readable", "required": True},
    {"name": "gcloud required APIs", "status": "OK", "detail": "9 enabled", "required": True},
    {"name": "ADC account", "status": "OK", "detail": "andrus@raudsalu.com", "required": False},
    {"name": "continuity bundle", "status": "OK", "detail": "fresh", "required": True},
    {"name": "subia integrity", "status": "OK", "detail": "164 files clean", "required": False},
]


class _StubReadiness:
    target = "gcp"
    timestamp = "2026-05-17T20:00:00+00:00"
    overall = "OK"
    probes = [type("P", (), p)() for p in _OK_PROBES]

    def to_dict(self):
        return {
            "target": self.target,
            "timestamp": self.timestamp,
            "overall": self.overall,
            "probes": _OK_PROBES,
        }


_cd.check_readiness = lambda target="gcp": _StubReadiness()

# ── Migrate API ─────────────────────────────────────────────────────
from app.control_plane.migrate_api import router as migrate_router  # noqa: E402
app.include_router(migrate_router)

# ── Settings alias (POST /api/cp/settings → forwards to /config) ────
from app.control_plane.settings_alias_api import router as settings_alias_router  # noqa: E402
app.include_router(settings_alias_router)

# ── /config/runtime_settings GET + POST (canonical runtime-settings) ──
from app.api.config_api import (  # noqa: E402
    get_runtime_settings_endpoint,
    set_runtime_settings_endpoint,
)


@app.get("/config/runtime_settings")
async def _rs_get():
    # The canonical getter is auth-gated; bypass in this preview server.
    from app.runtime_settings import snapshot
    return snapshot()


@app.post("/config/runtime_settings")
async def _rs_post(request: Request):
    # Bypass the gateway-secret check for preview. The setter dispatcher
    # still routes each key to its strict-typed setter.
    return await set_runtime_settings_endpoint(request)


# ── Stub endpoints the React Layout polls ───────────────────────────


@app.get("/api/cp/health")
async def _health():
    return {"status": "ok"}


@app.get("/api/cp/projects")
async def _projects():
    return []


@app.get("/api/cp/system-status")
async def _system_status():
    return {"checks": [], "by_category": {}, "overall": "ok", "updated_at": ""}


# ── Static dashboard (built React app) ──────────────────────────────


DASH_BUILD = REPO / "dashboard" / "serve-root" / "cp"
# Mount /cp at the dashboard build. The React app uses BrowserRouter
# basename="/cp", so client-side routes like /cp/migrate need an
# index.html fallback.

if DASH_BUILD.exists():
    # Serve assets at /cp/assets, /cp/*.svg etc.
    app.mount("/cp/assets", StaticFiles(directory=str(DASH_BUILD / "assets")), name="cp-assets")

    @app.get("/cp/{full_path:path}")
    async def _cp_spa(full_path: str):
        # SPA fallback: any non-asset path under /cp/ returns index.html
        # so React's BrowserRouter can handle the client-side route.
        target = DASH_BUILD / full_path
        if target.is_file() and not full_path.startswith("assets/"):
            return FileResponse(str(target))
        return FileResponse(str(DASH_BUILD / "index.html"))

    @app.get("/cp")
    async def _cp_root():
        return FileResponse(str(DASH_BUILD / "index.html"))

    @app.get("/")
    async def _root():
        return JSONResponse(
            {"hint": "open http://127.0.0.1:8000/cp/migrate"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
