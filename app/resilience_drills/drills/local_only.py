"""local_only — quarterly "Local-Only Day" drill (Q17.2).

DRY-RUN: never issues live LLM calls. Probes:
  1. Ollama TCP + /api/tags
  2. Non-dominant provider key formats (Groq/Gemini/DeepSeek/MiniMax)

Passes when ≥50% of non-dominant providers are ready. Below that
threshold, files a CR proposing operator action (start Ollama,
set GROQ_API_KEY, etc.).
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.resilience_drills.audit import (
    acquire_drill_lock,
    append_result,
    emit_landmark_for,
    last_result_for,
    release_drill_lock,
)
from app.resilience_drills.protocol import (
    DrillResult,
    DrillRisk,
    DrillSpec,
    DrillStatus,
    drill_enabled,
    register,
)

logger = logging.getLogger(__name__)


SPEC = DrillSpec(
    name="local_only",
    cadence_days=90,
    grace_days=30,
    risk=DrillRisk.LOW,
    description=(
        "Quarterly Local-Only Day inspection. Probes non-dominant LLM "
        "providers for structural readiness. NEVER issues live LLM calls."
    ),
    requires_master_switch="drill_local_only_enabled",
)


_OLLAMA_PROBE_TIMEOUT_S = 1.5
_OLLAMA_HTTP_TIMEOUT_S = 3.0

_PROVIDER_KEY_PATTERNS = {
    "groq": re.compile(r"^gsk_[A-Za-z0-9]{20,}$"),
    "gemini": re.compile(r"^[A-Za-z0-9_-]{30,}$"),
    "google": re.compile(r"^[A-Za-z0-9_-]{30,}$"),
    "deepseek": re.compile(r"^sk-[A-Za-z0-9]{20,}$"),
    "minimax": re.compile(r"^[A-Za-z0-9_-]{30,}$"),
}

_PROVIDER_ENV = {
    "ollama": "OLLAMA_BASE_URL",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


def _probe_ollama() -> dict[str, Any]:
    out: dict[str, Any] = {"provider": "ollama", "ready": False}
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    out["base_url"] = base
    host = "127.0.0.1"
    port = 11434
    if base.startswith("http://") or base.startswith("https://"):
        rest = base.split("://", 1)[1]
        host_port = rest.split("/", 1)[0]
        if ":" in host_port:
            host, port_s = host_port.split(":", 1)
            try:
                port = int(port_s)
            except ValueError:
                port = 11434
        else:
            host = host_port
            port = 80 if base.startswith("http://") else 443
    try:
        with socket.create_connection((host, port), timeout=_OLLAMA_PROBE_TIMEOUT_S):
            out["socket_ok"] = True
    except (socket.timeout, OSError) as exc:
        out["socket_ok"] = False
        out["socket_error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        req = urllib.request.Request(base.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=_OLLAMA_HTTP_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw) if raw.strip() else {}
        models = parsed.get("models") or []
        out["http_ok"] = True
        out["n_models"] = len(models)
        out["model_names"] = [m.get("name") for m in models if isinstance(m, dict)][:20]
        out["ready"] = bool(models)
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        out["http_ok"] = False
        out["http_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _probe_key(provider: str) -> dict[str, Any]:
    env_name = _PROVIDER_ENV.get(provider)
    out: dict[str, Any] = {"provider": provider, "ready": False, "env": env_name}
    if not env_name:
        return out
    val = os.environ.get(env_name, "").strip()
    out["env_set"] = bool(val)
    if not val:
        return out
    pat = _PROVIDER_KEY_PATTERNS.get(provider)
    if pat is None:
        out["format_ok"] = None
        out["ready"] = True
        return out
    out["format_ok"] = bool(pat.match(val))
    out["ready"] = out["format_ok"]
    return out


def _discover_cascade_providers() -> list[str]:
    candidates = {"ollama", "groq", "gemini", "google", "deepseek", "minimax"}
    return sorted(candidates)


def _persist_report(report: dict[str, Any]) -> Path | None:
    try:
        from app.paths import WORKSPACE_ROOT
        base = Path(WORKSPACE_ROOT) / "resilience" / "local_only"
    except Exception:
        base = Path("/app/workspace/resilience/local_only")
    base.mkdir(parents=True, exist_ok=True)
    fname = datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".json"
    out = base / fname
    try:
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return out
    except OSError:
        logger.debug("local_only: report write failed", exc_info=True)
        return None


def _file_readiness_cr(report: dict[str, Any]) -> str | None:
    try:
        from app.change_requests.lifecycle import create_request
        from app.change_requests.models import RiskClass
    except Exception:
        return None
    n = report.get("n_providers", 0)
    n_ready = report.get("n_ready", 0)
    if n == 0:
        return None
    ratio = n_ready / n
    if ratio >= 0.5:
        return None
    body = [
        f"LOCAL_ONLY_DRILL: only {n_ready}/{n} non-dominant providers ready.",
        "",
        "Per-provider readiness:",
    ]
    for p in report.get("providers", []):
        ready = "✓" if p.get("ready") else "✗"
        body.append(f"  {ready} {p.get('provider'):12s} env={p.get('env_set')} "
                    f"socket_ok={p.get('socket_ok')} format_ok={p.get('format_ok')}")
    try:
        cr = create_request(
            requestor="local_only_drill",
            path="docs/RESILIENCE_DRILLS.md",
            new_content="",
            old_content="",
            reason="\n".join(body),
            risk_class=RiskClass.STANDARD,
        )
        return getattr(cr, "id", None)
    except Exception:
        logger.debug("local_only: CR filing failed", exc_info=True)
        return None


def _run(*, dry_run: bool = True) -> DrillResult:
    started = datetime.now(timezone.utc)
    t0 = time.time()
    prior = last_result_for(SPEC.name)
    prior_status = prior.get("status") if prior else None
    if not drill_enabled(SPEC):
        result = DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.SKIPPED,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0,
            dry_run=dry_run,
            detail={"reason": "master switch off"},
        )
        append_result(result)
        return result
    lock = acquire_drill_lock(SPEC.name)
    if lock is None:
        result = DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.SKIPPED,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0,
            dry_run=dry_run,
            detail={"reason": "lock held"},
        )
        append_result(result)
        return result
    try:
        providers = _discover_cascade_providers()
        probes: list[dict[str, Any]] = []
        for p in providers:
            if p == "ollama":
                probes.append(_probe_ollama())
            else:
                probes.append(_probe_key(p))
        n_ready = sum(1 for p in probes if p.get("ready"))
        report = {
            "ts": started.isoformat(),
            "n_providers": len(providers),
            "n_ready": n_ready,
            "providers": probes,
            "ratio": (n_ready / len(providers)) if providers else 0.0,
        }
        path = _persist_report(report)
        cr_id = None
        status = DrillStatus.PASS if report["ratio"] >= 0.5 else DrillStatus.FAIL
        if status == DrillStatus.FAIL:
            cr_id = _file_readiness_cr(report)
        result = DrillResult(
            drill_name=SPEC.name,
            status=status,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0,
            dry_run=dry_run,
            detail={
                "report_path": str(path) if path else None,
                "n_providers": report["n_providers"],
                "n_ready": report["n_ready"],
                "ratio": report["ratio"],
                "cr_id": cr_id,
                "probes": probes,
            },
        )
        append_result(result)
        emit_landmark_for(result, prior_status=prior_status)
        return result
    except Exception as exc:
        logger.debug("local_only: drill errored", exc_info=True)
        result = DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.ERROR,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0,
            dry_run=dry_run,
            detail={},
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        append_result(result)
        emit_landmark_for(result, prior_status=prior_status)
        return result
    finally:
        release_drill_lock(SPEC.name, lock)


def run(*, dry_run: bool = True) -> DrillResult:
    return _run(dry_run=dry_run)


register(SPEC, run)
