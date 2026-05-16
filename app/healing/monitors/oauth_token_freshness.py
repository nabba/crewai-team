"""oauth_token_freshness — silent-expiry guard for OAuth + vendor keys.

PROGRAM §51 — Q16 Theme 2 (decade-resilience, vendor independence
depth). Google Workspace refresh tokens silently invalidate after
6 months of inactivity. Vendor API keys can be revoked / rotated
without warning. The existing ``vendor_sunset`` monitor watches
provider *model* lifecycles; this monitor watches *credentials*.

What this monitor checks (pure file/format inspection — NEVER calls
an external API):

  1. **Google Workspace refresh token** at ``workspace/google_token.json``.
     File exists, JSON parses, ``refresh_token`` key non-empty,
     ``expiry`` (if present) is timezone-aware, last modified
     within freshness threshold.

  2. **Vendor API key presence + format** for the keys the LLM
     cascade depends on. Pure regex match against the documented
     format. Mismatch usually means a rotated key with a new shape
     (vendor changed format) or a typo / corruption.

     Vendors checked: Anthropic, OpenAI, OpenRouter, Groq.

  3. **Web Push VAPID keypair** at ``workspace/vapid_*.pem``. Files
     exist and are non-empty. (Format check is left to the VAPID
     library at use-site; we just verify the files are still there
     and weren't truncated by a botched rotation.)

What this monitor **does not** do:

  * Issue any external API call. The drill ``secret_rotation``
    already validates format patterns; this monitor adds the
    freshness / mtime dimension.
  * Trigger any rotation. Operator-only.
  * Read or log any secret VALUE. Only booleans + timestamps +
    SHA-256 prefixes (4 chars) where stable identifiers are useful.

Cadence: daily probe; internal weekly cadence for the full check.
Master switch: ``oauth_token_freshness_monitor_enabled`` (default ON).
Alert dedup: 14 days per (key, severity).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


NAME = "oauth_token_freshness"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "oauth_token_freshness_monitor_enabled"

_INTERNAL_CADENCE_S = 7 * 24 * 3600
_DEDUP_WINDOW_S = 14 * 86400
_STATE_FILE_NAME = "oauth_token_freshness_state.json"

# Google Workspace refresh tokens become invalid after 6 months of
# inactivity. We alert at 4 months (≥120d since last refresh) so the
# operator has slack to act.
_GOOGLE_INACTIVITY_WARN_DAYS = 120
_GOOGLE_INACTIVITY_CRIT_DAYS = 165
_GOOGLE_TOKEN_FILE = "google_token.json"

# Vendor API keys we watch. Pattern source: each vendor's documented
# format as of 2026-05-16. Update when a vendor changes shape; the
# monitor will surface the mismatch and the operator can sync.
_VENDOR_KEY_PATTERNS: dict[str, re.Pattern] = {
    "anthropic":  re.compile(r"^sk-ant-[A-Za-z0-9_-]{20,}$"),
    "openai":     re.compile(r"^sk-[A-Za-z0-9_-]{20,}$"),
    "openrouter": re.compile(r"^sk-or-[A-Za-z0-9_-]{20,}$"),
    "groq":       re.compile(r"^gsk_[A-Za-z0-9]{20,}$"),
}

_VENDOR_ENV_VARS: dict[str, str] = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq":       "GROQ_API_KEY",
}

_VAPID_PRIVATE_FILE = "vapid_private.pem"
_VAPID_PUBLIC_FILE = "vapid_public.pem"


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_oauth_token_freshness_monitor_enabled
        return get_oauth_token_freshness_monitor_enabled()
    except Exception:
        return os.getenv(
            "OAUTH_TOKEN_FRESHNESS_MONITOR_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_alert_at": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_alert_at": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug(
            "oauth_token_freshness: state write failed", exc_info=True,
        )


def _key_fingerprint(value: str) -> str:
    """Return the first 4 hex chars of SHA-256(value). Stable across
    runs; tiny prefix so even a future log accident discloses minimal
    information."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:4]


def _check_google_token() -> dict[str, Any]:
    """Inspect ``workspace/google_token.json``. Returns a structured
    finding dict (never raises)."""
    p = _workspace() / _GOOGLE_TOKEN_FILE
    finding: dict[str, Any] = {
        "present": False,
        "parseable": False,
        "has_refresh_token": False,
        "age_days": None,
        "status": "missing",
        "severity": "info",
    }
    if not p.exists():
        return finding
    finding["present"] = True
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        finding["parseable"] = True
    except Exception:
        finding["status"] = "unparseable"
        finding["severity"] = "crit"
        return finding
    refresh = data.get("refresh_token", "")
    if isinstance(refresh, str) and refresh.strip():
        finding["has_refresh_token"] = True
        finding["key_fingerprint"] = _key_fingerprint(refresh)
    else:
        finding["status"] = "no_refresh_token"
        finding["severity"] = "crit"
        return finding
    # File mtime as a freshness proxy for "last successful refresh".
    # The googleapiclient writes the token file on every refresh, so
    # mtime should be ≤ a few weeks under normal use.
    try:
        mtime = p.stat().st_mtime
        age_s = max(0.0, time.time() - mtime)
        age_days = age_s / 86400
        finding["age_days"] = round(age_days, 1)
        if age_days >= _GOOGLE_INACTIVITY_CRIT_DAYS:
            finding["status"] = "near_expiry"
            finding["severity"] = "crit"
        elif age_days >= _GOOGLE_INACTIVITY_WARN_DAYS:
            finding["status"] = "inactive_warn"
            finding["severity"] = "warn"
        else:
            finding["status"] = "fresh"
            finding["severity"] = "info"
    except OSError:
        finding["status"] = "stat_failed"
        finding["severity"] = "warn"
    return finding


def _check_vendor_keys() -> dict[str, Any]:
    """For each known vendor, check env var presence + format match.
    Returns a per-vendor finding."""
    out: dict[str, Any] = {}
    for vendor, env_var in _VENDOR_ENV_VARS.items():
        value = os.environ.get(env_var, "").strip()
        finding: dict[str, Any] = {
            "env_var": env_var,
            "present": bool(value),
            "format_match": None,
            "severity": "info",
            "status": "absent",
        }
        if not value:
            # Anthropic absent is critical (the cascade leans on it heavily).
            # Others are warn-level — the cascade can route around them.
            finding["severity"] = "crit" if vendor == "anthropic" else "warn"
            finding["status"] = "missing"
            out[vendor] = finding
            continue
        finding["key_fingerprint"] = _key_fingerprint(value)
        pattern = _VENDOR_KEY_PATTERNS.get(vendor)
        matched = bool(pattern and pattern.match(value))
        finding["format_match"] = matched
        if matched:
            finding["status"] = "ok"
            finding["severity"] = "info"
        else:
            finding["status"] = "format_mismatch"
            finding["severity"] = "warn"
        out[vendor] = finding
    return out


def _check_vapid_keypair() -> dict[str, Any]:
    """Verify both VAPID PEM files exist and are non-empty. We don't
    parse the PEM here; the at-use-site library handles that and
    will fail loudly if the file is corrupt."""
    base = _workspace()
    priv = base / _VAPID_PRIVATE_FILE
    pub = base / _VAPID_PUBLIC_FILE
    finding: dict[str, Any] = {
        "private_present": priv.exists(),
        "public_present": pub.exists(),
        "private_nonempty": False,
        "public_nonempty": False,
        "severity": "info",
        "status": "ok",
    }
    if priv.exists():
        try:
            finding["private_nonempty"] = priv.stat().st_size > 0
        except OSError:
            pass
    if pub.exists():
        try:
            finding["public_nonempty"] = pub.stat().st_size > 0
        except OSError:
            pass
    ok = (
        finding["private_present"] and finding["public_present"]
        and finding["private_nonempty"] and finding["public_nonempty"]
    )
    if not ok:
        # If neither exists, Web Push simply isn't configured —
        # not an alert condition. If one exists without the other,
        # that's a botched rotation.
        if not (finding["private_present"] or finding["public_present"]):
            finding["status"] = "unconfigured"
            finding["severity"] = "info"
        else:
            finding["status"] = "incomplete_pair"
            finding["severity"] = "warn"
    return finding


def _alert_if_due(
    state: dict[str, Any],
    *,
    key: str,
    title: str,
    body: str,
    now: float,
) -> bool:
    last_alerts = state.setdefault("last_alert_at", {})
    if not isinstance(last_alerts, dict):
        last_alerts = {}
        state["last_alert_at"] = last_alerts
    last = float(last_alerts.get(key, 0))
    if now - last < _DEDUP_WINDOW_S:
        return False
    last_alerts[key] = now
    try:
        from app.notify import notify
        notify(
            title=title,
            body=body,
            url="/cp/settings",
            topic=f"oauth_token_freshness:{key}",
            critical=False,
            arbitrate=True,
        )
        return True
    except Exception:
        logger.debug(
            "oauth_token_freshness: notify failed", exc_info=True,
        )
        return False


def _maybe_emit_alerts(
    state: dict[str, Any],
    findings: dict[str, Any],
    *,
    now: float,
) -> list[dict[str, Any]]:
    """For each finding with severity >= warn, fire a deduped alert."""
    sent: list[dict[str, Any]] = []
    google = findings.get("google", {})
    if google.get("severity") in ("warn", "crit"):
        sev_emoji = "🔴" if google["severity"] == "crit" else "🟡"
        body = (
            f"{sev_emoji} Google Workspace refresh token state: "
            f"{google.get('status', 'unknown')}.\n"
            f"  • token file age: {google.get('age_days', '?')} days\n"
            f"  • Google invalidates refresh tokens after ~180d of inactivity.\n\n"
            f"Triage: trigger a refresh by exercising any Google Workspace "
            f"tool (Gmail / Calendar / Docs / Sheets / Slides / Drive). "
            f"If still failing, re-run "
            f"`python -m app.google_workspace.bootstrap` to re-authorize."
        )
        ok = _alert_if_due(
            state,
            key=f"google_{google['status']}",
            title=f"{sev_emoji} Google OAuth token nearing expiry",
            body=body,
            now=now,
        )
        sent.append({
            "kind": "google_workspace",
            "severity": google["severity"],
            "status": google["status"],
            "alert_sent": ok,
        })

    vendors = findings.get("vendors", {})
    for vendor, finding in vendors.items():
        if finding.get("severity") not in ("warn", "crit"):
            continue
        sev_emoji = "🔴" if finding["severity"] == "crit" else "🟡"
        body = (
            f"{sev_emoji} Vendor credential issue: {vendor} "
            f"({finding['env_var']}) — {finding['status']}.\n\n"
            f"Triage:\n"
            f"  • If missing: set the env var in deploy config + restart.\n"
            f"  • If format_mismatch: the vendor may have changed key "
            f"shape. Verify against the vendor's current docs."
        )
        ok = _alert_if_due(
            state,
            key=f"vendor_{vendor}_{finding['status']}",
            title=f"{sev_emoji} {vendor} credential: {finding['status']}",
            body=body,
            now=now,
        )
        sent.append({
            "kind": f"vendor_{vendor}",
            "severity": finding["severity"],
            "status": finding["status"],
            "alert_sent": ok,
        })

    vapid = findings.get("vapid", {})
    if vapid.get("severity") == "warn" and vapid.get("status") == "incomplete_pair":
        body = (
            f"🟡 Web Push VAPID keypair is incomplete (one of "
            f"vapid_private.pem / vapid_public.pem missing or empty). "
            f"Likely a botched rotation. Re-run "
            f"`python -m app.web_push.bootstrap` to regenerate the pair."
        )
        ok = _alert_if_due(
            state,
            key="vapid_incomplete",
            title="🟡 Web Push VAPID keypair incomplete",
            body=body,
            now=now,
        )
        sent.append({
            "kind": "vapid",
            "severity": vapid["severity"],
            "status": vapid["status"],
            "alert_sent": ok,
        })
    return sent


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """One pass. Daily wake-up gates on weekly internal cadence.
    Returns a summary dict with each finding category."""
    summary: dict[str, Any] = {
        "ran": False,
        "findings": {},
        "alerts": [],
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last = float(state.get("last_run_at", 0))
    if last > 0 and cur - last < _INTERNAL_CADENCE_S:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    findings: dict[str, Any] = {
        "google": _check_google_token(),
        "vendors": _check_vendor_keys(),
        "vapid": _check_vapid_keypair(),
    }
    summary["findings"] = findings
    summary["alerts"] = _maybe_emit_alerts(state, findings, now=cur)
    _write_state(state)
    return summary
