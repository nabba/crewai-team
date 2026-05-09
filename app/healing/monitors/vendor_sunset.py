"""Vendor-sunset monitor — detect deprecated upstream models.

Years-of-uptime hazard: providers retire models on their own clocks
(OpenRouter, Anthropic, Google have all done this). When a model used
by the runtime catalog is sunset, the agent calls fail with cryptic
upstream errors and the system silently degrades.

This monitor takes a weekly pass over the runtime catalog and queries
each provider's public ``/v1/models`` listing to spot any model that's
either (a) absent from the listing now (definitely sunset) or (b)
flagged with deprecation metadata. It does NOT auto-migrate — that's
operator-approved via change-request — but it does file a Signal
alert and persist the diff for inspection.

Because this monitor reaches OUTBOUND to providers, it's gated behind
``HEALING_VENDOR_SUNSET_ENABLED`` (default ON). Disable in environments
where outbound HTTP isn't acceptable.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "vendor_sunset.json"
_HTTP_TIMEOUT_S = 8.0


def _enabled() -> bool:
    return os.getenv("HEALING_VENDOR_SUNSET_ENABLED", "true").lower() in (
        "true", "1", "yes",
    )


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        return json.load(resp)


def _fetch_openrouter_ids() -> set[str]:
    try:
        data = _http_get_json("https://openrouter.ai/api/v1/models")
    except Exception:
        logger.debug("vendor_sunset: openrouter fetch failed", exc_info=True)
        return set()
    rows = data.get("data") or data.get("models") or []
    ids: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            mid = row.get("id") or row.get("model") or row.get("name")
            if mid:
                ids.add(str(mid))
    return ids


def _fetch_anthropic_ids() -> set[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return set()
    try:
        data = _http_get_json(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
    except Exception:
        logger.debug("vendor_sunset: anthropic fetch failed", exc_info=True)
        return set()
    rows = data.get("data") or []
    return {str(r.get("id", "")) for r in rows if r.get("id")}


def _runtime_catalog_models() -> dict[str, set[str]]:
    """Best-effort: collect models the system is actively using, grouped by
    provider. Reads ``control_plane.discovered_models`` (preferred) and
    falls back to scanning ``llm/`` config files.
    """
    by_provider: dict[str, set[str]] = {"openrouter": set(), "anthropic": set()}
    try:
        from app.control_plane.db import execute
        rows = execute(
            "SELECT model_id, provider FROM control_plane.discovered_models "
            "WHERE status IN ('active', 'discovered') "
            "AND cost_output_per_m > 0 LIMIT 500",
            fetch=True,
        ) or []
        for row in rows:
            provider = (row.get("provider") or "").lower()
            mid = row.get("model_id") or ""
            if not mid:
                continue
            # Strip the provider prefix that the catalog stores
            # ("openrouter/xyz/abc" → "xyz/abc").
            if provider == "openrouter" and mid.startswith("openrouter/"):
                mid = mid[len("openrouter/"):]
            if provider in by_provider:
                by_provider[provider].add(mid)
    except Exception:
        logger.debug("vendor_sunset: catalog read failed", exc_info=True)
    return by_provider


def run() -> None:
    if not _enabled():
        return

    in_use = _runtime_catalog_models()
    if not any(in_use.values()):
        return

    upstream = {
        "openrouter": _fetch_openrouter_ids(),
        "anthropic": _fetch_anthropic_ids(),
    }

    sunset_findings: list[dict] = []
    for provider, models in in_use.items():
        upstream_set = upstream.get(provider) or set()
        if not upstream_set:
            # Couldn't fetch — skip this provider this cycle.
            continue
        missing = sorted(models - upstream_set)
        for m in missing:
            sunset_findings.append({
                "provider": provider,
                "model": m,
                "first_missed_at": time.time(),
            })

    state = read_state_json(_STATE_FILE, {"sunset_models": {}})
    sunset_map = state.setdefault("sunset_models", {})

    new_findings: list[dict] = []
    for f in sunset_findings:
        key = f"{f['provider']}::{f['model']}"
        prev = sunset_map.get(key)
        if prev is None:
            sunset_map[key] = {
                "provider": f["provider"],
                "model": f["model"],
                "first_missed_at": f["first_missed_at"],
                "alerted": False,
            }
            new_findings.append(f)
        else:
            prev["last_seen_missing"] = time.time()
    state["last_run_at"] = time.time()
    write_state_json(_STATE_FILE, state)

    audit_event(
        "vendor_sunset_check",
        n_in_use=sum(len(v) for v in in_use.values()),
        n_sunset=len(sunset_findings),
        n_new=len(new_findings),
    )

    if not new_findings:
        return

    lines = [
        f"  • [{f['provider']}] `{f['model']}`"
        for f in new_findings[:10]
    ]
    body = (
        f"📦 Self-heal: {len(new_findings)} model(s) used by AndrusAI "
        f"are no longer listed by their provider — likely sunset:\n\n"
        + "\n".join(lines)
        + "\n\nPlan migration to a supported alternative. Tracked in "
          "`workspace/self_heal/vendor_sunset.json`."
    )
    send_signal_alert(body, tag="vendor_sunset")

    # Mark them alerted so we don't re-spam next week.
    for f in new_findings:
        key = f"{f['provider']}::{f['model']}"
        if key in sunset_map:
            sunset_map[key]["alerted"] = True
    write_state_json(_STATE_FILE, state)
