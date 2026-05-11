"""
config_api.py — Configuration management endpoints.

Extracted from main.py. Handles LLM mode switching.
"""

import hmac
import logging
import threading
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["config"])

# Rate limiter for config endpoints — max 5 changes per minute
_config_rate_bucket: list = []
_config_rate_lock = threading.Lock()


def _config_rate_check() -> bool:
    """Return True if within rate limit for config changes."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=1)
    with _config_rate_lock:
        _config_rate_bucket[:] = [t for t in _config_rate_bucket if t > cutoff]
        if len(_config_rate_bucket) >= 5:
            return False
        _config_rate_bucket.append(now)
        return True


def verify_gateway_secret(request: Request) -> bool:
    """Verify the forwarder is authenticated with the gateway secret."""
    from app.config import get_gateway_secret
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth[7:]
    return hmac.compare_digest(token, get_gateway_secret())


@router.get("/llm_mode")
async def get_llm_mode_endpoint():
    """Return the current LLM mode + the list of accepted values."""
    from app.llm_mode import get_mode, VALID_MODES
    return {"mode": get_mode(), "valid_modes": list(VALID_MODES)}


@router.post("/llm_mode")
async def set_llm_mode_endpoint(request: Request):
    """Switch LLM mode. See app.llm_mode for semantics."""
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _config_rate_check():
        raise HTTPException(status_code=429, detail="Too many config changes. Try again later.")
    payload = await request.json()
    mode = payload.get("mode", "").strip().lower()
    from app.llm_mode import VALID_MODES, set_mode
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Use one of: {', '.join(VALID_MODES)}",
        )
    from app.firebase_reporter import report_llm_mode
    set_mode(mode)
    report_llm_mode(mode)
    return {"status": "ok", "mode": mode}


@router.get("/creative_mode")
async def get_creative_mode_endpoint():
    """Return current creative-mode runtime settings (budget, originality weight)."""
    from app.creative_mode import snapshot
    return snapshot()


@router.post("/creative_mode")
async def set_creative_mode_endpoint(request: Request):
    """Update creative-mode runtime settings.

    Accepts any subset of: creative_run_budget_usd (float),
    originality_wiki_weight (float in [0, 1]).
    """
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _config_rate_check():
        raise HTTPException(status_code=429, detail="Too many config changes. Try again later.")
    payload = await request.json()
    from app.creative_mode import (
        set_budget_usd, set_originality_wiki_weight, snapshot,
    )

    if "creative_run_budget_usd" in payload:
        try:
            set_budget_usd(float(payload["creative_run_budget_usd"]))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if "originality_wiki_weight" in payload:
        try:
            set_originality_wiki_weight(float(payload["originality_wiki_weight"]))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return {"status": "ok", **snapshot()}


@router.get("/background_tasks")
async def get_background_tasks_endpoint():
    """Return whether the idle scheduler is currently allowed to run jobs.

    Mirrors the legacy Firestore ``config/background_tasks`` document
    used by the old HTML monitor — same kill-switch, surfaced over HTTP
    so the React Settings page doesn't need a Firebase client.
    """
    from app.idle_scheduler import is_enabled
    return {"enabled": bool(is_enabled())}


@router.post("/background_tasks")
async def set_background_tasks_endpoint(request: Request):
    """Toggle the idle-scheduler kill switch."""
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _config_rate_check():
        raise HTTPException(status_code=429, detail="Too many config changes. Try again later.")
    payload = await request.json()
    if "enabled" not in payload:
        raise HTTPException(status_code=400, detail="missing 'enabled'")
    enabled = bool(payload["enabled"])
    from app.idle_scheduler import set_enabled
    set_enabled(enabled)
    # Mirror to Firestore so the legacy HTML monitor and any other
    # listeners (the in-process Firestore listener at idle_scheduler.py:
    # 2581 included) stay in sync.
    try:
        from app.firebase.infra import _get_db, _now_iso
        db = _get_db()
        if db is not None:
            db.collection("config").document("background_tasks").set(
                {"enabled": enabled, "updated_at": _now_iso()},
                merge=True,
            )
    except Exception:
        logger.debug("background_tasks: firestore mirror failed", exc_info=True)
    return {"status": "ok", "enabled": enabled}


# ── Web Push (PWA notifications, Phase 4 — May 2026) ────────────────────────

@router.get("/vapid_public_key")
async def get_vapid_public_key():
    """Return the VAPID public key (or empty when not configured) so the
    React PWA can subscribe browsers via PushManager.subscribe."""
    from app.config import get_settings
    return {"public_key": get_settings().vapid_public_key or ""}


@router.post("/web_push/subscribe")
async def web_push_subscribe(request: Request):
    """Register a browser's PushSubscription so the gateway can notify it."""
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    payload = await request.json()
    from app.web_push import add_subscription
    ok = add_subscription(payload)
    if not ok:
        raise HTTPException(status_code=400, detail="invalid subscription payload")
    return {"status": "ok"}


@router.post("/web_push/unsubscribe")
async def web_push_unsubscribe(request: Request):
    """Remove a previously-registered subscription by endpoint."""
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    payload = await request.json()
    endpoint = (payload or {}).get("endpoint", "")
    from app.web_push import remove_subscription
    return {"removed": remove_subscription(endpoint)}


@router.get("/web_push/subscriptions")
async def web_push_list():
    """List currently-registered devices (for the Settings page indicator)."""
    from app.web_push import list_subscriptions, is_configured
    subs = list_subscriptions()
    return {
        "configured": is_configured(),
        "count": len(subs),
        "devices": [
            {
                "user_agent": s.get("user_agent", ""),
                "added_at": s.get("added_at", ""),
                "endpoint_host": _endpoint_host(s.get("endpoint", "")),
            }
            for s in subs
        ],
    }


@router.post("/web_push/test")
async def web_push_test(request: Request):
    """Send a test notification to all registered devices."""
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from app.web_push import send_to_all
    delivered = send_to_all(
        title="AndrusAI",
        body="Push notifications are working ✓",
        url="/cp/settings",
        tag="andrusai-test",
    )
    return {"delivered": delivered}


def _endpoint_host(endpoint: str) -> str:
    """Strip the path off an endpoint URL so the UI shows just 'fcm.googleapis.com' etc."""
    try:
        from urllib.parse import urlparse
        return urlparse(endpoint).netloc
    except Exception:
        return ""


# ── Personal-agent runtime settings (Phase 0 — May 2026) ───────────────────
# Voice mode, vision-driven computer-use cap, concierge persona toggle.
# Read path: app.runtime_settings.snapshot(); write path: these endpoints.

@router.get("/runtime_settings")
async def get_runtime_settings_endpoint():
    """Return the live runtime settings (voice mode + vision CU + concierge)."""
    from app.runtime_settings import snapshot
    return snapshot()


@router.post("/runtime_settings")
async def set_runtime_settings_endpoint(request: Request):
    """Update one or more runtime settings.

    Accepts any subset of:
      voice_mode (off|local|cloud), vision_cu_enabled (bool),
      vision_cu_monthly_cap_usd (float), concierge_persona_enabled (bool).
    """
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _config_rate_check():
        raise HTTPException(status_code=429, detail="Too many config changes. Try again later.")
    payload = await request.json()
    from app.runtime_settings import (
        set_voice_mode, set_vision_cu_enabled,
        set_vision_cu_monthly_cap_usd, set_concierge_persona_enabled,
        set_tier3_amendment_enabled,
        set_error_runbooks_enabled, set_tool_supervisor_enabled,
        set_recovery_loop_enabled,
        set_goodhart_hard_gate_disabled, set_goodhart_hard_gate_enforcing,
        set_structured_diagnosis_threshold_floor,
        set_structured_diagnosis_threshold_ceiling,
        set_structured_diagnosis_threshold_override,
        set_structured_diagnosis_auto_tune_enabled,
        set_embedding_migration_dual_write_enabled,
        set_embedding_migration_shadow_read_enabled,
        set_embedding_migration_cutover_enabled,
        # Q4.2 — person correlation
        set_person_correlation_enabled,
        set_person_correlation_decay_months,
        set_person_centrality_enabled,
        set_person_centrality_formula,
        set_person_suggestions_enabled,
        set_person_suggestions_dormancy_enabled,
        set_person_suggestions_responsiveness_enabled,
        set_person_correlation_social_graph_enabled,
        get_person_correlation_social_graph_enabled,
        set_graph_shortest_path_enabled,
        set_graph_communities_enabled,
        set_graph_bridges_enabled,
        set_graph_suggestions_enabled,
        get_graph_suggestions_enabled,
        set_graph_suggestions_cluster_dormancy_enabled,
        set_graph_suggestions_bridge_maintenance_enabled,
        set_graph_suggestions_weak_tie_enabled,
        snapshot,
    )

    try:
        if "voice_mode" in payload:
            set_voice_mode(str(payload["voice_mode"]))
        if "vision_cu_enabled" in payload:
            set_vision_cu_enabled(bool(payload["vision_cu_enabled"]))
        if "vision_cu_monthly_cap_usd" in payload:
            set_vision_cu_monthly_cap_usd(float(payload["vision_cu_monthly_cap_usd"]))
        if "concierge_persona_enabled" in payload:
            set_concierge_persona_enabled(bool(payload["concierge_persona_enabled"]))
        if "tier3_amendment_enabled" in payload:
            set_tier3_amendment_enabled(bool(payload["tier3_amendment_enabled"]))
        if "error_runbooks_enabled" in payload:
            set_error_runbooks_enabled(bool(payload["error_runbooks_enabled"]))
        if "tool_supervisor_enabled" in payload:
            set_tool_supervisor_enabled(bool(payload["tool_supervisor_enabled"]))
        if "recovery_loop_enabled" in payload:
            set_recovery_loop_enabled(bool(payload["recovery_loop_enabled"]))
        if "goodhart_hard_gate_disabled" in payload:
            set_goodhart_hard_gate_disabled(bool(payload["goodhart_hard_gate_disabled"]))
        if "goodhart_hard_gate_enforcing" in payload:
            set_goodhart_hard_gate_enforcing(bool(payload["goodhart_hard_gate_enforcing"]))
        # Structured-diagnosis threshold band (Q2 §39).
        if "structured_diagnosis_threshold_floor" in payload:
            set_structured_diagnosis_threshold_floor(
                float(payload["structured_diagnosis_threshold_floor"])
            )
        if "structured_diagnosis_threshold_ceiling" in payload:
            set_structured_diagnosis_threshold_ceiling(
                float(payload["structured_diagnosis_threshold_ceiling"])
            )
        if "structured_diagnosis_threshold_override" in payload:
            raw = payload["structured_diagnosis_threshold_override"]
            set_structured_diagnosis_threshold_override(
                None if raw is None else float(raw)
            )
        if "structured_diagnosis_auto_tune_enabled" in payload:
            set_structured_diagnosis_auto_tune_enabled(
                bool(payload["structured_diagnosis_auto_tune_enabled"])
            )
        # Embedding-migration master switches (PROGRAM §40 Item 12).
        if "embedding_migration_dual_write_enabled" in payload:
            set_embedding_migration_dual_write_enabled(
                bool(payload["embedding_migration_dual_write_enabled"])
            )
        if "embedding_migration_shadow_read_enabled" in payload:
            set_embedding_migration_shadow_read_enabled(
                bool(payload["embedding_migration_shadow_read_enabled"])
            )
        if "embedding_migration_cutover_enabled" in payload:
            set_embedding_migration_cutover_enabled(
                bool(payload["embedding_migration_cutover_enabled"])
            )

        # ── Q4.2 — person correlation (PROGRAM §42) ──────────────────
        if "person_correlation_enabled" in payload:
            set_person_correlation_enabled(bool(payload["person_correlation_enabled"]))
        if "person_correlation_decay_months" in payload:
            set_person_correlation_decay_months(
                int(payload["person_correlation_decay_months"])
            )
        if "person_centrality_enabled" in payload:
            set_person_centrality_enabled(bool(payload["person_centrality_enabled"]))
        if "person_centrality_formula" in payload:
            set_person_centrality_formula(str(payload["person_centrality_formula"]))
        if "person_suggestions_enabled" in payload:
            set_person_suggestions_enabled(bool(payload["person_suggestions_enabled"]))
        if "person_suggestions_dormancy_enabled" in payload:
            set_person_suggestions_dormancy_enabled(bool(payload["person_suggestions_dormancy_enabled"]))
        if "person_suggestions_responsiveness_enabled" in payload:
            set_person_suggestions_responsiveness_enabled(bool(payload["person_suggestions_responsiveness_enabled"]))

        # L4 master — typed-phrase gate on False→True transition.
        # The phrase MUST be present and match exactly OR the new value
        # must be False (disable doesn't require a phrase).
        if "person_correlation_social_graph_enabled" in payload:
            new_val = bool(payload["person_correlation_social_graph_enabled"])
            if new_val and not get_person_correlation_social_graph_enabled():
                # Enabling: require phrase.
                phrase = str(payload.get("social_graph_confirm_phrase") or "")
                if phrase != "ENABLE SOCIAL GRAPH":
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Enabling the social graph requires the "
                            "typed-phrase confirmation. Send "
                            "social_graph_confirm_phrase='ENABLE SOCIAL GRAPH' "
                            "alongside person_correlation_social_graph_enabled=true."
                        ),
                    )
            set_person_correlation_social_graph_enabled(new_val)

        if "graph_shortest_path_enabled" in payload:
            set_graph_shortest_path_enabled(bool(payload["graph_shortest_path_enabled"]))
        if "graph_communities_enabled" in payload:
            set_graph_communities_enabled(bool(payload["graph_communities_enabled"]))
        if "graph_bridges_enabled" in payload:
            set_graph_bridges_enabled(bool(payload["graph_bridges_enabled"]))

        # L4.4 master — SECOND typed-phrase gate on False→True.
        if "graph_suggestions_enabled" in payload:
            new_val = bool(payload["graph_suggestions_enabled"])
            if new_val and not get_graph_suggestions_enabled():
                phrase = str(payload.get("graph_suggestions_confirm_phrase") or "")
                if phrase != "ENABLE GRAPH-DRIVEN SUGGESTIONS":
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Enabling graph-driven suggestions requires the "
                            "typed-phrase confirmation. Send "
                            "graph_suggestions_confirm_phrase="
                            "'ENABLE GRAPH-DRIVEN SUGGESTIONS' alongside "
                            "graph_suggestions_enabled=true."
                        ),
                    )
            set_graph_suggestions_enabled(new_val)

        if "graph_suggestions_cluster_dormancy_enabled" in payload:
            set_graph_suggestions_cluster_dormancy_enabled(
                bool(payload["graph_suggestions_cluster_dormancy_enabled"])
            )
        if "graph_suggestions_bridge_maintenance_enabled" in payload:
            set_graph_suggestions_bridge_maintenance_enabled(
                bool(payload["graph_suggestions_bridge_maintenance_enabled"])
            )
        if "graph_suggestions_weak_tie_enabled" in payload:
            set_graph_suggestions_weak_tie_enabled(
                bool(payload["graph_suggestions_weak_tie_enabled"])
            )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Audit the change so the operator can see who flipped what.
    try:
        import json as _json
        from app.audit import log_security_event
        log_security_event(
            "runtime_settings_change",
            _json.dumps({"changed": list(payload.keys()), "after": snapshot()}),
        )
    except Exception:
        logger.debug("runtime_settings audit log failed", exc_info=True)

    return {"status": "ok", **snapshot()}


# ── Governance ratchet (Wave 3 #6 — May 2026) ──────────────────────────────
# Operator-controlled raising/relaxing of SAFETY_MINIMUM and
# QUALITY_MINIMUM above the hardcoded FLOOR in governance.py. Both
# routes are gateway-bearer-secret gated. Relax requires an explicit
# typed-phrase confirmation in the request body — UI-side typed
# confirmation is a UX gate; THIS check is the authoritative one.


@router.get("/governance_ratchet/state")
async def get_governance_ratchet_state():
    """Snapshot of every ratchet-controlled threshold + history."""
    from app.governance_ratchet import list_thresholds
    return {"thresholds": list_thresholds()}


@router.post("/governance_ratchet/set")
async def set_governance_ratchet_endpoint(request: Request):
    """Raise a threshold floor.

    Body: ``{"name": "safety_minimum"|"quality_minimum",
             "new_value": 0.0..1.0,
             "reason": str}``.

    Refuses (400) if ``new_value <= current`` (use ``/relax`` for
    downward changes). Refuses (400) if ``new_value > 1.0``.
    """
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _config_rate_check():
        raise HTTPException(status_code=429, detail="Too many config changes. Try again later.")
    payload = await request.json()
    from app.governance_ratchet import (
        set_ratchet, MonotonicViolation, CeilingViolation,
        UnknownThresholdViolation,
    )
    name = (payload.get("name") or "").strip()
    new_value = payload.get("new_value")
    reason = (payload.get("reason") or "").strip()
    try:
        state = set_ratchet(
            name=name,
            new_value=float(new_value),
            source="operator_react",
            reason=reason,
        )
    except (MonotonicViolation, CeilingViolation, UnknownThresholdViolation) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "state": state.to_dict()}


@router.get("/runbook_settings")
async def get_runbook_settings_endpoint():
    """List every registered self-heal runbook with its current
    ``enabled`` flag, ``min_recurrence``, and operator comment.

    Reads ``workspace/self_heal/runbook_settings.json``. Empty list
    when the file doesn't exist (no runbooks are configured yet).
    """
    import json as _json
    from pathlib import Path
    state_path = Path("/app/workspace/self_heal/runbook_settings.json")
    if not state_path.exists():
        return {"runbooks": {}}
    try:
        return _json.loads(state_path.read_text())
    except (OSError, _json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="runbook_settings.json malformed")


@router.post("/runbook_settings")
async def set_runbook_settings_endpoint(request: Request):
    """Toggle one runbook's ``enabled`` flag.

    Body: ``{"name": "<runbook_name>", "enabled": bool}``. Optionally
    include ``min_recurrence`` (int) to update the dispatch gate at the
    same time.

    The dispatcher re-reads ``runbook_settings.json`` on every anomaly,
    so the change takes effect immediately — no gateway restart.
    """
    import json as _json
    from pathlib import Path

    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _config_rate_check():
        raise HTTPException(status_code=429, detail="Too many config changes. Try again later.")
    payload = await request.json()
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="missing 'name'")
    if "enabled" not in payload:
        raise HTTPException(status_code=400, detail="missing 'enabled'")

    state_path = Path("/app/workspace/self_heal/runbook_settings.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        try:
            data = _json.loads(state_path.read_text())
        except (OSError, _json.JSONDecodeError):
            data = {"runbooks": {}}
    else:
        data = {"runbooks": {}}

    runbooks = data.setdefault("runbooks", {})
    entry = runbooks.setdefault(name, {})
    entry["enabled"] = bool(payload["enabled"])
    if "min_recurrence" in payload:
        try:
            mr = int(payload["min_recurrence"])
            entry["min_recurrence"] = max(1, min(100, mr))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="min_recurrence must be int")

    # Atomic write so a crash mid-update doesn't corrupt the JSON.
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(state_path)

    # Audit so the operator can grep /cp/audit for who-flipped-what.
    try:
        import json as __json
        from app.audit import log_security_event
        log_security_event(
            "runbook_setting_change",
            __json.dumps({"name": name, "after": entry}),
        )
    except Exception:
        logger.debug("runbook_settings audit log failed", exc_info=True)

    return {"status": "ok", **data}


@router.post("/governance_ratchet/relax")
async def relax_governance_ratchet_endpoint(request: Request):
    """Lower a threshold floor (operator-only, typed-confirmation gated).

    Body: ``{"name": ..., "new_value": ...,
             "confirmation": "RELAX <THRESHOLD>",
             "reason": str}``.

    The ``confirmation`` field is the typed-phrase gate. If the UI
    forgets to send it (or a script tries to bypass the React form),
    the request fails 400. This is *not* security — the gateway-secret
    is — but it's a UX correctness check that catches accidental
    automation.
    """
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _config_rate_check():
        raise HTTPException(status_code=429, detail="Too many config changes. Try again later.")
    payload = await request.json()
    from app.governance_ratchet import (
        relax_ratchet, MonotonicViolation, FloorViolation,
        CeilingViolation, UnknownThresholdViolation,
    )
    name = (payload.get("name") or "").strip()
    new_value = payload.get("new_value")
    reason = (payload.get("reason") or "").strip()
    confirmation = (payload.get("confirmation") or "").strip().upper()

    expected = f"RELAX {name.upper()}"
    if confirmation != expected:
        raise HTTPException(
            status_code=400,
            detail=(
                f"missing or wrong confirmation phrase — relax "
                f"requires confirmation == {expected!r}"
            ),
        )
    if not reason:
        raise HTTPException(
            status_code=400,
            detail="relax requires a non-empty 'reason'",
        )
    try:
        state = relax_ratchet(
            name=name,
            new_value=float(new_value),
            source="operator_react",
            reason=reason,
        )
    except (MonotonicViolation, FloorViolation, CeilingViolation,
            UnknownThresholdViolation) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "state": state.to_dict()}


@router.post("/creative_run")
async def creative_run_endpoint(request: Request):
    """Force-dispatch a task to the creative crew, bypassing the router.

    This is the "Try Creative Mode" entry point — it lets users explicitly
    invoke the multi-agent ideation pipeline without phrasing their request
    in a way the LLM router happens to recognize as creative.

    Accepts: {"task": str, "creativity": "high"|"medium"} — creativity
    defaults to "high". Budget cap (creative_run_budget_usd) still applies.

    Returns: {"final_output": str, "scores": dict, "cost_usd": float,
              "aborted_reason": str|None, "phases": int}.
    """
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    payload = await request.json()
    task = (payload.get("task") or "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")
    if len(task) > 4000:
        raise HTTPException(status_code=400, detail="Task too long (max 4000 chars)")
    creativity = payload.get("creativity", "high")
    if creativity not in ("high", "medium"):
        creativity = "high"

    try:
        from app.crews.creative_crew import run_creative_crew
        from app.rate_throttle import start_request_tracking, stop_request_tracking
        # Track cost so the budget cap can fire mid-run
        tracker = start_request_tracking(request_id=f"creative_run_{int(__import__('time').time())}")
        try:
            result = run_creative_crew(task, creativity=creativity)
        finally:
            stop_request_tracking()
        return {
            "final_output": result.final_output,
            "scores": result.scores,
            "cost_usd": result.cost_usd,
            "aborted_reason": result.aborted_reason,
            "phases": len(result.phase_1_outputs) + len(result.phase_2_outputs),
        }
    except Exception as exc:
        logger.exception("creative_run failed")
        raise HTTPException(status_code=500, detail=f"Creative run failed: {exc}")


# ── Life-companion control panel (2026-05-10) ────────────────────────
# GET returns the registry merged with current overrides — one
# call gives the React page everything it needs to render.  POST
# accepts {"feature_key", "enabled"?, "tunables"?}.  All bearer-
# auth gated and rate-limited like the other config endpoints.


@router.get("/life_companion")
async def get_life_companion_state_endpoint():
    """Return registry + current per-feature override state.

    Shape::

        {
          "master_enabled": bool,
          "features": [
            {
              "key": "act_now_digest",
              "name": "Act-now digest (LLM-graded)",
              "description": "...",
              "feature_env_key": "LIFE_COMPANION_ACT_NOW_DIGEST_ENABLED",
              "job_name": "life-companion-act-now-digest",
              "enabled": true,           # effective (override OR env)
              "enabled_source": "override"|"env"|"default",
              "tunables": [
                {
                  "env_key": "LIFE_COMPANION_ACT_NOW_TOP_K",
                  "label": "Top K", "description": "...",
                  "type": "int", "default": 7,
                  "min": 1, "max": 20, "options": [],
                  "current_value": "7",       # effective value
                  "value_source": "default"|"env"|"override"
                }
              ]
            }
          ]
        }

    No auth required — read-only.  Mutations go through POST below.
    """
    import os
    from app.life_companion import _common as lc_common
    from app.life_companion.feature_registry import list_features
    from app.runtime_settings import life_companion_get_overrides

    overrides = life_companion_get_overrides()
    features_out: list[dict] = []

    for feat in list_features():
        # Resolve effective enabled state + which layer wins.
        feat_override = overrides.get(feat.key) or {}
        override_enabled = feat_override.get("enabled")
        env_value = os.getenv(feat.feature_env_key)

        if override_enabled is not None:
            enabled = bool(override_enabled)
            enabled_source = "override"
        elif env_value is not None:
            enabled = env_value.lower() in ("true", "1", "yes")
            enabled_source = "env"
        else:
            enabled = feat.default_enabled
            enabled_source = "default"

        # Resolve each tunable's effective value + source.
        override_tuns = feat_override.get("tunables") or {}
        tuns_out: list[dict] = []
        for tun in feat.tunables:
            ovr_val = override_tuns.get(tun.env_key)
            env_val = os.getenv(tun.env_key)
            if ovr_val is not None and ovr_val != "":
                current = str(ovr_val)
                source = "override"
            elif env_val is not None and env_val != "":
                current = str(env_val)
                source = "env"
            else:
                current = str(tun.default)
                source = "default"
            tuns_out.append({
                "env_key": tun.env_key,
                "label": tun.label,
                "description": tun.description,
                "type": tun.type,
                "default": tun.default,
                "min": tun.min,
                "max": tun.max,
                "options": list(tun.options),
                "current_value": current,
                "value_source": source,
            })

        features_out.append({
            "key": feat.key,
            "name": feat.name,
            "description": feat.description,
            "feature_env_key": feat.feature_env_key,
            "job_name": feat.job_name,
            "enabled": enabled,
            "enabled_source": enabled_source,
            "tunables": tuns_out,
        })

    return {
        "master_enabled": lc_common._master_enabled(),
        "features": features_out,
    }


@router.post("/life_companion")
async def set_life_companion_feature_endpoint(request: Request):
    """Update one life-companion feature's enabled state and/or
    tunables.

    Body::

        {
          "feature_key": "act_now_digest",
          "enabled": true,                  # optional
          "tunables": {                     # optional
            "LIFE_COMPANION_ACT_NOW_TOP_K": "5"
          }
        }

    Pass ``"enabled": null`` to clear the toggle override (revert
    to env default).  Pass ``"tunables": {"<KEY>": ""}`` to clear
    a single tunable override.

    Bearer-auth gated; rate-limited; audited.
    """
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _config_rate_check():
        raise HTTPException(
            status_code=429,
            detail="Too many config changes. Try again later.",
        )
    payload = await request.json()
    feature_key = (payload.get("feature_key") or "").strip()
    if not feature_key:
        raise HTTPException(status_code=400, detail="Missing 'feature_key'")

    from app.life_companion.feature_registry import find_tunable, get_feature
    feat = get_feature(feature_key)
    if feat is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature_key: {feature_key!r}",
        )

    # Validate tunable keys belong to THIS feature (don't let one
    # feature's overrides leak into another's).
    raw_tunables = payload.get("tunables")
    validated_tunables: dict | None = None
    if raw_tunables is not None:
        if not isinstance(raw_tunables, dict):
            raise HTTPException(
                status_code=400, detail="'tunables' must be an object",
            )
        validated_tunables = {}
        valid_keys = {t.env_key for t in feat.tunables}
        for k, v in raw_tunables.items():
            if k not in valid_keys:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tunable {k!r} not declared for feature {feature_key!r}",
                )
            # Range-validate numeric tunables (UI should already do
            # this; backend is the authoritative gate).
            tun_def = next(t for t in feat.tunables if t.env_key == k)
            if v not in (None, "") and tun_def.type in ("int", "minutes", "hours", "secs"):
                try:
                    iv = int(v)
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{k} must be an integer",
                    )
                if tun_def.min is not None and iv < tun_def.min:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{k} below min ({tun_def.min})",
                    )
                if tun_def.max is not None and iv > tun_def.max:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{k} above max ({tun_def.max})",
                    )
            elif v not in (None, "") and tun_def.type == "float":
                try:
                    fv = float(v)
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400, detail=f"{k} must be a number",
                    )
                if tun_def.min is not None and fv < tun_def.min:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{k} below min ({tun_def.min})",
                    )
                if tun_def.max is not None and fv > tun_def.max:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{k} above max ({tun_def.max})",
                    )
            validated_tunables[k] = v

    # Three "enabled" paths the API supports:
    #   * key absent     — leave toggle untouched
    #   * "enabled": null — clear the override (revert to env)
    #   * "enabled": bool — set the override
    # Map to the runtime-settings function's kwargs explicitly.
    from app.runtime_settings import life_companion_set_feature_override

    if "enabled" not in payload:
        overrides = life_companion_set_feature_override(
            feature_key, tunables=validated_tunables,
        )
        enabled_for_audit = "leave"
    else:
        enabled_raw = payload["enabled"]
        # ``None`` is a valid signal: clear the override.
        if enabled_raw is None:
            overrides = life_companion_set_feature_override(
                feature_key, enabled=None, tunables=validated_tunables,
            )
        else:
            overrides = life_companion_set_feature_override(
                feature_key,
                enabled=bool(enabled_raw),
                tunables=validated_tunables,
            )
        enabled_for_audit = enabled_raw

    # Audit so the operator can see who flipped what.
    try:
        import json as _json
        from app.audit import log_security_event
        log_security_event(
            "life_companion_settings_change",
            _json.dumps({
                "feature_key": feature_key,
                "enabled": enabled_for_audit,
                "tunable_keys_changed": (
                    list(validated_tunables.keys())
                    if validated_tunables else []
                ),
            }),
        )
    except Exception:
        logger.debug("life_companion audit log failed", exc_info=True)

    return {"status": "ok", "overrides": overrides}


# ── Structured-diagnosis telemetry + state (Q2 §39) ─────────────────────


@router.get("/structured_diagnosis/state")
async def get_structured_diagnosis_state():
    """Return the current threshold band + the auto-tune state +
    a recent-window telemetry summary.

    React ``/cp/settings`` Structured-Diagnosis card consumes this
    for its inline display (current threshold, recent approval
    rate, last adjustment).
    """
    try:
        from app.healing.diagnosis_auto_tune import current_state
        from app.healing.diagnosis_telemetry import summary as telemetry_summary
        return {
            "threshold": current_state(),
            "telemetry_30d": telemetry_summary(window=30),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/structured_diagnosis/telemetry")
async def get_structured_diagnosis_telemetry(window: int = 30):
    """Return paginated telemetry rows (filed events with their
    resolutions joined). Used by the React ``/cp/structured-diagnosis``
    detail page."""
    if window < 1 or window > 200:
        raise HTTPException(status_code=400, detail="window must be in [1, 200]")
    try:
        from app.healing.diagnosis_telemetry import rolling_window_with_resolutions
        return {
            "window": window,
            "rows": rolling_window_with_resolutions(window=window),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
