"""
runtime_settings.py — File-backed runtime state for the personal-agent surface.

Holds the toggles the React dashboard can flip without a restart:

    voice_mode                       off | local | cloud
    vision_cu_enabled                bool
    vision_cu_monthly_cap_usd        float
    concierge_persona_enabled        bool
    tier3_amendment_enabled          bool

    # Self-heal subsystem master switches (Wave 4 follow-up, 2026-05-09):
    error_runbooks_enabled           bool
    tool_supervisor_enabled          bool
    recovery_loop_enabled            bool

    # Goodhart hard-gate three-way control (Wave 4 follow-up):
    goodhart_hard_gate_disabled      bool   # emergency disable
    goodhart_hard_gate_enforcing     bool   # advisory→blocking flip

State is initialised from `Settings` defaults on first read, then persisted
to ``workspace/runtime_settings.json`` so toggles survive process restarts.
This is the single read path for any subsystem that needs to know what mode
the user wants — do NOT read these values directly from `get_settings()`,
because that returns the env-default and ignores dashboard updates.

Default-seeding policy: the new healing/governance switches default to the
operator's current ``.env`` value at first read, so flipping the file-backed
runtime_settings in front of an existing env-true setup doesn't silently
turn things off. After the JSON file exists, IT is canonical.

Pattern mirrors `app.creative_mode` and `app.llm_mode`, with the added file
backing because these toggles drive user-facing behaviour and should not
silently revert on container restart.

Thread-safety: protected by a module-level lock around read-modify-write
of the JSON file. Fast path (cached read) is lock-free.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

from app.config import get_settings
from app.paths import WORKSPACE_ROOT


def _env_bool(name: str, default: bool = False) -> bool:
    """Read an env var as a boolean. Used to seed first-time defaults
    so an existing ``.env`` setup is preserved when the runtime_settings
    JSON is created."""
    raw = os.getenv(name, "").strip().lower()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    return default

logger = logging.getLogger(__name__)

VALID_VOICE_MODES = ("off", "local", "cloud")

_STATE_PATH = WORKSPACE_ROOT / "runtime_settings.json"
_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def _defaults() -> dict[str, Any]:
    s = get_settings()
    # Settings may not declare every key on older deployments or in
    # the v2 test shim — read defensively. Phase E #14 (2026-05-09):
    # made the previously-direct attribute reads use ``getattr`` with
    # explicit defaults so a stripped-down test ``Settings`` (or an
    # older deployment without these fields) doesn't crash on import.
    # The runtime-settings file is the single source of truth once
    # written; env defaults below are first-boot seeds.
    return {
        "voice_mode": getattr(s, "voice_mode", "off"),
        "vision_cu_enabled": bool(getattr(s, "vision_cu_enabled", False)),
        "vision_cu_monthly_cap_usd": float(getattr(s, "vision_cu_monthly_cap_usd", 10.0)),
        "concierge_persona_enabled": bool(getattr(s, "concierge_persona_enabled", False)),
        "tier3_amendment_enabled": bool(getattr(s, "tier3_amendment_enabled", False)),
        # Self-heal subsystem master switches.
        "error_runbooks_enabled": _env_bool("ERROR_RUNBOOKS_ENABLED", False),
        "tool_supervisor_enabled": _env_bool("TOOL_SUPERVISOR_ENABLED", False),
        "recovery_loop_enabled": _env_bool("RECOVERY_LOOP_ENABLED", False),
        # Goodhart hard-gate three-way control.
        "goodhart_hard_gate_disabled": _env_bool("GOODHART_HARD_GATE_DISABLED", False),
        "goodhart_hard_gate_enforcing": _env_bool("GOODHART_HARD_GATE_ENFORCING", False),
        # Life-companion per-feature overrides — populated by the
        # /cp/life-companion control panel.  Schema:
        #   {<feature_key>: {"enabled": bool|None,
        #                    "tunables": {<env_key>: <stringified value>}}}
        # ``enabled = None`` means "fall back to env / default";
        # missing tunable keys also fall back.
        "life_companion_overrides": {},
        # Model capability blocklists — populated by the
        # ``model_capability`` self-heal handlers when a model is
        # observed failing a structural capability check (chat
        # completion, function calling). Subsystems consult these
        # at routing time; an entry here means "do not route this
        # capability to this model." See
        # ``docs/SELF_HEAL_V3.md`` for the auto-action contract.
        "chat_blocked_models": [],
        "no_function_calling_models": [],
        # Structured-diagnosis confidence threshold band (Q2 §39).
        # The auto-tuner adjusts the active threshold within
        # ``[floor, ceiling]`` based on recent approval-rate
        # telemetry. ``override`` is a manual operator pin that
        # bypasses the auto-tuner entirely when set (None means
        # "let auto-tuner manage"). ``auto_tune_enabled=False``
        # also pins the auto-tuner; the difference is override is a
        # specific value, the disabled flag freezes whatever the
        # state file currently holds.
        "structured_diagnosis_threshold_floor": 0.50,
        "structured_diagnosis_threshold_ceiling": 0.95,
        "structured_diagnosis_threshold_override": None,
        "structured_diagnosis_auto_tune_enabled": True,
        # Embedding-migration master switches (PROGRAM §40 Item 12,
        # 2026-05-10). Default OFF — the entire framework is
        # observational until the operator opts in. ``state`` is the
        # state-machine blob; the three ``_enabled`` flags are user-
        # facing toggles surfaced on /cp/settings.
        "embedding_migration_dual_write_enabled": False,
        "embedding_migration_shadow_read_enabled": False,
        "embedding_migration_cutover_enabled": False,
        "embedding_migration_state": {},
        # Person-correlation (PROGRAM §42, 2026-05-11) — four-level
        # opt-in stack. ALL flags default OFF. Enabling L4 + L4.4
        # additionally requires a typed-phrase confirmation flowing
        # through ``app.api.config_api`` (the runtime_settings setter
        # itself does not enforce; the API endpoint does).
        #
        # L1 — Presence (counts only)
        "person_correlation_enabled": False,
        "person_correlation_decay_months": 12,
        # L2 — Centrality scores
        "person_centrality_enabled": False,
        "person_centrality_formula": "frequency",   # frequency | recency_weighted | cross_modal
        # L3 — Suggestions
        "person_suggestions_enabled": False,
        "person_suggestions_dormancy_enabled": False,
        "person_suggestions_responsiveness_enabled": False,
        # L4 — Social graph (requires typed-phrase "ENABLE SOCIAL GRAPH")
        "person_correlation_social_graph_enabled": False,
        # L4 sub-features
        "graph_shortest_path_enabled": False,
        "graph_communities_enabled": False,
        "graph_bridges_enabled": False,
        # L4.4 — Graph-driven suggestions (requires SECOND typed-phrase
        # "ENABLE GRAPH-DRIVEN SUGGESTIONS")
        "graph_suggestions_enabled": False,
        "graph_suggestions_cluster_dormancy_enabled": False,
        "graph_suggestions_bridge_maintenance_enabled": False,
        "graph_suggestions_weak_tie_enabled": False,

        # Q5 — Targeted sentience experiments (PROGRAM §43, 2026-05-13).
        # Each module reifies a functional approximation of a capability
        # the Butlin scorecard declares architecturally ABSENT. None of
        # these flip the scorecard — the evaluators check canonical
        # paths in ``app/subia/*``. These modules live in
        # ``app/sentience_experiments/`` and are observational only.
        #
        # Two modules default OFF for blast-radius reasons:
        #   * HOT-4 hooks live LoadableAgent telemetry; async-only,
        #     but worth keeping behind an explicit toggle until the
        #     latency-budget assertion has run in production.
        #
        # User explicitly approved these defaults during the Q5 plan.
        "sentience_ae2_enabled": True,
        "sentience_hot1_enabled": True,
        "sentience_hot4_enabled": True,
        "sentience_rpt1_enabled": True,
        # Philosophy decision panel (PROGRAM §43.1 — Q5.1) — pre-decision
        # multi-tradition consult surface for Tier-3 amendments,
        # identity-claim ratification, and welfare-bound calibration.
        # ON by default (cache-bounded; very low cost).
        "philosophy_panel_enabled": True,
        # Ledger-as-governor (PROGRAM §43.1 — Q5.1) — file-kind history
        # in addition to the existing per-path history.
        "ledger_governor_enabled": True,
        # LLM-prose gate for sentience modules. When OFF the modules
        # emit structured observations only — no inferred-affect
        # prose. ON enables hypothesis generation (passed through the
        # decentering filter regardless).
        "sentience_llm_hypothesis_enabled": True,

        # Q6 — Resilience drills (PROGRAM §44, 2026-05-13).
        # Quarterly exercises that verify recovery procedures work.
        # Master + per-drill gates. kill_the_gateway is OFF by default
        # because it is the only DISRUPTIVE drill (actually stops the
        # gateway container). Operator opts in via the React /cp/settings
        # toggle when ready to schedule a maintenance window.
        "resilience_drills_enabled": True,
        "drill_backup_restore_enabled": True,
        "drill_embedding_migration_enabled": True,
        "drill_secret_rotation_enabled": True,
        "drill_kill_the_gateway_enabled": False,  # OPT-IN
        "drill_staleness_monitor_enabled": True,
        # Q6.5 P2#3 (PROGRAM §44.5) — daily probe of
        # workspace/backups/dr/ mtime. Catches the "operator's backup-
        # sync cron died" failure mode without needing cloud SDKs.
        "backup_freshness_monitor_enabled": True,

        # Q7.1 — Architecture-request primitive (PROGRAM §45.1).
        # Top-level subsystem switch + per-feature adoption monitor.
        # Both default ON per operator decision.
        "architecture_requests_enabled": True,
        "architecture_adoption_monitor_enabled": True,

        # Q7.4 — Per-coding-session inline ShinkaEvolve (PROGRAM §45.4).
        # Gates ``app.coding_session.evolution_bridge.evolve_in_session``.
        # When OFF, the bridge returns ``status="disabled"`` instead of
        # invoking ShinkaEvolveRunner. The bulk subsystem
        # (``app.shinka_engine``) is gated separately.
        "shinka_inline_evolve_enabled": True,

        # Q9.3 — Travel monitor configuration (PROGRAM §46.6).
        # ``tripit_ical_url`` is the per-user TripIt iCal feed
        # (Settings → Calendar Sync → "Copy to your calendar" in
        # the TripIt account UI). Empty = TripIt source disabled.
        # ``aviationstack_api_key`` is the optional Aviationstack
        # API key for live flight status. Empty = no live status;
        # the TripIt segments themselves still surface.
        # Both fall back to the matching env vars (TRIPIT_ICAL_URL
        # / AVIATIONSTACK_API_KEY) for backward compatibility.
        "tripit_ical_url": "",
        "aviationstack_api_key": "",

        # Post-amendment restart-claim queue (PROGRAM §40.2 Item 1+9,
        # 2026-05-11). When a Tier-3 amendment applies a code change
        # whose effect requires reloading the running interpreter
        # (e.g. ``_EMBED_DIM`` substrate migration; future soul edits
        # whose hooks load at import), the amendment's post-apply
        # path appends a claim here. The gateway's startup self-check
        # consults this list; un-cleared claims surface as a loud
        # banner + Signal alert so the operator knows a restart is
        # the only thing that brings the new behavior live.
        #
        # Schema per claim:
        #   {
        #     "id": "<unique-id>",
        #     "issued_at": ISO-8601 UTC,
        #     "reason": "<short operator-readable>",
        #     "source": "<subsystem>",      # e.g. "embedding_migration.cutover"
        #     "tier3_proposal_id": "<id>",  # optional cross-link
        #     "claim_kind": "<kind>",       # e.g. "restart_required"
        #   }
        #
        # Cleared (popped) by the gateway after a confirmed boot that
        # observed the amendment in effect.
        "post_amendment_restart_claims": [],
    }


def _load() -> dict[str, Any]:
    """Read state from disk, falling back to env defaults for missing keys."""
    state = _defaults()
    if _STATE_PATH.exists():
        try:
            on_disk = json.loads(_STATE_PATH.read_text())
            if isinstance(on_disk, dict):
                # Merge — disk wins for known keys, unknown keys ignored.
                for k in state:
                    if k in on_disk:
                        state[k] = on_disk[k]
        except Exception as exc:
            logger.warning(f"runtime_settings: failed to load {_STATE_PATH}: {exc}")
    return state


def _save(state: dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(_STATE_PATH)


def _ensure_initialized() -> dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is None:
            _cache = _load()
        return _cache


def snapshot() -> dict[str, Any]:
    """Return a plain-dict view of current runtime settings."""
    return dict(_ensure_initialized())


def get_voice_mode() -> str:
    return _ensure_initialized()["voice_mode"]


def set_voice_mode(value: str) -> None:
    v = (value or "").strip().lower()
    if v not in VALID_VOICE_MODES:
        raise ValueError(f"voice_mode must be one of {VALID_VOICE_MODES}, got {value!r}")
    _update({"voice_mode": v})
    logger.info(f"runtime_settings: voice_mode set to {v!r}")


def get_vision_cu_enabled() -> bool:
    return bool(_ensure_initialized()["vision_cu_enabled"])


def set_vision_cu_enabled(value: bool) -> None:
    _update({"vision_cu_enabled": bool(value)})
    logger.info(f"runtime_settings: vision_cu_enabled set to {bool(value)}")


def get_vision_cu_monthly_cap_usd() -> float:
    return float(_ensure_initialized()["vision_cu_monthly_cap_usd"])


def set_vision_cu_monthly_cap_usd(value: float) -> None:
    v = float(value)
    if v < 0.0:
        raise ValueError("vision_cu_monthly_cap_usd must be non-negative")
    if v > 1000.0:
        raise ValueError("vision_cu_monthly_cap_usd exceeds sanity cap of $1000/mo")
    _update({"vision_cu_monthly_cap_usd": v})
    logger.info(f"runtime_settings: vision_cu_monthly_cap_usd set to ${v:.2f}")


def get_concierge_persona_enabled() -> bool:
    return bool(_ensure_initialized()["concierge_persona_enabled"])


def set_concierge_persona_enabled(value: bool) -> None:
    _update({"concierge_persona_enabled": bool(value)})
    logger.info(f"runtime_settings: concierge_persona_enabled set to {bool(value)}")


def get_tier3_amendment_enabled() -> bool:
    """Master switch for the Tier-3 amendment protocol.

    Read by ``app.governance_amendment.protocol.amendment_protocol_enabled``
    so the React dashboard can flip the gate without a gateway restart.
    Default is False — the protocol is opt-in.
    """
    return bool(_ensure_initialized()["tier3_amendment_enabled"])


def set_tier3_amendment_enabled(value: bool) -> None:
    _update({"tier3_amendment_enabled": bool(value)})
    logger.info(
        "runtime_settings: tier3_amendment_enabled set to %s", bool(value),
    )


# ── Self-heal subsystem master switches (2026-05-09) ────────────────────


def get_error_runbooks_enabled() -> bool:
    """Read by ``app.healing.runbooks.runbooks_enabled``."""
    return bool(_ensure_initialized()["error_runbooks_enabled"])


def set_error_runbooks_enabled(value: bool) -> None:
    _update({"error_runbooks_enabled": bool(value)})
    logger.info("runtime_settings: error_runbooks_enabled set to %s", bool(value))


def get_tool_supervisor_enabled() -> bool:
    """Read by ``app.tool_runtime.supervisor.is_enabled``."""
    return bool(_ensure_initialized()["tool_supervisor_enabled"])


def set_tool_supervisor_enabled(value: bool) -> None:
    _update({"tool_supervisor_enabled": bool(value)})
    logger.info("runtime_settings: tool_supervisor_enabled set to %s", bool(value))


def get_recovery_loop_enabled() -> bool:
    """Read by ``app.recovery.loop.is_enabled``."""
    return bool(_ensure_initialized()["recovery_loop_enabled"])


def set_recovery_loop_enabled(value: bool) -> None:
    _update({"recovery_loop_enabled": bool(value)})
    logger.info("runtime_settings: recovery_loop_enabled set to %s", bool(value))


# ── Goodhart hard-gate (2026-05-09) ─────────────────────────────────────


def get_goodhart_hard_gate_disabled() -> bool:
    """Emergency disable. Read by
    ``app.governance._goodhart_hard_gate_disabled``.
    """
    return bool(_ensure_initialized()["goodhart_hard_gate_disabled"])


def set_goodhart_hard_gate_disabled(value: bool) -> None:
    prior = get_goodhart_hard_gate_disabled()
    _update({"goodhart_hard_gate_disabled": bool(value)})
    logger.info(
        "runtime_settings: goodhart_hard_gate_disabled set to %s", bool(value),
    )
    if bool(prior) != bool(value):
        _emit_goodhart_governance_event(
            setting="goodhart_hard_gate_disabled",
            prior=bool(prior), new=bool(value),
        )


def get_goodhart_hard_gate_enforcing() -> bool:
    """Advisory→blocking flip. Read by
    ``app.governance._goodhart_hard_gate_enforcing``.
    """
    return bool(_ensure_initialized()["goodhart_hard_gate_enforcing"])


def set_goodhart_hard_gate_enforcing(value: bool) -> None:
    prior = get_goodhart_hard_gate_enforcing()
    _update({"goodhart_hard_gate_enforcing": bool(value)})
    logger.info(
        "runtime_settings: goodhart_hard_gate_enforcing set to %s", bool(value),
    )
    if bool(prior) != bool(value):
        _emit_goodhart_governance_event(
            setting="goodhart_hard_gate_enforcing",
            prior=bool(prior), new=bool(value),
        )


def _emit_goodhart_governance_event(
    *, setting: str, prior: bool, new: bool,
) -> None:
    """Record a Goodhart-gate mode flip as an identity-shaping
    governance event. Mirrors the existing ``governance_ratchet``
    emission pattern in ``app/governance_ratchet/protocol.py``;
    Goodhart enforcement changes are the same caliber of event.

    Best-effort: ledger / GW failures degrade silently — the setting
    is already persisted by ``_update``.
    """
    summary = (
        f"Goodhart hard gate {setting.replace('goodhart_hard_gate_', '')} "
        f"flipped {prior} → {new}"
    )
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="governance_ratchet",
            actor="operator",
            summary=summary,
            detail={
                "setting": setting,
                "prior": prior,
                "new": new,
                # Effective mode after the flip — useful for the
                # annual reflection drift summary.
                "effective_mode": _goodhart_effective_mode_label(),
            },
        )
    except Exception:
        logger.debug(
            "runtime_settings: continuity_ledger emission failed for %s",
            setting, exc_info=True,
        )
    try:
        from app.workspace_publish import publish_to_workspace
        publish_to_workspace(
            source="runtime-settings",
            content=summary,
            salience=0.6,  # governance changes are operator-relevant
            signal_type="disposition",
        )
    except Exception:
        logger.debug(
            "runtime_settings: GW publish failed for %s", setting,
            exc_info=True,
        )


def _goodhart_effective_mode_label() -> str:
    """Resolve the three-mode label from current state. Mirrors the
    ``app.governance._evaluate_goodhart_gate`` discrimination."""
    try:
        if get_goodhart_hard_gate_disabled():
            return "disabled"
        if get_goodhart_hard_gate_enforcing():
            return "enforcing"
        return "advisory"
    except Exception:
        return "unknown"


def _update(patch: dict[str, Any]) -> None:
    global _cache
    with _lock:
        state = _cache if _cache is not None else _load()
        state.update(patch)
        _save(state)
        _cache = state


# ── Life-companion per-feature overrides ────────────────────────────


def life_companion_get_overrides() -> dict[str, Any]:
    """Read-only snapshot of all life-companion feature overrides.

    Schema: ``{<feature_key>: {"enabled": bool|None, "tunables":
    {<env_key>: <str_value>}}}``.  Empty dict on first boot.
    Mutations go through :func:`life_companion_set_feature_override`.
    """
    return dict(_ensure_initialized().get("life_companion_overrides") or {})


def life_companion_get_feature_enabled(feature_key: str) -> bool | None:
    """Return the override-controlled enabled state for a feature, or
    None if no override is set (caller falls back to env default).

    Splits cleanly so ``feature_enabled()`` in life_companion._common
    can do: ``override or env-default``.
    """
    override = life_companion_get_overrides().get(feature_key)
    if not isinstance(override, dict):
        return None
    if "enabled" not in override:
        return None
    val = override["enabled"]
    if val is None:
        return None
    return bool(val)


def life_companion_get_tunable(env_key: str) -> str | None:
    """Return the override-controlled tunable value, or None when no
    override is set.

    Returned as a string so the caller can apply its own type
    coercion (mirrors os.getenv semantics).  This intentionally
    matches what the registry's UI sends — the React control
    panel emits everything as a string.
    """
    overrides = life_companion_get_overrides()
    for feat_key, entry in overrides.items():
        if not isinstance(entry, dict):
            continue
        tuns = entry.get("tunables") or {}
        if env_key in tuns and tuns[env_key] is not None:
            return str(tuns[env_key])
    return None


# Sentinel for "don't touch this kwarg" — distinguishes from
# ``None`` which explicitly clears the toggle override.
_LEAVE_UNTOUCHED = object()


def life_companion_set_feature_override(
    feature_key: str,
    *,
    enabled: bool | None | object = _LEAVE_UNTOUCHED,
    tunables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist an override for one life-companion feature.

    Three distinct paths for ``enabled``:

      * Omitted (default ``_LEAVE_UNTOUCHED``) — leave the toggle
        override exactly as it was; useful when the operator only
        edited tunables.
      * ``None`` — clear the toggle override; the feature reverts
        to its env-var default.
      * ``True`` / ``False`` — set the override explicitly.

    ``tunables`` is merged into the existing tunable dict — pass
    an empty value (``{"<key>": ""}``) to clear a single tunable
    override and let env defaults take back over.

    Returns the new overrides snapshot.
    """
    if not isinstance(feature_key, str) or not feature_key:
        raise ValueError("feature_key must be a non-empty string")

    global _cache
    with _lock:
        state = _cache if _cache is not None else _load()
        all_overrides = dict(state.get("life_companion_overrides") or {})
        entry = dict(all_overrides.get(feature_key) or {})
        existing_tunables = dict(entry.get("tunables") or {})

        if enabled is _LEAVE_UNTOUCHED:
            pass  # leave untouched
        elif enabled is None:
            entry.pop("enabled", None)  # clear override
        else:
            entry["enabled"] = bool(enabled)

        if tunables is not None:
            for k, v in tunables.items():
                if v in (None, ""):
                    existing_tunables.pop(k, None)
                else:
                    existing_tunables[k] = str(v)
            entry["tunables"] = existing_tunables

        # If the entry is now empty (no enabled override AND no
        # tunables), drop it so the JSON stays tidy.
        if (
            "enabled" not in entry
            and not (entry.get("tunables") or {})
        ):
            all_overrides.pop(feature_key, None)
        else:
            all_overrides[feature_key] = entry

        state["life_companion_overrides"] = all_overrides
        _save(state)
        _cache = state

    logger.info(
        "runtime_settings: life_companion override %s — enabled=%s tunables=%s",
        feature_key,
        "leave" if enabled is _LEAVE_UNTOUCHED else enabled,
        list(tunables.keys()) if tunables else [],
    )
    return all_overrides


# ── Model capability blocklists (Q2 self-heal auto-action) ──────────────


def get_chat_blocked_models() -> list[str]:
    """Models the LLM router should NOT consider for chat tasks.

    Populated by ``app.healing.handlers.model_capability`` when an
    embed-only model is observed being routed to chat. The selector
    consults this list at default-tier selection (see
    ``app.llm_selector.select_model``).
    """
    raw = _ensure_initialized().get("chat_blocked_models") or []
    return list(raw) if isinstance(raw, list) else []


def add_chat_blocked_model(model_name: str) -> bool:
    """Append ``model_name`` to the chat blocklist. Idempotent —
    returns ``True`` on first add, ``False`` if already present.
    Empty / non-string input is a no-op.
    """
    name = (model_name or "").strip()
    if not name:
        return False
    state = _ensure_initialized()
    current = list(state.get("chat_blocked_models") or [])
    if name in current:
        return False
    current.append(name)
    _update({"chat_blocked_models": current})
    logger.info(
        "runtime_settings: chat_blocked_models +%r (size=%d)",
        name, len(current),
    )
    return True


def remove_chat_blocked_model(model_name: str) -> bool:
    """Remove ``model_name`` from the blocklist. Returns True if the
    entry existed and was removed."""
    name = (model_name or "").strip()
    if not name:
        return False
    state = _ensure_initialized()
    current = list(state.get("chat_blocked_models") or [])
    if name not in current:
        return False
    current.remove(name)
    _update({"chat_blocked_models": current})
    logger.info(
        "runtime_settings: chat_blocked_models -%r (size=%d)",
        name, len(current),
    )
    return True


def get_no_function_calling_models() -> list[str]:
    """Models known to NOT support OpenAI-style function calling.

    Populated by ``app.healing.handlers.model_capability`` when Mem0
    LLM extraction (or any tool-using path) hits a model that doesn't
    accept ``tool_choice``. Consumer subsystems consult this to fall
    back to unstructured extraction.
    """
    raw = _ensure_initialized().get("no_function_calling_models") or []
    return list(raw) if isinstance(raw, list) else []


def add_no_function_calling_model(model_name: str) -> bool:
    """Idempotent append. Same shape as ``add_chat_blocked_model``."""
    name = (model_name or "").strip()
    if not name:
        return False
    state = _ensure_initialized()
    current = list(state.get("no_function_calling_models") or [])
    if name in current:
        return False
    current.append(name)
    _update({"no_function_calling_models": current})
    logger.info(
        "runtime_settings: no_function_calling_models +%r (size=%d)",
        name, len(current),
    )
    return True


def remove_no_function_calling_model(model_name: str) -> bool:
    name = (model_name or "").strip()
    if not name:
        return False
    state = _ensure_initialized()
    current = list(state.get("no_function_calling_models") or [])
    if name not in current:
        return False
    current.remove(name)
    _update({"no_function_calling_models": current})
    logger.info(
        "runtime_settings: no_function_calling_models -%r (size=%d)",
        name, len(current),
    )
    return True


# ── Structured-diagnosis threshold band (Q2 §39) ────────────────────────


def get_structured_diagnosis_threshold_floor() -> float:
    return float(_ensure_initialized().get(
        "structured_diagnosis_threshold_floor", 0.50,
    ))


def set_structured_diagnosis_threshold_floor(value: float) -> None:
    v = float(value)
    if not (0.0 <= v <= 0.99):
        raise ValueError(
            f"structured_diagnosis_threshold_floor must be in [0.0, 0.99], got {value!r}"
        )
    ceiling = get_structured_diagnosis_threshold_ceiling()
    if v >= ceiling:
        raise ValueError(
            f"floor {v} must be < ceiling {ceiling}; "
            f"adjust ceiling first OR pick a lower floor"
        )
    _update({"structured_diagnosis_threshold_floor": v})
    logger.info(
        "runtime_settings: structured_diagnosis_threshold_floor set to %.2f", v,
    )


def get_structured_diagnosis_threshold_ceiling() -> float:
    return float(_ensure_initialized().get(
        "structured_diagnosis_threshold_ceiling", 0.95,
    ))


def set_structured_diagnosis_threshold_ceiling(value: float) -> None:
    v = float(value)
    if not (0.01 <= v <= 1.0):
        raise ValueError(
            f"structured_diagnosis_threshold_ceiling must be in [0.01, 1.0], got {value!r}"
        )
    floor = get_structured_diagnosis_threshold_floor()
    if v <= floor:
        raise ValueError(
            f"ceiling {v} must be > floor {floor}; "
            f"adjust floor first OR pick a higher ceiling"
        )
    _update({"structured_diagnosis_threshold_ceiling": v})
    logger.info(
        "runtime_settings: structured_diagnosis_threshold_ceiling set to %.2f", v,
    )


def get_structured_diagnosis_threshold_override() -> float | None:
    """Returns None when no override is set (auto-tuner manages the
    threshold). Returns a float in (0, 1] when the operator has
    pinned a specific value."""
    raw = _ensure_initialized().get("structured_diagnosis_threshold_override")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def set_structured_diagnosis_threshold_override(value: float | None) -> None:
    if value is None:
        _update({"structured_diagnosis_threshold_override": None})
        logger.info("runtime_settings: structured_diagnosis_threshold_override CLEARED")
        return
    v = float(value)
    floor = get_structured_diagnosis_threshold_floor()
    ceiling = get_structured_diagnosis_threshold_ceiling()
    if not (floor <= v <= ceiling):
        raise ValueError(
            f"override {v} must be within [floor={floor}, ceiling={ceiling}]; "
            f"adjust the band first OR pick an in-band value"
        )
    _update({"structured_diagnosis_threshold_override": v})
    logger.info(
        "runtime_settings: structured_diagnosis_threshold_override set to %.2f", v,
    )


def get_structured_diagnosis_auto_tune_enabled() -> bool:
    return bool(_ensure_initialized().get(
        "structured_diagnosis_auto_tune_enabled", True,
    ))


def set_structured_diagnosis_auto_tune_enabled(value: bool) -> None:
    _update({"structured_diagnosis_auto_tune_enabled": bool(value)})
    logger.info(
        "runtime_settings: structured_diagnosis_auto_tune_enabled set to %s",
        bool(value),
    )


# ── Embedding-migration master switches (PROGRAM §40 Item 12) ───────────


def get_embedding_migration_dual_write_enabled() -> bool:
    return bool(_ensure_initialized().get(
        "embedding_migration_dual_write_enabled", False,
    ))


def set_embedding_migration_dual_write_enabled(value: bool) -> None:
    _update({"embedding_migration_dual_write_enabled": bool(value)})
    logger.info(
        "runtime_settings: embedding_migration_dual_write_enabled set to %s",
        bool(value),
    )


def get_embedding_migration_shadow_read_enabled() -> bool:
    return bool(_ensure_initialized().get(
        "embedding_migration_shadow_read_enabled", False,
    ))


def set_embedding_migration_shadow_read_enabled(value: bool) -> None:
    _update({"embedding_migration_shadow_read_enabled": bool(value)})
    logger.info(
        "runtime_settings: embedding_migration_shadow_read_enabled set to %s",
        bool(value),
    )


def get_embedding_migration_cutover_enabled() -> bool:
    return bool(_ensure_initialized().get(
        "embedding_migration_cutover_enabled", False,
    ))


def set_embedding_migration_cutover_enabled(value: bool) -> None:
    _update({"embedding_migration_cutover_enabled": bool(value)})
    logger.info(
        "runtime_settings: embedding_migration_cutover_enabled set to %s",
        bool(value),
    )


def get_embedding_migration_state() -> dict[str, Any]:
    """Read the embedding-migration state blob. Returns ``{}`` on
    first boot. Mutated by ``app.memory.embedding_migration.state``."""
    blob = _ensure_initialized().get("embedding_migration_state")
    if not isinstance(blob, dict):
        return {}
    return dict(blob)


def set_embedding_migration_state(value: dict[str, Any]) -> None:
    """Persist the embedding-migration state blob. The state-machine
    module owns the schema; runtime_settings is just the storage."""
    if not isinstance(value, dict):
        raise TypeError("embedding_migration_state must be a dict")
    _update({"embedding_migration_state": dict(value)})


# ── Post-amendment restart claims (PROGRAM §40.2) ────────────────────────


def get_post_amendment_restart_claims() -> list[dict[str, Any]]:
    """Return all outstanding restart claims. List of dicts; see the
    schema in ``_defaults``. Empty list = no pending restart."""
    raw = _ensure_initialized().get("post_amendment_restart_claims")
    if not isinstance(raw, list):
        return []
    return [dict(c) for c in raw if isinstance(c, dict)]


def append_post_amendment_restart_claim(claim: dict[str, Any]) -> None:
    """Append one claim. Idempotent on ``claim["id"]``: a claim with an
    id that already exists is silently dropped. The caller is
    responsible for generating a stable id (e.g. tier3_proposal_id +
    claim_kind)."""
    if not isinstance(claim, dict):
        raise TypeError("claim must be a dict")
    if not claim.get("id"):
        raise ValueError("claim must have a non-empty id")
    with _lock:
        state = _cache if _cache is not None else _load()
        existing = list(state.get("post_amendment_restart_claims") or [])
        ids = {c.get("id") for c in existing if isinstance(c, dict)}
        if claim["id"] in ids:
            return
        existing.append(dict(claim))
        state["post_amendment_restart_claims"] = existing
        _save(state)
        globals().update({"_cache": state})


def clear_post_amendment_restart_claims(
    ids: list[str] | None = None,
) -> int:
    """Drop claims. When ``ids`` is None, clears ALL — the gateway
    calls this after a confirmed boot that satisfied every outstanding
    claim. When ``ids`` is a list, drops only the matching ones (so
    a partial-satisfaction flow can clear what it knows is live).
    Returns the number of claims removed."""
    with _lock:
        state = _cache if _cache is not None else _load()
        existing = list(state.get("post_amendment_restart_claims") or [])
        if ids is None:
            removed = len(existing)
            state["post_amendment_restart_claims"] = []
        else:
            id_set = set(ids)
            keep = [c for c in existing if c.get("id") not in id_set]
            removed = len(existing) - len(keep)
            state["post_amendment_restart_claims"] = keep
        _save(state)
        globals().update({"_cache": state})
        return removed


# ── Person correlation (PROGRAM §42) — 14 getters + setters ───────────


def get_person_correlation_enabled() -> bool:
    return bool(_ensure_initialized().get("person_correlation_enabled", False))


def set_person_correlation_enabled(value: bool) -> None:
    prev = get_person_correlation_enabled()
    _update({"person_correlation_enabled": bool(value)})
    logger.info("runtime_settings: person_correlation_enabled = %s", bool(value))
    # Q4.2.2#1 — identity-shaping policy flip → continuity ledger.
    if prev != bool(value):
        _emit_person_correlation_policy_event(
            level="L1",
            enabled=bool(value),
        )


def get_person_correlation_decay_months() -> int:
    return int(_ensure_initialized().get("person_correlation_decay_months", 12))


def set_person_correlation_decay_months(value: int) -> None:
    v = max(1, min(60, int(value)))
    _update({"person_correlation_decay_months": v})


def get_person_centrality_enabled() -> bool:
    return bool(_ensure_initialized().get("person_centrality_enabled", False))


def set_person_centrality_enabled(value: bool) -> None:
    _update({"person_centrality_enabled": bool(value)})


def get_person_centrality_formula() -> str:
    return str(_ensure_initialized().get("person_centrality_formula", "frequency"))


def set_person_centrality_formula(value: str) -> None:
    if value not in {"frequency", "recency_weighted", "cross_modal"}:
        raise ValueError(f"person_centrality_formula must be one of frequency/recency_weighted/cross_modal, got {value!r}")
    _update({"person_centrality_formula": value})


def get_person_suggestions_enabled() -> bool:
    return bool(_ensure_initialized().get("person_suggestions_enabled", False))


def set_person_suggestions_enabled(value: bool) -> None:
    _update({"person_suggestions_enabled": bool(value)})


def get_person_suggestions_dormancy_enabled() -> bool:
    return bool(_ensure_initialized().get("person_suggestions_dormancy_enabled", False))


def set_person_suggestions_dormancy_enabled(value: bool) -> None:
    _update({"person_suggestions_dormancy_enabled": bool(value)})


def get_person_suggestions_responsiveness_enabled() -> bool:
    return bool(_ensure_initialized().get("person_suggestions_responsiveness_enabled", False))


def set_person_suggestions_responsiveness_enabled(value: bool) -> None:
    _update({"person_suggestions_responsiveness_enabled": bool(value)})


def get_person_correlation_social_graph_enabled() -> bool:
    return bool(_ensure_initialized().get("person_correlation_social_graph_enabled", False))


def set_person_correlation_social_graph_enabled(value: bool) -> None:
    """Master switch for L4. Enabling this from False→True requires
    a typed-phrase confirmation in the API surface — this function
    does NOT enforce that (the config_api endpoint does)."""
    prev = get_person_correlation_social_graph_enabled()
    _update({"person_correlation_social_graph_enabled": bool(value)})
    logger.info(
        "runtime_settings: person_correlation_social_graph_enabled = %s",
        bool(value),
    )
    # Q4.2.2#1 — L4 enablement is the most identity-shaping flip in the
    # stack (this is the typed-phrase one). Log it to the continuity
    # ledger so annual reflection picks it up.
    if prev != bool(value):
        _emit_person_correlation_policy_event(
            level="L4",
            enabled=bool(value),
        )


def get_graph_shortest_path_enabled() -> bool:
    return bool(_ensure_initialized().get("graph_shortest_path_enabled", False))


def set_graph_shortest_path_enabled(value: bool) -> None:
    _update({"graph_shortest_path_enabled": bool(value)})


def get_graph_communities_enabled() -> bool:
    return bool(_ensure_initialized().get("graph_communities_enabled", False))


def set_graph_communities_enabled(value: bool) -> None:
    _update({"graph_communities_enabled": bool(value)})


def get_graph_bridges_enabled() -> bool:
    return bool(_ensure_initialized().get("graph_bridges_enabled", False))


def set_graph_bridges_enabled(value: bool) -> None:
    _update({"graph_bridges_enabled": bool(value)})


def get_graph_suggestions_enabled() -> bool:
    return bool(_ensure_initialized().get("graph_suggestions_enabled", False))


def set_graph_suggestions_enabled(value: bool) -> None:
    """L4.4 master. From False→True requires SECOND typed-phrase
    'ENABLE GRAPH-DRIVEN SUGGESTIONS'. Enforced at config_api layer."""
    prev = get_graph_suggestions_enabled()
    _update({"graph_suggestions_enabled": bool(value)})
    logger.info("runtime_settings: graph_suggestions_enabled = %s", bool(value))
    # Q4.2.2#1 — L4.4 is the second typed-phrase gate; identity-shaping.
    if prev != bool(value):
        _emit_person_correlation_policy_event(
            level="L4.4",
            enabled=bool(value),
        )


def _emit_person_correlation_policy_event(*, level: str, enabled: bool) -> None:
    """Q4.2.2#1 helper — emit a ``person_correlation_policy`` event to
    the identity continuity ledger. Failure-isolated: never raises
    out to the setter."""
    try:
        from app.identity.continuity_ledger import record_event
        direction = "enabled" if enabled else "disabled"
        record_event(
            kind="person_correlation_policy",
            actor="operator",
            summary=f"person-correlation {level} {direction}",
            detail={
                "level": level,
                "enabled": enabled,
            },
        )
    except Exception:
        logger.debug("person_correlation_policy ledger emit failed", exc_info=True)


def get_graph_suggestions_cluster_dormancy_enabled() -> bool:
    return bool(_ensure_initialized().get("graph_suggestions_cluster_dormancy_enabled", False))


def set_graph_suggestions_cluster_dormancy_enabled(value: bool) -> None:
    _update({"graph_suggestions_cluster_dormancy_enabled": bool(value)})


def get_graph_suggestions_bridge_maintenance_enabled() -> bool:
    return bool(_ensure_initialized().get("graph_suggestions_bridge_maintenance_enabled", False))


def set_graph_suggestions_bridge_maintenance_enabled(value: bool) -> None:
    _update({"graph_suggestions_bridge_maintenance_enabled": bool(value)})


def get_graph_suggestions_weak_tie_enabled() -> bool:
    return bool(_ensure_initialized().get("graph_suggestions_weak_tie_enabled", False))


def set_graph_suggestions_weak_tie_enabled(value: bool) -> None:
    _update({"graph_suggestions_weak_tie_enabled": bool(value)})


# ── Q5 — Targeted sentience experiments (PROGRAM §43) ────────────────


def get_sentience_ae2_enabled() -> bool:
    return bool(_ensure_initialized().get("sentience_ae2_enabled", True))


def set_sentience_ae2_enabled(value: bool) -> None:
    _update({"sentience_ae2_enabled": bool(value)})


def get_sentience_hot1_enabled() -> bool:
    return bool(_ensure_initialized().get("sentience_hot1_enabled", True))


def set_sentience_hot1_enabled(value: bool) -> None:
    _update({"sentience_hot1_enabled": bool(value)})


def get_sentience_hot4_enabled() -> bool:
    return bool(_ensure_initialized().get("sentience_hot4_enabled", True))


def set_sentience_hot4_enabled(value: bool) -> None:
    _update({"sentience_hot4_enabled": bool(value)})


def get_sentience_rpt1_enabled() -> bool:
    return bool(_ensure_initialized().get("sentience_rpt1_enabled", True))


def set_sentience_rpt1_enabled(value: bool) -> None:
    _update({"sentience_rpt1_enabled": bool(value)})


def get_philosophy_panel_enabled() -> bool:
    return bool(_ensure_initialized().get("philosophy_panel_enabled", True))


def set_philosophy_panel_enabled(value: bool) -> None:
    _update({"philosophy_panel_enabled": bool(value)})


def get_ledger_governor_enabled() -> bool:
    return bool(_ensure_initialized().get("ledger_governor_enabled", True))


def set_ledger_governor_enabled(value: bool) -> None:
    _update({"ledger_governor_enabled": bool(value)})


def get_sentience_llm_hypothesis_enabled() -> bool:
    return bool(_ensure_initialized().get("sentience_llm_hypothesis_enabled", True))


def set_sentience_llm_hypothesis_enabled(value: bool) -> None:
    _update({"sentience_llm_hypothesis_enabled": bool(value)})


# ── Q6 — Resilience drills (PROGRAM §44) ─────────────────────────────


def get_resilience_drills_enabled() -> bool:
    return bool(_ensure_initialized().get("resilience_drills_enabled", True))


def set_resilience_drills_enabled(value: bool) -> None:
    _update({"resilience_drills_enabled": bool(value)})


def get_drill_backup_restore_enabled() -> bool:
    return bool(_ensure_initialized().get("drill_backup_restore_enabled", True))


def set_drill_backup_restore_enabled(value: bool) -> None:
    _update({"drill_backup_restore_enabled": bool(value)})


def get_drill_embedding_migration_enabled() -> bool:
    return bool(_ensure_initialized().get("drill_embedding_migration_enabled", True))


def set_drill_embedding_migration_enabled(value: bool) -> None:
    _update({"drill_embedding_migration_enabled": bool(value)})


def get_drill_secret_rotation_enabled() -> bool:
    return bool(_ensure_initialized().get("drill_secret_rotation_enabled", True))


def set_drill_secret_rotation_enabled(value: bool) -> None:
    _update({"drill_secret_rotation_enabled": bool(value)})


def get_drill_kill_the_gateway_enabled() -> bool:
    """OFF by default — the only DISRUPTIVE drill. Operator must
    explicitly enable via /cp/settings before scheduler will emit
    'due' notifications. Even when ON, execution requires the
    external script + typed-phrase confirmation."""
    return bool(_ensure_initialized().get("drill_kill_the_gateway_enabled", False))


def set_drill_kill_the_gateway_enabled(value: bool) -> None:
    _update({"drill_kill_the_gateway_enabled": bool(value)})


def get_drill_staleness_monitor_enabled() -> bool:
    return bool(_ensure_initialized().get("drill_staleness_monitor_enabled", True))


def set_drill_staleness_monitor_enabled(value: bool) -> None:
    _update({"drill_staleness_monitor_enabled": bool(value)})


def get_backup_freshness_monitor_enabled() -> bool:
    return bool(_ensure_initialized().get("backup_freshness_monitor_enabled", True))


def set_backup_freshness_monitor_enabled(value: bool) -> None:
    _update({"backup_freshness_monitor_enabled": bool(value)})


# ── Q7.1 — Architecture-request primitive (PROGRAM §45.1) ────────────


def get_architecture_requests_enabled() -> bool:
    return bool(_ensure_initialized().get("architecture_requests_enabled", True))


def set_architecture_requests_enabled(value: bool) -> None:
    _update({"architecture_requests_enabled": bool(value)})


def get_architecture_adoption_monitor_enabled() -> bool:
    return bool(_ensure_initialized().get("architecture_adoption_monitor_enabled", True))


def set_architecture_adoption_monitor_enabled(value: bool) -> None:
    _update({"architecture_adoption_monitor_enabled": bool(value)})


# ── Q7.4 — Inline ShinkaEvolve per coding session (PROGRAM §45.4) ────


def get_shinka_inline_evolve_enabled() -> bool:
    return bool(_ensure_initialized().get("shinka_inline_evolve_enabled", True))


def set_shinka_inline_evolve_enabled(value: bool) -> None:
    _update({"shinka_inline_evolve_enabled": bool(value)})


# ── Q9.3 — Travel monitor (PROGRAM §46.6) ─────────────────────────────


def get_tripit_ical_url() -> str:
    """Operator-supplied TripIt iCal feed URL. Returns empty when
    not configured; the travel module degrades to env-var fallback
    then to silent no-op."""
    return str(_ensure_initialized().get("tripit_ical_url", "") or "")


def set_tripit_ical_url(value: str) -> None:
    """Persist TripIt iCal URL. Operator-set value; sane validation:
    must be empty OR start with ``https://`` and contain ``tripit``
    in the hostname (defensive — refuse paste of random URLs)."""
    v = (value or "").strip()
    if v:
        lower = v.lower()
        if not lower.startswith("https://"):
            raise ValueError("tripit_ical_url must start with https://")
        # Conservative hostname check — TripIt iCal feeds live under
        # *.tripit.com. Operators copying from the right place will
        # always have "tripit" in the URL.
        if "tripit" not in lower.split("/")[2]:
            raise ValueError(
                "tripit_ical_url hostname must contain 'tripit'"
            )
    _update({"tripit_ical_url": v})


def get_aviationstack_api_key() -> str:
    """Aviationstack API key for live flight status. Empty when not
    configured."""
    return str(_ensure_initialized().get("aviationstack_api_key", "") or "")


def set_aviationstack_api_key(value: str) -> None:
    """Persist Aviationstack API key. Defensive validation: must be
    empty OR a hex-ish 32-char token (Aviationstack format)."""
    v = (value or "").strip()
    if v and len(v) < 16:
        raise ValueError(
            "aviationstack_api_key looks too short to be valid"
        )
    _update({"aviationstack_api_key": v})
