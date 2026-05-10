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
