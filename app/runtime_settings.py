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
    _update({"goodhart_hard_gate_disabled": bool(value)})
    logger.info(
        "runtime_settings: goodhart_hard_gate_disabled set to %s", bool(value),
    )


def get_goodhart_hard_gate_enforcing() -> bool:
    """Advisory→blocking flip. Read by
    ``app.governance._goodhart_hard_gate_enforcing``.
    """
    return bool(_ensure_initialized()["goodhart_hard_gate_enforcing"])


def set_goodhart_hard_gate_enforcing(value: bool) -> None:
    _update({"goodhart_hard_gate_enforcing": bool(value)})
    logger.info(
        "runtime_settings: goodhart_hard_gate_enforcing set to %s", bool(value),
    )


def _update(patch: dict[str, Any]) -> None:
    global _cache
    with _lock:
        state = _cache if _cache is not None else _load()
        state.update(patch)
        _save(state)
        _cache = state
