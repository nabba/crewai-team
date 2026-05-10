"""Shared utilities for the life-companion subsystem.

This package adds three proactive features on top of the existing
agent-tools surface:

  * ``email_monitor``     — triages unread inbox every ~10 min.
  * ``daily_briefing``    — synthesises a morning / evening / weekly digest.
  * ``routine_detector``  — surfaces day-of-week + time-of-day patterns.

They share these helpers:

  * ``RateLimiter``        — per-feature cooldown.
  * ``state_path / read_state_json / write_state_json`` —
    atomic JSON state under ``workspace/life_companion/``.
  * ``send_signal_alert``  — bounded best-effort Signal notify.
  * ``audit_event``        — typed audit with ``actor='life_companion'``.
  * ``user_email_address`` — best-effort owner-email lookup.
  * ``feature_enabled``    — env-flag check honoring the master switch.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Master + per-feature switches ─────────────────────────────────────────


def _master_enabled() -> bool:
    return os.getenv("LIFE_COMPANION_ENABLED", "true").lower() in ("true", "1", "yes")


def feature_enabled(feature: str) -> bool:
    """``feature`` is one of ``email`` / ``briefing`` / ``routines`` etc.

    Resolution order:

      1. Master switch ``LIFE_COMPANION_ENABLED`` — if false, every
         feature is off regardless of per-feature flags.
      2. Runtime override (set via the React /cp/life-companion
         control panel).  Persisted in
         ``workspace/runtime_settings.json`` so toggles survive
         restarts.
      3. Env-var fallback ``LIFE_COMPANION_<FEATURE>_ENABLED``
         (boot default).
    """
    if not _master_enabled():
        return False
    # Runtime override wins over env so operators can flip features
    # via the React control panel without a gateway restart.
    try:
        from app.runtime_settings import life_companion_get_feature_enabled
        override = life_companion_get_feature_enabled(feature)
        if override is not None:
            return override
    except Exception:
        pass
    var = f"LIFE_COMPANION_{feature.upper()}_ENABLED"
    return os.getenv(var, "true").lower() in ("true", "1", "yes")


def get_tunable(env_key: str, default: str = "") -> str:
    """Read a tunable env var with runtime-override priority.

    Resolution order:

      1. Runtime override (from the React /cp/life-companion
         control panel) — persisted in
         ``workspace/runtime_settings.json``.
      2. Env-var fallback (the boot default).
      3. ``default`` argument (only used when the env var is also
         unset).

    Always returns a string, mirroring ``os.getenv`` semantics.
    Callers do their own type coercion — same shape they had when
    they read straight from the env.
    """
    try:
        from app.runtime_settings import life_companion_get_tunable
        override = life_companion_get_tunable(env_key)
        if override is not None:
            return override
    except Exception:
        pass
    return os.getenv(env_key, default)


# ── State files ───────────────────────────────────────────────────────────


_STATE_DIR = Path(__file__).resolve().parents[2] / "workspace" / "life_companion"


def state_path(name: str) -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR / name


def write_state_json(name: str, payload: dict[str, Any]) -> None:
    """Atomic JSON write to ``workspace/life_companion/<name>``. Best effort."""
    try:
        path = state_path(name)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        tmp.replace(path)
    except Exception:
        logger.debug("life_companion: state-write failed for %s", name, exc_info=True)


def read_state_json(name: str, default: Optional[dict] = None) -> dict[str, Any]:
    try:
        path = state_path(name)
        if not path.exists():
            return dict(default or {})
        return json.loads(path.read_text())
    except Exception:
        return dict(default or {})


# ── Rate limiter (single-allowance per cooldown_seconds) ──────────────────


class RateLimiter:
    def __init__(self, cooldown_seconds: float):
        self.cooldown = float(cooldown_seconds)
        self._last_fire: float = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if now - self._last_fire < self.cooldown:
                return False
            self._last_fire = now
            return True

    def reset(self) -> None:
        with self._lock:
            self._last_fire = 0.0


# ── Signal alert helper ───────────────────────────────────────────────────


def send_signal_alert(text: str, *, tag: str = "life_companion") -> bool:
    """Best-effort notify. Skips silently when Signal client isn't set up."""
    try:
        from app.signal_client import send_message
        from app.config import get_settings
        recipient = (get_settings().signal_owner_number or "").strip()
        if not recipient:
            logger.debug("life_companion[%s]: no signal_owner_number configured", tag)
            return False
        send_message(recipient, text)
        return True
    except Exception:
        logger.debug("life_companion[%s]: signal send failed", tag, exc_info=True)
        return False


# ── Audit helper ──────────────────────────────────────────────────────────


def audit_event(action: str, **detail: Any) -> None:
    """Hash-chained audit with ``actor='life_companion'`` so the operator can
    filter via ``/api/cp/audit?actor=life_companion``.
    """
    try:
        from app.control_plane.audit import get_audit
        get_audit().log(actor="life_companion", action=action, detail=detail)
    except Exception:
        logger.debug("life_companion: audit write failed", exc_info=True)


# ── User email lookup ─────────────────────────────────────────────────────

_DEFAULT_USER_EMAIL = "andrus.raudsalu@plgmoments.com"


def user_email_address() -> str:
    """Return the operator's email — used by the email-importance scorer
    to detect direct-to-user vs cc-only messages.

    Resolution order:
      1. ``USER_EMAIL`` env var
      2. ``settings.user_email`` if present
      3. Fallback constant (current owner of this gateway)
    """
    env_val = os.getenv("USER_EMAIL", "").strip()
    if env_val:
        return env_val
    try:
        from app.config import get_settings
        s = get_settings()
        for attr in ("user_email", "owner_email", "operator_email"):
            v = getattr(s, attr, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass
    return _DEFAULT_USER_EMAIL


# ── Idle / kill-switch awareness ──────────────────────────────────────────


def background_enabled() -> bool:
    """Honor the global idle-scheduler kill switch when available.

    The dashboard toggle in Firestore ``config/background_tasks`` is read
    by ``app.idle_scheduler.is_enabled()``. We respect it so operators can
    pause every life-companion pass without restarting the gateway.
    """
    try:
        from app import idle_scheduler
        return bool(idle_scheduler.is_enabled())
    except Exception:
        # If idle_scheduler isn't importable (tests), default to ON.
        return True
