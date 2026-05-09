"""Shared utilities for healing runbook handlers.

Each handler module registers one or more runbooks against the dispatcher
in ``app/healing/runbooks.py``. This module provides:

  - ``compute_signature(logger, message)`` — mirrors the SHA-1 signature
    rule in ``app.observability.error_monitor._signature`` so handlers can
    register against the exact hash a real anomaly will carry.
  - ``RateLimiter`` — per-handler cooldown to prevent self-DOS when an
    anomaly is firing continuously.
  - ``send_signal_alert(...)`` — bounded best-effort notify to the operator.
  - ``audit_event(...)`` — typed audit helper for handler activity.
  - ``write_state_json(...)`` — atomic JSON write for handler-owned state
    files under ``workspace/self_heal/``.

None of these touch TIER_IMMUTABLE files.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Mirror the strip patterns + normalization rules from
# app/observability/error_monitor.py:_signature so registered hashes line
# up with what the monitor actually emits. Order MATTERS — identical to
# error_monitor's list (UUIDs first, then timestamps, then numbers, etc.).
_STRIP_PATTERNS = [
    (re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.I,
    ), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\S*\b"), "<ts>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    (re.compile(r"'[^']{0,200}'"), "'<str>'"),
    (re.compile(r'"[^"]{0,200}"'), '"<str>"'),
    (re.compile(r"/[\w./-]+\.(?:py|md|json|yaml|sql)\b"), "<path>"),
]


def compute_signature(logger_name: str, message: str) -> str:
    """Mirror of ``error_monitor._signature``: SHA-1[:16] over normalized
    ``"{module}::{normalized_message[:120]}"``.

    Used at handler-registration time to compute the hash that will land
    in ``anomaly['pattern_signature']`` for the canonical message.
    """
    module = (logger_name or "unknown").lower()
    msg = (message or "").strip()  # error_monitor strips before truncating
    norm = msg[:200]
    for pat, repl in _STRIP_PATTERNS:
        norm = pat.sub(repl, norm)
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    raw = f"{module}::{norm[:120]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ── Rate limiter (per-handler cooldown) ────────────────────────────────────

class RateLimiter:
    """Token-bucket-ish single-allowance rate limiter.

    ``allow()`` returns True at most once per ``cooldown_seconds``. Used by
    handlers to avoid storming an action on a sustained spike.
    """

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


# ── State-file writer ──────────────────────────────────────────────────────

_STATE_DIR = Path(__file__).resolve().parents[3] / "workspace" / "self_heal"


def state_path(name: str) -> Path:
    """Return a path under ``workspace/self_heal/`` for handler state."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR / name


def write_state_json(name: str, payload: dict[str, Any]) -> None:
    """Atomic JSON write to ``workspace/self_heal/<name>``.

    Best-effort: any failure logs at debug and returns without raising —
    state files are advisory, never on the hot path.
    """
    try:
        path = state_path(name)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)
    except Exception:
        logger.debug("healing.handlers: state-write failed for %s", name, exc_info=True)


def read_state_json(name: str, default: Optional[dict] = None) -> dict[str, Any]:
    try:
        path = state_path(name)
        if not path.exists():
            return dict(default or {})
        return json.loads(path.read_text())
    except Exception:
        return dict(default or {})


# ── Signal alert helper ────────────────────────────────────────────────────


def send_signal_alert(text: str, *, tag: str = "self_heal") -> bool:
    """Best-effort Signal notify to the operator. Returns True on dispatch.

    Never raises. Skips silently if the Signal client isn't configured —
    the runbook framework never depends on Signal being live.
    """
    try:
        from app.signal_client import send_message
        from app.config import get_settings
        recipient = (get_settings().signal_owner_number or "").strip()
        if not recipient:
            logger.debug("healing.handlers[%s]: no signal_owner_number configured", tag)
            return False
        send_message(recipient, text)
        return True
    except Exception:
        logger.debug("healing.handlers[%s]: signal send failed", tag, exc_info=True)
        return False


# ── Audit helper ───────────────────────────────────────────────────────────


def audit_event(action: str, **detail: Any) -> None:
    """Best-effort hash-chained audit. Actor is fixed to the healing surface
    so the operator can ``/api/cp/audit?actor=self_heal_handler`` to filter.
    """
    try:
        from app.control_plane.audit import get_audit
        get_audit().log(actor="self_heal_handler", action=action, detail=detail)
    except Exception:
        logger.debug("healing.handlers: audit write failed", exc_info=True)


# ── Change-request helper ──────────────────────────────────────────────────


def file_change_request(
    *,
    path: str,
    new_content: str,
    old_content: str,
    reason: str,
    requestor: str = "self_heal_handler",
    send_signal: bool = True,
) -> Optional[str]:
    """Create a change-request and (optionally) send the Signal ASK.

    Returns the request_id on success, None on failure or TIER_IMMUTABLE
    refusal. Never raises — operators can always opt out by leaving the
    runbook disabled in ``runbook_settings.json``.
    """
    try:
        from app.change_requests import create_request, send_ask, Status
    except Exception:
        logger.debug("healing.handlers: change_requests import failed", exc_info=True)
        return None

    try:
        cr = create_request(
            requestor=requestor,
            path=path,
            new_content=new_content,
            old_content=old_content,
            reason=reason,
        )
    except Exception:
        logger.debug("healing.handlers: create_request failed", exc_info=True)
        return None

    if cr.status != Status.PENDING:
        # Either TIER_IMMUTABLE_REFUSED or validation REJECTED — nothing more to do.
        logger.info(
            "healing.handlers: change_request not pending — status=%s path=%s",
            cr.status, path,
        )
        return None

    if send_signal:
        try:
            send_ask(cr.id)
        except Exception:
            logger.debug("healing.handlers: send_ask failed", exc_info=True)

    return cr.id


# ── Sample-substring guard ────────────────────────────────────────────────


def sample_contains(anomaly: dict[str, Any], *needles: str) -> bool:
    """Defensive check: confirm ``pattern_sample`` contains expected text.

    The dispatcher already matched on ``pattern_signature`` (SHA-1), but
    if the canonical message subtly drifts the hash will break and we
    don't want a stale handler firing on a different error that happened
    to collide. So handlers double-check the sample text.
    """
    sample = (anomaly.get("pattern_sample") or anomaly.get("sample") or "").lower()
    if not sample:
        return False
    return all(needle.lower() in sample for needle in needles)
