"""Auto-revert watcher for AUTO_APPLY change requests.

When an auto-apply CR lands, the originating error pattern is
registered here. The watcher monitors error_monitor for recurrences
of that pattern; if the pattern fires within ``_WATCH_WINDOW_S``
after apply, the CR is automatically rolled back.

This is the safety guarantee that lets us bypass the operator gate
for narrow patterns — if the patch makes the situation worse, we
notice and undo it before the operator wakes up.

Mechanics:

  * ``register(...)`` — called by ``lifecycle.auto_approve`` after
    a successful apply. Persists ``(cr_id, signature, applied_at)``
    to ``workspace/change_requests/auto_revert_watch.json``.
  * Daemon thread polls every ``_POLL_INTERVAL_S`` seconds. For
    each watched entry:
      - If we're past the watch window → unregister (success).
      - Else: query error_monitor for the pattern's count since
        applied_at. If the count exceeds ``_REVERT_THRESHOLD``,
        trigger ``rollback_change`` and unregister.

Failure modes:
  * Watch state corruption → re-read from disk on each pass; never
    crash the daemon thread.
  * error_monitor unavailable → fail OPEN (don't auto-revert on our
    own bug; the operator gate is the fallback).
  * rollback_change fails → log + Signal alert + leave entry in
    place so the operator can manually intervene.

Master switch: ``CHANGE_REQUESTS_AUTO_REVERT_ENABLED`` (default
``true``). Disabling deactivates the watcher; existing registrations
remain on disk and get cleaned up next start.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────


_DAEMON_THREAD_NAME = "change-requests-auto-revert"
_WARMUP_S = 60
_POLL_INTERVAL_S = 60   # cheap polling — error_monitor read is in-memory

# Window during which a recurrence of the origin pattern triggers
# rollback. After this, the auto-apply is considered successful and
# the entry is unregistered.
_WATCH_WINDOW_S = 30 * 60   # 30 minutes

# Threshold: how many recurrences of the origin signature within the
# window triggers rollback. Set to 1 — even a single recurrence
# means the patch didn't help (or made things worse).
_REVERT_THRESHOLD = 1

_STATE_FILENAME = "auto_revert_watch.json"


# ── State ────────────────────────────────────────────────────────────


def _state_path() -> Path:
    """Where the watch registry lives. Co-located with the rest of
    the change-requests workspace dir."""
    base = Path(os.environ.get("CHANGE_REQUESTS_DIR")
                or "/app/workspace/change_requests")
    return base / _STATE_FILENAME


@dataclass
class _Watch:
    cr_id: str
    origin_pattern_signature: str
    applied_at_iso: str
    # The error_monitor count at register time. Recurrences are
    # measured AS AN INCREASE relative to this baseline so we don't
    # spuriously trigger on the original-event count itself.
    baseline_count: int = 0


def _load_watches() -> list[_Watch]:
    p = _state_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("auto_revert: state read failed", exc_info=True)
        return []
    if not isinstance(raw, list):
        return []
    out: list[_Watch] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(_Watch(
                cr_id=str(entry["cr_id"]),
                origin_pattern_signature=str(entry["origin_pattern_signature"]),
                applied_at_iso=str(entry["applied_at_iso"]),
                baseline_count=int(entry.get("baseline_count", 0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _save_watches(watches: list[_Watch]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps([asdict(w) for w in watches], indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)
    except OSError:
        logger.debug("auto_revert: state write failed", exc_info=True)


# ── Public API ───────────────────────────────────────────────────────


def register(
    *,
    cr_id: str,
    origin_pattern_signature: str,
    applied_at_iso: str,
) -> None:
    """Register an auto-applied CR for the watch loop.

    Idempotent on cr_id — re-registering replaces the existing entry.
    Empty ``origin_pattern_signature`` is accepted but never triggers
    rollback (no pattern to monitor); the entry still gets cleaned up
    when the watch window elapses.
    """
    if not cr_id:
        return
    baseline = _signature_count(origin_pattern_signature)
    watches = [w for w in _load_watches() if w.cr_id != cr_id]
    watches.append(_Watch(
        cr_id=cr_id,
        origin_pattern_signature=origin_pattern_signature,
        applied_at_iso=applied_at_iso,
        baseline_count=baseline,
    ))
    _save_watches(watches)
    logger.info(
        "auto_revert: registered %s (pattern=%s, baseline=%d)",
        cr_id, origin_pattern_signature, baseline,
    )


def unregister(cr_id: str) -> bool:
    """Remove a watch entry. Returns True if removed."""
    watches = _load_watches()
    new_watches = [w for w in watches if w.cr_id != cr_id]
    if len(new_watches) == len(watches):
        return False
    _save_watches(new_watches)
    return True


def list_active_watches() -> list[dict]:
    """Read-only view of the current watch registry — for tests +
    operator dashboards."""
    return [asdict(w) for w in _load_watches()]


# ── Pattern recurrence ───────────────────────────────────────────────


def _signature_count(signature: str) -> int:
    """Current 24-hour count of ``signature`` in error_monitor.
    Returns 0 on miss (uninitialised monitor or unknown signature).
    """
    if not signature:
        return 0
    try:
        from app.observability.error_monitor import _hourly_rate  # type: ignore
        return int(_hourly_rate(signature, hours=24))
    except Exception:
        return 0


def _enabled() -> bool:
    return os.getenv("CHANGE_REQUESTS_AUTO_REVERT_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── Pass logic ───────────────────────────────────────────────────────


def run_one_pass() -> dict[str, int]:
    """One watcher pass. Returns counters."""
    counters = {
        "watched": 0,
        "expired": 0,    # window elapsed without recurrence (success)
        "reverted": 0,
        "revert_failed": 0,
    }
    if not _enabled():
        return counters

    watches = _load_watches()
    if not watches:
        return counters

    now = datetime.now(timezone.utc)
    keep: list[_Watch] = []

    for watch in watches:
        counters["watched"] += 1
        applied_at = _parse_iso(watch.applied_at_iso)
        if applied_at is None:
            # Malformed entry — drop it.
            continue
        elapsed = (now - applied_at).total_seconds()

        if elapsed > _WATCH_WINDOW_S:
            counters["expired"] += 1
            logger.info(
                "auto_revert: watch expired for %s (no recurrence in %ds)",
                watch.cr_id, _WATCH_WINDOW_S,
            )
            continue

        if not watch.origin_pattern_signature:
            keep.append(watch)
            continue

        current = _signature_count(watch.origin_pattern_signature)
        recurrences = current - watch.baseline_count
        if recurrences >= _REVERT_THRESHOLD:
            ok = _trigger_rollback(watch, current=current, baseline=watch.baseline_count)
            if ok:
                counters["reverted"] += 1
            else:
                counters["revert_failed"] += 1
                # Keep the watch entry so operator visibility is preserved.
                keep.append(watch)
            continue

        keep.append(watch)

    _save_watches(keep)
    return counters


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _trigger_rollback(
    watch: _Watch,
    *,
    current: int,
    baseline: int,
) -> bool:
    """Roll back the auto-applied CR. Returns True on success."""
    try:
        from app.change_requests.apply import rollback_change
    except Exception:
        logger.warning(
            "auto_revert: rollback_change import failed", exc_info=True,
        )
        return False

    operator = "auto_revert_watcher"
    logger.warning(
        "auto_revert: rolling back %s — pattern %s recurred "
        "(baseline=%d, current=%d)",
        watch.cr_id, watch.origin_pattern_signature, baseline, current,
    )
    try:
        result = rollback_change(watch.cr_id, operator=operator)
    except Exception:
        logger.warning("auto_revert: rollback raised", exc_info=True)
        _alert_revert_failure(watch, "rollback raised")
        return False

    if not result.ok:
        _alert_revert_failure(watch, result.error or "unknown")
        return False

    _alert_revert_success(watch, current=current, baseline=baseline)
    _publish_auto_revert(watch, current=current, baseline=baseline)
    return True


def _alert_revert_success(watch: _Watch, *, current: int, baseline: int) -> None:
    try:
        from app.signal_client import send_message
        from app.config import get_settings
        recipient = (get_settings().signal_owner_number or "").strip()
        if not recipient:
            return
        send_message(
            recipient,
            f"🔁 Auto-revert: CR `{watch.cr_id}` was rolled back because "
            f"its origin pattern `{watch.origin_pattern_signature}` "
            f"recurred (baseline={baseline}, current={current}). "
            f"The patch did not solve the issue; reverting to the prior "
            f"state. See /cp/changes/{watch.cr_id} for details.",
        )
    except Exception:
        logger.debug("auto_revert: alert send failed", exc_info=True)


def _alert_revert_failure(watch: _Watch, reason: str) -> None:
    try:
        from app.signal_client import send_message
        from app.config import get_settings
        recipient = (get_settings().signal_owner_number or "").strip()
        if not recipient:
            return
        send_message(
            recipient,
            f"⚠️  Auto-revert FAILED for CR `{watch.cr_id}`: {reason[:300]}. "
            f"Manual intervention needed via /cp/changes/{watch.cr_id}.",
        )
    except Exception:
        logger.debug("auto_revert: failure alert send failed", exc_info=True)


def _publish_auto_revert(watch: _Watch, *, current: int, baseline: int) -> None:
    try:
        from app.workspace_publish import publish_to_workspace
        publish_to_workspace(
            source="change-requests-auto-revert",
            content=(
                f"auto-reverted CR {watch.cr_id} "
                f"(pattern={watch.origin_pattern_signature}, "
                f"baseline={baseline}, current={current})"
            ),
            salience=0.7,  # operator-relevant: system corrected itself
            signal_type="trend_reversal",
        )
    except Exception:
        logger.debug("auto_revert: GW publish failed", exc_info=True)


# ── Daemon driver ────────────────────────────────────────────────────


_driver_lock = threading.Lock()
_driver_started = False
_stop_event = threading.Event()


def _is_running() -> bool:
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


def _driver() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            counters = run_one_pass()
            if any(counters.get(k) for k in ("reverted", "revert_failed")):
                logger.info("auto_revert: %s", counters)
        except Exception:
            logger.debug("auto_revert: pass raised", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start() -> None:
    """Idempotent daemon launch."""
    global _driver_started
    if not _enabled():
        logger.info(
            "auto_revert: disabled via CHANGE_REQUESTS_AUTO_REVERT_ENABLED",
        )
        return
    with _driver_lock:
        if _is_running():
            return
        if _driver_started:
            logger.warning(
                "auto_revert: previous thread is dead, re-spawning",
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info(
            "auto_revert: daemon started (warm-up=%ds, poll=%ds, "
            "watch-window=%ds, revert-threshold=%d)",
            _WARMUP_S, _POLL_INTERVAL_S, _WATCH_WINDOW_S, _REVERT_THRESHOLD,
        )


def stop() -> None:
    _stop_event.set()


# Eager start at import — same pattern as healing/monitors and the
# proposal_bridge promoter.
start()
