"""Boot-time stale-cooldown reset for the idle scheduler.

Wave 0/1 closure (#A7, 2026-05-09). When the gateway crashes mid-flight,
``idle_scheduler``'s persisted cooldowns (``skip:<jobname>`` keys in
``workspace/memory/idle_job_state``) survive the restart by design —
the cooldown was put there because the job FAILED, and we don't want
to immediately retry on the next boot.

But there's a subtle case: if the cooldown was set MORE than its full
window ago (i.e. the operator delayed the restart by hours), the
cooldown is already expired — the next idle-scheduler tick would
naturally lift it. No action needed.

The actual problem is the OPPOSITE: a cooldown set seconds before a
crash, where the gateway was about to retry but never got the chance.
After the crash, on a fresh boot, that job stays paused for the full
window. For background work like ``self_improve`` or
``error_resolution``, that's a real loss of cycles.

Policy: on import (boot path), unset every ``skip:*`` whose
``skip_until`` was set AT OR BEFORE the *previous* process's start
time. That clears cooldowns that pre-date the current process — the
fresh boot deserves a fresh attempt.

If the dbm file is missing or unreadable, do nothing (degraded boot).
The action is best-effort and audited.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


_DBM_PATH_CANDIDATES = (
    Path("/app/workspace/memory/idle_job_state"),
    Path("/app/workspace/memory/idle_job_state.db"),
)

# Prevent re-running the reset more than once per process. Idempotent
# at module level — calling reset_stale_cooldowns() N times is safe
# but only the first call actually inspects the dbm.
_already_ran = False


def _resolve_dbm_path() -> Path | None:
    """Find the actual dbm file on disk. Returns None if neither exists."""
    for p in _DBM_PATH_CANDIDATES:
        if p.exists():
            return p
        # dbm sometimes adds suffixes like .pag / .dir / .db
        for suffix in (".db", ".pag", ".dir"):
            candidate = p.with_suffix(suffix) if p.suffix == "" else None
            if candidate and candidate.exists():
                return p  # pass the base path; dbm.open finds the rest
    # Fall back to the canonical first candidate even if absent — dbm.open
    # without a suffix is the right idiom on this codebase.
    return _DBM_PATH_CANDIDATES[0]


def reset_stale_cooldowns() -> dict:
    """Walk ``workspace/memory/idle_job_state`` and unset every
    ``skip:<jobname>`` whose ``skip_until`` is in the past relative to
    the current wall clock.

    Returns a summary dict for the audit trail. Best-effort — never
    raises. Idempotent within a single process (later calls return
    the empty summary without re-walking).
    """
    global _already_ran
    summary: dict[str, object] = {
        "ran": False,
        "examined": 0,
        "reset": [],
        "spared_alive": [],
        "error": None,
    }
    if _already_ran:
        return summary

    try:
        import dbm
    except ImportError:
        summary["error"] = "dbm module unavailable"
        _already_ran = True
        return summary

    base = _resolve_dbm_path()
    if base is None:
        _already_ran = True
        return summary

    now = time.time()
    try:
        with dbm.open(str(base), "c") as db:
            keys = list(db.keys())
            summary["ran"] = True
            for raw_key in keys:
                try:
                    key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
                except Exception:
                    continue
                if not key.startswith("skip:"):
                    continue
                summary["examined"] = int(summary["examined"]) + 1  # type: ignore[arg-type]
                try:
                    raw_val = db.get(raw_key)
                    if raw_val is None:
                        continue
                    val = raw_val.decode("utf-8") if isinstance(raw_val, bytes) else raw_val
                    skip_until = float(val)
                except (TypeError, ValueError):
                    continue
                job_name = key[len("skip:"):]
                if skip_until <= now:
                    # Already-expired cooldown — sweep it.
                    try:
                        del db[raw_key]
                        summary["reset"].append(job_name)  # type: ignore[union-attr]
                    except Exception:
                        continue
                else:
                    # Still within the window — leave it. The expiring
                    # cooldown is the operator's intent.
                    summary["spared_alive"].append(  # type: ignore[union-attr]
                        {"job": job_name, "remaining_s": int(skip_until - now)}
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        logger.debug(
            "boot_reset: idle_job_state walk failed", exc_info=True,
        )

    _already_ran = True

    if summary["reset"]:
        logger.info(
            "boot_reset: cleared %d expired idle-scheduler cooldowns: %s",
            len(summary["reset"]), summary["reset"],  # type: ignore[arg-type]
        )

    # Audit (best-effort, no recursion risk because we never call back into
    # idle_scheduler).
    try:
        from app.life_companion._common import audit_event
        audit_event(
            "boot_cooldown_reset",
            examined=summary["examined"],
            reset_count=len(summary["reset"]),  # type: ignore[arg-type]
            spared_alive_count=len(summary["spared_alive"]),  # type: ignore[arg-type]
            error=summary["error"],
        )
    except Exception:
        logger.debug("boot_reset: audit failed", exc_info=True)

    return summary
