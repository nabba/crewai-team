"""state.py — compose deployment state from multiple sources.

The single ``get_system_state(window_hours=24)`` function is the
entry point. It collects:

  * **git** — host source-repo state via the bridge if available;
    falls back to in-container `/app/workspace`'s git (workspace
    state, NOT source state — clearly labeled), then to the
    ``BUILD_SHA`` env var, then to "unknown".
  * **gateway** — uptime + start timestamp (read from /proc/1/stat).
  * **recent_crew_runs** — from the in-memory ring buffer at
    ``app.system_state.crew_runs``.
  * **tier_immutable_count** — ``len(TIER_IMMUTABLE)`` from
    auto_deployer.
  * **tools_registered** — ``len(ToolRegistry.instance().all())``.

Every source degrades gracefully: if it can't be read, the
returned dict's corresponding section is ``{"available": False,
"reason": "..."}``. Callers (the routing layer, the
``request_restricted_write`` tool, the React control plane) read
``available`` to know what to trust.

Speed: ~30-50ms cold (subprocess + bridge HTTP); cached for 5s
afterwards. Every consumer can call freely; we only do the work
when the cache expires.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


_CACHE: dict[tuple, dict[str, Any]] = {}  # key: (window_hours, crew_runs_limit)
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SEC = 5.0


# Conservative default — operators can override per-call.
_HOST_REPO_PATH_ENV = "HOST_REPO_PATH"
_HOST_REPO_PATH_DEFAULT = "/Users/andrus/BotArmy/crewai-team"


def _host_repo_path() -> str:
    return os.environ.get(_HOST_REPO_PATH_ENV) or _HOST_REPO_PATH_DEFAULT


# ── git ─────────────────────────────────────────────────────────────


def _git_via_bridge() -> dict[str, Any] | None:
    """Use the host bridge to read source-repo git state. Returns None
    if the bridge isn't reachable, ``{"available": True, ...}`` on
    success."""
    try:
        from app.bridge_client import get_bridge
    except Exception:
        return None
    bridge = get_bridge("system_state")
    if bridge is None or not bridge.is_available():
        return None

    cwd = _host_repo_path()
    try:
        sha = bridge.execute(
            ["git", "rev-parse", "HEAD"], working_dir=cwd, timeout=5,
        ).get("stdout", "").strip()
        if not sha:
            return None
        msg = bridge.execute(
            ["git", "log", "-1", "--format=%s"], working_dir=cwd, timeout=5,
        ).get("stdout", "").strip()
        head_iso = bridge.execute(
            ["git", "log", "-1", "--format=%cI"], working_dir=cwd, timeout=5,
        ).get("stdout", "").strip()
        dirty = bool(bridge.execute(
            ["git", "status", "--porcelain"], working_dir=cwd, timeout=5,
        ).get("stdout", "").strip())
        files_24h = bridge.execute(
            ["git", "log", "--since=24 hours ago", "--name-only", "--pretty=format:"],
            working_dir=cwd, timeout=5,
        ).get("stdout", "").strip()
        files_list = sorted(set(
            line.strip() for line in files_24h.splitlines() if line.strip()
        ))[:50]
        head_age_min = _age_minutes_from_iso(head_iso)
        return {
            "available": True,
            "source": "host bridge (source repo)",
            "head_sha": sha,
            "head_message": msg[:200],
            "head_age_min": head_age_min,
            "head_committed_at": head_iso,
            "uncommitted_changes": dirty,
            "files_changed_last_24h": files_list,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("system_state git via bridge failed: %s", exc)
        return None


def _git_via_subprocess(cwd: str) -> dict[str, Any] | None:
    """Try plain subprocess git in ``cwd``. Returns None on failure
    (no git repo, command unavailable, etc)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        ).stdout.strip()
        if not sha:
            return None
        msg = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        ).stdout.strip()
        head_iso = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        ).stdout.strip())
        files_24h = subprocess.run(
            ["git", "log", "--since=24 hours ago", "--name-only", "--pretty=format:"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        ).stdout.strip()
        files_list = sorted(set(
            line.strip() for line in files_24h.splitlines() if line.strip()
        ))[:50]
        head_age_min = _age_minutes_from_iso(head_iso)
        return {
            "available": True,
            "source": f"in-container subprocess ({cwd})",
            "head_sha": sha,
            "head_message": msg[:200],
            "head_age_min": head_age_min,
            "head_committed_at": head_iso,
            "uncommitted_changes": dirty,
            "files_changed_last_24h": files_list,
        }
    except Exception:
        return None


def _git_via_env() -> dict[str, Any] | None:
    """Last-resort: read BUILD_SHA from env. Returns None when unset."""
    sha = os.environ.get("BUILD_SHA", "").strip()
    if not sha:
        return None
    return {
        "available": True,
        "source": "BUILD_SHA env (no live git access)",
        "head_sha": sha,
        "head_message": "(unknown — read from BUILD_SHA env)",
        "head_age_min": None,
        "head_committed_at": None,
        "uncommitted_changes": None,
        "files_changed_last_24h": [],
    }


def _age_minutes_from_iso(iso: str) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0, int(delta.total_seconds() // 60))
    except Exception:
        return None


def _git_state() -> dict[str, Any]:
    """Try sources in order; return the first that succeeds, with a
    fallback chain making the source explicit."""
    for fn in (_git_via_bridge, lambda: _git_via_subprocess("/app"),
               lambda: _git_via_subprocess("/app/workspace"), _git_via_env):
        result = fn()
        if result is not None:
            return result
    return {"available": False, "reason": "no git source reachable"}


# ── gateway ─────────────────────────────────────────────────────────


def _gateway_state() -> dict[str, Any]:
    """Read PID 1's start time from /proc/1/stat. Linux-only;
    degrades cleanly elsewhere."""
    try:
        with open("/proc/1/stat") as f:
            stat = f.read().split()
        # Field 22 (zero-indexed: starttime in clock ticks since boot).
        # /proc/uptime gives boot time relative to now.
        starttime_clk = int(stat[21])
        clk_tck = os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
        boot_epoch = time.time() - uptime_s
        proc1_started_at_epoch = boot_epoch + (starttime_clk / clk_tck)
        uptime_min = int((time.time() - proc1_started_at_epoch) // 60)
        started_iso = datetime.fromtimestamp(
            proc1_started_at_epoch, tz=timezone.utc
        ).isoformat()
        return {
            "available": True,
            "uptime_min": uptime_min,
            "started_at": started_iso,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"/proc unavailable: {exc}"}


# ── tier-immutable + tools ──────────────────────────────────────────


def _tier_immutable_state() -> dict[str, Any]:
    try:
        from app.auto_deployer import TIER_IMMUTABLE
        return {
            "available": True,
            "count": len(TIER_IMMUTABLE),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


def _tools_state() -> dict[str, Any]:
    try:
        from app.tool_registry import ToolRegistry
        reg = ToolRegistry.instance()
        return {
            "available": True,
            "count": len(reg.all()),
            "names": sorted(s.name for s in reg.all())[:50],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


# ── crew runs ───────────────────────────────────────────────────────


def _crew_runs_state(window_hours: int, limit_per_crew: int) -> dict[str, Any]:
    try:
        from app.system_state.crew_runs import recent_runs, stats
        return {
            "available": True,
            "by_crew": recent_runs(limit=limit_per_crew),
            "buffer_sizes": stats(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


# ── public API ──────────────────────────────────────────────────────


def get_system_state(
    *,
    window_hours: int = 24,
    crew_runs_limit: int = 10,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compose deployment state from all sources.

    Returns a dict where every top-level key has an ``available``
    bool. Callers MUST check ``available`` before reading fields —
    sources can be unreachable transiently and we degrade rather
    than fail.

    Cached for 5 seconds. Pass ``use_cache=False`` to force refresh
    (rarely needed; cache is short).
    """
    now = time.time()
    cache_key = (window_hours, crew_runs_limit)
    if use_cache:
        with _CACHE_LOCK:
            entry = _CACHE.get(cache_key)
            if entry is not None and entry["expires_at"] > now:
                return entry["value"]

    state = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "git": _git_state(),
        "gateway": _gateway_state(),
        "tier_immutable": _tier_immutable_state(),
        "tools": _tools_state(),
        "recent_crew_runs": _crew_runs_state(window_hours, crew_runs_limit),
    }

    with _CACHE_LOCK:
        _CACHE[cache_key] = {
            "value": state,
            "expires_at": now + _CACHE_TTL_SEC,
        }

    return state


def reset_cache_for_tests() -> None:
    """Clear the cache so tests see live results."""
    with _CACHE_LOCK:
        _CACHE.clear()
