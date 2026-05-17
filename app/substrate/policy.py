"""
substrate.policy — single decision point for "should heavy work defer?"

Used by the idle scheduler (T2.5) to honor host resource posture
WITHOUT silently dropping work. When this function returns a non-None
reason, the caller MUST emit a visible ``idle_job_deferred`` event
naming the reason — silent deferral is the failure mode this exists
to prevent.

Pure data → decision. No I/O. Caller passes in a ``SubstrateStatus``
snapshot (or lets the policy gather one). Predicates are conservative
defaults; operators can tune via ``ResourcePolicy`` instance or by
flipping runtime_settings (future T-tier work).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourcePolicy:
    """Local-mode resource budget — what must remain free for the human user.

    Cloud-mode policy is separate (deferred per the productization plan
    until cloud workload exists). These defaults are tuned for a
    single-operator MacBook running the gateway alongside the user's
    own work; bump them if the system runs on a dedicated host.
    """
    min_free_disk_gb: float = 2.0
    """Below this, heavy idle work defers."""

    min_free_disk_gb_critical: float = 0.5
    """Below this, even LIGHT idle work defers."""

    max_inflight_tasks: int = 8
    """Above this, defer heavy work — let the foreground request finish first."""


_DEFAULT_POLICY = ResourcePolicy()


def should_defer_heavy_work(
    snapshot=None,
    policy: Optional[ResourcePolicy] = None,
) -> Optional[str]:
    """Return a defer_reason string if heavy work should pause, else None.

    Callers (idle scheduler) MUST emit a visible event when this returns
    a reason — silent deferral is the pattern this function is designed
    to prevent. Pass ``snapshot=None`` to gather one inline; pass a cached
    one if you have it (saves ~50 ms).

    Never raises. On internal probe failure, returns None (fail-open) so
    a broken policy doesn't stall the system.
    """
    pol = policy or _DEFAULT_POLICY

    if snapshot is None:
        try:
            from app.substrate.status import gather_substrate_status
            snapshot = gather_substrate_status()
        except Exception as exc:
            logger.debug("substrate.policy: snapshot failed (fail-open): %s", exc)
            return None

    try:
        # Disk pressure — hard floor.
        disk_free = (snapshot.resources or {}).get("disk_free_gb")
        if disk_free is not None:
            if disk_free < pol.min_free_disk_gb_critical:
                return f"disk_free={disk_free:.1f}GB < critical={pol.min_free_disk_gb_critical}GB"
            if disk_free < pol.min_free_disk_gb:
                return f"disk_free={disk_free:.1f}GB < min={pol.min_free_disk_gb}GB"

        # Inflight-tasks pressure.
        inflight = int(getattr(snapshot, "inflight_tasks", 0) or 0)
        if inflight > pol.max_inflight_tasks:
            return f"inflight_tasks={inflight} > max={pol.max_inflight_tasks}"

        # Host substrate alerts — if Q16 monitor recently fired a disk-horizon
        # or memory-headroom alert, defer until cleared.
        alerts = (snapshot.resources or {}).get("host_substrate_alerts") or []
        for a in alerts:
            kind = a.get("kind") or ""
            if kind in ("disk_horizon", "low_memory_headroom", "uptime_stale"):
                return f"host_substrate_alert={kind}"
    except Exception as exc:
        logger.debug("substrate.policy: predicate eval failed (fail-open): %s", exc)
        return None

    return None
