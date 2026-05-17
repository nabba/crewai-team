"""
substrate.status — single read aggregator for the living system.

``gather_substrate_status()`` walks every existing health surface
(memory stores, SubIA integrity, self-improvement queues, host capacity,
identity continuity ledger, healing monitors, runtime settings, DR
freshness) and returns a typed snapshot. Each probe is failure-isolated:
broken probes record an entry in ``errors`` and the snapshot continues.

Used by:
  - the /cp/status dashboard page (T2.4)
  - the `botarmy status` and `botarmy doctor --deep` commands (T2.2)
  - idle scheduler resource gating via substrate.policy (T2.5)

Performance: target <100 ms wall time on a healthy laptop. Each probe
times out implicitly through the underlying subsystem's own timeouts;
no probe spawns threads, sleeps, or runs LLMs.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SubstrateStatus:
    """A snapshot of the living system at one moment in time.

    Every field has a safe default. Fields populated by failed probes
    carry placeholder values; the matching error string lands in
    ``errors``. Operator surfaces should render the snapshot AND surface
    the errors list — silent gaps are worse than visible ones.
    """
    timestamp: str = ""

    # ── Live operations ─────────────────────────────────────────────
    inflight_tasks: int = 0
    inbound_queue_depth: int = 0
    dlq_depth: int = 0

    # ── Memory subsystems ───────────────────────────────────────────
    memory: dict[str, Any] = field(default_factory=dict)

    # ── SubIA + identity ────────────────────────────────────────────
    subia: dict[str, Any] = field(default_factory=dict)

    # ── Self-improvement ────────────────────────────────────────────
    self_improvement: dict[str, Any] = field(default_factory=dict)

    # ── Resources + host posture ────────────────────────────────────
    resources: dict[str, Any] = field(default_factory=dict)

    # ── Continuity (backup + drill freshness) ──────────────────────
    continuity: dict[str, Any] = field(default_factory=dict)

    # ── Health monitors ─────────────────────────────────────────────
    health: dict[str, Any] = field(default_factory=dict)

    # ── Active feature toggles ─────────────────────────────────────
    settings: dict[str, Any] = field(default_factory=dict)

    # ── Per-probe errors ────────────────────────────────────────────
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Internal probes ─────────────────────────────────────────────────


def _probe_inflight(snap: SubstrateStatus) -> None:
    try:
        from app import main as _main
        snap.inflight_tasks = int(getattr(_main, "_inflight_tasks", 0) or 0)
    except Exception as exc:
        snap.errors.append(f"inflight: {exc}")


def _probe_queues(snap: SubstrateStatus) -> None:
    try:
        from app.dead_letter_inbound import queue_depth as _dlq_depth
        snap.dlq_depth = int(_dlq_depth() or 0)
    except Exception as exc:
        snap.errors.append(f"dlq_depth: {exc}")

    try:
        from app.conversation_store import _get_conn
        conn = _get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM inbound_queue WHERE status IN ('queued', 'processing')"
        ).fetchone()
        snap.inbound_queue_depth = int(row[0]) if row else 0
    except Exception as exc:
        snap.errors.append(f"inbound_queue_depth: {exc}")


def _probe_memory(snap: SubstrateStatus) -> None:
    out: dict[str, Any] = {
        "chromadb_ok": None,
        "postgres_ok": None,
        "neo4j_ok": None,
        "mem0_ok": None,
        "drift_30d": None,
    }
    # ChromaDB: simplest signal — can we list collections?
    try:
        from app.memory.chromadb_manager import get_kb_client
        client = get_kb_client("memory")
        # cheap probe: count collections
        try:
            collections = client.list_collections() if client else None
            out["chromadb_ok"] = collections is not None
            out["chromadb_collections"] = len(collections) if collections else 0
        except Exception:
            out["chromadb_ok"] = False
    except Exception as exc:
        out["chromadb_ok"] = False
        snap.errors.append(f"memory.chromadb: {exc}")

    # Identity continuity drift summary (30-day window)
    try:
        from app.identity.continuity_ledger import summarise_drift
        drift = summarise_drift(window_days=30)
        out["drift_30d"] = {
            "n_events": drift.n_events,
            "by_kind": dict(drift.by_kind),
            "first_seen": drift.first_seen,
            "last_seen": drift.last_seen,
        }
    except Exception as exc:
        snap.errors.append(f"memory.drift: {exc}")

    snap.memory = out


def _probe_subia(snap: SubstrateStatus) -> None:
    out: dict[str, Any] = {
        "live_enabled": False,
        "integrity_ok": None,
        "integrity_drift": None,
        "kernel_loop_count": None,
    }

    # Live-integration toggle (runtime_settings first, env fallback)
    try:
        from app.runtime_settings import get_subia_live_enabled
        out["live_enabled"] = bool(get_subia_live_enabled())
    except Exception as exc:
        snap.errors.append(f"subia.flag: {exc}")

    # Integrity manifest
    try:
        from app.subia.integrity import verify_integrity
        r = verify_integrity(strict=False)
        out["integrity_ok"] = bool(r.ok)
        out["integrity_drift"] = bool(r.has_drift)
        out["integrity_n_files"] = int(getattr(r, "n_files", 0))
        out["integrity_mismatched"] = len(getattr(r, "mismatched", []) or [])
        out["integrity_extra"] = len(getattr(r, "extra", []) or [])
        out["integrity_missing"] = len(getattr(r, "missing", []) or [])
    except Exception as exc:
        snap.errors.append(f"subia.integrity: {exc}")

    # Kernel state (if live)
    try:
        from app.subia.kernel import get_active_kernel
        k = get_active_kernel()
        if k is not None:
            out["kernel_loop_count"] = int(getattr(k, "loop_count", 0) or 0)
    except Exception as exc:
        snap.errors.append(f"subia.kernel: {exc}")

    snap.subia = out


def _probe_self_improvement(snap: SubstrateStatus) -> None:
    out: dict[str, Any] = {
        "pending_change_requests": None,
        "pending_proposals": None,
        "last_deploy": None,
        "active_coding_sessions": None,
    }

    # Change requests (PENDING count)
    try:
        from app.change_requests.store import list_all
        all_reqs = list_all() or []
        out["pending_change_requests"] = sum(
            1 for r in all_reqs
            if getattr(r, "status", None) == "PENDING"
            or (isinstance(r, dict) and r.get("status") == "PENDING")
        )
        out["total_change_requests"] = len(all_reqs)
    except Exception as exc:
        snap.errors.append(f"self_improvement.crs: {exc}")

    # Proposals (pending file count under workspace/proposals/)
    try:
        from app.paths import WORKSPACE_ROOT
        prop_dir = WORKSPACE_ROOT / "proposals"
        if prop_dir.exists():
            pending = 0
            for sub in prop_dir.iterdir():
                if not sub.is_dir():
                    continue
                status_file = sub / "status.json"
                if status_file.exists():
                    import json
                    try:
                        s = json.loads(status_file.read_text())
                        if s.get("status") == "pending":
                            pending += 1
                    except Exception:
                        continue
            out["pending_proposals"] = pending
    except Exception as exc:
        snap.errors.append(f"self_improvement.proposals: {exc}")

    # Last deploy log row
    try:
        from app.auto_deployer import DEPLOY_LOG
        if DEPLOY_LOG.exists():
            import json
            log = json.loads(DEPLOY_LOG.read_text())
            if log:
                last = log[-1]
                out["last_deploy"] = {
                    "ts": last.get("ts"),
                    "status": last.get("status"),
                    "reason": (last.get("reason") or "")[:80],
                    "files": last.get("files", [])[:3],
                    "evidence_source": (last.get("evidence") or {}).get("source"),
                }
    except Exception as exc:
        snap.errors.append(f"self_improvement.last_deploy: {exc}")

    snap.self_improvement = out


def _probe_resources(snap: SubstrateStatus) -> None:
    out: dict[str, Any] = {
        "disk_free_gb": None,
        "disk_total_gb": None,
        "host_substrate_state": None,
        "host_substrate_alerts": [],
    }

    # Disk free under workspace
    try:
        from app.paths import WORKSPACE_ROOT
        usage = shutil.disk_usage(str(WORKSPACE_ROOT))
        out["disk_free_gb"] = round(usage.free / 1024 / 1024 / 1024, 2)
        out["disk_total_gb"] = round(usage.total / 1024 / 1024 / 1024, 2)
    except Exception as exc:
        snap.errors.append(f"resources.disk: {exc}")

    # Host substrate state (from Q16 monitor)
    try:
        from app.healing.monitors.host_substrate_health import _read_state, _state_path
        if _state_path().exists():
            state = _read_state()
            out["host_substrate_state"] = {
                "last_run_at": state.get("last_run_at"),
                "fingerprint_hostname": (state.get("fingerprint") or {}).get("hostname"),
                "uptime_days": state.get("uptime_days"),
            }
            # Recent alerts
            alerts = state.get("recent_alerts") or []
            out["host_substrate_alerts"] = [
                {"kind": a.get("kind"), "ts": a.get("ts"), "msg": (a.get("msg") or "")[:120]}
                for a in alerts[-3:]
            ]
    except Exception as exc:
        snap.errors.append(f"resources.host_substrate: {exc}")

    snap.resources = out


def _probe_continuity(snap: SubstrateStatus) -> None:
    out: dict[str, Any] = {
        "last_backup": None,
        "last_backup_age_days": None,
        "last_drill_ok": None,
        "ledger_event_count_30d": None,
    }

    # DR backup freshness — read from monitor state if available
    try:
        from app.paths import WORKSPACE_ROOT
        backup_dir = WORKSPACE_ROOT / "backups" / "dr"
        if backup_dir.exists():
            tarballs = sorted(
                backup_dir.glob("*.tar.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if tarballs:
                newest = tarballs[0]
                age_seconds = datetime.now(timezone.utc).timestamp() - newest.stat().st_mtime
                out["last_backup"] = newest.name
                out["last_backup_age_days"] = round(age_seconds / 86400, 1)
    except Exception as exc:
        snap.errors.append(f"continuity.backup: {exc}")

    # Continuity ledger event count (30d)
    try:
        from app.identity.continuity_ledger import summarise_drift
        d = summarise_drift(window_days=30)
        out["ledger_event_count_30d"] = int(d.n_events)
    except Exception as exc:
        snap.errors.append(f"continuity.ledger: {exc}")

    snap.continuity = out


def _probe_health_monitors(snap: SubstrateStatus) -> None:
    """Count healing-monitor outcomes by classification.

    The monitor framework keeps state under workspace/self_heal/. We
    read directory listings rather than importing each monitor, so a
    broken monitor module doesn't take down the probe.
    """
    out: dict[str, Any] = {
        "n_state_files": 0,
        "recent_alerts": [],
    }
    try:
        from app.paths import WORKSPACE_ROOT
        sh_dir = WORKSPACE_ROOT / "self_heal"
        if sh_dir.exists():
            state_files = list(sh_dir.glob("*.json"))
            out["n_state_files"] = len(state_files)
    except Exception as exc:
        snap.errors.append(f"health.monitors: {exc}")

    snap.health = out


def _probe_settings(snap: SubstrateStatus) -> None:
    out: dict[str, Any] = {}
    try:
        from app import runtime_settings as rs
        out = {
            "subia_live_enabled": rs.get_subia_live_enabled(),
            "subia_grounding_enabled": rs.get_subia_grounding_enabled(),
            "subia_idle_jobs_enabled": rs.get_subia_idle_jobs_enabled(),
            "recovery_loop_enabled": rs.get_recovery_loop_enabled(),
            "tool_supervisor_enabled": rs.get_tool_supervisor_enabled(),
            "error_runbooks_enabled": rs.get_error_runbooks_enabled(),
            "tier3_amendment_enabled": rs.get_tier3_amendment_enabled(),
        }
    except Exception as exc:
        snap.errors.append(f"settings: {exc}")
    snap.settings = out


# ── Public entry point ──────────────────────────────────────────────


def gather_substrate_status() -> SubstrateStatus:
    """Walk every existing health surface and return a typed snapshot.

    Pure read. Never raises. Per-probe failures land in ``errors`` and
    the snapshot continues. Designed to be safe to call from any
    operator surface at any cadence.
    """
    snap = SubstrateStatus(
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    # Order is for readability only — probes are independent.
    _probe_inflight(snap)
    _probe_queues(snap)
    _probe_memory(snap)
    _probe_subia(snap)
    _probe_self_improvement(snap)
    _probe_resources(snap)
    _probe_continuity(snap)
    _probe_health_monitors(snap)
    _probe_settings(snap)
    return snap
