"""host_substrate_health — long-horizon host-substrate health probe.

PROGRAM §51 — Q16 Theme 1 (decade-resilience hardening, substrate
longevity). The existing 34 monitors all watch the system *inside*
the host. None watch the **host itself** — its disk, its uptime
pattern, its kernel/OS freshness. Over 5+ years of single-host
operation, this is the single biggest unaddressed risk.

What this monitor observes (from inside the container — non-invasive):

  1. **Workspace volume free-space trend.** ``disk_quota`` watches
     the threshold; this watches the *slope*. Linear regression
     over rolling weekly samples; predicts ``days_until_full`` at
     the current burn rate. Alerts at <60d horizon long before the
     ``disk_quota`` threshold trips.

  2. **Workspace bytes growth.** Total size of ``workspace/`` tracked
     weekly. Sustained week-over-week growth >10% gets surfaced as a
     "what's growing?" question (often a leaky log file or a runaway
     index).

  3. **Process uptime + restart cadence.** Records the gateway's own
     boot timestamp on first probe of each lifetime. Alerts on
     >3 restarts/24h (instability) and on uptime >180d with no
     observable code change (stale code in production).

  4. **Memory headroom.** Reads ``/proc/meminfo`` when available
     (Linux containers). Alerts when ``MemAvailable`` is sustained
     below 10% of ``MemTotal``. Mac Docker Desktop containers see
     the VM's view, not the host's, so this is a coarse signal.

  5. **Host-side telemetry pass-through.** If a host-side companion
     writes to ``workspace/healing/host_metrics.jsonl`` (SMART data,
     ``sw_vers``, kernel version — anything the container can't see
     directly), this monitor surfaces the latest row in its summary
     without imposing a strict schema. See the §51 ``SUBSTRATE_
     MIGRATION.md`` playbook for the optional host-side companion
     pattern (parallels Q15's two-process split for browse
     ingestion).

What this monitor **does not** do:

  * No auto-action. Disk-quota's auto-retention is already the lever
    for "running out of space"; substrate-health alerts are slow-burn
    warnings the operator must triage.
  * No host modifications. Reads only.
  * No external network calls.

Cadence: daily probe; internal weekly cadence for trend computation.
Master switch: ``host_substrate_health_monitor_enabled`` (default ON).
Alert dedup: 14 days per (kind, severity).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


NAME = "host_substrate_health"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "host_substrate_health_monitor_enabled"

# Internal cadence — weekly samples; the daily probe is just the
# wake-up. We only compute trends + emit alerts once per week.
_WEEKLY_S = 7 * 24 * 3600

# Alert thresholds.
_DAYS_UNTIL_FULL_WARN = 60  # alert when burn-rate projects <60d to full
_GROWTH_WOW_WARN_PCT = 10.0  # alert on sustained week-over-week >10%
_GROWTH_SUSTAINED_WEEKS = 4  # number of consecutive weeks above threshold
_MEMORY_HEADROOM_WARN_PCT = 10.0  # alert when MemAvailable < 10% of total
_RESTART_BURST_WINDOW_S = 24 * 3600
_RESTART_BURST_THRESHOLD = 3
_UPTIME_STALE_DAYS = 180

_DEDUP_WINDOW_S = 14 * 86400
_STATE_FILE_NAME = "host_substrate_health_state.json"
_HOST_METRICS_FILE = "host_metrics.jsonl"
_MAX_HISTORY_SAMPLES = 52  # one year of weekly samples

# Module-level marker set on the first probe of each process lifetime
# (process-local; not persisted). Used to detect restarts and to
# compute uptime. Tests can reset via ``reset_process_marker()``.
_PROCESS_BOOT_AT: Optional[float] = None


def reset_process_marker() -> None:
    """Test hook: clear the in-process boot marker so the next ``run()``
    treats the call as a fresh restart."""
    global _PROCESS_BOOT_AT
    _PROCESS_BOOT_AT = None


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_host_substrate_health_monitor_enabled
        return get_host_substrate_health_monitor_enabled()
    except Exception:
        return os.getenv(
            "HOST_SUBSTRATE_HEALTH_MONITOR_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _host_metrics_path() -> Path:
    return _workspace() / "healing" / _HOST_METRICS_FILE


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {
            "last_run_at": 0.0,
            "boot_at": 0.0,
            "restart_log": [],
            "weekly_samples": [],
            "last_alert_at": {},
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "last_run_at": 0.0,
            "boot_at": 0.0,
            "restart_log": [],
            "weekly_samples": [],
            "last_alert_at": {},
        }


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("host_substrate_health: state write failed", exc_info=True)


def _disk_usage_bytes() -> tuple[int, int]:
    """Returns ``(free_bytes, total_bytes)`` for the workspace volume.
    ``(0, 0)`` on failure."""
    try:
        import shutil
        usage = shutil.disk_usage(str(_workspace()))
        return usage.free, usage.total
    except Exception:
        return 0, 0


def _workspace_bytes() -> int:
    """Total bytes under the workspace tree. Falls back to 0 on
    failure. Walks once per weekly sample, not per probe."""
    total = 0
    try:
        root = _workspace()
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    continue
    except Exception:
        logger.debug("host_substrate_health: walk failed", exc_info=True)
    return total


def _substrate_fingerprint() -> dict[str, Any]:
    """Cheap, cross-platform host-identity probe. Combination of values
    that change when the gateway is moved to a different machine but
    NOT when the same machine restarts. Each lookup is failure-isolated
    so a missing field never destabilises the comparison."""
    fp: dict[str, Any] = {}
    try:
        import platform
        fp["hostname"] = platform.node() or ""
        fp["system"] = platform.system() or ""
        fp["machine"] = platform.machine() or ""
        fp["python_version"] = platform.python_version() or ""
    except Exception:
        fp.setdefault("hostname", "")
        fp.setdefault("system", "")
        fp.setdefault("machine", "")
        fp.setdefault("python_version", "")
    try:
        import shutil
        usage = shutil.disk_usage(str(_workspace()))
        # Round to GB so minor expansion of the same volume doesn't
        # register as a transition.
        fp["total_gb"] = int(usage.total / 1024 / 1024 / 1024)
    except Exception:
        fp["total_gb"] = 0
    return fp


def _substrate_transition(
    prior: Optional[dict[str, Any]],
    current: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Compare prior + current fingerprints. Returns a description of
    the transition iff a *material* difference exists. None otherwise.

    Material = hostname OR machine OR system changed, OR total_gb
    differs by >10% (different volume, not just expanded same one).
    Python-version drift alone does NOT count (operator upgrade is a
    normal event)."""
    if not prior:
        return None
    if not isinstance(prior, dict):
        return None

    diffs: dict[str, dict[str, Any]] = {}
    for key in ("hostname", "system", "machine"):
        pv = prior.get(key, "")
        cv = current.get(key, "")
        if pv and cv and pv != cv:
            diffs[key] = {"from": pv, "to": cv}

    prior_gb = float(prior.get("total_gb", 0))
    cur_gb = float(current.get("total_gb", 0))
    if prior_gb > 0 and cur_gb > 0:
        rel = abs(cur_gb - prior_gb) / prior_gb
        if rel > 0.10:
            diffs["total_gb"] = {
                "from": int(prior_gb),
                "to": int(cur_gb),
                "rel_change": round(rel, 3),
            }

    if not diffs:
        return None
    # Python version is informational, not a primary signal — but
    # surface it when it differs alongside another signal so the
    # operator has full context.
    if "python_version" not in diffs:
        pv = prior.get("python_version", "")
        cv = current.get("python_version", "")
        if pv and cv and pv != cv:
            diffs["python_version"] = {"from": pv, "to": cv}
    return diffs


def _emit_substrate_migration(
    transition: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Emit a continuity-ledger ``substrate_migration`` event +
    loud Signal alert. Failure-isolated."""
    try:
        from app.identity.continuity_ledger import record_event
        summary_bits = []
        for k, v in transition.items():
            if isinstance(v, dict) and "from" in v and "to" in v:
                summary_bits.append(f"{k} {v['from']!r}→{v['to']!r}")
        summary = (
            "host substrate change detected: " + "; ".join(summary_bits)
            if summary_bits else "host substrate change detected"
        )
        record_event(
            kind="substrate_migration",
            actor="host_substrate_health",
            summary=summary,
            detail={
                "transition": transition,
                "current_fingerprint": current,
                "source": "host_substrate_health.weekly_probe",
            },
        )
    except Exception:
        logger.debug(
            "host_substrate_health: ledger emit failed", exc_info=True,
        )
    try:
        from app.notify import notify
        bits = []
        for k, v in transition.items():
            if isinstance(v, dict) and "from" in v and "to" in v:
                bits.append(f"  • {k}: `{v['from']}` → `{v['to']}`")
        body = (
            "🏠 **Substrate transition detected** — the gateway is "
            "running on a host that looks different from the one it "
            "saw on the previous weekly probe.\n\n"
            + "\n".join(bits)
            + "\n\nIf this is intentional (laptop migration, VPS move, "
            "Docker Desktop reset), no action needed — a "
            "`substrate_migration` event has landed in the continuity "
            "ledger so the annual reflection will pick it up. If "
            "unintentional, investigate."
        )
        notify(
            title="🏠 Substrate transition",
            body=body,
            url="/cp/monitor",
            topic="host_substrate_health:substrate_transition",
            critical=True,    # operator should always know
            arbitrate=False,
        )
    except Exception:
        logger.debug(
            "host_substrate_health: notify failed", exc_info=True,
        )


def _memory_headroom() -> Optional[dict[str, Any]]:
    """Returns ``{mem_available_bytes, mem_total_bytes, headroom_pct}``
    on Linux. None on macOS / Windows / failure."""
    p = Path("/proc/meminfo")
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return None
    mem_total = None
    mem_available = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        rest = rest.strip().split()
        if not rest:
            continue
        try:
            value_kb = int(rest[0])
        except ValueError:
            continue
        if key == "MemTotal":
            mem_total = value_kb * 1024
        elif key == "MemAvailable":
            mem_available = value_kb * 1024
    if mem_total is None or mem_available is None or mem_total == 0:
        return None
    return {
        "mem_available_bytes": mem_available,
        "mem_total_bytes": mem_total,
        "headroom_pct": 100.0 * mem_available / mem_total,
    }


def _read_host_metrics_tail() -> Optional[dict[str, Any]]:
    """Reads the last row of the optional ``host_metrics.jsonl`` that
    an out-of-band host-side companion may write. No schema enforced
    — surfaces whatever's there. Returns None if the file is missing
    or unreadable."""
    p = _host_metrics_path()
    if not p.exists():
        return None
    try:
        last_line = None
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return None
        return json.loads(last_line)
    except Exception:
        return None


def _linear_slope(samples: list[tuple[float, float]]) -> float:
    """Least-squares slope (y per unit x) for ``samples = [(x, y), ...]``.
    Returns 0.0 for fewer than 2 points."""
    n = len(samples)
    if n < 2:
        return 0.0
    sum_x = sum(p[0] for p in samples)
    sum_y = sum(p[1] for p in samples)
    sum_xy = sum(p[0] * p[1] for p in samples)
    sum_xx = sum(p[0] * p[0] for p in samples)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def _project_days_until_full(samples: list[dict[str, Any]]) -> Optional[float]:
    """Given weekly samples with ``ts`` + ``free_bytes``, returns the
    projected days until free_bytes hits zero at the current trend.
    Returns None if there's no negative slope (free space is growing
    or flat) or fewer than 2 samples."""
    if len(samples) < 2:
        return None
    # Newest samples first; need at least 4 weeks of data for a
    # stable trend.
    if len(samples) < 4:
        return None
    points = [
        (float(s.get("ts", 0)), float(s.get("free_bytes", 0)))
        for s in samples[-8:]  # last 8 weeks
    ]
    slope = _linear_slope(points)  # bytes per second
    if slope >= 0:
        return None
    current_free = points[-1][1]
    if current_free <= 0:
        return 0.0
    seconds_until_empty = -current_free / slope
    return seconds_until_empty / 86400.0


def _record_restart(state: dict[str, Any], now: float) -> bool:
    """Records a restart on first probe of the lifetime. Returns True
    iff this is the first probe (restart was just recorded).

    Uses the module-level ``_PROCESS_BOOT_AT`` marker. Tests can reset
    via :func:`reset_process_marker` to simulate a fresh restart."""
    global _PROCESS_BOOT_AT
    if _PROCESS_BOOT_AT is not None:
        return False
    _PROCESS_BOOT_AT = now
    log = state.setdefault("restart_log", [])
    if not isinstance(log, list):
        log = []
        state["restart_log"] = log
    log.append(now)
    # Cap at 50 most recent — we only care about recent bursts.
    if len(log) > 50:
        del log[: len(log) - 50]
    return True


def _alert_if_due(
    state: dict[str, Any],
    *,
    key: str,
    title: str,
    body: str,
    now: float,
) -> bool:
    """Issues a Signal alert if the dedup window has elapsed. Returns
    True iff sent. Failure-isolated."""
    last_alerts = state.setdefault("last_alert_at", {})
    if not isinstance(last_alerts, dict):
        last_alerts = {}
        state["last_alert_at"] = last_alerts
    last = float(last_alerts.get(key, 0))
    if now - last < _DEDUP_WINDOW_S:
        return False
    last_alerts[key] = now
    try:
        from app.notify import notify
        notify(
            title=title,
            body=body,
            url="/cp/health",
            topic=f"host_substrate_health:{key}",
            critical=False,
            arbitrate=True,
        )
        return True
    except Exception:
        logger.debug("host_substrate_health: notify failed", exc_info=True)
        return False


def _gather_weekly_sample(now: float) -> dict[str, Any]:
    """One-shot sample of every signal we trend on. Cheap enough to
    take once a week; the workspace walk is the slow part."""
    free_b, total_b = _disk_usage_bytes()
    sample = {
        "ts": now,
        "free_bytes": free_b,
        "total_bytes": total_b,
        "workspace_bytes": _workspace_bytes(),
    }
    mem = _memory_headroom()
    if mem is not None:
        sample["mem_available_bytes"] = mem["mem_available_bytes"]
        sample["mem_total_bytes"] = mem["mem_total_bytes"]
        sample["headroom_pct"] = round(mem["headroom_pct"], 2)
    return sample


def _check_disk_horizon(
    state: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    now: float,
) -> Optional[dict[str, Any]]:
    """If projected days-until-full < threshold, emit alert. Returns
    the alert payload (for the summary) or None."""
    days = _project_days_until_full(samples)
    if days is None:
        return None
    if days > _DAYS_UNTIL_FULL_WARN:
        return None
    latest = samples[-1] if samples else {}
    free_gb = latest.get("free_bytes", 0) / 1024 / 1024 / 1024
    total_gb = latest.get("total_bytes", 0) / 1024 / 1024 / 1024
    body = (
        f"📉 Workspace free-space trend projects exhaustion in "
        f"~{days:.0f} days at the current burn rate.\n"
        f"  • free now: {free_gb:.1f} GB of {total_gb:.1f} GB\n"
        f"  • horizon threshold: {_DAYS_UNTIL_FULL_WARN} days\n\n"
        f"This is a slow-burn warning, not an imminent failure. "
        f"Triage: review what's growing under workspace/, consider "
        f"earlier retention runs, or expand the volume."
    )
    sent = _alert_if_due(
        state,
        key="disk_horizon_warn",
        title="📉 Disk-free horizon < 60 days",
        body=body,
        now=now,
    )
    return {
        "kind": "disk_horizon_warn",
        "days_until_full": round(days, 1),
        "free_gb": round(free_gb, 2),
        "total_gb": round(total_gb, 2),
        "alert_sent": sent,
    }


def _check_growth_trend(
    state: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    now: float,
) -> Optional[dict[str, Any]]:
    """Sustained week-over-week workspace growth > threshold gets a
    warning. Need ``_GROWTH_SUSTAINED_WEEKS`` consecutive weeks."""
    if len(samples) < _GROWTH_SUSTAINED_WEEKS + 1:
        return None
    recent = samples[-(_GROWTH_SUSTAINED_WEEKS + 1):]
    pcts = []
    for prev, cur in zip(recent, recent[1:]):
        prev_b = prev.get("workspace_bytes", 0)
        cur_b = cur.get("workspace_bytes", 0)
        if prev_b <= 0:
            return None
        pcts.append(100.0 * (cur_b - prev_b) / prev_b)
    if not all(p >= _GROWTH_WOW_WARN_PCT for p in pcts):
        return None
    latest_gb = recent[-1].get("workspace_bytes", 0) / 1024 / 1024 / 1024
    median_pct = sorted(pcts)[len(pcts) // 2]
    body = (
        f"📦 Workspace has grown ≥{_GROWTH_WOW_WARN_PCT:.0f}% week-over-"
        f"week for {_GROWTH_SUSTAINED_WEEKS} consecutive weeks "
        f"(median {median_pct:.1f}%).\n"
        f"  • current size: {latest_gb:.1f} GB\n\n"
        f"Likely causes: a leaky log writer, runaway index, "
        f"or accumulating jsonl without rotation. Check the per-"
        f"subdirectory size via 'du -sh workspace/*'."
    )
    sent = _alert_if_due(
        state,
        key="workspace_growth_burst",
        title="📦 Workspace growth burst",
        body=body,
        now=now,
    )
    return {
        "kind": "workspace_growth_burst",
        "median_wow_pct": round(median_pct, 1),
        "current_gb": round(latest_gb, 2),
        "alert_sent": sent,
    }


def _check_restart_burst(
    state: dict[str, Any],
    *,
    now: float,
) -> Optional[dict[str, Any]]:
    """Too many restarts in a short window = instability."""
    log = state.get("restart_log", [])
    if not isinstance(log, list):
        return None
    recent = [t for t in log if isinstance(t, (int, float)) and now - t <= _RESTART_BURST_WINDOW_S]
    if len(recent) < _RESTART_BURST_THRESHOLD:
        return None
    body = (
        f"🔁 Gateway has restarted {len(recent)} times in the last "
        f"{_RESTART_BURST_WINDOW_S // 3600}h. Possible OOM, panic, or "
        f"crash loop.\n\n"
        f"Check 'docker logs' / journal for the latest exit reasons. "
        f"The watchdog auto-restarts daemons but cannot restart the "
        f"gateway itself — repeated process-level death is operator-"
        f"actionable."
    )
    sent = _alert_if_due(
        state,
        key="restart_burst",
        title="🔁 Gateway restart burst",
        body=body,
        now=now,
    )
    return {
        "kind": "restart_burst",
        "n_restarts_24h": len(recent),
        "alert_sent": sent,
    }


def _check_uptime_stale(
    state: dict[str, Any],
    *,
    now: float,
) -> Optional[dict[str, Any]]:
    """Process uptime > N days without restart — likely running stale
    code. Restart is the lever to pick up self-applied amendments."""
    if _PROCESS_BOOT_AT is None:
        return None
    uptime_s = now - _PROCESS_BOOT_AT
    uptime_days = uptime_s / 86400
    if uptime_days < _UPTIME_STALE_DAYS:
        return None
    body = (
        f"⏰ Gateway process has been running for {uptime_days:.0f} days "
        f"without restart.\n\n"
        f"Self-applied amendments and hot-reload-incompatible "
        f"settings only take effect after a restart. Consider a "
        f"controlled restart at the next operator-convenient window."
    )
    sent = _alert_if_due(
        state,
        key="uptime_stale",
        title="⏰ Gateway uptime > 180 days",
        body=body,
        now=now,
    )
    return {
        "kind": "uptime_stale",
        "uptime_days": round(uptime_days, 1),
        "alert_sent": sent,
    }


def _check_memory_headroom(
    state: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    now: float,
) -> Optional[dict[str, Any]]:
    """Sustained MemAvailable < 10% of MemTotal. Requires
    ``_GROWTH_SUSTAINED_WEEKS`` consecutive weekly samples below
    threshold so a transient spike doesn't alert."""
    recent = samples[-_GROWTH_SUSTAINED_WEEKS:]
    if len(recent) < _GROWTH_SUSTAINED_WEEKS:
        return None
    pcts = [s.get("headroom_pct") for s in recent]
    if not all(isinstance(p, (int, float)) for p in pcts):
        return None
    if not all(p < _MEMORY_HEADROOM_WARN_PCT for p in pcts):
        return None
    latest_pct = pcts[-1]
    body = (
        f"🧠 Memory headroom has been < {_MEMORY_HEADROOM_WARN_PCT:.0f}% "
        f"for {_GROWTH_SUSTAINED_WEEKS} consecutive weeks "
        f"(latest {latest_pct:.1f}%).\n\n"
        f"Sustained pressure usually points to an unbounded cache "
        f"or a worker pool growing past its useful size. On macOS "
        f"Docker Desktop, this is the VM's view; if the actual "
        f"host's Activity Monitor disagrees, the VM allocation is "
        f"the bottleneck."
    )
    sent = _alert_if_due(
        state,
        key="memory_pressure",
        title="🧠 Sustained memory pressure",
        body=body,
        now=now,
    )
    return {
        "kind": "memory_pressure",
        "headroom_pct": round(latest_pct, 1),
        "alert_sent": sent,
    }


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """One probe pass. Daily-cadence wake-up; weekly-cadence
    sampling + trend evaluation. Returns a summary dict.

    Failure-isolated — never raises."""
    summary: dict[str, Any] = {
        "ran": False,
        "sampled": False,
        "n_samples": 0,
        "alerts": [],
        "host_metrics_present": False,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()

    # Always record restart on first probe of this lifetime.
    _record_restart(state, cur)

    last_run = float(state.get("last_run_at", 0))
    if last_run > 0 and cur - last_run < _WEEKLY_S:
        # Daily wake-up but not yet time for the weekly sample.
        # Still check fast signals: restart burst + uptime.
        state["last_run_at"] = cur  # record the daily probe
        fast_alerts: list[dict[str, Any]] = []
        burst = _check_restart_burst(state, now=cur)
        if burst is not None:
            fast_alerts.append(burst)
        stale = _check_uptime_stale(state, now=cur)
        if stale is not None:
            fast_alerts.append(stale)
        summary["ran"] = True
        summary["alerts"] = fast_alerts
        _write_state(state)
        return summary

    # Weekly sample + full trend evaluation.
    state["last_run_at"] = cur
    summary["ran"] = True
    summary["sampled"] = True

    sample = _gather_weekly_sample(cur)
    samples = state.setdefault("weekly_samples", [])
    if not isinstance(samples, list):
        samples = []
    samples.append(sample)
    if len(samples) > _MAX_HISTORY_SAMPLES:
        del samples[: len(samples) - _MAX_HISTORY_SAMPLES]
    state["weekly_samples"] = samples
    summary["n_samples"] = len(samples)
    summary["latest_sample"] = sample

    # Substrate-transition detection (Q16 Theme 1 follow-on, automated
    # ledger emission). Compare current fingerprint to previously-
    # observed one; on material divergence, emit the continuity-ledger
    # event + loud Signal alert. First run: just record.
    current_fp = _substrate_fingerprint()
    prior_fp = state.get("substrate_fingerprint")
    transition = _substrate_transition(prior_fp, current_fp)
    if transition is not None:
        _emit_substrate_migration(transition, current_fp)
        summary["substrate_transition"] = transition
    state["substrate_fingerprint"] = current_fp

    # Optional host-side companion telemetry.
    host_metrics = _read_host_metrics_tail()
    if host_metrics is not None:
        summary["host_metrics_present"] = True
        summary["host_metrics_latest"] = host_metrics

    alerts: list[dict[str, Any]] = []
    for check in (
        lambda: _check_disk_horizon(state, samples, now=cur),
        lambda: _check_growth_trend(state, samples, now=cur),
        lambda: _check_restart_burst(state, now=cur),
        lambda: _check_uptime_stale(state, now=cur),
        lambda: _check_memory_headroom(state, samples, now=cur),
    ):
        try:
            result = check()
        except Exception:
            logger.debug(
                "host_substrate_health: check raised", exc_info=True,
            )
            result = None
        if result is not None:
            alerts.append(result)

    summary["alerts"] = alerts
    _write_state(state)
    return summary
