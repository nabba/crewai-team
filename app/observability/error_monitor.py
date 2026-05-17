"""
error_monitor.py — Permanent error anomaly monitor for the React dashboard.

Tails ``workspace/logs/errors.jsonl`` every 5 minutes, groups errors by
a normalized signature (logger module + cleaned message prefix), and
detects three classes of anomaly:

  1. ``new_pattern`` — signature first observed in the last 60 minutes
                       AND occurring > 5× in that window
  2. ``rate_spike``  — signature's 1h rate > 3× its 24h rolling average
                       AND > 5/hour absolute
  3. ``total_rate``  — total errors/hour deviates > 2σ from the 24h
                       rolling mean (delegated to ``app.anomaly_detector``)

State persistence:
  * Byte offset of last-read errors.jsonl is stored at
    ``workspace/observability/error_monitor_state.json`` so each scan is
    incremental — full re-reads of the 21 MB log are wasteful and there
    is no semantic value in re-detecting historical anomalies.
  * Detected anomalies persist to ``control_plane.error_anomalies`` so
    open alerts survive gateway restarts.

The monitor is strictly read-only on errors.jsonl and only writes to its
own state file + the new anomalies table. It touches none of the
protected subsystems (SubIA, beliefs, MAP-Elites, affective layer, KBs).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────────

ERRORS_LOG_PATH = Path("/app/workspace/logs/errors.jsonl")
STATE_PATH = Path("/app/workspace/observability/error_monitor_state.json")

# Rolling window cap — events older than this are evicted from memory.
WINDOW_HOURS = 24
# OOM guard: even if the log floods, never hold more than this many events
# in the in-memory window.
MAX_EVENTS_IN_MEMORY = 50_000

# Detection thresholds.
SPIKE_RATIO = 3.0           # 1h rate must exceed this × 24h average to spike
SPIKE_MIN_HOURLY = 5        # absolute floor — ignore noise on very-rare patterns
NEW_PATTERN_WINDOW_MIN = 60 # signature unseen before this window = "new"
NEW_PATTERN_MIN_COUNT = 5   # plus must occur this many times to flag

# Auto-resolution.
AUTO_RESOLVE_RATIO = 0.5    # if rate falls below this × detection threshold
AUTO_RESOLVE_HOURS = 2      # for this many consecutive hours → resolve

# Severity ladder (ratio of current vs baseline).
SEVERITY_INFO_RATIO = 3.0
SEVERITY_WARN_RATIO = 5.0
SEVERITY_CRIT_RATIO = 10.0

# ── Pattern signature ───────────────────────────────────────────────────────

# Strip varying parts so structurally-identical errors group.
_STRIP_PATTERNS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\S*\b"), "<ts>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    (re.compile(r"'[^']{0,200}'"), "'<str>'"),
    (re.compile(r'"[^"]{0,200}"'), '"<str>"'),
    (re.compile(r"/[\w./-]+\.(?:py|md|json|yaml|sql)\b"), "<path>"),
]


def _signature(record: dict) -> tuple[str, str]:
    """Return ``(signature_hash, sample_message)`` for an error record.

    The signature is stable across structurally-identical errors;
    the sample preserves the first occurrence's actual text for context
    in the dashboard.
    """
    module = (record.get("module") or record.get("logger") or "unknown").lower()
    msg = (record.get("message") or record.get("msg") or "").strip()
    sample = msg[:300]
    # Normalize message for signature.
    norm = msg[:200]
    for pat, repl in _STRIP_PATTERNS:
        norm = pat.sub(repl, norm)
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    raw = f"{module}::{norm[:120]}"
    sig = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return sig, sample


# ── State persistence ───────────────────────────────────────────────────────

_state_lock = threading.Lock()
_window_lock = threading.Lock()

# In-memory rolling window: signature → deque[(unix_ts, sample)].
_window: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_EVENTS_IN_MEMORY // 100))
# Total events across all signatures — capped separately so a single hot
# signature can't consume the whole budget.
_total_events: deque = deque(maxlen=MAX_EVENTS_IN_MEMORY)
# Signatures we've ever seen (since process start) — persisted via state file.
_known_signatures: set[str] = set()
# Auto-resolution tracking: signature → (consecutive hours below threshold).
_below_threshold_streak: dict[str, int] = defaultdict(int)
# Warm-up flag: after process start the in-memory window is empty even if
# errors.jsonl has 24h of history. The first scan back-fills the window so
# the dashboard has data immediately and rate-spike detection has a baseline.
_warmup_done = False


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"offset": 0, "known_signatures": []}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"offset": 0, "known_signatures": []}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


# ── Warm-up (back-fill the rolling window on process start) ────────────────

# Read this many bytes from the end of errors.jsonl on first scan after
# process start. ~5 MB typically covers > 24h of records on this system
# (errors.jsonl grows ~1-2 MB/day in steady state).
_WARMUP_TAIL_BYTES = 5 * 1024 * 1024


def _warmup_window() -> int:
    """Back-fill the in-memory window from the tail of errors.jsonl.

    Runs once per process start. Without this, after a restart the window
    is empty until new errors arrive — the dashboard would show zeros for
    the first ~5 minutes of every restart, and rate-spike detection would
    have no baseline to compare against.

    Returns the number of records ingested during warmup.
    """
    if not ERRORS_LOG_PATH.exists():
        return 0
    size = ERRORS_LOG_PATH.stat().st_size
    backward = min(size, _WARMUP_TAIL_BYTES)
    try:
        with ERRORS_LOG_PATH.open("rb") as f:
            f.seek(size - backward)
            chunk = f.read(backward)
    except Exception as exc:
        logger.debug(f"error_monitor: warmup read failed: {exc}")
        return 0
    text = chunk.decode("utf-8", errors="replace")
    # Drop the first partial line (we seeked into the middle of one).
    lines = text.split("\n")
    if backward < size and lines:
        lines = lines[1:]
    ingested = _ingest_records(lines)
    # Mark every signature currently in the window as historically known
    # so the very next detection pass doesn't fire spurious "new pattern"
    # alerts on signatures that have actually been around for hours/days.
    with _window_lock:
        _known_signatures.update(_window.keys())
    logger.info(
        f"error_monitor: warmup back-filled {ingested} records, "
        f"{len(_window)} signatures"
    )
    return ingested


# ── Tailing errors.jsonl ────────────────────────────────────────────────────

def _read_new_lines(state: dict) -> tuple[list[str], int]:
    """Return (new_lines, new_offset). Handles log rotation by detecting
    file-size shrink and starting from 0.
    """
    if not ERRORS_LOG_PATH.exists():
        return [], state.get("offset", 0)
    size = ERRORS_LOG_PATH.stat().st_size
    offset = state.get("offset", 0)
    if offset > size:
        # File truncated / rotated — restart from beginning.
        offset = 0
    if offset == size:
        return [], offset
    with ERRORS_LOG_PATH.open("rb") as f:
        f.seek(offset)
        chunk = f.read(size - offset)
    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        return [], size
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return lines, size


def _ingest_records(lines: list[str]) -> int:
    """Parse and add error records to the rolling window. Returns count
    of valid records ingested.
    """
    now = time.time()
    cutoff = now - WINDOW_HOURS * 3600
    ingested = 0
    with _window_lock:
        # Evict stale entries from total window first.
        while _total_events and _total_events[0][0] < cutoff:
            _total_events.popleft()
        for ln in lines:
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            ts_str = rec.get("ts") or rec.get("timestamp") or rec.get("time")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() if ts_str else now
            except Exception:
                ts = now
            if ts < cutoff:
                continue
            sig, sample = _signature(rec)
            _total_events.append((ts, sig))
            _window[sig].append((ts, sample))
            # Cap per-signature deque so a single hot signature can't blow memory.
            while len(_window[sig]) > 2000:
                _window[sig].popleft()
            ingested += 1
    return ingested


# ── Detection ───────────────────────────────────────────────────────────────

def _hourly_rate(sig: str, hours: int = 1) -> int:
    """Count events for ``sig`` in the last ``hours`` hours."""
    cutoff = time.time() - hours * 3600
    with _window_lock:
        return sum(1 for ts, _ in _window.get(sig, ()) if ts >= cutoff)


def _avg_hourly_rate_24h(sig: str) -> float:
    """Average hourly rate over the rolling 24h window."""
    with _window_lock:
        n = len(_window.get(sig, ()))
    return n / WINDOW_HOURS


def _is_first_seen_within(sig: str, minutes: int) -> bool:
    """True if the earliest in-window event for ``sig`` is more recent
    than ``minutes`` ago. Combined with the persisted known-set, this
    answers "is this signature genuinely new?"
    """
    if sig in _known_signatures:
        return False
    cutoff = time.time() - minutes * 60
    with _window_lock:
        events = _window.get(sig, ())
        if not events:
            return False
        return events[0][0] >= cutoff


def _severity_for_ratio(ratio: float) -> str:
    if ratio >= SEVERITY_CRIT_RATIO: return "critical"
    if ratio >= SEVERITY_WARN_RATIO: return "warning"
    return "info"


def _open_anomaly_exists(sig: str) -> bool:
    """Has this signature already got an open anomaly?  Keeps the table
    de-duplicated so a sustained spike doesn't flood it.
    """
    try:
        from app.control_plane.db import execute_one
        row = execute_one(
            "SELECT 1 FROM control_plane.error_anomalies "
            "WHERE pattern_signature = %s AND status = 'open' LIMIT 1",
            (sig,),
        )
        return row is not None
    except Exception:
        return False


def _record_anomaly(
    sig: str, sample: str, anomaly_type: str,
    hourly_rate: float, baseline: float, severity: str,
) -> None:
    # PR 3 (2026-05-16): INSERT uses ``execute_required`` so a DB
    # failure now fires the WARNING log (was DEBUG/silent under the
    # old ``execute`` which swallowed exceptions). Without this, an
    # anomaly that fires the runbook below could leave no row in
    # ``error_anomalies`` — operators inspecting the table would
    # see "nothing wrong" while the runbook had already attempted
    # remediation.
    try:
        from app.control_plane.db import execute_required
        execute_required(
            """
            INSERT INTO control_plane.error_anomalies
                (pattern_signature, pattern_sample, anomaly_type,
                 severity, hourly_rate, baseline_rate)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (sig, sample[:1000], anomaly_type, severity,
             round(hourly_rate, 4), round(baseline, 4) if baseline else None),
        )
        logger.info(
            "error_monitor: anomaly logged — type=%s sev=%s sig=%s rate=%.1f baseline=%.1f",
            anomaly_type, severity, sig, hourly_rate, baseline,
        )
    except Exception as exc:
        logger.warning(f"error_monitor: anomaly insert failed: {exc}")

    # Hook: dispatch a runbook if one is registered for this signature.
    # No-op when ``ERROR_RUNBOOKS_ENABLED`` is unset. Wrapped so a
    # runbook bug can never break anomaly recording.
    try:
        from app.healing.runbooks import maybe_run_runbook
        maybe_run_runbook({
            "pattern_signature": sig,
            "pattern_sample": sample[:1000],
            "anomaly_type": anomaly_type,
            "severity": severity,
            "hourly_rate": hourly_rate,
            "baseline": baseline,
        })
    except Exception as exc:
        logger.debug(f"error_monitor: runbook dispatch hook failed: {exc}")


def _auto_resolve_open_anomalies() -> int:
    """For each open anomaly, if its signature's current rate has been
    below ``AUTO_RESOLVE_RATIO × SPIKE_MIN_HOURLY`` for ``AUTO_RESOLVE_HOURS``
    consecutive checks, mark it resolved.
    """
    threshold = AUTO_RESOLVE_RATIO * SPIKE_MIN_HOURLY
    resolved = 0
    try:
        from app.control_plane.db import execute
        rows = execute(
            "SELECT id, pattern_signature FROM control_plane.error_anomalies WHERE status = 'open'",
            fetch=True,
        ) or []
        for row in rows:
            sig = row.get("pattern_signature") if isinstance(row, dict) else row[1]
            anomaly_id = row.get("id") if isinstance(row, dict) else row[0]
            current = _hourly_rate(sig, hours=1)
            if current < threshold:
                _below_threshold_streak[sig] += 1
                if _below_threshold_streak[sig] >= AUTO_RESOLVE_HOURS:
                    execute(
                        "UPDATE control_plane.error_anomalies "
                        "SET status='resolved', resolved_at=NOW() "
                        "WHERE id = %s AND status='open'",
                        (anomaly_id,),
                    )
                    _below_threshold_streak.pop(sig, None)
                    resolved += 1
            else:
                _below_threshold_streak.pop(sig, None)
    except Exception as exc:
        logger.debug(f"error_monitor: auto-resolve scan failed: {exc}")
    return resolved


# ── Public API: scan() and snapshot() ───────────────────────────────────────

def scan() -> dict:
    """One pass: read new lines, ingest, run detectors, persist anomalies.

    Returns a small dict for logging / tests. Safe to call concurrently
    (state file write is atomic via tmp + rename; detectors use an internal
    lock).
    """
    global _warmup_done
    with _state_lock:
        state = _load_state()
        # Restore known-signatures from prior runs.
        if not _known_signatures:
            _known_signatures.update(state.get("known_signatures", []))
        # First scan after process start: back-fill the in-memory window
        # so the dashboard isn't empty and detectors have a baseline.
        if not _warmup_done:
            _warmup_window()
            _warmup_done = True
        lines, new_offset = _read_new_lines(state)
        ingested = _ingest_records(lines) if lines else 0

        # Detection: walk every signature seen in the rolling window.
        new_anomalies = 0
        with _window_lock:
            sigs_in_window = list(_window.keys())
        for sig in sigs_in_window:
            if _open_anomaly_exists(sig):
                continue
            current_hr = _hourly_rate(sig, hours=1)
            avg_hr = _avg_hourly_rate_24h(sig)
            sample = ""
            with _window_lock:
                events = _window.get(sig, ())
                if events:
                    sample = events[-1][1]
            # Rule 1: new pattern
            if current_hr >= NEW_PATTERN_MIN_COUNT and _is_first_seen_within(sig, NEW_PATTERN_WINDOW_MIN):
                ratio = current_hr / max(avg_hr, 0.1)
                _record_anomaly(sig, sample, "new_pattern", current_hr, avg_hr,
                                _severity_for_ratio(ratio))
                new_anomalies += 1
            # Rule 2: rate spike (only for already-known signatures with > min hourly)
            elif current_hr >= SPIKE_MIN_HOURLY and avg_hr > 0:
                ratio = current_hr / avg_hr
                if ratio >= SPIKE_RATIO and sig in _known_signatures:
                    _record_anomaly(sig, sample, "rate_spike", current_hr, avg_hr,
                                    _severity_for_ratio(ratio))
                    new_anomalies += 1
            _known_signatures.add(sig)

        # Rule 3: total-rate σ-anomaly via existing engine.
        try:
            from app.anomaly_detector import _get_window as _ad_window, AnomalyAlert
            with _window_lock:
                total_last_hr = sum(1 for ts, _ in _total_events if ts >= time.time() - 3600)
            ad = _ad_window("errors_per_hour")
            if ad.is_anomalous(total_last_hr):
                # Reuse the existing anomaly_detector alert deque so it
                # surfaces via the existing /api/cp/anomalies endpoint.
                from app.anomaly_detector import _alerts as _ad_alerts
                _ad_alerts.append(AnomalyAlert(
                    metric="errors_per_hour",
                    current_value=float(total_last_hr),
                    mean=ad.mean,
                    stddev=ad.stddev,
                    sigma_distance=(total_last_hr - ad.mean) / max(ad.stddev, 0.001),
                    direction="high" if total_last_hr > ad.mean else "low",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    alert_type="total_error_rate",
                ))
            ad.add(float(total_last_hr))
        except Exception as exc:
            logger.debug(f"error_monitor: total-rate σ check skipped: {exc}")

        resolved = _auto_resolve_open_anomalies()

        # Persist state.
        _save_state({
            "offset": new_offset,
            "known_signatures": sorted(_known_signatures)[-5000:],  # cap memory
            "last_scan": datetime.now(timezone.utc).isoformat(),
        })

    return {
        "ingested": ingested,
        "new_anomalies": new_anomalies,
        "resolved": resolved,
        "total_signatures": len(_known_signatures),
    }


def snapshot() -> dict:
    """Aggregated view for the GET /api/cp/error_audit endpoint.

    Returns a single payload the React component renders directly.
    """
    now = time.time()
    h_buckets: dict[int, int] = defaultdict(int)  # hour-of-day → count
    pattern_counts: dict[str, int] = defaultdict(int)
    pattern_samples: dict[str, str] = {}
    total_24h = 0
    total_1h = 0

    with _window_lock:
        cutoff_24h = now - 24 * 3600
        cutoff_1h = now - 3600
        for ts, sig in _total_events:
            if ts < cutoff_24h:
                continue
            total_24h += 1
            if ts >= cutoff_1h:
                total_1h += 1
            # Bucket by floor-of-hour timestamp for the trend chart.
            bucket = int(ts // 3600) * 3600
            h_buckets[bucket] += 1
            pattern_counts[sig] += 1
        for sig in pattern_counts:
            events = _window.get(sig, ())
            if events:
                pattern_samples[sig] = events[-1][1]

    top_patterns = sorted(pattern_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top_patterns_out = [{
        "signature": sig,
        "sample": pattern_samples.get(sig, "")[:200],
        "count": cnt,
        "share_pct": round(100.0 * cnt / max(total_24h, 1), 2),
    } for sig, cnt in top_patterns]

    trend = [
        {"hour": datetime.fromtimestamp(b, timezone.utc).isoformat(), "count": c}
        for b, c in sorted(h_buckets.items())
    ]

    active = []
    try:
        from app.control_plane.db import execute
        rows = execute(
            """
            SELECT id, pattern_signature, pattern_sample, anomaly_type, severity,
                   hourly_rate, baseline_rate, detected_at
            FROM control_plane.error_anomalies
            WHERE status = 'open'
            ORDER BY detected_at DESC
            LIMIT 50
            """,
            fetch=True,
        ) or []
        for r in rows:
            d = dict(r) if not isinstance(r, dict) else r
            active.append({
                "id": str(d.get("id", "")),
                "signature": d.get("pattern_signature", ""),
                "sample": (d.get("pattern_sample") or "")[:300],
                "type": d.get("anomaly_type", ""),
                "severity": d.get("severity", "info"),
                "hourly_rate": float(d.get("hourly_rate") or 0),
                "baseline_rate": float(d.get("baseline_rate") or 0),
                "detected_at": d.get("detected_at").isoformat() if d.get("detected_at") else None,
            })
    except Exception as exc:
        logger.debug(f"error_monitor: snapshot anomalies query failed: {exc}")

    hourly_avg = total_24h / 24.0
    trend_dir = "rising" if total_1h > hourly_avg * 1.3 else "falling" if total_1h < hourly_avg * 0.7 else "stable"

    return {
        "summary": {
            "total_24h": total_24h,
            "total_1h": total_1h,
            "hourly_avg_24h": round(hourly_avg, 2),
            "trend": trend_dir,
        },
        "top_patterns_24h": top_patterns_out,
        "trend_hourly": trend,
        "active_anomalies": active,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def acknowledge(anomaly_id: str) -> bool:
    """Mark an open anomaly as acknowledged (manual silence; preserves
    history). Returns True on success, False on DB failure.

    PR 3 (2026-05-16): UPDATE uses ``execute_required`` so a DB
    failure now returns False (was previously returning True
    because ``execute`` silently swallowed exceptions). The operator
    clicking "acknowledge" now actually gets the right answer when
    the DB is unreachable.
    """
    try:
        from app.control_plane.db import execute_required
        execute_required(
            "UPDATE control_plane.error_anomalies "
            "SET status='acknowledged' "
            "WHERE id = %s::uuid AND status='open'",
            (anomaly_id,),
        )
        return True
    except Exception as exc:
        logger.warning(f"error_monitor: acknowledge failed: {exc}")
        return False
