"""
anomaly_detector.py — Statistical anomaly detection for agent system health.

Computes rolling averages and standard deviations for key metrics. Fires
alerts when any metric deviates > 2σ from its 24h rolling average.

No LLM calls — pure statistics on existing SQLite/metric data.
Called from the heartbeat loop (every 60s) and feeds anomaly events
into the self-healing layer and Firebase dashboard.

Based on: JISEM 2025 self-healing infrastructure patterns.
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

# M9: Cooldown tracking for auto-remediation (max 1 per hour per type)
_REMEDIATION_COOLDOWN = 3600  # 1 hour in seconds
_last_remediation: dict[str, float] = {}
_remediation_lock = threading.Lock()


def _check_remediation_cooldown(alert_type: str) -> bool:
    """Return True if auto-remediation is allowed (cooldown expired)."""
    now = time.monotonic()
    with _remediation_lock:
        last = _last_remediation.get(alert_type, 0)
        if now - last < _REMEDIATION_COOLDOWN:
            return False
        _last_remediation[alert_type] = now
        return True

logger = logging.getLogger(__name__)

# Rolling window: 24h of 60s samples = 1440 data points
_WINDOW_SIZE = 1440
_SIGMA_THRESHOLD = 2.0  # alert when > 2σ from mean
_WARN_SIGMA_THRESHOLD = 5.0  # only emit WARN for extreme deviations;
# 2–5σ is recorded in `_alerts` (visible to dashboard / API) and logged
# at INFO so the audit trail still shows it, but it doesn't pollute
# errors.jsonl with stat-noise that re-signs every time the rolling mean
# shifts (pattern_learner sees one signature per outlier — pure churn).
_MIN_SAMPLES = 30  # need at least 30 samples before alerting


@dataclass
class MetricWindow:
    """Rolling window of metric values with online mean/stddev."""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=_WINDOW_SIZE))

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def stddev(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        variance = sum((v - m) ** 2 for v in self.values) / len(self.values)
        return math.sqrt(variance)

    def add(self, value: float) -> None:
        self.values.append(value)

    def is_anomalous(self, value: float) -> bool:
        """Check if value is > 2σ from rolling mean."""
        if len(self.values) < _MIN_SAMPLES:
            return False
        sd = self.stddev
        if sd < 0.001:  # near-zero variance = stable metric
            return abs(value - self.mean) > 0.1  # use absolute threshold instead
        return abs(value - self.mean) > _SIGMA_THRESHOLD * sd


@dataclass
class AnomalyAlert:
    """A detected anomaly event."""
    metric: str
    current_value: float
    mean: float
    stddev: float
    sigma_distance: float
    direction: str  # "high" or "low"
    timestamp: str
    alert_type: str  # "high_error_rate", "slow_response", "quality_drift", "token_spike"


# Global metric windows
_windows: dict[str, MetricWindow] = {}
_alerts: deque[AnomalyAlert] = deque(maxlen=100)
_lock = threading.Lock()


def _get_window(name: str) -> MetricWindow:
    if name not in _windows:
        _windows[name] = MetricWindow(name=name)
    return _windows[name]


def record_sample(metric_name: str, value: float) -> AnomalyAlert | None:
    """Record a metric sample and return an alert if anomalous."""
    with _lock:
        window = _get_window(metric_name)
        is_anom = window.is_anomalous(value)
        window.add(value)

        if not is_anom:
            return None

        sd = window.stddev
        sigma_dist = abs(value - window.mean) / sd if sd > 0.001 else 0
        direction = "high" if value > window.mean else "low"

        # Map metric names to alert types
        alert_type_map = {
            "error_rate_1h": "high_error_rate",
            "avg_response_time": "slow_response",
            "output_quality": "quality_drift",
            "token_usage_1h": "token_spike",
        }
        alert_type = alert_type_map.get(metric_name, f"anomaly_{metric_name}")

        alert = AnomalyAlert(
            metric=metric_name,
            current_value=round(value, 4),
            mean=round(window.mean, 4),
            stddev=round(sd, 4),
            sigma_distance=round(sigma_dist, 2),
            direction=direction,
            timestamp=datetime.now(timezone.utc).isoformat(),
            alert_type=alert_type,
        )
        _alerts.append(alert)
        # Log level is sigma-aware: extreme deviations (≥5σ) WARN so they
        # surface in errors.jsonl + Signal feed; routine 2–5σ noise INFOs
        # (still in _alerts for dashboard, but not in the error stream).
        msg = (
            f"ANOMALY: {metric_name}={value:.4f} "
            f"(mean={window.mean:.4f} ±{sd:.4f}, {sigma_dist:.1f}σ {direction})"
        )
        if sigma_dist >= _WARN_SIGMA_THRESHOLD:
            logger.warning(msg)
        else:
            logger.info(msg)
        return alert


def collect_and_check() -> list[AnomalyAlert]:
    """Collect current metrics and check for anomalies. Called from heartbeat.

    Returns list of new alerts (empty if all normal).
    """
    alerts = []

    # Error rate (last 1 hour)
    try:
        from app.metrics import _error_rate_1h
        alert = record_sample("error_rate_1h", _error_rate_1h())
        if alert:
            alerts.append(alert)
    except Exception:
        pass

    # Average response time
    try:
        from app.metrics import _avg_response_time
        resp = _avg_response_time()
        if resp > 0:
            alert = record_sample("avg_response_time", resp)
            if alert:
                alerts.append(alert)
    except Exception:
        pass

    # Output quality (from self-reports)
    try:
        from app.metrics import _output_quality_score
        alert = record_sample("output_quality", _output_quality_score())
        if alert:
            alerts.append(alert)
    except Exception:
        pass

    return alerts


def handle_alerts(alerts: list[AnomalyAlert]) -> None:
    """Dispatch anomaly alerts to appropriate handlers.

    - high_error_rate → trigger error resolution immediately
    - quality_drift → trigger retrospective
    - All alerts → push to Firebase for dashboard
    """
    for alert in alerts:
        # Push to Firebase
        try:
            from app.firebase_reporter import _fire, _get_db, _now_iso
            def _report(a=alert):
                db = _get_db()
                if not db:
                    return
                db.collection("activities").add({
                    "ts": _now_iso(),
                    "event": "anomaly_detected",
                    "crew": "system",
                    "detail": f"⚠️ {a.alert_type}: {a.metric}={a.current_value} ({a.sigma_distance}σ {a.direction})",
                })
            _fire(_report)
        except Exception:
            pass

        # M9: Rate-limit auto-remediation — max 1 per hour per type
        _auto_remediation_allowed = _check_remediation_cooldown(alert.alert_type)
        if not _auto_remediation_allowed:
            logger.info(f"Anomaly handler: skipping auto-remediation (cooldown) for {alert.alert_type}")
            return alert

        # Trigger appropriate handler
        if alert.alert_type == "high_error_rate" and alert.direction == "high":
            try:
                from app.auditor import run_error_resolution
                logger.info("Anomaly handler: triggering error resolution")
                threading.Thread(
                    target=run_error_resolution, daemon=True,
                    name="anomaly-error-resolve"
                ).start()
            except Exception:
                logger.debug("Anomaly handler: error resolution trigger failed", exc_info=True)

        elif alert.alert_type == "quality_drift" and alert.direction == "low":
            try:
                from app.crews.retrospective_crew import RetrospectiveCrew
                logger.info("Anomaly handler: triggering retrospective analysis")
                threading.Thread(
                    target=RetrospectiveCrew().run, daemon=True,
                    name="anomaly-retrospective"
                ).start()
            except Exception:
                logger.debug("Anomaly handler: retrospective trigger failed", exc_info=True)


def get_recent_alerts(n: int = 10) -> list[dict]:
    """Return recent alerts for display."""
    with _lock:
        return [
            {
                "metric": a.metric,
                "value": a.current_value,
                "mean": a.mean,
                "sigma": a.sigma_distance,
                "direction": a.direction,
                "type": a.alert_type,
                "ts": a.timestamp,
            }
            for a in list(_alerts)[-n:]
        ]
