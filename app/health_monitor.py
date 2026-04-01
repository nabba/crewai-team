"""
health_monitor.py — Continuous health monitoring with dimensional tracking.

Tracks per-interaction operational metrics across six dimensions:
  - error_rate: task failures / total tasks
  - avg_latency_ms: mean response time
  - hallucination_rate: vetting rejections / total vetted
  - cascade_fallback_rate: premium tier usage for simple tasks
  - memory_retrieval_accuracy: memory hits / total queries
  - safety_violations: count of safety-related issues

Uses a sliding window (last 100 interactions) for aggregation.
Threshold-based alerting with three severity levels (warning, critical, emergency).

IMMUTABLE — infrastructure-level module.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── Per-interaction metrics ──────────────────────────────────────────────────


@dataclass
class InteractionMetrics:
    """Metrics captured from a single user interaction."""
    timestamp: float = 0.0  # monotonic
    task_id: str = ""
    sender_id: str = ""

    # Core dimensions
    success: bool = True         # did the task complete without error?
    latency_ms: float = 0.0     # total response time in milliseconds
    vetted: bool = False         # was the response vetted?
    vetting_passed: bool = True  # did vetting pass?
    model_tier: str = ""         # "budget", "mid", "premium", "local"
    task_difficulty: int = 3     # 1-10 estimated difficulty
    used_premium: bool = False   # hit premium tier?
    memory_queried: bool = False # was memory/RAG used?
    memory_hit: bool = False     # did memory return useful results?
    safety_issue: bool = False   # any safety-related issue detected?

    # Optional detail
    crew_used: str = ""
    error_type: str = ""


# ── Aggregated health state ─────────────────────────────────────────────────


@dataclass
class HealthState:
    """Aggregated health across all dimensions."""
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    hallucination_rate: float = 0.0
    cascade_fallback_rate: float = 0.0
    memory_retrieval_accuracy: float = 1.0
    safety_violations: int = 0
    sample_size: int = 0
    window_start: str = ""
    window_end: str = ""


# ── Health alerts ────────────────────────────────────────────────────────────


@dataclass
class HealthAlert:
    """A single health threshold violation."""
    severity: str  # "warning", "critical", "emergency"
    dimension: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    auto_remediate: bool = True
    message: str = ""


# ── IMMUTABLE thresholds ─────────────────────────────────────────────────────

THRESHOLDS = {
    "error_rate": {
        "warning": 0.05,
        "critical": 0.15,
        "emergency": 0.30,
    },
    "avg_latency_ms": {
        "warning": 5000,
        "critical": 10000,
        "emergency": 30000,
    },
    "hallucination_rate": {
        "warning": 0.05,
        "critical": 0.10,
        "emergency": 0.20,
    },
    "cascade_fallback_rate": {
        "warning": 0.30,
        "critical": 0.50,
        "emergency": 0.70,
    },
    "memory_retrieval_accuracy": {
        # Inverted: LOWER is worse
        "warning": 0.85,
        "critical": 0.70,
        "emergency": 0.50,
    },
    "safety_violations": {
        "warning": 1,
        "critical": 1,
        "emergency": 1,  # Any safety violation is emergency
    },
}

# Dimensions where lower = worse (inverted comparison)
_INVERTED_DIMENSIONS = {"memory_retrieval_accuracy"}

# Minimum interactions before alerting (avoid noise from small samples)
MIN_SAMPLE_SIZE = 10


# ── Health Monitor ───────────────────────────────────────────────────────────


class HealthMonitor:
    """Continuous health monitoring with dimensional tracking and alerting."""

    def __init__(self, window_size: int = 100):
        self._window: deque[InteractionMetrics] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._alert_callbacks: list = []
        self._last_alerts: list[HealthAlert] = []
        self._alert_cooldown: dict[str, float] = {}  # dimension → last alert time
        self._cooldown_seconds = 300  # 5 minutes between alerts per dimension

    def record(self, metrics: InteractionMetrics) -> None:
        """Record metrics from a single interaction."""
        if metrics.timestamp == 0:
            metrics.timestamp = time.monotonic()
        with self._lock:
            self._window.append(metrics)

    def evaluate(self) -> list[HealthAlert]:
        """Evaluate current health against IMMUTABLE thresholds.

        Returns list of alerts (empty if healthy).
        """
        with self._lock:
            if len(self._window) < MIN_SAMPLE_SIZE:
                return []
            # Copy for thread safety
            window = list(self._window)

        state = self._aggregate(window)
        alerts = []
        now = time.monotonic()

        for dimension, thresholds in THRESHOLDS.items():
            value = getattr(state, dimension, None)
            if value is None:
                continue

            # Check cooldown
            last_alert = self._alert_cooldown.get(dimension, 0)
            if (now - last_alert) < self._cooldown_seconds:
                continue

            # Check thresholds from most severe to least
            for severity in ("emergency", "critical", "warning"):
                threshold = thresholds[severity]

                if dimension in _INVERTED_DIMENSIONS:
                    triggered = value < threshold
                elif dimension == "safety_violations":
                    triggered = value >= threshold
                else:
                    triggered = value > threshold

                if triggered:
                    alert = HealthAlert(
                        severity=severity,
                        dimension=dimension,
                        current_value=value,
                        threshold=threshold,
                        auto_remediate=(severity != "emergency"),
                        message=(
                            f"{dimension}: {value:.3f} "
                            f"{'<' if dimension in _INVERTED_DIMENSIONS else '>'} "
                            f"{threshold} ({severity})"
                        ),
                    )
                    alerts.append(alert)
                    self._alert_cooldown[dimension] = now
                    break  # Only highest severity per dimension

        self._last_alerts = alerts

        # Fire callbacks
        if alerts:
            for cb in self._alert_callbacks:
                try:
                    cb(alerts)
                except Exception:
                    pass

        return alerts

    def get_health_state(self) -> HealthState:
        """Return current aggregated health state."""
        with self._lock:
            if len(self._window) < 1:
                return HealthState()
            window = list(self._window)
        return self._aggregate(window)

    def get_recent_alerts(self) -> list[HealthAlert]:
        """Return alerts from last evaluation."""
        return self._last_alerts

    def on_alert(self, callback) -> None:
        """Register a callback for health alerts."""
        self._alert_callbacks.append(callback)

    def _aggregate(self, window: list[InteractionMetrics]) -> HealthState:
        """Aggregate a window of metrics into a health state."""
        n = len(window)
        if n == 0:
            return HealthState()

        errors = sum(1 for m in window if not m.success)
        total_latency = sum(m.latency_ms for m in window)
        vetted = [m for m in window if m.vetted]
        vetting_fails = sum(1 for m in vetted if not m.vetting_passed)
        simple_tasks = [m for m in window if m.task_difficulty <= 4]
        premium_on_simple = sum(1 for m in simple_tasks if m.used_premium)
        memory_queries = [m for m in window if m.memory_queried]
        memory_hits = sum(1 for m in memory_queries if m.memory_hit)
        safety = sum(1 for m in window if m.safety_issue)

        return HealthState(
            error_rate=errors / n,
            avg_latency_ms=total_latency / n,
            hallucination_rate=(vetting_fails / len(vetted)) if vetted else 0.0,
            cascade_fallback_rate=(
                (premium_on_simple / len(simple_tasks)) if simple_tasks else 0.0
            ),
            memory_retrieval_accuracy=(
                (memory_hits / len(memory_queries)) if memory_queries else 1.0
            ),
            safety_violations=safety,
            sample_size=n,
            window_start=datetime.fromtimestamp(
                window[0].timestamp if window[0].timestamp > 1e9 else time.time(),
                tz=timezone.utc,
            ).isoformat() if window else "",
            window_end=datetime.fromtimestamp(
                window[-1].timestamp if window[-1].timestamp > 1e9 else time.time(),
                tz=timezone.utc,
            ).isoformat() if window else "",
        )

    def format_health_report(self) -> str:
        """Generate human-readable health report."""
        state = self.get_health_state()
        if state.sample_size == 0:
            return "No interaction data yet."

        def _status(dim: str, val: float) -> str:
            thresholds = THRESHOLDS.get(dim, {})
            inverted = dim in _INVERTED_DIMENSIONS
            for sev in ("emergency", "critical", "warning"):
                t = thresholds.get(sev, 0)
                if inverted and val < t:
                    return f"{'🔴' if sev == 'emergency' else '🟡' if sev == 'critical' else '🟠'} {sev}"
                elif not inverted and dim != "safety_violations" and val > t:
                    return f"{'🔴' if sev == 'emergency' else '🟡' if sev == 'critical' else '🟠'} {sev}"
                elif dim == "safety_violations" and val >= t:
                    return f"🔴 {sev}"
            return "🟢 healthy"

        lines = [
            "📊 Health Monitor Report",
            f"   Sample: {state.sample_size} interactions",
            "",
            f"   Error rate:        {state.error_rate:.1%}  {_status('error_rate', state.error_rate)}",
            f"   Avg latency:       {state.avg_latency_ms:.0f}ms  {_status('avg_latency_ms', state.avg_latency_ms)}",
            f"   Hallucination:     {state.hallucination_rate:.1%}  {_status('hallucination_rate', state.hallucination_rate)}",
            f"   Cascade fallback:  {state.cascade_fallback_rate:.1%}  {_status('cascade_fallback_rate', state.cascade_fallback_rate)}",
            f"   Memory accuracy:   {state.memory_retrieval_accuracy:.1%}  {_status('memory_retrieval_accuracy', state.memory_retrieval_accuracy)}",
            f"   Safety violations: {state.safety_violations}  {_status('safety_violations', state.safety_violations)}",
        ]
        return "\n".join(lines)


# ── Module-level singleton ───────────────────────────────────────────────────

_monitor: HealthMonitor | None = None
_monitor_lock = threading.Lock()


def get_monitor() -> HealthMonitor:
    """Get or create the singleton health monitor."""
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = HealthMonitor()
        return _monitor


def record_interaction(metrics: InteractionMetrics) -> None:
    """Convenience: record metrics on the singleton monitor."""
    get_monitor().record(metrics)


def evaluate_health() -> list[HealthAlert]:
    """Convenience: evaluate health on the singleton monitor."""
    return get_monitor().evaluate()
