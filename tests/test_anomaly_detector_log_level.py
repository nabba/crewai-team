"""Regression: ANOMALY: messages must be sigma-aware in their log level.

Pre-fix shape (the operator-reported bug):

  Every 2σ deviation logged at WARNING, including 2.1σ, 2.7σ, 4.1σ —
  all routine statistical noise. errors.jsonl filled with hundreds of
  ANOMALY: avg_response_time=… messages per week. Each new outlier
  generated a new signature (because the f-string includes the actual
  numeric value), so pattern_learner saw 10+ "uncovered" patterns
  that were really one stat-detector.

Post-fix:
  • ≥5σ (extreme) → WARNING (visible in errors.jsonl + Signal alert)
  •  <5σ (routine) → INFO (still in _alerts for the dashboard, but
    not in the error stream; pattern_learner ignores INFO)

The alert object is appended to `_alerts` regardless of sigma — that's
the source of truth for the dashboard. We only changed the log
verbosity for the human-visible error stream.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def fresh_detector():
    """Start each test with empty windows so prior samples don't pollute
    the rolling baseline."""
    from app import anomaly_detector
    anomaly_detector._windows.clear()
    anomaly_detector._alerts.clear()
    yield anomaly_detector
    anomaly_detector._windows.clear()
    anomaly_detector._alerts.clear()


def _seed_baseline(det, metric: str, value: float, n: int) -> None:
    """Push n samples of the same value so mean is `value` and stddev is
    near zero. Then a deviation of `delta` registers as ~delta/0.001σ
    (effectively unbounded), which the detector treats specially via
    the absolute-threshold branch. To get a defined sigma_dist we
    seed with a small spread instead."""
    import random
    rng = random.Random(42)
    for _ in range(n):
        det.record_sample(metric, value + rng.uniform(-1.0, 1.0))


class TestSigmaAwareLogLevel:

    def test_low_sigma_emits_info_not_warning(
        self, fresh_detector, caplog: pytest.LogCaptureFixture,
    ) -> None:
        det = fresh_detector
        _seed_baseline(det, "test_metric", 100.0, 60)
        # 3σ should be informative but not a warning under the new rule.
        # mean≈100, stddev≈0.6 (uniform width 2 / sqrt(12)) → 3σ ≈ 101.8
        with caplog.at_level(logging.INFO, logger="app.anomaly_detector"):
            alert = det.record_sample("test_metric", 102.0)
        assert alert is not None, "should still record an alert object"
        # Find the ANOMALY: line
        anomaly_lines = [r for r in caplog.records if "ANOMALY:" in r.message]
        assert len(anomaly_lines) == 1, (
            f"expected exactly one ANOMALY line, got {len(anomaly_lines)}"
        )
        assert anomaly_lines[0].levelno == logging.INFO, (
            f"low-sigma anomaly should INFO not WARN; got {anomaly_lines[0].levelname}"
        )

    def test_high_sigma_still_emits_warning(
        self, fresh_detector, caplog: pytest.LogCaptureFixture,
    ) -> None:
        det = fresh_detector
        _seed_baseline(det, "test_metric", 100.0, 60)
        # ~10σ extreme deviation: this is a real anomaly; should WARN.
        with caplog.at_level(logging.INFO, logger="app.anomaly_detector"):
            alert = det.record_sample("test_metric", 200.0)
        assert alert is not None
        anomaly_lines = [r for r in caplog.records if "ANOMALY:" in r.message]
        assert len(anomaly_lines) == 1
        assert anomaly_lines[0].levelno == logging.WARNING, (
            f"extreme-sigma anomaly should WARN; got {anomaly_lines[0].levelname}"
        )

    def test_alert_appended_regardless_of_log_level(
        self, fresh_detector,
    ) -> None:
        """Whatever the log level, _alerts must capture the event.
        That's the source of truth for the dashboard / API."""
        det = fresh_detector
        _seed_baseline(det, "test_metric", 100.0, 60)
        before = len(det._alerts)
        det.record_sample("test_metric", 102.5)  # low-sigma → INFO
        det.record_sample("test_metric", 200.0)  # high-sigma → WARN
        after = len(det._alerts)
        # Both alerts captured, regardless of WARN vs INFO log level.
        assert after - before == 2, (
            f"expected 2 alerts in _alerts; got {after - before}"
        )


class TestThresholdConstantExposed:
    """The threshold must be importable so operators can tune it from
    config without touching the function body."""

    def test_warn_sigma_threshold_constant_exists(self) -> None:
        from app.anomaly_detector import _WARN_SIGMA_THRESHOLD
        assert _WARN_SIGMA_THRESHOLD == 5.0
