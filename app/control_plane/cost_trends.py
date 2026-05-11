"""Cost-trends helpers for the React dashboard's CostTrendsCard.

PROGRAM §40 (2026-05-10/11) — Q3 Item 14.

Produces:

  * **Monthly rollup** — N months of cost / token / call totals from
    ``control_plane.audit_log``. Trims gracefully when fewer months
    of history exist.
  * **6-month forecast** — pure-stdlib OLS linear regression on the
    monthly totals; 95% CI band derived from regression residuals.
  * **Anomaly detection** — daily z-score against a rolling 30-day
    window. Flags days with |z| ≥ 3.

  * **Summary** — human-friendly trend % per month, total over the
    historical window, projected next-12-month total. The summary is
    the single piece the React card shows above the fold; the rest is
    progressive disclosure.

Design constraints:

  * **No numpy dep** — kept the system portable; a single inverted-2x2
    matrix solve is plenty for OLS with one regressor.
  * **Read-only on existing schemas** — taps ``audit_log.cost_usd`` +
    ``audit_log.timestamp``; nothing here writes anywhere.
  * **Goodhart-resistant** — never auto-acts on the forecast. Operators
    read it; the system does not optimize against it.
  * **Cheap on cold cache** — one round-trip to Postgres per call. No
    background scheduler.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_HISTORY_MONTHS = 12       # rollup window for the trend computation
DEFAULT_FORECAST_MONTHS = 6       # projection horizon
DEFAULT_ANOMALY_WINDOW = 30       # rolling-window size for z-scoring
DEFAULT_ANOMALY_Z = 3.0           # |z| threshold for daily anomalies


# ── Data fetch ────────────────────────────────────────────────────────────


def _fetch_monthly_rollup(months: int, project_id: str | None = None) -> list[dict]:
    """Pull monthly totals from ``control_plane.audit_log``.

    Each row is ``{"month": "YYYY-MM", "total_cost_usd": float,
    "total_tokens": int, "call_count": int}``. Returned oldest→newest
    so charts render left-to-right naturally.
    """
    try:
        from app.control_plane.db import execute
    except Exception:
        logger.debug("cost_trends: control_plane.db unavailable", exc_info=True)
        return []

    sql = """
        SELECT to_char(date_trunc('month', timestamp), 'YYYY-MM') AS month,
               COALESCE(SUM(cost_usd), 0)::float8        AS total_cost_usd,
               COALESCE(SUM(tokens), 0)::bigint          AS total_tokens,
               COUNT(*)::bigint                          AS call_count
          FROM control_plane.audit_log
         WHERE cost_usd IS NOT NULL
           AND timestamp >= date_trunc('month', NOW()) - INTERVAL '%s months'
           AND (%s IS NULL OR project_id::text = %s)
      GROUP BY date_trunc('month', timestamp)
      ORDER BY date_trunc('month', timestamp) ASC
    """
    try:
        rows = execute(sql, (int(months), project_id, project_id), fetch=True) or []
    except Exception:
        logger.debug("cost_trends: monthly rollup query failed", exc_info=True)
        return []
    return [
        {
            "month": r["month"],
            "total_cost_usd": float(r["total_cost_usd"] or 0.0),
            "total_tokens": int(r["total_tokens"] or 0),
            "call_count": int(r["call_count"] or 0),
        }
        for r in rows
    ]


def _fetch_daily_rollup(days: int, project_id: str | None = None) -> list[dict]:
    """Pull daily totals — oldest→newest. Used for anomaly detection."""
    try:
        from app.control_plane.db import execute
    except Exception:
        return []
    sql = """
        SELECT to_char(date_trunc('day', timestamp), 'YYYY-MM-DD') AS day,
               COALESCE(SUM(cost_usd), 0)::float8 AS total_cost_usd
          FROM control_plane.audit_log
         WHERE cost_usd IS NOT NULL
           AND timestamp >= NOW() - (INTERVAL '1 day' * %s)
           AND (%s IS NULL OR project_id::text = %s)
      GROUP BY date_trunc('day', timestamp)
      ORDER BY date_trunc('day', timestamp) ASC
    """
    try:
        rows = execute(sql, (int(days), project_id, project_id), fetch=True) or []
    except Exception:
        logger.debug("cost_trends: daily rollup query failed", exc_info=True)
        return []
    return [
        {"day": r["day"], "total_cost_usd": float(r["total_cost_usd"] or 0.0)}
        for r in rows
    ]


# ── OLS linear regression (pure stdlib) ───────────────────────────────────


def _ols_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Returns ``(slope, intercept, residual_stdev)`` for ``y ≈ slope·x +
    intercept``. ``residual_stdev`` is the sample stdev of residuals
    (n-2 dof; falls back to 0 when n < 3).

    Pure stdlib — no numpy. Stable for the small N the forecast uses
    (≤24 months).
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0, (ys[0] if ys else 0.0), 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    if n < 3:
        return slope, intercept, 0.0
    residuals = [(y - (slope * x + intercept)) for x, y in zip(xs, ys)]
    rss = sum(r * r for r in residuals)
    # n-2 dof for OLS with one regressor + intercept
    var = rss / (n - 2)
    return slope, intercept, math.sqrt(max(0.0, var))


def _next_month_label(label: str) -> str:
    """``2026-05`` → ``2026-06``."""
    yr, mo = label.split("-")
    yr_i, mo_i = int(yr), int(mo)
    mo_i += 1
    if mo_i > 12:
        yr_i += 1
        mo_i = 1
    return f"{yr_i:04d}-{mo_i:02d}"


def _build_forecast(
    monthly: list[dict], horizon: int = DEFAULT_FORECAST_MONTHS,
) -> list[dict]:
    """OLS forecast ``horizon`` months past the last observed month.

    Uses ``y = total_cost_usd`` against the integer index of months.
    95% CI = projection ± 1.96·residual_stdev (Gaussian assumption,
    fine for cost data dominated by smooth growth + noise).
    """
    if len(monthly) < 2 or horizon <= 0:
        return []
    xs = [float(i) for i in range(len(monthly))]
    ys = [float(r["total_cost_usd"]) for r in monthly]
    slope, intercept, sigma = _ols_fit(xs, ys)
    out: list[dict] = []
    cursor = monthly[-1]["month"]
    for k in range(1, horizon + 1):
        cursor = _next_month_label(cursor)
        x = float(len(monthly) - 1 + k)
        proj = slope * x + intercept
        ci = 1.96 * sigma
        out.append({
            "month": cursor,
            "projected_usd": round(max(0.0, proj), 6),
            "ci_low": round(max(0.0, proj - ci), 6),
            "ci_high": round(max(0.0, proj + ci), 6),
        })
    return out


# ── Anomaly detection ─────────────────────────────────────────────────────


def _detect_anomalies(
    daily: list[dict],
    window: int = DEFAULT_ANOMALY_WINDOW,
    z_threshold: float = DEFAULT_ANOMALY_Z,
) -> list[dict]:
    """Flag days where the daily total exceeds rolling-mean+z·rolling-σ.

    Window is rolling-trailing — we don't peek at the future. Need at
    least ``window`` data points before the first day can be checked.
    """
    if len(daily) < window + 1:
        return []
    flagged: list[dict] = []
    values = [float(r["total_cost_usd"]) for r in daily]
    for i in range(window, len(daily)):
        chunk = values[i - window:i]
        mean = sum(chunk) / window
        var = sum((v - mean) ** 2 for v in chunk) / max(1, window - 1)
        std = math.sqrt(max(0.0, var))
        if std <= 0:
            continue
        z = (values[i] - mean) / std
        if abs(z) >= z_threshold:
            flagged.append({
                "day": daily[i]["day"],
                "total_cost_usd": round(values[i], 6),
                "expected_usd": round(mean, 6),
                "z_score": round(z, 3),
                "kind": "spike" if z > 0 else "drop",
            })
    # Newest-first per dashboard convention.
    flagged.sort(key=lambda x: x["day"], reverse=True)
    return flagged


# ── Summary ───────────────────────────────────────────────────────────────


def _build_summary(monthly: list[dict], forecast: list[dict]) -> dict[str, Any]:
    if not monthly:
        return {
            "total_history_usd": 0.0,
            "trend_pct_per_month": None,
            "projected_next_12mo_usd": 0.0,
            "history_months_observed": 0,
        }
    total_history = sum(r["total_cost_usd"] for r in monthly)
    # Trend %/month: simple last/first compounded growth, cleaner than
    # slope/mean (which can blow up with near-zero baseline).
    trend_pct: float | None = None
    if len(monthly) >= 3:
        first_avg = sum(r["total_cost_usd"] for r in monthly[:3]) / 3
        last_avg = sum(r["total_cost_usd"] for r in monthly[-3:]) / 3
        if first_avg > 0.001:
            n = len(monthly) - 3  # months between centers of the two windows
            if n > 0:
                ratio = last_avg / first_avg
                trend_pct = (ratio ** (1.0 / n) - 1.0) * 100.0
    proj_12 = sum(f["projected_usd"] for f in forecast[:12])
    return {
        "total_history_usd": round(total_history, 6),
        "trend_pct_per_month": (
            round(trend_pct, 3) if trend_pct is not None else None
        ),
        "projected_next_12mo_usd": round(proj_12, 6),
        "history_months_observed": len(monthly),
    }


# ── Public entry point ────────────────────────────────────────────────────


def get_cost_trends(
    history_months: int = DEFAULT_HISTORY_MONTHS,
    forecast_months: int = DEFAULT_FORECAST_MONTHS,
    anomaly_window: int = DEFAULT_ANOMALY_WINDOW,
    anomaly_z: float = DEFAULT_ANOMALY_Z,
    project_id: str | None = None,
) -> dict[str, Any]:
    """One-shot trend bundle for the React CostTrendsCard.

    Cheap (≤2 SQL round-trips). Safe to call per page-load. No caching
    layer — Postgres is already fast enough on the relatively small
    monthly-rollup over a 12-month window.
    """
    monthly = _fetch_monthly_rollup(history_months, project_id=project_id)
    daily_window_days = max(60, anomaly_window * 2)
    daily = _fetch_daily_rollup(daily_window_days, project_id=project_id)
    forecast = _build_forecast(monthly, horizon=forecast_months)
    anomalies = _detect_anomalies(daily, window=anomaly_window, z_threshold=anomaly_z)
    summary = _build_summary(monthly, forecast)
    return {
        "summary": summary,
        "monthly": monthly,
        "forecast": forecast,
        "anomalies": anomalies,
        "params": {
            "history_months": history_months,
            "forecast_months": forecast_months,
            "anomaly_window": anomaly_window,
            "anomaly_z_threshold": anomaly_z,
        },
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
