"""
decentered.py — Complementary "no-self" reflection pass over the affect trace.

OBSERVATIONAL ONLY. Operates on the same raw trace as the Narrative-Self
track (salience.py / episodes.py / narrative.py) but without imposing a
first-person frame, identity coherence, or daily arc.

Strict invariants:
    - Does NOT write to the experiential KB.
    - Does NOT mutate identity_claims.json.
    - Does NOT emit entry_type=chapter or entry_type=episode.
    - Read-only on the trace + salience surfaces.

Two passes:

  (A) Structural cluster pass — groups salience events by
      (kind, attractor, prev_attractor, sorted out-of-band variables) and
      tightens groups via greedy complete-linkage on (V, A, C). Surfaces
      shapes that recur regardless of when, which agent, or under what
      storyline. Cross-day spans are explicitly counted — those are the
      patterns the daily chapter consolidator cannot see by construction.

  (B) Statistical anomaly pass — over the rolling V/A/C/total_error/
      epistemic_uncertainty time series, computes per-variable rolling
      z-scores. Flags points whose composite |z|_max exceeds threshold
      regardless of whether they crossed any salience trigger. Surfaces
      "statistically unusual moments" that the per-variable salience
      filter dropped.

Output:
    /app/workspace/affect/decentered/<YYYY-MM-DD>.json
    Idempotent on date — re-running same day overwrites.

Self-Improver permissions: read-only on this module. The no-self pass
shapes how the system observes itself; the same self-modeling integrity
invariant that protects salience.py and narrative.py applies here.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.utils.jsonl_retention import read_archive

logger = logging.getLogger(__name__)


# ── Tunable parameters (module-level for auditability) ───────────────────────

# Cluster pass
_CLUSTER_VAC_THRESHOLD = 0.35     # max Euclidean dist over (V,A,C) inside a cluster
_MIN_CLUSTER_SIZE = 3             # clusters below this are skipped

# Anomaly pass
_ANOMALY_WINDOW = 256             # rolling baseline length (in trace events)
_ANOMALY_Z_THRESHOLD = 3.0        # |z|_max above this flags an anomaly
_ANOMALY_MIN_BASELINE = 32        # need this many points before any flagging

# Cross-day motif gate (the experiment's primary criterion)
_MIN_CROSS_DAY_SPAN = 3           # cluster spans ≥ this many distinct UTC days

# Read budget (safety cap; ~14 days at typical rate is well under)
_MAX_LINES = 200_000

# Report caps
_TOP_CLUSTERS = 20
_TOP_ANOMALIES = 20


# ── Public entry point ───────────────────────────────────────────────────────

def run_decentered_pass(window_hours: int = 14 * 24) -> dict:
    """Run both passes over the last `window_hours`. Persist + return summary."""
    from app.paths import AFFECT_TRACE, AFFECT_SALIENCE, AFFECT_ROOT

    cutoff = datetime.now(timezone.utc).timestamp() - window_hours * 3600

    salience_events = _load_salience(AFFECT_SALIENCE, cutoff)
    trace_points = _load_trace(AFFECT_TRACE, cutoff)

    clusters = _cluster_salience(salience_events)
    anomalies = _detect_anomalies(trace_points)
    summary = _summarise(clusters, anomalies, salience_events, trace_points, window_hours)

    out_dir = AFFECT_ROOT / "decentered"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = out_dir / f"{today}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(
        "affect.decentered: wrote %s "
        "(clusters=%d cross_day=%d anomalies=%d trace=%d salience=%d)",
        out_path.name,
        len(clusters),
        summary["clusters"]["cross_day"],
        len(anomalies),
        len(trace_points),
        len(salience_events),
    )
    return summary


# ── Pass A: structural clusters ──────────────────────────────────────────────

def _cluster_salience(events: list[dict]) -> list[dict]:
    """Group by structural fingerprint, then split each bucket on (V, A, C)."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for ev in events:
        fp = (
            ev.get("kind", ""),
            ev.get("attractor", "neutral"),
            ev.get("prev_attractor") or "",
            tuple(sorted(ev.get("out_of_band") or [])),
        )
        buckets[fp].append(ev)

    clusters: list[dict] = []
    for fp, group in buckets.items():
        if len(group) < _MIN_CLUSTER_SIZE:
            continue
        for sub in _split_by_vac(group, _CLUSTER_VAC_THRESHOLD):
            if len(sub) < _MIN_CLUSTER_SIZE:
                continue
            clusters.append(_cluster_summary(fp, sub))

    clusters.sort(key=lambda c: (c["days_spanned"], c["size"]), reverse=True)
    return clusters


def _split_by_vac(events: list[dict], threshold: float) -> list[list[dict]]:
    """Greedy complete-linkage on (V, A, C). Mirrors consolidator pattern."""
    out: list[list[dict]] = []
    for ev in events:
        v = _vac(ev)
        best_idx = -1
        best_max = float("inf")
        for ci, cluster in enumerate(out):
            mx = max(_euclid(v, _vac(e)) for e in cluster)
            if mx <= threshold and mx < best_max:
                best_idx = ci
                best_max = mx
        if best_idx >= 0:
            out[best_idx].append(ev)
        else:
            out.append([ev])
    return out


def _vac(ev: dict) -> tuple[float, float, float]:
    return (
        float(ev.get("valence", 0.0)),
        float(ev.get("arousal", 0.0)),
        float(ev.get("controllability", 0.5)),
    )


def _euclid(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _cluster_summary(fp: tuple, members: list[dict]) -> dict:
    timestamps = sorted(m.get("ts", "") for m in members if m.get("ts"))
    days = {ts[:10] for ts in timestamps if len(ts) >= 10}
    centroid = tuple(
        sum(_vac(m)[i] for m in members) / len(members) for i in range(3)
    )
    spread = (
        statistics.fmean(_euclid(_vac(m), centroid) for m in members)
        if members else 0.0
    )

    kind, attractor, prev_attractor, oob = fp
    fp_str = f"{kind}|{attractor}"
    if prev_attractor:
        fp_str += f"←{prev_attractor}"
    if oob:
        fp_str += f"|oob={','.join(oob)}"

    return {
        "fingerprint": fp_str,
        "size": len(members),
        "vac_centroid": [round(x, 3) for x in centroid],
        "spread": round(spread, 4),
        "first_ts": timestamps[0] if timestamps else "",
        "last_ts": timestamps[-1] if timestamps else "",
        "days_spanned": len(days),
        "days": sorted(days),
        "severities": dict(Counter(m.get("severity", "info") for m in members)),
        "sample_details": [m.get("detail", "")[:160] for m in members[:3]],
    }


# ── Pass B: statistical anomalies ────────────────────────────────────────────

def _detect_anomalies(trace_points: list[dict]) -> list[dict]:
    """Rolling z-score anomaly detection.

    Maintains a rolling window per variable. Once the baseline has
    enough samples, any point whose composite |z|_max exceeds the
    threshold is flagged. Composite = max abs z-score across the 5
    variables (cheaper than Mahalanobis but captures "unusual along
    any axis" — sufficient for first-pass surfacing).
    """
    if len(trace_points) < _ANOMALY_MIN_BASELINE:
        return []

    history: dict[str, list[float]] = {
        "valence": [],
        "arousal": [],
        "controllability": [],
        "total_error": [],
        "epistemic_uncertainty": [],
    }
    anomalies: list[dict] = []

    for pt in trace_points:
        affect = pt.get("affect") or {}
        viability = pt.get("viability") or {}
        viab_values = viability.get("values") or {}

        sample = {
            "valence": float(affect.get("valence", 0.0)),
            "arousal": float(affect.get("arousal", 0.0)),
            "controllability": float(affect.get("controllability", 0.5)),
            "total_error": float(viability.get("total_error", 0.0)),
            "epistemic_uncertainty": float(viab_values.get("epistemic_uncertainty", 0.3)),
        }

        baseline_ready = all(
            len(hist) >= _ANOMALY_MIN_BASELINE for hist in history.values()
        )
        if baseline_ready:
            max_z = 0.0
            max_var = ""
            for name, hist in history.items():
                z = _z_score(sample[name], hist)
                if abs(z) > abs(max_z):
                    max_z = z
                    max_var = name
            if abs(max_z) >= _ANOMALY_Z_THRESHOLD:
                anomalies.append({
                    "ts": affect.get("ts", ""),
                    "max_var": max_var,
                    "z": round(max_z, 3),
                    "vac": [
                        round(sample["valence"], 3),
                        round(sample["arousal"], 3),
                        round(sample["controllability"], 3),
                    ],
                    "total_error": round(sample["total_error"], 3),
                    "epistemic_uncertainty": round(sample["epistemic_uncertainty"], 3),
                    "attractor": affect.get("attractor", ""),
                })

        for name, val in sample.items():
            hist = history[name]
            hist.append(val)
            if len(hist) > _ANOMALY_WINDOW:
                hist.pop(0)

    return anomalies


def _z_score(x: float, hist: list[float]) -> float:
    if len(hist) < 2:
        return 0.0
    mean = statistics.fmean(hist)
    sd = statistics.pstdev(hist)
    if sd <= 1e-6:
        return 0.0
    return (x - mean) / sd


# ── Summary ──────────────────────────────────────────────────────────────────

def _summarise(
    clusters: list[dict],
    anomalies: list[dict],
    salience_events: list[dict],
    trace_points: list[dict],
    window_hours: int,
) -> dict:
    cross_day_clusters = [c for c in clusters if c["days_spanned"] >= _MIN_CROSS_DAY_SPAN]
    salience_ts_index = {ev.get("ts", "") for ev in salience_events if ev.get("ts")}
    anomalies_outside_salience = [
        a for a in anomalies if a["ts"] and a["ts"] not in salience_ts_index
    ]

    first_ts = ""
    last_ts = ""
    if trace_points:
        first_ts = (trace_points[0].get("affect") or {}).get("ts", "") or ""
        last_ts = (trace_points[-1].get("affect") or {}).get("ts", "") or ""

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "input": {
            "trace_points": len(trace_points),
            "salience_events": len(salience_events),
            "trace_window_first_ts": first_ts,
            "trace_window_last_ts": last_ts,
        },
        "clusters": {
            "total": len(clusters),
            "cross_day": len(cross_day_clusters),
            "top": clusters[:_TOP_CLUSTERS],
        },
        "anomalies": {
            "total": len(anomalies),
            "outside_salience": len(anomalies_outside_salience),
            "top": sorted(anomalies, key=lambda a: -abs(a["z"]))[:_TOP_ANOMALIES],
        },
        "experiment_criterion": {
            "min_cross_day_span": _MIN_CROSS_DAY_SPAN,
            "min_cluster_size": _MIN_CLUSTER_SIZE,
            "anomaly_z_threshold": _ANOMALY_Z_THRESHOLD,
        },
    }


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _iter_with_archive(path: Path) -> Iterator[str]:
    """Yield every line across all monthly archives + the live file, in
    chronological order. Q3.1 (2026-05-11) — archive rotation moved
    historical lines out of the live file; daily passes (24h window) are
    still served from live alone, but operator-initiated long-window
    passes (e.g. 365d) must walk the archive to see what's actually
    there. Falls back to live-only when the archive directory is empty.
    """
    yielded_live = False
    try:
        for line in read_archive(path, include_live=True):
            yielded_live = True
            yield line
    except Exception:
        logger.debug(
            "decentered: archive iterator failed, falling back to live",
            exc_info=True,
        )
    if not yielded_live and path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                yield from f
        except Exception:
            logger.debug("decentered: live fallback failed", exc_info=True)


def _load_salience(path: Path, cutoff_unix: float) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for i, line in enumerate(_iter_with_archive(path)):
            if i > _MAX_LINES:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_unix = _ts_to_unix(row.get("ts", ""))
            if ts_unix is None or ts_unix < cutoff_unix:
                continue
            out.append(row)
    except Exception:
        logger.debug("decentered: load salience failed", exc_info=True)
    return out


def _load_trace(path: Path, cutoff_unix: float) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for i, line in enumerate(_iter_with_archive(path)):
            if i > _MAX_LINES:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            affect = row.get("affect") or {}
            ts_unix = _ts_to_unix(affect.get("ts", ""))
            if ts_unix is None or ts_unix < cutoff_unix:
                continue
            out.append(row)
    except Exception:
        logger.debug("decentered: load trace failed", exc_info=True)
    return out


def _ts_to_unix(ts: str) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# ── Idle-scheduler / CLI entry points ────────────────────────────────────────

def run_daily_pass() -> None:
    """Idle-scheduler hook. Runs over the last 24h. Failures are swallowed."""
    try:
        run_decentered_pass(window_hours=24)
    except Exception:
        logger.exception("affect.decentered: daily pass failed")


def main() -> None:
    """CLI: python -m app.affect.decentered [window_hours]"""
    import sys
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 14 * 24
    summary = run_decentered_pass(window_hours=hours)
    print(json.dumps({
        "trace_points": summary["input"]["trace_points"],
        "salience_events": summary["input"]["salience_events"],
        "clusters": summary["clusters"]["total"],
        "cross_day_clusters": summary["clusters"]["cross_day"],
        "anomalies": summary["anomalies"]["total"],
        "anomalies_outside_salience": summary["anomalies"]["outside_salience"],
    }, indent=2))


if __name__ == "__main__":
    main()
