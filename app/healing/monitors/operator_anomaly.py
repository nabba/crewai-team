"""operator_anomaly — softer-signal companion to ``unauthorized_sender``.

PROGRAM §51 — Q16 Theme 3 (decade-resilience, operator-unavailable
autonomy + impersonation detection). The existing ``unauthorized_
sender`` audit event catches calls from numbers that aren't on the
allow-list — that's the hard boundary. This monitor watches the
*authorized* sender's behaviour pattern and surfaces dramatic shifts
that may signal:

  * Account compromise (Signal-CLI keys stolen, attacker now
    impersonating the operator)
  * Coercion (operator forced to send something at gunpoint —
    extreme case but worth surfacing)
  * Operator distress (sudden very-late-night message bursts can
    indicate a mental-health flag the companion should respond to)
  * Operator unavailability (sudden quiet for many days)

What this monitor observes (from ``workspace/audit.log`` JSONL, the
existing audit surface — no new instrumentation needed):

  1. **Hour-of-day shift.** Baseline fraction of messages in each
     6h bucket (night/morning/afternoon/evening); alert when a
     bucket's recent fraction is ≥3× the baseline.

  2. **Cadence shift.** Daily-message count ratio recent-7-day vs
     baseline-30-day. Alert at <0.5× (operator gone quiet) or >2×
     (sudden chatter). The "quiet" signal is non-critical — it
     might mean vacation, hospitalisation, or just a busy week.

  3. **Message-length shift.** Median message length in recent
     50 messages vs prior 200. Alert when the median moves by ≥4×
     (someone else is talking; their tempo is different).

  4. **New authorized sender.** Senders seen in last 7 days that
     have NOT been seen in the prior 90 days. (Past the
     unauthorized-sender filter, so this is a new ALLOWED sender —
     usually an operator opt-in, but surface it.)

  5. **Failure-isolated weighted score.** Each signal contributes a
     bounded float; sum drives the master alert decision so a
     single noisy signal can't dominate.

What this monitor **does not** do:

  * Block any messages or refuse any commands. Observation only.
    The defense lever is the future ``vacation_mode`` switch (also
    Q16 Theme 3, deferred to a separate change so the security
    contract gets its own review).
  * Read message *content*. The audit log records lengths and
    timestamps, not message bodies. Privacy boundary preserved.
  * Train any model. Pure statistics on JSONL rows.

Cadence: daily probe; internal weekly cadence for full evaluation.
Master switch: ``operator_anomaly_monitor_enabled`` (default ON).
Alert dedup: 14 days per signal kind.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


NAME = "operator_anomaly"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "operator_anomaly_monitor_enabled"

_INTERNAL_CADENCE_S = 7 * 24 * 3600
_DEDUP_WINDOW_S = 14 * 86400
_STATE_FILE_NAME = "operator_anomaly_state.json"

# Window sizes.
_RECENT_DAYS = 7
_BASELINE_DAYS = 30
_LONG_LOOKBACK_DAYS = 90
_MIN_BASELINE_EVENTS = 30   # below this, signals are unreliable

# Alert thresholds.
_HOUR_BUCKET_RATIO_WARN = 3.0   # recent bucket fraction ≥ 3× baseline
_CADENCE_RATIO_HIGH = 2.0       # recent count ≥ 2× baseline rate
_CADENCE_RATIO_LOW = 0.5        # recent count ≤ 0.5× baseline rate
_LENGTH_RATIO_WARN = 4.0        # median length moved ≥ 4×
_NEW_SENDER_MIN_MESSAGES = 3    # ignore one-off accidental sends


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_operator_anomaly_monitor_enabled
        return get_operator_anomaly_monitor_enabled()
    except Exception:
        return os.getenv(
            "OPERATOR_ANOMALY_MONITOR_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _audit_log_path() -> Path:
    return _workspace() / "audit.log"


# ── Public read API (Q16.1 Item 7 — decoupling cross-module readers)


def last_critical_alert_at(kind: str) -> Optional[float]:
    """Return the epoch-second timestamp of the most recent alert for
    ``kind`` (e.g. ``"new_sender"``, ``"hour_shift"``, ``"cadence_quiet"``).
    Returns None when no alert has fired.

    Replaces direct reads of ``operator_anomaly_state.json`` from
    other modules. Cross-module consumers (e.g. ``vacation_mode``)
    should call this rather than reading the state file directly —
    the state-file schema is private and may change without notice.

    Failure-isolated: returns None on any I/O / parse error."""
    try:
        state = _read_state()
    except Exception:
        return None
    last_alerts = state.get("last_alert_at", {})
    if not isinstance(last_alerts, dict):
        return None
    value = last_alerts.get(kind)
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_alert_at": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_alert_at": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("operator_anomaly: state write failed", exc_info=True)


def _parse_ts(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        # Tolerate both "+00:00" and "Z" suffixes.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _load_request_received(
    *, cutoff_ts: float, until_ts: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Read JSONL audit log, return ``request_received`` rows since
    ``cutoff_ts`` and (optionally) before ``until_ts``. Tolerant of
    broken lines.

    Escalates to monthly-rotated archives via ``jsonl_retention.read_archive``
    when ``cutoff_ts`` predates the live file's oldest row (Q3.1
    §40.1.1c pattern — the 30d / 90d baselines this monitor needs would
    otherwise silently truncate once retention rotation kicks in).
    """
    p = _audit_log_path()
    if not p.exists():
        return []
    # Cheap heuristic: if the live file's mtime is more recent than the
    # cutoff we want, the live file alone is enough; otherwise walk the
    # archive too. This keeps the hot path zero-overhead in steady state.
    use_archive = False
    try:
        live_oldest_ts = _live_oldest_ts(p)
        if live_oldest_ts is None or live_oldest_ts > cutoff_ts:
            use_archive = True
    except Exception:
        # Failure-open — if we can't tell, walk the archive to be safe.
        use_archive = True

    out: list[dict[str, Any]] = []

    def _consume(iterable) -> None:
        for line in iterable:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") != "request_received":
                continue
            ts = _parse_ts(row.get("ts"))
            if ts is None or ts < cutoff_ts:
                continue
            if until_ts is not None and ts >= until_ts:
                continue
            row["_ts"] = ts
            out.append(row)

    if use_archive:
        try:
            from app.utils.jsonl_retention import read_archive
            _consume(read_archive(p, include_live=True))
            return out
        except Exception:
            # Fall through to live-only read.
            pass

    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            _consume(f)
    except OSError:
        return []
    return out


def _live_oldest_ts(path: Path) -> Optional[float]:
    """Read the first usable timestamp from the live audit file."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = _parse_ts(row.get("ts"))
                if ts is not None:
                    return ts
    except OSError:
        return None
    return None


def _bucket_for_hour(hour: int) -> str:
    if 0 <= hour < 6:
        return "night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def _hour_bucket_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: Counter = Counter()
    for r in rows:
        ts = r.get("_ts")
        if not isinstance(ts, (int, float)):
            continue
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        counts[_bucket_for_hour(hour)] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: counts[k] / total for k in ("night", "morning", "afternoon", "evening")}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float(s[n // 2 - 1] + s[n // 2]) / 2.0


def _detect_hour_shift(
    recent: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Bucket-fraction ratio. Alert when any bucket's recent fraction
    is ≥ ``_HOUR_BUCKET_RATIO_WARN`` × its baseline fraction."""
    if len(baseline) < _MIN_BASELINE_EVENTS:
        return None
    rd = _hour_bucket_distribution(recent)
    bd = _hour_bucket_distribution(baseline)
    if not rd or not bd:
        return None
    flagged: list[tuple[str, float, float]] = []
    for bucket in ("night", "morning", "afternoon", "evening"):
        # Lower-floor on the baseline so a near-zero baseline doesn't
        # produce inf ratios for a single shifted message.
        b = max(bd.get(bucket, 0.0), 0.05)
        r = rd.get(bucket, 0.0)
        if r >= _HOUR_BUCKET_RATIO_WARN * b:
            flagged.append((bucket, r, b))
    if not flagged:
        return None
    return {
        "kind": "hour_shift",
        "flagged_buckets": [
            {
                "bucket": b,
                "recent_fraction": round(r, 3),
                "baseline_fraction": round(bl, 3),
            }
            for b, r, bl in flagged
        ],
    }


def _detect_cadence_shift(
    recent: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if len(baseline) < _MIN_BASELINE_EVENTS:
        return None
    recent_per_day = len(recent) / max(_RECENT_DAYS, 1)
    baseline_per_day = len(baseline) / max(_BASELINE_DAYS, 1)
    if baseline_per_day < 0.5:
        return None  # too sparse to characterise
    ratio = recent_per_day / baseline_per_day
    if ratio >= _CADENCE_RATIO_HIGH:
        return {
            "kind": "cadence_spike",
            "ratio": round(ratio, 2),
            "recent_per_day": round(recent_per_day, 2),
            "baseline_per_day": round(baseline_per_day, 2),
        }
    if ratio <= _CADENCE_RATIO_LOW:
        return {
            "kind": "cadence_quiet",
            "ratio": round(ratio, 2),
            "recent_per_day": round(recent_per_day, 2),
            "baseline_per_day": round(baseline_per_day, 2),
        }
    return None


def _detect_length_shift(
    recent: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if len(recent) < 10 or len(baseline) < _MIN_BASELINE_EVENTS:
        return None
    recent_lens = [
        float(r.get("message_length", 0)) for r in recent
        if isinstance(r.get("message_length"), (int, float))
    ]
    baseline_lens = [
        float(r.get("message_length", 0)) for r in baseline
        if isinstance(r.get("message_length"), (int, float))
    ]
    if not recent_lens or not baseline_lens:
        return None
    rm = _median(recent_lens)
    bm = _median(baseline_lens)
    if bm < 1.0:
        return None
    ratio = rm / bm
    if ratio >= _LENGTH_RATIO_WARN or (ratio > 0 and ratio <= 1.0 / _LENGTH_RATIO_WARN):
        return {
            "kind": "length_shift",
            "ratio": round(ratio, 2),
            "recent_median": round(rm, 1),
            "baseline_median": round(bm, 1),
        }
    return None


def _detect_new_sender(
    recent: list[dict[str, Any]],
    long_lookback: list[dict[str, Any]],
    *,
    recent_cutoff_ts: float,
) -> Optional[dict[str, Any]]:
    """Sender seen in recent window but not in the prior 90d window."""
    recent_senders = Counter(r.get("sender", "") for r in recent if r.get("sender"))
    prior_senders: set[str] = set()
    for r in long_lookback:
        # Exclude rows already in the recent window so we compare
        # disjoint sets.
        ts = r.get("_ts")
        if isinstance(ts, (int, float)) and ts >= recent_cutoff_ts:
            continue
        s = r.get("sender")
        if s:
            prior_senders.add(s)
    new = [
        (s, c) for s, c in recent_senders.items()
        if s and s not in prior_senders and c >= _NEW_SENDER_MIN_MESSAGES
    ]
    if not new:
        return None
    return {
        "kind": "new_sender",
        "senders": [
            {"sender": s, "messages": c} for s, c in new
        ],
    }


def _alert_if_due(
    state: dict[str, Any],
    *,
    key: str,
    title: str,
    body: str,
    now: float,
    critical: bool = False,
) -> bool:
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
            url="/cp/monitor",
            topic=f"operator_anomaly:{key}",
            critical=critical,
            arbitrate=True,
        )
        return True
    except Exception:
        logger.debug("operator_anomaly: notify failed", exc_info=True)
        return False


def _emit_alerts(
    state: dict[str, Any],
    signals: list[dict[str, Any]],
    *,
    now: float,
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    for sig in signals:
        kind = sig.get("kind")
        if kind == "new_sender":
            # New authorized sender is the most security-relevant —
            # surface as critical so it bypasses arbiter suppression.
            n = len(sig.get("senders", []))
            body = (
                f"🔒 {n} new authorized sender(s) appeared in the last "
                f"{_RECENT_DAYS} days — none seen in the prior "
                f"{_LONG_LOOKBACK_DAYS - _RECENT_DAYS} days.\n\n"
                f"This is past the unauthorized-sender filter (the "
                f"hard boundary is intact), so this number is on the "
                f"allow-list. Worth confirming you intended to add it."
            )
            ok = _alert_if_due(
                state, key="new_sender",
                title="🔒 New authorized sender",
                body=body, now=now, critical=True,
            )
            sig["alert_sent"] = ok
            sent.append(sig)
        elif kind == "hour_shift":
            buckets = ", ".join(
                f"{b['bucket']} ({b['recent_fraction']:.0%} vs "
                f"{b['baseline_fraction']:.0%})"
                for b in sig.get("flagged_buckets", [])
            )
            body = (
                f"🕐 Time-of-day distribution shift: {buckets}.\n\n"
                f"Recent {_RECENT_DAYS}-day pattern is ≥3× baseline "
                f"in one or more buckets. Could mean a different "
                f"schedule, distress signal, or a different sender."
            )
            ok = _alert_if_due(
                state, key="hour_shift",
                title="🕐 Operator time-of-day shift",
                body=body, now=now,
            )
            sig["alert_sent"] = ok
            sent.append(sig)
        elif kind == "cadence_spike":
            body = (
                f"📈 Operator messages/day jumped {sig['ratio']:.1f}× "
                f"({sig['baseline_per_day']:.1f}/day → "
                f"{sig['recent_per_day']:.1f}/day).\n\n"
                f"Sudden activity surge."
            )
            ok = _alert_if_due(
                state, key="cadence_spike",
                title="📈 Operator activity surge",
                body=body, now=now,
            )
            sig["alert_sent"] = ok
            sent.append(sig)
        elif kind == "cadence_quiet":
            body = (
                f"📉 Operator messages/day dropped to "
                f"{sig['ratio']:.1f}× baseline "
                f"({sig['baseline_per_day']:.1f}/day → "
                f"{sig['recent_per_day']:.1f}/day).\n\n"
                f"Could be vacation, busy week, or unavailability. "
                f"Informational — non-critical."
            )
            ok = _alert_if_due(
                state, key="cadence_quiet",
                title="📉 Operator gone quiet",
                body=body, now=now,
            )
            sig["alert_sent"] = ok
            sent.append(sig)
        elif kind == "length_shift":
            body = (
                f"✏️ Message-length median shifted {sig['ratio']:.1f}× "
                f"({sig['baseline_median']:.0f} chars → "
                f"{sig['recent_median']:.0f} chars).\n\n"
                f"Could be tone change, voice mode flip, or a "
                f"different sender."
            )
            ok = _alert_if_due(
                state, key="length_shift",
                title="✏️ Operator message-length shift",
                body=body, now=now,
            )
            sig["alert_sent"] = ok
            sent.append(sig)
    return sent


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """One probe pass. Daily wake-up gates on weekly internal
    cadence."""
    summary: dict[str, Any] = {
        "ran": False,
        "n_recent": 0,
        "n_baseline": 0,
        "signals": [],
        "alerts": [],
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last = float(state.get("last_run_at", 0))
    if last > 0 and cur - last < _INTERNAL_CADENCE_S:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    recent_cutoff = cur - _RECENT_DAYS * 86400
    baseline_cutoff = cur - _BASELINE_DAYS * 86400
    long_cutoff = cur - _LONG_LOOKBACK_DAYS * 86400

    long_rows = _load_request_received(cutoff_ts=long_cutoff)
    recent = [r for r in long_rows if r["_ts"] >= recent_cutoff]
    baseline = [
        r for r in long_rows
        if baseline_cutoff <= r["_ts"] < recent_cutoff
    ]
    summary["n_recent"] = len(recent)
    summary["n_baseline"] = len(baseline)

    signals: list[dict[str, Any]] = []
    for detector in (
        lambda: _detect_hour_shift(recent, baseline),
        lambda: _detect_cadence_shift(recent, baseline),
        lambda: _detect_length_shift(recent, baseline),
        lambda: _detect_new_sender(
            recent, long_rows, recent_cutoff_ts=recent_cutoff,
        ),
    ):
        try:
            sig = detector()
        except Exception:
            logger.debug(
                "operator_anomaly: detector raised", exc_info=True,
            )
            sig = None
        if sig is not None:
            signals.append(sig)

    summary["signals"] = signals
    summary["alerts"] = _emit_alerts(state, signals, now=cur)
    _write_state(state)
    return summary
