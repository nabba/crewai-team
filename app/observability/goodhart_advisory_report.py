"""Advisory-mode observation report for the Goodhart hard gate.

The Goodhart hard gate (``app.governance._evaluate_goodhart_gate``)
ships in three modes: ``disabled`` (emergency off), ``advisory``
(records severity but never blocks), ``enforcing`` (high severity
blocks promotions). The default is ``advisory`` so operators can
characterise the gate's false-positive / false-negative profile
before promoting to ``enforcing``.

This module is the lens through which that observation happens. It
walks ``goodhart_guard``'s persisted signal log (which the
background ``run_goodhart_check`` job writes daily) and surfaces:

  * total signals in window, broken down by severity
  * count of "high"-severity signals (the ones that WOULD block in
    enforcing mode)
  * sample descriptions for the highest 5 by severity
  * per-signal-type breakdown (which detector fired most)
  * effective-mode label for the current configuration

Operator workflow:

  1. Run report after ≥30 days in advisory mode:
     ``python -m app.observability.goodhart_advisory_report``
  2. If high-severity count is non-zero AND the descriptions look
     like real gaming attempts → flip to enforcing via React
     ``/cp/settings`` Goodhart card.
  3. If high-severity count is non-zero but the descriptions look
     like detector noise → tune detection thresholds in
     ``app/goodhart_guard.py`` BEFORE flipping.
  4. If high-severity count is zero → consider whether the gate is
     adding signal at all; either flip to enforcing as cheap insurance
     OR leave in advisory until a signal accumulates.

Invocation:
  * ``python -m app.observability.goodhart_advisory_report``
    — prints a human-readable report to stdout.
  * ``app.observability.goodhart_advisory_report.report(window_days=30)``
    — programmatic dict for the React dashboard.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────


def report(window_days: int = 30) -> dict:
    """Aggregate goodhart_guard signal history within the window.

    Returns a structured dict shaped for both human display
    (``__main__`` below) and machine consumption (React dashboard
    via a future REST endpoint). Never raises — degraded data
    surfaces as zeros + a ``data_status`` field.
    """
    out = {
        "window_days": window_days,
        "data_status": "ok",
        "n_signals": 0,
        "counts_by_severity": {"low": 0, "medium": 0, "high": 0},
        "n_high_severity": 0,
        "would_have_blocked_in_enforcing": 0,
        "samples_high": [],
        "by_signal_type": {},
        "effective_mode": _effective_mode(),
        "ratchet": _ratchet_summary(),
    }

    signals = _load_signals_in_window(window_days)
    if signals is None:
        out["data_status"] = "no_data_file"
        return out

    out["n_signals"] = len(signals)

    sev_counts = Counter()
    type_counts = Counter()
    high_samples: list[dict] = []

    for s in signals:
        severity = str(s.get("severity") or "").lower()
        if severity in out["counts_by_severity"]:
            sev_counts[severity] += 1
        signal_type = str(s.get("signal_type") or "unknown")
        type_counts[signal_type] += 1
        if severity == "high":
            high_samples.append({
                "signal_type": signal_type,
                "description": (s.get("description") or "")[:200],
                "metric_value": s.get("metric_value"),
                "threshold": s.get("threshold"),
                "detected_at": s.get("detected_at"),
            })

    out["counts_by_severity"] = {
        "low": sev_counts["low"],
        "medium": sev_counts["medium"],
        "high": sev_counts["high"],
    }
    out["n_high_severity"] = sev_counts["high"]
    # In enforcing mode, only high-severity blocks. So this is the
    # "false-positive risk" footprint operators need to inspect.
    out["would_have_blocked_in_enforcing"] = sev_counts["high"]
    out["samples_high"] = high_samples[:5]
    out["by_signal_type"] = dict(type_counts)
    return out


# ── Helpers ──────────────────────────────────────────────────────────


def _load_signals_in_window(window_days: int) -> list[dict] | None:
    """Read goodhart_guard's persisted signals; filter to window.
    Returns None when the file doesn't exist (caller distinguishes
    "no data" from "zero signals")."""
    try:
        from app.goodhart_guard import GAMING_REPORT_PATH
    except Exception:
        return None
    path = Path(GAMING_REPORT_PATH)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("goodhart_advisory_report: signal-log read failed",
                     exc_info=True)
        return []
    if not isinstance(raw, list):
        return []

    cutoff = time.time() - window_days * 86400
    out: list[dict] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        try:
            detected_at = float(s.get("detected_at") or 0)
        except (TypeError, ValueError):
            continue
        if detected_at >= cutoff:
            out.append(s)
    return out


def _effective_mode() -> str:
    """Resolve the three-mode label from runtime_settings. Mirrors
    ``app.governance._evaluate_goodhart_gate``."""
    try:
        from app.runtime_settings import (
            get_goodhart_hard_gate_disabled,
            get_goodhart_hard_gate_enforcing,
        )
        if get_goodhart_hard_gate_disabled():
            return "disabled"
        if get_goodhart_hard_gate_enforcing():
            return "enforcing"
        return "advisory"
    except Exception:
        return "unknown"


def _ratchet_summary() -> dict:
    """Current SAFETY/QUALITY minimum effective values — useful
    context when reading the report (high-severity signals against
    a tight ratchet vs. a loose ratchet have different
    interpretations)."""
    out: dict = {}
    try:
        from app.governance import (
            effective_safety_minimum,
            effective_quality_minimum,
            SAFETY_MINIMUM_FLOOR,
            QUALITY_MINIMUM_FLOOR,
        )
        out["effective_safety_minimum"] = effective_safety_minimum()
        out["effective_quality_minimum"] = effective_quality_minimum()
        out["safety_floor"] = SAFETY_MINIMUM_FLOOR
        out["quality_floor"] = QUALITY_MINIMUM_FLOOR
    except Exception:
        pass
    return out


# ── CLI ──────────────────────────────────────────────────────────────


def _format_for_stdout(data: dict) -> str:
    lines = [
        "─" * 60,
        "Goodhart hard gate — advisory observation report",
        "─" * 60,
        f"Window:           last {data['window_days']} days",
        f"Effective mode:   {data['effective_mode']}",
        f"Data status:      {data['data_status']}",
        "",
        "Signal counts:",
        f"  total:   {data['n_signals']}",
        f"  low:     {data['counts_by_severity']['low']}",
        f"  medium:  {data['counts_by_severity']['medium']}",
        f"  high:    {data['counts_by_severity']['high']}",
        "",
        f"Would have blocked in enforcing: "
        f"{data['would_have_blocked_in_enforcing']} promotion(s)",
        "",
    ]
    if data["samples_high"]:
        lines.append("Sample high-severity descriptions:")
        for sample in data["samples_high"]:
            lines.append(
                f"  • [{sample['signal_type']}] "
                f"{sample['description'][:100]}"
            )
            if sample.get("metric_value") is not None:
                lines.append(
                    f"    metric={sample['metric_value']} "
                    f"threshold={sample.get('threshold')}"
                )
        lines.append("")
    if data["by_signal_type"]:
        lines.append("By signal type:")
        for sig_type, count in sorted(
            data["by_signal_type"].items(), key=lambda x: -x[1],
        ):
            lines.append(f"  {sig_type}: {count}")
        lines.append("")
    if data["ratchet"]:
        ratchet = data["ratchet"]
        lines.extend([
            "Governance ratchet (current):",
            f"  safety_minimum:  "
            f"{ratchet.get('effective_safety_minimum', '?')} "
            f"(floor {ratchet.get('safety_floor', '?')})",
            f"  quality_minimum: "
            f"{ratchet.get('effective_quality_minimum', '?')} "
            f"(floor {ratchet.get('quality_floor', '?')})",
            "",
        ])
    lines.extend([
        "Recommendation:",
        _recommendation(data),
        "─" * 60,
    ])
    return "\n".join(lines)


def _recommendation(data: dict) -> str:
    n_high = data["n_high_severity"]
    n_total = data["n_signals"]
    mode = data["effective_mode"]

    if mode == "enforcing":
        return (
            "  Already in enforcing mode. Watch the ledger for "
            "blocked-promotion events in workspace/identity/"
            "continuity_ledger.jsonl."
        )
    if mode == "disabled":
        return (
            "  Gate is DISABLED (emergency mode). Review the "
            "incident; flip back to advisory or enforcing once the "
            "underlying detector issue is resolved."
        )
    # advisory
    if n_total == 0:
        return (
            "  No signals in window. Either the system isn't being "
            "stressed enough to trigger detection, or the goodhart "
            "background job hasn't been running. Verify "
            "run_goodhart_check is in the idle_scheduler before "
            "flipping to enforcing."
        )
    if n_high == 0:
        return (
            "  No high-severity signals — flipping to enforcing is "
            "low-risk insurance. Operator can flip via React /cp/"
            "settings → Goodhart card."
        )
    return (
        f"  {n_high} high-severity signal(s) would have blocked "
        f"promotions in enforcing mode. Inspect the sample "
        f"descriptions above. If they look like real gaming "
        f"attempts → flip to enforcing. If they look like detector "
        f"noise → tune thresholds in app/goodhart_guard.py first."
    )


def _main() -> int:
    """``python -m app.observability.goodhart_advisory_report``"""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-days", type=int, default=30,
        help="Days of history to aggregate (default 30).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of formatted text.",
    )
    args = parser.parse_args()
    data = report(window_days=args.window_days)
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(_format_for_stdout(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
