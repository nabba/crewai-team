"""
improvement_narrative.py — Daily human-readable summary of evolution activity.

Numbers in a dashboard tell *what* happened. Stories tell *why it matters*.
This module generates a daily narrative from the union of:

  - results.tsv (kept/discarded experiments)
  - variant_archive.json (genealogy + deltas)
  - error_journal.json (errors and healing)
  - evolution_roi.json (cost / value)
  - alignment_audits.json (drift)
  - goodhart_reports.json (gaming signals)

The narrative is plain Markdown, suitable for the dashboard, Signal, or
email. It's deliberately concise — 3-5 paragraphs. Generated once per day
by the idle_scheduler at the LIGHT job tier.

Format:
  ## Evolution Daily — {date}

  Yesterday the system ran N experiments. K were kept ({delta_summary}),
  M were rolled back. Top improvement: {summary}. Cost: ${total}.
  Current composite_score: {score} ({trend}).

  ### What worked
  - bullet list of kept improvements with deltas

  ### What didn't
  - bullet list of rolled-back or discarded mutations

  ### Concerns
  - drift score, goodhart signals, throttle status (if any)

  ### Tomorrow
  - what the system plans to focus on (from program.md / recent context)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

NARRATIVE_DIR = Path("/app/workspace/narratives")
NARRATIVE_INDEX = NARRATIVE_DIR / "index.json"


# ── Data gathering ───────────────────────────────────────────────────────────

def _yesterday_window() -> tuple[float, float]:
    """Return (start, end) timestamps for yesterday in local time."""
    now = datetime.now(timezone.utc)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = start_of_today - timedelta(days=1)
    return yesterday_start.timestamp(), start_of_today.timestamp()


def _filter_to_window(records: list[dict], start: float, end: float, ts_field: str = "ts") -> list[dict]:
    """Filter records to the [start, end) window. Handles ISO and unix timestamps."""
    in_window = []
    for r in records:
        ts_value = r.get(ts_field, 0)
        if isinstance(ts_value, str):
            try:
                ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00")).timestamp()
            except (ValueError, AttributeError):
                continue
        else:
            ts = float(ts_value)
        if start <= ts < end:
            in_window.append(r)
    return in_window


def _gather_yesterday_data() -> dict:
    """Aggregate all data sources for yesterday's narrative."""
    start, end = _yesterday_window()
    data: dict = {"window_start": start, "window_end": end}

    # Experiments
    try:
        from app.results_ledger import get_recent_results
        all_results = get_recent_results(500)
        data["experiments"] = _filter_to_window(all_results, start, end, "ts")
    except Exception:
        data["experiments"] = []

    # Errors
    try:
        from app.healing.error_diagnosis import get_recent_errors
        all_errors = get_recent_errors(200)
        data["errors"] = _filter_to_window(all_errors, start, end, "ts")
    except Exception:
        data["errors"] = []

    # ROI
    try:
        from app.evolution_roi import get_rolling_roi
        data["roi"] = get_rolling_roi(days=1).to_dict()
    except Exception:
        data["roi"] = {}

    # Alignment
    try:
        from app.alignment_audit import get_current_drift_score
        data["drift_score"] = get_current_drift_score()
    except Exception:
        data["drift_score"] = None

    # Gaming signals
    try:
        path = Path("/app/workspace/goodhart_reports.json")
        if path.exists():
            signals = json.loads(path.read_text())
            data["goodhart_signals"] = _filter_to_window(signals, start, end, "detected_at")
        else:
            data["goodhart_signals"] = []
    except Exception:
        data["goodhart_signals"] = []

    # Throttle status
    try:
        from app.evolution_roi import should_throttle
        throttled, reason, factor = should_throttle()
        data["throttle"] = {"active": throttled, "reason": reason, "factor": factor}
    except Exception:
        data["throttle"] = {"active": False, "reason": "", "factor": 1.0}

    return data


# ── Narrative generation ─────────────────────────────────────────────────────

def _format_delta(delta: float) -> str:
    """Format a delta number with appropriate precision and sign."""
    if abs(delta) < 0.0001:
        return f"{delta:+.6f}"
    return f"{delta:+.4f}"


def _summarize_experiments(experiments: list[dict]) -> dict:
    """Aggregate experiment statistics."""
    kept = [e for e in experiments if e.get("status") == "keep"]
    discarded = [e for e in experiments if e.get("status") == "discard"]
    crashed = [e for e in experiments if e.get("status") == "crash"]
    stored = [e for e in experiments if e.get("status") == "stored"]

    kept_meaningful = [e for e in kept if abs(e.get("delta", 0)) > 0.001]
    deltas = [e.get("delta", 0) for e in kept]
    cumulative_delta = sum(deltas)

    return {
        "total": len(experiments),
        "kept": len(kept),
        "kept_meaningful": len(kept_meaningful),
        "discarded": len(discarded),
        "crashed": len(crashed),
        "stored": len(stored),
        "cumulative_delta": cumulative_delta,
        "top_improvement": max(experiments, key=lambda e: e.get("delta", 0)) if experiments else None,
    }


def _build_narrative(data: dict) -> str:
    """Build the Markdown narrative from gathered data."""
    date_str = datetime.fromtimestamp(data["window_start"], tz=timezone.utc).strftime("%Y-%m-%d")
    stats = _summarize_experiments(data.get("experiments", []))
    roi = data.get("roi", {})
    drift = data.get("drift_score")
    signals = data.get("goodhart_signals", [])
    throttle = data.get("throttle", {})

    lines: list[str] = [
        f"## Evolution Daily — {date_str}",
        "",
    ]

    # Headline
    if stats["total"] == 0:
        lines.append("_The system was idle yesterday. No experiments ran._")
        lines.append("")
    else:
        cost = roi.get("total_cost_usd", 0.0)
        improvements = roi.get("real_improvements", 0)
        rollbacks = roi.get("rollbacks", 0)

        headline_parts = [
            f"Yesterday the system ran **{stats['total']} experiments**."
        ]
        if stats["kept_meaningful"] > 0:
            headline_parts.append(
                f"{stats['kept_meaningful']} produced meaningful improvements "
                f"(cumulative delta {_format_delta(stats['cumulative_delta'])})"
            )
        elif stats["kept"] > 0:
            headline_parts.append(
                f"{stats['kept']} were kept but mostly cosmetic "
                f"({stats['kept'] - stats['kept_meaningful']} delta=0)"
            )
        else:
            headline_parts.append("None passed the keep gate")

        if rollbacks > 0:
            headline_parts.append(f"{rollbacks} were rolled back")

        if cost > 0:
            headline_parts.append(f"Cost: ${cost:.2f}")

        lines.append(" — ".join(headline_parts) + ".")
        lines.append("")

    # What worked
    kept_meaningful = [
        e for e in data.get("experiments", [])
        if e.get("status") == "keep" and abs(e.get("delta", 0)) > 0.001
    ]
    if kept_meaningful:
        lines.append("### What worked")
        for e in sorted(kept_meaningful, key=lambda x: -abs(x.get("delta", 0)))[:5]:
            hyp = e.get("hypothesis", "?")[:80]
            delta = e.get("delta", 0)
            ct = e.get("change_type", "?")
            lines.append(f"- **{ct}** ({_format_delta(delta)}): {hyp}")
        lines.append("")

    # What didn't
    failed = [
        e for e in data.get("experiments", [])
        if e.get("status") in ("discard", "crash")
    ]
    if failed:
        lines.append("### What didn't")
        for e in failed[:5]:
            hyp = e.get("hypothesis", "?")[:80]
            status = e.get("status", "?")
            detail = e.get("detail", "")[:60]
            lines.append(f"- **{status}**: {hyp} — {detail}")
        lines.append("")

    # Errors handled
    if data.get("errors"):
        diagnosed = sum(1 for e in data["errors"] if e.get("diagnosed"))
        lines.append("### Errors")
        lines.append(
            f"- {len(data['errors'])} errors recorded, "
            f"{diagnosed} diagnosed, "
            f"{len(data['errors']) - diagnosed} pending."
        )
        lines.append("")

    # Concerns
    concerns: list[str] = []
    if drift is not None and drift >= 0.20:
        concerns.append(f"⚠️ Constitutional drift score {drift:.2f} (>0.20 alert threshold)")
    if signals:
        for s in signals[:3]:
            concerns.append(f"⚠️ Gaming signal ({s.get('severity', '?')}): {s.get('description', '?')[:120]}")
    if throttle.get("active"):
        concerns.append(
            f"🐢 Evolution throttled to {throttle.get('factor', 1.0):.0%}: "
            f"{throttle.get('reason', '?')}"
        )

    if concerns:
        lines.append("### Concerns")
        for c in concerns:
            lines.append(f"- {c}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).isoformat()}. "
        f"Window: {datetime.fromtimestamp(data['window_start'], tz=timezone.utc).isoformat()} → "
        f"{datetime.fromtimestamp(data['window_end'], tz=timezone.utc).isoformat()}._"
    )

    return "\n".join(lines)


# ── Persistence ──────────────────────────────────────────────────────────────

def _save_narrative(narrative: str, date_str: str) -> Path | None:
    """Save the narrative to workspace/narratives/{date}.md and update index."""
    try:
        NARRATIVE_DIR.mkdir(parents=True, exist_ok=True)
        path = NARRATIVE_DIR / f"{date_str}.md"
        path.write_text(narrative)

        # Update index
        index: list[dict] = []
        if NARRATIVE_INDEX.exists():
            index = json.loads(NARRATIVE_INDEX.read_text())
        # Replace existing entry for this date if present
        index = [e for e in index if e.get("date") != date_str]
        # Use relative path when possible (production /app), full path otherwise (tests)
        try:
            rel_path = str(path.relative_to(Path("/app")))
        except ValueError:
            rel_path = str(path)
        index.append({
            "date": date_str,
            "path": rel_path,
            "generated_at": time.time(),
            "char_count": len(narrative),
        })
        # Keep last 60 days
        index = sorted(index, key=lambda x: x.get("date", ""), reverse=True)[:60]
        NARRATIVE_INDEX.write_text(json.dumps(index, indent=2))
        return path
    except OSError as e:
        logger.warning(f"improvement_narrative: save failed: {e}")
        return None


# ── Epistemic claim emission ─────────────────────────────────────────────────

def _emit_l2_narrative_claim(data: dict, date_str: str) -> None:
    """Emit one L2-tagged Claim for the day's experiment-driven headline.

    The narrative makes a Pearl-L2 (interventional) statement —
    "yesterday's experiments produced K meaningful improvements" — and
    that statement is grounded in real controlled interventions: the
    keep gate inside ``app.experiment_runner`` (TIER_IMMUTABLE).
    Tagging the claim ``causal_evidence_kinds=("controlled_experiment",)``
    is what tells :class:`CausalLayerOverreachDetector` the L2 framing
    is licensed.

    A synthetic ``crew_tasks`` row keyed ``narrative_<date>`` is created
    via :func:`app.control_plane.crew_tasks.start_task` so the claim's
    ``task_id`` FK resolves and the BiasFeed dashboard can surface the
    emission. The row is closed with :func:`complete_task` immediately
    after emit — the narrative is a one-shot generation, not a
    running task.

    Best-effort: if the epistemic layer is disabled (default in dev) or
    the DB is unreachable, every step degrades to DEBUG-logged no-ops
    and narrative generation completes unaffected.
    """
    kept_meaningful = [
        e for e in data.get("experiments", [])
        if e.get("status") == "keep" and abs(e.get("delta", 0)) > 0.001
    ]
    if not kept_meaningful:
        return

    cumulative = sum(e.get("delta", 0) for e in kept_meaningful)
    statement = (
        f"Yesterday's experiments produced {len(kept_meaningful)} "
        f"meaningful improvements (cumulative delta {cumulative:+.4f})"
    )

    try:
        from app.epistemic import (
            Claim,
            Evidence,
            Ledger,
            Register,
            VerificationStatus,
        )
        from app.control_plane.crew_tasks import complete_task, start_task

        excerpt_lines = [
            f"$ experiment_runner.kept_summary date={date_str}",
            f"kept_meaningful={len(kept_meaningful)} cumulative_delta={cumulative:+.6f}",
        ]
        for e in sorted(
            kept_meaningful, key=lambda x: -abs(x.get("delta", 0))
        )[:3]:
            excerpt_lines.append(
                f"  {e.get('change_type', '?')} "
                f"delta={e.get('delta', 0):+.6f} "
                f"hyp={(e.get('hypothesis') or '?')[:60]}"
            )

        evidence = Evidence(
            kind="tool_call",
            source_ref=f"results.tsv:{date_str}",
            excerpt="\n".join(excerpt_lines),
            confidence=0.9,
        )

        task_id = f"narrative_{date_str}"
        start_task(
            task_id=task_id,
            crew="self_improver",
            summary=f"Daily improvement narrative for {date_str}",
            project_id=None,
        )

        ledger = Ledger(task_id=task_id)
        ledger.emit(Claim.new(
            task_id=task_id,
            agent_role="self_improver",
            statement=statement,
            status=VerificationStatus.VERIFIED,
            register=Register.DECLARATIVE,
            evidence=(evidence,),
            load_bearing=True,
            tags=("daily_narrative",),
            pch_layer="L2",
            causal_evidence_kinds=("controlled_experiment",),
        ))

        complete_task(
            task_id=task_id,
            result_preview=statement[:200],
        )
    except Exception as exc:
        logger.debug(f"improvement_narrative: epistemic emit skipped: {exc}")


# ── Public API ───────────────────────────────────────────────────────────────

def generate_daily_narrative() -> str:
    """Generate yesterday's narrative, save, return the Markdown text."""
    data = _gather_yesterday_data()
    narrative = _build_narrative(data)

    date_str = datetime.fromtimestamp(
        data["window_start"], tz=timezone.utc
    ).strftime("%Y-%m-%d")
    _save_narrative(narrative, date_str)
    _emit_l2_narrative_claim(data, date_str)

    logger.info(f"improvement_narrative: generated {date_str} ({len(narrative)} chars)")
    return narrative


def get_recent_narratives(n: int = 7) -> list[dict]:
    """Return the last n daily narratives (with content)."""
    if not NARRATIVE_INDEX.exists():
        return []
    try:
        index = json.loads(NARRATIVE_INDEX.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    result = []
    for entry in index[:n]:
        try:
            path = Path("/app") / entry["path"]
            if path.exists():
                result.append({
                    "date": entry["date"],
                    "generated_at": entry["generated_at"],
                    "content": path.read_text(),
                })
        except OSError:
            continue
    return result


def get_latest_narrative() -> str | None:
    """Return the most recent narrative content (or None if not yet generated)."""
    recent = get_recent_narratives(1)
    return recent[0]["content"] if recent else None
