"""
calibration.py — Daily reflection cycle scaffold.

Phase 1 SCOPE: this module reads recent affect trace, replays the reference
panel, computes the healthy-dynamics predicate, and writes a reflection
report. It does NOT yet propose or apply calibration deltas — that is
Phase 2 work.

The scheduled task is registered separately in hooks.install() at process
startup; this module exposes only the cycle entry point.

Output: /app/workspace/affect/reflections/YYYY-MM-DD.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.affect.schemas import AffectState, utc_now_iso

logger = logging.getLogger(__name__)

from app.paths import (  # noqa: E402  workspace-aware paths
    AFFECT_ROOT as _AFFECT_DIR,
    AFFECT_TRACE as _TRACE_FILE,
    AFFECT_REFLECTIONS_DIR as _REFLECTIONS_DIR,
)


# ── Trace replay ────────────────────────────────────────────────────────────


def load_recent_trace(hours: int = 24) -> list[AffectState]:
    """Read the last N hours of affect trace as AffectState objects."""
    states, _ = load_recent_trace_with_viability(hours)
    return states


def load_recent_trace_with_viability(hours: int = 24) -> tuple[list[AffectState], list[dict]]:
    """Read the last N hours of trace; return (affect_states, viability_frames).

    Two parallel lists, same length. viability_frames are the raw dicts
    persisted alongside each affect snapshot in trace.jsonl. Used by the
    Phase-2 calibration backtest.
    """
    if not _TRACE_FILE.exists():
        return [], []
    cutoff = (datetime.now(timezone.utc).timestamp() - hours * 3600)
    states: list[AffectState] = []
    frames: list[dict] = []
    # Q3.1 (2026-05-11) — archive-aware iteration. The 24h default window
    # is fully served by the live file; long-window operator backfills
    # (e.g. annual calibration review) need the archive to see what was
    # actually there. The escalation is early-exit-guarded: the live
    # file's oldest entry tells us whether we need to walk archives.
    live_oldest_ts: float | None = None
    try:
        with _TRACE_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                affect = row.get("affect", {})
                viability = row.get("viability", {})
                ts_str = affect.get("ts", "")
                try:
                    ts_unix = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00"),
                    ).timestamp()
                except ValueError:
                    continue
                if live_oldest_ts is None or ts_unix < live_oldest_ts:
                    live_oldest_ts = ts_unix
                if ts_unix < cutoff:
                    continue
                states.append(AffectState(
                    valence=float(affect.get("valence", 0.0)),
                    arousal=float(affect.get("arousal", 0.0)),
                    controllability=float(affect.get("controllability", 0.5)),
                    valence_source=str(affect.get("valence_source", "")),
                    arousal_source=str(affect.get("arousal_source", "")),
                    controllability_source=str(affect.get("controllability_source", "")),
                    attractor=str(affect.get("attractor", "neutral")),
                    internal_state_id=affect.get("internal_state_id"),
                    viability_frame_ts=affect.get("viability_frame_ts"),
                    ts=ts_str,
                ))
                frames.append(viability)
    except Exception:
        logger.debug("calibration: trace read failed", exc_info=True)

    # If the live file's earliest entry is already older than cutoff,
    # we have full coverage. Else escalate to the archive.
    if live_oldest_ts is None or live_oldest_ts <= cutoff:
        return states, frames

    try:
        from app.utils.jsonl_retention import read_archive
        archive_states: list[AffectState] = []
        archive_frames: list[dict] = []
        for line in read_archive(_TRACE_FILE, include_live=False):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            affect = row.get("affect", {})
            viability = row.get("viability", {})
            ts_str = affect.get("ts", "")
            try:
                ts_unix = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00"),
                ).timestamp()
            except ValueError:
                continue
            if ts_unix < cutoff:
                continue
            archive_states.append(AffectState(
                valence=float(affect.get("valence", 0.0)),
                arousal=float(affect.get("arousal", 0.0)),
                controllability=float(affect.get("controllability", 0.5)),
                valence_source=str(affect.get("valence_source", "")),
                arousal_source=str(affect.get("arousal_source", "")),
                controllability_source=str(affect.get("controllability_source", "")),
                attractor=str(affect.get("attractor", "neutral")),
                internal_state_id=affect.get("internal_state_id"),
                viability_frame_ts=affect.get("viability_frame_ts"),
                ts=ts_str,
            ))
            archive_frames.append(viability)
        # Archives are oldest-first, live is appended after.
        return archive_states + states, archive_frames + frames
    except Exception:
        logger.debug("calibration: archive read failed", exc_info=True)
        return states, frames


# ── Reflection cycle entry ──────────────────────────────────────────────────


def run_reflection_cycle(window_hours: int = 24) -> dict:
    """Run the daily reflection. Returns the report dict (also written to disk).

    Phase 2: calibration deltas are now proposed and (when guardrails pass)
    applied. The 6-guardrail flow lives in calibration_proposals.evaluate_and_apply.
    """
    from app.affect.welfare import (
        healthy_dynamics_predicate,
        maybe_audit_monotonic_drift,
        read_audit,
    )
    from app.affect.reference_panel import replay_panel
    from app.affect.calibration_proposals import evaluate_and_apply

    window, viability_frames = load_recent_trace_with_viability(hours=window_hours)
    if window:
        valences = [s.valence for s in window]
        arousals = [s.arousal for s in window]
        controllabilities = [s.controllability for s in window]
        attractor_counts: dict[str, int] = {}
        for s in window:
            attractor_counts[s.attractor] = attractor_counts.get(s.attractor, 0) + 1

        stats = {
            "n": len(window),
            "mean_valence": round(sum(valences) / len(valences), 4),
            "mean_arousal": round(sum(arousals) / len(arousals), 4),
            "mean_controllability": round(sum(controllabilities) / len(controllabilities), 4),
            "attractor_counts": attractor_counts,
        }
    else:
        stats = {"n": 0}

    healthy, diags = healthy_dynamics_predicate(window) if window else (False, {"reason": "no_trace"})

    panel_results = replay_panel()
    drift_counts = {"ok": 0, "numbness": 0, "over_reactive": 0, "wrong_attractor": 0, "drift": 0}
    for r in panel_results:
        drift_counts[r.drift_signature] = drift_counts.get(r.drift_signature, 0) + 1

    audit_window = read_audit(limit=200)
    # Filter to window
    audit_in_window = []
    cutoff = (datetime.now(timezone.utc).timestamp() - window_hours * 3600)
    for a in audit_window:
        try:
            if datetime.fromisoformat(a.get("ts", "").replace("Z", "+00:00")).timestamp() >= cutoff:
                audit_in_window.append(a)
        except (ValueError, AttributeError):
            continue

    # Long-window monotonic drift: consume the L9 daily snapshots
    # written by app.affect.l9_snapshots at 04:35. Closes the loop
    # between observability (snapshot writer) and welfare (slow-drift
    # detector), which previously had no consumer.
    monotonic_drift = {}
    try:
        monotonic_drift = maybe_audit_monotonic_drift()
    except Exception:
        logger.debug("affect.calibration: monotonic_drift_check failed", exc_info=True)

    report = {
        "ts": utc_now_iso(),
        "window_hours": window_hours,
        "stats": stats,
        "healthy_dynamics": {"passes": healthy, "diagnostics": diags},
        "monotonic_drift": monotonic_drift,
        "reference_panel": {
            "drift_counts": drift_counts,
            "results": [r.to_dict() for r in panel_results],
        },
        "welfare_audit_in_window": audit_in_window,
        "calibration_proposal": evaluate_and_apply(
            affect_history=window,
            viability_window=viability_frames,
        ),
        "attachment": _check_attachment_at_reflection(),
    }

    _write_report(report)

    # ── Retention / compaction (runs once per day) ─────────────────────
    # Tied to the daily reflection cycle so we don't need a separate
    # cron entry. All operations are best-effort + audit-logged; a
    # rotation failure must never block the reflection report itself.
    rotation = {}
    try:
        rotation["trace"] = rotate_trace_jsonl(retain_days=7, archive=True)
    except Exception:
        logger.debug("affect.calibration: trace rotation failed", exc_info=True)
    try:
        rotation["phase5_proposals"] = compact_phase5_proposals(
            stale_pending_days=14, drop_reviewed_after_days=30,
        )
    except Exception:
        logger.debug("affect.calibration: phase5 compaction failed", exc_info=True)
    if rotation:
        report["retention"] = rotation

    logger.info(
        f"affect.calibration: reflection complete — n={stats.get('n', 0)} healthy={healthy} "
        f"drift={drift_counts} retention={rotation}"
    )
    return report


# ── Retention / compaction helpers ───────────────────────────────────────────
#
# These keep the affect persistence files bounded. Every file the affect
# layer writes accumulates indefinitely without maintenance; the daily
# reflection cycle is the natural place to do the maintenance because it
# already loads + analyzes the recent window, so the marginal cost of
# a single archival pass is small.


def rotate_trace_jsonl(
    retain_days: int = 7,
    archive: bool = True,
) -> dict:
    """Rotate trace.jsonl: keep only the last `retain_days` of entries
    in the live file, archive the rest as a daily-bucketed gzip.

    Archive layout: AFFECT_ROOT/trace_archive/YYYY-MM.jsonl.gz
    (one file per UTC month, append-mode so multi-day archives merge).

    Returns a small report dict for the reflection log:
        {"kept": N, "archived": M, "archive_files": [...], "skipped": "..."}

    Never raises. If the trace file doesn't exist, returns {"skipped": "no trace"}.
    """
    if not _TRACE_FILE.exists():
        return {"skipped": "no trace"}

    cutoff_ts = datetime.now(timezone.utc).timestamp() - retain_days * 86400
    keep_lines: list[str] = []
    archive_buckets: dict[str, list[str]] = {}  # YYYY-MM → [lines]

    try:
        with _TRACE_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                # Cheap parse: only need the affect.ts to decide.
                try:
                    row = json.loads(stripped)
                    ts_str = row.get("affect", {}).get("ts", "")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except (json.JSONDecodeError, ValueError, AttributeError):
                    # Malformed line — keep it so a human can inspect later.
                    keep_lines.append(line if line.endswith("\n") else line + "\n")
                    continue
                if ts >= cutoff_ts:
                    keep_lines.append(line if line.endswith("\n") else line + "\n")
                else:
                    bucket = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
                    archive_buckets.setdefault(bucket, []).append(
                        line if line.endswith("\n") else line + "\n"
                    )
    except Exception:
        logger.debug("rotate_trace_jsonl: read failed", exc_info=True)
        return {"skipped": "read failed"}

    archived_count = sum(len(v) for v in archive_buckets.values())
    archive_files: list[str] = []

    if archive and archive_buckets:
        import gzip
        archive_dir = _AFFECT_DIR / "trace_archive"
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            for bucket, lines in archive_buckets.items():
                target = archive_dir / f"{bucket}.jsonl.gz"
                # Append-mode gzip: preserves earlier entries from the
                # same month if rotation runs across multiple days.
                with gzip.open(target, "ab") as gz:
                    gz.write("".join(lines).encode("utf-8"))
                archive_files.append(target.name)
        except Exception:
            logger.debug("rotate_trace_jsonl: archive write failed", exc_info=True)
            # If archiving fails, fall through to NOT rewriting the live
            # file — preserve everything rather than risk data loss.
            return {
                "kept": len(keep_lines),
                "archived": 0,
                "skipped": "archive write failed; live file untouched",
            }

    if archived_count > 0:
        # Atomic rewrite of trace.jsonl with only the recent entries.
        try:
            tmp = _TRACE_FILE.with_suffix(_TRACE_FILE.suffix + ".tmp")
            tmp.write_text("".join(keep_lines), encoding="utf-8")
            tmp.replace(_TRACE_FILE)
        except Exception:
            logger.debug("rotate_trace_jsonl: live rewrite failed", exc_info=True)
            return {
                "kept": "<unknown>",
                "archived": archived_count,
                "skipped": "live rewrite failed",
            }

    return {
        "kept": len(keep_lines),
        "archived": archived_count,
        "archive_files": archive_files,
        "retain_days": retain_days,
    }


def compact_phase5_proposals(
    stale_pending_days: int = 14,
    drop_reviewed_after_days: int = 30,
) -> dict:
    """Compact app/affect/phase5_proposals.jsonl.

    Two policies enforced together so the file stays bounded:
      - Pending proposals older than `stale_pending_days` are flipped
        to status="auto_deferred" (still kept in the file, but no
        longer block the queue from progressing).
      - Reviewed proposals (status not in {"pending"}) older than
        `drop_reviewed_after_days` are dropped from the file. The
        decision is preserved in the welfare audit log so the trail
        survives compaction.

    Returns a report dict; never raises.
    """
    from app.paths import AFFECT_PHASE5_PROPOSALS as proposals_file
    if not proposals_file.exists():
        return {"skipped": "no proposals file"}

    now_ts = datetime.now(timezone.utc).timestamp()
    auto_defer_cutoff = now_ts - stale_pending_days * 86400
    drop_cutoff = now_ts - drop_reviewed_after_days * 86400

    kept: list[dict] = []
    auto_deferred = 0
    dropped = 0

    try:
        with proposals_file.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                status = str(row.get("status", "pending"))
                ts_str = str(row.get("ts", "") or row.get("submitted_at", ""))
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    ts = now_ts  # treat undated as fresh

                if status == "pending" and ts < auto_defer_cutoff:
                    row["status"] = "auto_deferred"
                    row["auto_defer_ts"] = utc_now_iso()
                    row["auto_defer_reason"] = (
                        f"pending > {stale_pending_days}d without human review"
                    )
                    auto_deferred += 1
                    kept.append(row)
                    continue

                if status not in ("pending",) and ts < drop_cutoff:
                    dropped += 1
                    # Audit-log the drop so the trail survives.
                    try:
                        from app.affect.welfare import _AUDIT_FILE  # type: ignore
                        from threading import Lock as _Lock
                        with proposals_file.open("a") if False else open(_AUDIT_FILE, "a", encoding="utf-8") as af:
                            af.write(json.dumps({
                                "kind": "phase5_proposal_archived",
                                "severity": "info",
                                "ts": utc_now_iso(),
                                "name": row.get("name") or row.get("feature_name", "?"),
                                "final_status": status,
                                "submitted": ts_str,
                            }) + "\n")
                    except Exception:
                        pass
                    continue

                kept.append(row)
    except Exception:
        logger.debug("compact_phase5_proposals: read failed", exc_info=True)
        return {"skipped": "read failed"}

    if auto_deferred or dropped:
        try:
            tmp = proposals_file.with_suffix(proposals_file.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for row in kept:
                    f.write(json.dumps(row, default=str) + "\n")
            tmp.replace(proposals_file)
        except Exception:
            logger.debug("compact_phase5_proposals: rewrite failed", exc_info=True)
            return {"skipped": "rewrite failed"}

    return {
        "kept": len(kept),
        "auto_deferred": auto_deferred,
        "dropped_reviewed": dropped,
        "stale_pending_days": stale_pending_days,
        "drop_reviewed_after_days": drop_reviewed_after_days,
    }


def _check_attachment_at_reflection() -> dict:
    """Phase 3: during the daily reflection, evaluate separation analog status
    and care policy modifiers. Latent only — no auto-actions are taken.
    """
    out: dict = {"phase": "phase-3", "candidates_generated": []}
    try:
        from app.affect.attachment import (
            check_separation_analog,
            list_all_others,
            primary_user_identity,
        )
        from app.affect.care_policies import current_modifiers

        # Generate (at most one per cooldown) check-in candidate for the primary user
        cand = check_separation_analog(primary_user_identity())
        if cand is not None:
            out["candidates_generated"].append(cand)

        out["modifiers"] = current_modifiers().to_dict()
        out["others"] = [m.to_dict() for m in list_all_others()]
    except Exception:
        logger.debug("affect.calibration: attachment check failed", exc_info=True)
    return out


def _write_report(report: dict) -> None:
    try:
        _REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = _REFLECTIONS_DIR / f"{date_str}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception:
        logger.error("affect.calibration: report write failed", exc_info=True)


def latest_report() -> dict | None:
    """Most recent reflection report, or None."""
    if not _REFLECTIONS_DIR.exists():
        return None
    try:
        files = sorted(_REFLECTIONS_DIR.glob("*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
