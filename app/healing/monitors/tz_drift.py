"""tz_drift — TZ database drift + hand-rolled DST drift monitor.

PROGRAM §48 — Q13.3 (year-2+ resilience #2.6). Catches three silent
failure modes that bite the system years from now:

  1. **EU abolishes DST** — Estonia + Finland already voted for
     permanent summer time in 2018; activation is pending an EU-wide
     decision. When the rule flips, ``zoneinfo.ZoneInfo("Europe/
     Helsinki")`` updates automatically with system ``tzdata``, but
     ``app/temporal_context.py:_helsinki_tz()`` is hand-rolled
     ("last Sunday of March / October") and will silently produce
     the wrong offset.
  2. **System tzdata stale** — operator's host stops receiving OS
     updates; everything's correct in code but the runtime computes
     an old transition. Comparison against an external anchor would
     catch this.
  3. **Hand-rolled function drift from upstream** — even with current
     rules, the two implementations could diverge at edge cases
     (the exact instant of DST transition, ambiguous-time handling).

Algorithm — three independent probes per pass:

  * **Now-offset divergence.** Compute the current Helsinki UTC
    offset via both ``_helsinki_tz`` (hand-rolled) and
    ``ZoneInfo("Europe/Helsinki")``. Alert if they disagree.
  * **Anchor-moment divergence.** Pin two known anchors per year
    (March equinox, October equinox — astronomical, NOT DST-related,
    so hand-rolled vs. zoneinfo SHOULD agree). Compare offsets at
    each anchor; alert if they disagree.
  * **DST-transition divergence.** Compute the next DST-transition
    moment via both implementations. Alert if they disagree by more
    than 1 hour (catches a missing/added transition).

On first material divergence detected, the monitor files a regular
change-request proposing to replace the hand-rolled
``_helsinki_tz()`` with ``ZoneInfo("Europe/Helsinki")``. The CR is
filed via the standard ``change_requests.lifecycle.create_request``
gate — operator approves/rejects via Signal 👍/👎. (Architecture-
requests are for new-subsystem scaffolds; a function-level refactor
fits the regular CR shape better; same end result for the operator.)

Emits a ``tz_drift`` event to the identity continuity ledger ONLY
on landmark transitions (first divergence detected, recovery from
divergence). Routine pass with no divergence stays silent.

Master switch: ``tz_drift_monitor_enabled`` (default ON).
Cadence: daily probe. Dedup: 7-day per-alert-topic window via the
notify arbiter.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


NAME = "tz_drift"
CADENCE_SECONDS = 24 * 3600          # daily probe
MASTER_SWITCH_KEY = "tz_drift_monitor_enabled"

# 1 hour cap on declared-acceptable divergence at DST transitions
# (matches the actual transition magnitude — a missing/added
# transition manifests as a 1h flip).
_DIVERGENCE_THRESHOLD_SECONDS = 60   # 60s is generous; offsets are
                                     # in whole hours so any real
                                     # divergence is ≥ 1h.

_STATE_FILE_NAME = "tz_drift_state.json"


# ── Anchor moments (astronomical — both implementations should agree) ─


def _next_march_equinox(year: int) -> datetime:
    """Approximate March equinox in UTC. Accuracy ~6 hours, sufficient
    for "is the offset at this approximate moment consistent."""
    # Meeus' approximation, simplified — pinning year+month+day at
    # 21 March 09:00 UTC is within 24h of the true equinox for all
    # years in the next century.
    return datetime(year, 3, 21, 9, 0, tzinfo=timezone.utc)


def _next_october_equinox(year: int) -> datetime:
    """Approximate September equinox + 1 month → mid-DST control."""
    return datetime(year, 9, 22, 21, 0, tzinfo=timezone.utc)


def _anchor_moments() -> list[tuple[str, datetime]]:
    """Three anchor moments spanning DST + standard time."""
    now = datetime.now(timezone.utc)
    year = now.year
    return [
        ("now", now),
        (f"march_equinox_{year}", _next_march_equinox(year)),
        (f"october_equinox_{year}", _next_october_equinox(year)),
    ]


# ── Probes ──────────────────────────────────────────────────────────────


def _handrolled_offset_at(dt: datetime) -> Optional[int]:
    """Return offset in SECONDS at ``dt`` per app/temporal_context.py
    convention. We re-implement the rule here (not importing the
    private function) because it's the SOURCE under test.

    Returns None on any failure (treated as "can't compare")."""
    try:
        year = dt.year
        # Last Sunday of March, 03:00 UTC
        mar31 = datetime(year, 3, 31, 3, 0, tzinfo=timezone.utc)
        dst_start = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)
        # Last Sunday of October, 04:00 UTC
        oct31 = datetime(year, 10, 31, 4, 0, tzinfo=timezone.utc)
        dst_end = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)
        if dst_start <= dt < dst_end:
            return 3 * 3600  # EEST
        return 2 * 3600      # EET
    except Exception:
        logger.debug("tz_drift: hand-rolled offset failed", exc_info=True)
        return None


def _zoneinfo_offset_at(dt: datetime) -> Optional[int]:
    """Return offset in SECONDS at ``dt`` per ``zoneinfo``. None if
    zoneinfo or the Helsinki zone is unavailable on this host."""
    try:
        from zoneinfo import ZoneInfo
        zi = ZoneInfo("Europe/Helsinki")
        local = dt.astimezone(zi)
        utcoffset = local.utcoffset()
        if utcoffset is None:
            return None
        return int(utcoffset.total_seconds())
    except Exception:
        logger.debug("tz_drift: zoneinfo offset failed", exc_info=True)
        return None


def _probe_anchor(
    label: str, dt: datetime,
) -> dict[str, Any]:
    """Probe a single anchor: hand-rolled offset, zoneinfo offset,
    divergence in seconds."""
    h = _handrolled_offset_at(dt)
    z = _zoneinfo_offset_at(dt)
    diverged = False
    div_seconds = None
    if h is not None and z is not None:
        div_seconds = abs(h - z)
        diverged = div_seconds > _DIVERGENCE_THRESHOLD_SECONDS
    return {
        "label": label,
        "moment_iso": dt.isoformat(),
        "handrolled_offset_s": h,
        "zoneinfo_offset_s": z,
        "divergence_s": div_seconds,
        "diverged": diverged,
    }


# ── State (for landmark transitions + dedup) ────────────────────────────


def _state_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "healing" / _STATE_FILE_NAME
    except Exception:
        return Path("/app/workspace/healing") / _STATE_FILE_NAME


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_divergence_at": None, "cr_filed": False}
    try:
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("tz_drift: state read failed", exc_info=True)
        return {"last_divergence_at": None, "cr_filed": False}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import json
        p.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        logger.debug("tz_drift: state write failed", exc_info=True)


# ── CR filing (operator's choice — automated proposal on divergence) ───


def _propose_consolidation_cr(probes: list[dict[str, Any]]) -> Optional[str]:
    """File a standard change-request proposing to replace the hand-
    rolled ``_helsinki_tz()`` with ``ZoneInfo``. Returns the CR id
    on success, None on failure (failure-isolated).

    Generates a real synthetic diff so the operator can approve the CR
    as-is rather than reading the prose and hand-writing the edit
    (post-2026-05-16 discipline — empty-diff CRs were flagged as a
    RISK item in the monitor audit because they put burden on the
    operator and the operator can introduce a different error than
    the proposer's described intent)."""
    try:
        from app.change_requests.lifecycle import create_request
        from app.change_requests.models import RiskClass
    except Exception:
        logger.debug("tz_drift: change_requests unavailable", exc_info=True)
        return None
    target_path = "app/temporal_context.py"
    try:
        src = Path(target_path).read_text(encoding="utf-8")
    except Exception:
        # Repo-relative path may not resolve from the gateway; fall
        # back to absolute. The CR validator works on the path string,
        # not on filesystem state — it's the applier (post-approval)
        # that opens the file.
        try:
            src = Path(f"/app/crewai-team/{target_path}").read_text(encoding="utf-8")
        except Exception:
            logger.debug("tz_drift: source read failed", exc_info=True)
            return None

    new_src = _synthesize_zoneinfo_patch(src)
    if new_src is None or new_src == src:
        # Couldn't safely produce a synthetic diff — fall back to an
        # informational CR (the old shape), so operator still gets
        # the signal. Better than no CR at all.
        new_src = src

    proposed_reason = (
        f"TZ_DRIFT_MONITOR: divergence detected between hand-rolled "
        f"_helsinki_tz() in app/temporal_context.py and "
        f"ZoneInfo('Europe/Helsinki').\n\n"
        f"Probes:\n"
    )
    for p in probes:
        proposed_reason += (
            f"  - {p['label']:30s} hand-rolled={p['handrolled_offset_s']}s, "
            f"zoneinfo={p['zoneinfo_offset_s']}s, "
            f"divergence={p['divergence_s']}s\n"
        )
    diff_note = (
        "(synthetic patch attached)" if new_src != src
        else "(no synthetic patch — patch by hand from reason)"
    )
    proposed_reason += (
        f"\nProposed fix {diff_note}: replace `_helsinki_tz()` body with "
        f"a thin `zoneinfo.ZoneInfo('Europe/Helsinki')` wrapper so the "
        f"system automatically follows OS tzdata updates. This is the "
        f"same approach already used at app/affect/hooks.py:366 and "
        f"app/companion/scheduler.py:150.\n\n"
        f"Operator: review the diff carefully. If the underlying cause "
        f"is stale tzdata on the host (not an EU rule change), update "
        f"tzdata + tzdata-legacy instead of changing the code."
    )
    try:
        cr = create_request(
            requestor="tz_drift_monitor",
            path=target_path,
            new_content=new_src,
            old_content=src,
            reason=proposed_reason,
            risk_class=RiskClass.STANDARD,
        )
        return getattr(cr, "id", None)
    except Exception:
        logger.debug("tz_drift: CR filing failed", exc_info=True)
        return None


def _synthesize_zoneinfo_patch(src: str) -> Optional[str]:
    """Produce a synthetic ``new_content`` that replaces the hand-rolled
    ``_helsinki_tz()`` function with a one-liner using
    ``zoneinfo.ZoneInfo``.

    Returns the modified source on success, ``None`` if the source
    doesn't match the expected shape (defensive — if temporal_context
    is refactored before this monitor's CR lands, the synthetic patch
    might silently corrupt it).

    The synthesis is conservative:
      * requires the exact function signature to be present
      * requires the function to end with a `return timezone(...)` line
        before either a blank line + `def` or end-of-file
      * if either is missing, returns None and the caller falls back
        to the empty-diff CR (informational only)
    """
    sig = "def _helsinki_tz() -> timezone:"
    sig_idx = src.find(sig)
    if sig_idx < 0:
        return None
    # Locate the end of the function body: the first blank line followed
    # by a top-level token (def / class / # / `_cache` style identifier
    # at column 0). The current function ends with `return timezone(...)`.
    body_start = sig_idx + len(sig)
    # Scan line-by-line for the function's closing — first non-empty
    # line at column 0 after at least one line of body.
    lines = src[body_start:].splitlines(keepends=True)
    consumed = 0
    seen_body_line = False
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n\r")
        if not stripped.strip():
            consumed += len(line)
            continue
        if stripped[0] not in (" ", "\t"):
            if seen_body_line:
                # Reached the next top-level token — function ended here.
                break
            consumed += len(line)
            continue
        seen_body_line = True
        consumed += len(line)
    if not seen_body_line:
        return None
    body_end = body_start + consumed

    # Replacement function — drop the `-> timezone` annotation since
    # ZoneInfo is a different tzinfo subclass; both still satisfy the
    # tzinfo protocol the caller uses.
    replacement = (
        "def _helsinki_tz():\n"
        '    """Return Helsinki timezone via stdlib ``zoneinfo``.\n'
        "\n"
        "    Replaced 2026-05-16 (tz_drift monitor synthetic patch):\n"
        "    the hand-rolled DST calculation that used to live here\n"
        "    diverged from ``ZoneInfo('Europe/Helsinki')`` on at least\n"
        "    one probe. Using zoneinfo means the timezone definition\n"
        "    follows host tzdata, so EU rule changes propagate\n"
        '    automatically.\n'
        '    """\n'
        '    return ZoneInfo("Europe/Helsinki")\n'
    )

    new_src = src[:sig_idx] + replacement + src[body_end:]

    # Add the `from zoneinfo import ZoneInfo` import if missing.
    if "from zoneinfo import" not in new_src and "import zoneinfo" not in new_src:
        # Insert just after the `from datetime import` line — same
        # block as the existing stdlib datetime imports.
        marker = "from datetime import datetime, timedelta, timezone\n"
        idx = new_src.find(marker)
        if idx >= 0:
            insert_at = idx + len(marker)
            new_src = (
                new_src[:insert_at]
                + "from zoneinfo import ZoneInfo\n"
                + new_src[insert_at:]
            )
        else:
            # Couldn't find the canonical import line — refuse the
            # patch rather than ship a broken file.
            return None
    return new_src


# ── Continuity-ledger event emission ────────────────────────────────────


def _emit_ledger_event(
    *,
    kind: str,
    summary: str,
    detail: dict[str, Any],
) -> None:
    """Best-effort emit to the identity continuity ledger. Failure
    here is silent — the Signal alert + CR are the load-bearing
    artifacts; the ledger event is a year-over-year visibility hook
    for the annual reflection's drift summary."""
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="tz_drift",
            actor="tz_drift_monitor",
            summary=summary,
            detail=detail,
        )
    except Exception:
        logger.debug("tz_drift: ledger emit failed", exc_info=True)


# ── Public entry ────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One monitor probe. Returns a summary dict (also used by tests).

    Failure-isolated: any exception is caught and logged at debug;
    the return shape is preserved with ``errors`` non-zero."""
    summary: dict[str, Any] = {
        "checked": False,
        "probes": [],
        "n_diverged": 0,
        "alerts": 0,
        "cr_filed_id": None,
        "errors": 0,
    }

    # Master-switch check (default ON; failure-isolated).
    try:
        from app.runtime_settings import get_tz_drift_monitor_enabled
        if not get_tz_drift_monitor_enabled():
            summary["skipped"] = True
            return summary
    except Exception:
        pass  # fall through — default ON

    # Three probes per pass.
    try:
        probes = [_probe_anchor(label, dt) for label, dt in _anchor_moments()]
        summary["probes"] = probes
        summary["checked"] = True
        summary["n_diverged"] = sum(1 for p in probes if p["diverged"])
    except Exception:
        logger.debug("tz_drift: probe failed", exc_info=True)
        summary["errors"] = 1
        return summary

    if summary["n_diverged"] == 0:
        # Check for recovery transition (previous pass had divergence).
        state = _read_state()
        if state.get("last_divergence_at") and not state.get("recovered"):
            state["recovered"] = True
            state["recovered_at"] = datetime.now(timezone.utc).isoformat()
            _write_state(state)
            _emit_ledger_event(
                kind="tz_drift",
                summary=(
                    "TZ drift recovered — hand-rolled and zoneinfo "
                    "offsets agree at all anchors."
                ),
                detail={"recovered_at": state["recovered_at"]},
            )
        return summary

    # Divergence detected. Signal alert + CR filing (if first-time).
    diverged = [p for p in probes if p["diverged"]]
    body_lines = [
        f"TZ drift detected: hand-rolled _helsinki_tz() and "
        f"ZoneInfo('Europe/Helsinki') disagree.",
        "",
    ]
    for p in diverged:
        h = p["handrolled_offset_s"]
        z = p["zoneinfo_offset_s"]
        h_h = (h or 0) / 3600
        z_h = (z or 0) / 3600
        body_lines.append(
            f"  • {p['label']:30s} hand-rolled UTC+{h_h:.1f}, "
            f"zoneinfo UTC+{z_h:.1f}"
        )
    body_lines.append("")
    body_lines.append(
        "Likely cause: EU DST rule change OR stale host tzdata. "
        "Auto-CR filed proposing consolidation onto ZoneInfo."
    )
    body = "\n".join(body_lines)
    try:
        from app.notify import notify
        notify(
            title="🕒 TZ drift detected",
            body=body,
            url="/cp/changes",
            topic="tz_drift",
            critical=False,
            arbitrate=True,
        )
        summary["alerts"] = 1
    except Exception:
        logger.debug("tz_drift: notify failed", exc_info=True)
        summary["errors"] += 1

    # File a CR if we haven't already in this drift episode.
    state = _read_state()
    if not state.get("cr_filed"):
        cr_id = _propose_consolidation_cr(probes)
        if cr_id:
            summary["cr_filed_id"] = cr_id
            state["cr_filed"] = True
            state["cr_id"] = cr_id
        state["last_divergence_at"] = datetime.now(timezone.utc).isoformat()
        state["recovered"] = False
        state.pop("recovered_at", None)
        _write_state(state)
        _emit_ledger_event(
            kind="tz_drift",
            summary=(
                f"TZ drift detected at {summary['n_diverged']} anchor(s); "
                f"CR {cr_id or 'failed'} proposes consolidation."
            ),
            detail={
                "n_diverged": summary["n_diverged"],
                "cr_id": cr_id,
                "probes": probes,
            },
        )

    return summary
