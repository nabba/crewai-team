"""Quarterly self-improvement velocity digest — automated consumer
for the Theme 4 velocity surface.

PROGRAM §51 Q16.1 Item 9. The original Theme 4 (commit 4c135b02)
shipped a velocity aggregator + REST endpoint but no automated
consumer — operator had to poll the endpoint manually. This
module closes the loop: a quarterly idle job that surfaces the
current quarter's velocity into a Signal digest and writes the
snapshot to ``workspace/self_improvement/velocity_digests/
<year>q<n>.md``.

What this digest does
=====================

  * Runs the velocity aggregator (window_days=90).
  * Compares the current snapshot to the most recent prior
    snapshot for the SAME quarter-over-quarter signal.
  * Surfaces the deltas in a markdown digest + Signal alert.
  * Stores snapshots in ``workspace/self_improvement/velocity_
    digests/<year>q<n>.md`` for year-over-year visibility.

What this digest deliberately doesn't do
========================================

  * No LLM. Pure aggregation.
  * No gating. The numbers INFORM the operator; nothing acts on them
    automatically.
  * No alarm on absolute thresholds. Only relative quarter-over-
    quarter shifts (e.g. applied-rate dropped >15pp) get
    highlighted. Catches drift, not noise.

Cadence: daily probe; internal quarterly cadence (≥80 days since
last digest).
Master switch: ``velocity_digest_enabled`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_STATE_FILE = "velocity_digest_state.json"
_MIN_DAYS_BETWEEN_DIGESTS = 80
_APPLIED_RATE_ALERT_DELTA = 0.15   # 15 percentage points
_CR_TOTAL_RATIO_ALERT = 2.0        # current/prior ratio


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_velocity_digest_enabled
        return get_velocity_digest_enabled()
    except Exception:
        return os.getenv(
            "VELOCITY_DIGEST_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "self_improvement" / _STATE_FILE


def _digests_dir() -> Path:
    return _workspace() / "self_improvement" / "velocity_digests"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_snapshot": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_snapshot": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug(
            "velocity_digest: state write failed", exc_info=True,
        )


def _current_quarter_label(now: float) -> str:
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}q{q}"


def _compute_deltas(
    current: dict[str, Any],
    prior: dict[str, Any],
) -> list[str]:
    """Return a list of operator-readable delta-bullets. Only items
    with material change are surfaced."""
    out: list[str] = []
    cur_crs = current.get("change_requests") or {}
    pri_crs = prior.get("change_requests") or {}

    # Applied-rate shift.
    cur_rate = cur_crs.get("applied_rate_overall")
    pri_rate = pri_crs.get("applied_rate_overall")
    if (
        isinstance(cur_rate, (int, float))
        and isinstance(pri_rate, (int, float))
    ):
        delta = cur_rate - pri_rate
        if abs(delta) >= _APPLIED_RATE_ALERT_DELTA:
            arrow = "▲" if delta > 0 else "▼"
            out.append(
                f"{arrow} **CR applied-rate** shifted by "
                f"{delta * 100:+.1f}pp ({pri_rate:.2%} → {cur_rate:.2%})"
            )

    # CR volume ratio.
    cur_total = int(cur_crs.get("n_total", 0) or 0)
    pri_total = int(pri_crs.get("n_total", 0) or 0)
    if pri_total > 0 and cur_total > 0:
        ratio = cur_total / pri_total
        if ratio >= _CR_TOTAL_RATIO_ALERT or ratio <= 1.0 / _CR_TOTAL_RATIO_ALERT:
            out.append(
                f"📊 **CR volume** changed {ratio:.1f}× "
                f"({pri_total} → {cur_total})"
            )

    # Recipe success-rate shift.
    cur_rec = current.get("recipes") or {}
    pri_rec = prior.get("recipes") or {}
    cur_rec_rate = cur_rec.get("global_success_rate")
    pri_rec_rate = pri_rec.get("global_success_rate")
    if (
        isinstance(cur_rec_rate, (int, float))
        and isinstance(pri_rec_rate, (int, float))
    ):
        delta = cur_rec_rate - pri_rec_rate
        if abs(delta) >= 0.10:
            arrow = "▲" if delta > 0 else "▼"
            out.append(
                f"{arrow} **Recipe success-rate** shifted by "
                f"{delta * 100:+.1f}pp "
                f"({pri_rec_rate:.2%} → {cur_rec_rate:.2%})"
            )

    # Architecture rollback-threshold count.
    cur_arch = current.get("architecture_adoption") or {}
    pri_arch = prior.get("architecture_adoption") or {}
    cur_below = int(cur_arch.get("below_rollback_threshold", 0) or 0)
    pri_below = int(pri_arch.get("below_rollback_threshold", 0) or 0)
    if cur_below > pri_below:
        out.append(
            f"⚠️ **Architecture-adoption rollback candidates** rose "
            f"from {pri_below} to {cur_below} requests below the 0.20 "
            f"adoption threshold"
        )

    # Lessons-learned growth.
    cur_lessons = current.get("lessons_learned") or {}
    pri_lessons = prior.get("lessons_learned") or {}
    cur_lessons_total = int(cur_lessons.get("n_total", 0) or 0)
    pri_lessons_total = int(pri_lessons.get("n_total", 0) or 0)
    if cur_lessons_total - pri_lessons_total >= 20:
        out.append(
            f"📚 **Lessons-learned KB** grew by "
            f"{cur_lessons_total - pri_lessons_total} entries "
            f"({pri_lessons_total} → {cur_lessons_total})"
        )

    return out


def compose_digest(*, now: Optional[float] = None) -> Optional[Path]:
    """Compose the current quarter's digest. Returns the written path
    on success, None on skip."""
    if not _enabled():
        return None
    try:
        from app.self_improvement.velocity import velocity_summary
        current = velocity_summary(window_days=90)
    except Exception:
        logger.debug(
            "velocity_digest: velocity_summary failed", exc_info=True,
        )
        return None
    cur = float(now) if now is not None else time.time()
    state = _read_state()
    prior = state.get("last_snapshot") or {}

    deltas = _compute_deltas(current, prior) if prior else []
    label = _current_quarter_label(cur)
    target = _digests_dir() / f"{label}.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    cr_block = current.get("change_requests") or {}
    rec_block = current.get("recipes") or {}
    arch_block = current.get("architecture_adoption") or {}
    lessons_block = current.get("lessons_learned") or {}
    forge_block = current.get("forge_graduations") or {}

    lines: list[str] = [
        f"# Self-improvement velocity digest — {label}",
        "",
        f"_Composed at {datetime.now(timezone.utc).isoformat()}._",
        "",
        "## Quarter at a glance",
        "",
        f"- Change requests: **{cr_block.get('n_total', 0)}** total; "
        f"applied-rate **{(cr_block.get('applied_rate_overall') or 0):.0%}**",
        f"- Active recipes: **{rec_block.get('n_active', 0)}**; "
        f"global success-rate **{(rec_block.get('global_success_rate') or 0):.0%}**",
        f"- Architecture-adoption probe: **{arch_block.get('n_measured', 0)}** measured; "
        f"**{arch_block.get('below_rollback_threshold', 0)}** below 0.20 threshold",
        f"- Lessons-learned KB: **{lessons_block.get('n_total', 0)}** entries "
        f"({lessons_block.get('n_added_last_30d', 0)} added last 30d)",
        f"- Forge graduations (90d): **{forge_block.get('n_last_90d', 0)}**",
        "",
    ]

    if deltas:
        lines.append("## Quarter-over-quarter deltas")
        lines.append("")
        for delta in deltas:
            lines.append(f"- {delta}")
        lines.append("")
    elif prior:
        lines.append(
            "No material quarter-over-quarter deltas. The system's "
            "self-improvement velocity is consistent with the prior "
            "quarter."
        )
        lines.append("")

    if cr_block.get("by_requestor"):
        lines.append("## Top CR requestors")
        lines.append("")
        for req, count in list(cr_block["by_requestor"].items())[:8]:
            lines.append(f"- `{req}`: {count}")
        lines.append("")

    lines.extend([
        "## What to look at next",
        "",
        "  - If applied-rate dropped: which requestor's CRs are being",
        "    rejected? `GET /api/cp/self-improvement/velocity?window_days=90`",
        "    has the breakdown.",
        "  - If architecture-rollback candidates rose: investigate the",
        "    relevant `architecture_adoption` healing-monitor alerts.",
        "  - If lessons-learned KB didn't grow: structured_diagnosis may",
        "    be silent (no errors) OR may be filing dups (consult HOT-1).",
        "",
        "Prior digests at `workspace/self_improvement/velocity_digests/`",
        "provide year-over-year visibility.",
        "",
    ])
    target.write_text("\n".join(lines), encoding="utf-8")
    state["last_snapshot"] = current
    state["last_run_at"] = cur
    _write_state(state)

    # Signal digest (failure-isolated).
    try:
        from app.notify import notify
        if deltas:
            body = (
                f"Velocity digest for {label}: "
                f"{len(deltas)} material change(s) since prior quarter.\n\n"
                + "\n".join(f"  • {d}" for d in deltas)
                + f"\n\nFull markdown: `{target.relative_to(_workspace())}`"
            )
        else:
            body = (
                f"Velocity digest for {label}: no material changes "
                f"since prior quarter. Snapshot written to "
                f"`{target.relative_to(_workspace())}`."
            )
        notify(
            title=f"📊 Velocity digest — {label}",
            body=body,
            url="/cp/self-improvement",
            topic=f"velocity_digest:{label}",
            critical=False,
            arbitrate=True,
        )
    except Exception:
        logger.debug(
            "velocity_digest: notify failed", exc_info=True,
        )
    return target


def run_once(*, now: Optional[float] = None) -> dict[str, Any]:
    """Idle-job entry. Quarterly cadence gate."""
    summary: dict[str, Any] = {"ran": False, "wrote": None}
    if not _enabled():
        summary["skipped"] = True
        return summary
    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last = float(state.get("last_run_at", 0))
    if last > 0 and (cur - last) < _MIN_DAYS_BETWEEN_DIGESTS * 86400:
        return summary
    summary["ran"] = True
    try:
        path = compose_digest(now=cur)
    except Exception:
        logger.debug(
            "velocity_digest: compose failed", exc_info=True,
        )
        path = None
    if path is not None:
        summary["wrote"] = str(path)
    return summary
