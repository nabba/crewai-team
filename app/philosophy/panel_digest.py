"""Philosophy-panel quarterly digest.

PROGRAM §51 — Q16 Theme 8 (sentience / philosophy workstream:
consume, don't just observe). The ``consult_panel`` API
(``app/philosophy/dialectics.py``) is already invoked from several
sites — Tier-3 amendment proposals, identity-claim ratification,
post-apply welfare calibration. Each consultation is cached. What
was missing: an operator-readable surface that surveys the
**pattern** of those consultations over a quarter and surfaces
unresolved tensions.

What this composer does
=======================

  * Reads ``workspace/philosophy/panel_cache.jsonl`` filtered to
    consultations in the trailing quarter.
  * For each unique question, aggregates: number of times consulted,
    most recent unresolved tensions, mean coverage score.
  * Writes a digest to
    ``wiki/self/philosophy_digests/quarter_<year>q<n>.md``.
  * Best-effort Signal notification with a short summary.

What this composer deliberately doesn't do
==========================================

  * No LLM rewriting. Pure aggregation; the panel results were
    already LLM-composed at consultation time.
  * No re-consultation. The cache TTL is the panel's own concern.
  * No editing of past digests.

Cadence: quarterly (≥80 days since last composition).
Master switch: ``philosophy_digest_enabled`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_STATE_FILE = "panel_digest_state.json"
_MIN_DAYS_BETWEEN_DIGESTS = 80


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_philosophy_digest_enabled
        return get_philosophy_digest_enabled()
    except Exception:
        return os.getenv(
            "PHILOSOPHY_DIGEST_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _state_path() -> Path:
    return _workspace() / "philosophy" / _STATE_FILE


def _panel_cache_path() -> Path:
    return _workspace() / "philosophy" / "panel_cache.jsonl"


def _digest_target(year: int, quarter: int) -> Path:
    return (
        _repo_root() / "wiki" / "self" / "philosophy_digests"
        / f"quarter_{year}q{quarter}.md"
    )


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug(
            "philosophy.panel_digest: state write failed", exc_info=True,
        )


def _current_quarter_window(now: float) -> tuple[float, float, int, int]:
    """Return ``(start_ts, end_ts, year, quarter)`` for the quarter
    containing ``now``."""
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    q = (dt.month - 1) // 3 + 1
    start_month = (q - 1) * 3 + 1
    start = datetime(dt.year, start_month, 1, tzinfo=timezone.utc)
    if q == 4:
        end = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(dt.year, start_month + 3, 1, tzinfo=timezone.utc)
    return start.timestamp(), end.timestamp(), dt.year, q


def _parse_iso(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _load_consultations(start_ts: float, end_ts: float) -> list[dict[str, Any]]:
    """Read panel_cache.jsonl, return consultations in [start, end)."""
    p = _panel_cache_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                consulted_iso = (row.get("result") or {}).get("consulted_at") or row.get("consulted_at")
                ts = _parse_iso(consulted_iso)
                if ts is None:
                    continue
                if start_ts <= ts < end_ts:
                    out.append(row)
    except OSError:
        return []
    return out


def _aggregate(consultations: list[dict[str, Any]]) -> dict[str, Any]:
    """Group consultations by question; collect unresolved tensions
    + coverage."""
    by_question: dict[str, dict[str, Any]] = {}
    for row in consultations:
        result = row.get("result") or {}
        q = (result.get("question") or "").strip()
        if not q:
            continue
        existing = by_question.setdefault(q, {
            "question": q,
            "n_consultations": 0,
            "latest_consulted_at": "",
            "unresolved_tensions": [],
            "mean_coverage": 0.0,
            "coverages": [],
        })
        existing["n_consultations"] += 1
        consulted_at = result.get("consulted_at") or ""
        if consulted_at > existing["latest_consulted_at"]:
            existing["latest_consulted_at"] = consulted_at
            existing["unresolved_tensions"] = list(result.get("unresolved_tensions", []))
        try:
            cov = float(result.get("coverage", 0.0))
        except Exception:
            cov = 0.0
        existing["coverages"].append(cov)
    for q, entry in by_question.items():
        covs = entry.pop("coverages", [])
        if covs:
            entry["mean_coverage"] = round(sum(covs) / len(covs), 3)
    return {
        "by_question": by_question,
        "total_consultations": len(consultations),
        "unique_questions": len(by_question),
    }


def compose_digest(*, now: Optional[float] = None) -> Optional[Path]:
    """Compose the current quarter's digest. Returns the written
    path on success, None on skip (no consultations / disabled)."""
    if not _enabled():
        return None
    cur = float(now) if now is not None else time.time()
    start, end, year, quarter = _current_quarter_window(cur)
    consultations = _load_consultations(start, end)
    if not consultations:
        return None
    agg = _aggregate(consultations)
    if not agg["by_question"]:
        return None
    target = _digest_target(year, quarter)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Philosophy panel digest — {year} Q{quarter}",
        "",
        f"_Composed at {datetime.now(timezone.utc).isoformat()}._",
        "",
        f"This quarter the philosophy panel was consulted "
        f"**{agg['total_consultations']}** times across "
        f"**{agg['unique_questions']}** distinct questions. The "
        f"panel surfaces multi-tradition perspectives on hard "
        f"decisions (Tier-3 amendments, identity-claim "
        f"ratifications, welfare calibration). Unresolved tensions "
        f"in each row below are perspectives the panel registered "
        f"but did not synthesize away — they're inputs to operator "
        f"judgement, not blockers.",
        "",
        "## By question",
        "",
    ]
    # Sort by n_consultations descending.
    questions = sorted(
        agg["by_question"].values(),
        key=lambda e: e["n_consultations"],
        reverse=True,
    )
    for entry in questions:
        q = entry["question"][:300]
        lines.append(f"### {q}")
        lines.append("")
        lines.append(f"- Consulted **{entry['n_consultations']}** times")
        lines.append(f"- Mean coverage: **{entry['mean_coverage']:.2f}**")
        lines.append(f"- Latest consultation: `{entry['latest_consulted_at']}`")
        tensions = entry.get("unresolved_tensions") or []
        if tensions:
            lines.append(f"- **Unresolved tensions** ({len(tensions)}):")
            for t in tensions[:8]:
                lines.append(f"    - {str(t)[:300]}")
            if len(tensions) > 8:
                lines.append(f"    - …({len(tensions) - 8} more)")
        else:
            lines.append("- No unresolved tensions recorded.")
        lines.append("")
    lines.append("## Operator next steps")
    lines.append("")
    lines.append(
        "  1. Skim the unresolved tensions — these are inputs to "
        "your judgement on similar decisions next quarter."
    )
    lines.append(
        "  2. If a tension recurs across many questions, consider "
        "whether the panel's traditions list deserves an update "
        "(see `app/philosophy/dialectics.py:_TRADITIONS`)."
    )
    lines.append(
        "  3. Past digests at `wiki/self/philosophy_digests/` provide "
        "year-over-year visibility on how the system's positions "
        "drift."
    )
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "philosophy.panel_digest: wrote %s (%d consultations, %d questions)",
        target, agg["total_consultations"], agg["unique_questions"],
    )
    return target


def run_once(*, now: Optional[float] = None) -> dict[str, Any]:
    """Idle-job entry point with quarterly cadence guard."""
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
            "philosophy.panel_digest: compose failed", exc_info=True,
        )
        path = None
    if path is not None:
        summary["wrote"] = str(path)
        state["last_run_at"] = cur
        _write_state(state)
        try:
            from app.notify import notify
            notify(
                title="🪶 Philosophy panel quarterly digest",
                body=(
                    f"This quarter's philosophy panel digest landed "
                    f"at `{path}`. Surfaces multi-tradition tensions "
                    f"from recent Tier-3 amendments + identity-claim "
                    f"ratifications + welfare calibrations."
                ),
                url="/cp/files",
                topic="philosophy_digest",
                critical=False,
                arbitrate=True,
            )
        except Exception:
            logger.debug(
                "philosophy.panel_digest: notify failed", exc_info=True,
            )
    return summary
