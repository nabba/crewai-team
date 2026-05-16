"""End-of-vacation digest composer.

PROGRAM §51 — Q16 Theme 3 follow-on. When an engagement ends
(manual disengagement or auto-expiry), produce a single markdown
summary of everything that happened during the window:

  * Engagement window (start, end, duration_actual, reason).
  * Auto-apply log rows (from ``workspace/vacation_mode/auto_apply_log.jsonl``)
    filtered to the engagement window.
  * Aggregates: total auto-applied, by-requestor, by-path-prefix.
  * Any rate-limited rejections during the window.

The composer reads only what it needs (no external network calls,
no LLM). Output lands at ``workspace/vacation_mode/digests/
<engaged_at-iso>.md``. The path is deterministic so re-running
``compose_digest`` for the same engagement overwrites cleanly.

Failure-isolated: caller (``state.disengage``) wraps in try/except.
This module itself does not raise on file-read failures — it
produces a best-effort digest with whatever rows it could read.
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

from app.vacation_mode.state import VacationEngagement

logger = logging.getLogger(__name__)


_AUTO_APPLY_LOG = "auto_apply_log.jsonl"
_DIGEST_DIR = "digests"


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _log_path() -> Path:
    return _workspace() / "vacation_mode" / _AUTO_APPLY_LOG


def _digests_dir() -> Path:
    return _workspace() / "vacation_mode" / _DIGEST_DIR


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


def _load_rows_in_window(
    start_ts: float, end_ts: float,
) -> list[dict[str, Any]]:
    p = _log_path()
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
                ts = _parse_iso(row.get("ts"))
                if ts is None:
                    continue
                if start_ts <= ts <= end_ts:
                    out.append(row)
    except OSError:
        return []
    return out


def _common_prefix(path: str) -> str:
    """Pick a coarse bucket for the digest's per-prefix table —
    the first two path segments, or the whole path if shorter."""
    if not path:
        return "(unknown)"
    parts = path.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2]) + "/"
    return parts[0]


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def compose_digest(
    *,
    engagement: VacationEngagement,
    ended_at: float,
    output_dir: Optional[Path] = None,
) -> Path:
    """Compose a markdown digest for the closed engagement and write
    it to ``workspace/vacation_mode/digests/<iso>.md``. Returns the
    written path.

    Failure-isolated: never raises. Returns the path even if write
    failed (caller can decide whether to surface).
    """
    target_dir = output_dir if output_dir is not None else _digests_dir()
    start_ts = float(engagement.engaged_at)
    end_ts = float(ended_at)
    duration_s = max(0.0, end_ts - start_ts)

    rows = _load_rows_in_window(start_ts, end_ts)
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_failed = n_total - n_ok

    # By requestor.
    requestor_counts: Counter = Counter(
        (r.get("requestor") or "(unknown)") for r in rows
    )
    # By coarse path prefix.
    prefix_counts: Counter = Counter(
        _common_prefix(r.get("path") or "") for r in rows
    )

    start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()
    safe_slug = start_iso.replace(":", "-")

    lines = [
        f"# Vacation digest — {start_iso[:10]} → {end_iso[:10]}",
        "",
        "## Engagement",
        "",
        f"- Engaged by: `{engagement.engaged_by}`",
        f"- Started: `{start_iso}`",
        f"- Ended: `{end_iso}`",
        f"- Duration: `{_format_duration(duration_s)}`",
        f"- Reason: `{engagement.reason or '(none)'}`",
        f"- Allowlist requestors: `{', '.join(engagement.frozen_allowlist.requestor_allowlist) or '(empty)'}`",
        f"- Allowlist path prefixes: `{', '.join(engagement.frozen_allowlist.path_prefix_allowlist) or '(empty)'}`",
        f"- Max diff lines: `{engagement.frozen_allowlist.max_diff_lines}`",
        "",
        "## Auto-applies during the window",
        "",
        f"- Total: **{n_total}**",
        f"- Successful: **{n_ok}**",
        f"- Failed: **{n_failed}**",
    ]

    if requestor_counts:
        lines.extend([
            "",
            "### By requestor",
            "",
        ])
        for req, count in sorted(
            requestor_counts.items(), key=lambda kv: kv[1], reverse=True,
        ):
            lines.append(f"- `{req}`: {count}")

    if prefix_counts:
        lines.extend([
            "",
            "### By path prefix",
            "",
        ])
        for prefix, count in sorted(
            prefix_counts.items(), key=lambda kv: kv[1], reverse=True,
        ):
            lines.append(f"- `{prefix}`: {count}")

    if rows:
        lines.extend([
            "",
            "## Detail (newest first)",
            "",
            "| Time | Status | Requestor | Path | Error |",
            "| --- | --- | --- | --- | --- |",
        ])
        # Newest first.
        for row in sorted(rows, key=lambda r: r.get("ts", ""), reverse=True):
            ts = (row.get("ts") or "")[:19]
            status = "✅" if row.get("ok") else "⚠️"
            req = row.get("requestor") or "(unknown)"
            path = row.get("path") or ""
            error = (row.get("error") or "").replace("|", "\\|")[:80]
            lines.append(f"| {ts} | {status} | `{req}` | `{path}` | {error} |")

    if not rows:
        lines.extend([
            "",
            "*No auto-applies occurred during this window.*",
            "",
            "(This is a perfectly valid outcome — vacation mode is a "
            "safety-net that only acts when a matching CR happens to "
            "be filed during the window. Many engagements will have "
            "zero auto-applies.)",
        ])

    lines.extend([
        "",
        "## Next steps for the operator",
        "",
        "- Review each row above and ensure the change is what you'd",
        "  have approved manually.",
        "- Check the CR detail at `/cp/changes` for full diffs.",
        "- Any unexpected entry: investigate the requestor and",
        "  consider tightening the path allowlist before the next",
        "  engagement.",
        "- The auto-revert watcher's 60-min window has already",
        "  closed for these CRs (it activates at apply time). Manual",
        "  revert is still available via `/cp/changes`.",
        "",
        f"_Composed at `{datetime.now(timezone.utc).isoformat()}`._",
        "",
    ])

    body = "\n".join(lines)
    target_path = target_dir / f"{safe_slug}.md"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(body, encoding="utf-8")
        logger.info(
            "vacation_mode.digest: wrote %s (%d auto-applies)",
            target_path, n_total,
        )
    except OSError:
        logger.debug(
            "vacation_mode.digest: write failed", exc_info=True,
        )
    return target_path


def list_digests() -> list[Path]:
    """Return all digest files in chronological order (newest last)."""
    d = _digests_dir()
    if not d.exists():
        return []
    try:
        return sorted(d.glob("*.md"))
    except OSError:
        return []


def read_digest(path: Path) -> str:
    """Read a digest file. Returns empty string on any error."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""
