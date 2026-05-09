"""Failure-pattern learner (Phase C #4, 2026-05-09).

The Self-Heal v3 dispatcher only fires for hand-registered runbooks
(see ``app/healing/handlers/__init__.py``). Errors that don't match
any registered pattern fall through to ``log_only`` and accumulate
silently in ``workspace/logs/errors.jsonl``. Over weeks that's where
the next class of solvable problems lives — buried in the long tail
the dispatcher can't see.

This module mines the error stream weekly and flags recurring
signatures that are NOT yet covered by any registered runbook.
For each flag, it writes a markdown scaffold to
``workspace/proposed_runbooks/<sig>.md`` for the operator to flesh
out into a real handler.

Cadence: 24 h (the daemon driver pings daily; we self-cadence to once/day).

Algorithm:

  1. Walk ``workspace/logs/errors.jsonl`` (last ``LOOKBACK_DAYS``).
  2. Compute the same SHA-1 signature the runbook dispatcher uses
     (via ``handlers._common.compute_signature``).
  3. Group by signature; count occurrences in the lookback window.
  4. Filter:
       * occurrence count ≥ ``MIN_OCCURRENCES`` (default 10).
       * signature NOT in any registered runbook.
       * not in the proposed-runbooks index already (dedup).
  5. For each surviving signature, write a markdown scaffold:
       title, signature, sample messages, suggested action template.
  6. Emit one Signal alert summarizing the top-5 new patterns + paths
     to the scaffolds. Per-pattern dedup: 14 days.

Master switch: ``HEALING_PATTERN_LEARNER_ENABLED`` (default ON).
Disabling halts both the JSONL scan and the Signal alert.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    compute_signature,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "pattern_learner.json"
_ERRORS_LOG = Path("/app/workspace/logs/errors.jsonl")
# Phase E #9 (2026-05-09): unified with auditor_bridge's mirror dir
# (``docs/proposed_fixes/``) so operators have one location to scan
# for proposed self-heal scaffolds. The two writers use distinct
# filename conventions (``learner_<sig>.md`` vs
# ``<pattern>__attempt_<n>.md``) so they don't collide.
_PROPOSED_DIR = Path("/app/docs/proposed_fixes")
_RUN_CADENCE_S = 24 * 3600
_LOOKBACK_DAYS = 7
_MIN_OCCURRENCES = 10
_TOP_N_ALERT = 5
_DEDUP_WINDOW_S = 14 * 86400


def _enabled() -> bool:
    return os.getenv("HEALING_PATTERN_LEARNER_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── Source iteration ──────────────────────────────────────────────────────


def _iter_error_records(lookback_days: int) -> list[dict]:
    """Read errors.jsonl + rotated suffixes; return rows newer than lookback."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[dict] = []

    candidates: list[Path] = []
    if _ERRORS_LOG.exists():
        candidates.append(_ERRORS_LOG)
    # Include rotated suffixes (errors.jsonl.1, .2, .3) — they're still
    # in the lookback window for the first day or two after rotation.
    if _ERRORS_LOG.parent.exists():
        candidates.extend(sorted(_ERRORS_LOG.parent.glob("errors.jsonl.*")))

    for p in candidates:
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts_str = row.get("ts", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(
                            str(ts_str).replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        row["_ts"] = ts
                        out.append(row)
        except OSError:
            logger.debug("pattern_learner: read failed for %s", p, exc_info=True)
            continue
    return out


# ── Signature grouping ────────────────────────────────────────────────────


def _group_by_signature(rows: list[dict]) -> dict[str, dict]:
    """Cluster rows by ``compute_signature(logger, message)``.

    Returns ``{signature: {count, sample_messages, sample_logger,
    first_seen, last_seen}}``.
    """
    groups: dict[str, dict] = {}
    for row in rows:
        # Only consider WARN/ERROR. INFO is conversational; DEBUG is noise.
        level = (row.get("level") or "").upper()
        if level not in ("WARNING", "ERROR", "CRITICAL"):
            continue
        msg = row.get("message") or ""
        logger_name = row.get("logger") or ""
        if not msg:
            continue
        sig = compute_signature(logger_name, msg)
        bucket = groups.setdefault(sig, {
            "count": 0, "sample_messages": [],
            "logger_names": Counter(), "modules": Counter(),
            "first_seen": row["_ts"], "last_seen": row["_ts"],
        })
        bucket["count"] += 1
        bucket["logger_names"][logger_name] += 1
        bucket["modules"][row.get("module") or ""] += 1
        if len(bucket["sample_messages"]) < 3:
            bucket["sample_messages"].append(msg[:200])
        if row["_ts"] > bucket["last_seen"]:
            bucket["last_seen"] = row["_ts"]
        if row["_ts"] < bucket["first_seen"]:
            bucket["first_seen"] = row["_ts"]
    return groups


def _registered_signatures() -> set[str]:
    """Return signatures already covered by registered runbooks.

    The runbooks module is TIER_IMMUTABLE; we read its private state
    directly (``_REGISTERED_RUNBOOKS`` keyed by runbook name; locking
    via ``_registry_lock``). Wrapped in try/except so a future symbol
    rename never silently breaks our covered-signature check — the
    fallback returns an empty set, which makes us re-propose patterns
    that ARE covered, which then surfaces the rename instead of
    hiding it.
    """
    try:
        from app.healing.runbooks import _REGISTERED_RUNBOOKS, _registry_lock
    except Exception:
        logger.debug(
            "pattern_learner: runbooks registry symbols unavailable",
            exc_info=True,
        )
        return set()
    sigs: set[str] = set()
    try:
        with _registry_lock:
            for name, entry in _REGISTERED_RUNBOOKS.items():
                # Two registration shapes co-exist in the dispatcher:
                #   (a) by SHA-1 signature — name IS the hex hash.
                #   (b) by regex pattern — name is human-readable.
                # We treat (a) as a covered signature directly, and skip
                # catch-all regex patterns (can't claim every signature).
                if (
                    isinstance(name, str)
                    and len(name) >= 8
                    and all(c in "0123456789abcdef" for c in name)
                ):
                    sigs.add(name)
                    continue
                pat = getattr(entry, "pattern", None)
                pat_str = ""
                if pat is not None:
                    pat_str = getattr(pat, "pattern", str(pat))
                if (
                    pat_str
                    and pat_str not in (".*", ".+")
                    and len(pat_str) >= 8
                    and all(c in "0123456789abcdef" for c in pat_str)
                ):
                    sigs.add(pat_str)
    except Exception:
        logger.debug("pattern_learner: registry walk failed", exc_info=True)
    return sigs


# ── Scaffold writer ───────────────────────────────────────────────────────


def _write_scaffold(signature: str, group: dict) -> Path:
    """Materialize a markdown scaffold for one new signature."""
    _PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    short = signature[:12]
    # ``learner_`` prefix distinguishes from auditor_bridge's
    # ``<pattern>__attempt_<n>.md`` files in the same dir.
    target = _PROPOSED_DIR / f"learner_{short}.md"

    top_logger = group["logger_names"].most_common(1)
    top_logger_str = top_logger[0][0] if top_logger else "(unknown)"
    top_module = group["modules"].most_common(1)
    top_module_str = top_module[0][0] if top_module else "(unknown)"

    samples = "\n".join(f"- {m}" for m in group["sample_messages"])

    body = f"""# Proposed runbook: {short}

Generated by `app.healing.pattern_learner` on {datetime.now(timezone.utc).isoformat()}.

## Summary

A failure pattern with signature `{signature}` was observed
**{group['count']}** time(s) over the last {_LOOKBACK_DAYS} days. It is
NOT yet covered by any registered runbook handler.

- First seen: `{group['first_seen'].isoformat()}`
- Last seen:  `{group['last_seen'].isoformat()}`
- Top logger: `{top_logger_str}`
- Top module: `{top_module_str}`

## Sample messages

{samples}

## Suggested next steps

1. Decide whether this pattern represents:
   - a real fault that warrants a remediation handler
     → write a handler in `app/healing/handlers/` and register
       under signature `{signature}`
   - a benign warning that shouldn't appear in errors.jsonl at all
     → adjust the upstream logger to use INFO instead of WARNING
   - a transient issue handled correctly by retry/backoff
     → mark this scaffold "noise" and ignore (or move it under
       `workspace/proposed_runbooks/_noise/` for archival)

2. If a handler is appropriate, scaffold:

```python
# app/healing/handlers/<topic>.py
def my_runbook(*, sample: str, **_) -> dict:
    if "<distinguishing fragment>" not in sample:
        return {{"applied": False, "reason": "sample mismatch"}}
    # remediation logic here
    return {{"applied": True, "reason": "..."}}

# in app/healing/handlers/__init__.py:
register_runbook(
    name="<topic>",
    pattern_signature="{signature}",
    fn=my_runbook,
    rate_limit_per_hour=1,
)
```

3. After registering, this scaffold becomes obsolete. The pattern
   learner will skip it on the next pass.

## Auto-suppression

Re-running the learner won't overwrite this file unless it's deleted.
Operator owns the lifecycle.
"""
    target.write_text(body, encoding="utf-8")
    return target


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "candidates": 0, "new_proposals": 0,
        "alerted": False, "lookback_days": _LOOKBACK_DAYS,
    }
    if not _enabled():
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "alerted_signatures": {},
    })
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    rows = _iter_error_records(_LOOKBACK_DAYS)
    if not rows:
        write_state_json(_STATE_FILE, state)
        return summary

    groups = _group_by_signature(rows)
    covered = _registered_signatures()
    alerted: dict = state.setdefault("alerted_signatures", {})

    candidates: list[tuple[str, dict, Path]] = []
    for sig, group in groups.items():
        if group["count"] < _MIN_OCCURRENCES:
            continue
        if sig in covered:
            continue
        prev = alerted.setdefault(sig, {"last_alert_at": 0.0})
        if now_ts - float(prev.get("last_alert_at", 0)) < _DEDUP_WINDOW_S:
            continue
        try:
            target = _write_scaffold(sig, group)
        except Exception:
            logger.debug("pattern_learner: scaffold write failed for %s", sig,
                         exc_info=True)
            continue
        candidates.append((sig, group, target))
        prev["last_alert_at"] = now_ts
        summary["new_proposals"] += 1

    summary["candidates"] = len(candidates)
    write_state_json(_STATE_FILE, state)

    if candidates:
        # Top-N by occurrence count.
        candidates.sort(key=lambda t: t[1]["count"], reverse=True)
        lines = [
            f"🧪 Self-heal: {len(candidates)} new failure pattern(s) "
            f"observed but NOT covered by any runbook (last "
            f"{_LOOKBACK_DAYS}d, ≥{_MIN_OCCURRENCES} occurrences):\n",
        ]
        for sig, group, target in candidates[:_TOP_N_ALERT]:
            sample = group["sample_messages"][0] if group["sample_messages"] else ""
            lines.append(
                f"  • `{sig[:8]}…` ×{group['count']} — {sample[:80]}"
            )
            try:
                rel = target.relative_to(Path("/app"))
                lines.append(f"    scaffold: `{rel}`")
            except ValueError:
                lines.append(f"    scaffold: `{target}`")
        if len(candidates) > _TOP_N_ALERT:
            lines.append(f"\n  …and {len(candidates) - _TOP_N_ALERT} more.")
        try:
            send_signal_alert("\n".join(lines), tag="pattern_learner")
            summary["alerted"] = True
        except Exception:
            logger.debug("pattern_learner: alert send failed", exc_info=True)

    audit_event(
        "pattern_learner_pass",
        candidates=summary["candidates"],
        new_proposals=summary["new_proposals"],
        alerted=summary["alerted"],
    )
    return summary
