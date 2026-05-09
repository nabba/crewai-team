"""Auditor → Signal + change-request bridge.

The error-resolution cron job in ``app/auditor.py`` produces fix
proposals (``error_fix_proposed`` events in
``workspace/audit_journal.json``) but those proposals sit in a
``proposals`` table that nobody opens — the journal records show
"0 resolved, 1 attempted" every 30 min for the past month.

This bridge runs as a daemon poll on the audit_journal: any new
``error_fix_proposed`` event triggers two surfacing actions:

  1. **Signal alert** — single message with the proposal text + a
     deep-link to ``/cp/proposals``. Lets the operator triage from
     their phone in seconds.
  2. **Change-request mirror** — files a CR under
     ``docs/proposed_fixes/<pattern>__attempt_<n>.md`` whose
     content is a structured markdown record of the proposal. The
     CR is operator-approvable via Signal 👍 or ``/cp/changes``;
     once approved, the markdown lands in the repo as a permanent
     paper trail of "we tried this fix for that pattern."

The bridge does NOT auto-apply code changes — the auditor outputs
prose, not diffs. The CR mirror creates a documentation artefact
that goes through the same approval gate as code-CRs, giving
operators a unified approval surface.

Dedup keys are (pattern_key, attempt_n) so re-attempts under the
same pattern get separate alerts + CRs (each attempt is a fresh
fix to evaluate), but the same (pattern, attempt) doesn't repeat.

Master switch: ``HEALING_AUDITOR_BRIDGE_ENABLED`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    file_change_request,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "auditor_bridge.json"
_POLL_INTERVAL_S = 5 * 60  # check every 5 min — proposals are O(30 min)
_WARMUP_S = 90

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_JOURNAL_CANDIDATES = [
    _REPO_ROOT / "workspace" / "audit_journal.json",
    Path("/app/workspace/audit_journal.json"),
]


def _enabled() -> bool:
    return os.getenv("HEALING_AUDITOR_BRIDGE_ENABLED", "true").lower() in (
        "true", "1", "yes",
    )


def _audit_path() -> Path | None:
    for p in _AUDIT_JOURNAL_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_journal() -> list[dict[str, Any]]:
    p = _audit_path()
    if p is None:
        return []
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def _proposal_key(detail: str) -> str:
    """Best-effort dedup key. The auditor format is::

        "Pattern <pattern_key> attempt #N: <free text>"

    We extract ``<pattern_key>`` + ``#N`` so re-attempts under the same
    pattern get separate alerts (each attempt is a fresh fix to evaluate),
    but the SAME (pattern, attempt) doesn't repeat.
    """
    if not detail:
        return ""
    head = detail[:200]
    # Try to split on " attempt #" to isolate pattern_key + attempt number.
    if " attempt #" in head:
        before, after = head.split(" attempt #", 1)
        attempt = after.split(":", 1)[0].strip()
        pattern_key = before.replace("Pattern ", "").strip()
        return f"{pattern_key}#{attempt}"
    return head[:80]


# ── CR mirror ────────────────────────────────────────────────────────────


_PATH_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _proposal_to_path(key: str) -> str:
    """Convert a proposal key (e.g. ``"coding:RuntimeError#1"``) into a
    filesystem-safe markdown path under ``docs/proposed_fixes/``.
    """
    safe = _PATH_SAFE_RE.sub("_", key).strip("_")
    if not safe:
        safe = "unknown"
    return f"docs/proposed_fixes/{safe}.md"


def _proposal_to_markdown(*, key: str, entry: dict[str, Any]) -> str:
    """Render a structured markdown record of an auditor proposal.

    The markdown is operator-readable, repo-friendly, and (once the CR
    is approved) lives as a permanent paper trail.
    """
    detail = (entry.get("detail") or "").strip()
    ts = entry.get("ts") or "unknown"
    pattern_part, _, attempt_part = key.partition("#")
    files_changed = entry.get("files_changed") or []
    return f"""# Auditor proposal — `{pattern_part}` (attempt {attempt_part or '?'})

> Auto-mirrored from `workspace/audit_journal.json` by the
> ``app.healing.auditor_bridge`` daemon. The auditor's
> ``run_error_resolution`` cron produced this fix proposal but the
> proposals system surface (`/cp/proposals`) was unattended; this
> file gives the change-request gate (`/cp/changes`) a concrete
> artefact to approve.

- **Pattern:** `{pattern_part}`
- **Attempt:** {attempt_part or '?'}
- **Proposed at:** {ts}
- **Files referenced:** {", ".join(files_changed) if files_changed else "—"}

## Proposed fix (auditor's own description)

```
{detail}
```

## Operator action

The auditor's description is prose, not a runnable diff. Approving
this CR lands this markdown as a record. Apply the actual code change
yourself based on the description above, then the next pass of
``auditor.run_error_resolution`` will mark the pattern resolved if no
new errors of the same shape appear within 24 hours.

If the proposal turns out to be wrong, **reject** the CR. The
auditor's progressive-refinement loop will try a different angle on
attempt #{(int(attempt_part) + 1) if attempt_part.isdigit() else 'N+1'}.
"""


def _file_proposal_cr(*, key: str, entry: dict[str, Any]) -> str | None:
    """File a CR mirroring this proposal. Returns the CR id or None.

    Failure (CR system unavailable, validator rejection) is non-fatal:
    the Signal alert is the primary surfacing surface; the CR mirror
    is the durable trail.
    """
    path = _proposal_to_path(key)
    body = _proposal_to_markdown(key=key, entry=entry)
    pattern_part, _, attempt_part = key.partition("#")
    return file_change_request(
        path=path,
        new_content=body,
        old_content="",
        reason=(
            f"Self-heal mirror: auditor proposal for pattern "
            f"`{pattern_part}` (attempt {attempt_part or '?'}). "
            f"Approve to record the proposal as a paper trail under "
            f"`{path}`. The actual code fix must still be applied "
            f"manually based on the proposal description."
        ),
        requestor="self_heal_handler",
    )


def _emit_alert(entry: dict[str, Any], state: dict[str, Any]) -> bool:
    detail = entry.get("detail") or ""
    key = _proposal_key(detail)
    if not key:
        return False
    seen = state.setdefault("seen", {})
    if key in seen:
        return False

    # File the CR mirror first so its id can land in the Signal alert.
    cr_id: str | None = None
    try:
        cr_id = _file_proposal_cr(key=key, entry=entry)
    except Exception:
        logger.debug("auditor_bridge: CR mirror failed", exc_info=True)
        cr_id = None

    seen[key] = {
        "ts": entry.get("ts"),
        "alerted_at": time.time(),
        "cr_id": cr_id,
    }
    audit_event(
        "auditor_bridge_alert",
        proposal_key=key,
        proposal_ts=entry.get("ts"),
        cr_id=cr_id,
    )

    # Truncate long fix descriptions for Signal readability.
    cr_line = (
        f"\n\n📌 Mirrored as CR `{cr_id}` — approve in /cp/changes "
        f"to land the record."
        if cr_id
        else "\n\n(CR mirror unavailable; only Signal alert sent.)"
    )
    body = (
        f"🩺 Self-heal: the auditor proposed a fix that's been sitting "
        f"un-applied:\n\n"
        f"`{detail[:400]}`\n\n"
        f"Original proposal at `/cp/proposals`."
        f"{cr_line}"
    )
    send_signal_alert(body, tag="auditor_bridge")
    return True


def run_one_pass() -> int:
    """Single pass over the audit journal. Returns count of alerts emitted."""
    if not _enabled():
        return 0

    entries = _load_journal()
    if not entries:
        return 0

    # Only consider events from the last 7 days — older proposals are
    # almost certainly stale and re-alerting them is noise.
    cutoff_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S",
        time.gmtime(time.time() - 7 * 86400),
    )
    relevant = [
        e for e in entries
        if e.get("event") == "error_fix_proposed"
        and (e.get("ts") or "") >= cutoff_iso
    ]

    if not relevant:
        return 0

    state = read_state_json(_STATE_FILE, {"seen": {}})
    # Garbage-collect: drop seen-entries older than 14 days so the file
    # doesn't grow forever.
    cutoff = time.time() - 14 * 86400
    seen = state.setdefault("seen", {})
    for k in list(seen.keys()):
        ts = seen[k].get("alerted_at", 0)
        if ts < cutoff:
            seen.pop(k, None)

    emitted = 0
    for entry in relevant:
        if _emit_alert(entry, state):
            emitted += 1

    state["last_run_at"] = time.time()
    write_state_json(_STATE_FILE, state)
    return emitted


# ── Daemon driver ──────────────────────────────────────────────────────────

_started = False
_start_lock = threading.Lock()
_stop_event = threading.Event()


def _driver() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            n = run_one_pass()
            if n:
                logger.info("auditor_bridge: %d alert(s) emitted", n)
        except Exception:
            logger.debug("auditor_bridge: pass failed", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start() -> None:
    """Start the bridge daemon. Idempotent."""
    global _started
    if not _enabled():
        logger.info("auditor_bridge: disabled via HEALING_AUDITOR_BRIDGE_ENABLED")
        return
    with _start_lock:
        if _started:
            return
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name="healing-auditor-bridge", daemon=True,
        )
        thread.start()
        _started = True
        logger.info("auditor_bridge: daemon started (poll=%ds)", _POLL_INTERVAL_S)


def stop() -> None:
    _stop_event.set()


# Eager start on import.
start()
