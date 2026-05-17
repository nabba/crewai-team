"""source_ledger_replay — quarterly KB rebuild-from-ledger drill (§56).

PROGRAM §56. 7th resilience drill. Proves that the source-ledger →
KB reconstruction pipeline actually works, on real data, without
touching the live KB.

What the drill does
===================

  1. Pick a random KB from the live set
  2. Replay its ledger into a scratch dir (chromadb.PersistentClient
     against ``<workspace>/.drill_scratch/<kb>/`` — never touches the
     live KB)
  3. Compare the scratch KB's row count vs the ledger row count
  4. PASS when they match within a small tolerance; FAIL otherwise
  5. Clean up the scratch dir

Risk: LOW. Read-only against the live KB; the scratch dir is a
fresh temporary chromadb instance. Never modifies anything operator
data depends on.

Cadence: quarterly (90 days). Same as the other LOW-risk drills.

What this catches
=================

  * The ledger has been silently corrupted (rows now unreplayable)
  * The embedding model has changed in a way that breaks replay
  * The KB schema has drifted from what the ledger expects
  * Some new chromadb version changed the API in a way the replay
    code doesn't handle

All of those would otherwise only surface during a real recovery
event — which is exactly when you don't want to discover them.
"""
from __future__ import annotations

import json
import logging
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.resilience_drills.audit import (
    acquire_drill_lock,
    append_result,
    emit_landmark_for,
    last_result_for,
    release_drill_lock,
)
from app.resilience_drills.protocol import (
    DrillResult,
    DrillRisk,
    DrillSpec,
    DrillStatus,
    drill_enabled,
    register,
)

logger = logging.getLogger(__name__)


SPEC = DrillSpec(
    name="source_ledger_replay",
    cadence_days=90,
    grace_days=30,
    risk=DrillRisk.LOW,
    description=(
        "Quarterly drill that picks one KB at random, replays its "
        "source ledger into a scratch chromadb, and verifies the row "
        "count matches. Proves that the §56 reconstruction pipeline "
        "actually works without touching live data."
    ),
    requires_master_switch="drill_source_ledger_replay_enabled",
)


# Tolerance for the row-count comparison. Replay can legitimately
# end with fewer rows than the ledger if some rows had empty text
# (skipped during replay). 5% is generous and catches the cases
# where replay genuinely diverged from the ledger.
_ACCEPTABLE_LOSS_PCT = 0.05

# Cap per-drill row replay so we don't run a 30-minute drill against
# a huge KB. The 10k cap is more than enough to confirm the
# mechanism works; the daily drift-detection daemon handles the
# full-KB rebuild.
_DRILL_REPLAY_CAP = 10_000


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT  # type: ignore
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _scratch_root() -> Path:
    return _workspace_root() / ".drill_scratch"


def _pick_kb() -> str | None:
    """Pick one KB at random that has a non-trivial ledger.

    Skip KBs with empty ledgers — there's nothing meaningful to
    drill. If every KB is empty the drill is SKIPPED, not failed.
    """
    try:
        from app.memory.source_ledger import list_kbs, count_rows
    except Exception:
        logger.debug("source_ledger_replay: import failed", exc_info=True)
        return None
    candidates = []
    for kb in list_kbs():
        try:
            if count_rows(kb) > 0:
                candidates.append(kb)
        except Exception:
            continue
    if not candidates:
        return None
    return random.choice(candidates)


def _run(*, dry_run: bool = True) -> DrillResult:
    started = datetime.now(timezone.utc)
    t0 = time.time()
    prior = last_result_for(SPEC.name)
    prior_status = prior.get("status") if prior else None

    if not drill_enabled(SPEC):
        result = DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.SKIPPED,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0,
            dry_run=dry_run,
            detail={"reason": "master switch off"},
        )
        append_result(result)
        return result

    lock = acquire_drill_lock(SPEC.name)
    if lock is None:
        result = DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.SKIPPED,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0,
            dry_run=dry_run,
            detail={"reason": "lock held"},
        )
        append_result(result)
        return result

    try:
        kb = _pick_kb()
        if kb is None:
            result = DrillResult(
                drill_name=SPEC.name,
                status=DrillStatus.SKIPPED,
                started_at=started.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_s=time.time() - t0,
                dry_run=dry_run,
                detail={"reason": "no KB with non-empty ledger"},
            )
            append_result(result)
            emit_landmark_for(result, prior_status=prior_status)
            return result

        detail: dict[str, Any] = {"kb_name": kb}

        # Step 1 — count ledger rows we'll attempt to replay.
        try:
            from app.memory.source_ledger import count_rows, replay_kb, verify_chain
        except Exception as exc:
            result = DrillResult(
                drill_name=SPEC.name,
                status=DrillStatus.ERROR,
                started_at=started.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_s=time.time() - t0,
                dry_run=dry_run,
                detail=detail,
                errors=[f"import failed: {exc}"],
            )
            append_result(result)
            emit_landmark_for(result, prior_status=prior_status)
            return result

        ledger_rows = min(count_rows(kb), _DRILL_REPLAY_CAP)
        detail["ledger_rows"] = ledger_rows

        # Step 2 — verify chain integrity FIRST. A broken chain is a
        # hard failure independent of the replay outcome.
        chain = verify_chain(kb)
        detail["chain_verify"] = chain.to_dict()
        if not chain.ok:
            result = DrillResult(
                drill_name=SPEC.name,
                status=DrillStatus.FAIL,
                started_at=started.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_s=time.time() - t0,
                dry_run=dry_run,
                detail=detail,
                errors=[f"chain broken at row {chain.first_bad_row}: {chain.first_bad_reason}"],
            )
            append_result(result)
            emit_landmark_for(result, prior_status=prior_status)
            return result

        # Step 3 — replay into a scratch dir.
        scratch = _scratch_root() / kb / started.strftime("%Y%m%dT%H%M%SZ")
        # Cleanup any half-written scratch from a previous failure.
        if scratch.exists():
            try:
                shutil.rmtree(scratch)
            except OSError:
                pass
        scratch.mkdir(parents=True, exist_ok=True)
        replay = replay_kb(kb, target_path=scratch, max_rows=_DRILL_REPLAY_CAP)
        detail["replay"] = replay.to_dict()

        # Step 4 — verify the scratch KB has the expected rows.
        scratch_rows = 0
        try:
            import chromadb  # type: ignore
            client = chromadb.PersistentClient(path=str(scratch))
            for col in client.list_collections():
                try:
                    scratch_rows += int(col.count())
                except Exception:
                    pass
        except Exception:
            logger.debug("source_ledger_replay: scratch introspect failed", exc_info=True)
        detail["scratch_rows"] = scratch_rows
        detail["scratch_path"] = str(scratch)

        # Step 5 — verdict.
        if ledger_rows == 0:
            status = DrillStatus.SKIPPED
        else:
            loss_pct = max(0.0, (ledger_rows - scratch_rows) / ledger_rows)
            detail["loss_pct"] = round(loss_pct, 4)
            if loss_pct <= _ACCEPTABLE_LOSS_PCT:
                status = DrillStatus.PASS
            else:
                status = DrillStatus.FAIL

        # Step 6 — clean up scratch (always, regardless of verdict).
        try:
            shutil.rmtree(scratch)
        except OSError:
            pass
        try:
            # Try to remove the per-KB parent if empty.
            scratch.parent.rmdir()
        except OSError:
            pass

        result = DrillResult(
            drill_name=SPEC.name,
            status=status,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0,
            dry_run=dry_run,
            detail=detail,
        )
        append_result(result)
        emit_landmark_for(result, prior_status=prior_status)
        return result

    except Exception as exc:
        logger.debug("source_ledger_replay: drill errored", exc_info=True)
        result = DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.ERROR,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0,
            dry_run=dry_run,
            detail={},
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        append_result(result)
        emit_landmark_for(result, prior_status=prior_status)
        return result
    finally:
        release_drill_lock(SPEC.name, lock)


def run(*, dry_run: bool = True) -> DrillResult:
    """Public entry point — drills run via the standard scheduler."""
    return _run(dry_run=dry_run)


# Register at import time, same pattern as the other drills.
register(SPEC, run)
