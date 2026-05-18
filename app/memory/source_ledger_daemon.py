"""
source_ledger_daemon.py — Background daemon that keeps the per-KB
source ledger in sync with live chromadb contents.

PROGRAM §56 (2026-05-17) — companion to ``app/memory/source_ledger.py``.

Three responsibilities:

  1. **Bootstrap**  — One-time per KB. Walks every collection's rows
     and appends to the ledger anything missing. Idempotent on
     ``(collection, doc_id)``.

  2. **Drift check** — Daily. Compares live KB row count vs ledger row
     count. If KB is short by >5%, triggers a replay (the chromadb
     file lost rows we still have in the ledger). If ledger is short,
     queues another bootstrap pass.

  3. **Verify chain** — Daily. Runs ``verify_chain()`` on each KB's
     ledger; alerts if hash chain is broken (bit-rot, tampering, or a
     write that lost a fsync).

Boot-anchored from ``app.healing.__init__`` like the other
observational daemons.

Master switches (all default ON):

  chromadb_source_ledger_enabled          — kill switch for everything
  chromadb_ledger_bootstrap_enabled       — bootstrap branch
  chromadb_ledger_drift_replay_enabled    — drift-detection branch
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


_BOOTSTRAP_INITIAL_DELAY_S = 300       # 5 min after gateway boot — quiet warm-up
_BOOTSTRAP_PASS_INTERVAL_S = 24 * 3600  # daily — catches any rows missed by direct hooks
_DRIFT_CHECK_INTERVAL_S = 24 * 3600     # daily
_VERIFY_CHAIN_INTERVAL_S = 24 * 3600    # daily
_COMPACTION_INTERVAL_S = 7 * 24 * 3600  # weekly — see source_ledger.compact_ledger gates

_DAEMON_STARTED = False
_DAEMON_LOCK = threading.Lock()
_STOP_EVENT = threading.Event()


def _enabled() -> bool:
    return os.getenv("HEALING_MONITORS_ENABLED", "true").lower() in ("true", "1", "yes")


def _gate_master() -> bool:
    """Master gate read from runtime_settings — flippable without restart."""
    try:
        from app.runtime_settings import get_chromadb_source_ledger_enabled
        return get_chromadb_source_ledger_enabled()
    except Exception:
        return True


def _gate(name: str) -> bool:
    try:
        from app import runtime_settings  # type: ignore
        getter = getattr(runtime_settings, f"get_{name}", None)
        if getter is not None:
            return bool(getter())
    except Exception:
        pass
    return True


def _run_one_pass() -> dict:
    """Single daemon iteration. Returns summary dict for logging.

    Each branch is independently gated. Failure in one branch never
    breaks the others — every step in try/except.
    """
    summary: dict[str, Any] = {
        "bootstraps": {},
        "drifts": {},
        "chain_verifications": {},
    }
    if not _gate_master():
        summary["skipped"] = "master_disabled"
        return summary

    try:
        from app.memory import source_ledger as sl
    except Exception:
        logger.exception("source_ledger_daemon: import failed")
        summary["error"] = "import_failed"
        return summary

    kbs = sl.list_kbs()
    summary["kbs"] = kbs

    # ── Bootstrap branch ────────────────────────────────────────
    if _gate("chromadb_ledger_bootstrap_enabled"):
        for kb_name in kbs:
            try:
                # Cap rows per pass so the bootstrap doesn't lock the
                # daemon for a long time on a big KB. The daemon
                # re-runs daily — over a few iterations it converges.
                info = sl.bootstrap_kb(kb_name, max_rows=5000)
                summary["bootstraps"][kb_name] = {
                    "ok": info.get("ok"),
                    "rows_added": info.get("rows_added", 0),
                    "rows_already_present": info.get("rows_already_present", 0),
                    "error": info.get("error"),
                }
            except Exception:
                logger.exception(
                    "source_ledger_daemon: bootstrap raised for %s", kb_name,
                )

    # ── Drift branch ────────────────────────────────────────────
    if _gate("chromadb_ledger_drift_replay_enabled"):
        for kb_name in kbs:
            try:
                drift = sl.check_drift(kb_name)
                summary["drifts"][kb_name] = drift.to_dict()
                if drift.needs_replay:
                    logger.warning(
                        "source_ledger_daemon: drift detected on %s — "
                        "ledger=%d kb=%d pct=%.2f%%; replaying",
                        kb_name, drift.ledger_rows, drift.kb_rows_total,
                        100 * drift.drift_pct,
                    )
                    replay = sl.replay_kb(kb_name)
                    summary["drifts"][kb_name]["replay_result"] = replay.to_dict()
                    _alert_drift_replay(kb_name, drift, replay)
            except Exception:
                logger.exception(
                    "source_ledger_daemon: drift check raised for %s", kb_name,
                )

    # ── Chain verification branch ───────────────────────────────
    # Uses the incremental verifier (B4, 2026-05-18): keeps a per-KB
    # checkpoint of ``(rows_verified, hash_at_that_row, first_row_hash)``
    # and resumes from the appended tail rather than walking from
    # genesis daily. Compaction invalidates the checkpoint via the
    # first_row_hash sentinel, falling back to a full walk on the next
    # pass. Drills + dashboard + audit_chain_check monitor still call
    # the full ``verify_chain`` for authoritative end-to-end checks.
    for kb_name in kbs:
        try:
            verify = sl.verify_chain_incremental(kb_name)
            summary["chain_verifications"][kb_name] = verify.to_dict()
            if not verify.ok:
                logger.error(
                    "source_ledger_daemon: chain broken on %s at row %d: %s",
                    kb_name, verify.first_bad_row, verify.first_bad_reason,
                )
                _alert_chain_break(kb_name, verify)
        except Exception:
            logger.exception(
                "source_ledger_daemon: chain verify raised for %s", kb_name,
            )

    # ── Compaction branch ──────────────────────────────────────
    # Weekly fold-and-snapshot pass. Gated internally: compaction
    # skips when ledger is small or reduction would be minor. The
    # daemon loop runs daily so the "weekly cadence" is enforced
    # by a per-KB state row that records the last successful pass.
    summary["compactions"] = {}
    if _gate("chromadb_ledger_compaction_enabled"):
        try:
            from pathlib import Path as _P
            ws_root = None
            try:
                from app.paths import WORKSPACE_ROOT
                ws_root = _P(WORKSPACE_ROOT)
            except Exception:
                ws_root = _P("/app/workspace")
            state_path = ws_root / "healing" / "source_ledger_compaction_state.json"
            try:
                import json as _json
                last_runs = _json.loads(state_path.read_text()) if state_path.exists() else {}
            except Exception:
                last_runs = {}
            now_ts = __import__("time").time()
            updated = False
            for kb_name in kbs:
                last = float(last_runs.get(kb_name, 0))
                if now_ts - last < _COMPACTION_INTERVAL_S:
                    continue
                try:
                    cresult = sl.compact_ledger(kb_name)
                    summary["compactions"][kb_name] = cresult.to_dict()
                    last_runs[kb_name] = now_ts
                    updated = True
                except Exception:
                    logger.exception(
                        "source_ledger_daemon: compaction raised for %s",
                        kb_name,
                    )
            if updated:
                try:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
                    tmp.write_text(__import__("json").dumps(last_runs, indent=2))
                    tmp.replace(state_path)
                except Exception:
                    logger.debug(
                        "source_ledger_daemon: compaction state save failed",
                        exc_info=True,
                    )
        except Exception:
            logger.exception("source_ledger_daemon: compaction branch failed")

    # ── Off-host upload branches ────────────────────────────────
    # Both are gated by their own runtime_settings switches; with
    # OFF defaults this whole section is silent no-ops until the
    # operator opts in by wiring S3 / Google Drive credentials.
    summary["offhost"] = {"s3": {}, "gdrive": {}}
    try:
        from app.memory.source_ledger_offhost import s3 as offhost_s3
        for kb_name in kbs:
            try:
                r = offhost_s3.upload_kb_ledger(kb_name)
                summary["offhost"]["s3"][kb_name] = r.to_dict()
            except Exception:
                logger.debug(
                    "source_ledger_daemon: s3 upload raised for %s",
                    kb_name, exc_info=True,
                )
    except Exception:
        logger.debug("source_ledger_daemon: s3 backend import failed", exc_info=True)

    try:
        from app.memory.source_ledger_offhost import gdrive as offhost_gdrive
        for kb_name in kbs:
            try:
                r = offhost_gdrive.upload_kb_ledger(kb_name)
                summary["offhost"]["gdrive"][kb_name] = r.to_dict()
            except Exception:
                logger.debug(
                    "source_ledger_daemon: gdrive upload raised for %s",
                    kb_name, exc_info=True,
                )
    except Exception:
        logger.debug("source_ledger_daemon: gdrive backend import failed", exc_info=True)

    return summary


def _alert_drift_replay(kb_name: str, drift, replay) -> None:
    """Best-effort Signal alert on drift-triggered replay."""
    try:
        from app.life_companion._common import send_signal_alert  # type: ignore
        send_signal_alert(
            f"🔁 ChromaDB source-ledger drift on `{kb_name}`: "
            f"ledger has {drift.ledger_rows} rows, KB has {drift.kb_rows_total}. "
            f"Auto-replayed; upserted {replay.rows_upserted} rows in {replay.duration_s:.1f}s. "
            f"See /cp/settings → ChromaDB integrity.",
            tag=f"source_ledger_drift:{kb_name}",
        )
    except Exception:
        logger.debug("source_ledger_daemon: drift alert failed", exc_info=True)


def _alert_chain_break(kb_name: str, verify) -> None:
    """Critical alert on a hash-chain integrity failure — the ledger
    itself has been damaged. This is a different class of problem from
    drift: the source-of-truth is wrong.
    """
    try:
        from app.life_companion._common import send_signal_alert  # type: ignore
        send_signal_alert(
            f"🚨 ChromaDB source-ledger hash chain broken on `{kb_name}` at row "
            f"{verify.first_bad_row}: {verify.first_bad_reason}. "
            f"The on-disk ledger has been damaged or tampered with. "
            f"Restore from warm-spare / S3 / Google Drive — see "
            f"docs/CHROMADB_INTEGRITY.md §recovery.",
            tag=f"source_ledger_chain_break:{kb_name}",
        )
    except Exception:
        logger.debug("source_ledger_daemon: chain alert failed", exc_info=True)


def _daemon_loop() -> None:
    """Daemon thread body. Single quiet warm-up, then daily passes."""
    if _STOP_EVENT.wait(_BOOTSTRAP_INITIAL_DELAY_S):
        return
    while not _STOP_EVENT.is_set():
        try:
            summary = _run_one_pass()
            n_added = sum(
                b.get("rows_added", 0) for b in summary.get("bootstraps", {}).values()
            )
            replayed = [
                k for k, d in summary.get("drifts", {}).items() if d.get("replay_result")
            ]
            chain_bad = [
                k for k, v in summary.get("chain_verifications", {}).items()
                if not v.get("ok", True)
            ]
            if n_added or replayed or chain_bad:
                logger.info(
                    "source_ledger_daemon: pass complete — bootstrapped %d rows, "
                    "replayed %s, chain-bad %s",
                    n_added, replayed, chain_bad,
                )
        except Exception:
            logger.exception("source_ledger_daemon: pass raised")
        # Sleep until next pass. Use the smallest interval so all three
        # branches get equal cadence; in practice they're all daily.
        next_interval = min(
            _BOOTSTRAP_PASS_INTERVAL_S,
            _DRIFT_CHECK_INTERVAL_S,
            _VERIFY_CHAIN_INTERVAL_S,
        )
        if _STOP_EVENT.wait(next_interval):
            return


def start() -> None:
    """Idempotent daemon start. Called at module import (via
    app.healing.__init__ anchoring) so the daemon starts exactly once
    per process.
    """
    global _DAEMON_STARTED
    with _DAEMON_LOCK:
        if _DAEMON_STARTED:
            return
        if not _enabled():
            logger.debug("source_ledger_daemon: HEALING_MONITORS_ENABLED off, skipping")
            return
        t = threading.Thread(
            target=_daemon_loop, name="source_ledger_daemon", daemon=True,
        )
        t.start()
        _DAEMON_STARTED = True
        logger.info(
            "source_ledger_daemon: started (warm-up=%ds, pass=24h)",
            _BOOTSTRAP_INITIAL_DELAY_S,
        )


def stop() -> None:
    _STOP_EVENT.set()


# Eager start at import — anchored by app.healing.__init__ which is
# imported from app.main at boot. This matches the pattern used by
# proposal_bridge, capability_gap_analyzer, library_radar, etc.
start()
