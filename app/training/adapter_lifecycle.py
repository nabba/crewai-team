"""Adapter-lifecycle management — orphan cleanup, disk monitoring, drift.

The MLX QLoRA training pipeline writes adapters to
``workspace/training_adapters/`` and fused models to
``workspace/trained_models/``. The registry at
``workspace/training_adapters/registry.json`` is the source of truth
for which adapter paths are *active*. Anything on disk under those
roots that ISN'T in the registry is an orphan — left over from an
aborted training run, a renamed adapter, or a manual cleanup that
forgot to remove the file.

This module runs monthly via the healing monitors daemon and:

  1. **Orphan cleanup** — files under ``ADAPTERS_DIR`` / ``MODELS_DIR``
     not referenced by ``registry.json`` AND older than
     ``_ORPHAN_AGE_S`` (default 30 days) are deleted. Two-condition
     guard so we don't race with an in-flight training run.

  2. **Dead-pointer detection** — registry entries whose
     ``adapter_path`` doesn't exist on disk are surfaced via Signal.
     The pointer is left in place for the operator to fix manually
     (deleting it would lose the only record of "we used to have an
     adapter named X").

  3. **Disk-usage report** — total bytes under each tracked root.
     Alert at ``_BLOAT_THRESHOLD_GB``.

  4. **Drift surface** — every pass writes the current registry
     snapshot to ``workspace/healing/adapter_lifecycle_history.jsonl``
     so a future audit can reconstruct "when was this adapter
     promoted, with what eval_score". Closes the rollback-blind-spot
     where the training pipeline overwrites adapter slots without
     keeping a trail.

This module imports from ``app.training_pipeline`` (TIER_IMMUTABLE)
read-only — no modifications. ``training_pipeline.py`` itself is
unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

# Mirror the training_pipeline constants — read directly so we don't
# depend on the orchestrator class being instantiated.
_ADAPTERS_DIR = Path("/app/workspace/training_adapters")
_MODELS_DIR = Path("/app/workspace/trained_models")
_REGISTRY_PATH = _ADAPTERS_DIR / "registry.json"

_STATE_FILE = "adapter_lifecycle.json"
_HISTORY_FILE = "adapter_lifecycle_history.jsonl"

# Cadence — once per ~30 days. The daemon driver calls run() more
# frequently; we cadence-guard internally.
_RUN_CADENCE_S = 30 * 24 * 3600

# An orphan file must be older than this to be deletable.
_ORPHAN_AGE_S = 30 * 24 * 3600

# Bloat alert threshold (gigabytes).
_BLOAT_THRESHOLD_GB = 5.0


# ── Helpers ──────────────────────────────────────────────────────────────


def _read_registry() -> dict[str, dict[str, Any]]:
    """Read the adapter registry. Returns ``{adapter_name: info_dict}``.

    Empty dict on missing or malformed file — fail-soft, let the
    monitor surface the absence rather than crashing.
    """
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(_REGISTRY_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.debug("adapter_lifecycle: registry read failed", exc_info=True)
        return {}


def _registry_paths(registry: dict[str, dict[str, Any]]) -> set[str]:
    """Return the absolute paths referenced by the registry."""
    out: set[str] = set()
    for entry in registry.values():
        if not isinstance(entry, dict):
            continue
        p = entry.get("adapter_path") or ""
        if p:
            out.add(str(Path(p).resolve()))
    return out


def _walk_root_paths(root: Path) -> list[Path]:
    """List immediate children of ``root`` (files + dirs). Returns [] if
    ``root`` doesn't exist.

    We don't recurse — adapters are typically directories at the top
    level. If a future format changes that, we revisit.
    """
    if not root.exists() or not root.is_dir():
        return []
    try:
        return sorted(p for p in root.iterdir() if p.name != "registry.json")
    except OSError:
        logger.debug("adapter_lifecycle: %s iterdir failed", root, exc_info=True)
        return []


def _path_size_bytes(p: Path) -> int:
    """Total size of file or directory tree, in bytes. Best-effort."""
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        for sub in p.rglob("*"):
            if sub.is_file():
                try:
                    total += sub.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def _path_mtime(p: Path) -> float:
    """Best-effort mtime for a file or directory tree. Returns 0 on error."""
    try:
        if p.is_file():
            return p.stat().st_mtime
        # Walk and use the most recent mtime in the tree.
        latest = 0.0
        for sub in p.rglob("*"):
            try:
                m = sub.stat().st_mtime
                if m > latest:
                    latest = m
            except OSError:
                continue
        if latest == 0.0:
            latest = p.stat().st_mtime
        return latest
    except OSError:
        return 0.0


def _is_orphan(path: Path, registry_paths: set[str]) -> bool:
    """A path is orphan iff its resolved absolute form isn't in the
    registry path set. The walker passes only top-level entries.
    """
    return str(path.resolve()) not in registry_paths


def _delete_orphan(path: Path) -> bool:
    """Best-effort delete. Returns True on success."""
    try:
        if path.is_dir():
            shutil.rmtree(str(path))
        else:
            path.unlink()
        return True
    except OSError:
        logger.debug("adapter_lifecycle: delete failed for %s", path, exc_info=True)
        return False


# ── Pure pass logic (testable) ───────────────────────────────────────────


def _do_one_pass() -> dict[str, Any]:
    """Run one lifecycle pass. Returns a summary dict for tests + audit.

    Side effects: deletes orphan files, writes history JSONL, emits
    Signal alerts for dead pointers + bloat.
    """
    registry = _read_registry()
    registry_paths = _registry_paths(registry)
    now = time.time()

    summary = {
        "registry_size": len(registry),
        "orphans_examined": 0,
        "orphans_deleted": 0,
        "orphans_too_young": 0,
        "dead_pointers": [],
        "total_bytes": 0,
        "bloat_alert": False,
    }

    # ── Walk both roots, treat children as orphans-or-not ────────────
    for root in (_ADAPTERS_DIR, _MODELS_DIR):
        for path in _walk_root_paths(root):
            summary["total_bytes"] += _path_size_bytes(path)
            if not _is_orphan(path, registry_paths):
                continue
            summary["orphans_examined"] += 1
            mtime = _path_mtime(path)
            if mtime == 0.0 or now - mtime < _ORPHAN_AGE_S:
                summary["orphans_too_young"] += 1
                continue
            # Old AND not in registry → delete.
            if _delete_orphan(path):
                summary["orphans_deleted"] += 1

    # ── Dead-pointer detection ───────────────────────────────────────
    for name, info in registry.items():
        if not isinstance(info, dict):
            continue
        p = info.get("adapter_path") or ""
        if p and not Path(p).exists():
            summary["dead_pointers"].append({
                "adapter_name": name,
                "missing_path": p,
                "training_run_id": info.get("training_run_id", ""),
            })

    # ── Bloat detection ──────────────────────────────────────────────
    if summary["total_bytes"] > _BLOAT_THRESHOLD_GB * (1024 ** 3):
        summary["bloat_alert"] = True

    return summary


def _append_history_snapshot(registry: dict[str, dict[str, Any]]) -> None:
    """Append one timestamped snapshot of the registry to the history
    JSONL. Operators can grep this to reconstruct "when was X promoted,
    what eval_score did it have at the time" — closes the rollback gap
    where the live registry only has the most recent state.
    """
    try:
        from app.life_companion._common import state_path
        path = state_path(_HISTORY_FILE)
        snapshot = {
            "ts": time.time(),
            "registry": {
                name: {
                    k: v for k, v in info.items()
                    if k in (
                        "name", "adapter_path", "training_run_id",
                        "eval_score", "examples_count", "promoted",
                        "created_at", "agent_roles",
                    )
                }
                for name, info in registry.items()
                if isinstance(info, dict)
            },
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, default=str) + "\n")
    except Exception:
        logger.debug("adapter_lifecycle: history append failed", exc_info=True)


# ── Public entry point ───────────────────────────────────────────────────


def run() -> None:
    """One pass — cadence-guarded internally."""
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return

    state["last_run_at"] = now

    try:
        summary = _do_one_pass()
        _append_history_snapshot(_read_registry())
    except Exception:
        logger.debug("adapter_lifecycle: pass raised", exc_info=True)
        write_state_json(_STATE_FILE, state)
        return

    audit_event(
        "adapter_lifecycle_pass",
        registry_size=summary["registry_size"],
        orphans_examined=summary["orphans_examined"],
        orphans_deleted=summary["orphans_deleted"],
        dead_pointers=len(summary["dead_pointers"]),
        total_bytes=summary["total_bytes"],
    )

    # ── Alerts ──────────────────────────────────────────────────────
    alerts: list[str] = []

    if summary["orphans_deleted"] > 0:
        # Routine cleanup — don't alert. Audit is enough.
        pass

    if summary["dead_pointers"]:
        lines = [
            f"  • `{d['adapter_name']}` → missing `{d['missing_path']}`"
            for d in summary["dead_pointers"][:10]
        ]
        alerts.append(
            "🧬 Self-heal: adapter registry has dead pointers — entries\n"
            "reference paths that no longer exist on disk:\n\n"
            + "\n".join(lines)
            + "\n\nFix in the registry manually OR re-run training. "
            "Snapshot trail in `workspace/healing/adapter_lifecycle_history.jsonl`."
        )

    if summary["bloat_alert"]:
        gb = summary["total_bytes"] / (1024 ** 3)
        alerts.append(
            f"💾 Self-heal: training-adapter / trained-model storage "
            f"is at {gb:.1f} GB across "
            f"`{_ADAPTERS_DIR}` + `{_MODELS_DIR}` — past the "
            f"{_BLOAT_THRESHOLD_GB} GB threshold. Inspect for stale "
            f"adapters that survived the orphan check."
        )

    state["last_summary"] = summary
    write_state_json(_STATE_FILE, state)

    for body in alerts:
        send_signal_alert(body, tag="adapter_lifecycle")
