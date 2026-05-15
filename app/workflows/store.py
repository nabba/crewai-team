"""Workflow template persistence — per-record JSON.

Mirrors the threads + change_requests + architecture_requests
pattern. One file per template at
``workspace/workflows/templates/<id>.json``. Runs land under
``workspace/workflows/runs/<run_id>.json`` (see queue.py).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.workflows.models import WorkflowTemplate

logger = logging.getLogger(__name__)


_DEFAULT_BASE_DIR = Path("/app/workspace/workflows")
_base_dir_override: Path | None = None
_LOCK = threading.RLock()
_INDEX: dict[str, WorkflowTemplate] | None = None


def _base_dir() -> Path:
    return _base_dir_override or _DEFAULT_BASE_DIR


def get_base_dir() -> Path:
    return _base_dir()


def _templates_dir() -> Path:
    d = _base_dir() / "templates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    """Shared with queue.py for per-run JSON storage."""
    d = _base_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_path(template_id: str) -> Path:
    return _templates_dir() / f"{template_id}.json"


def _index() -> dict[str, WorkflowTemplate]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        loaded: dict[str, WorkflowTemplate] = {}
        try:
            for f in _templates_dir().glob("*.json"):
                try:
                    t = WorkflowTemplate.from_dict(json.loads(f.read_text()))
                    loaded[t.id] = t
                except Exception as exc:  # noqa: BLE001
                    logger.warning("workflows: cannot load %s: %s", f, exc)
        except OSError:
            logger.debug("workflows: templates dir unreadable", exc_info=True)
        _INDEX = loaded
        return _INDEX


def _persist(template: WorkflowTemplate) -> None:
    path = _record_path(template.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(template.to_dict(), indent=2, default=str))
    tmp.replace(path)


def save(template: WorkflowTemplate) -> None:
    """Persist a validated template. Caller must call
    :func:`app.workflows.validator.validate_template` first."""
    with _LOCK:
        idx = _index()
        idx[template.id] = template
        _persist(template)


def get(template_id: str) -> WorkflowTemplate | None:
    return _index().get(template_id)


def list_all(*, limit: int = 200) -> list[WorkflowTemplate]:
    items = list(_index().values())
    items.sort(
        key=lambda t: (t.last_run_at or t.created_at or ""),
        reverse=True,
    )
    return items[:limit]


def delete(template_id: str) -> bool:
    """Remove the template. Returns True on success, False if absent.
    Does NOT touch run history (operator-visible audit trail)."""
    with _LOCK:
        idx = _index()
        if template_id not in idx:
            return False
        idx.pop(template_id, None)
        path = _record_path(template_id)
        if path.exists():
            path.unlink()
        return True


def record_run_outcome(template_id: str, *, success: bool, finished_at: str) -> None:
    """Bump counters after a run terminates."""
    with _LOCK:
        idx = _index()
        t = idx.get(template_id)
        if t is None:
            return
        t.run_count += 1
        if success:
            t.success_count += 1
        t.last_run_at = finished_at
        _persist(t)


def reset_for_tests(base_dir: Path | None = None) -> None:
    global _INDEX, _base_dir_override
    with _LOCK:
        _INDEX = None
        _base_dir_override = base_dir
