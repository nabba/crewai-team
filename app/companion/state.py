"""Runtime state sidecar — vruntime, last_tick_at, daily cost.

Stored at ``workspace/companion/state/<project_id>.json``. Atomic writes via
write-temp-then-rename. Lives outside CP ``config_json`` because this is
ephemeral process state, not durable user intent.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.environ.get("COMPANION_STATE_DIR", "workspace/companion/state"))
_LOCK = Lock()


@dataclass
class WorkspaceState:
    """Runtime counters for one workspace's Companion loop."""
    project_id: str = ""
    vruntime_s: float = 0.0
    last_tick_at: float = 0.0
    daily_cost_usd: float = 0.0
    cost_day_key: str = ""
    cycles_total: int = 0
    last_skip_reason: str | None = None


def _path_for(project_id: str) -> Path:
    safe = "".join(c for c in project_id if c.isalnum() or c in "-_") or "default"
    return _STATE_DIR / f"{safe}.json"


def load(project_id: str) -> WorkspaceState:
    """Read state; returns a default-initialised WorkspaceState on miss."""
    p = _path_for(project_id)
    if not p.exists():
        return WorkspaceState(project_id=project_id)
    try:
        raw = json.loads(p.read_text())
        fields = WorkspaceState.__dataclass_fields__
        kwargs = {k: raw[k] for k in fields if k in raw}
        kwargs["project_id"] = project_id
        return WorkspaceState(**kwargs)
    except Exception as exc:
        logger.warning("companion.state: load failed for %s: %s", project_id, exc)
        return WorkspaceState(project_id=project_id)


def save(state: WorkspaceState) -> None:
    """Atomically persist state via temp-file + rename."""
    p = _path_for(state.project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(asdict(state), indent=2, sort_keys=True)
    with _LOCK:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=p.parent, delete=False,
            prefix=".tmp.", suffix=".json",
        ) as tmp:
            tmp.write(serialised)
            tmp_path = tmp.name
        os.replace(tmp_path, p)


def utc_day_key(now_unix: float | None = None) -> str:
    ts = now_unix if now_unix is not None else time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
