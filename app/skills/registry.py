"""
skills.registry — JSON-backed skill registry.

One file, ``workspace/skills_registry.json``, lock-protected against
concurrent writes. The schema is intentionally small so the operator can
hand-edit the file in a pinch:

    {
      "morning briefing": {
        "name": "morning briefing",
        "description": "Today's calendar + top emails + weather",
        "task_template": "Summarize my calendar today, top 3 urgent emails ...",
        "args_schema": [],
        "force_tier": null,
        "extra_tools": [],
        "task_hint": "personal_briefing",
        "created_at": "2026-05-08T20:00:00Z",
        "last_run_at": null,
        "run_count": 0,
        "success_count": 0
      },
      ...
    }

Names are normalised (lower-cased, whitespace-collapsed) so ``/skill run
Morning Briefing`` and ``/skill run morning briefing`` hit the same row.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.paths import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

_STORE_PATH = WORKSPACE_ROOT / "skills_registry.json"
_lock = threading.Lock()
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass
class Skill:
    name: str
    task_template: str
    description: str = ""
    args_schema: list[str] = field(default_factory=list)
    force_tier: Optional[str] = None
    extra_tools: list[str] = field(default_factory=list)
    task_hint: str = ""
    created_at: str = ""
    last_run_at: Optional[str] = None
    run_count: int = 0
    success_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        return cls(
            name=d.get("name", ""),
            task_template=d.get("task_template", ""),
            description=d.get("description", ""),
            args_schema=list(d.get("args_schema") or []),
            force_tier=d.get("force_tier"),
            extra_tools=list(d.get("extra_tools") or []),
            task_hint=d.get("task_hint", ""),
            created_at=d.get("created_at", ""),
            last_run_at=d.get("last_run_at"),
            run_count=int(d.get("run_count", 0)),
            success_count=int(d.get("success_count", 0)),
        )


def normalise(name: str) -> str:
    """Normalise a skill name: lowercase + collapse internal whitespace."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def extract_placeholders(template: str) -> list[str]:
    """Return ordered, deduped list of ``{placeholder}`` names in a template."""
    seen: list[str] = []
    for m in _PLACEHOLDER_RE.finditer(template or ""):
        name = m.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def _load() -> dict[str, dict[str, Any]]:
    if not _STORE_PATH.exists():
        return {}
    try:
        data = json.loads(_STORE_PATH.read_text())
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(f"skills.registry: failed to load: {exc}")
    return {}


def _save(state: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(_STORE_PATH)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save_skill(
    name: str,
    task_template: str,
    *,
    description: str = "",
    force_tier: Optional[str] = None,
    extra_tools: Optional[list[str]] = None,
    task_hint: str = "",
) -> Skill:
    """Create or overwrite a skill. Returns the persisted Skill."""
    if not name or not name.strip():
        raise ValueError("skill name must be non-empty")
    if not task_template or not task_template.strip():
        raise ValueError("task_template must be non-empty")

    canonical = normalise(name)
    args = extract_placeholders(task_template)
    skill = Skill(
        name=canonical,
        task_template=task_template.strip(),
        description=(description or "").strip(),
        args_schema=args,
        force_tier=force_tier,
        extra_tools=list(extra_tools or []),
        task_hint=(task_hint or "").strip(),
        created_at=_now_iso(),
    )
    with _lock:
        state = _load()
        # Preserve existing counters when overwriting.
        prev = state.get(canonical)
        if prev:
            skill.run_count = int(prev.get("run_count", 0))
            skill.success_count = int(prev.get("success_count", 0))
            skill.last_run_at = prev.get("last_run_at")
            # Preserve the original creation timestamp on rewrite.
            skill.created_at = prev.get("created_at") or skill.created_at
        state[canonical] = skill.to_dict()
        _save(state)
    logger.info(f"skills.registry: saved skill {canonical!r} (args={args})")
    return skill


def get_skill(name: str) -> Optional[Skill]:
    canonical = normalise(name)
    with _lock:
        state = _load()
    row = state.get(canonical)
    if not row:
        return None
    return Skill.from_dict(row)


def list_skills() -> list[Skill]:
    with _lock:
        state = _load()
    skills = [Skill.from_dict(r) for r in state.values()]
    # Sort: recently-run first, then unused alphabetically.
    skills.sort(key=lambda s: (s.last_run_at or "", s.name), reverse=True)
    return skills


def delete_skill(name: str) -> bool:
    canonical = normalise(name)
    with _lock:
        state = _load()
        if canonical not in state:
            return False
        del state[canonical]
        _save(state)
    logger.info(f"skills.registry: deleted {canonical!r}")
    return True


def record_run_result(name: str, *, success: bool) -> None:
    """Bump per-skill run/success counters after a run completes."""
    canonical = normalise(name)
    with _lock:
        state = _load()
        row = state.get(canonical)
        if not row:
            return
        row["run_count"] = int(row.get("run_count", 0)) + 1
        if success:
            row["success_count"] = int(row.get("success_count", 0)) + 1
        row["last_run_at"] = _now_iso()
        state[canonical] = row
        _save(state)
