"""successor — designated-successor declaration file (Q17.4).

Operator-authored, system never acts on it. Carries handoff
instructions, signal_id, email, signature_phrase, paths to consult.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_FILE_NAME = "successor.json"
_DEFAULT_HANDOFF_PATHS = (
    "wiki/self/",
    "identity/continuity_ledger.jsonl",
    "self_model/agreement_ledger.jsonl",
    "epistemic/",
    "lessons_learned/",
    "personality/",
    "souls/",
)


@dataclass
class SuccessorDeclaration:
    ts: str
    successor_name: str
    successor_signal_id: str | None = None
    successor_email: str | None = None
    trigger_conditions: list[str] = field(default_factory=lambda: ["explicit_designation"])
    instructions: str = ""
    signature_phrase: str = ""
    knowledge_handoff_paths: list[str] = field(default_factory=lambda: list(_DEFAULT_HANDOFF_PATHS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _successor_path() -> Path:
    return _workspace_root() / "operator_transition" / _FILE_NAME


def load_successor() -> SuccessorDeclaration | None:
    p = _successor_path()
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return SuccessorDeclaration(
            ts=raw["ts"],
            successor_name=raw["successor_name"],
            successor_signal_id=raw.get("successor_signal_id"),
            successor_email=raw.get("successor_email"),
            trigger_conditions=list(raw.get("trigger_conditions") or []),
            instructions=raw.get("instructions", ""),
            signature_phrase=raw.get("signature_phrase", ""),
            knowledge_handoff_paths=list(raw.get("knowledge_handoff_paths") or []),
        )
    except KeyError:
        return None


def save_successor(decl: SuccessorDeclaration) -> Path:
    p = _successor_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(decl.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="q17_landmark",
            actor="operator_transition",
            summary="successor declaration updated",
            detail={"subsystem": "operator_transition", "successor_name": decl.successor_name, "trigger_conditions": decl.trigger_conditions},
        )
    except Exception:
        pass
    return p


def declare_successor(
    successor_name: str,
    *,
    signal_id: str | None = None,
    email: str | None = None,
    instructions: str = "",
    signature_phrase: str = "",
) -> SuccessorDeclaration:
    decl = SuccessorDeclaration(
        ts=datetime.now(timezone.utc).isoformat(),
        successor_name=successor_name,
        successor_signal_id=signal_id,
        successor_email=email,
        instructions=instructions,
        signature_phrase=signature_phrase,
    )
    save_successor(decl)
    return decl
