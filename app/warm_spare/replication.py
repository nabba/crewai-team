"""replication — rsync command generation for warm-spare mirroring.

Q17.1. We never execute rsync ourselves — operator's crontab /
LaunchAgent does. Reads the partner target from
``workspace/warm_spare/activation.json`` (preferred) or from
``runtime_settings.warm_spare_partner_target``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_EXCLUDE_PATTERNS = (
    "__pycache__/",
    ".cache/",
    "tmp/",
    "*.pyc",
    "*.tmp",
    ".env",
    ".env.local",
    "secrets/",
    "*.crt",
    "*.key",
    "*.pem",
)


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _activation_path() -> Path:
    return _workspace_root() / "warm_spare" / "activation.json"


def get_partner_target() -> str:
    """Prefer activation.json over runtime_settings (latter gets
    rewritten by the gateway on every save)."""
    p = _activation_path()
    if p.exists():
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
            v = blob.get("partner_target")
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass
    try:
        from app.runtime_settings import get_warm_spare_partner_target
        v = get_warm_spare_partner_target() or ""
        if v:
            return v
    except Exception:
        pass
    return os.environ.get("WARM_SPARE_PARTNER_TARGET", "")


def set_partner_target(value: str) -> None:
    try:
        from app.runtime_settings import set_warm_spare_partner_target
        set_warm_spare_partner_target(value)
    except Exception:
        logger.debug("warm_spare.replication: set_partner_target failed", exc_info=True)


def build_rsync_command(*, dry_run: bool = False) -> dict[str, Any]:
    target = get_partner_target()
    excludes = list(_EXCLUDE_PATTERNS)
    root = _workspace_root()
    args = ["rsync", "-av", "--update", "--partial"]
    if dry_run:
        args.append("--dry-run")
    for pat in excludes:
        args.append(f"--exclude={pat}")
    args.append(str(root).rstrip("/") + "/")
    args.append(target if target else "<configure partner target>")
    return {
        "cmd": " ".join(args),
        "excludes": excludes,
        "target": target,
        "workspace_root": str(root),
        "configured": bool(target),
    }


def write_recipe_file() -> Path:
    out = _workspace_root() / "warm_spare" / "replication_recipe.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = build_rsync_command()
    lines = [
        "# Warm-Spare replication recipe (Q17.1)",
        "#",
        "# OPERATOR-USE. Copy the command line below into your crontab",
        "# (or LaunchAgent, or systemd timer) on the canonical host.",
        "#",
        f"# Configured partner target: {blob['target'] or '<UNCONFIGURED>'}",
        f"# Excludes: {len(blob['excludes'])} patterns",
        "#",
        "# Suggested crontab entry (hourly, top of hour):",
        "#",
        "#   0 * * * * " + blob["cmd"],
        "",
        blob["cmd"],
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
