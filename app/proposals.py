"""
proposals.py — Improvement proposal management.

Agents create proposals for system improvements (new skills, tools,
code changes). Proposals are stored in /app/workspace/proposals/ and
require explicit user approval via Signal before being applied.

Proposal types:
  skill   — new .md files for workspace/skills/ (low risk)
  code    — new/modified Python files (requires rebuild)
  config  — .env or docker-compose changes (requires restart)
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROPOSALS_DIR = Path("/app/workspace/proposals")
SKILLS_DIR = Path("/app/workspace/skills")
APPLIED_CODE_DIR = Path("/app/workspace/applied_code")


def _next_id() -> int:
    """Return the next sequential proposal ID."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(d.name.split("_")[0])
        for d in PROPOSALS_DIR.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ]
    return max(existing, default=0) + 1


def create_proposal(
    title: str,
    description: str,
    proposal_type: str = "skill",
    files: dict[str, str] | None = None,
) -> int:
    """
    Create a new improvement proposal.

    Args:
        title: Short title (used in Signal summary)
        description: Detailed description of what and why
        proposal_type: "skill", "code", or "config"
        files: dict of {relative_path: file_content} to be applied on approval

    Returns:
        The proposal ID (integer)
    """
    pid = _next_id()
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40].strip()
    dirname = f"{pid:03d}_{safe_title.replace(' ', '_')}"
    pdir = PROPOSALS_DIR / dirname
    pdir.mkdir(parents=True, exist_ok=True)

    # Write human-readable proposal
    (pdir / "proposal.md").write_text(
        f"# Proposal #{pid}: {title}\n\n"
        f"**Type:** {proposal_type}\n"
        f"**Created:** {datetime.now(timezone.utc).isoformat()}\n\n"
        f"## Description\n\n{description}\n\n"
        f"## Files\n\n"
        + (("\n".join(f"- `{p}`" for p in files) + "\n") if files else "None\n")
    )

    # Write status
    status = {
        "id": pid,
        "title": title,
        "type": proposal_type,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied_at": None,
        "dirname": dirname,
    }
    (pdir / "status.json").write_text(json.dumps(status, indent=2))

    # Write proposed files
    if files:
        files_dir = pdir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        for fpath, fcontent in files.items():
            target = files_dir / fpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fcontent)

    logger.info(f"Proposal #{pid} created: {title} ({proposal_type})")
    return pid


def list_proposals(status_filter: str = "pending") -> list[dict]:
    """List proposals filtered by status."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for d in sorted(PROPOSALS_DIR.iterdir()):
        sf = d / "status.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
                if status_filter == "all" or s.get("status") == status_filter:
                    results.append(s)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def get_proposal(proposal_id: int) -> dict | None:
    """Get a proposal by ID, including file contents."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    for d in PROPOSALS_DIR.iterdir():
        sf = d / "status.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
                if s.get("id") == proposal_id:
                    s["description"] = ""
                    pm = d / "proposal.md"
                    if pm.exists():
                        s["description"] = pm.read_text()
                    # List files
                    files_dir = d / "files"
                    s["files"] = {}
                    if files_dir.exists():
                        for f in files_dir.rglob("*"):
                            if f.is_file():
                                rel = str(f.relative_to(files_dir))
                                s["files"][rel] = f.read_text()[:8000]
                    return s
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def _get_proposal_dir(proposal_id: int) -> Path | None:
    """Find the directory for a proposal by ID."""
    for d in PROPOSALS_DIR.iterdir():
        sf = d / "status.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
                if s.get("id") == proposal_id:
                    return d
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def approve_proposal(proposal_id: int) -> str:
    """
    Approve and apply a proposal.

    - skill proposals: copies .md files to workspace/skills/
    - code proposals: copies files to workspace/applied_code/ for hot-reload
    - config proposals: saves but requires manual restart

    Returns a status message.
    """
    pdir = _get_proposal_dir(proposal_id)
    if not pdir:
        return f"Proposal #{proposal_id} not found."

    status = json.loads((pdir / "status.json").read_text())
    if status.get("status") != "pending":
        return f"Proposal #{proposal_id} is already {status.get('status')}."

    ptype = status.get("type", "skill")
    files_dir = pdir / "files"
    applied = []

    if files_dir.exists():
        for f in files_dir.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(files_dir)

            if ptype == "skill":
                # Skills go directly to workspace/skills/
                dest = SKILLS_DIR / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                applied.append(f"skills/{rel}")

            elif ptype in ("code", "config"):
                # Code goes to applied_code/ for entrypoint overlay
                dest = APPLIED_CODE_DIR / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                applied.append(f"applied_code/{rel}")

    # Update status
    status["status"] = "approved"
    status["applied_at"] = datetime.now(timezone.utc).isoformat()
    status["applied_files"] = applied
    (pdir / "status.json").write_text(json.dumps(status, indent=2))

    logger.info(f"Proposal #{proposal_id} approved: {applied}")

    msg = f"Proposal #{proposal_id} approved and applied."
    if applied:
        msg += f" Files: {', '.join(applied)}"
    if ptype in ("code", "config"):
        msg += " Note: code changes will take effect after container restart."
    return msg


def reject_proposal(proposal_id: int) -> str:
    """Reject a proposal."""
    pdir = _get_proposal_dir(proposal_id)
    if not pdir:
        return f"Proposal #{proposal_id} not found."

    status = json.loads((pdir / "status.json").read_text())
    if status.get("status") != "pending":
        return f"Proposal #{proposal_id} is already {status.get('status')}."

    status["status"] = "rejected"
    status["rejected_at"] = datetime.now(timezone.utc).isoformat()
    (pdir / "status.json").write_text(json.dumps(status, indent=2))

    logger.info(f"Proposal #{proposal_id} rejected")
    return f"Proposal #{proposal_id} rejected."
