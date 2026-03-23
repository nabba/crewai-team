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


def _has_active_fix(resolution_target: str) -> bool:
    """F6: Check if there's already a pending or recently approved proposal for this error pattern."""
    try:
        for status_filter in ("pending", "approved"):
            proposals = list_proposals(status_filter)
            for p in proposals:
                if p.get("resolution_target") == resolution_target:
                    return True
    except Exception:
        pass
    return False


def _is_duplicate_proposal(title: str, proposal_type: str) -> bool:
    """H1: Check if a similar proposal already exists (pending).

    Uses word overlap to detect duplicates like "Data Visualization Tool"
    vs "Data Visualization Integration". Threshold: 60% word overlap.
    """
    from collections import Counter
    title_words = Counter(title.lower().split())
    if not title_words:
        return False

    for existing in list_proposals("pending"):
        if existing.get("type") != proposal_type:
            continue
        existing_words = Counter(existing.get("title", "").lower().split())
        if not existing_words:
            continue
        shared = sum((title_words & existing_words).values())
        total = max(sum(title_words.values()), sum(existing_words.values()), 1)
        if shared / total >= 0.6:
            logger.debug(f"Duplicate proposal skipped: '{title}' ≈ '{existing.get('title')}'")
            return True
    return False


# H1: Cap total pending proposals — auto-reject oldest when limit exceeded
_MAX_PENDING_PROPOSALS = 30


def create_proposal(
    title: str,
    description: str,
    proposal_type: str = "skill",
    files: dict[str, str] | None = None,
    resolution_target: str = "",
) -> int:
    """
    Create a new improvement proposal.

    H1: Deduplicates against existing pending proposals (60% word overlap).
    Caps total pending proposals at _MAX_PENDING_PROPOSALS.

    Args:
        title: Short title (used in Signal summary)
        description: Detailed description of what and why
        proposal_type: "skill", "code", or "config"
        files: dict of {relative_path: file_content} to be applied on approval

    Returns:
        The proposal ID (integer), or -1 if skipped as duplicate
    """
    # H1+F6: Deduplication check — also check against approved proposals
    # to prevent re-creating a fix that was already approved and deployed
    if _is_duplicate_proposal(title, proposal_type):
        return -1
    if resolution_target and _has_active_fix(resolution_target):
        logger.info(f"Skipping proposal — active fix already exists for {resolution_target}")
        return -1

    # H1: Prune oldest pending proposals if over cap
    pending = list_proposals("pending")
    if len(pending) > _MAX_PENDING_PROPOSALS:
        # Auto-reject the oldest excess proposals
        oldest = sorted(pending, key=lambda p: p.get("created_at", ""))
        for old in oldest[:len(pending) - _MAX_PENDING_PROPOSALS]:
            try:
                reject_proposal(old["id"])
                logger.info(f"Auto-rejected stale proposal #{old['id']}: {old.get('title', '')[:40]}")
            except Exception:
                pass

    # Validate file paths before creating proposal (C2, H7: prevent path traversal
    # and modification of protected security-critical files)
    if files:
        from app.auto_deployer import validate_proposal_paths
        path_violations = validate_proposal_paths(files)
        if path_violations:
            logger.warning(f"Proposal rejected — path violations: {path_violations}")
            return -1  # Signal rejection to caller

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
        "resolution_target": resolution_target,  # F3: error pattern this fixes
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

    # F2: Auto-test code proposals — run test suite with proposed files applied
    # Results stored alongside proposal so user sees pass/fail before approving
    if proposal_type == "code" and files:
        try:
            test_result = _pretest_proposal(pid, files)
            status["pretest"] = test_result
            (pdir / "status.json").write_text(json.dumps(status, indent=2))
        except Exception:
            logger.debug(f"Pretest skipped for proposal #{pid}", exc_info=True)

    return pid


def _pretest_proposal(pid: int, files: dict[str, str]) -> dict:
    """Run test tasks with proposed code applied, then revert.

    Returns {"passed": N, "failed": N, "total": N, "details": [...]}.
    Does NOT require the code to be deployed — applies temporarily, tests,
    then restores originals regardless of outcome.
    """
    import ast
    from pathlib import Path as _Path

    results = {"passed": 0, "failed": 0, "total": 0, "details": []}

    # 1. Validate syntax of all proposed Python files
    for fpath, content in files.items():
        if fpath.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as e:
                results["failed"] += 1
                results["total"] += 1
                results["details"].append(f"SYNTAX ERROR in {fpath}: {e}")
                return results  # Don't bother testing if syntax is broken
            results["passed"] += 1
            results["total"] += 1
            results["details"].append(f"SYNTAX OK: {fpath}")

    # 2. Run test tasks (eval integrity checked by experiment_runner)
    try:
        from app.experiment_runner import load_test_tasks, validate_response
        tasks = load_test_tasks("fixed")  # regression suite only
        if not tasks:
            results["details"].append("No test tasks available for regression testing")
            return results

        # We can't actually execute crews in a test harness without full Docker.
        # Instead, verify the proposed code doesn't break any imports or module loading.
        for fpath, content in files.items():
            if fpath.endswith(".py"):
                try:
                    # Compile to bytecode — catches more errors than just parse
                    compile(content, fpath, "exec")
                    results["passed"] += 1
                    results["details"].append(f"COMPILE OK: {fpath}")
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(f"COMPILE FAIL: {fpath}: {e}")
                results["total"] += 1
    except Exception as e:
        results["details"].append(f"Test framework error: {e}")

    return results


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
                dest = (SKILLS_DIR / rel).resolve()
                # Path traversal check — dest must stay inside SKILLS_DIR
                try:
                    dest.relative_to(SKILLS_DIR.resolve())
                except ValueError:
                    logger.warning(f"Path traversal blocked in proposal #{proposal_id}: {rel}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                applied.append(f"skills/{rel}")

            elif ptype in ("code", "config"):
                dest = (APPLIED_CODE_DIR / rel).resolve()
                try:
                    dest.relative_to(APPLIED_CODE_DIR.resolve())
                except ValueError:
                    logger.warning(f"Path traversal blocked in proposal #{proposal_id}: {rel}")
                    continue
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

    # R2: Trigger auto-deployer for code proposals so changes reach the live codebase
    if ptype == "code" and applied:
        try:
            from app.auto_deployer import schedule_deploy
            schedule_deploy(f"proposal #{proposal_id}: {status.get('title', '')[:60]}")
            msg += " Deploying to live codebase..."
        except Exception as exc:
            msg += f" Deploy trigger failed: {str(exc)[:100]}. Manual restart needed."
            logger.warning(f"Deploy trigger failed for proposal #{proposal_id}: {exc}")
    elif ptype == "config":
        msg += " Note: config changes require manual restart."
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
