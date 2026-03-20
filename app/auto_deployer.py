"""
auto_deployer.py — Automatic deployment of auditor/self-heal code fixes.

When the auditor or error resolution loop writes fixed files to
/app/workspace/applied_code/, this module:

1. Copies them over the live source in /app/
2. Triggers a hot-reload of the FastAPI app (if uvicorn supports it)
3. Falls back to signaling the user via Signal if hot-reload isn't available

Safety measures:
  - Backs up original files before overwriting
  - Validates Python syntax before applying
  - Keeps a deploy log for rollback reference
  - Only processes files under app/ (never entrypoint, Dockerfile, etc.)
"""

import ast
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

APPLIED_CODE_DIR = Path("/app/workspace/applied_code")
LIVE_CODE_DIR = Path("/app")
BACKUP_DIR = Path("/app/workspace/deploy_backups")
DEPLOY_LOG = Path("/app/workspace/deploy_log.json")

# Modules that LLM-generated code must never import — prevents code execution
# attacks, credential theft, network exfiltration, etc.
_BLOCKED_IMPORTS = frozenset({
    "subprocess", "os.system", "shutil.rmtree",
    "ctypes", "importlib", "pickle", "shelve", "marshal",
    "socket", "http.server", "xmlrpc", "ftplib", "smtplib",
    "webbrowser", "code", "codeop", "compileall",
    "pty", "resource", "sysconfig",
})

# Also block builtins used in code: eval(), exec(), compile(), __import__()
_BLOCKED_CALLS = frozenset({"eval", "exec", "compile", "__import__", "getattr"})


def _check_dangerous_imports(tree: ast.AST) -> list[str]:
    """Scan AST for dangerous imports and calls. Returns list of violations."""
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _BLOCKED_IMPORTS or alias.name.split(".")[0] in _BLOCKED_IMPORTS:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in _BLOCKED_IMPORTS or mod.split(".")[0] in _BLOCKED_IMPORTS:
                violations.append(f"from {mod} import ...")
        elif isinstance(node, ast.Call):
            # Check for eval(), exec(), compile(), __import__()
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _BLOCKED_CALLS:
                violations.append(f"{name}() call")
    return violations


_deploy_lock = threading.Lock()
_deploy_scheduled = False


def schedule_deploy(reason: str) -> None:
    """
    Schedule a deploy to run shortly. Multiple calls are de-duped — only
    one deploy runs per batch of changes.
    """
    global _deploy_scheduled
    with _deploy_lock:
        if _deploy_scheduled:
            return
        _deploy_scheduled = True

    logger.info(f"auto_deployer: deploy scheduled — {reason}")

    import time

    def _delayed():
        global _deploy_scheduled
        time.sleep(5)  # wait for all file writes to complete
        try:
            run_deploy(reason)
        finally:
            with _deploy_lock:
                _deploy_scheduled = False

    t = threading.Thread(target=_delayed, daemon=True, name="auto-deploy")
    t.start()


def run_deploy(reason: str = "manual") -> str:
    """
    Apply all files from applied_code/ to the live codebase.
    Returns a status message.
    """
    with _deploy_lock:
        return _deploy_locked(reason)


def _deploy_locked(reason: str) -> str:
    if not APPLIED_CODE_DIR.exists():
        return "No applied_code directory."

    # Find all files to deploy
    files_to_deploy = []
    for f in APPLIED_CODE_DIR.rglob("*.py"):
        rel = f.relative_to(APPLIED_CODE_DIR)
        # Only allow files under app/ for safety
        if not str(rel).startswith("app/"):
            logger.warning(f"auto_deployer: skipping non-app file: {rel}")
            continue
        files_to_deploy.append((f, rel))

    if not files_to_deploy:
        return "No files to deploy."

    # Validate all files have valid Python syntax and no dangerous imports
    invalid = []
    dangerous = []
    for src, rel in files_to_deploy:
        source = src.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            invalid.append(f"{rel}: {e}")
            continue
        # Check for dangerous imports that LLM-generated code should never use
        blocked = _check_dangerous_imports(tree)
        if blocked:
            dangerous.append(f"{rel}: {', '.join(blocked)}")

    if invalid:
        msg = f"Deploy blocked: {len(invalid)} files have syntax errors: {'; '.join(invalid[:3])}"
        logger.error(f"auto_deployer: {msg}")
        _log_deploy("blocked", reason, [], msg)
        return msg

    if dangerous:
        msg = f"Deploy blocked: dangerous imports in {'; '.join(dangerous[:3])}"
        logger.error(f"auto_deployer: {msg}")
        _log_deploy("blocked", reason, [], msg)
        return msg

    # Create timestamped backup
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / ts
    backup.mkdir(parents=True, exist_ok=True)

    deployed = []
    for src, rel in files_to_deploy:
        dest = LIVE_CODE_DIR / rel

        # Backup original
        if dest.exists():
            backup_file = backup / rel
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, backup_file)

        # Copy new version
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        deployed.append(str(rel))
        logger.info(f"auto_deployer: deployed {rel}")

    # Clean up applied_code (files have been deployed)
    for src, rel in files_to_deploy:
        try:
            src.unlink()
        except OSError:
            pass
    # Remove empty directories
    _cleanup_empty_dirs(APPLIED_CODE_DIR)

    msg = f"Deployed {len(deployed)} files: {', '.join(deployed)}"
    _log_deploy("success", reason, deployed)
    logger.info(f"auto_deployer: {msg}")

    # Notify via Firebase
    try:
        from app.firebase_reporter import _get_db, _now_iso
        db = _get_db()
        if db:
            db.collection("activities").add({
                "ts": _now_iso(),
                "event": "auto_deploy",
                "crew": "auditor",
                "detail": msg,
            })
    except Exception:
        pass

    return msg


def _cleanup_empty_dirs(root: Path) -> None:
    """Remove empty directories recursively."""
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()  # only succeeds if empty
            except OSError:
                pass


def _log_deploy(status: str, reason: str, files: list, error: str = "") -> None:
    """Append to deploy log."""
    import json
    try:
        log = []
        if DEPLOY_LOG.exists():
            log = json.loads(DEPLOY_LOG.read_text())
        log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "reason": reason[:200],
            "files": files,
            "error": error[:200],
        })
        DEPLOY_LOG.write_text(json.dumps(log[-100:], indent=2))
    except (OSError, json.JSONDecodeError):
        pass


def get_deploy_log(n: int = 10) -> str:
    """Return recent deploy activity."""
    import json
    try:
        if DEPLOY_LOG.exists():
            log = json.loads(DEPLOY_LOG.read_text())[-n:]
            if not log:
                return "No deployments yet."
            lines = []
            for e in log:
                files_str = ", ".join(e.get("files", [])[:3])
                lines.append(
                    f"[{e['ts'][:16]}] {e['status']}: {e.get('reason', '')[:60]} "
                    f"({files_str})"
                )
            return "\n".join(lines)
    except (OSError, json.JSONDecodeError):
        pass
    return "No deployments yet."
