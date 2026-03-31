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
    "subprocess", "os", "sys", "shutil",
    "ctypes", "importlib", "pickle", "shelve", "marshal",
    "socket", "http.server", "xmlrpc", "ftplib", "smtplib",
    "webbrowser", "code", "codeop", "compileall",
    "pty", "resource", "sysconfig",
    "yaml", "signal", "multiprocessing", "tempfile",
})

# Block dangerous builtins — both direct calls and attribute-based access.
# "open" blocks arbitrary file I/O; "type" blocks __subclasses__ gadget chains;
# "breakpoint" blocks debugger access.
_BLOCKED_CALLS = frozenset({
    "eval", "exec", "compile", "__import__", "getattr",
    "globals", "locals", "vars", "dir", "delattr", "setattr",
    "open", "type", "breakpoint", "input", "help",
})

# Attribute names that must NEVER appear in code — even as obj.attr access.
# These enable sandbox escapes via Python's object model.
_BLOCKED_ATTRS = frozenset({
    "__builtins__", "__subclasses__", "__bases__", "__mro__",
    "__class__", "__globals__", "__code__", "__func__",
    "__self__", "__dict__",  # introspection used in gadget chains
})

# Files that self-modification systems (evolution, self-heal, auditor) must
# NEVER be allowed to modify.  These enforce the security boundary.
PROTECTED_FILES = frozenset({
    "app/sanitize.py",
    "app/security.py",
    "app/vetting.py",
    "app/auto_deployer.py",
    "app/rate_throttle.py",
    "app/circuit_breaker.py",
    "app/config.py",
    "app/main.py",
    "app/experiment_runner.py",
    "app/evolution.py",
    "app/proposals.py",
    "app/signal_client.py",
    "app/firebase_reporter.py",
    "entrypoint.sh",
    "Dockerfile",
    "docker-compose.yml",
    "dashboard/firestore.rules",
    # Soul / constitution files — define system values, identity, and behavioral
    # constraints.  These must never be modified by self-improvement proposals.
    # Internal Python code only READS these files (via loader.py); nothing writes them.
    "app/souls/constitution.md",
    "app/souls/commander.md",
    "app/souls/loader.py",
    "app/souls/style.md",
    "app/souls/agents_protocol.md",
    "app/souls/coder.md",
    "app/souls/critic.md",
    "app/souls/researcher.md",
    "app/souls/writer.md",
    "app/souls/media_analyst.md",
    "app/souls/self_improver.md",
    # Homeostatic regulation module — contains immutable set-point TARGETS.
    # Runtime state lives in workspace/homeostasis.json (not protected).
    "app/self_awareness/homeostasis.py",
})


def validate_proposal_paths(files: dict[str, str]) -> list[str]:
    """Validate all file paths in a proposal. Returns list of violations.

    Checks:
      - No path traversal (.. or absolute paths)
      - No protected files
      - Only allowed directories (app/, skills/, or bare filenames like skill.md)
    """
    violations = []
    for fpath in files:
        # Block path traversal
        if ".." in fpath or fpath.startswith("/"):
            violations.append(f"Path traversal blocked: {fpath}")
            continue
        # Normalize
        normalized = str(Path(fpath))
        # Block protected files
        if normalized in PROTECTED_FILES:
            violations.append(f"Protected file: {normalized}")
            continue
        # Allow: app/ subdirs, skills/ subdir, or bare filenames (skills, configs)
        has_dir = "/" in normalized
        if has_dir and not (normalized.startswith("app/") or normalized.startswith("skills/")):
            violations.append(f"Outside allowed directories: {normalized}")
    return violations


# Constitutional invariants — evolved code must NEVER remove these.
# Checked at deploy time (not just AST safety).
_CONSTITUTIONAL_IMPORTS = {
    # Security essentials that must never be removed from a file that has them
    "app.sanitize": ["sanitize_input", "wrap_user_input"],
    "app.vetting": ["vet_response"],
    "app.security": ["is_authorized_sender"],
}


def _check_constitutional_invariants(src_path: Path, new_source: str) -> list[str]:
    """Verify that evolved code doesn't remove constitutional security imports.

    Compares the new source against the existing live file. If the live file
    imports a constitutional module and the new version doesn't, it's blocked.
    """
    violations = []
    dest = LIVE_CODE_DIR / src_path
    if not dest.exists():
        return []  # new file, no baseline to compare

    try:
        old_source = dest.read_text()
    except OSError:
        return []

    for module, functions in _CONSTITUTIONAL_IMPORTS.items():
        # Check if the old file imported this module
        if module in old_source:
            # Verify the new file still imports it
            if module not in new_source:
                violations.append(
                    f"Constitutional violation: removed import of {module}"
                )
            # Check individual functions
            for func in functions:
                if func in old_source and func not in new_source:
                    violations.append(
                        f"Constitutional violation: removed {func} from {module}"
                    )
    return violations


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
            # Block dangerous builtin calls (bare Name nodes)
            # e.g., block `eval(x)` but allow `re.compile(pattern)`
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                # Block dangerous method calls on any object
                if func.attr in ("eval", "exec", "__import__", "open"):
                    name = func.attr
            if name in _BLOCKED_CALLS:
                violations.append(f"{name}() call")
        elif isinstance(node, ast.Attribute):
            # Block access to dangerous dunder attributes used in sandbox escapes
            # e.g., obj.__subclasses__, obj.__bases__, obj.__mro__
            if node.attr in _BLOCKED_ATTRS:
                violations.append(f"blocked attribute access: {node.attr}")
        elif isinstance(node, ast.Name):
            # Block direct reference to __builtins__ (used in __builtins__["eval"])
            if node.id in _BLOCKED_ATTRS:
                violations.append(f"blocked name reference: {node.id}")
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

    # Notify user via Signal that auto-deploy is starting
    try:
        from app.signal_client import send_message
        from app.config import get_settings
        s = get_settings()
        send_message(
            s.signal_owner_number,
            f"🚀 AUTO-DEPLOY INITIATED: {reason[:100]}\n"
            f"Validating syntax, safety checks, deploying in 5s...",
        )
    except Exception:
        pass

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

    # Step 8: Constitutional invariant check — evolved code must not remove security imports
    constitutional = []
    for src, rel in files_to_deploy:
        violations = _check_constitutional_invariants(rel, src.read_text())
        constitutional.extend(violations)
    if constitutional:
        msg = f"Deploy blocked: {'; '.join(constitutional[:3])}"
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
            try:
                shutil.copy2(dest, backup_file)
            except OSError:
                pass  # backup is best-effort

        # Copy new version — may fail on read-only filesystem (Docker read_only: true)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            deployed.append(str(rel))
            logger.info(f"auto_deployer: deployed {rel}")
        except OSError as exc:
            # R2: Read-only filesystem — log clearly so user knows to remove
            # read_only:true from docker-compose.yml if they want code self-modification.
            logger.warning(
                f"auto_deployer: cannot deploy {rel} — filesystem is read-only. "
                f"Remove 'read_only: true' from docker-compose.yml gateway service "
                f"to enable code self-modification. Error: {exc}"
            )
            _log_deploy("blocked", reason, [], f"Read-only filesystem: {rel}")
            return (
                f"Deploy blocked: filesystem is read-only. "
                f"To enable code self-modification, remove 'read_only: true' "
                f"from the gateway service in docker-compose.yml and restart."
            )

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

    # F4: Hot-reload with verification + auto-rollback on failure.
    # If any module fails to reload, restore ALL files from backup and abort.
    reloaded, reload_errors = _hot_reload_modules_safe(deployed, backup)
    if reload_errors:
        msg += f" RELOAD FAILURES (auto-reverted): {', '.join(reload_errors)}"
        logger.error(f"auto_deployer: {len(reload_errors)} reload failures — reverted all files")
    elif reloaded:
        msg += f" Hot-reloaded: {', '.join(reloaded)}"
        logger.info(f"auto_deployer: hot-reloaded {len(reloaded)} modules")

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

    # Notify user via Signal about successful deploy
    if deployed and not reload_errors:
        try:
            from app.signal_client import send_message
            from app.config import get_settings
            s = get_settings()
            send_message(
                s.signal_owner_number,
                f"✅ DEPLOYED: {', '.join(deployed[:3])} ({reason[:60]}). "
                f"Monitoring for 60s...",
            )
        except Exception:
            pass

    # Start post-deploy error monitoring in background
    if deployed and not reload_errors:
        monitor = threading.Thread(
            target=_post_deploy_monitor,
            args=(deployed, backup, reason),
            daemon=True,
            name="deploy-monitor",
        )
        monitor.start()

    return msg


def _hot_reload_modules_safe(deployed_files: list[str], backup_dir: Path | None) -> tuple[list[str], list[str]]:
    """Hot-reload deployed modules with verification and auto-rollback.

    F4: If ANY module fails to reload, restore ALL deployed files from backup
    and return the error list. This prevents the system from entering a
    hybrid state (disk has new code, memory has old code).

    Returns (reloaded_modules, error_list). If error_list is non-empty,
    rollback was performed.
    """
    import importlib
    import sys
    import shutil

    reloaded = []
    errors = []
    py_modules = []

    for filepath in deployed_files:
        if not filepath.endswith(".py"):
            continue
        module_name = filepath[:-3].replace("/", ".")
        if module_name in sys.modules:
            py_modules.append((filepath, module_name))

    # Attempt reload of all modules
    for filepath, module_name in py_modules:
        try:
            importlib.reload(sys.modules[module_name])
            reloaded.append(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            logger.error(f"auto_deployer: reload FAILED for {module_name}: {exc}")

    # F4: If any reload failed, rollback ALL deployed files from backup
    if errors and backup_dir and backup_dir.exists():
        logger.warning(f"auto_deployer: {len(errors)} reload failures — rolling back ALL files")
        rollback_count = 0
        for filepath in deployed_files:
            backup_file = backup_dir / filepath
            live_file = Path("/app") / filepath
            if backup_file.exists():
                try:
                    shutil.copy2(backup_file, live_file)
                    rollback_count += 1
                except OSError as e:
                    logger.error(f"auto_deployer: rollback failed for {filepath}: {e}")
        logger.info(f"auto_deployer: rolled back {rollback_count}/{len(deployed_files)} files")

        # Re-reload the restored modules to get back to known state
        for filepath, module_name in py_modules:
            try:
                importlib.reload(sys.modules[module_name])
            except Exception:
                pass  # best effort — at least files are restored on disk

        _log_deploy("rollback", f"Auto-rollback: {'; '.join(errors)}", deployed_files)

    return reloaded, errors


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


def _post_deploy_monitor(deployed_files: list[str], backup_dir: Path, reason: str) -> None:
    """Monitor for error spike after deploy and auto-rollback if detected.

    Runs in a background thread. Waits 60s, then checks if error rate spiked.
    If errors increased significantly, restores backup and notifies via Signal.
    """
    import time as _time

    # Wait for deployed code to be exercised
    _time.sleep(60)

    try:
        from app.self_heal import get_recent_errors
        errors = get_recent_errors(20)
        # Count errors in last 2 minutes (should include post-deploy period)
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        recent = [e for e in errors if e.get("ts", "") > cutoff]

        if len(recent) >= 3:
            # Error spike detected — rollback
            logger.warning(
                f"auto_deployer: {len(recent)} errors in 2 min after deploy — "
                f"auto-rolling back {len(deployed_files)} files"
            )
            import shutil as _shutil
            rolled = 0
            for filepath in deployed_files:
                backup_file = backup_dir / filepath
                live_file = Path("/app") / filepath
                if backup_file.exists():
                    try:
                        _shutil.copy2(backup_file, live_file)
                        rolled += 1
                    except OSError:
                        pass

            # Hot-reload restored modules
            import importlib, sys
            for filepath in deployed_files:
                if filepath.endswith(".py"):
                    mod = filepath[:-3].replace("/", ".")
                    if mod in sys.modules:
                        try:
                            importlib.reload(sys.modules[mod])
                        except Exception:
                            pass

            _log_deploy("auto_rollback", f"Error spike ({len(recent)} in 2min) after: {reason}", deployed_files)

            # Notify user via Signal
            try:
                from app.signal_client import send_message
                from app.config import get_settings
                s = get_settings()
                send_message(
                    s.signal_owner_number,
                    f"⚠️ AUTO-ROLLBACK: {len(recent)} errors detected after deploying "
                    f"{', '.join(deployed_files[:3])}. Reverted {rolled} files to backup. "
                    f"Reason: {reason[:80]}",
                )
            except Exception:
                pass

            logger.info(f"auto_deployer: auto-rollback complete — {rolled} files restored")
        else:
            logger.info(f"auto_deployer: post-deploy check OK — {len(recent)} errors in 2min (threshold: 3)")
    except Exception as exc:
        logger.debug(f"auto_deployer: post-deploy monitor error: {exc}")


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
