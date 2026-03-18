from crewai.tools import tool
import pathlib
from app.audit import log_tool_blocked

WORKSPACE = pathlib.Path("/app/workspace").resolve()

# Subdirectories agents are allowed to write to.
# Everything else (memory/, conversations.db, .git/, crewai_storage/) is blocked.
_WRITABLE_DIRS = {"output", "skills", "proposals"}

# Paths that must never be written to, even if under an allowed dir.
_BLOCKED_NAMES = {"conversations.db", ".git", "crewai_storage", "audit.log"}


def _is_writable(target: pathlib.Path) -> bool:
    """Check that target is inside one of the allowed writable subdirectories."""
    try:
        rel = target.relative_to(WORKSPACE)
    except ValueError:
        return False
    # First path component must be in the allowlist
    parts = rel.parts
    if not parts or parts[0] not in _WRITABLE_DIRS:
        return False
    # Block sensitive file/dir names anywhere in the path
    for part in parts:
        if part in _BLOCKED_NAMES:
            return False
    return True


@tool("file_manager")
def file_manager(action: str, path: str, content: str = "") -> str:
    """
    Read and write files in the workspace.
    action: 'read' or 'write'
    path: relative path within workspace/ (e.g., 'output/report.md' or 'skills/python.md')
    content: text to write (only for 'write' action)

    Writable directories: output/, skills/, proposals/
    """
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    target = (WORKSPACE / path).resolve()

    # Path traversal check — must stay within workspace
    try:
        target.relative_to(WORKSPACE)
    except ValueError:
        log_tool_blocked("file_manager", "unknown", f"path traversal attempt: {path[:100]!r}")
        return "Error: Path traversal detected. Access denied."

    if action == "read":
        if not target.exists():
            return f"Error: File not found: {path}"
        return target.read_text()[:32000]
    elif action == "write":
        if not _is_writable(target):
            log_tool_blocked("file_manager", "unknown",
                             f"write to restricted path: {path[:100]!r}")
            return (f"Error: Cannot write to '{path}'. "
                    f"Writable directories: {', '.join(sorted(_WRITABLE_DIRS))}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Written to {path} ({len(content)} chars)"
    else:
        return f"Error: Unknown action '{action}'. Use 'read' or 'write'."
