from crewai.tools import tool
import pathlib
from app.audit import log_tool_blocked

WORKSPACE = pathlib.Path("/app/workspace/output").resolve()


@tool("file_manager")
def file_manager(action: str, path: str, content: str = "") -> str:
    """
    Read and write files scoped to workspace/output/ only.
    action: 'read' or 'write'
    path: relative path within workspace/output/ (e.g., 'report.md')
    content: text to write (only for 'write' action)
    """
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    # Resolve and validate path — must stay within workspace.
    # relative_to() is used instead of startswith() to avoid the classic
    # prefix bypass: /app/workspace/output_evil passes startswith("/app/workspace/output")
    target = (WORKSPACE / path).resolve()
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
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Written to {path} ({len(content)} chars)"
    else:
        return f"Error: Unknown action '{action}'. Use 'read' or 'write'."
