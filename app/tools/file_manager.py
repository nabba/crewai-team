from crewai.tools import tool
import pathlib
from app.audit import log_tool_blocked

WORKSPACE = pathlib.Path("/app/workspace").resolve()

# Subdirectories agents are allowed to write to.
# Everything else (memory/, conversations.db, .git/, crewai_storage/) is blocked.
_WRITABLE_DIRS = {"output", "skills", "proposals"}

# Paths that must never be read or written, even if under an allowed dir.
_BLOCKED_NAMES = {"conversations.db", ".git", "crewai_storage", "audit.log"}

# Files and directories that must never be readable (contain secrets or sensitive data)
_READ_BLOCKED_NAMES = {
    ".git", ".env", "crewai_storage", "audit.log",
    "conversations.db", "llm_benchmarks.db",
    "firebase-service-account.json",
}
# Extensions that should never be readable
_READ_BLOCKED_EXTENSIONS = {".db", ".sqlite", ".sqlite3", ".key", ".pem", ".p12"}


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
def file_manager(action: str, path: str = "", content: str = "") -> str:
    """
    Read, write, or list files in the workspace.

    actions:
      - 'read'  : return file contents (up to 32 KB, path required)
      - 'write' : create/overwrite a file (path + content required)
      - 'list'  : enumerate files in a directory, sorted newest first
                  (path is the directory; defaults to 'output/responses/'
                  when empty — the common case for "find the latest
                  response report")

    path: relative path within workspace/ (e.g., 'output/report.md' or
          'output/responses/' for listing).

    Writable directories: output/, skills/, proposals/.

    Example usage to find the latest response file then read it::

        file_manager(action='list',  path='output/responses/')
        # → "Found 3 file(s) in output/responses/ (newest first):
        #       - response_20260424_074649.md  (9633 bytes, 2026-04-24 10:46)
        #       - ..."
        file_manager(action='read',  path='output/responses/response_20260424_074649.md')
    """
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    # Default list path — the most common discovery target.
    if action == "list" and not path:
        path = "output/responses/"

    target = (WORKSPACE / path).resolve()

    # Path traversal check — must stay within workspace
    try:
        target.relative_to(WORKSPACE)
    except ValueError:
        log_tool_blocked("file_manager", "unknown", f"path traversal attempt: {path[:100]!r}")
        return "Error: Path traversal detected. Access denied."

    if action == "read":
        if not target.exists():
            return (
                f"Error: File not found: {path}.  Tip: use "
                f"action='list' to enumerate available files first."
            )
        # Block reading sensitive files
        for part in target.relative_to(WORKSPACE).parts:
            if part in _READ_BLOCKED_NAMES:
                log_tool_blocked("file_manager", "unknown",
                                 f"read of sensitive path: {path[:100]!r}")
                return f"Error: Access denied to '{path}'. This file contains sensitive data."
        if target.suffix.lower() in _READ_BLOCKED_EXTENSIONS:
            log_tool_blocked("file_manager", "unknown",
                             f"read of blocked extension: {path[:100]!r}")
            return f"Error: Access denied to '{path}'. File type not allowed."
        # Check file size before reading to prevent OOM on huge files
        file_size = target.stat().st_size
        if file_size > 10_000_000:  # 10 MB
            return f"Error: File too large ({file_size:,} bytes). Max 10 MB."
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
    elif action == "list":
        if not target.exists():
            return f"Directory not found: {path}"
        if not target.is_dir():
            return (
                f"Path {path!r} is not a directory.  Use action='read' "
                f"to read a file."
            )
        # Skip any part of the path in the sensitive-names blocklist.
        for part in target.relative_to(WORKSPACE).parts:
            if part in _READ_BLOCKED_NAMES:
                log_tool_blocked("file_manager", "unknown",
                                 f"list of sensitive path: {path[:100]!r}")
                return f"Error: Access denied to '{path}'. Sensitive directory."
        from datetime import datetime, timezone
        entries = []
        for p in target.iterdir():
            if p.name.startswith("."):
                continue  # skip dotfiles
            if p.name in _READ_BLOCKED_NAMES:
                continue
            try:
                st = p.stat()
                entries.append((
                    p.name,
                    st.st_size,
                    st.st_mtime,
                    "dir" if p.is_dir() else "file",
                ))
            except Exception:
                continue
        # Newest first — matches the common "find the latest" use case.
        entries.sort(key=lambda e: -e[2])
        if not entries:
            return f"{path}: (empty)"
        lines = [f"Found {len(entries)} entries in {path} (newest first):"]
        for name, size, mtime, kind in entries[:50]:
            ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
            suffix = "/" if kind == "dir" else ""
            lines.append(f"  - {name}{suffix}  ({size:,} bytes, {ts})")
        if len(entries) > 50:
            lines.append(f"  ... and {len(entries) - 50} more (showing 50 newest)")
        return "\n".join(lines)
    else:
        return (
            f"Error: Unknown action {action!r}. "
            f"Use 'read', 'write', or 'list'."
        )


# ── Tool registry annotation (Phase 1a, passive) ────────────────────
try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="file_manager",
        capabilities=["reads-file", "writes-file"],
        description=(
            "Read, write, or list files in the workspace. Actions: "
            "`read` (path required), `write` (path + content), "
            "`list` (path = directory, defaults to "
            "'output/responses/'). Path is relative to the workspace "
            "root; absolute paths and traversal are sanitized."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _file_manager_registry_factory():
        return file_manager
except ImportError:
    pass
