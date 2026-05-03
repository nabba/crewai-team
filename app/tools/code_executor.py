from crewai.tools import tool
import docker
import logging
import tempfile
import pathlib
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

WORKSPACE = pathlib.Path("/app/workspace/output").resolve()
# Dedicated temp dir for sandbox — isolates code from other output files
SANDBOX_TMPDIR = pathlib.Path("/app/workspace/output/.sandbox_tmp").resolve()

_WORKSPACE_ROOT = "/app/workspace"

def _to_host_path(container_path: str) -> str:
    """Translate a gateway-container path to its host-OS equivalent.

    The Docker daemon mounting volumes lives on the host, not inside this
    container.  When we ask it to bind-mount SANDBOX_TMPDIR (which is a
    *gateway-internal* path like /app/workspace/output/.sandbox_tmp),
    Docker would try to resolve that path on the host's filesystem — and
    fail, because the host sees it as
    <WORKSPACE_HOST_PATH>/output/.sandbox_tmp instead.  Without this
    translation the sandbox dies with a silent ContainerError before any
    code executes — surfaced to agents as the opaque "sandbox failed to
    run" message.
    """
    host_ws = settings.workspace_host_path or ""
    if host_ws and container_path.startswith(_WORKSPACE_ROOT):
        return host_ws + container_path[len(_WORKSPACE_ROOT):]
    return container_path


@tool("execute_code")
def execute_code(language: str, code: str) -> str:
    """
    Execute code safely inside a Docker sandbox.
    language: 'python', 'bash', 'node', or 'ruby'
    code: the source code to execute
    Returns stdout + stderr, max 4000 chars.
    """
    # Normalize to lowercase so "Python" / "PYTHON" are accepted
    language = language.lower().strip()

    # cmd is a list so it can include flags without shell interpretation risks
    ALLOWED_LANGUAGES: dict[str, tuple[str, list[str]]] = {
        "python": (".py", ["python3"]),
        "bash":   (".sh", ["bash"]),
        # --max-old-space-size caps V8 heap to prevent OOM abuse
        "node":   (".js", ["node", "--max-old-space-size=256"]),
        "ruby":   (".rb", ["ruby"]),
    }

    lang = ALLOWED_LANGUAGES.get(language)
    if lang is None:
        return f"Unsupported language: {language!r}. Allowed: {', '.join(ALLOWED_LANGUAGES)}"
    ext, cmd_parts = lang

    # Prevent agents from submitting huge payloads that fill the shared workspace
    MAX_CODE_BYTES = 512_000  # 512 KB
    if len(code.encode()) > MAX_CODE_BYTES:
        return f"Code too large (max {MAX_CODE_BYTES // 1024} KB)."

    # Use a dedicated temp dir so sandbox can't read other output files
    SANDBOX_TMPDIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=ext, dir=SANDBOX_TMPDIR, delete=False, mode="w"
    ) as f:
        f.write(code)
        host_path = pathlib.Path(f.name)

    container_path = f"/sandbox/{host_path.name}"
    # timeout= sets the HTTP socket timeout for the Docker daemon connection itself,
    # distinct from the container execution timeout below
    client = docker.from_env(timeout=10)

    # Docker daemon runs on the HOST — bind-mount paths must be host
    # paths, not gateway-internal paths.  _to_host_path() handles the
    # translation using WORKSPACE_HOST_PATH from the environment.
    sandbox_mount_host = _to_host_path(str(SANDBOX_TMPDIR))

    try:
        result = client.containers.run(
            settings.sandbox_image,
            command=[*cmd_parts, container_path],  # List form avoids shell injection
            volumes={sandbox_mount_host: {"bind": "/sandbox", "mode": "ro"}},
            network_disabled=True,  # No network in sandbox
            read_only=True,  # No writing to container FS
            mem_limit=settings.sandbox_memory_limit,
            nano_cpus=int(settings.sandbox_cpu_limit * 1e9),
            cap_drop=["ALL"],  # Drop all Linux capabilities
            security_opt=["no-new-privileges:true"],
            remove=True,  # Auto-remove after run
            timeout=settings.sandbox_timeout_seconds,
            stdout=True,
            stderr=True,
        )
        output = result.decode("utf-8", errors="replace")
    except docker.errors.ContainerError as e:
        output = f"Runtime error:\n{e.stderr.decode('utf-8', errors='replace')}"
    except Exception as exc:
        # Surface the actual exception so agents (and diagnostics) see WHY
        # sandboxing failed — not the previous opaque fallback message.
        # Common causes: image not present, volume mount host-path wrong,
        # docker-proxy ACL blocking /containers/create, CPU/memory
        # over-commit.
        logger.exception("code_executor: sandbox run failed — %s: %s",
                         type(exc).__name__, exc)
        output = (
            f"Execution error: sandbox failed to run "
            f"({type(exc).__name__}: {str(exc)[:300]})"
        )
    finally:
        try:
            host_path.unlink()
        except Exception:
            pass

    return output[:4000]


# ── Tool registry annotation (Phase 1a, passive) ────────────────────
try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="execute_code",
        capabilities=["executes-code"],
        description=(
            "Run Python code in the in-process sandbox. 512m RAM, 0.5 "
            "CPU, 30s wall-clock, no network. Use this BEFORE "
            "delivering code to the user — never deliver code you "
            "haven't executed. Returns stdout/stderr (truncated to "
            "4000 chars). Common languages: python, bash."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _execute_code_registry_factory():
        return execute_code
except ImportError:
    pass
