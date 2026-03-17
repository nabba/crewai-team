from crewai.tools import tool
import docker
import tempfile
import pathlib
from app.config import get_settings

settings = get_settings()

WORKSPACE = pathlib.Path("/app/workspace/output").resolve()


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

    # Ensure workspace exists
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    # Write code to a temp file in workspace (readable by sandbox)
    with tempfile.NamedTemporaryFile(
        suffix=ext, dir=WORKSPACE, delete=False, mode="w"
    ) as f:
        f.write(code)
        host_path = pathlib.Path(f.name)

    container_path = f"/sandbox/{host_path.name}"
    # timeout= sets the HTTP socket timeout for the Docker daemon connection itself,
    # distinct from the container execution timeout below
    client = docker.from_env(timeout=10)

    try:
        result = client.containers.run(
            settings.sandbox_image,
            command=[*cmd_parts, container_path],  # List form avoids shell injection
            volumes={str(WORKSPACE): {"bind": "/sandbox", "mode": "ro"}},
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
    except Exception:
        output = "Execution error: sandbox failed to run."
    finally:
        try:
            host_path.unlink()
        except Exception:
            pass

    return output[:4000]
