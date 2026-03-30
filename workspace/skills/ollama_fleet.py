\"""
ollama_fleet.py — Containerized LLM fleet with dynamic spawn/shutdown.

Each model runs in its own Docker container (ollama/ollama image).
All containers share a Docker volume for model storage, so models
are downloaded once and reused across container restarts.

Lifecycle:
  spawn_model(model) → starts container, pulls model if needed, returns API URL
  stop_model(model)  → stops container (model persists in volume)
  stop_idle()        → stops containers idle for >N minutes
  stop_all()         → stops all fleet containers

The gateway reaches fleet containers via Docker networking.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone

import docker
import requests

logger = logging.getLogger(__name__)

OLLAMA_IMAGE = "ollama/ollama:latest"
VOLUME_NAME = "crewai_ollama_models"
CONTAINER_PREFIX = "crewai-ollama-"
NETWORK_NAME = "crewai-team_external"  # docker-compose creates this
BASE_PORT = 11440  # fleet containers use ports 11440+
MAX_CONTAINERS = 3
IDLE_MINUTES = 10
STARTUP_TIMEOUT = 120  # seconds to wait for Ollama to be ready

_fleet_lock = threading.Lock()

# Registry: model_name → {container_name, port, last_used, container_id}
_fleet: dict[str, dict] = {}


def _safe_name(model: str) -> str:
    """Convert model name to a safe container name suffix."""
    return model.replace(":", "-").replace("/", "-").replace(".", "-")


def _get_client() -> docker.DockerClient:
    """Get Docker client. Uses the Docker socket proxy if available."""
    try:
        return docker.DockerClient(base_url="tcp://docker-proxy:2375", timeout=15)
    except Exception:
        return docker.from_env(timeout=15)


def ensure_volume() -> None:
    """Create the shared model volume if it doesn't exist."""
    try:
        client = _get_client()
        try:
            client.volumes.get(VOLUME_NAME)
            logger.debug(f"ollama_fleet: volume {VOLUME_NAME} exists")
        except docker.errors.NotFound:
            client.volumes.create(VOLUME_NAME)
            logger.info(f"ollama_fleet: created volume {VOLUME_NAME}")
    except Exception as exc:
        logger.warning(f"ollama_fleet: could not ensure volume: {exc}")


def _find_free_port() -> int:
    """Find a free port for a new fleet container."""
    used_ports = {info["port"] for info in _fleet.values()}
    for p in range(BASE_PORT, BASE_PORT + 20):
        if p not in used_ports:
            return p
    return BASE_PORT + len(_fleet)


def _discover_existing() -> None:
    """Discover fleet containers from a previous session."""
    try:
        client = _get_client()
        for c in client.containers.list(all=True, filters={"name": CONTAINER_PREFIX}):
            name = c.name
            model_suffix = name.replace(CONTAINER_PREFIX, "")
            # Try to find the model name from container labels or env
            model = None
            for env_var in (c.attrs.get("Config", {}).get("Env", [])):
                if env_var.startswith("OLLAMA_MODEL="):
                    model = env_var.split("=", 1)[1]
                    break
            if not model:
                continue

            # Find the mapped port
            ports = c.attrs.get("NetworkSettings", {}).get("Ports", {})
            port = None
            for k, v in ports.items():
                if v and "11434" in k:
                    port = int(v[0]["HostPort"])
                    break

            if port and model not in _fleet:
                _fleet[model] = {
                    "container_name": name,
                    "container_id": c.id,
                    "port": port,
                    "last_used": time.monotonic(),
                    "status": c.status,
                }
                logger.info(f"ollama_fleet: discovered existing container for {model} on port {port}")
    except Exception as exc:
        logger.debug(f"ollama_fleet: discovery failed: {exc}")


def spawn_model(model: str) -> str:
    """
    Ensure a container is running for this model. Returns the API base URL.

    1. If container exists and is running → return URL
    2. If container exists but stopped → restart it
    3. If no container → create new one
    4. If MAX_CONTAINERS reached → stop least-recently-used first
    5. Pull model into container if not already available
    """
    with _fleet_lock:
        return _spawn_locked(model)


def _spawn_locked(model: str) -> str:
    # Discover containers from previous sessions on first call
    if not _fleet:
        _discover_existing()

    # Check if already running
    if model in _fleet:
        info = _fleet[model]
        info["last_used"] = time.monotonic()
        try:
            client = _get_client()
            c = client.containers.get(info["container_name"])
            if c.status == "running":
                url = f"http://host.docker.internal:{info['port']}"
                if _wait_ready(url, timeout=5):
                    return url
            # Container exists but not running — restart
            c.start()
            url = f"http://host.docker.internal:{info['port']}"
            if _wait_ready(url, timeout=30):
                _ensure_model_in_container(url, model)
                return url
        except docker.errors.NotFound:
            del _fleet[model]  # container was removed externally
        except Exception as exc:
            logger.warning(f"ollama_fleet: restart failed for {model}: {exc}")
            _fleet.pop(model, None)

    # Enforce container limit — stop LRU
    while len([v for v in _fleet.values() if v.get("status") == "running"]) >= MAX_CONTAINERS:
        _stop_lru()

    # Create new container
    port = _find_free_port()
    safe = _safe_name(model)
    container_name = f"{CONTAINER_PREFIX}{safe}"

    try:
        client = _get_client()

        # Remove any leftover container with same name
        try:
            old = client.containers.get(container_name)
            old.remove(force=True)
        except docker.errors.NotFound:
            pass

        logger.info(f"ollama_fleet: spawning container for {model} on port {port}")

        container = client.containers.run(
            OLLAMA_IMAGE,
            detach=True,
            name=container_name,
            ports={"11434/tcp": ("0.0.0.0", port)},
            volumes={VOLUME_NAME: {"bind": "/root/.ollama", "mode": "rw"}},
            environment={"OLLAMA_MODEL": model},
            mem_limit="24g",
            restart_policy={"Name": "no"},
        )

        _fleet[model] = {
            "container_name": container_name,
            "container_id": container.id,
            "port": port,
            "last_used": time.monotonic(),
            "status": "running",
        }

        url = f"http://host.docker.internal:{port}"
        if not _wait_ready(url, timeout=STARTUP_TIMEOUT):
            logger.error(f"ollama_fleet: container for {model} failed to start")
            return url  # return anyway, caller will get connection error

        # Pull/load the model inside the container
        _ensure_model_in_container(url, model)
        logger.info(f"ollama_fleet: {model} ready at {url}")
        return url

    except Exception as exc:
        logger.error(f"ollama_fleet: failed to spawn container for {model}: {exc}")
        # Fallback: try host Ollama
        from app.config import get_settings
        return get_settings().local_llm_base_url


def _wait_ready(url: str, timeout: int = 30) -> bool:
    """Wait for Ollama API to be responsive."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/api/tags", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _ensure_model_in_container(url: str, model: str) -> None:
    """Pull the model inside the container if not already present."""
    try:
        r = requests.get(f"{url}/api/tags", timeout=10)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            if model in models:
                # Pre-load into memory
                requests.post(
                    f"{url}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": -1},
                    timeout=120,
                )
                return

        # Pull the model
        logger.info(f"ollama_fleet: pulling {model} inside container...")
        with requests.post(f"{url}/api/pull", json={"name": model},
                           stream=True, timeout=600) as r:
            for line in r.iter_lines():
                pass  # consume stream
        logger.info(f"ollama_fleet: {model} pulled successfully")

        # Pre-load
        requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": -1},
            timeout=120,
        )
    except Exception as exc:
        logger.warning(f"ollama_fleet: model setup failed for {model}: {exc}")


def stop_model(model: str) -> None:
    """Stop the container for a model."""
    with _fleet_lock:
        info = _fleet.get(model)
        if not info:
            return
        try:
            client = _get_client()
            c = client.containers.get(info["container_name"])
            c.stop(timeout=10)
            c.remove()  # Add cleanup of stopped container
            info["status"] = "stopped"
            logger.info(f"ollama_fleet: stopped container for {model}")
        except Exception as exc:
            logger.debug(f"ollama_fleet: stop failed for {model}: {exc}")
        finally:
            _fleet.pop(model, None)  # Clean up fleet registry


def _stop_lru() -> None:
    """Stop the least-recently-used running container."""
    running = {m: i for m, i in _fleet.items() if i.get("status") == "running"}
    if not running:
        return
    lru_model = min(running, key=lambda m: running[m]["last_used"])
    logger.info(f"ollama_fleet: stopping LRU container: {lru_model}")
    info = _fleet[lru_model]
    try:
        client = _get_client()
        c = client.containers.get(info["container_name"])
        c.stop(timeout=10)
        c.remove()  # Add cleanup of stopped container
        info["status"] = "stopped"
    except Exception:
        _fleet.pop(lru_model, None)


def stop_idle_containers(idle_minutes: int = IDLE_MINUTES) -> None:
    """Stop containers that haven't been used recently."""
    cutoff = time.monotonic() - (idle_minutes * 60)
    with _fleet_lock:
        to_stop = [
            m for m, i in _fleet.items()
            if i.get("status") == "running" and i["last_used"] < cutoff
        ]
    for model in to_stop:
        stop_model(model)
        logger.info(f"ollama_fleet: idle-stopped {model} (>{idle_minutes}min)")


def stop_all() -> None:
    """Stop all fleet containers."""
    with _fleet_lock:
        models = list(_fleet.keys())
    for model in models:
        stop_model(model)
    logger.info("ollama_fleet: all containers stopped")


def get_fleet_status() -> list[dict]:
    """Return status of all managed containers."""
    with _fleet_lock:
        status = []
        for model, info in _fleet.items():
            idle_min = (time.monotonic() - info["last_used"]) / 60
            status.append({
                "model": model,
                "port": info["port"],
                "status": info.get("status", "unknown"),
                "idle_minutes": round(idle_min, 1),
                "container": info["container_name"],
            })
        return status


def get_available_models() -> list[str]:
    """
    Check which models are already downloaded in the shared volume.
    Spawns a temporary container to query the volume, caches for 5 minutes.
    """
    # Quick check: if any fleet container is running, query it
    with _fleet_lock:
        for model, info in _fleet.items():
            if info.get("status") == "running":
                try:
                    url = f"http://host.docker.internal:{info['port']}"
                    r = requests.get(f"{url}/api/tags", timeout=5)
                    if r.status_code == 200:
                        return [m.get("name", "") for m in r.json().get("models", [])]
                except Exception:
                    pass

    # Fallback: check host Ollama (might still be running)
    try:
        from app.config import get_settings
        base = get_settings().local_llm_base_url.rstrip("/")
        r = requests.get(f"{base}/api/tags", timeout=3)
        if r.status_code == 200:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass

    return []


def format_fleet_status() -> str:
    """Format fleet status for Signal display."""
    status = get_fleet_status()
    if not status:
        return "No LLM containers running."
    lines = ["LLM Fleet Status:
"]
    for s in status:
        icon = "🟢" if s["status"] == "running" else "⚫"
        lines.append(
            f"  {icon} {s['model']} — port {s['port']}, "
            f"idle {s['idle_minutes']:.0f}min"
        )
    return "\n".join(lines)