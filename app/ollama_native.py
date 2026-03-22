"""
ollama_native.py — Native Ollama integration for macOS Metal GPU acceleration.

Replaces ollama_fleet.py (Docker-based) with direct access to the native
Ollama installation.  Native Ollama on macOS uses Metal GPU automatically,
giving 5-10x faster inference compared to Docker containers (which run
in a Linux VM without GPU access).

Drop-in replacement: same API surface as ollama_fleet.py so llm_factory.py
needs minimal changes.

Lifecycle:
  spawn_model(model) → ensures model is pulled and loaded, returns API URL
  stop_model(model)  → unloads model from memory
  stop_idle_containers() → no-op (native Ollama manages memory)
  stop_all()         → unloads all models
"""

import logging
import threading
import time

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

# S3: Per-model locks so different models can spawn concurrently.
# Also a fast-path check outside the lock for "already loaded" case.
_model_locks: dict[str, threading.Lock] = {}
_model_locks_lock = threading.Lock()  # protects the dict itself
_last_used: dict[str, float] = {}  # model → monotonic timestamp


def _get_model_lock(model: str) -> threading.Lock:
    """Get or create a per-model lock."""
    if model not in _model_locks:
        with _model_locks_lock:
            if model not in _model_locks:
                _model_locks[model] = threading.Lock()
    return _model_locks[model]

PULL_TIMEOUT = 600  # seconds to wait for model pull
STARTUP_TIMEOUT = 60  # seconds to wait for model to load


def _base_url() -> str:
    """Get the native Ollama base URL."""
    return get_settings().ollama_base_url.rstrip("/")


def _gateway_url() -> str:
    """URL for the gateway container to reach host Ollama.

    Inside Docker for Mac, host services are at host.docker.internal.
    Outside Docker (e.g., direct Python), use localhost.
    """
    base = _base_url()
    # Replace localhost with host.docker.internal for containers
    return base.replace("localhost", "host.docker.internal").replace(
        "127.0.0.1", "host.docker.internal"
    )


def _is_running() -> bool:
    """Check if native Ollama is responding (tries gateway URL first for Docker)."""
    for url in (_gateway_url(), _base_url()):
        try:
            r = requests.get(f"{url}/api/tags", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            continue
    return False


def _reachable_url() -> str | None:
    """Return the first Ollama URL that responds (gateway or base)."""
    for url in (_gateway_url(), _base_url()):
        try:
            r = requests.get(f"{url}/api/tags", timeout=3)
            if r.status_code == 200:
                return url
        except Exception:
            continue
    return None


def _get_loaded_models() -> list[str]:
    """Get models currently loaded in GPU memory."""
    url = _reachable_url()
    if not url:
        return []
    try:
        r = requests.get(f"{url}/api/ps", timeout=5)
        if r.status_code == 200:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def ensure_volume() -> None:
    """No-op — native Ollama manages its own storage at ~/.ollama/models/."""
    pass


def spawn_model(model: str) -> str | None:
    """Ensure a model is available and loaded. Returns the API base URL.

    S3: Uses per-model locks so different models spawn concurrently.
    Fast-path: if model is already loaded, returns immediately without locking.
    """
    # Fast-path: check if already loaded without acquiring the lock
    url = _gateway_url()
    try:
        loaded = _get_loaded_models()
        if any(model in m for m in loaded):
            _last_used[model] = time.monotonic()
            return url
    except Exception:
        pass

    # Need to spawn — acquire per-model lock (doesn't block other models)
    with _get_model_lock(model):
        return _spawn_locked(model)


def _spawn_locked(model: str) -> str | None:
    reach = _reachable_url()
    if not reach:
        logger.warning("ollama_native: Ollama not running at %s", _base_url())
        return None

    _last_used[model] = time.monotonic()
    url = _gateway_url()

    # Check if already loaded in GPU memory
    loaded = _get_loaded_models()
    if any(model in m for m in loaded):
        logger.debug("ollama_native: %s already loaded in GPU", model)
        return url

    # Check if model exists locally
    available = get_available_models()
    model_exists = any(model in m for m in available)

    if not model_exists:
        # Pull the model
        logger.info("ollama_native: pulling %s (this may take a while)...", model)
        try:
            with requests.post(
                f"{reach}/api/pull",
                json={"name": model},
                stream=True,
                timeout=PULL_TIMEOUT,
            ) as r:
                for line in r.iter_lines():
                    pass  # consume stream to completion
            logger.info("ollama_native: %s pulled successfully", model)
        except Exception as exc:
            logger.error("ollama_native: failed to pull %s: %s", model, exc)
            return None

    # Warm up — load into GPU memory with a dummy request
    logger.info("ollama_native: loading %s into GPU memory...", model)
    try:
        start = time.monotonic()
        requests.post(
            f"{reach}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "10m"},
            timeout=STARTUP_TIMEOUT,
        )
        load_ms = int((time.monotonic() - start) * 1000)
        logger.info("ollama_native: %s ready (load: %dms)", model, load_ms)
    except Exception as exc:
        logger.warning("ollama_native: warm-up failed for %s: %s", model, exc)
        # Return URL anyway — model might load on first real request
        return url

    return url


def stop_model(model: str) -> None:
    """Unload a model from GPU memory."""
    reach = _reachable_url()
    if not reach:
        return
    try:
        requests.post(
            f"{reach}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "0"},
            timeout=10,
        )
        _last_used.pop(model, None)
        logger.info("ollama_native: unloaded %s from GPU", model)
    except Exception as exc:
        logger.debug("ollama_native: unload failed for %s: %s", model, exc)


def stop_idle_containers(idle_minutes: int = 10) -> None:
    """No-op — native Ollama handles memory management via keep_alive."""
    pass


def stop_all() -> None:
    """Unload all models from GPU memory."""
    for model in list(_last_used.keys()):
        stop_model(model)
    logger.info("ollama_native: all models unloaded")


def get_available_models() -> list[str]:
    """Get all models downloaded in the native Ollama installation."""
    reach = _reachable_url()
    if not reach:
        return []
    try:
        r = requests.get(f"{reach}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def get_fleet_status() -> list[dict]:
    """Return status of loaded models (compatible with fleet API)."""
    loaded = []
    reach = _reachable_url()
    if not reach:
        return loaded
    try:
        r = requests.get(f"{reach}/api/ps", timeout=5)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                name = m.get("name", "unknown")
                size_gb = m.get("size", 0) / 1e9
                vram_gb = m.get("size_vram", 0) / 1e9
                loaded.append({
                    "model": name,
                    "size_gb": round(size_gb, 1),
                    "vram_gb": round(vram_gb, 1),
                    "status": "loaded (Metal GPU)" if vram_gb > 0 else "loaded (CPU)",
                })
    except Exception:
        pass
    return loaded


def format_fleet_status() -> str:
    """Format native Ollama status for Signal display."""
    if not _is_running():
        return "Ollama not running. Start with: ollama serve"

    loaded = get_fleet_status()
    available = get_available_models()

    lines = ["LLM Status (Native Ollama + Metal GPU):\n"]

    if loaded:
        lines.append("Loaded in GPU:")
        for m in loaded:
            lines.append(
                f"  {m['model']} — {m['vram_gb']}GB VRAM, {m['status']}"
            )
    else:
        lines.append("No models currently loaded in GPU.")

    if available:
        lines.append(f"\nAvailable on disk ({len(available)} models):")
        for name in available[:10]:
            marker = " *" if any(name in m["model"] for m in loaded) else ""
            lines.append(f"  {name}{marker}")

    return "\n".join(lines)
