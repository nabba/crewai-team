\"""
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