"""
voice.local — host-binary STT/TTS backend.

Runs whisper.cpp and Piper on the macOS host via the host bridge so the
work happens on Apple Silicon (Metal-accelerated) instead of inside the
amd64 Docker container. Audio bytes are staged through the shared
``workspace/voice_tmp/`` directory (mounted into both Docker and the host)
so the bridge can read them by host path.

Required host binaries:
    whisper-cli         (brew install whisper-cpp)
    piper               (pip install piper-tts, or download release binary)
    ffmpeg              (brew install ffmpeg) — only used if Piper output
                         needs format conversion (not by MVP)

Required model files on host (downloaded once by host_bridge/install_voice.sh):
    ~/whisper-models/ggml-large-v3.bin
    ~/piper-voices/en_US-lessac-medium.onnx + .json
    ~/piper-voices/et_EE-mart-medium.onnx + .json
    ~/piper-voices/fi_FI-harri-medium.onnx + .json

Override the binary paths or model directories with env vars:
    WHISPER_CPP_BIN, WHISPER_CPP_MODEL, PIPER_BIN, PIPER_VOICE_DIR

Failure modes return "" (STT) or None (TTS) so the dispatcher can fall
back to the cloud backend.
"""
from __future__ import annotations

import logging
import os
import shlex
import uuid
from pathlib import Path

from app.config import get_settings
from app.paths import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

_AGENT_ID = "voice"  # capability identity for bridge_client

# Per-language Piper voice file basenames (without .onnx). Override via env
# var ``PIPER_VOICE_<LANG>``.
#
# Estonian (et_EE) is intentionally absent: rhasspy/piper-voices does NOT
# ship one as of May 2026, and ``_piper_voice_for`` falls back to the
# English voice for unknown languages. For Estonian TTS that actually
# sounds Estonian, run cloud mode (Google Cloud Neural2 has
# ``et-EE-Standard-A``); the dispatcher will route there automatically.
_PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "fi": "fi_FI-harri-medium",
}

_VOICE_TMP_DIR_NAME = "voice_tmp"


def _bridge():
    """Return a bridge client or None if unavailable."""
    try:
        from app.bridge_client import get_bridge
        b = get_bridge(_AGENT_ID)
        if b is None or not b.is_available():
            logger.debug("voice.local: bridge not available")
            return None
        return b
    except Exception:
        logger.debug("voice.local: bridge import failed", exc_info=True)
        return None


def _voice_tmp_dir() -> Path:
    """Docker-side path to the staging directory for voice files."""
    p = WORKSPACE_ROOT / _VOICE_TMP_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_host_path(docker_path: Path) -> str | None:
    """Translate a Docker workspace path to the host equivalent for the bridge."""
    s = get_settings()
    host_root = s.workspace_host_path
    if not host_root:
        # When running natively (no Docker), the host path *is* the workspace path.
        return str(docker_path)
    docker_str = str(docker_path)
    docker_root = "/app/workspace"
    if not docker_str.startswith(docker_root):
        return None
    return host_root.rstrip("/") + docker_str[len(docker_root):]


def _whisper_bin() -> str:
    return os.environ.get("WHISPER_CPP_BIN", "whisper-cli")


def _whisper_model() -> str:
    return os.environ.get(
        "WHISPER_CPP_MODEL",
        str(Path.home() / "whisper-models" / "ggml-large-v3.bin"),
    )


def _piper_bin() -> str:
    return os.environ.get("PIPER_BIN", "piper")


def _piper_voice_dir() -> str:
    return os.environ.get(
        "PIPER_VOICE_DIR",
        str(Path.home() / "piper-voices"),
    )


def _piper_voice_for(language: str) -> str:
    """Resolve a Piper voice basename for the given language."""
    lang = (language or "en").split("-")[0].lower()
    override = os.environ.get(f"PIPER_VOICE_{lang.upper()}")
    if override:
        return override
    return _PIPER_VOICES.get(lang, _PIPER_VOICES["en"])


def transcribe(
    audio_bytes: bytes,
    *,
    audio_format: str = "m4a",
    language: str | None = None,
) -> str:
    """STT via whisper.cpp on the host. Returns "" on any failure."""
    bridge = _bridge()
    if bridge is None:
        return ""

    tmp_dir = _voice_tmp_dir()
    stem = uuid.uuid4().hex
    in_path = tmp_dir / f"{stem}.{audio_format.lstrip('.')}"
    try:
        in_path.write_bytes(audio_bytes)
    except OSError as exc:
        logger.warning(f"voice.local.transcribe: failed to stage audio: {exc}")
        return ""

    host_audio = _to_host_path(in_path)
    if not host_audio:
        logger.warning("voice.local.transcribe: cannot translate to host path")
        _safe_unlink(in_path)
        return ""

    cmd = [
        _whisper_bin(),
        "-m", _whisper_model(),
        "-f", host_audio,
        "-nt",                 # no timestamps — plain transcript on stdout
        "-otxt", "false",
    ]
    if language:
        cmd += ["-l", language]
    else:
        cmd += ["-l", "auto"]

    try:
        result = bridge.execute(cmd, working_dir="/tmp", timeout=60)
    finally:
        _safe_unlink(in_path)

    if not isinstance(result, dict) or "error" in result:
        logger.info(
            f"voice.local.transcribe: bridge error {result.get('error') if isinstance(result, dict) else result}"
        )
        return ""
    if result.get("returncode", 1) != 0:
        logger.info(
            f"voice.local.transcribe: whisper-cli exit {result.get('returncode')}: "
            f"{(result.get('stderr') or '')[:200]}"
        )
        return ""
    return (result.get("stdout") or "").strip()


def synthesize(text: str, *, language: str = "en") -> bytes | None:
    """TTS via Piper on the host. Returns WAV bytes, or None on failure."""
    bridge = _bridge()
    if bridge is None:
        return None

    voice = _piper_voice_for(language)
    voice_path = Path(_piper_voice_dir()) / f"{voice}.onnx"
    # voice_path is on the host filesystem — the bridge resolves it.

    tmp_dir = _voice_tmp_dir()
    stem = uuid.uuid4().hex
    out_path = tmp_dir / f"{stem}.wav"
    host_out = _to_host_path(out_path)
    if not host_out:
        return None

    # Piper reads text from stdin. Bridge.execute has no stdin pipe, so
    # wrap with bash -c using shlex for safe quoting.
    piper_cmd = (
        f"{shlex.quote(_piper_bin())} "
        f"--model {shlex.quote(str(voice_path))} "
        f"--output_file {shlex.quote(host_out)}"
    )
    bash_cmd = f"echo {shlex.quote(text)} | {piper_cmd}"
    cmd = ["bash", "-lc", bash_cmd]

    try:
        result = bridge.execute(cmd, working_dir="/tmp", timeout=30)
        if not isinstance(result, dict) or "error" in result:
            logger.info(
                f"voice.local.synthesize: bridge error "
                f"{result.get('error') if isinstance(result, dict) else result}"
            )
            return None
        if result.get("returncode", 1) != 0:
            logger.info(
                f"voice.local.synthesize: piper exit {result.get('returncode')}: "
                f"{(result.get('stderr') or '')[:200]}"
            )
            return None
        if not out_path.exists():
            logger.info("voice.local.synthesize: piper produced no output file")
            return None
        return out_path.read_bytes()
    finally:
        _safe_unlink(out_path)


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass
