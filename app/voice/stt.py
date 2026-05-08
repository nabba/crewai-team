"""
voice.stt — speech-to-text dispatcher.

`transcribe(audio_bytes, audio_format, language) -> str` reads the live
`voice_mode` runtime setting and routes to either the host-binary backend
(whisper.cpp via the bridge) or the cloud backend (Groq Whisper). Every
failure path returns "" so the caller can fall back to plain text.

Dispatch precedence:
  - voice_mode == "off"   → return "" (caller treats inbound as text-less)
  - voice_mode == "local" → local whisper.cpp; on failure, try cloud as a
                            secondary so a bad host install doesn't drop
                            the user's voice note silently
  - voice_mode == "cloud" → Groq Whisper; on failure, try local
"""
from __future__ import annotations

import logging

from app.runtime_settings import get_voice_mode

logger = logging.getLogger(__name__)

# MIME prefixes that the gateway treats as voice attachments.
# signal-cli has been observed to deliver: audio/aac, audio/mp4, audio/m4a,
# audio/ogg, audio/opus. The "audio/" prefix catches all of them.
AUDIO_MIME_PREFIXES: tuple[str, ...] = ("audio/",)


def transcribe(
    audio_bytes: bytes,
    *,
    audio_format: str = "m4a",
    language: str | None = None,
) -> str:
    """Return the text transcript of an audio clip.

    Args:
        audio_bytes: raw bytes of the audio file (any format Whisper accepts —
            m4a, aac, mp3, mp4, ogg, opus, wav).
        audio_format: hint used to name the temp file when the local backend
            writes to disk for whisper.cpp. Cloud backend ignores it.
        language: ISO 639-1 code (e.g. "en", "et", "fi"). None = autodetect.

    Returns:
        Transcribed text on success. Empty string when voice mode is off or
        every backend failed.
    """
    if not audio_bytes:
        return ""

    mode = get_voice_mode()
    if mode == "off":
        return ""

    primary, secondary = _resolve_chain(mode)
    text = primary(audio_bytes, audio_format=audio_format, language=language)
    if text:
        return text
    if secondary is not None:
        text = secondary(audio_bytes, audio_format=audio_format, language=language)
        if text:
            logger.info("voice.stt: primary backend empty, secondary succeeded")
            return text
    return ""


def _resolve_chain(mode: str):
    """Return (primary, secondary) callables based on the requested mode."""
    from app.voice.local import transcribe as local_transcribe
    from app.voice.cloud import transcribe as cloud_transcribe

    if mode == "local":
        return local_transcribe, cloud_transcribe
    if mode == "cloud":
        return cloud_transcribe, local_transcribe
    # Defensive: an unknown mode should never reach here because the
    # validator rejects it, but if it does, treat it as "off".
    return (lambda *a, **kw: ""), None
