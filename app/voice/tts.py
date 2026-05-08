"""
voice.tts — text-to-speech dispatcher.

`synthesize(text, language) -> bytes | None` reads the live `voice_mode`
runtime setting and routes to either the host-binary backend (Piper via
the bridge) or the cloud backend (Google Cloud Neural2 over REST). Every
failure path returns None so the caller can degrade to a text-only reply.

Output format is OGG/Opus when produced by the cloud backend (Google's
``audioEncoding`` set to ``OGG_OPUS``) and WAV when produced by Piper.
Both formats are accepted by signal-cli as plain attachments. Add a
post-process step (ffmpeg via bridge) if true Signal voice-note rendering
on iOS is needed — left out of MVP.
"""
from __future__ import annotations

import logging

from app.runtime_settings import get_voice_mode

logger = logging.getLogger(__name__)

# Set by whichever backend last produced output, so callers can name the
# file with the right suffix (`.opus` for Google, `.wav` for Piper).
TTS_OUTPUT_FORMAT: dict[str, str] = {"latest": ""}


def synthesize(text: str, *, language: str = "en") -> bytes | None:
    """Return audio bytes for the given text, or None on failure / off mode.

    Args:
        text: text to speak. Truncated to a sane upper bound to avoid runaway
            cloud bills if a paragraph slips through.
        language: ISO 639-1 code. Used by both backends to pick a voice.

    Returns:
        Audio bytes (Opus or WAV) on success. None when voice mode is off
        or every backend failed.
    """
    if not text:
        return None

    # Cap to a reasonable upper bound — Google charges per character, Piper
    # blocks the bridge thread on long strings. 3000 chars ≈ ~25 seconds.
    text = text[:3000]

    mode = get_voice_mode()
    if mode == "off":
        return None

    primary, secondary, primary_fmt, secondary_fmt = _resolve_chain(mode)
    audio = primary(text, language=language)
    if audio:
        TTS_OUTPUT_FORMAT["latest"] = primary_fmt
        return audio
    if secondary is not None:
        audio = secondary(text, language=language)
        if audio:
            logger.info("voice.tts: primary backend empty, secondary succeeded")
            TTS_OUTPUT_FORMAT["latest"] = secondary_fmt
            return audio
    return None


def _resolve_chain(mode: str):
    """Return (primary, secondary, primary_fmt, secondary_fmt)."""
    from app.voice.local import synthesize as local_synthesize
    from app.voice.cloud import synthesize as cloud_synthesize

    if mode == "local":
        return local_synthesize, cloud_synthesize, "wav", "opus"
    if mode == "cloud":
        return cloud_synthesize, local_synthesize, "opus", "wav"
    return (lambda *a, **kw: None), None, "", ""
