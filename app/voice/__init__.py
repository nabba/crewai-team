"""
app.voice — speech-to-text + text-to-speech with switchable local/cloud backends.

Public surface:

    from app.voice import transcribe, synthesize, AUDIO_MIME_PREFIXES

    text = transcribe(audio_bytes, audio_format="m4a", language="en")
    audio_bytes = synthesize("Hello there.", language="en")

Both functions read the live `voice_mode` from `app.runtime_settings` and
dispatch to either the host-binary backend (whisper.cpp + Piper via the
bridge) or the cloud backend (Groq Whisper + Google Cloud Neural2). Both
return safely on failure — empty string for STT, None for TTS — so the
caller can degrade gracefully to plain text.

Audio detection helpers expose the prefixes Signal-cli uses on inbound
attachments so the gateway can decide whether to invoke STT.
"""
from __future__ import annotations

from app.voice.stt import transcribe, AUDIO_MIME_PREFIXES
from app.voice.tts import synthesize, TTS_OUTPUT_FORMAT
from app.voice.inbound_state import (
    mark_voice_inbound, is_voice_active, clear as clear_voice_state,
)

__all__ = [
    "transcribe", "synthesize",
    "AUDIO_MIME_PREFIXES", "TTS_OUTPUT_FORMAT",
    "mark_voice_inbound", "is_voice_active", "clear_voice_state",
]
