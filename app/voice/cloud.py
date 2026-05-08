"""
voice.cloud — cloud STT/TTS backend.

STT: Groq Whisper-large-v3 over their OpenAI-compatible REST endpoint.
TTS: Google Cloud Neural2 over the standard Text-to-Speech v1 REST API,
authenticated by API key.

Zero extra dependencies — everything goes through stdlib `urllib` and
`http.client` so the same module runs in Docker without adding wheels
to the image. Returns "" / None on failure so the dispatcher can fall
back to the local backend.

API references (current as of May 2026):
  - https://console.groq.com/docs/speech-text
  - https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import uuid

from app.config import get_groq_api_key, get_google_cloud_tts_key

logger = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_GROQ_MODEL = "whisper-large-v3"

_GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

# Neural2 voices per language. en-US has a male (J) and female (F) variant
# — pick whichever; override via env if desired (left to a follow-up).
_GOOGLE_VOICES = {
    "en": ("en-US", "en-US-Neural2-J"),
    "et": ("et-EE", "et-EE-Standard-A"),    # Neural2 not yet for et — fall back to Standard
    "fi": ("fi-FI", "fi-FI-Wavenet-A"),     # Wavenet ~= Neural2 quality for fi
}


# ── STT (Groq) ─────────────────────────────────────────────────────────────

def transcribe(
    audio_bytes: bytes,
    *,
    audio_format: str = "m4a",
    language: str | None = None,
) -> str:
    """STT via Groq Whisper. Returns "" on any failure."""
    key = get_groq_api_key()
    if not key:
        logger.debug("voice.cloud.transcribe: GROQ_API_KEY not set")
        return ""

    boundary = "----GroqVoice" + uuid.uuid4().hex
    body = _build_multipart(
        boundary=boundary,
        audio_bytes=audio_bytes,
        audio_format=audio_format,
        model=_GROQ_MODEL,
        language=language,
    )
    req = urllib.request.Request(
        _GROQ_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.info(f"voice.cloud.transcribe: Groq HTTP {exc.code}: {exc.read()[:200]}")
        return ""
    except Exception as exc:
        logger.info(f"voice.cloud.transcribe: Groq request failed: {exc}")
        return ""
    return (payload.get("text") or "").strip()


def _build_multipart(
    *,
    boundary: str,
    audio_bytes: bytes,
    audio_format: str,
    model: str,
    language: str | None,
) -> bytes:
    """Hand-rolled multipart/form-data body — keeps dep tree clean."""
    crlf = b"\r\n"
    fmt = audio_format.lstrip(".") or "m4a"
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(f"--{boundary}".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"'.encode()
        )
        parts.append(b"")
        parts.append(value.encode("utf-8"))

    add_field("model", model)
    if language:
        add_field("language", language)
    add_field("response_format", "json")

    parts.append(f"--{boundary}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="audio.{fmt}"'.encode()
    )
    parts.append(f"Content-Type: audio/{fmt}".encode())
    parts.append(b"")
    parts.append(audio_bytes)

    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    return crlf.join(parts)


# ── TTS (Google Cloud Neural2) ─────────────────────────────────────────────

def synthesize(text: str, *, language: str = "en") -> bytes | None:
    """TTS via Google Cloud Text-to-Speech. Returns Opus bytes or None."""
    key = get_google_cloud_tts_key()
    if not key:
        logger.debug("voice.cloud.synthesize: GOOGLE_CLOUD_TTS_KEY not set")
        return None

    lang = (language or "en").split("-")[0].lower()
    code, voice = _GOOGLE_VOICES.get(lang, _GOOGLE_VOICES["en"])

    body = json.dumps({
        "input": {"text": text},
        "voice": {"languageCode": code, "name": voice},
        "audioConfig": {
            "audioEncoding": "OGG_OPUS",
            "speakingRate": 1.0,
            "pitch": 0.0,
        },
    }).encode("utf-8")

    url = _GOOGLE_TTS_URL + "?" + urllib.parse.urlencode({"key": key})
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.info(f"voice.cloud.synthesize: Google HTTP {exc.code}: {exc.read()[:200]}")
        return None
    except Exception as exc:
        logger.info(f"voice.cloud.synthesize: Google request failed: {exc}")
        return None

    audio_b64 = payload.get("audioContent")
    if not audio_b64:
        return None
    try:
        return base64.b64decode(audio_b64)
    except Exception:
        return None
