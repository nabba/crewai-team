"""Image → vision-extracted markdown note inbox handler.

PROGRAM §46.7 (Q9.4). Anthropic Haiku 4.5 vision reads the image
and emits a structured markdown note at
``workspace/notes/<stem>.vision.md``. The model prompt is conservative:

  - Bullet-extract text + structure (no creative writing)
  - Note what kind of artefact it appears to be (whiteboard /
    handwritten / screenshot / photo / diagram)
  - Refuse if the image contains nothing legible
"""
from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB — Claude vision cap
_MODEL = "claude-haiku-4-5-20251001"
_MAX_OUTPUT_TOKENS = 1500

_SYSTEM = (
    "You are an image-to-notes extractor. You read photos of "
    "whiteboards, handwritten notes, screenshots, diagrams, and "
    "documents and produce a clean structured markdown note. "
    "Rules:\n"
    "  - Bullet-extract what is written / drawn; do not invent "
    "content not visible in the image.\n"
    "  - Open with one line stating the artefact type "
    "(whiteboard / handwritten / screenshot / photo / diagram / "
    "document).\n"
    "  - Preserve structure (headers / bullets / arrows / boxes) "
    "by indentation.\n"
    "  - If the image is illegible or empty, output exactly: "
    "ILLEGIBLE.\n"
    "  - No commentary about the image's aesthetics, no first-"
    "person voice, no preamble like 'I see...'."
)


def run(path: Path) -> str:
    size = path.stat().st_size if path.exists() else 0
    if size > _MAX_FILE_BYTES:
        raise RuntimeError(
            f"image {size / 1024 / 1024:.1f} MB exceeds "
            f"{_MAX_FILE_BYTES / 1024 / 1024:.0f} MB cap"
        )

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            f"anthropic SDK unavailable: {exc}"
        ) from exc

    try:
        with open(path, "rb") as f:
            blob = f.read()
        encoded = base64.standard_b64encode(blob).decode("ascii")
    except OSError as exc:
        raise RuntimeError(f"image read failed: {exc}") from exc

    media_type = _media_type(path)
    client = anthropic.Anthropic()
    try:
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract the content as a structured markdown note.",
                    },
                ],
            }],
        )
    except Exception as exc:
        raise RuntimeError(f"vision call failed: {exc}") from exc

    text_parts = [
        getattr(b, "text", "")
        for b in (msg.content or [])
        if getattr(b, "type", "") == "text"
    ]
    body = "".join(text_parts).strip()
    if not body:
        raise RuntimeError("vision returned empty output")
    if body == "ILLEGIBLE":
        raise RuntimeError("vision marked image ILLEGIBLE")

    dest = _notes_dir() / f"{path.stem}.vision.md"
    if dest.exists():
        stem = dest.stem
        i = 1
        while True:
            cand = _notes_dir() / f"{stem}.{i}.md"
            if not cand.exists():
                dest = cand
                break
            i += 1
    dest.write_text(_render(path, body), encoding="utf-8")
    return f"vision → {dest.name} ({len(body)} chars)"


def _media_type(path: Path) -> str:
    suf = path.suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "heic": "image/heic",
        "gif": "image/gif",
    }.get(suf, "image/png")


def _notes_dir() -> Path:
    from app.paths import WORKSPACE_ROOT
    d = Path(os.getenv("INBOX_NOTES_DIR", str(WORKSPACE_ROOT / "notes")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _render(src: Path, body: str) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    return (
        f"# Vision extract: {src.name}\n\n"
        f"_Source image dropped at {ts}; extracted by Claude Haiku 4.5 vision._\n\n"
        f"---\n\n"
        f"{body}\n"
    )
