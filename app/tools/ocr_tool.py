"""
OCR tool — extract text from images using GLM-OCR via Ollama.

GLM-OCR is a 0.9B vision-language model optimized for document understanding:
tables, multi-column layouts, receipts, screenshots, handwriting.

The model runs locally on Ollama (Metal GPU) — no cloud API needed.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default OCR model — GLM-OCR is tiny (0.9B) and fast
OCR_MODEL = os.getenv("OCR_MODEL", "glm-ocr")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL_HOST", "http://host.docker.internal:11434")


def ocr_from_file(image_path: str, prompt: str = "Extract all text from this image.") -> str:
    """Extract text from an image file using GLM-OCR.

    Args:
        image_path: Path to image file (PNG, JPG, WEBP, etc.)
        prompt: Instruction for the OCR model (default: extract all text)

    Returns:
        Extracted text, or error message.
    """
    path = Path(image_path)
    if not path.exists():
        return f"Error: file not found: {image_path}"

    # Read and encode image
    try:
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception as e:
        return f"Error reading image: {e}"

    return _call_ocr(image_data, prompt)


def ocr_from_base64(image_b64: str, prompt: str = "Extract all text from this image.") -> str:
    """Extract text from a base64-encoded image."""
    return _call_ocr(image_b64, prompt)


def ocr_from_url(url: str, prompt: str = "Extract all text from this image.") -> str:
    """Download an image from URL and extract text."""
    import httpx
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        image_data = base64.b64encode(resp.content).decode("utf-8")
        return _call_ocr(image_data, prompt)
    except Exception as e:
        return f"Error fetching image from {url}: {e}"


def _call_ocr(image_b64: str, prompt: str) -> str:
    """Send image to GLM-OCR via Ollama API and return extracted text."""
    import httpx

    # Try host bridge Ollama first, then container-accessible Ollama
    urls_to_try = [OLLAMA_URL, "http://localhost:11434"]

    for base_url in urls_to_try:
        try:
            resp = httpx.post(
                f"{base_url.rstrip('/')}/api/generate",
                json={
                    "model": OCR_MODEL,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for accurate extraction
                        "num_predict": 4096,  # Allow long output for dense documents
                    },
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("response", "")
                if text:
                    logger.info(f"OCR extracted {len(text)} chars using {OCR_MODEL}")
                    return text
                return "OCR returned empty response."
            else:
                logger.debug(f"OCR failed on {base_url}: {resp.status_code}")
                continue
        except Exception as e:
            logger.debug(f"OCR failed on {base_url}: {e}")
            continue

    return f"OCR failed: {OCR_MODEL} not available. Pull it with: ollama pull {OCR_MODEL}"


def ocr_attachment(attachment: dict) -> str:
    """Extract text from a Signal attachment dict.

    The attachment dict has: contentType, filename, id, size.
    The actual file is at /app/workspace/attachments/{id} or similar.
    """
    filename = attachment.get("filename", "")
    content_type = attachment.get("contentType", "")

    # Only process image types
    if not content_type.startswith("image/"):
        return ""

    # Try common attachment paths
    att_id = attachment.get("id", "")
    for base in ["/app/workspace/attachments", "/tmp/signal-attachments", "/app/workspace"]:
        for name in [att_id, filename, f"{att_id}.jpg", f"{att_id}.png"]:
            path = Path(base) / name
            if path.exists():
                return ocr_from_file(str(path))

    return f"OCR: attachment file not found (id={att_id}, filename={filename})"


# ── CrewAI tool wrapper ──────────────────────────────────────────────────────

def create_ocr_tool():
    """Create a CrewAI-compatible OCR tool. Returns None if OCR model unavailable."""
    try:
        from crewai.tools import tool

        @tool("ocr_extract_text")
        def ocr_tool(image_path: str, prompt: str = "Extract all text from this image.") -> str:
            """Extract text from an image using OCR. Supports photos, screenshots, documents, receipts.
            Provide a file path to a local image. Optionally customize the extraction prompt."""
            return ocr_from_file(image_path, prompt)

        return ocr_tool
    except Exception:
        return None
