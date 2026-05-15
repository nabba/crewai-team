"""PDF → receipt-or-note inbox handler.

PROGRAM §46.7 (Q9.4). Anthropic Haiku 4.5 reads the PDF (sent as a
``document`` content block, which the SDK supports natively) and
emits either:

  1. **Receipt JSONL row** at ``workspace/finance/expenses.jsonl``
     when the model identifies vendor + amount + currency + date.
  2. **Markdown note** at ``workspace/notes/<stem>.pdf.md`` otherwise.

The model is asked for strict JSON; we parse it and route by the
``kind`` field. Failure-isolated: malformed JSON → treat as note;
empty content → raise so the file stays in inbox.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_MAX_FILE_BYTES = 32 * 1024 * 1024  # 32 MB — Anthropic doc cap
_MODEL = "claude-haiku-4-5-20251001"
_MAX_OUTPUT_TOKENS = 2000

_SYSTEM = (
    "You read a single PDF file and produce a strict JSON document. "
    "Decide the kind:\n"
    '  - kind="receipt" if the PDF is a receipt, invoice, or '
    "expense voucher.\n"
    '  - kind="document" otherwise.\n\n'
    "Schema for receipt:\n"
    '  {"kind":"receipt","vendor":"...","amount":<number>,'
    '"currency":"EUR|USD|...","date":"YYYY-MM-DD",'
    '"category":"food|transport|software|hotel|other",'
    '"summary":"one-line description"}\n\n'
    "Schema for document:\n"
    '  {"kind":"document","title":"...","summary":"...",'
    '"key_points":["...","..."]}\n\n'
    "Output ONLY the JSON object. No markdown fences, no preamble. "
    "If the PDF is empty/unreadable: output {\"kind\":\"document\","
    "\"title\":\"UNREADABLE\",\"summary\":\"\",\"key_points\":[]}."
)


def run(path: Path) -> str:
    size = path.stat().st_size if path.exists() else 0
    if size > _MAX_FILE_BYTES:
        raise RuntimeError(
            f"PDF {size / 1024 / 1024:.1f} MB exceeds "
            f"{_MAX_FILE_BYTES / 1024 / 1024:.0f} MB cap"
        )

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(f"anthropic SDK unavailable: {exc}") from exc

    try:
        with open(path, "rb") as f:
            blob = f.read()
        encoded = base64.standard_b64encode(blob).decode("ascii")
    except OSError as exc:
        raise RuntimeError(f"PDF read failed: {exc}") from exc

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
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": encoded,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract per the system instructions.",
                    },
                ],
            }],
        )
    except Exception as exc:
        raise RuntimeError(f"PDF extract call failed: {exc}") from exc

    text_parts = [
        getattr(b, "text", "")
        for b in (msg.content or [])
        if getattr(b, "type", "") == "text"
    ]
    raw = "".join(text_parts).strip()
    if not raw:
        raise RuntimeError("PDF extract returned empty output")

    parsed = _parse_json(raw)
    if parsed is None:
        # Fall back to markdown note with the raw output as body
        return _write_note(path, raw)
    if parsed.get("title") == "UNREADABLE":
        raise RuntimeError("PDF marked UNREADABLE by extractor")

    if parsed.get("kind") == "receipt":
        return _write_receipt(path, parsed)
    return _write_note(path, _render_document(parsed))


# ─────────────────────────────────────────────────────────────────────
#   Writers
# ─────────────────────────────────────────────────────────────────────


def _write_receipt(path: Path, data: dict[str, Any]) -> str:
    """Append to workspace/finance/expenses.jsonl."""
    from app.paths import WORKSPACE_ROOT
    ledger = Path(
        os.getenv("INBOX_EXPENSE_LEDGER",
                  str(WORKSPACE_ROOT / "finance" / "expenses.jsonl"))
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "source_file": path.name,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "vendor": str(data.get("vendor", "") or "")[:200],
        "amount": _coerce_float(data.get("amount")),
        "currency": str(data.get("currency", "") or "")[:8],
        "date": str(data.get("date", "") or "")[:32],
        "category": str(data.get("category", "other") or "other")[:32],
        "summary": str(data.get("summary", "") or "")[:300],
    }
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return (
        f"receipt → expenses.jsonl: {row['vendor'][:32]} "
        f"{row['amount']} {row['currency']}"
    )


def _write_note(path: Path, body: str) -> str:
    from app.paths import WORKSPACE_ROOT
    notes_dir = Path(
        os.getenv("INBOX_NOTES_DIR", str(WORKSPACE_ROOT / "notes"))
    )
    notes_dir.mkdir(parents=True, exist_ok=True)
    dest = notes_dir / f"{path.stem}.pdf.md"
    if dest.exists():
        stem = dest.stem
        i = 1
        while True:
            cand = notes_dir / f"{stem}.{i}.md"
            if not cand.exists():
                dest = cand
                break
            i += 1
    dest.write_text(
        f"# PDF extract: {path.name}\n\n"
        f"_Extracted by Claude Haiku 4.5 at "
        f"{datetime.now(timezone.utc).isoformat()}._\n\n---\n\n"
        f"{body.strip()}\n",
        encoding="utf-8",
    )
    return f"pdf → {dest.name} ({len(body)} chars)"


def _render_document(data: dict[str, Any]) -> str:
    lines: list[str] = []
    title = data.get("title")
    if title:
        lines.append(f"## {title}")
    summary = data.get("summary")
    if summary:
        lines.append("")
        lines.append(summary)
    points = data.get("key_points") or []
    if points:
        lines.append("")
        lines.append("### Key points")
        for p in points:
            lines.append(f"- {p}")
    return "\n".join(lines)


def _parse_json(raw: str) -> dict[str, Any] | None:
    # Try strict first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to strip a markdown code fence
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find the first {...} substring
    brace = re.search(r"\{.*\}", raw, re.S)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _coerce_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
