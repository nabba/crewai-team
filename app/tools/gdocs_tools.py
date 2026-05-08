"""
gdocs_tools.py — agent-callable Google Docs operations.

Three CrewAI tools:

    create_google_doc        new doc with optional body text
    read_google_doc          full plain-text body of an existing doc
    append_to_google_doc     append paragraphs to the end of a doc

Document URLs and bare doc IDs are both accepted by the read/append tools;
the helper extracts the canonical id from a URL of the form
``https://docs.google.com/document/d/<ID>/edit?...``.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_DOC_URL_RE = re.compile(r"docs\.google\.com/document/d/([A-Za-z0-9_-]{20,})")


def _docs():
    from app.google_workspace import get_service
    return get_service("docs")


def _drive():
    from app.google_workspace import get_service
    return get_service("drive")


def _doc_id(value: str) -> str:
    """Accept either a Doc URL or a raw id. Returns the id portion."""
    value = (value or "").strip()
    m = _DOC_URL_RE.search(value)
    return m.group(1) if m else value


def _create(title: str, body: str = "") -> dict[str, Any]:
    docs = _docs()
    if docs is None:
        return {"error": "Docs not configured"}
    doc = docs.documents().create(body={"title": title}).execute()
    doc_id = doc.get("documentId")
    if body:
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": [{
                "insertText": {"location": {"index": 1}, "text": body}
            }]},
        ).execute()
    return {
        "id": doc_id,
        "url": f"https://docs.google.com/document/d/{doc_id}/edit",
        "title": title,
    }


def _read(doc_id: str) -> dict[str, Any]:
    docs = _docs()
    if docs is None:
        return {"error": "Docs not configured"}
    real_id = _doc_id(doc_id)
    doc = docs.documents().get(documentId=real_id).execute()

    body_parts: list[str] = []
    for el in doc.get("body", {}).get("content", []) or []:
        para = el.get("paragraph")
        if not para:
            continue
        line: list[str] = []
        for run in para.get("elements", []) or []:
            text_run = run.get("textRun", {})
            text = text_run.get("content", "")
            if text:
                line.append(text)
        if line:
            body_parts.append("".join(line).rstrip("\n"))

    return {
        "id": real_id,
        "title": doc.get("title", ""),
        "url": f"https://docs.google.com/document/d/{real_id}/edit",
        "body": "\n".join(body_parts)[:30_000],
    }


def _append(doc_id: str, text: str) -> dict[str, Any]:
    docs = _docs()
    if docs is None:
        return {"error": "Docs not configured"}
    real_id = _doc_id(doc_id)
    # Find the end-of-body insertion index — Docs requires us to insert at
    # ``endIndex - 1`` (the trailing newline of the body element).
    doc = docs.documents().get(documentId=real_id, fields="body(content(endIndex))").execute()
    end_index = 1
    for el in doc.get("body", {}).get("content", []) or []:
        if "endIndex" in el:
            end_index = max(end_index, int(el["endIndex"]))
    insert_at = max(1, end_index - 1)
    payload = "\n" + text if not text.startswith("\n") else text
    docs.documents().batchUpdate(
        documentId=real_id,
        body={"requests": [{
            "insertText": {"location": {"index": insert_at}, "text": payload}
        }]},
    ).execute()
    return {"id": real_id, "appended_chars": len(payload)}


# ── CrewAI tool factory ────────────────────────────────────────────────────

def create_gdocs_tools(agent_id: str = "writer") -> list:
    try:
        from app.google_workspace import is_configured
        if not is_configured():
            return []
    except Exception:
        return []
    if _docs() is None:
        return []

    try:
        from crewai.tools import tool
    except ImportError:
        return []

    @tool("create_google_doc")
    def create_tool(title: str, body: str = "") -> str:
        """Create a new Google Doc. Returns JSON {id, url, title}."""
        import json
        return json.dumps(_create(title=title, body=body))

    @tool("read_google_doc")
    def read_tool(doc_id: str) -> str:
        """Read the plain-text body of a Google Doc by id or URL. Returns JSON."""
        import json
        return json.dumps(_read(doc_id), ensure_ascii=False)

    @tool("append_to_google_doc")
    def append_tool(doc_id: str, text: str) -> str:
        """Append ``text`` to the end of the Google Doc. Returns JSON."""
        import json
        return json.dumps(_append(doc_id, text))

    return [create_tool, read_tool, append_tool]
