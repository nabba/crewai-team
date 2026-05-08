"""
gslides_tools.py — agent-callable Google Slides operations.

Three CrewAI tools:

    create_google_slides_deck   new deck with a title slide
    add_google_slide            append a content slide (title + body bullets)
    set_google_slide_text       overwrite the text of a specific slide

Closes the slide-editing gap that python-pptx alone can't fill: this path
edits decks **inside** Google Drive directly, so the user can open the
result on docs.google.com immediately.

Both URLs and bare presentation ids are accepted; the helper extracts the
canonical id from URLs of the form
``https://docs.google.com/presentation/d/<ID>/edit``.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_DECK_URL_RE = re.compile(r"docs\.google\.com/presentation/d/([A-Za-z0-9_-]{20,})")


def _service():
    from app.google_workspace import get_service
    return get_service("slides")


def _deck_id(value: str) -> str:
    value = (value or "").strip()
    m = _DECK_URL_RE.search(value)
    return m.group(1) if m else value


def _create(title: str, subtitle: str = "") -> dict[str, Any]:
    svc = _service()
    if svc is None:
        return {"error": "Slides not configured"}
    deck = svc.presentations().create(body={"title": title}).execute()
    deck_id = deck.get("presentationId")

    # The auto-created title slide has TITLE + SUBTITLE placeholders.
    # Insert text into them so the deck opens with the requested title.
    requests = []
    title_slide = (deck.get("slides") or [{}])[0]
    for ph in title_slide.get("pageElements", []) or []:
        ph_type = (ph.get("shape") or {}).get("placeholder", {}).get("type", "")
        oid = ph.get("objectId")
        if not oid:
            continue
        if ph_type == "CENTERED_TITLE" or ph_type == "TITLE":
            requests.append({"insertText": {"objectId": oid, "text": title}})
        elif ph_type == "SUBTITLE" and subtitle:
            requests.append({"insertText": {"objectId": oid, "text": subtitle}})
    if requests:
        svc.presentations().batchUpdate(
            presentationId=deck_id, body={"requests": requests},
        ).execute()

    return {
        "id": deck_id,
        "url": f"https://docs.google.com/presentation/d/{deck_id}/edit",
        "title": title,
    }


def _add_slide(deck_id: str, title: str, body: str = "") -> dict[str, Any]:
    svc = _service()
    if svc is None:
        return {"error": "Slides not configured"}
    real_id = _deck_id(deck_id)
    slide_id = f"slide_{uuid.uuid4().hex[:12]}"

    create_request = {
        "createSlide": {
            "objectId": slide_id,
            "slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"},
        }
    }
    resp = svc.presentations().batchUpdate(
        presentationId=real_id, body={"requests": [create_request]},
    ).execute()

    # Fetch the new slide so we can locate the title + body placeholder ids.
    deck = svc.presentations().get(
        presentationId=real_id,
        fields="slides(objectId,pageElements(objectId,shape(placeholder(type))))",
    ).execute()
    title_oid = body_oid = ""
    for slide in deck.get("slides", []) or []:
        if slide.get("objectId") != slide_id:
            continue
        for el in slide.get("pageElements", []) or []:
            ptype = (el.get("shape") or {}).get("placeholder", {}).get("type", "")
            oid = el.get("objectId")
            if ptype == "TITLE":
                title_oid = oid
            elif ptype == "BODY":
                body_oid = oid

    text_requests: list[dict] = []
    if title_oid and title:
        text_requests.append({"insertText": {"objectId": title_oid, "text": title}})
    if body_oid and body:
        text_requests.append({"insertText": {"objectId": body_oid, "text": body}})
    if text_requests:
        svc.presentations().batchUpdate(
            presentationId=real_id, body={"requests": text_requests},
        ).execute()

    return {
        "id": real_id,
        "slide_id": slide_id,
        "url": f"https://docs.google.com/presentation/d/{real_id}/edit#slide=id.{slide_id}",
    }


def _set_text(deck_id: str, slide_id: str, title: str = "", body: str = "") -> dict[str, Any]:
    """Overwrite TITLE + BODY placeholder text on a specific slide."""
    svc = _service()
    if svc is None:
        return {"error": "Slides not configured"}
    real_id = _deck_id(deck_id)

    deck = svc.presentations().get(
        presentationId=real_id,
        fields="slides(objectId,pageElements(objectId,shape(placeholder(type))))",
    ).execute()
    title_oid = body_oid = ""
    for slide in deck.get("slides", []) or []:
        if slide.get("objectId") != slide_id:
            continue
        for el in slide.get("pageElements", []) or []:
            ptype = (el.get("shape") or {}).get("placeholder", {}).get("type", "")
            oid = el.get("objectId")
            if ptype == "TITLE":
                title_oid = oid
            elif ptype == "BODY":
                body_oid = oid

    if not (title_oid or body_oid):
        return {"error": f"slide {slide_id!r} not found in deck {real_id!r}"}

    requests: list[dict] = []
    if title_oid:
        requests += [
            {"deleteText": {"objectId": title_oid, "textRange": {"type": "ALL"}}},
            {"insertText": {"objectId": title_oid, "text": title or ""}},
        ]
    if body_oid:
        requests += [
            {"deleteText": {"objectId": body_oid, "textRange": {"type": "ALL"}}},
            {"insertText": {"objectId": body_oid, "text": body or ""}},
        ]

    svc.presentations().batchUpdate(
        presentationId=real_id, body={"requests": requests},
    ).execute()
    return {"id": real_id, "slide_id": slide_id, "updated": True}


# ── CrewAI tool factory ────────────────────────────────────────────────────

def create_gslides_tools(agent_id: str = "writer") -> list:
    try:
        from app.google_workspace import is_configured
        if not is_configured():
            return []
    except Exception:
        return []
    if _service() is None:
        return []

    try:
        from crewai.tools import tool
    except ImportError:
        return []

    @tool("create_google_slides_deck")
    def create_tool(title: str, subtitle: str = "") -> str:
        """Create a new Google Slides deck. Returns JSON {id, url, title}."""
        return json.dumps(_create(title=title, subtitle=subtitle))

    @tool("add_google_slide")
    def add_tool(deck_id: str, title: str, body: str = "") -> str:
        """Append a content slide (TITLE_AND_BODY layout) to a Slides deck.

        Returns JSON {id, slide_id, url}. ``body`` is plain text — line breaks
        become bullet boundaries when Slides applies the BODY layout.
        """
        return json.dumps(_add_slide(deck_id=deck_id, title=title, body=body))

    @tool("set_google_slide_text")
    def set_text_tool(deck_id: str, slide_id: str, title: str = "", body: str = "") -> str:
        """Overwrite the title and/or body placeholder text on an existing slide."""
        return json.dumps(_set_text(deck_id, slide_id, title=title, body=body))

    return [create_tool, add_tool, set_text_tool]
