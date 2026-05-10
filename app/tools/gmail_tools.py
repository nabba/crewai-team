"""
gmail_tools.py — agent-callable Gmail operations via the Gmail API.

Four CrewAI tools:

    list_recent_gmail   most recent N messages with from/subject/snippet
    read_gmail          full body + headers of a specific message id
    send_gmail          send a new email (text body, optional CC)
    label_gmail         apply or remove a label by id (or by name)

All run through the credentials loaded from the bootstrap-saved
refresh token; the factory returns ``[]`` if Gmail is not configured
or the API client is missing.

For Gmail-hosted accounts the agent should prefer this over
``email_tools.py`` (the IMAP/SMTP fallback for non-Gmail providers)
because OAuth credentials work even when the user has 2FA on without
needing an App Password.
"""
from __future__ import annotations

import base64
import logging
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)

# Gmail API helpers ────────────────────────────────────────────────────────


def _service():
    from app.google_workspace import get_service
    return get_service("gmail")


# Headers we ask Gmail to include with each message stub.  The bulk
# markers (List-Unsubscribe / List-ID / Auto-Submitted / Precedence)
# are critical for the email_importance scorer — without them the
# scorer is blind to the most reliable bulk-mail signal and surfaces
# marketing emails as "urgent unread".  Threading headers (In-Reply-
# To / References) let the scorer credit personal replies.  Adding
# headers to ``metadataHeaders`` is FREE — same API call, more
# fields come back.  See `app/tools/email_importance.py` for the
# weight each signal carries.
_GMAIL_METADATA_HEADERS: tuple[str, ...] = (
    "From", "To", "Cc", "Subject", "Date",
    "List-Unsubscribe", "List-Id",
    "Auto-Submitted", "Precedence",
    "In-Reply-To", "References",
)


def _list_recent(limit: int = 10, query: str = "in:inbox") -> list[dict]:
    svc = _service()
    if svc is None:
        return []
    msgs_resp = (
        svc.users()
        .messages()
        .list(userId="me", q=query, maxResults=max(1, min(50, limit)))
        .execute()
    )
    out: list[dict] = []
    for stub in msgs_resp.get("messages", []) or []:
        full = (
            svc.users()
            .messages()
            .get(userId="me", id=stub["id"], format="metadata",
                 metadataHeaders=list(_GMAIL_METADATA_HEADERS))
            .execute()
        )
        headers = {h["name"].lower(): h["value"] for h in full.get("payload", {}).get("headers", [])}
        out.append({
            "id": full.get("id"),
            "thread_id": full.get("threadId"),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "cc": headers.get("cc", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "snippet": full.get("snippet", ""),
            "label_ids": full.get("labelIds", []),
            # Bulk markers — surfaced verbatim so the scorer can apply
            # weights without needing to re-fetch.  None when absent.
            "list_unsubscribe": headers.get("list-unsubscribe"),
            "list_id": headers.get("list-id"),
            "auto_submitted": headers.get("auto-submitted"),
            "precedence": headers.get("precedence"),
            "in_reply_to": headers.get("in-reply-to"),
            "references": headers.get("references"),
        })
    return out


def _read_one(msg_id: str) -> dict[str, Any]:
    svc = _service()
    if svc is None:
        return {"error": "Gmail not configured"}
    msg = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}

    body_text = _extract_body(msg.get("payload", {}))
    return {
        "id": msg.get("id"),
        "thread_id": msg.get("threadId"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "body": body_text[:30_000],
        "label_ids": msg.get("labelIds", []),
    }


def _extract_body(part: dict[str, Any]) -> str:
    """Recursively walk MIME parts to find text/plain (preferred) or text/html."""
    if not part:
        return ""
    mime = part.get("mimeType", "")
    if mime == "text/plain":
        data = part.get("body", {}).get("data", "")
        return _b64d(data)
    if mime.startswith("multipart/"):
        # Prefer text/plain part if present, else fall back to text/html.
        for sub in part.get("parts", []) or []:
            if sub.get("mimeType") == "text/plain":
                return _extract_body(sub)
        for sub in part.get("parts", []) or []:
            if sub.get("mimeType") == "text/html":
                return _extract_body(sub)
        # Generic recursion for nested multiparts
        for sub in part.get("parts", []) or []:
            t = _extract_body(sub)
            if t:
                return t
    if mime == "text/html":
        data = part.get("body", {}).get("data", "")
        return _b64d(data)
    return ""


def _b64d(b64url: str) -> str:
    if not b64url:
        return ""
    try:
        return base64.urlsafe_b64decode(b64url.encode("ascii")).decode(
            "utf-8", errors="replace"
        )
    except Exception:
        return ""


def _send(to: str, subject: str, body: str, cc: str = "") -> dict[str, Any]:
    svc = _service()
    if svc is None:
        return {"error": "Gmail not configured"}
    mime = MIMEText(body, _charset="utf-8")
    mime["To"] = to
    mime["Subject"] = subject
    if cc:
        mime["Cc"] = cc
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
    sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"id": sent.get("id"), "thread_id": sent.get("threadId"), "status": "sent"}


def _resolve_label_id(name_or_id: str) -> str | None:
    """Accept either a literal label id (e.g. 'Label_42') or a human name."""
    svc = _service()
    if svc is None:
        return None
    if name_or_id.startswith("Label_") or name_or_id in {
        "INBOX", "STARRED", "IMPORTANT", "UNREAD", "TRASH", "SPAM",
    }:
        return name_or_id
    labels = svc.users().labels().list(userId="me").execute().get("labels", [])
    for lbl in labels:
        if lbl.get("name", "").lower() == name_or_id.lower():
            return lbl.get("id")
    return None


def _modify_labels(msg_id: str, add: list[str] | None, remove: list[str] | None) -> dict:
    svc = _service()
    if svc is None:
        return {"error": "Gmail not configured"}
    body: dict[str, list[str]] = {}
    if add:
        ids = [_resolve_label_id(n) for n in add]
        body["addLabelIds"] = [i for i in ids if i]
    if remove:
        ids = [_resolve_label_id(n) for n in remove]
        body["removeLabelIds"] = [i for i in ids if i]
    if not body:
        return {"error": "no labels to add or remove"}
    res = svc.users().messages().modify(userId="me", id=msg_id, body=body).execute()
    return {"id": res.get("id"), "label_ids": res.get("labelIds", [])}


# ── CrewAI tool factory ────────────────────────────────────────────────────

def create_gmail_tools(agent_id: str = "pim") -> list:
    """Build CrewAI BaseTool instances for Gmail. Returns [] if not configured."""
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

    @tool("list_recent_gmail")
    def list_recent_tool(limit: int = 10, query: str = "in:inbox") -> str:
        """List the most recent Gmail messages. Returns JSON.

        Args:
            limit: 1–50 messages.
            query: Gmail search expression (e.g. "is:unread", "from:alice@example.com",
                "subject:invoice newer_than:7d"). Default "in:inbox".
        """
        import json
        return json.dumps(_list_recent(limit=limit, query=query), ensure_ascii=False)

    @tool("read_gmail")
    def read_tool(message_id: str) -> str:
        """Read full body + headers of a Gmail message by id. Returns JSON."""
        import json
        return json.dumps(_read_one(message_id), ensure_ascii=False)

    @tool("send_gmail")
    def send_tool(to: str, subject: str, body: str, cc: str = "") -> str:
        """Send a Gmail message. Returns JSON {id, thread_id, status}."""
        import json
        return json.dumps(_send(to=to, subject=subject, body=body, cc=cc))

    @tool("label_gmail")
    def label_tool(message_id: str, add: str = "", remove: str = "") -> str:
        """Apply or remove Gmail labels by name or id (comma-separated).

        System labels: INBOX, STARRED, IMPORTANT, UNREAD, TRASH, SPAM.
        User labels match by name (case-insensitive). Returns JSON.
        """
        import json
        add_list = [s.strip() for s in add.split(",") if s.strip()]
        remove_list = [s.strip() for s in remove.split(",") if s.strip()]
        return json.dumps(_modify_labels(message_id, add_list or None, remove_list or None))

    return [list_recent_tool, read_tool, send_tool, label_tool]
