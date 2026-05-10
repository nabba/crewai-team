"""Email-draft action handler.

The agent proposes an email; the operator approves; this handler
sends it via the existing :mod:`app.delivery.email_send`.

Data payload shape::

    {
        "to":          "x@example.com" | ["a@x", "b@x"],   # one or many
        "subject":     "...",
        "body":        "...",
        "attachments": ["/path/1", "/path/2"],   # optional
        "html":        false,                    # optional, default False
    }
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.action_requests.handlers.base import ActionHandler, ApplyResult
from app.action_requests.models import ActionType

logger = logging.getLogger(__name__)


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_MAX_BODY_CHARS = 100_000
_MAX_SUBJECT_CHARS = 998   # RFC 5322 hard limit


def _coerce_recipients(to: Any) -> list[str]:
    if isinstance(to, str):
        return [to.strip()] if to.strip() else []
    if isinstance(to, list):
        return [str(t).strip() for t in to if str(t).strip()]
    return []


class EmailDraftHandler(ActionHandler):
    @property
    def action_type(self):
        return ActionType.EMAIL_DRAFT

    def validate(self, data: dict[str, Any]) -> tuple[bool, str | None]:
        recipients = _coerce_recipients(data.get("to"))
        if not recipients:
            return False, "to is required and must be a non-empty string or list"
        for r in recipients:
            if not _EMAIL_RE.match(r):
                return False, f"invalid email address: {r!r}"
        subject = data.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            return False, "subject is required"
        if len(subject) > _MAX_SUBJECT_CHARS:
            return False, f"subject exceeds {_MAX_SUBJECT_CHARS} chars"
        body = data.get("body")
        if not isinstance(body, str) or not body.strip():
            return False, "body is required"
        if len(body) > _MAX_BODY_CHARS:
            return False, f"body exceeds {_MAX_BODY_CHARS} chars"
        attachments = data.get("attachments", [])
        if attachments is not None and not isinstance(attachments, list):
            return False, "attachments must be a list of paths or omitted"
        html = data.get("html", False)
        if not isinstance(html, bool):
            return False, "html must be a boolean"
        return True, None

    def apply(self, data: dict[str, Any]) -> ApplyResult:
        try:
            from app.delivery.email_send import send_via_email
        except Exception as exc:  # noqa: BLE001
            return ApplyResult(
                ok=False,
                error=f"email_send import failed: {exc}",
            )
        recipients = _coerce_recipients(data.get("to"))
        try:
            result = send_via_email(
                to=recipients[0] if len(recipients) == 1 else recipients,
                subject=data["subject"],
                body=data["body"],
                attachment_paths=data.get("attachments") or [],
                html=bool(data.get("html", False)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("email_draft: send_via_email raised: %s", exc, exc_info=True)
            return ApplyResult(ok=False, error=f"send raised: {exc}")
        # send_via_email may return a dict, a bool, or None depending on
        # the implementation. Normalise.
        if isinstance(result, dict):
            ok = bool(result.get("ok", True))
            return ApplyResult(
                ok=ok,
                artifact={k: v for k, v in result.items() if k != "ok"},
                error=str(result.get("error", "")) if not ok else "",
            )
        if isinstance(result, bool):
            return ApplyResult(ok=result, error="" if result else "send returned False")
        # None or other → assume success (caller didn't surface an error).
        return ApplyResult(ok=True, artifact={"recipients": recipients})

    def render_summary(self, data: dict[str, Any]) -> str:
        recipients = _coerce_recipients(data.get("to"))
        n_recipients = len(recipients)
        first = recipients[0] if recipients else "(no recipient)"
        more = f" (+{n_recipients - 1} more)" if n_recipients > 1 else ""
        subject = (data.get("subject") or "")[:80]
        return f"📧 email to {first}{more} — “{subject}”"
