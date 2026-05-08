"""
delivery.email_send — SMTP send with attachments.

The existing ``email_tools.py`` wraps SMTP behind a CrewAI tool factory
that doesn't expose a plain function. This module is the direct surface
for the React /cp/files page and the agent tool wrappers.

Uses the same SMTP config as ``email_tools.py`` (EMAIL_* env vars).
"""
from __future__ import annotations

import logging
import mimetypes
import smtplib
import email.utils
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_MAX_TOTAL_BYTES = 25 * 1024 * 1024  # match Gmail's typical attachment cap


def send_via_email(
    to: str,
    subject: str,
    body: str,
    attachment_paths: Iterable[str | Path] = (),
    *,
    html: bool = False,
) -> tuple[bool, str]:
    """Send an email with optional attachments.

    Returns ``(ok, detail)``. Reads SMTP config from settings; falls back
    to ``(False, ...)`` when ``email_enabled`` is off or required fields
    are missing.
    """
    try:
        from app.config import get_settings
    except Exception as exc:
        return False, f"settings load failed: {exc}"
    s = get_settings()
    if not s.email_enabled:
        return False, "EMAIL_ENABLED is false"
    if not (s.email_smtp_host and s.email_address):
        return False, "SMTP host or address missing"

    to = (to or "").strip()
    if "@" not in to:
        return False, f"invalid recipient {to!r}"

    msg = MIMEMultipart("mixed")
    msg["From"] = s.email_address
    msg["To"] = to
    msg["Subject"] = subject or "(no subject)"
    msg["Date"] = email.utils.formatdate(localtime=True)

    body_part = MIMEMultipart("alternative")
    content_type = "html" if html else "plain"
    body_part.attach(MIMEText(body or "", content_type, "utf-8"))
    msg.attach(body_part)

    paths = [Path(p) for p in attachment_paths]
    if paths:
        ok, err = _attach_files(msg, paths)
        if not ok:
            return False, err

    try:
        password = s.email_password.get_secret_value() if s.email_password else ""
        with smtplib.SMTP(s.email_smtp_host, s.email_smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            if password:
                server.login(s.email_address, password)
            server.send_message(msg)
    except Exception as exc:
        return False, f"SMTP send failed: {type(exc).__name__}: {exc}"

    detail = f"sent to {to}"
    if paths:
        detail += f" with {len(paths)} attachment(s)"
    return True, detail


def _attach_files(msg: MIMEMultipart, paths: list[Path]) -> tuple[bool, str]:
    """Attach files. Aborts at the first invalid path or size cap."""
    total_bytes = 0
    for p in paths:
        if not p.exists():
            return False, f"attachment not found: {p}"
        if not p.is_file():
            return False, f"attachment is not a file: {p}"
        size = p.stat().st_size
        if size <= 0:
            return False, f"attachment is empty: {p.name}"
        total_bytes += size
        if total_bytes > _MAX_TOTAL_BYTES:
            return False, f"attachments exceed {_MAX_TOTAL_BYTES // (1024 * 1024)} MB total"

        ctype, encoding = mimetypes.guess_type(p.name)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        try:
            with p.open("rb") as fp:
                part = MIMEApplication(fp.read(), _subtype=subtype)
            part.add_header("Content-Disposition", "attachment", filename=p.name)
            part.add_header("Content-Type", ctype, name=p.name)
            msg.attach(part)
        except OSError as exc:
            return False, f"failed to read {p.name}: {exc}"
    return True, ""
