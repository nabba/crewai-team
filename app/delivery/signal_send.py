"""
delivery.signal_send — direct Signal attachment delivery for non-tool callers.

Wraps the same validation + container→host path translation used by the
``signal_send_attachment`` agent tool, but exposes a plain function so the
``/cp/files`` REST endpoint and Discord-relay code can use it directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Caps mirrored from app/tools/signal_attachment.py — keep aligned.
_MAX_BODY_CHARS = 2000
_MAX_TOTAL_BYTES = 25 * 1024 * 1024
_MAX_ATTACHMENTS = 5


def send_via_signal(
    paths: Iterable[str | Path],
    body: str = "",
) -> tuple[bool, str]:
    """Deliver one or more files to the configured Signal owner.

    Returns ``(ok, detail)``. ``ok`` is False on any validation or
    transport failure; ``detail`` is a single-line human-readable status.
    """
    try:
        from app.config import get_settings
        from app.tools.signal_attachment import (
            _validate_attachments, _container_to_host,
        )
        from app.signal_client import send_message
    except Exception as exc:
        return False, f"signal stack unavailable: {exc}"

    settings = get_settings()
    recipient = (settings.signal_owner_number or "").strip()
    if not recipient:
        return False, "SIGNAL_OWNER_NUMBER not configured"
    if not (settings.workspace_host_path or "").strip():
        return False, "WORKSPACE_HOST_PATH not configured"

    str_paths = [str(p) for p in paths]
    if not str_paths:
        return False, "no attachments provided"
    if len(str_paths) > _MAX_ATTACHMENTS:
        return False, f"too many attachments ({len(str_paths)} > {_MAX_ATTACHMENTS})"

    valid_paths, validation_err = _validate_attachments(str_paths)
    if not valid_paths:
        return False, f"no valid attachments: {validation_err or 'unknown'}"

    body_text = (body or "").strip()
    if len(body_text) > _MAX_BODY_CHARS:
        body_text = body_text[: _MAX_BODY_CHARS] + "…"

    host_paths = _container_to_host(valid_paths, settings.workspace_host_path)
    try:
        send_message(recipient, body_text, attachments=host_paths)
    except Exception as exc:
        return False, f"signal-cli send failed: {type(exc).__name__}: {exc}"

    summary = f"Signal delivered: {len(valid_paths)} file(s)"
    if validation_err:
        summary += f" (note: {validation_err.splitlines()[0]})"
    return True, summary
