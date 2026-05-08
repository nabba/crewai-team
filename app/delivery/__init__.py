"""
app.delivery — direct (non-tool) helpers for sending generated artifacts.

Three entry points:

  - ``send_via_signal(path, body)`` — drop a file into the owner's Signal
    chat with a one-line description. Uses signal-cli through the
    existing host bridge.

  - ``send_via_email(to, subject, body, attachment_paths)`` — SMTP send
    with attachments. Uses the existing IMAP/SMTP config from .env.

  - ``send_via_discord(user_id, body, attachment_paths)`` — DM the
    configured Discord owner with optional file attachments.

These are the surfaces used by the React /cp/files "Send via …" buttons,
by the Discord/Signal forwarders, and by the agent-callable tools that
already wrap them.

Each function returns ``(ok: bool, detail: str)`` so callers (REST
handlers, tool wrappers) can surface a clear status without parsing
exceptions.
"""
from __future__ import annotations

from app.delivery.signal_send import send_via_signal
from app.delivery.email_send import send_via_email
# Discord lives under app.discord_client; re-export here for symmetry
# so /cp/files and the gateway reply path can import from one place.
from app.discord_client.sender import send_via_discord

__all__ = ["send_via_signal", "send_via_email", "send_via_discord"]
