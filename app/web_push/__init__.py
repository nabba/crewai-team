"""
app.web_push — Web Push notifications for the React PWA.

Two pieces:

  - subscriptions.py — JSON-backed device registry under
    ``workspace/web_push_subscriptions.json``. One row per browser, identified
    by the endpoint URL it gives us.
  - sender.py — VAPID-signed push delivery via pywebpush. Returns the count of
    successful sends and prunes 410-Gone subscriptions automatically.

VAPID keys are generated once with:

    python -m app.web_push.bootstrap

The public key gets injected into the React build via a `/config/vapid_public_key`
endpoint so the SettingsPage can subscribe browsers without a rebuild.

When VAPID keys aren't configured, ``send_to_all()`` is a no-op (returns 0)
so callers can wire it into the notification pipeline without conditional
logic — Phase 4 ships the wiring; the operator opts-in by running bootstrap.
"""
from __future__ import annotations

from app.web_push.subscriptions import (
    add_subscription, remove_subscription, list_subscriptions, prune_subscription,
)
from app.web_push.sender import send_to_all, send_to_one, is_configured

__all__ = [
    "add_subscription", "remove_subscription", "list_subscriptions",
    "prune_subscription", "send_to_all", "send_to_one", "is_configured",
]
