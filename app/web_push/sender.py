"""
web_push.sender — VAPID-signed delivery via pywebpush.

`send_to_all(message, url)` fans out to every registered subscription and
returns the count of successful deliveries. 410-Gone responses (the
browser uninstalled the PWA) auto-prune the dead endpoint from the store.

When VAPID keys aren't configured the helpers are no-ops — the function
still returns 0 so the caller can wire it into the notification pipeline
without conditional checks.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.config import get_settings
from app.web_push.subscriptions import list_subscriptions, prune_subscription

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    """True if VAPID public + private keys are both present."""
    s = get_settings()
    pub = getattr(s, "vapid_public_key", None)
    priv = getattr(s, "vapid_private_key", None)
    if pub and hasattr(pub, "get_secret_value"):
        pub = pub.get_secret_value()
    if priv and hasattr(priv, "get_secret_value"):
        priv = priv.get_secret_value()
    return bool(pub and priv)


def _vapid_claims() -> dict[str, str]:
    s = get_settings()
    contact = getattr(s, "vapid_contact_email", "") or "andrus@raudsalu.com"
    return {"sub": f"mailto:{contact}"}


def _vapid_private_key() -> str:
    s = get_settings()
    pk = getattr(s, "vapid_private_key", None)
    if pk and hasattr(pk, "get_secret_value"):
        pk = pk.get_secret_value()
    return pk or ""


def send_to_one(subscription: dict[str, Any], title: str, body: str = "",
                url: str = "/cp/", tag: str = "andrusai") -> bool:
    """Send a single push. Returns True on 2xx."""
    if not is_configured():
        return False
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        logger.debug("pywebpush not installed — web push disabled")
        return False

    payload = json.dumps({"title": title, "body": body, "url": url, "tag": tag})
    try:
        webpush(
            subscription_info={
                "endpoint": subscription["endpoint"],
                "keys": subscription["keys"],
            },
            data=payload,
            vapid_private_key=_vapid_private_key(),
            vapid_claims=_vapid_claims(),
        )
        return True
    except WebPushException as exc:
        # 404 / 410 → the subscription is dead; prune it.
        status = getattr(exc.response, "status_code", 0) if exc.response is not None else 0
        if status in (404, 410):
            prune_subscription(subscription["endpoint"])
            logger.info(f"web_push: pruned dead subscription (status {status})")
        else:
            logger.warning(f"web_push: send failed (status {status}): {exc}")
        return False
    except Exception as exc:
        logger.warning(f"web_push: unexpected send failure: {exc}")
        return False


def send_to_all(title: str, body: str = "", url: str = "/cp/",
                tag: str = "andrusai") -> int:
    """Fan out to every registered subscription. Returns count delivered."""
    if not is_configured():
        return 0
    subs = list_subscriptions()
    if not subs:
        return 0
    delivered = 0
    for sub in subs:
        if send_to_one(sub, title=title, body=body, url=url, tag=tag):
            delivered += 1
    if delivered:
        logger.info(f"web_push: delivered to {delivered}/{len(subs)} devices")
    return delivered
