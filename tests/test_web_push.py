"""
test_web_push — JSON store + sender configuration tests.

No real Web Push servers are contacted. ``send_to_one`` is exercised only
indirectly via the configured/not-configured branches.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Redirect the JSON store to a tmp file."""
    from app.web_push import subscriptions as subs
    store = tmp_path / "subs.json"
    monkeypatch.setattr(subs, "_STORE_PATH", store)
    return store


def test_add_subscription_persists(isolated_store):
    from app.web_push import add_subscription, list_subscriptions
    ok = add_subscription({
        "endpoint": "https://fcm.googleapis.com/fcm/send/abc",
        "keys": {"p256dh": "p_abc", "auth": "a_abc"},
        "userAgent": "Mozilla/5.0 iPhone",
    })
    assert ok is True
    rows = list_subscriptions()
    assert len(rows) == 1
    assert rows[0]["endpoint"].endswith("/abc")
    assert rows[0]["user_agent"].startswith("Mozilla")


def test_add_subscription_rejects_missing_keys(isolated_store):
    from app.web_push import add_subscription, list_subscriptions
    assert add_subscription({"endpoint": "x"}) is False
    assert add_subscription({"endpoint": "x", "keys": {}}) is False
    assert list_subscriptions() == []


def test_add_subscription_overwrites_same_endpoint(isolated_store):
    from app.web_push import add_subscription, list_subscriptions
    payload = {
        "endpoint": "https://example.com/push/X",
        "keys": {"p256dh": "p", "auth": "a"},
        "userAgent": "v1",
    }
    add_subscription(payload)
    payload["userAgent"] = "v2"
    add_subscription(payload)
    rows = list_subscriptions()
    assert len(rows) == 1
    assert rows[0]["user_agent"] == "v2"


def test_remove_subscription(isolated_store):
    from app.web_push import add_subscription, remove_subscription, list_subscriptions
    add_subscription({
        "endpoint": "https://example.com/push/Z",
        "keys": {"p256dh": "p", "auth": "a"},
    })
    assert remove_subscription("https://example.com/push/Z") is True
    assert remove_subscription("https://example.com/push/Z") is False
    assert list_subscriptions() == []


def test_send_to_all_noop_without_vapid(isolated_store, monkeypatch):
    """When VAPID keys aren't set, send_to_all returns 0 without erroring."""
    from app.web_push import add_subscription, send_to_all
    add_subscription({
        "endpoint": "https://example.com/push/Q",
        "keys": {"p256dh": "p", "auth": "a"},
    })
    # Force the configured check to false
    from app.web_push import sender
    monkeypatch.setattr(sender, "is_configured", lambda: False)
    assert send_to_all("test", "body") == 0


def test_endpoint_host_extraction():
    from app.api.config_api import _endpoint_host
    assert _endpoint_host("https://fcm.googleapis.com/fcm/send/abc123") == "fcm.googleapis.com"
    assert _endpoint_host("https://updates.push.services.mozilla.com/wpush/v2/X") == "updates.push.services.mozilla.com"
    assert _endpoint_host("") == ""
    assert _endpoint_host("not a url") == ""
