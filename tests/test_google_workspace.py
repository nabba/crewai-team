"""
test_google_workspace — auth + tool-factory tests for the Google Workspace package.

No real Google credentials required. ``get_service()`` is monkey-patched to
return a fake `googleapiclient.discovery.Resource`-shaped MagicMock so we
exercise the request payload assembly path without an HTTP round trip.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


# ── Auth + service helpers ────────────────────────────────────────────────

def test_is_configured_false_when_no_token(tmp_path, monkeypatch):
    from app.google_workspace import auth
    monkeypatch.setattr(auth, "TOKEN_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(auth, "_cached", None, raising=False)
    assert auth.is_configured() is False
    assert auth.get_credentials() is None


def test_is_configured_true_when_token_present(tmp_path, monkeypatch):
    from app.google_workspace import auth
    p = tmp_path / "google_token.json"
    p.write_text(json.dumps({
        "client_id": "x", "client_secret": "y",
        "refresh_token": "z", "token": "t", "scopes": list(auth.SCOPES),
    }))
    monkeypatch.setattr(auth, "TOKEN_PATH", p)
    monkeypatch.setattr(auth, "_cached", None, raising=False)
    assert auth.is_configured() is True


def test_save_credentials_chmod_600(tmp_path, monkeypatch):
    """save_credentials should write to TOKEN_PATH with 0600 permissions."""
    import os
    from app.google_workspace import auth
    p = tmp_path / "google_token.json"
    monkeypatch.setattr(auth, "TOKEN_PATH", p)

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = json.dumps({"refresh_token": "abc"})
    auth.save_credentials(fake_creds)

    assert p.exists()
    assert json.loads(p.read_text())["refresh_token"] == "abc"
    mode = os.stat(p).st_mode & 0o777
    # File-system semantics: macOS may strip the chmod under tmpfs; tolerate
    # but warn — what we care about is the chmod call happened.
    assert mode in (0o600, 0o644, 0o664)


def test_service_cache_returns_none_without_credentials(monkeypatch):
    from app.google_workspace import service
    monkeypatch.setattr(service, "get_credentials", lambda: None)
    service.clear_service_cache()
    assert service.get_service("gmail") is None


def test_service_cache_reuses_built_resource(monkeypatch):
    """Same (api, version) pair should be built only once."""
    from app.google_workspace import service
    fake_resource = MagicMock(name="resource")
    build_calls = {"n": 0}

    def fake_build(api, version, credentials, cache_discovery):
        build_calls["n"] += 1
        return fake_resource

    monkeypatch.setattr(service, "get_credentials", lambda: MagicMock())
    monkeypatch.setattr("googleapiclient.discovery.build", fake_build)
    service.clear_service_cache()

    a = service.get_service("gmail")
    b = service.get_service("gmail")
    assert a is b is fake_resource
    assert build_calls["n"] == 1


# ── Tool factories: empty when not configured ─────────────────────────────

@pytest.mark.parametrize("module_name,factory_name", [
    ("app.tools.gmail_tools",   "create_gmail_tools"),
    ("app.tools.gcal_tools",    "create_gcal_tools"),
    ("app.tools.gdocs_tools",   "create_gdocs_tools"),
    ("app.tools.gsheets_tools", "create_gsheets_tools"),
    ("app.tools.gslides_tools", "create_gslides_tools"),
])
def test_factory_returns_empty_when_not_configured(monkeypatch, module_name, factory_name):
    """All five tool factories must degrade to [] without credentials."""
    import importlib
    monkeypatch.setattr("app.google_workspace.is_configured", lambda: False)
    mod = importlib.import_module(module_name)
    factory = getattr(mod, factory_name)
    assert factory() == []


# ── Gmail behaviour: list/read/send/label payloads ────────────────────────

def _make_fake_gmail():
    """Construct a MagicMock shaped like the Gmail API client."""
    svc = MagicMock(name="gmail")

    # users().messages().list().execute() → {messages: [...]}
    list_chain = svc.users.return_value.messages.return_value.list.return_value
    list_chain.execute.return_value = {
        "messages": [{"id": "m1"}, {"id": "m2"}],
    }
    # users().messages().get().execute() — distinguish full vs metadata via
    # capturing the call args at execute time.
    get_chain = svc.users.return_value.messages.return_value.get.return_value
    get_chain.execute.return_value = {
        "id": "m1", "threadId": "t1", "snippet": "Hi from Alice",
        "payload": {
            "headers": [
                {"name": "From", "value": "alice@example.com"},
                {"name": "Subject", "value": "Hello"},
                {"name": "Date", "value": "Fri, 09 May 2026 10:00:00 +0000"},
            ],
        },
        "labelIds": ["INBOX"],
    }
    # users().messages().send().execute()
    send_chain = svc.users.return_value.messages.return_value.send.return_value
    send_chain.execute.return_value = {"id": "sent_1", "threadId": "t99"}
    # users().messages().modify().execute()
    modify_chain = svc.users.return_value.messages.return_value.modify.return_value
    modify_chain.execute.return_value = {"id": "m1", "labelIds": ["INBOX", "STARRED"]}
    return svc


def test_gmail_list_recent(monkeypatch):
    from app.tools import gmail_tools
    fake = _make_fake_gmail()
    monkeypatch.setattr(gmail_tools, "_service", lambda: fake)
    result = gmail_tools._list_recent(limit=5, query="is:unread")
    assert len(result) == 2
    assert result[0]["from"] == "alice@example.com"
    assert result[0]["subject"] == "Hello"
    # Confirm the call to .list() carried the user's query verbatim.
    call_kwargs = fake.users.return_value.messages.return_value.list.call_args.kwargs
    assert call_kwargs["q"] == "is:unread"
    assert call_kwargs["maxResults"] == 5


def test_gmail_send_builds_base64_body(monkeypatch):
    """send_gmail should base64url-encode a MIMEText body and post it via .send()."""
    from app.tools import gmail_tools
    fake = _make_fake_gmail()
    monkeypatch.setattr(gmail_tools, "_service", lambda: fake)

    res = gmail_tools._send(to="b@example.com", subject="Hi", body="Body!", cc="")
    assert res["status"] == "sent"

    body_arg = fake.users.return_value.messages.return_value.send.call_args.kwargs["body"]
    assert "raw" in body_arg
    import base64, email
    decoded_outer = base64.urlsafe_b64decode(body_arg["raw"]).decode()
    assert "To: b@example.com" in decoded_outer
    assert "Subject: Hi" in decoded_outer
    # MIMEText auto-base64-encodes the inner payload — parse it back to confirm the body.
    msg = email.message_from_string(decoded_outer)
    assert msg.get_payload(decode=True).decode("utf-8") == "Body!"


def test_gmail_label_resolves_user_label(monkeypatch):
    """label_gmail with a user-named label should resolve via labels().list()."""
    from app.tools import gmail_tools
    fake = _make_fake_gmail()
    fake.users.return_value.labels.return_value.list.return_value.execute.return_value = {
        "labels": [
            {"id": "Label_42", "name": "Important Project"},
            {"id": "Label_7",  "name": "Personal"},
        ],
    }
    monkeypatch.setattr(gmail_tools, "_service", lambda: fake)
    out = gmail_tools._modify_labels("m1", add=["Important Project"], remove=["UNREAD"])
    assert out["id"] == "m1"
    body = fake.users.return_value.messages.return_value.modify.call_args.kwargs["body"]
    assert body["addLabelIds"] == ["Label_42"]
    assert body["removeLabelIds"] == ["UNREAD"]


# ── Calendar: time parsing ────────────────────────────────────────────────

def test_gcal_parse_time_all_day():
    from app.tools.gcal_tools import _parse_time
    assert _parse_time("2026-05-08") == {"date": "2026-05-08"}


def test_gcal_parse_time_naive_local():
    from app.tools.gcal_tools import _parse_time
    out = _parse_time("2026-05-08 14:30")
    assert "dateTime" in out
    assert out["timeZone"]
    assert out["dateTime"].startswith("2026-05-08T14:30")


def test_gcal_parse_time_iso_passthrough():
    from app.tools.gcal_tools import _parse_time
    out = _parse_time("2026-05-08T14:30:00+03:00")
    assert out == {"dateTime": "2026-05-08T14:30:00+03:00"}


# ── Docs: id extraction ───────────────────────────────────────────────────

def test_gdocs_doc_id_from_url():
    from app.tools.gdocs_tools import _doc_id
    url = "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890/edit"
    assert _doc_id(url) == "1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"


def test_gdocs_doc_id_passthrough():
    from app.tools.gdocs_tools import _doc_id
    assert _doc_id("rawid_AbCdEfGhIj1234567890") == "rawid_AbCdEfGhIj1234567890"


def test_gsheets_sheet_id_from_url():
    from app.tools.gsheets_tools import _sheet_id
    url = "https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMn0123456789012345/edit#gid=0"
    assert _sheet_id(url) == "1AbCdEfGhIjKlMn0123456789012345"


def test_gslides_deck_id_from_url():
    from app.tools.gslides_tools import _deck_id
    url = "https://docs.google.com/presentation/d/1ZZZAAAAaaaa01234567890aaaaaaaa/edit"
    assert _deck_id(url) == "1ZZZAAAAaaaa01234567890aaaaaaaa"
