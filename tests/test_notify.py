"""
test_notify — completion notifications + decorator.

The Signal/Web-Push channels are stubbed so we exercise the dispatch
logic without touching real services.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _stub_channels(monkeypatch):
    """Capture every signal + web-push send for assertion."""
    sent: dict[str, list] = {"signal": [], "web_push": []}

    def fake_signal(title: str, body: str) -> bool:
        sent["signal"].append((title, body))
        return True

    def fake_web_push(title: str, body: str, *, url: str, tag: str) -> int:
        sent["web_push"].append({"title": title, "body": body, "url": url, "tag": tag})
        return len(sent["web_push"])

    monkeypatch.setattr("app.notify.api._send_signal", fake_signal)
    monkeypatch.setattr("app.notify.api._send_web_push", fake_web_push)
    return sent


# ── notify() fan-out ──────────────────────────────────────────────────────

def test_notify_fans_out_to_both_channels(_stub_channels):
    from app.notify import notify
    out = notify("Hello", "world", url="/cp/skills")
    assert out["signal"] is True
    assert out["web_push_count"] == 1
    assert _stub_channels["signal"] == [("Hello", "world")]
    assert _stub_channels["web_push"][0]["url"] == "/cp/skills"


def test_notify_can_disable_signal(_stub_channels):
    from app.notify import notify
    out = notify("only push", "body", signal=False)
    assert out["signal"] is False
    assert _stub_channels["signal"] == []
    assert out["web_push_count"] == 1


def test_notify_can_disable_web_push(_stub_channels):
    from app.notify import notify
    out = notify("only signal", "body", web_push=False)
    assert out["signal"] is True
    assert out["web_push_count"] == 0
    assert _stub_channels["web_push"] == []


# ── @notify_on_complete on sync functions ────────────────────────────────

def test_decorator_pings_on_success(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete(label="my job")
    def work() -> int:
        return 42

    assert work() == 42
    assert len(_stub_channels["signal"]) == 1
    title, body = _stub_channels["signal"][0]
    assert title == "my job"
    assert body.startswith("✓ done")


def test_decorator_pings_on_failure(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete(label="boom")
    def kaboom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        kaboom()

    assert len(_stub_channels["signal"]) == 1
    title, body = _stub_channels["signal"][0]
    assert title == "boom"
    assert body.startswith("✗ failed:")
    assert "RuntimeError" in body
    assert "nope" in body


def test_decorator_failure_only_skips_success(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete(label="quiet job", notify_on_failure_only=True)
    def quiet() -> None:
        return None

    quiet()
    assert _stub_channels["signal"] == []


def test_decorator_failure_only_still_notifies_failure(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete(label="quiet boom", notify_on_failure_only=True)
    def boom() -> None:
        raise ValueError("bad")

    with pytest.raises(ValueError):
        boom()

    assert len(_stub_channels["signal"]) == 1
    assert "✗ failed:" in _stub_channels["signal"][0][1]


def test_decorator_silent_kill_switch(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete(label="never", silent=True)
    def silent() -> None:
        return None

    @notify_on_complete(label="never_either", silent=True)
    def silent_boom() -> None:
        raise RuntimeError("ignored")

    silent()
    with pytest.raises(RuntimeError):
        silent_boom()

    assert _stub_channels["signal"] == []
    assert _stub_channels["web_push"] == []


def test_decorator_uses_function_qualname_when_label_missing(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete()
    def my_job() -> None:
        return None

    my_job()
    title, _ = _stub_channels["signal"][0]
    assert "my_job" in title


def test_decorator_skips_keyboard_interrupt(_stub_channels):
    """Operator-initiated SIGINT shouldn't spam a "✗ failed" notification."""
    from app.notify import notify_on_complete

    @notify_on_complete(label="job")
    def cancelled() -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        cancelled()
    assert _stub_channels["signal"] == []


def test_decorator_preserves_return_value_and_args(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete(label="echo")
    def echo(x: int, *, scale: int = 1) -> int:
        return x * scale

    assert echo(5, scale=3) == 15


# ── @notify_on_complete on async functions ───────────────────────────────

def test_decorator_handles_async_success(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete(label="async job")
    async def aok() -> int:
        await asyncio.sleep(0)
        return 7

    result = asyncio.run(aok())
    assert result == 7
    assert len(_stub_channels["signal"]) == 1
    assert _stub_channels["signal"][0][0] == "async job"
    assert "done" in _stub_channels["signal"][0][1]


def test_decorator_handles_async_failure(_stub_channels):
    from app.notify import notify_on_complete

    @notify_on_complete(label="async boom")
    async def aboom() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("async bad")

    with pytest.raises(RuntimeError):
        asyncio.run(aboom())
    assert len(_stub_channels["signal"]) == 1
    assert "RuntimeError" in _stub_channels["signal"][0][1]


# ── Duration formatting ───────────────────────────────────────────────────

@pytest.mark.parametrize("seconds,expected_substr", [
    (0.05, "ms"),
    (3.7, "s"),
    (90, "1m30s"),
    (3725, "1h02m"),
])
def test_human_duration_formats(seconds, expected_substr):
    from app.notify.api import _human_duration
    assert expected_substr in _human_duration(seconds)
