"""Tests for app/dashboard_links.py.

The operator approves things from two devices: iPhone (PWA via
Tailscale Funnel HTTPS) and Macbook (Vite dev server on the Tailnet).
Every approval Signal message must surface BOTH URLs so the operator
can tap whichever device is at hand.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.dashboard_links import (  # noqa: E402
    DEFAULT_IPHONE_HOST,
    DEFAULT_MAC_HOST,
    signal_links_block,
    url_iphone,
    url_macbook,
)


class TestIphoneURL:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://override.test")
        assert url_iphone("/cp/foo") == "https://override.test/cp/foo"

    def test_default_is_https_funnel(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_PUBLIC_URL", raising=False)
        assert url_iphone("/cp/foo").startswith("https://")
        assert url_iphone("/cp/foo").endswith("/cp/foo")

    def test_path_gets_leading_slash(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://x.test")
        assert url_iphone("cp/foo") == "https://x.test/cp/foo"

    def test_trailing_slash_in_base_normalized(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://x.test/")
        assert url_iphone("/cp/foo") == "https://x.test/cp/foo"


class TestMacbookURL:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_MAC_URL", "http://laptop.local:3100")
        assert url_macbook("/cp/foo") == "http://laptop.local:3100/cp/foo"

    def test_default_includes_dev_port(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_MAC_URL", raising=False)
        # Default should be the Tailnet hostname + Vite port.
        url = url_macbook("/cp/foo")
        assert ":3100" in url
        assert url.endswith("/cp/foo")

    def test_default_is_http_not_https(self, monkeypatch):
        """The Mac default is plain HTTP because the dev server doesn't
        terminate TLS — Funnel HTTPS is the iPhone path, not the Mac path."""
        monkeypatch.delenv("DASHBOARD_MAC_URL", raising=False)
        assert url_macbook("/cp/foo").startswith("http://")


class TestLinksBlock:
    def test_block_has_both_devices(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://phone.test")
        monkeypatch.setenv("DASHBOARD_MAC_URL", "http://mac.test:3100")
        block = signal_links_block("/cp/foo")
        assert "📱" in block
        assert "💻" in block
        assert "https://phone.test/cp/foo" in block
        assert "http://mac.test:3100/cp/foo" in block

    def test_block_is_two_lines(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://phone.test")
        monkeypatch.setenv("DASHBOARD_MAC_URL", "http://mac.test:3100")
        block = signal_links_block("/cp/foo")
        assert block.count("\n") == 1, (
            "Two-link block should be exactly two lines (one newline). "
            "Got:\n" + block
        )

    def test_iphone_listed_first(self, monkeypatch):
        """Operators are more often on the phone when a Signal alert
        lands — iPhone link comes first so it's the first tap target."""
        block = signal_links_block("/cp/foo")
        lines = block.split("\n")
        assert "📱" in lines[0]
        assert "💻" in lines[1]

    def test_defaults_render_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_PUBLIC_URL", raising=False)
        monkeypatch.delenv("DASHBOARD_MAC_URL", raising=False)
        block = signal_links_block("/cp/governance")
        assert DEFAULT_IPHONE_HOST in block
        assert DEFAULT_MAC_HOST in block
        assert "/cp/governance" in block
