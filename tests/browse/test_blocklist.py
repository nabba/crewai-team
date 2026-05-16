"""Tests for app.browse.blocklist."""
from __future__ import annotations

from pathlib import Path

from app.browse import blocklist


def test_seed_blocks_banking() -> None:
    assert blocklist.is_blocked("paypal.com")
    assert blocklist.is_blocked("op.fi")
    assert blocklist.is_blocked("revolut.com")


def test_seed_blocks_health_portals() -> None:
    assert blocklist.is_blocked("kanta.fi")
    assert blocklist.is_blocked("digilugu.ee")


def test_seed_blocks_subdomains_via_suffix_match() -> None:
    """A seed entry of ``example.com`` should also block ``api.example.com``."""
    assert blocklist.is_blocked("api.paypal.com")
    assert blocklist.is_blocked("login.kanta.fi")


def test_seed_does_not_block_lookalikes() -> None:
    """``notpaypal.com`` should NOT match ``paypal.com``."""
    assert not blocklist.is_blocked("notpaypal.com")
    assert not blocklist.is_blocked("paypal-imposter.com")


def test_unblocked_domain_passes() -> None:
    assert not blocklist.is_blocked("github.com")
    assert not blocklist.is_blocked("en.wikipedia.org")


def test_empty_domain_not_blocked() -> None:
    assert not blocklist.is_blocked("")


def test_www_prefix_handled() -> None:
    """A query with ``www.`` prefix should match the same seed pattern."""
    assert blocklist.is_blocked("www.paypal.com")


def test_operator_file_adds_entries(_reset_browse_state: Path) -> None:
    bl_path = _reset_browse_state / "blocklist.txt"
    bl_path.write_text("# custom\nexample-bank.test\n*.private.corp\n", encoding="utf-8")
    blocklist.reset_cache()
    assert blocklist.is_blocked("example-bank.test")
    assert blocklist.is_blocked("foo.private.corp")
    assert blocklist.is_blocked("private.corp")


def test_operator_file_comments_ignored(_reset_browse_state: Path) -> None:
    bl_path = _reset_browse_state / "blocklist.txt"
    bl_path.write_text("#this-is-a-comment\n  \n", encoding="utf-8")
    blocklist.reset_cache()
    assert not blocklist.is_blocked("this-is-a-comment")


def test_mute_domain_is_idempotent(_reset_browse_state: Path) -> None:
    bl_path = _reset_browse_state / "blocklist.txt"
    assert blocklist.mute_domain("first.example") is True
    assert blocklist.mute_domain("first.example") is False  # already muted
    text = bl_path.read_text(encoding="utf-8")
    assert text.count("first.example") == 1


def test_mute_domain_persists(_reset_browse_state: Path) -> None:
    assert blocklist.mute_domain("muted.example.com")
    assert blocklist.is_blocked("muted.example.com")
    assert blocklist.is_blocked("sub.muted.example.com")


def test_list_seed_entries_returns_tuple_contents() -> None:
    seeds = blocklist.list_seed_entries()
    assert "paypal.com" in seeds
    assert "kanta.fi" in seeds
    assert len(seeds) > 10
