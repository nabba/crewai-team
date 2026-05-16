"""Tests for app.browse.url_canon — the structural privacy contract."""
from __future__ import annotations

from app.browse.url_canon import canonicalize, truncate_title


def test_canonicalize_strips_query_string() -> None:
    """PRIVACY PIN: query strings must never survive canonicalisation.

    This is the structural guarantee — if the canon strips ``?…``, no
    downstream code path can persist a search term or session token
    no matter what bugs accumulate later.
    """
    c = canonicalize("https://example.com/search?q=secret-search-term")
    assert c is not None
    assert c.url == "https://example.com/search"
    assert "secret-search-term" not in c.path
    assert "q=" not in c.path


def test_canonicalize_strips_fragment() -> None:
    """PRIVACY PIN: fragments stripped too."""
    c = canonicalize("https://example.com/page#section-3")
    assert c is not None
    assert c.url == "https://example.com/page"


def test_canonicalize_lowercases_host() -> None:
    c = canonicalize("HTTPS://Example.COM/Foo")
    assert c is not None
    assert c.domain == "example.com"
    assert c.path == "/Foo"  # path case preserved


def test_canonicalize_strips_www_prefix() -> None:
    c = canonicalize("https://www.example.com/")
    assert c is not None
    assert c.domain == "example.com"


def test_canonicalize_rejects_non_http_schemes() -> None:
    for url in (
        "chrome://settings/",
        "about:blank",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<h1>x</h1>",
    ):
        assert canonicalize(url) is None, url


def test_canonicalize_rejects_empty_and_invalid() -> None:
    assert canonicalize("") is None
    assert canonicalize("not a url") is None
    assert canonicalize("https://") is None


def test_canonicalize_strips_trailing_slash_on_non_root() -> None:
    a = canonicalize("https://example.com/foo/")
    b = canonicalize("https://example.com/foo")
    assert a is not None and b is not None
    assert a.path == b.path == "/foo"


def test_canonicalize_keeps_root_slash() -> None:
    c = canonicalize("https://example.com")
    assert c is not None
    assert c.path == "/"


def test_truncate_title_caps_at_200_chars() -> None:
    long = "x" * 500
    out = truncate_title(long)
    assert out is not None
    assert len(out) == 200
    assert out.endswith("…")


def test_truncate_title_passthrough_short_strings() -> None:
    assert truncate_title("Hello") == "Hello"


def test_truncate_title_empty_to_none() -> None:
    assert truncate_title("") is None
    assert truncate_title("   ") is None
    assert truncate_title(None) is None
