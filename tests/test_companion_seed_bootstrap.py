"""Tests for app.companion.seed_bootstrap — cold-start seed derivation."""

from unittest.mock import patch

import pytest

from app.companion import seed_bootstrap as _sb


def _stub_project(name="PLG", mission="Build the ticketing platform"):
    return {"id": "ws-1", "name": name, "mission": mission}


# ── Activation gates ───────────────────────────────────────────────────────

def test_returns_none_for_unknown_workspace():
    with patch("app.companion.seed_bootstrap._get_project", lambda ws: None):
        assert _sb.derive_seed("nope") is None


def test_returns_none_for_blocklisted_name():
    """The 'default' workspace is the catch-all and shouldn't bootstrap."""
    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project(name="default")), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: ["a"] * 5):
        assert _sb.derive_seed("ws-1") is None


def test_returns_none_with_no_mission_and_no_tickets():
    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project(mission="")), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: []):
        assert _sb.derive_seed("ws-1") is None


def test_returns_none_with_short_mission_and_no_tickets():
    """Mission below MIN_MISSION_CHARS doesn't count as signal."""
    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project(mission="hi")), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: []):
        assert _sb.derive_seed("ws-1") is None


# ── Source signal classification ───────────────────────────────────────────

def test_uses_mission_only_when_no_tickets():
    raw = "SEED: a great seed about ticketing\nRATIONALE: r"
    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project()), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: []), \
         patch("app.companion.seed_bootstrap._invoke_synthesizer",
               lambda p: raw):
        d = _sb.derive_seed("ws-1")
    assert d is not None
    assert d.source_signal == "mission_only"
    assert d.ticket_count == 0
    assert d.has_mission is True


def test_uses_tickets_only_when_no_mission():
    raw = "SEED: distilled from your tickets\nRATIONALE: r"
    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project(mission="")), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: ["build chart", "write essay"]), \
         patch("app.companion.seed_bootstrap._invoke_synthesizer",
               lambda p: raw):
        d = _sb.derive_seed("ws-1")
    assert d is not None
    assert d.source_signal == "tickets_only"
    assert d.ticket_count == 2
    assert d.has_mission is False


def test_combines_mission_plus_tickets():
    raw = "SEED: combined seed\nRATIONALE: r"
    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project()), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: ["one", "two", "three"]), \
         patch("app.companion.seed_bootstrap._invoke_synthesizer",
               lambda p: raw):
        d = _sb.derive_seed("ws-1")
    assert d is not None
    assert d.source_signal == "mission+tickets"
    assert d.ticket_count == 3


# ── LLM failure modes ─────────────────────────────────────────────────────

def test_handles_llm_failure():
    def _broken(p):
        raise RuntimeError("LLM down")

    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project()), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: ["x"]), \
         patch("app.companion.seed_bootstrap._invoke_synthesizer", _broken):
        assert _sb.derive_seed("ws-1") is None


def test_handles_unparseable_response():
    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project()), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: ["x"]), \
         patch("app.companion.seed_bootstrap._invoke_synthesizer",
               lambda p: "I refuse to answer."):
        assert _sb.derive_seed("ws-1") is None


def test_clamps_long_seed_to_max_chars():
    """Even if the LLM ignores the 250-char cap, we enforce it."""
    long_seed = "x" * 1000
    raw = f"SEED: {long_seed}\nRATIONALE: r"
    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project()), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: ["x"]), \
         patch("app.companion.seed_bootstrap._invoke_synthesizer",
               lambda p: raw):
        d = _sb.derive_seed("ws-1")
    assert d is not None
    assert len(d.text) <= _sb.SEED_MAX_CHARS


def test_recent_tickets_swallows_failure_in_unconfigured_env():
    """The real _recent_tickets returns [] when CP DB is unavailable —
    the import + execute call itself is wrapped in try/except. In the
    test env psycopg2 isn't installed, so the import fails and we get []."""
    out = _sb._recent_tickets("ws-1", limit=10)
    assert out == []


def test_get_project_swallows_failure_in_unconfigured_env():
    """The real _get_project returns None when CP DB is unavailable."""
    assert _sb._get_project("ws-1") is None


# ── Prompt construction ───────────────────────────────────────────────────

def test_passes_project_name_into_prompt():
    captured: list[str] = []

    def _capture(p):
        captured.append(p)
        return "SEED: ok\nRATIONALE: r"

    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project(name="Eesti mets",
                                         mission="Estonian forests")), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               lambda ws, **kw: ["forest age graphic", "forest health essay"]), \
         patch("app.companion.seed_bootstrap._invoke_synthesizer", _capture):
        _sb.derive_seed("ws-1")

    prompt = captured[0]
    assert "Eesti mets" in prompt
    assert "Estonian forests" in prompt
    assert "forest age graphic" in prompt
    assert "forest health essay" in prompt


def test_caps_ticket_count_in_prompt():
    """MAX_TICKETS_IN_PROMPT bounds how many tickets reach the LLM."""
    big = [f"ticket {i}" for i in range(50)]
    captured: list[str] = []

    def _capture(p):
        captured.append(p)
        return "SEED: ok\nRATIONALE: r"

    # _recent_tickets is called with limit=MAX_TICKETS_IN_PROMPT, and
    # the (mocked) function returns at most that many. So the prompt
    # cannot exceed it. Verify by counting "- ticket" lines.
    def _bounded_tickets(ws, *, limit):
        return big[:limit]

    with patch("app.companion.seed_bootstrap._get_project",
               lambda ws: _stub_project()), \
         patch("app.companion.seed_bootstrap._recent_tickets",
               _bounded_tickets), \
         patch("app.companion.seed_bootstrap._invoke_synthesizer", _capture):
        _sb.derive_seed("ws-1")

    bullet_count = captured[0].count("\n- ticket ")
    assert bullet_count <= _sb.MAX_TICKETS_IN_PROMPT


# ── Parser ─────────────────────────────────────────────────────────────────

def test_parse_extracts_both_fields():
    raw = ("SEED: cathedral-scale forest decision support over generations\n"
           "RATIONALE: ties ecology to indigenous knowledge to measurement.")
    seed, rat = _sb._parse(raw)
    assert "cathedral-scale" in seed
    assert "ecology" in rat


def test_parse_handles_missing_rationale():
    seed, rat = _sb._parse("SEED: just the seed")
    assert seed == "just the seed"
    assert rat == ""


def test_parse_returns_empty_on_garbage():
    seed, rat = _sb._parse("totally unrelated output")
    assert seed == ""
    assert rat == ""


def test_parse_collapses_whitespace_in_rationale():
    raw = "SEED: x\nRATIONALE: line one  \n  line two   line three"
    _, rat = _sb._parse(raw)
    assert "  " not in rat
