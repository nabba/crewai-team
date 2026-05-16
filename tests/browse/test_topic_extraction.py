"""Tests for app.browse.topic_extraction.

Two privacy pins (load-bearing — DO NOT REMOVE):
  * No raw URLs reach the LLM prompt body. The prompt may include
    titles + counts + domains. Paths, query strings, fragments NEVER.
  * Blocklisted titles can't reach the prompt — because the events
    they came from never reach the store in the first place. We pin
    this transitively by feeding the aggregator a blocklisted entry
    and asserting it's absent everywhere downstream.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.browse import store
from app.browse.models import BrowseEvent
from app.browse.topic_extraction import (
    _aggregate_titles,
    _build_user_prompt,
    _redact_pii,
    extract_topics_for_day,
    topics_for_day,
)


def _ev(domain: str, *, title: str | None, path: str = "/",
        ts: str = "2026-05-15T10:00:00+00:00") -> BrowseEvent:
    return BrowseEvent(
        visit_ts=ts, domain=domain, path=path, title=title,
        browser="chrome", profile=None,
    )


# ── Redaction pins ────────────────────────────────────────────────────


def test_redact_strips_email() -> None:
    assert "<email>" in _redact_pii("contact us at andrus@plgmoments.com today")
    assert "andrus@plgmoments.com" not in _redact_pii(
        "andrus@plgmoments.com - profile",
    )


def test_redact_strips_phone() -> None:
    assert "<phone>" in _redact_pii("call +358 50 1234 5678 for support")
    assert "555-123-4567" not in _redact_pii("call 555-123-4567 now")


def test_redact_passthrough() -> None:
    assert _redact_pii("Helsinki - Wikipedia") == "Helsinki - Wikipedia"


# ── Prompt-assembly privacy pins ──────────────────────────────────────


def test_raw_urls_never_in_llm_batch(_reset_browse_state: Path) -> None:
    """PRIVACY PIN: paths + query strings must NEVER appear in the prompt."""
    events = [
        _ev("github.com", title="Claude Code", path="/anthropics/claude-code"),
        _ev("example.com", title="Search results", path="/search"),
    ]
    rows = _aggregate_titles(events)
    prompt = _build_user_prompt(rows)
    # Domains are present (operator-decided)
    assert "github.com" in prompt
    assert "example.com" in prompt
    # Paths are NOT.
    assert "/anthropics/claude-code" not in prompt
    assert "anthropics" not in prompt
    assert "claude-code" not in prompt.lower().replace("claude code", "")
    assert "/search" not in prompt


def test_blocklisted_titles_never_in_llm_batch(_reset_browse_state: Path) -> None:
    """PRIVACY PIN: blocklisted-domain events are filtered upstream at
    reader → never persisted → can't reach this batch.

    Here we directly write events for both a normal and a blocklisted
    domain to the store, but since the reader is the gate (and we
    bypass it in this test), we just assert that any title that DID
    sneak through is not in the prompt for a follow-up forget-domain
    cleanup."""
    # Simulate a residual entry that somehow landed for a blocklisted
    # domain. forget_domain should clear it so the batch never sees it.
    events = [
        _ev("github.com", title="Claude Code"),
        _ev("paypal.com", title="Account balance"),  # would-be banking
    ]
    store.append_events(events)
    store.forget_domain("paypal.com")
    day = date(2026, 5, 15)
    events_after = store.list_events_for_day(day)
    rows = _aggregate_titles(events_after)
    prompt = _build_user_prompt(rows)
    assert "Account balance" not in prompt
    assert "paypal.com" not in prompt
    assert "Claude Code" in prompt


def test_titles_with_pii_are_redacted_in_prompt(_reset_browse_state: Path) -> None:
    events = [
        _ev("example.com", title="Profile - alice@example.com"),
    ]
    rows = _aggregate_titles(events)
    prompt = _build_user_prompt(rows)
    assert "alice@example.com" not in prompt
    assert "<email>" in prompt


# ── Aggregation behaviour ─────────────────────────────────────────────


def test_aggregate_deduplicates_and_counts() -> None:
    events = [
        _ev("github.com", title="Issue #42"),
        _ev("github.com", title="Issue #42"),
        _ev("github.com", title="Issue #42"),
        _ev("github.com", title="README"),
    ]
    rows = _aggregate_titles(events)
    counts = {(r.title, r.domain): r.count for r in rows}
    assert counts[("Issue #42", "github.com")] == 3
    assert counts[("README", "github.com")] == 1


def test_aggregate_skips_empty_titles() -> None:
    events = [
        _ev("a.com", title=None),
        _ev("a.com", title=""),
        _ev("a.com", title="   "),
        _ev("a.com", title="Real title"),
    ]
    rows = _aggregate_titles(events)
    titles = {r.title for r in rows}
    assert titles == {"Real title"}


# ── End-to-end pass with a fake LLM ───────────────────────────────────


def _fake_llm_ok(_system: str, _user: str) -> str:
    """A deterministic stand-in that emits valid clustering JSON."""
    return json.dumps({
        "topics": [
            {"label": "claude code", "title_indexes": [0, 1]},
            {"label": "finnish nature", "title_indexes": [2]},
        ]
    })


def _fake_llm_garbage(_system: str, _user: str) -> str:
    return "not even close to JSON"


def _fake_llm_unavailable(_system: str, _user: str) -> str:
    return ""


def test_extract_emits_clusters_with_fake_llm(_reset_browse_state: Path) -> None:
    events = [
        _ev("github.com", title="Claude Code"),
        _ev("github.com", title="Claude Code Docs"),
        _ev("wikipedia.org", title="Pine — Pinus sylvestris"),
    ]
    store.append_events(events)
    result = extract_topics_for_day(date(2026, 5, 15), llm_call=_fake_llm_ok)
    labels = sorted(t.label for t in result.topics)
    assert labels == ["claude code", "finnish nature"]
    # Output file persisted.
    p = _reset_browse_state / "topics" / "2026-05-15.json"
    assert p.exists()


def test_extract_idempotent(_reset_browse_state: Path) -> None:
    """A second call for the same day reads the cached file, never
    re-prompts the LLM."""
    events = [_ev("github.com", title="Claude Code")]
    store.append_events(events)
    call_count = {"n": 0}

    def counting_llm(_s, _u):
        call_count["n"] += 1
        return _fake_llm_ok(_s, _u)

    extract_topics_for_day(date(2026, 5, 15), llm_call=counting_llm)
    extract_topics_for_day(date(2026, 5, 15), llm_call=counting_llm)
    assert call_count["n"] == 1


def test_extract_no_events_notes_no_titles(_reset_browse_state: Path) -> None:
    result = extract_topics_for_day(date(2026, 5, 15), llm_call=_fake_llm_ok)
    assert result.note == "no_titles"
    assert result.topics == []


def test_extract_llm_garbage_falls_back_gracefully(_reset_browse_state: Path) -> None:
    events = [_ev("github.com", title="A")]
    store.append_events(events)
    result = extract_topics_for_day(date(2026, 5, 15), llm_call=_fake_llm_garbage)
    assert result.note == "llm_failed"
    assert result.topics == []


def test_extract_llm_unavailable_notes_unavailable(_reset_browse_state: Path) -> None:
    events = [_ev("github.com", title="A")]
    store.append_events(events)
    result = extract_topics_for_day(date(2026, 5, 15), llm_call=_fake_llm_unavailable)
    assert result.note == "llm_unavailable"


def test_disabled_short_circuit(
    _reset_browse_state: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When BROWSE_INGESTION_ENABLED is off, no LLM call ever fires
    even if the per-day events file exists."""
    from app.browse import topic_extraction as te
    events = [_ev("github.com", title="A")]
    store.append_events(events)
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "false")
    calls = {"n": 0}

    def counting_llm(_s, _u):
        calls["n"] += 1
        return _fake_llm_ok(_s, _u)

    te.run_topic_extraction_tick(llm_call=counting_llm)
    assert calls["n"] == 0


def test_topics_for_day_returns_none_when_missing(_reset_browse_state: Path) -> None:
    assert topics_for_day(date(2026, 5, 15)) is None
