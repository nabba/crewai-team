"""Tests for the act-now email digest (LLM-graded thrice-daily
synthesis of the last 48h unread inbox).

Sibling of email_monitor — different shape: 3-hour cadence between
07:00–22:00 local, 48h lookback, LLM content analysis, top-7 with
"why" + "action" + Gmail link, heuristic fallback when LLM fails.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.life_companion.act_now_digest import (
    _BULK_LABELS_DROP,
    _Candidate,
    _build_emails_block,
    _format_digest,
    _gmail_link,
    _pre_filter,
    _slot_key_for,
    _FIRE_HOURS,
)


# ── Cadence + business-hours gating ────────────────────────────────


class TestSlotKeyFor:
    """Slot key gates business-hour fire windows AND prevents
    double-firing within the same 30-min tolerance window."""

    def test_at_07_00_returns_key(self) -> None:
        now = datetime(2026, 5, 11, 7, 0, 0)
        assert _slot_key_for(now) == "2026-05-11-07"

    def test_at_07_14_within_tolerance(self) -> None:
        now = datetime(2026, 5, 11, 7, 14, 30)
        assert _slot_key_for(now) == "2026-05-11-07"

    def test_at_07_16_outside_tolerance(self) -> None:
        """Past +15 min from 07:00 — none of the slots match."""
        now = datetime(2026, 5, 11, 7, 16, 0)
        assert _slot_key_for(now) is None

    def test_at_22_00_is_last_slot(self) -> None:
        now = datetime(2026, 5, 11, 22, 0, 0)
        assert _slot_key_for(now) == "2026-05-11-22"

    def test_at_03_00_outside_window(self) -> None:
        """Middle of the night — no slot."""
        assert _slot_key_for(datetime(2026, 5, 11, 3, 0, 0)) is None

    def test_at_23_30_outside_window(self) -> None:
        """Late night — past 22:15 → no slot."""
        assert _slot_key_for(datetime(2026, 5, 11, 23, 30, 0)) is None

    def test_all_six_slots_fire(self) -> None:
        for hour in _FIRE_HOURS:
            now = datetime(2026, 5, 11, hour, 0, 0)
            assert _slot_key_for(now) == f"2026-05-11-{hour:02d}"

    def test_distinct_keys_for_different_days(self) -> None:
        d1 = datetime(2026, 5, 11, 7, 0, 0)
        d2 = datetime(2026, 5, 12, 7, 0, 0)
        assert _slot_key_for(d1) != _slot_key_for(d2)


# ── Pre-filter: drop bulk before the LLM call ─────────────────────


def _cand(
    *, id_: str = "a1", labels: tuple[str, ...] = ("INBOX", "UNREAD"),
    from_: str = "Alice <a@x.com>", subject: str = "test", body: str = "",
) -> _Candidate:
    return _Candidate(
        id=id_, from_=from_, subject=subject,
        date="Mon, 10 May 2026 09:00:00 +0000",
        snippet="", body=body, label_ids=labels,
    )


class TestPreFilter:

    def test_drops_promotions(self) -> None:
        bulk = _cand(id_="b1", labels=("INBOX", "UNREAD", "CATEGORY_PROMOTIONS"))
        assert _pre_filter([bulk]) == []

    def test_drops_social(self) -> None:
        bulk = _cand(id_="b2", labels=("CATEGORY_SOCIAL",))
        assert _pre_filter([bulk]) == []

    def test_drops_forums(self) -> None:
        bulk = _cand(id_="b3", labels=("CATEGORY_FORUMS",))
        assert _pre_filter([bulk]) == []

    def test_keeps_updates(self) -> None:
        """Flight changes / banking / package tracking can be act-now —
        keep CATEGORY_UPDATES even though Gmail tab-categorizes it."""
        c = _cand(id_="u1", labels=("INBOX", "UNREAD", "CATEGORY_UPDATES"))
        assert _pre_filter([c]) == [c]

    def test_keeps_inbox_only(self) -> None:
        c = _cand(id_="i1", labels=("INBOX", "UNREAD"))
        assert _pre_filter([c]) == [c]

    def test_drop_set_matches_constants(self) -> None:
        """The drop set must include exactly the three bulk
        categories — adding CATEGORY_UPDATES would over-filter."""
        assert _BULK_LABELS_DROP == frozenset({
            "CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS",
        })


# ── Gmail link format ─────────────────────────────────────────────


class TestGmailLink:

    def test_standard_message_id(self) -> None:
        assert _gmail_link("abc123") == (
            "https://mail.google.com/mail/u/0/#inbox/abc123"
        )

    def test_empty_id_returns_empty(self) -> None:
        assert _gmail_link("") == ""

    def test_handles_long_thread_id(self) -> None:
        # Gmail message ids are sometimes >20 chars — must pass through.
        long_id = "1817b3a4f2e9d6c7"
        link = _gmail_link(long_id)
        assert long_id in link


# ── Output format ──────────────────────────────────────────────────


class TestFormatDigest:

    def test_empty_ranked_emits_nothing_actionable_message(self) -> None:
        out = _format_digest(
            ranked=[],
            candidates_by_id={},
            total_unread=10, pre_filtered_dropped=8,
        )
        assert "Top 0" in out
        assert "Nothing requires action" in out

    def test_full_digest_includes_link_why_action(self) -> None:
        c = _cand(id_="m1", from_="CFO <cfo@acme.com>",
                  subject="Q3 board deck — sign-off needed by EOD",
                  body="Please review and sign off today.")
        ranked = [{
            "email_id": "m1",
            "why_now": "explicit deadline today",
            "suggested_action": "review draft + reply with sign-off",
            "deadline_hint": "EOD today",
            "rank": 1,
        }]
        out = _format_digest(
            ranked, candidates_by_id={"m1": c},
            total_unread=14, pre_filtered_dropped=5,
        )
        # Header tells the operator the filter math.
        assert "Top 1" in out
        assert "14 unread" in out
        assert "9 after bulk-filter" in out
        # Payload contains all four signals.
        assert "CFO <cfo@acme.com>" in out
        assert "Q3 board deck" in out
        assert "explicit deadline today" in out
        assert "review draft" in out
        assert "EOD today" in out
        assert "https://mail.google.com/mail/u/0/#inbox/m1" in out
        assert "📨" in out

    def test_skips_entry_without_matching_candidate(self) -> None:
        """If the LLM hallucinates an email_id that's not in the
        candidate set, we silently drop the entry rather than
        rendering a bogus one."""
        ranked = [{
            "email_id": "ghost", "why_now": "x",
            "suggested_action": "y", "deadline_hint": "", "rank": 1,
        }]
        out = _format_digest(
            ranked, candidates_by_id={}, total_unread=1, pre_filtered_dropped=0,
        )
        # Header still shows "Top 1" because that's what the LLM
        # returned, but the body has no per-email block.
        assert "ghost" not in out

    def test_omits_optional_fields_when_empty(self) -> None:
        c = _cand(id_="m2")
        ranked = [{
            "email_id": "m2", "why_now": "",
            "suggested_action": "", "deadline_hint": "", "rank": 1,
        }]
        out = _format_digest(
            ranked, candidates_by_id={"m2": c},
            total_unread=1, pre_filtered_dropped=0,
        )
        assert "why:" not in out
        assert "action:" not in out
        assert "deadline:" not in out
        # But the link + sender + subject still render.
        assert "https://mail.google.com/mail/u/0/#inbox/m2" in out


# ── LLM contract ──────────────────────────────────────────────────


class TestRankWithLlm:

    def test_well_formed_response_parsed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.life_companion import act_now_digest as mod

        class _StubLLM:
            def call(self, messages):
                # Return a clean JSON object the LLM is supposed to emit.
                return json.dumps({
                    "ranked": [
                        {"email_id": "a1", "why_now": "deadline tonight",
                         "suggested_action": "reply with approval",
                         "deadline_hint": "tonight", "rank": 1},
                        {"email_id": "a2", "why_now": "flight change",
                         "suggested_action": "confirm seat",
                         "deadline_hint": "", "rank": 2},
                    ]
                })

        monkeypatch.setattr(mod, "_get_ranking_llm", lambda: _StubLLM())

        candidates = [_cand(id_="a1"), _cand(id_="a2"), _cand(id_="a3")]
        out = mod._rank_with_llm(candidates, top_k=7)
        assert out is not None
        assert len(out) == 2
        assert out[0]["email_id"] == "a1"
        assert out[0]["why_now"] == "deadline tonight"

    def test_drops_hallucinated_id(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the LLM fabricates an email_id not present in the
        candidate set, the entry must be dropped silently."""
        from app.life_companion import act_now_digest as mod

        class _StubLLM:
            def call(self, messages):
                return json.dumps({
                    "ranked": [
                        {"email_id": "real1", "why_now": "x",
                         "suggested_action": "y", "deadline_hint": "",
                         "rank": 1},
                        {"email_id": "FAKE_ID_NOT_REAL", "why_now": "x",
                         "suggested_action": "y", "deadline_hint": "",
                         "rank": 2},
                    ]
                })

        monkeypatch.setattr(mod, "_get_ranking_llm", lambda: _StubLLM())

        out = mod._rank_with_llm([_cand(id_="real1")], top_k=7)
        assert out == [{
            "email_id": "real1", "why_now": "x",
            "suggested_action": "y", "deadline_hint": "", "rank": 1,
        }]

    def test_malformed_json_returns_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.life_companion import act_now_digest as mod

        class _StubLLM:
            def call(self, messages):
                return "this is not JSON, prose only"

        monkeypatch.setattr(mod, "_get_ranking_llm", lambda: _StubLLM())

        out = mod._rank_with_llm([_cand()], top_k=7)
        assert out is None  # caller falls back to heuristic

    def test_empty_ranked_list_is_valid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the LLM legitimately decides no email is act-now,
        empty list must round-trip cleanly (NOT trigger fallback)."""
        from app.life_companion import act_now_digest as mod

        class _StubLLM:
            def call(self, messages):
                return json.dumps({"ranked": []})

        monkeypatch.setattr(mod, "_get_ranking_llm", lambda: _StubLLM())

        out = mod._rank_with_llm([_cand()], top_k=7)
        assert out == []


# ── Prompt contract ──────────────────────────────────────────────


class TestEmailsBlock:
    """The LLM input must include id + sender + subject + body for
    every candidate so the model can rank them."""

    def test_includes_id_sender_subject_body(self) -> None:
        c = _cand(
            id_="msg-9", from_="Alice <a@x.com>",
            subject="Quick question", body="Can you confirm Tuesday?",
        )
        block = _build_emails_block([c])
        assert "id: msg-9" in block
        assert "from: Alice <a@x.com>" in block
        assert "subject: Quick question" in block
        assert "Can you confirm Tuesday?" in block

    def test_falls_back_to_snippet_when_body_empty(self) -> None:
        c = _Candidate(
            id="x", from_="A", subject="S", date="d",
            snippet="snippet here", body="", label_ids=(),
        )
        block = _build_emails_block([c])
        assert "snippet here" in block


# ── Wiring contract ──────────────────────────────────────────────


class TestWiredIntoIdleScheduler:

    def test_act_now_digest_in_get_idle_jobs(self) -> None:
        from app.life_companion import get_idle_jobs
        names = [job[0] for job in get_idle_jobs()]
        assert "life-companion-act-now-digest" in names, (
            "act_now_digest must be registered as an idle job — "
            "otherwise it will never run"
        )
