"""Regression: email triage must not surface marketing emails as
"urgent unread" (operator-reported 2026-05-10).

Pre-fix shape:

  📬 Email triage — 3 urgent unread:
    • DailyOM <today@dailyom.com>
      Last Chance: The Seven Strengths Free Live Online Event
      score=2.5
    • Wild Gym <info@wildgym.com>
      Your bones need a reason to stay strong
      score=2.5
    • "Swimmer.com.au" <news@swimmer.com.au>
      💦 The Ultimate in Swim Backpacks
      score=2.5

Root cause — TWO architectural gaps in the input pipeline:

  1. ``_list_recent`` only requested ``metadataHeaders=["From",
     "Subject", "Date"]`` from the Gmail API, leaving the bulk
     markers (List-Unsubscribe / List-ID / Auto-Submitted /
     Precedence) unfetched.  The scorer had the right weights
     (-3 / -2 / -2 / -2) but the input was blind.

  2. ``_build_headers`` in email_monitor hardcoded every bulk
     marker to ``None``, AND the scorer didn't understand
     Gmail's tab-category labels (CATEGORY_PROMOTIONS / SOCIAL /
     UPDATES / FORUMS) — which were already in the stub but
     unread.  Google's own ML calls these "Promotions" with
     higher precision than any single header rule.

Each marketing email scored:
  +1 (human From: display name + non-noreply addr)
  +1 (unread)
  +0.5 (recent)
  ≈ +2.5

…above the threshold (1.0) → false alarm.

Post-fix:
  • ``_list_recent`` widens metadataHeaders to include
    List-Unsubscribe, List-Id, Auto-Submitted, Precedence,
    In-Reply-To, References, To, Cc.
  • ``EmailHeaders`` adds ``gmail_labels: tuple[str, ...]``.
  • ``score_email`` penalizes Gmail bulk-category labels
    (Promotions: -4, Social: -3, Updates: -2, Forums: -2),
    overriding the +2.5 personal-marker noise.
  • ``_build_headers`` in email_monitor pulls these fields
    through from the stub.

Result: a CATEGORY_PROMOTIONS email scores ~-1.5, well below
the 1.0 threshold.  Genuinely-personal mail still wins.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.tools.email_importance import (
    EmailHeaders,
    _GMAIL_BULK_LABEL_PENALTIES,
    score_email,
)


# ── Layer A: gmail_tools._list_recent must request bulk headers ────


class TestListRecentRequestsBulkHeaders:
    """Source-grep contract: the metadataHeaders list MUST include the
    bulk-marker headers, otherwise no amount of scorer-side weight
    will fire."""

    def test_metadataheaders_includes_list_unsubscribe(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "tools" / "gmail_tools.py"
        ).read_text(encoding="utf-8")
        # The List-Unsubscribe header is the strongest bulk signal
        # (RFC 2369). It MUST appear in the metadataHeaders request.
        assert '"List-Unsubscribe"' in src, (
            "_list_recent must request List-Unsubscribe — without it "
            "the scorer is blind to RFC 2369 bulk-mail signal"
        )

    def test_metadataheaders_includes_list_id(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "tools" / "gmail_tools.py"
        ).read_text(encoding="utf-8")
        assert '"List-Id"' in src

    def test_metadataheaders_includes_auto_submitted_and_precedence(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "tools" / "gmail_tools.py"
        ).read_text(encoding="utf-8")
        assert '"Auto-Submitted"' in src
        assert '"Precedence"' in src

    def test_stub_includes_bulk_marker_fields(self) -> None:
        """The stub returned by _list_recent must surface the new
        headers as fields the scorer expects."""
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "tools" / "gmail_tools.py"
        ).read_text(encoding="utf-8")
        for field in (
            '"list_unsubscribe":', '"list_id":',
            '"auto_submitted":', '"precedence":',
            '"in_reply_to":', '"references":',
        ):
            assert field in src, f"_list_recent stub missing field: {field}"


# ── Layer B: scorer + EmailHeaders Gmail-label awareness ───────────


class TestEmailHeadersHasGmailLabels:

    def test_gmail_labels_field_exists(self) -> None:
        h = EmailHeaders(gmail_labels=("CATEGORY_PROMOTIONS", "UNREAD"))
        assert h.gmail_labels == ("CATEGORY_PROMOTIONS", "UNREAD")

    def test_default_is_empty_tuple(self) -> None:
        """Non-Gmail providers (IMAP) leave gmail_labels empty;
        the scorer must tolerate that without raising."""
        h = EmailHeaders()
        assert h.gmail_labels == ()


class TestPenaltyTable:

    def test_promotions_is_strongest_penalty(self) -> None:
        """CATEGORY_PROMOTIONS must outweigh +2.5 of personal-marker
        noise so a clear marketing email never crosses the 1.0
        threshold."""
        assert _GMAIL_BULK_LABEL_PENALTIES["CATEGORY_PROMOTIONS"] <= -4

    def test_all_four_categories_present(self) -> None:
        for label in (
            "CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL",
            "CATEGORY_UPDATES",    "CATEGORY_FORUMS",
        ):
            assert label in _GMAIL_BULK_LABEL_PENALTIES


# ── Functional: the actual operator-reported emails no longer surface ──


def _marketing_headers(
    *, from_: str, subject: str, gmail_labels: tuple[str, ...] = (),
    list_unsubscribe: str | None = None,
) -> EmailHeaders:
    """Build a marketing-shaped EmailHeaders with optional bulk
    markers — mimics what the FIXED pipeline would deliver."""
    return EmailHeaders(
        from_=from_,
        to="user@example.com",
        cc="",
        subject=subject,
        list_unsubscribe=list_unsubscribe,
        gmail_labels=gmail_labels,
        unread=True,
        date=datetime.now(timezone.utc) - timedelta(hours=1),
    )


class TestActualOperatorReportedEmails:
    """The three emails from the 2026-05-10 alert must score below
    the 1.0 urgency threshold under the post-fix scorer."""

    def test_dailyom_with_promotions_label(self) -> None:
        h = _marketing_headers(
            from_="DailyOM <today@dailyom.com>",
            subject="Last Chance: The Seven Strengths Free Live Online Event",
            gmail_labels=("CATEGORY_PROMOTIONS", "INBOX", "UNREAD"),
        )
        result = score_email(h, user_address="user@example.com")
        assert result.score < 1.0, (
            f"DailyOM marketing email must score below 1.0; "
            f"got {result.score} with reasons={result.reasons}"
        )

    def test_wildgym_with_promotions_label(self) -> None:
        h = _marketing_headers(
            from_="Wild Gym <info@wildgym.com>",
            subject="Your bones need a reason to stay strong",
            gmail_labels=("CATEGORY_PROMOTIONS", "INBOX", "UNREAD"),
        )
        result = score_email(h, user_address="user@example.com")
        assert result.score < 1.0, (
            f"Wild Gym marketing email must score below 1.0; "
            f"got {result.score} with reasons={result.reasons}"
        )

    def test_swimmer_with_promotions_label(self) -> None:
        h = _marketing_headers(
            from_='"Swimmer.com.au" <news@swimmer.com.au>',
            subject="💦 The Ultimate in Swim Backpacks",
            gmail_labels=("CATEGORY_PROMOTIONS", "INBOX", "UNREAD"),
        )
        result = score_email(h, user_address="user@example.com")
        assert result.score < 1.0


class TestListUnsubscribeAlone:
    """Even without Gmail labels, a List-Unsubscribe header (RFC 2369)
    must penalize sufficiently — IMAP / Outlook users get this path."""

    def test_list_unsubscribe_drops_below_threshold(self) -> None:
        h = _marketing_headers(
            from_="Marketing Team <hello@brand.com>",
            subject="Big Spring Sale Inside",
            list_unsubscribe="<https://brand.com/u>",
            # No gmail_labels — pretend non-Gmail provider.
        )
        result = score_email(h, user_address="user@example.com")
        # -3 (List-Unsubscribe) + 1 (human From) + 1 (unread) +
        # 0.5 (recent) - 1 (marketing keyword "sale") = -1.5
        assert result.score < 1.0


class TestNoOverPenalizationOnRealMail:
    """Negative test — a genuine personal email must NOT be tagged
    as bulk just because the user has the wrong gmail_labels mix."""

    def test_human_personal_email_above_threshold(self) -> None:
        h = EmailHeaders(
            from_="Alice Smith <alice@example.com>",
            to="user@example.com",
            cc="",
            subject="Quick question about Tuesday",
            list_unsubscribe=None,
            list_id=None,
            gmail_labels=("INBOX", "UNREAD"),  # no CATEGORY_*
            unread=True,
            date=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        result = score_email(h, user_address="user@example.com")
        # +1 direct To + 1 human From + 1 unread + 0.5 recent = 3.5
        assert result.score >= 1.0, (
            f"genuine personal email must clear threshold; got {result.score}"
        )

    def test_no_double_count_when_multiple_categories(self) -> None:
        """When Gmail tags a message with BOTH Promotions and Updates,
        we apply the strongest penalty once — not the sum.  Otherwise
        the operator could see a -8 score from two overlapping labels
        which would also penalize legitimate Updates (e.g. flight
        confirmations, banking)."""
        h = EmailHeaders(
            gmail_labels=("CATEGORY_PROMOTIONS", "CATEGORY_UPDATES"),
        )
        result = score_email(h, user_address="user@example.com")
        # Should be exactly the PROMOTIONS penalty (-4), not -6.
        assert result.score == -4
        # And the reason should name the chosen label.
        assert any("CATEGORY_PROMOTIONS" in r for r in result.reasons)


# ── Email monitor build_headers wiring ────────────────────────────


class TestBuildHeadersWiresThroughBulkMarkers:
    """Source-grep contract: ``_build_headers`` must populate every
    bulk-marker field from the stub, not hardcode None.  Pre-fix
    this hardcoding made every Gmail-fetched email look bulk-marker-
    free regardless of what the API returned."""

    def test_no_hardcoded_none_for_bulk_markers(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "life_companion" / "email_monitor.py"
        ).read_text(encoding="utf-8")
        # Slice the function.
        idx_start = src.index("def _build_headers")
        idx_end = src.index("\ndef ", idx_start + 50)
        body = src[idx_start:idx_end]
        # The pre-fix anti-pattern: a series of literal Nones for the
        # bulk-marker fields.  Lock that out.
        forbidden_lines = [
            "list_unsubscribe=None",
            "list_id=None",
            "auto_submitted=None",
            "precedence=None",
            "in_reply_to=None",
            "references=None",
        ]
        for line in forbidden_lines:
            assert line not in body, (
                f"{line!r} hardcoded in _build_headers — bulk-marker "
                f"signal lost"
            )

    def test_gmail_labels_passed_through(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "life_companion" / "email_monitor.py"
        ).read_text(encoding="utf-8")
        idx_start = src.index("def _build_headers")
        idx_end = src.index("\ndef ", idx_start + 50)
        body = src[idx_start:idx_end]
        assert "gmail_labels=" in body, (
            "_build_headers must populate gmail_labels from the stub"
        )
