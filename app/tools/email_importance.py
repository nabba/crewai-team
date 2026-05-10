"""
email_importance.py — Heuristic importance scorer for inbox triage.

Why this exists
---------------
The PIM agent's ``check_email`` returns emails by recency. When the
user asks "rank my top 25 most important emails", the agent has no
better signal than "newest first" — which surfaces marketing blasts
and notification noise above genuinely-important messages.

This module adds a fast, explainable, LLM-free scorer over standard
email headers + envelope state. The signals are well-known inbox-
triage primitives:

  * **Bulk markers** are negative — ``List-Unsubscribe`` / ``List-ID``
    / ``Auto-Submitted`` / ``Precedence: bulk`` / ``noreply@`` senders.
    Marketing copy in the subject (``%``, ``OFF``, ``deal``, emojis)
    adds further deductions.

  * **Personal markers** are positive — direct ``To:`` (vs Cc/Bcc),
    threaded reply (``In-Reply-To`` / ``References``), human-form
    senders, action verbs in the subject (``urgent``, ``review``,
    ``deadline``).

  * **Allowlist** (``EMAIL_IMPORTANT_SENDERS`` env) provides a
    user-curated upweight that overrides the heuristic when the
    operator has named-someone-known.

  * **State** (unread, recent) provides a small final tiebreak.

The scorer returns ``(score, reasons)`` so the rank_emails tool can
present *why* a message ranked where it did — non-magical, auditable,
and tunable without retraining anything.

Heuristic, not ML — kept explainable on purpose. The agent can still
ask the user to refine the allowlist or override.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Score weights ────────────────────────────────────────────────────
#
# Tuned conservatively: a personal email from someone known scores
# ~+10; a marketing blast scores ~-7. Five-bucket separation lets
# downstream display group results into clear tiers.

# Bulk / marketing penalties
_BULK_LIST_UNSUBSCRIBE = -3
_BULK_LIST_ID          = -2
_BULK_AUTO_SUBMITTED   = -2
_BULK_PRECEDENCE_BULK  = -2
_BULK_NOREPLY_SENDER   = -2

# Gmail's tab-categorization labels.  When the user has Gmail's
# Promotions / Social / Updates / Forums tabs enabled, the server
# auto-labels bulk mail with these category labels — that's
# Google's own ML telling us "this is a marketing email" with
# higher precision than any single-header rule we can encode here.
# Pre-2026-05-10 the email_monitor ignored these labels entirely
# (the scorer didn't even have a field for gmail_labels), which is
# why marketing emails surfaced with score=2.5: the bulk-marker
# weights above couldn't fire because metadataHeaders=["From",
# "Subject", "Date"] never requested List-Unsubscribe et al.,
# while the labels were in the stub but unread.  Penalty values
# tuned so that a clear Promotions tag overrides "human From +
# unread + recent" (which sums to +2.5).
_BULK_CATEGORY_PROMOTIONS = -4
_BULK_CATEGORY_SOCIAL     = -3
_BULK_CATEGORY_UPDATES    = -2
_BULK_CATEGORY_FORUMS     = -2

_GMAIL_BULK_LABEL_PENALTIES: dict[str, int] = {
    "CATEGORY_PROMOTIONS": _BULK_CATEGORY_PROMOTIONS,
    "CATEGORY_SOCIAL":     _BULK_CATEGORY_SOCIAL,
    "CATEGORY_UPDATES":    _BULK_CATEGORY_UPDATES,
    "CATEGORY_FORUMS":     _BULK_CATEGORY_FORUMS,
}

# Per-marketing-keyword in subject (capped at _MARKETING_KEYWORD_CAP)
_BULK_MARKETING_KEYWORD = -1
_MARKETING_KEYWORD_CAP  = -3   # i.e. cap penalty at -3 even if 5 keywords match

# Personal upweights
_PERSONAL_DIRECT_TO    = 1
_PERSONAL_THREADED     = 2
_PERSONAL_HUMAN_FROM   = 1     # name + address ("Alice <alice@...>"), not "Notifications <noreply@...>"

# Per-action-keyword in subject (capped at _ACTION_KEYWORD_CAP)
_PERSONAL_ACTION_KEYWORD = 1
_ACTION_KEYWORD_CAP      = 3

# Allowlist (env-curated)
_ALLOWLIST_HIT = 5

# State signals
_STATE_UNREAD          = 1
_STATE_RECENT_HOURS    = 6
_STATE_RECENT_BONUS    = 0.5


# ── Marker dictionaries ──────────────────────────────────────────────

_NOREPLY_RE = re.compile(
    r"\b(?:no[-_.]?reply|donotreply|notifications?|alerts?|mailer-daemon|"
    r"automated|auto[-_.]?reply|do[-_.]?not[-_.]?reply)\b@",
    re.IGNORECASE,
)

# Marketing tells. Kept lowercase; matching is case-insensitive substring.
_MARKETING_KEYWORDS: tuple[str, ...] = (
    "% off", "%off",
    " off ", " off!", " off:",
    "discount", "deal", "deals",
    "sale", "saletime", "buy now", "shop now",
    "limited time", "limited-time", "only today", "today only", "ends today",
    "ends tonight", "expires", "exclusive", "save big", "save up to",
    "free shipping", "free trial", "subscribe", "newsletter",
    "💥", "🎉", "🛒", "🎁", "⚡", "🌸", "🎙️",  # emoji-heavy subjects
    "preorder", "presale", "early access", "members only",
    "vip", "click here",
)

# Action / urgency tells. The agent should bubble these up.
_ACTION_KEYWORDS: tuple[str, ...] = (
    "urgent", "asap", "action required", "action needed", "please review",
    "please approve", "approval needed", "approval required", "review needed",
    "deadline", "due date", "overdue", "follow up", "follow-up",
    "kindly", "can you", "could you", "would you", "please",
    "confirm", "confirmation", "important",
    "invoice", "payment", "receipt", "contract",
    "meeting", "call scheduled", "interview",
    # Estonian/Finnish parallels (user is in Helsinki/Tallinn)
    "kiire", "tähtaeg", "palun kinnita", "vajalik tegevus",
)


# ── Public types ─────────────────────────────────────────────────────

@dataclass
class EmailHeaders:
    """Lightweight envelope/header bundle the scorer accepts.

    This shape decouples the scorer from imaplib's quirks — the email
    tool builds one of these per message after fetching headers.

    Fields use the canonical RFC822 names lowercased; everything is a
    string (or None when the header was absent). ``unread`` and
    ``date`` are extracted from envelope/flags rather than headers.
    """
    from_: str = ""              # raw From header (e.g. "Alice <alice@x.com>")
    to: str = ""
    cc: str = ""
    subject: str = ""
    list_unsubscribe: str | None = None
    list_id: str | None = None
    auto_submitted: str | None = None
    precedence: str | None = None
    in_reply_to: str | None = None
    references: str | None = None
    date: datetime | None = None
    unread: bool = False
    # Gmail tab-category labels (CATEGORY_PROMOTIONS / SOCIAL /
    # UPDATES / FORUMS) when fetched via Gmail API.  Empty tuple
    # for non-Gmail providers (IMAP / Outlook / etc.) where the
    # bulk-marker headers above carry the signal instead.
    gmail_labels: tuple[str, ...] = ()


@dataclass
class EmailScore:
    """Result of scoring one email. ``reasons`` is the audit trail
    (one short string per signal that fired)."""
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, weight: float, label: str) -> None:
        if weight == 0:
            return
        self.score += weight
        sign = "+" if weight > 0 else ""
        self.reasons.append(f"{sign}{weight:g} {label}")


# ── Scorer ───────────────────────────────────────────────────────────

def _parse_allowlist(raw: str) -> tuple[str, ...]:
    """Split env-style allowlist into lowercase tokens."""
    if not raw:
        return ()
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


def _matches_allowlist(from_: str, allowlist: tuple[str, ...]) -> str | None:
    """Return the matching allowlist token, or None. Substring match
    on the lowercased From header (so a domain like ``@acme.com``
    matches any sender at that domain)."""
    if not from_ or not allowlist:
        return None
    needle = from_.lower()
    for token in allowlist:
        if token in needle:
            return token
    return None


def _is_human_from(from_: str) -> bool:
    """Return True if the From header looks like a human (a name part
    plus an address) rather than a bot. Examples:

        "Alice Smith <alice@example.com>"  → True
        "Marketing Team <noreply@x.com>"   → False (noreply addr)
        "alice@example.com"                → False (no display name)
        "Notifications <notify@x.com>"     → False (bot-shaped name)
    """
    if not from_:
        return False
    if _NOREPLY_RE.search(from_):
        return False
    # Need both a display name and an address-like fragment.
    if "<" not in from_ or ">" not in from_:
        return False
    name_part = from_.split("<", 1)[0].strip().strip('"')
    if not name_part:
        return False
    bot_names = (
        "notification", "notifications", "alert", "alerts",
        "noreply", "no-reply", "do-not-reply", "automated",
    )
    name_lower = name_part.lower()
    return not any(b in name_lower for b in bot_names)


def _direct_to_user(to: str, cc: str, user_address: str) -> bool:
    """True if user_address appears in To: (not Cc:). Cc-only delivery
    means the message is informational — usually less urgent than mail
    addressed directly to you."""
    if not user_address or not to:
        return False
    needle = user_address.lower()
    return needle in to.lower() and (not cc or needle not in cc.lower())


def _count_keyword_hits(haystack: str, keywords: Iterable[str]) -> int:
    if not haystack:
        return 0
    hl = haystack.lower()
    return sum(1 for k in keywords if k in hl)


def score_email(
    headers: EmailHeaders,
    *,
    user_address: str,
    important_senders: tuple[str, ...] = (),
    now: datetime | None = None,
) -> EmailScore:
    """Score one email by composable signals. Returns explainable result.

    ``user_address`` is the operator's own email — used to detect
    direct-to-user vs bcc/cc-only delivery.

    ``important_senders`` is the parsed allowlist (use
    :func:`_parse_allowlist` on the env string).

    ``now`` is injectable for tests; defaults to ``datetime.now(UTC)``.
    """
    out = EmailScore()
    from_ = headers.from_ or ""
    subject = headers.subject or ""

    # ── Bulk markers ──
    if headers.list_unsubscribe:
        out.add(_BULK_LIST_UNSUBSCRIBE, "List-Unsubscribe present (bulk)")
    if headers.list_id:
        out.add(_BULK_LIST_ID, "List-ID present (mailing list)")
    if headers.auto_submitted and headers.auto_submitted.lower() not in ("no", ""):
        out.add(_BULK_AUTO_SUBMITTED, "Auto-Submitted (system-generated)")
    if headers.precedence and headers.precedence.lower() in ("bulk", "list", "junk"):
        out.add(_BULK_PRECEDENCE_BULK, f"Precedence: {headers.precedence.lower()}")
    if _NOREPLY_RE.search(from_):
        out.add(_BULK_NOREPLY_SENDER, "noreply-style sender")

    # Gmail tab-category labels — Google's own ML "this is bulk".
    # We apply the strongest penalty in this group (a message tagged
    # CATEGORY_PROMOTIONS may also have CATEGORY_UPDATES, but the
    # promo tag is the more reliable bulk signal).
    gmail_label_set = {str(label) for label in (headers.gmail_labels or ())}
    matching_penalties = [
        (label, _GMAIL_BULK_LABEL_PENALTIES[label])
        for label in gmail_label_set
        if label in _GMAIL_BULK_LABEL_PENALTIES
    ]
    if matching_penalties:
        # Pick the most negative (strongest) — don't compound, the
        # categories overlap and double-counting would over-penalize.
        worst_label, worst_penalty = min(matching_penalties, key=lambda lp: lp[1])
        out.add(worst_penalty, f"Gmail label: {worst_label}")

    # Marketing keywords in subject — capped to prevent runaway penalty
    mk_hits = _count_keyword_hits(subject, _MARKETING_KEYWORDS)
    if mk_hits:
        penalty = max(_MARKETING_KEYWORD_CAP, _BULK_MARKETING_KEYWORD * mk_hits)
        out.add(penalty, f"marketing keywords in subject ({mk_hits})")

    # ── Personal markers ──
    if _direct_to_user(headers.to, headers.cc, user_address):
        out.add(_PERSONAL_DIRECT_TO, "direct To: (not Cc only)")
    if headers.in_reply_to or headers.references:
        out.add(_PERSONAL_THREADED, "thread reply (In-Reply-To/References)")
    if _is_human_from(from_):
        out.add(_PERSONAL_HUMAN_FROM, "human sender (display name + address)")

    # Action keywords — capped
    ak_hits = _count_keyword_hits(subject, _ACTION_KEYWORDS)
    if ak_hits:
        bonus = min(_ACTION_KEYWORD_CAP, _PERSONAL_ACTION_KEYWORD * ak_hits)
        out.add(bonus, f"action keywords in subject ({ak_hits})")

    # ── Allowlist ──
    hit = _matches_allowlist(from_, important_senders)
    if hit:
        out.add(_ALLOWLIST_HIT, f"allowlisted sender ({hit})")

    # ── State ──
    if headers.unread:
        out.add(_STATE_UNREAD, "unread")
    if headers.date is not None:
        ref = now or datetime.now(timezone.utc)
        # Normalise both sides to UTC-aware
        try:
            d = headers.date
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            age_hours = (ref - d).total_seconds() / 3600.0
            if 0 <= age_hours <= _STATE_RECENT_HOURS:
                out.add(_STATE_RECENT_BONUS, f"recent (<{_STATE_RECENT_HOURS}h old)")
        except Exception:
            pass

    return out
