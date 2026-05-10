"""Single source of truth for life-companion features + tunables.

The React /cp/life-companion control panel consumes this registry to
render its on/off + tunable-edit UI; the runtime-settings override
machinery (``app.runtime_settings.life_companion_*``) reads it to
validate keys; the per-module ``run()`` functions read tunables
through helpers that consult overrides first, env second, registry
default last.

Why a registry (not module-level constants)?

  * Cross-cutting use: the React page and the runtime-settings
    accessors both need the same shape; declaring it once removes
    "where does the UI know about LIFE_COMPANION_ACT_NOW_TOP_K?"
    from the answer.

  * Schema-as-data: tunable types + bounds + defaults travel with
    the metadata, so the UI can render an int-input with min/max
    instead of a freeform text field, and the override-setter can
    typecheck before persisting.

  * Discoverable: a new feature just adds an entry here. The
    UI picks it up automatically on next page load (registry
    is read at request time).

This module has NO behavior — only data. Adding logic here would
re-create the coupling we're trying to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# Tunable types the UI knows how to render + validate.
TunableType = Literal["int", "float", "str", "bool", "secs", "minutes", "hours"]


@dataclass(frozen=True)
class Tunable:
    """One env-var-shaped knob a feature exposes."""
    env_key: str           # canonical env var ("LIFE_COMPANION_ACT_NOW_TOP_K")
    label: str             # short UI label
    description: str       # help text the UI shows under the input
    type: TunableType      # for input rendering + validation
    default: Any           # baseline default the registry advertises
    min: float | None = None       # optional clamp for int/float
    max: float | None = None
    options: tuple[str, ...] = ()  # only for type="str" — render dropdown


@dataclass(frozen=True)
class Feature:
    """One life-companion job + its on/off toggle + tunables."""
    key: str               # short feature key — what feature_enabled() takes
    name: str              # UI title
    description: str       # 1-2 sentence purpose
    feature_env_key: str   # the env var that toggles it
    job_name: str          # the idle-scheduler job name (in get_idle_jobs)
    default_enabled: bool = True
    tunables: tuple[Tunable, ...] = field(default_factory=tuple)


# ── Registry ────────────────────────────────────────────────────────


# Order matters — UI cards render in this order.
FEATURES: tuple[Feature, ...] = (
    Feature(
        key="email",
        name="Email monitor (real-time)",
        description=(
            "Triages unread inbox every ~10 min using a heuristic "
            "scorer (no LLM). Surfaces top-3 above a threshold to "
            "Signal. Pairs with the act-now digest below."
        ),
        feature_env_key="LIFE_COMPANION_EMAIL_ENABLED",
        job_name="life-companion-email",
        tunables=(
            Tunable(
                env_key="LIFE_COMPANION_EMAIL_CHECK_MIN",
                label="Check interval (min)",
                description="How often to scan unread inbox.",
                type="minutes", default=10, min=2, max=120,
            ),
            Tunable(
                env_key="LIFE_COMPANION_EMAIL_URGENCY_THRESHOLD",
                label="Urgency threshold",
                description=(
                    "Score floor for surfacing. The heuristic typically "
                    "produces -7..+10 — 1.0 catches genuine action items "
                    "while filtering bulk."
                ),
                type="float", default=1.0, min=-5, max=10,
            ),
        ),
    ),
    Feature(
        key="act_now_digest",
        name="Act-now digest (LLM-graded)",
        description=(
            "Thrice-daily synthesis of the last 48 h unread inbox "
            "via Sonnet. Surfaces top emails the user must ACT on now "
            "with why / action / Gmail link. Sibling of email monitor."
        ),
        feature_env_key="LIFE_COMPANION_ACT_NOW_DIGEST_ENABLED",
        job_name="life-companion-act-now-digest",
        tunables=(
            Tunable(
                env_key="LIFE_COMPANION_ACT_NOW_TOP_K",
                label="Top K",
                description="Max items in each digest.",
                type="int", default=7, min=1, max=20,
            ),
            Tunable(
                env_key="LIFE_COMPANION_ACT_NOW_LOOKBACK_HOURS",
                label="Lookback (hours)",
                description="How far back to search unread inbox.",
                type="hours", default=48, min=6, max=168,
            ),
            Tunable(
                env_key="LIFE_COMPANION_ACT_NOW_MAX_CANDIDATES",
                label="Max candidates",
                description=(
                    "Hard cap on emails sent to the LLM (cost + latency)."
                ),
                type="int", default=30, min=5, max=100,
            ),
            Tunable(
                env_key="LIFE_COMPANION_ACT_NOW_BODY_CHARS",
                label="Body excerpt (chars)",
                description="Truncation for each email body in the prompt.",
                type="int", default=500, min=100, max=2000,
            ),
        ),
    ),
    Feature(
        key="briefing",
        name="Daily briefing",
        description=(
            "Morning / evening / weekly digest synthesizing calendar + "
            "email + tickets + companion ideas."
        ),
        feature_env_key="LIFE_COMPANION_BRIEFING_ENABLED",
        job_name="life-companion-briefing",
        tunables=(
            Tunable(
                env_key="LIFE_COMPANION_BRIEFING_MORNING",
                label="Morning time (HH:MM)",
                description="Local-clock time for the morning briefing.",
                type="str", default="07:00",
            ),
            Tunable(
                env_key="LIFE_COMPANION_BRIEFING_EVENING",
                label="Evening time (HH:MM)",
                description="Local-clock time for the evening briefing.",
                type="str", default="20:00",
            ),
        ),
    ),
    Feature(
        key="routines",
        name="Routine detector",
        description=(
            "Surfaces day-of-week + time-of-day patterns from the affect "
            "trace. Nudges 30 min before recurring activities."
        ),
        feature_env_key="LIFE_COMPANION_ROUTINES_ENABLED",
        job_name="life-companion-routines",
    ),
    Feature(
        key="long_arc",
        name="Long-arc commitment follow-up",
        description=(
            "Multi-week commitment tracker. Pings about open follow-ups."
        ),
        feature_env_key="LIFE_COMPANION_LONG_ARC_ENABLED",
        job_name="life-companion-long-arc",
    ),
    Feature(
        key="calendar_prep",
        name="Calendar prep",
        description="30-min pre-meeting briefing with attendees + context.",
        feature_env_key="LIFE_COMPANION_CALENDAR_PREP_ENABLED",
        job_name="life-companion-calendar-prep",
    ),
    Feature(
        key="personalized_digest",
        name="Personalized weekly digest",
        description="RSS / GitHub / arXiv personalized weekly summary.",
        feature_env_key="LIFE_COMPANION_PERSONALIZED_DIGEST_ENABLED",
        job_name="life-companion-personalized-digest",
    ),
    Feature(
        key="calendar_horizon",
        name="72 h calendar horizon",
        description="Daily scan for upcoming conflicts + prep notes.",
        feature_env_key="LIFE_COMPANION_CALENDAR_HORIZON_ENABLED",
        job_name="life-companion-calendar-horizon",
    ),
    Feature(
        key="topic_dormancy",
        name="Topic dormancy",
        description="Long-arc nudge when a topic goes silent for too long.",
        feature_env_key="LIFE_COMPANION_TOPIC_DORMANCY_ENABLED",
        job_name="life-companion-topic-dormancy",
    ),
    Feature(
        key="seasonal_nudges",
        name="Seasonal nudges (Finland)",
        description="Sauna-season / berry-season / sunlight reminders.",
        feature_env_key="LIFE_COMPANION_SEASONAL_NUDGES_ENABLED",
        job_name="life-companion-seasonal-nudges",
    ),
)


# ── Lookups ─────────────────────────────────────────────────────────


_BY_KEY: dict[str, Feature] = {f.key: f for f in FEATURES}


def get_feature(key: str) -> Feature | None:
    """Return the feature definition for ``key``, or None."""
    return _BY_KEY.get(key)


def list_features() -> tuple[Feature, ...]:
    """Tuple of every registered feature in display order."""
    return FEATURES


def find_tunable(env_key: str) -> tuple[Feature, Tunable] | None:
    """Reverse-lookup a tunable by its env-var name."""
    for feat in FEATURES:
        for tun in feat.tunables:
            if tun.env_key == env_key:
                return feat, tun
    return None


__all__ = [
    "Feature", "Tunable", "FEATURES",
    "get_feature", "list_features", "find_tunable",
]
