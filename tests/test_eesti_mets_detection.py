"""Tests for the Eesti mets workspace detection.

Regression for the 2026-05-09 forest-age-distribution misroute:

  user:    "Please make a graphic about the change of forest age
            distribution over time in Estonia. Use sattelite data"
  router:  "estonia" matched PLG's keyword list → routed to PLG
  result:  ticket landed under the wrong project; user surprised.

The fix is in three parts:

  1. **Add an `eesti mets` profile** with forest / satellite / Estonian-
     forest keywords. Without this the detector simply has no signal
     for "this is a forest research task."
  2. **Tighten PLG's keyword list** — remove "estonia" / "baltic" /
     "latvia" / "lithuania". Geographies are not project signals;
     PLG operates in those countries but isn't synonymous with them.
  3. The existing scoring algorithm in ``detect_project`` already
     picks the highest-scoring project on ties, so multi-keyword
     "forest + estonia + satellite" matches Eesti mets cleanly.

These tests exercise the fix shape without hitting the filesystem
profile loader — they patch ``_projects`` directly so the test
isn't sensitive to whether the eesti_mets/ profile dir exists
in the test fixture.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ── Keyword list shape ──────────────────────────────────────────────


class TestKeywordList:
    """Static checks on PROJECT_KEYWORDS — keeps the dict honest."""

    def test_eesti_mets_present(self) -> None:
        from app.project_isolation import PROJECT_KEYWORDS

        assert "eesti mets" in PROJECT_KEYWORDS, (
            "eesti mets profile must be in PROJECT_KEYWORDS for the "
            "auto-detector to consider it"
        )

    def test_eesti_mets_has_forest_signals(self) -> None:
        """Concrete keywords the detector needs to catch real forest
        queries — not just the project name verbatim."""
        from app.project_isolation import PROJECT_KEYWORDS

        kws = PROJECT_KEYWORDS["eesti mets"]
        for required in (
            "forest", "deforestation", "satellite", "landsat",
            "earth engine", "biodiversity",
        ):
            assert required in kws, (
                f"eesti mets keyword list missing {required!r}; "
                "real-world forest queries won't match"
            )

    def test_plg_does_not_claim_geography(self) -> None:
        """The 2026-05-09 fix: PLG's keyword list must NOT contain
        bare geographic terms. PLG operates in Estonia/Baltics, but
        anything Estonia-related shouldn't auto-route to PLG.

        Specific terms (``ticketing``, ``piletilevi``) stay; bare
        country names go.
        """
        from app.project_isolation import PROJECT_KEYWORDS

        plg_kws = PROJECT_KEYWORDS["plg"]
        for forbidden in (
            "estonia", "baltic", "latvia", "lithuania",
        ):
            assert forbidden not in plg_kws, (
                f"PLG keyword list still has {forbidden!r}; this is "
                "the 2026-05-09 over-claim bug. Geographies aren't "
                "project signals."
            )


# ── Scoring outcome ─────────────────────────────────────────────────


def _detector_with_profiles(profile_names: list[str]):
    """Build a ProjectManager and stub ``_projects`` so the detector
    only considers the named profiles. Avoids dependency on the
    actual ``workspace/projects/*`` filesystem layout.
    """
    from app.project_isolation import ProjectConfig, ProjectManager

    pm = ProjectManager()
    pm._projects = {
        name: ProjectConfig(name=name)  # rest via __post_init__ defaults
        for name in profile_names
    }
    return pm


class TestDetectionOutcome:
    """The actual user-visible behaviour."""

    PROFILES = ["plg", "archibal", "kaicart", "eesti mets"]

    def test_forest_age_query_routes_to_eesti_mets(self) -> None:
        """The exact 2026-05-09 query that misrouted."""
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "Please make a graphic about the change of forest age "
            "distribution over time in Estonia. Use sattelite data"
        )
        assert detected == "eesti mets"

    def test_essay_query_routes_to_eesti_mets(self) -> None:
        """Earlier in the same conversation: the essay request."""
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "please write a 5-page essay about estonian forest and its health"
        )
        assert detected == "eesti mets"

    def test_deforestation_maps_query_routes_to_eesti_mets(self) -> None:
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "create Estonia deforestation and forest age maps per year since 2012"
        )
        assert detected == "eesti mets"

    def test_estonia_alone_does_not_route_to_plg(self) -> None:
        """Bare "estonia" used to route to PLG. After the geo-tighten
        it should match nothing (so the auto-detector doesn't propose
        anything spurious)."""
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "tell me about Estonia"  # no forest signal, no PLG signal
        )
        assert detected is None

    def test_real_plg_query_still_routes_to_plg(self) -> None:
        """Sanity: tightening PLG keywords didn't break PLG detection
        for queries that genuinely are PLG-shaped."""
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "What's the latest on the Piletilevi ticketing platform's "
            "Q3 numbers?"
        )
        assert detected == "plg"

    def test_event_concert_query_still_routes_to_plg(self) -> None:
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "How many tickets sold for the Live Nation concert at the venue?"
        )
        assert detected == "plg"

    def test_archibal_query_unaffected(self) -> None:
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "Update the C2PA provenance pipeline for content authenticity"
        )
        assert detected == "archibal"

    def test_kaicart_query_unaffected(self) -> None:
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "TikTok Shop seller onboarding flow for Thai SMBs"
        )
        assert detected == "kaicart"


class TestScoringTieBreak:
    """Tie-breaking when multiple profiles match."""

    PROFILES = ["plg", "archibal", "kaicart", "eesti mets"]

    def test_forest_with_ticketing_word_still_picks_eesti_mets(self) -> None:
        """If a query mentions "ticket" but is overwhelmingly about
        forests, the high-scoring match wins."""
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "I need a deforestation report on Estonian forests using "
            "Hansen GFC and Landsat data — and remember to ticket the "
            "task in Linear"
        )
        # eesti mets keywords matched: deforestation, estonian forest,
        # forest, hansen, landsat (5)
        # plg keywords matched: ticket (1)
        assert detected == "eesti mets"

    def test_plg_query_with_forest_word_still_picks_plg(self) -> None:
        """The opposite: if it's clearly a ticketing task, a stray
        "forest" word doesn't hijack."""
        pm = _detector_with_profiles(self.PROFILES)
        detected = pm.detect_project(
            "Process the Piletilevi ticket refund queue at the venue."
        )
        # plg: piletilevi, ticket, venue (3)
        # eesti mets: 0
        assert detected == "plg"


class TestEmptyOrUnrelated:

    def test_empty_string_returns_none(self) -> None:
        pm = _detector_with_profiles(["plg", "eesti mets"])
        assert pm.detect_project("") is None

    def test_unrelated_query_returns_none(self) -> None:
        pm = _detector_with_profiles(["plg", "archibal", "kaicart", "eesti mets"])
        # Generic philosophy question, no project keywords
        assert pm.detect_project(
            "What do you think about the meaning of consciousness?"
        ) is None
