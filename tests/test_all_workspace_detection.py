"""Auto-detect coverage tests for ALL four workspaces.

Operator request 2026-05-09 (after the eesti_mets fix landed):

    "I need auto detect for other workspaces as well — if it is asian
     e-commerce suggestion must be to move to KaiCart, if it is AI
     provenance suggestions must be Archibal, if it is ticketing,
     suggestion must be PLG."

The infrastructure (3-mode auto-detect → propose-via-Signal flow in
``app/main.py:1600+``) was already in place; only the keyword lists
were too narrow. This file pins down the expanded keyword lists so
each project catches its natural domain vocabulary.

Tests stub ``_projects`` directly so the real workspace/projects/
dir layout doesn't matter.
"""
from __future__ import annotations

import pytest


def _detector_with_all_profiles():
    from app.project_isolation import ProjectConfig, ProjectManager

    pm = ProjectManager()
    pm._projects = {
        name: ProjectConfig(name=name)
        for name in ("plg", "archibal", "kaicart", "eesti mets")
    }
    return pm


# ── KaiCart: Asian e-commerce ───────────────────────────────────────


class TestKaiCartDetection:
    """Operator's spec: 'if it is asian e-commerce suggestion must be
    to move to KaiCart'."""

    def test_asian_ecommerce_directly(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "What's the latest on the Asian e-commerce market?",
            "I'm researching Southeast Asia e-commerce trends",
            "ASEAN commerce regulations update",
            "How is the SEA market performing?",
        ):
            assert pm.detect_project(q) == "kaicart", f"missed: {q!r}"

    def test_specific_sea_countries(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "TikTok Shop sellers in Vietnam",
            "Indonesian merchants on Lazada",
            "Filipino fulfilment partners",
            "Malaysian online seller onboarding",
            "Singapore marketplace integration",
            "Thai SMB seller acquisition flow",
        ):
            assert pm.detect_project(q) == "kaicart", f"missed: {q!r}"

    def test_partner_platforms(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "Shopee fulfilment integration",
            "Lazada seller migration",
            "Shopify merchant analytics",
            "TikTok Shop seller onboarding",
        ):
            assert pm.detect_project(q) == "kaicart", f"missed: {q!r}"

    def test_generic_marketplace_terms(self) -> None:
        """KaiCart is the only e-commerce project, so it can claim
        generic marketplace vocabulary safely."""
        pm = _detector_with_all_profiles()
        for q in (
            "Online marketplace conversion rate",
            "Live commerce strategy for small sellers",
            "Drop-shipping vs fulfilment trade-offs",
            "Social commerce growth metrics",
        ):
            assert pm.detect_project(q) == "kaicart", f"missed: {q!r}"


# ── Archibal: AI provenance ─────────────────────────────────────────


class TestArchibalDetection:
    """Operator's spec: 'if it is AI provenance suggestions must be
    Archibal'."""

    def test_ai_provenance_directly(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "Update the AI provenance pipeline",
            "Implement digital provenance tracking for media",
            "How do we add content provenance to the upload flow?",
            "Media provenance audit log integration",
        ):
            assert pm.detect_project(q) == "archibal", f"missed: {q!r}"

    def test_c2pa_and_content_credentials(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "C2PA signing certificate rotation",
            "Add Content Credentials to the export pipeline",
            "Image authentication via C2PA",
            "Content authenticity for video uploads",
        ):
            assert pm.detect_project(q) == "archibal", f"missed: {q!r}"

    def test_synthetic_media_and_watermarking(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "Build a deepfake detection service",
            "Watermark all AI-generated images",
            "Synthetic media classifier for uploads",
            "AI detection accuracy benchmarks",
        ):
            assert pm.detect_project(q) == "archibal", f"missed: {q!r}"

    def test_pki_and_clearance(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "PKI certificate management for content signing",
            "Rights clearance workflow for stock images",
            "Public key infrastructure rollout for content auth",
        ):
            assert pm.detect_project(q) == "archibal", f"missed: {q!r}"


# ── PLG: Ticketing & live events ────────────────────────────────────


class TestPLGDetection:
    """Operator's spec: 'if it is ticketing, suggestion must be PLG'."""

    def test_ticketing_directly(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "How is the ticketing platform handling the spike?",
            "Improve ticket sales conversion",
            "Ticket pricing dynamic strategy",
            "Box office revenue last month",
            "Event ticketing UX redesign",
            "Event registration flow A/B test",
        ):
            assert pm.detect_project(q) == "plg", f"missed: {q!r}"

    def test_live_events(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "Concert tour schedule update",
            "Festival vendor onboarding",
            "Venue capacity planning",
            "Live event promoter dashboard",
            "Live Nation partnership terms",
        ):
            assert pm.detect_project(q) == "plg", f"missed: {q!r}"

    def test_brand_partners(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "PLG quarterly review",
            "Piletilevi inventory sync",
            "iaBilet integration status",
            "Protect Group escalation queue",
        ):
            assert pm.detect_project(q) == "plg", f"missed: {q!r}"

    def test_estonia_alone_does_not_route_to_plg(self) -> None:
        """Regression for the 2026-05-09 fix — bare geographies removed."""
        pm = _detector_with_all_profiles()
        assert pm.detect_project("tell me about Estonia") is None


# ── Eesti mets: Forest research (PR #69 baseline) ───────────────────


class TestEestiMetsDetection:
    """Sanity that the PR #69 detection still works after expansion
    of the other three projects' keyword lists."""

    def test_forest_research(self) -> None:
        pm = _detector_with_all_profiles()
        for q in (
            "Forest age distribution over time in Estonia",
            "Estonian forest health report",
            "Hansen Global Forest Change v1.10 reduction",
            "Landsat time-series analysis for deforestation",
            "Sentinel imagery for tree cover change",
        ):
            assert pm.detect_project(q) == "eesti mets", f"missed: {q!r}"


# ── Cross-project disambiguation ────────────────────────────────────


class TestCrossProjectDisambiguation:
    """When a query mentions vocabulary from multiple projects, the
    higher-scoring one wins. These cases catch any over-claim regression."""

    def test_estonian_forest_with_ticket_word(self) -> None:
        """Forest research mentioning 'ticket' (e.g. Linear ticket)
        should still route to Eesti mets."""
        pm = _detector_with_all_profiles()
        detected = pm.detect_project(
            "Open a ticket: investigate forest age distribution in Estonia "
            "using Hansen GFC and Landsat data"
        )
        assert detected == "eesti mets"

    def test_thai_forest_research(self) -> None:
        """Forest research about Thailand (Eesti mets's domain wins
        over KaiCart's geo-claim of 'thai')."""
        pm = _detector_with_all_profiles()
        detected = pm.detect_project(
            "Compare deforestation rates: Estonian forest vs Thai forest "
            "from satellite imagery"
        )
        assert detected == "eesti mets"

    def test_ecommerce_provenance_query(self) -> None:
        """E-commerce + provenance — should pick whichever has more
        keyword matches. Test the natural shape of the query."""
        pm = _detector_with_all_profiles()
        detected = pm.detect_project(
            "Add C2PA content credentials to product images on TikTok Shop"
        )
        # archibal: c2pa, content credentials = 2
        # kaicart: tiktok shop = 1
        assert detected == "archibal"

    def test_event_in_tropical_country(self) -> None:
        """Concert in Vietnam — KaiCart's vietnam vs PLG's concert.
        Both are single-keyword matches; tie-broken by dict iteration
        order in Python 3.7+. Just assert it's deterministic."""
        pm = _detector_with_all_profiles()
        result = pm.detect_project(
            "Live music concert tour in Vietnam"
        )
        # concert + live music + tour = 3 PLG signals
        # vietnam = 1 KaiCart signal
        assert result == "plg"

    def test_unrelated_returns_none(self) -> None:
        pm = _detector_with_all_profiles()
        assert pm.detect_project(
            "What is the meaning of life?"
        ) is None
