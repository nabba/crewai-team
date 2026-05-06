"""Tests for app.dossier.schema — provenance + reconciliation invariants.

Mocks: none required.  Pure-Python unit tests on the schema.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.dossier.schema import (
    CompanyDossier,
    CompanyRef,
    Confidence,
    DossierField,
    FieldStatus,
    FundingRound,
    Owner,
    Source,
    merge_field,
)


# ══════════════════════════════════════════════════════════════════════
# DossierField factories
# ══════════════════════════════════════════════════════════════════════


class TestDossierFieldFactories:

    def test_known_factory_carries_value_and_source(self):
        src = Source(adapter="sec_edgar", url="https://sec.gov/cik/10-K")
        f = DossierField[float].known(
            12.5e9, source=src, confidence=Confidence.EXACT,
            as_of=date(2024, 12, 31),
        )
        assert f.is_known is True
        assert f.status == FieldStatus.KNOWN
        assert f.value == 12.5e9
        assert f.source.adapter == "sec_edgar"
        assert f.confidence == Confidence.EXACT
        assert f.as_of == date(2024, 12, 31)
        assert f.conflicts == []

    def test_not_disclosed_carries_reason(self):
        f = DossierField[float].not_disclosed(reason="company is private")
        assert f.is_known is False
        assert f.status == FieldStatus.NOT_DISCLOSED
        assert f.value is None
        assert "private" in f.reason

    def test_not_applicable_renders_silently(self):
        f = DossierField[float].not_applicable()
        assert f.status == FieldStatus.NOT_APPLICABLE
        assert f.render_value() == "not applicable"

    def test_unresolved_default(self):
        f = DossierField[float].unresolved()
        assert f.status == FieldStatus.UNRESOLVED
        assert f.render_value() == "data unavailable"


class TestRenderValue:

    def test_currency_billions(self):
        src = Source(adapter="x")
        f = DossierField[float].known(12_500_000_000.0, source=src,
                                      confidence=Confidence.EXACT)
        assert f.render_value() == "$12.50B"

    def test_currency_millions(self):
        src = Source(adapter="x")
        f = DossierField[float].known(380_000_000.0, source=src,
                                      confidence=Confidence.HIGH)
        assert f.render_value() == "$380.0M"

    def test_int_thousands(self):
        src = Source(adapter="x")
        f = DossierField[int].known(7400, source=src,
                                    confidence=Confidence.EXACT)
        assert f.render_value() == "7.4K"

    def test_date_iso(self):
        src = Source(adapter="x")
        f = DossierField[date].known(date(2006, 4, 23), source=src,
                                     confidence=Confidence.MEDIUM)
        assert f.render_value() == "2006-04-23"

    def test_tuple(self):
        src = Source(adapter="x")
        f = DossierField[tuple[str, ...]].known(
            ("Daniel Ek", "Martin Lorentzon"),
            source=src, confidence=Confidence.MEDIUM,
        )
        assert f.render_value() == "Daniel Ek, Martin Lorentzon"


# ══════════════════════════════════════════════════════════════════════
# Confidence mapping to evidence confidence
# ══════════════════════════════════════════════════════════════════════


class TestConfidenceMapping:

    def test_exact_is_one(self):
        assert Confidence.EXACT.to_evidence_confidence() == 1.0

    def test_estimated_low(self):
        assert Confidence.ESTIMATED.to_evidence_confidence() == 0.3

    def test_descending(self):
        scores = [
            Confidence.EXACT.to_evidence_confidence(),
            Confidence.HIGH.to_evidence_confidence(),
            Confidence.MEDIUM.to_evidence_confidence(),
            Confidence.LOW.to_evidence_confidence(),
            Confidence.ESTIMATED.to_evidence_confidence(),
        ]
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════
# merge_field — the reconciliation invariant
# ══════════════════════════════════════════════════════════════════════


class TestMergeField:

    PRIORITY = {"sec_edgar": 100, "wikidata": 60, "web": 20}

    def test_higher_priority_wins(self):
        existing = DossierField[float].known(
            10.0, source=Source(adapter="wikidata"),
            confidence=Confidence.MEDIUM,
        )
        merged = merge_field(
            existing=existing,
            new_value=12.5,
            source=Source(adapter="sec_edgar"),
            confidence=Confidence.EXACT,
            as_of=None,
            source_priority=self.PRIORITY,
        )
        assert merged.value == 12.5
        assert merged.source.adapter == "sec_edgar"
        assert len(merged.conflicts) == 1
        assert merged.conflicts[0].value == 10.0

    def test_lower_priority_loses(self):
        existing = DossierField[float].known(
            12.5, source=Source(adapter="sec_edgar"),
            confidence=Confidence.EXACT,
        )
        merged = merge_field(
            existing=existing,
            new_value=10.0,
            source=Source(adapter="wikidata"),
            confidence=Confidence.MEDIUM,
            as_of=None,
            source_priority=self.PRIORITY,
        )
        assert merged.value == 12.5
        assert merged.source.adapter == "sec_edgar"
        assert len(merged.conflicts) == 1
        assert merged.conflicts[0].value == 10.0
        assert merged.conflicts[0].source.adapter == "wikidata"

    def test_unresolved_field_accepts_first_value(self):
        existing = DossierField[float].unresolved()
        merged = merge_field(
            existing=existing,
            new_value=42.0,
            source=Source(adapter="web"),
            confidence=Confidence.LOW,
            as_of=None,
            source_priority=self.PRIORITY,
        )
        assert merged.is_known is True
        assert merged.value == 42.0
        assert merged.conflicts == []

    def test_priority_tie_broken_by_confidence(self):
        # Both have priority 60 (wikidata).  Higher confidence wins.
        existing = DossierField[float].known(
            10.0, source=Source(adapter="wikidata"),
            confidence=Confidence.LOW,
        )
        priority = {"wikidata": 60}
        merged = merge_field(
            existing=existing,
            new_value=11.0,
            source=Source(adapter="wikidata"),
            confidence=Confidence.HIGH,
            as_of=None,
            source_priority=priority,
        )
        assert merged.value == 11.0


# ══════════════════════════════════════════════════════════════════════
# Coverage helpers
# ══════════════════════════════════════════════════════════════════════


class TestCoverageHelpers:

    def test_empty_dossier_zero_coverage(self):
        d = CompanyDossier(ref=CompanyRef(name="X"))
        assert d.known_field_count() == 0
        assert d.coverage_pct() == 0.0
        assert d.total_field_count() > 30  # invariant: many fields

    def test_partial_population(self):
        d = CompanyDossier(ref=CompanyRef(name="X"))
        d.legal_name = DossierField[str].known(
            "X Inc.", source=Source(adapter="sec_edgar"),
            confidence=Confidence.EXACT,
        )
        d.revenue_usd = DossierField[float].known(
            1.0, source=Source(adapter="sec_edgar"),
            confidence=Confidence.EXACT,
        )
        assert d.known_field_count() == 2
        assert 0.0 < d.coverage_pct() < 1.0


# ══════════════════════════════════════════════════════════════════════
# Sub-models
# ══════════════════════════════════════════════════════════════════════


class TestFundingRound:

    def test_construct(self):
        fr = FundingRound(
            round_type="series_a",
            amount_usd=20_000_000.0,
            announced_on=date(2020, 6, 1),
            lead_investors=("Acme Capital",),
        )
        assert fr.round_type == "series_a"
        assert fr.lead_investors == ("Acme Capital",)


class TestOwner:

    def test_construct(self):
        o = Owner(name="Founder Holdings", kind="founder",
                  pct_ownership=0.45)
        assert o.kind == "founder"
        assert o.pct_ownership == 0.45
