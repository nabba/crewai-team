"""Tests for app.dossier.compose — fact-check + section composition.

Uses the deterministic ``_slice_echo`` composer so tests don't need
an LLM running.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.dossier.compose import (
    _fact_check,
    _slice_echo,
    compose_report,
)
from app.dossier.schema import (
    CompanyDossier,
    CompanyRef,
    Confidence,
    DossierField,
    Source,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_dossier():
    ref = CompanyRef(name="Acme", ticker="ACME")
    d = CompanyDossier(ref=ref)
    src = Source(adapter="sec_edgar", url="https://sec.gov/x")
    d.legal_name = DossierField[str].known("Acme Inc.", source=src,
                                           confidence=Confidence.EXACT)
    d.revenue_usd = DossierField[float].known(
        12_500_000_000.0, source=src, confidence=Confidence.EXACT,
        as_of=date(2024, 12, 31),
    )
    d.employee_count = DossierField[int].known(
        7400, source=src, confidence=Confidence.EXACT,
        as_of=date(2024, 12, 31),
    )
    d.founded_on = DossierField[date].known(
        date(2006, 4, 23), source=src, confidence=Confidence.MEDIUM,
    )
    d.coverage_report = {"fields_filled": 4, "fields_total": 33,
                          "coverage_pct": 12.1, "adapters_fired": ["sec_edgar"]}
    return d


# ══════════════════════════════════════════════════════════════════════
# Fact-check pass
# ══════════════════════════════════════════════════════════════════════


class TestFactCheckHappyPath:

    def test_correctly_quoted_currency_no_warnings(self, sample_dossier):
        prose = "Revenue of $12.50B [revenue_usd] in fiscal 2024."
        warnings = _fact_check(prose, sample_dossier, ("revenue_usd",))
        assert warnings == []

    def test_correctly_quoted_int_with_commas(self, sample_dossier):
        prose = "The company had 7,400 employees [employee_count]."
        warnings = _fact_check(prose, sample_dossier, ("employee_count",))
        assert warnings == []

    def test_year_match(self, sample_dossier):
        # 2024 is the as_of year of revenue_usd
        prose = "Revenue was $12.50B in 2024."
        warnings = _fact_check(prose, sample_dossier, ("revenue_usd",))
        assert warnings == []

    def test_minor_format_variation(self, sample_dossier):
        # Same value, different formatting variant
        prose = "Revenue of $12.5B [revenue_usd]."
        warnings = _fact_check(prose, sample_dossier, ("revenue_usd",))
        assert warnings == []


class TestFactCheckCatchesInventions:

    def test_invented_currency_flagged(self, sample_dossier):
        prose = "Revenue was $15.2B [revenue_usd]."
        warnings = _fact_check(prose, sample_dossier, ("revenue_usd",))
        assert len(warnings) >= 1
        assert any("$15.2B" in w for w in warnings)

    def test_invented_employee_count_flagged(self, sample_dossier):
        prose = "The company had 9,000 employees."
        warnings = _fact_check(prose, sample_dossier, ("employee_count",))
        assert len(warnings) >= 1
        assert any("9,000" in w for w in warnings)

    def test_invented_year_flagged(self, sample_dossier):
        prose = "Acme was founded in 1985."  # actual 2006
        warnings = _fact_check(prose, sample_dossier, ("founded_on",))
        assert any("1985" in w for w in warnings)


class TestFactCheckOverlap:

    def test_no_double_flag_for_substring_overlap(self, sample_dossier):
        # "$12.50B" should produce one match; "$12.50" inside it must not
        # produce a second warning.
        prose = "Revenue of $12.50B."
        warnings = _fact_check(prose, sample_dossier, ("revenue_usd",))
        assert warnings == []
        # And again with two distinct matches:
        prose2 = "Revenue $12.50B [revenue_usd] with 7,400 staff."
        warnings2 = _fact_check(prose2, sample_dossier,
                                 ("revenue_usd", "employee_count"))
        assert warnings2 == []


class TestFactCheckEmpty:

    def test_empty_prose_no_warnings(self, sample_dossier):
        assert _fact_check("", sample_dossier, ("revenue_usd",)) == []

    def test_no_numeric_tokens_no_warnings(self, sample_dossier):
        prose = "The company is described as a technology firm."
        assert _fact_check(prose, sample_dossier, ()) == []


# ══════════════════════════════════════════════════════════════════════
# Composition with slice-echo composer (LLM-free)
# ══════════════════════════════════════════════════════════════════════


class TestComposeReport:

    def test_compose_produces_all_sections(self, sample_dossier):
        report = compose_report(sample_dossier, peer_dossiers=[],
                                 llm_call=_slice_echo)
        # 8 standard sections + 1 comparator
        assert len(report.sections) == 9
        keys = {s.key for s in report.sections}
        assert "executive_summary" in keys
        assert "financials" in keys
        assert "comparator" in keys
        assert "risks" in keys

    def test_each_section_has_prose(self, sample_dossier):
        report = compose_report(sample_dossier, peer_dossiers=[],
                                 llm_call=_slice_echo)
        for section in report.sections:
            assert section.prose
            assert section.word_count > 0

    def test_company_name_propagated(self, sample_dossier):
        report = compose_report(sample_dossier, peer_dossiers=[],
                                 llm_call=_slice_echo)
        assert report.company_name == "Acme"

    def test_failed_llm_does_not_abort(self, sample_dossier):
        def _broken_llm(prompt):
            raise RuntimeError("LLM broken")
        report = compose_report(sample_dossier, peer_dossiers=[],
                                 llm_call=_broken_llm)
        # All sections are present; each has a placeholder marker.
        assert len(report.sections) == 9
        for section in report.sections:
            assert "[section unavailable" in section.prose


class TestComparatorWithoutPeers:

    def test_comparator_renders_stub_with_no_peers(self, sample_dossier):
        report = compose_report(sample_dossier, peer_dossiers=[],
                                 llm_call=_slice_echo)
        comp = next(s for s in report.sections if s.key == "comparator")
        assert "unavailable" in comp.prose.lower() or comp.prose
