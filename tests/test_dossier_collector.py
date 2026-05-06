"""Tests for app.dossier.collector — pipeline + reconciliation + breaker.

Mocks the adapter registry with deterministic adapters to keep these
tests offline and fast.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.dossier import adapters as adapter_pkg
from app.dossier.adapters import (
    DossierAdapterResult,
    FieldUpdate,
    register_adapter,
    reset_for_tests,
)
from app.dossier.collector import collect_dossier
from app.dossier.schema import CompanyRef, Confidence


# ══════════════════════════════════════════════════════════════════════
# Adapter mocks
# ══════════════════════════════════════════════════════════════════════


class _Mock:
    """Minimal adapter for the registry."""

    def __init__(self, name, priority, *, can=True, configured=True,
                 fields=None, ref_enrichment=None, raises=None,
                 base_url=""):
        self.name = name
        self.priority = priority
        self._can = can
        self._configured = configured
        self._fields = fields or []
        self._enrichment = ref_enrichment or {}
        self._raises = raises
        self._base_url = base_url
        self.calls = 0

    def can_collect(self, ref):
        return self._can

    def is_configured(self):
        return self._configured

    def collect(self, ref):
        self.calls += 1
        if self._raises:
            raise self._raises
        return DossierAdapterResult(
            adapter_name=self.name,
            base_url=self._base_url,
            fields=list(self._fields),
            ref_enrichment=dict(self._enrichment),
        )


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Each test gets a clean registry; restore the production registry
    after the test by re-running install_defaults.
    """
    reset_for_tests()
    adapter_pkg._INSTALL_DONE = True  # block install_defaults from rebuilding
    yield
    reset_for_tests()
    adapter_pkg._INSTALL_DONE = False
    adapter_pkg.install_defaults()


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════


class TestSingleAdapter:

    def test_one_adapter_fills_dossier(self):
        register_adapter(_Mock(
            name="sec_edgar", priority=100,
            base_url="https://sec.gov/x",
            fields=[
                FieldUpdate("revenue_usd", 12.5e9, Confidence.EXACT,
                            as_of=date(2024, 12, 31)),
                FieldUpdate("legal_name", "Acme Inc.", Confidence.EXACT),
            ],
        ))
        ref = CompanyRef(name="Acme", ticker="ACME")
        d = collect_dossier(ref)
        assert d.revenue_usd.is_known
        assert d.revenue_usd.value == 12.5e9
        assert d.legal_name.value == "Acme Inc."
        assert d.coverage_report["adapters_fired"] == ["sec_edgar"]


class TestReconciliation:

    def test_higher_priority_wins_records_conflict(self):
        register_adapter(_Mock(
            name="sec_edgar", priority=100,
            fields=[FieldUpdate("revenue_usd", 12.5e9, Confidence.EXACT)],
        ))
        register_adapter(_Mock(
            name="wikidata", priority=60,
            fields=[FieldUpdate("revenue_usd", 11.0e9, Confidence.MEDIUM)],
        ))
        d = collect_dossier(CompanyRef(name="X", ticker="X"))
        assert d.revenue_usd.value == 12.5e9
        assert d.revenue_usd.source.adapter == "sec_edgar"
        assert len(d.revenue_usd.conflicts) == 1
        assert d.revenue_usd.conflicts[0].source.adapter == "wikidata"

    def test_lower_priority_runs_first_still_loses(self):
        # Same as above; the merge order is priority-stable regardless
        # of registration order.
        register_adapter(_Mock(
            name="wikidata", priority=60,
            fields=[FieldUpdate("revenue_usd", 11.0e9, Confidence.MEDIUM)],
        ))
        register_adapter(_Mock(
            name="sec_edgar", priority=100,
            fields=[FieldUpdate("revenue_usd", 12.5e9, Confidence.EXACT)],
        ))
        d = collect_dossier(CompanyRef(name="X", ticker="X"))
        assert d.revenue_usd.value == 12.5e9


class TestRefEnrichment:

    def test_enrichment_lands_on_ref(self):
        register_adapter(_Mock(
            name="wikidata", priority=60,
            ref_enrichment={"wikidata_id": "Q123", "ticker": "ACME"},
        ))
        ref = CompanyRef(name="Acme")
        collect_dossier(ref)
        assert ref.wikidata_id == "Q123"
        assert ref.ticker == "ACME"

    def test_enrichment_does_not_overwrite_explicit(self):
        register_adapter(_Mock(
            name="wikidata", priority=60,
            ref_enrichment={"ticker": "WRONG"},
        ))
        ref = CompanyRef(name="Acme", ticker="RIGHT")
        collect_dossier(ref)
        assert ref.ticker == "RIGHT"


class TestUnconfigured:

    def test_unconfigured_adapter_skipped(self):
        register_adapter(_Mock(
            name="companies_house", priority=95,
            configured=False,
            fields=[FieldUpdate("legal_name", "Z Ltd", Confidence.EXACT)],
        ))
        d = collect_dossier(CompanyRef(name="Z", ticker="Z"))
        assert "companies_house" in d.coverage_report["adapters_skipped"]
        assert d.coverage_report["adapters_skipped"]["companies_house"] == \
            "not configured"


class TestErrorHandling:

    def test_adapter_exception_recorded_not_fatal(self):
        # Adapter that raises unconditionally — collector must not blow up.
        register_adapter(_Mock(
            name="exploder", priority=50,
            raises=RuntimeError("boom"),
        ))
        register_adapter(_Mock(
            name="good", priority=40,
            fields=[FieldUpdate("legal_name", "Good Inc.", Confidence.EXACT)],
        ))
        d = collect_dossier(CompanyRef(name="X", ticker="X"))
        assert "exploder" in d.coverage_report["adapters_errored"]
        assert d.legal_name.value == "Good Inc."

    def test_can_collect_false_skipped_with_reason(self):
        register_adapter(_Mock(
            name="picky", priority=50, can=False,
        ))
        d = collect_dossier(CompanyRef(name="X"))  # no ticker
        assert d.coverage_report["adapters_skipped"].get("picky") == \
            "ref lacks required identity"


class TestCoverageReport:

    def test_report_populated(self):
        register_adapter(_Mock(
            name="a", priority=50,
            fields=[FieldUpdate("legal_name", "X", Confidence.EXACT)],
        ))
        d = collect_dossier(CompanyRef(name="X", ticker="X"))
        rpt = d.coverage_report
        assert rpt["fields_filled"] == 1
        assert rpt["fields_total"] >= 30
        assert 0 < rpt["coverage_pct"] < 100
        assert rpt["elapsed_seconds"] >= 0
