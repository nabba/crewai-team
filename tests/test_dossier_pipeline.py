"""Tests for app.dossier.pipeline — identity parsing + e2e with mocks."""
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
from app.dossier.compose import _slice_echo
from app.dossier.pipeline import build_dossier, parse_identity
from app.dossier.schema import CompanyRef, Confidence


# ══════════════════════════════════════════════════════════════════════
# Identity parsing
# ══════════════════════════════════════════════════════════════════════


class TestParseIdentity:

    def test_paren_ticker(self):
        ref = parse_identity("Build a dossier for Spotify (SPOT)")
        assert ref.ticker == "SPOT"
        assert "spotify" in ref.name.lower()

    def test_dollar_ticker(self):
        ref = parse_identity("Investment review for $AAPL please")
        assert ref.ticker == "AAPL"

    def test_explicit_ticker_keyword(self):
        ref = parse_identity("Profile for ticker: NVDA")
        assert ref.ticker == "NVDA"

    def test_nyse_prefix(self):
        ref = parse_identity("NYSE:JPM compete review")
        assert ref.ticker == "JPM"

    def test_just_company_name(self):
        ref = parse_identity("Tony's Chocolonely review")
        assert ref.name.lower().replace("'", "").startswith("tony")
        assert ref.ticker == ""

    def test_empty_query_empty_name(self):
        ref = parse_identity("")
        assert ref.name == ""
        assert ref.ticker == ""


# ══════════════════════════════════════════════════════════════════════
# End-to-end build
# ══════════════════════════════════════════════════════════════════════


class _MockAdapter:
    def __init__(self, name, priority, fields=None, base_url=""):
        self.name = name
        self.priority = priority
        self._fields = fields or []
        self._base_url = base_url

    def can_collect(self, ref):
        return bool(ref.name or ref.ticker)

    def is_configured(self):
        return True

    def collect(self, ref):
        return DossierAdapterResult(
            adapter_name=self.name,
            base_url=self._base_url,
            fields=list(self._fields),
        )


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_for_tests()
    adapter_pkg._INSTALL_DONE = True
    yield
    reset_for_tests()
    adapter_pkg._INSTALL_DONE = False
    adapter_pkg.install_defaults()


class TestBuildDossier:

    def test_e2e_with_mocked_adapters(self, tmp_path):
        register_adapter(_MockAdapter(
            name="sec_edgar", priority=100,
            base_url="https://sec.gov/x",
            fields=[
                FieldUpdate("legal_name", "Acme Inc.", Confidence.EXACT),
                FieldUpdate("revenue_usd", 12.5e9, Confidence.EXACT,
                            as_of=date(2024, 12, 31)),
                FieldUpdate("employee_count", 7400, Confidence.EXACT,
                            as_of=date(2024, 12, 31)),
                FieldUpdate("industry_codes",
                            ("SIC 7372 (Prepackaged Software)",),
                            Confidence.EXACT),
            ],
        ))
        out = tmp_path / "acme.pdf"
        build = build_dossier(
            query="Acme (ACME) dossier",
            output_path=str(out),
            include_peers=False,
            llm_call=_slice_echo,
        )
        assert build.ref.ticker == "ACME"
        assert build.ref.name.lower().startswith("acme")
        assert build.dossier.revenue_usd.value == 12.5e9
        assert build.pdf_path == out
        assert out.exists()
        assert out.stat().st_size > 1000  # non-empty PDF

    def test_summary_string_shape(self, tmp_path):
        register_adapter(_MockAdapter(
            name="sec_edgar", priority=100,
            fields=[FieldUpdate("legal_name", "X", Confidence.EXACT)],
        ))
        build = build_dossier(
            query="X (X)", output_path=str(tmp_path / "x.pdf"),
            include_peers=False, llm_call=_slice_echo,
        )
        s = build.summary()
        assert "Dossier for" in s
        assert "fields populated" in s
        assert "PDF:" in s

    def test_no_identity_raises(self, tmp_path):
        with pytest.raises(ValueError):
            build_dossier(query="", output_path=str(tmp_path / "x.pdf"))


class TestProgressCallback:

    def test_callback_fired_at_each_stage(self, tmp_path):
        register_adapter(_MockAdapter(
            name="sec_edgar", priority=100,
            fields=[FieldUpdate("legal_name", "Y", Confidence.EXACT)],
        ))
        steps: list[str] = []
        build_dossier(
            query="Y (Y)", output_path=str(tmp_path / "y.pdf"),
            include_peers=False, llm_call=_slice_echo,
            progress_callback=steps.append,
        )
        # We expect at least 3 steps: collect, compose, typeset.
        assert len(steps) >= 3
        assert any("Collecting" in s for s in steps)
        assert any("Composing" in s for s in steps)
        assert any("Typesetting" in s for s in steps)
