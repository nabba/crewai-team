"""
test_currency_tools.py — Unit tests for ECB-backed currency conversion.

Run: pytest tests/test_currency_tools.py -v
"""
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools import currency_tools


# A trimmed but realistic ECB daily payload.
DAILY_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01" xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">
  <gesmes:subject>Reference rates</gesmes:subject>
  <gesmes:Sender><gesmes:name>European Central Bank</gesmes:name></gesmes:Sender>
  <Cube>
    <Cube time="2024-06-03">
      <Cube currency="USD" rate="1.0890"/>
      <Cube currency="GBP" rate="0.8540"/>
      <Cube currency="JPY" rate="170.50"/>
      <Cube currency="CHF" rate="0.9750"/>
      <Cube currency="SEK" rate="11.4500"/>
    </Cube>
  </Cube>
</gesmes:Envelope>
"""

# A multi-day historical payload spanning a weekend (May 31 Fri, Jun 3 Mon).
HIST_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01" xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">
  <gesmes:subject>Reference rates</gesmes:subject>
  <gesmes:Sender><gesmes:name>European Central Bank</gesmes:name></gesmes:Sender>
  <Cube>
    <Cube time="2024-06-03">
      <Cube currency="USD" rate="1.0890"/>
      <Cube currency="GBP" rate="0.8540"/>
      <Cube currency="JPY" rate="170.50"/>
    </Cube>
    <Cube time="2024-05-31">
      <Cube currency="USD" rate="1.0850"/>
      <Cube currency="GBP" rate="0.8500"/>
      <Cube currency="JPY" rate="170.00"/>
    </Cube>
    <Cube time="2024-05-30">
      <Cube currency="USD" rate="1.0820"/>
      <Cube currency="GBP" rate="0.8480"/>
      <Cube currency="JPY" rate="169.50"/>
    </Cube>
  </Cube>
</gesmes:Envelope>
"""


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the module-level cache between tests."""
    currency_tools._cache.clear()
    yield
    currency_tools._cache.clear()


def _patch_fetch(payload: bytes):
    """Patch _fetch_feed to return the parsed payload — avoids needing `requests`."""
    parsed = currency_tools._parse_ecb_xml(payload)
    return patch.object(currency_tools, "_fetch_feed", return_value=parsed)


# ── Parsing ───────────────────────────────────────────────────────────


class TestParsing:
    def test_parses_eur_with_rate_one(self):
        out = currency_tools._parse_ecb_xml(DAILY_XML)
        assert date(2024, 6, 3) in out
        assert out[date(2024, 6, 3)]["EUR"] == Decimal("1")

    def test_parses_decimal_rates(self):
        out = currency_tools._parse_ecb_xml(DAILY_XML)
        rates = out[date(2024, 6, 3)]
        assert rates["USD"] == Decimal("1.0890")
        assert rates["JPY"] == Decimal("170.50")

    def test_parses_multiple_dates(self):
        out = currency_tools._parse_ecb_xml(HIST_XML)
        assert date(2024, 6, 3) in out
        assert date(2024, 5, 31) in out
        assert date(2024, 5, 30) in out


# ── get_ecb_rates ─────────────────────────────────────────────────────


class TestGetEcbRates:
    def test_returns_eur_base(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                effective, rates = currency_tools.get_ecb_rates(date(2024, 6, 3))
        assert effective == date(2024, 6, 3)
        assert rates["EUR"] == Decimal("1")
        assert rates["USD"] == Decimal("1.0890")

    def test_weekend_falls_back_to_friday(self):
        # Sunday June 2 → should resolve to Friday May 31
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(HIST_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_HIST_90D_URL, 60)):
                effective, rates = currency_tools.get_ecb_rates(date(2024, 6, 2))
        assert effective == date(2024, 5, 31)
        assert rates["USD"] == Decimal("1.0850")

    def test_rejects_future_date(self):
        with pytest.raises(ValueError, match="future"):
            currency_tools.get_ecb_rates(date(9999, 1, 1))

    def test_rejects_pre_1999_date(self):
        with pytest.raises(ValueError, match="1999"):
            currency_tools.get_ecb_rates(date(1998, 12, 31))

    def test_caches_subsequent_calls(self):
        mock_fetch = MagicMock(return_value=currency_tools._parse_ecb_xml(DAILY_XML))
        with patch.object(currency_tools, "_fetch_feed", mock_fetch):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 3600)):
                currency_tools.get_ecb_rates(date(2024, 6, 3))
                currency_tools.get_ecb_rates(date(2024, 6, 3))
        # Second call should hit the cache, not the network.
        assert mock_fetch.call_count == 1


# ── convert_currency ──────────────────────────────────────────────────


class TestConvertCurrency:
    def test_eur_to_usd(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                converted, effective = currency_tools.convert_currency(
                    Decimal("100"), "EUR", "USD", date(2024, 6, 3)
                )
        assert converted == Decimal("108.90")
        assert effective == date(2024, 6, 3)

    def test_usd_to_eur(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                converted, _ = currency_tools.convert_currency(
                    Decimal("108.90"), "USD", "EUR", date(2024, 6, 3)
                )
        # 108.90 USD / 1.0890 = exactly 100 EUR
        assert converted == Decimal("100")

    def test_cross_currency_usd_to_jpy(self):
        # 100 USD → EUR → JPY:  100 / 1.0890 * 170.50
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                converted, _ = currency_tools.convert_currency(
                    Decimal("100"), "USD", "JPY", date(2024, 6, 3)
                )
        expected = Decimal("100") / Decimal("1.0890") * Decimal("170.50")
        assert converted == expected

    def test_same_currency_is_identity(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                converted, _ = currency_tools.convert_currency(
                    Decimal("42.50"), "USD", "USD", date(2024, 6, 3)
                )
        assert converted == Decimal("42.50")

    def test_accepts_lowercase_currency(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                converted, _ = currency_tools.convert_currency(
                    Decimal("100"), "eur", "usd", date(2024, 6, 3)
                )
        assert converted == Decimal("108.90")

    def test_unknown_currency_raises(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                with pytest.raises(ValueError, match="XYZ"):
                    currency_tools.convert_currency(
                        Decimal("100"), "XYZ", "USD", date(2024, 6, 3)
                    )

    def test_accepts_float_and_string_amounts(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                from_float, _ = currency_tools.convert_currency(100.0, "EUR", "USD", date(2024, 6, 3))
                from_str, _ = currency_tools.convert_currency("100", "EUR", "USD", date(2024, 6, 3))
        assert from_float == Decimal("108.90")
        assert from_str == Decimal("108.90")


# ── CrewAI tool wrapper ───────────────────────────────────────────────


_crewai_available = True
try:
    import crewai  # noqa: F401
except ImportError:
    _crewai_available = False


@pytest.mark.skipif(not _crewai_available, reason="crewai not installed")
class TestCurrencyConvertTool:
    def _tool(self, name: str):
        tools = currency_tools.create_currency_tools("test")
        return next(t for t in tools if t.name == name)

    def test_factory_returns_two_tools(self):
        tools = currency_tools.create_currency_tools("test")
        assert {t.name for t in tools} == {"currency_convert", "currency_rates"}

    def test_convert_basic(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                tool = self._tool("currency_convert")
                result = tool._run(
                    amount=100.0,
                    from_currency="EUR",
                    to_currency="USD",
                    target_date="2024-06-03",
                )
        assert "100.00 EUR" in result
        assert "108.90 USD" in result
        assert "2024-06-03" in result

    def test_convert_invalid_date_format(self):
        tool = self._tool("currency_convert")
        result = tool._run(
            amount=100.0,
            from_currency="EUR",
            to_currency="USD",
            target_date="not-a-date",
        )
        assert "Invalid date" in result

    def test_convert_unknown_currency(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                tool = self._tool("currency_convert")
                result = tool._run(
                    amount=100.0,
                    from_currency="XYZ",
                    to_currency="USD",
                    target_date="2024-06-03",
                )
        assert "error" in result.lower() or "XYZ" in result

    def test_convert_weekend_notes_fallback(self):
        # Sunday June 2 → falls back to Friday May 31
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(HIST_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_HIST_90D_URL, 60)):
                tool = self._tool("currency_convert")
                result = tool._run(
                    amount=100.0,
                    from_currency="EUR",
                    to_currency="USD",
                    target_date="2024-06-02",
                )
        assert "did not publish" in result
        assert "2024-05-31" in result


@pytest.mark.skipif(not _crewai_available, reason="crewai not installed")
class TestCurrencyRatesTool:
    def _tool(self):
        tools = currency_tools.create_currency_tools("test")
        return next(t for t in tools if t.name == "currency_rates")

    def test_lists_all_rates(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                result = self._tool()._run(target_date="2024-06-03")
        assert "EUR" in result
        assert "USD" in result
        assert "1.0890" in result

    def test_filters_by_currency_list(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                result = self._tool()._run(target_date="2024-06-03", currencies="USD,JPY")
        assert "USD" in result
        assert "JPY" in result
        assert "GBP" not in result

    def test_unknown_currency_in_filter_reports_missing(self):
        with patch.object(currency_tools, "_fetch_feed", return_value=currency_tools._parse_ecb_xml(DAILY_XML)):
            with patch.object(currency_tools, "_select_feed", return_value=(currency_tools.ECB_DAILY_URL, 60)):
                result = self._tool()._run(target_date="2024-06-03", currencies="USD,XYZ")
        assert "XYZ" in result
        assert "not in" in result.lower()
