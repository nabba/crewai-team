"""yfinance_market — Market data for public companies.

yfinance is the cheap, reliable source for live market data: market
cap, P/E, EV/EBITDA, share price.  SEC EDGAR has the audited financials
but not market valuation — yfinance fills that gap.

Why this is a separate adapter from ``sec_edgar``
=================================================
SEC EDGAR is regulator-sourced (audited) — issued by the company.
Market valuation is derived from share price × shares outstanding,
which is exchange-sourced and changes minute-by-minute.  Different
provenance, different freshness story, different confidence label.

Confidence policy
=================
Market cap / P/E / EV/EBITDA are tagged ``HIGH`` (not ``EXACT``):
yfinance is a redistributor, not the source of record.  When the
prose composer says "as of T, market cap was X" it cites the
exchange close on day T via yfinance; it does NOT call this an
"audited disclosure."
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from app.dossier.adapters._base import (
    DossierAdapterResult,
    FieldUpdate,
    cache_lookup,
    cache_store,
    register_adapter,
)
from app.dossier.schema import CompanyRef, Confidence

logger = logging.getLogger(__name__)


class YFinanceMarketAdapter:
    """Live market valuation for tickers."""

    name = "yfinance_market"
    priority = 85  # high but below regulator (sec_edgar=100)

    def can_collect(self, ref: CompanyRef) -> bool:
        return bool(ref.ticker)

    def is_configured(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            return False

    def collect(self, ref: CompanyRef) -> DossierAdapterResult:
        cache_key = ref.ticker.upper()
        cached = cache_lookup(self.name, cache_key)
        if cached is not None:
            return cached
        result = self._collect_uncached(ref)
        cache_store(self.name, cache_key, result)
        return result

    def _collect_uncached(self, ref: CompanyRef) -> DossierAdapterResult:
        try:
            import yfinance as yf
        except ImportError:
            return DossierAdapterResult(
                adapter_name=self.name, error="yfinance not installed",
            )

        cite_url = f"https://finance.yahoo.com/quote/{ref.ticker}"
        result = DossierAdapterResult(
            adapter_name=self.name, base_url=cite_url,
        )

        try:
            ticker = yf.Ticker(ref.ticker.upper())
            info = ticker.info or {}
        except Exception as exc:
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"yfinance fetch failed: {type(exc).__name__}: {exc}",
            )

        if not info:
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"empty info dict for ticker {ref.ticker!r}",
            )

        as_of_today = datetime.now(timezone.utc).date()

        def _push(field_name: str, value, *, confidence: Confidence = Confidence.HIGH,
                  note: str = ""):
            result.fields.append(FieldUpdate(
                field_name=field_name, value=value,
                confidence=confidence, as_of=as_of_today, note=note,
            ))

        # ── Market valuation ─────────────────────────────────────────
        if (mcap := info.get("marketCap")):
            _push("market_cap_usd", float(mcap),
                  note="Yahoo Finance, share price × shares outstanding")

        if (ev := info.get("enterpriseValue")):
            _push("enterprise_value_usd", float(ev))

        if (pe := info.get("trailingPE")):
            _push("pe_ratio", float(pe))

        if (eve := info.get("enterpriseToEbitda")):
            _push("ev_ebitda", float(eve))

        # ── Identity ─────────────────────────────────────────────────
        if (long_name := info.get("longName")):
            _push("legal_name", str(long_name), confidence=Confidence.HIGH)

        if (website := info.get("website")):
            _push("website_url", str(website), confidence=Confidence.HIGH)
            # Enrich domain.
            try:
                from urllib.parse import urlparse
                host = urlparse(website).netloc.replace("www.", "")
                if host and not ref.website_domain:
                    result.ref_enrichment["website_domain"] = host
            except Exception:
                pass

        if (sector := info.get("sector")):
            industry = info.get("industry") or ""
            label = f"{sector} / {industry}" if industry else sector
            _push("industry_codes", (label,),
                  confidence=Confidence.MEDIUM,
                  note="Yahoo sector classification (not SIC/NAICS)")

        if (country := info.get("country")):
            _push("incorporated_in", str(country), confidence=Confidence.MEDIUM)

        # ── Workforce (from yfinance info, when present) ─────────────
        if (employees := info.get("fullTimeEmployees")):
            _push("employee_count", int(employees), confidence=Confidence.HIGH,
                  note="Yahoo Finance, latest disclosure")

        return result


def install() -> None:
    register_adapter(YFinanceMarketAdapter())
