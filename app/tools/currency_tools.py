"""
currency_tools.py — Currency conversion using European Central Bank (ECB) reference rates.

Data source:
  - ECB daily euro foreign exchange reference rates (free, no API key)
    https://www.ecb.europa.eu/stats/eurofxref/

Three feeds are used depending on the requested date:
  - eurofxref-daily.xml         — latest published rates
  - eurofxref-hist-90d.xml      — last 90 days
  - eurofxref-hist.xml          — full history back to 1999-01-04

ECB does not publish on weekends, TARGET holidays, or before the daily fixing
(~16:00 CET). For such dates this module falls back to the most recent earlier
business day with published rates.

Usage:
    from app.tools.currency_tools import convert_currency, create_currency_tools

    # Programmatic
    eur = convert_currency(Decimal("100"), "USD", "EUR")
    on_date = convert_currency(Decimal("100"), "USD", "JPY", date(2024, 6, 3))

    # As a CrewAI tool
    tools = create_currency_tools("agent_id")
"""

from __future__ import annotations

import logging
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

logger = logging.getLogger(__name__)

ECB_DAILY_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
ECB_HIST_90D_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml"
ECB_HIST_FULL_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.xml"

ECB_HISTORY_START = date(1999, 1, 4)

_NS = {
    "gesmes": "http://www.gesmes.org/xml/2002-08-01",
    "ecb": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref",
}

_DAILY_TTL_SECONDS = 60 * 60  # 1 hour — daily feed updates once per business day
_HIST_TTL_SECONDS = 12 * 60 * 60  # 12 hours
_REQUEST_TIMEOUT = 15


@dataclass
class _CacheEntry:
    rates_by_date: dict[date, dict[str, Decimal]]
    fetched_at: datetime


_cache: dict[str, _CacheEntry] = {}
_cache_lock = threading.Lock()


# ── ECB feed parsing ──────────────────────────────────────────────────


def _parse_ecb_xml(xml_bytes: bytes) -> dict[date, dict[str, Decimal]]:
    """Parse an ECB eurofxref XML payload into {date: {currency: rate_per_eur}}.

    EUR is included with rate 1, since ECB quotes everything as 1 EUR = X currency.
    """
    root = ET.fromstring(xml_bytes)
    cube_root = root.find("ecb:Cube", _NS)
    if cube_root is None:
        raise ValueError("Malformed ECB XML: missing top-level Cube element")

    out: dict[date, dict[str, Decimal]] = {}
    for day_cube in cube_root.findall("ecb:Cube", _NS):
        time_attr = day_cube.get("time")
        if not time_attr:
            continue
        try:
            d = date.fromisoformat(time_attr)
        except ValueError:
            continue
        rates: dict[str, Decimal] = {"EUR": Decimal("1")}
        for entry in day_cube.findall("ecb:Cube", _NS):
            cur = entry.get("currency")
            rate = entry.get("rate")
            if cur and rate:
                try:
                    rates[cur] = Decimal(rate)
                except Exception:
                    continue
        out[d] = rates
    return out


def _fetch_feed(url: str) -> dict[date, dict[str, Decimal]]:
    import requests

    resp = requests.get(
        url,
        timeout=_REQUEST_TIMEOUT,
        headers={"User-Agent": "BotArmy/1.0 currency_tools"},
    )
    resp.raise_for_status()
    return _parse_ecb_xml(resp.content)


def _get_feed_cached(url: str, ttl_seconds: int) -> dict[date, dict[str, Decimal]]:
    now = datetime.now(timezone.utc)
    with _cache_lock:
        entry = _cache.get(url)
        if entry and (now - entry.fetched_at).total_seconds() < ttl_seconds:
            return entry.rates_by_date

    rates = _fetch_feed(url)
    with _cache_lock:
        _cache[url] = _CacheEntry(rates_by_date=rates, fetched_at=now)
    return rates


def _select_feed(target: date) -> tuple[str, int]:
    """Pick the smallest ECB feed that should contain the target date."""
    today = date.today()
    if target >= today - timedelta(days=2):
        return ECB_DAILY_URL, _DAILY_TTL_SECONDS
    if target >= today - timedelta(days=85):
        return ECB_HIST_90D_URL, _HIST_TTL_SECONDS
    return ECB_HIST_FULL_URL, _HIST_TTL_SECONDS


# ── Public API ────────────────────────────────────────────────────────


def get_ecb_rates(target_date: Optional[date] = None) -> tuple[date, dict[str, Decimal]]:
    """Return ECB rates for `target_date` or the most recent earlier business day.

    Returns (effective_date, {currency_code: rate_per_eur}). EUR is always present
    with rate 1.

    Raises:
        ValueError: target_date is in the future or before 1999-01-04.
        LookupError: no rates available within 14 days before target_date
            (only happens for unusually long ECB outages).
    """
    if target_date is None:
        target_date = date.today()

    if target_date < ECB_HISTORY_START:
        raise ValueError(
            f"ECB reference rates only go back to {ECB_HISTORY_START.isoformat()}; "
            f"requested {target_date.isoformat()}"
        )
    if target_date > date.today():
        raise ValueError(f"Requested date {target_date.isoformat()} is in the future")

    url, ttl = _select_feed(target_date)
    feed = _get_feed_cached(url, ttl)

    # If the daily feed doesn't cover the request (e.g. asked for yesterday before
    # today's fixing), fall through to the 90-day feed.
    if not any(d <= target_date for d in feed) and url == ECB_DAILY_URL:
        url, ttl = ECB_HIST_90D_URL, _HIST_TTL_SECONDS
        feed = _get_feed_cached(url, ttl)

    # Walk back up to 14 days to skip weekends and TARGET holidays.
    for offset in range(15):
        candidate = target_date - timedelta(days=offset)
        if candidate in feed:
            return candidate, feed[candidate]

    # Last resort: try the full history feed if we weren't already on it.
    if url != ECB_HIST_FULL_URL:
        feed = _get_feed_cached(ECB_HIST_FULL_URL, _HIST_TTL_SECONDS)
        for offset in range(15):
            candidate = target_date - timedelta(days=offset)
            if candidate in feed:
                return candidate, feed[candidate]

    raise LookupError(
        f"No ECB rates available within 14 days before {target_date.isoformat()}"
    )


def convert_currency(
    amount: Decimal | float | int | str,
    from_currency: str,
    to_currency: str,
    target_date: Optional[date] = None,
) -> tuple[Decimal, date]:
    """Convert `amount` from one currency to another using ECB rates.

    Returns (converted_amount, effective_date). The effective_date may be earlier
    than the requested date when the request lands on a weekend or holiday.

    Raises:
        ValueError: unsupported currency, or invalid date.
    """
    src = from_currency.strip().upper()
    dst = to_currency.strip().upper()
    amt = amount if isinstance(amount, Decimal) else Decimal(str(amount))

    effective_date, rates = get_ecb_rates(target_date)

    if src not in rates:
        raise ValueError(f"Currency '{src}' not in ECB reference rates for {effective_date.isoformat()}")
    if dst not in rates:
        raise ValueError(f"Currency '{dst}' not in ECB reference rates for {effective_date.isoformat()}")

    # ECB quotes as: 1 EUR = rate[X] units of X.  So  amount_X / rate[X]  is in EUR.
    amount_in_eur = amt / rates[src]
    converted = amount_in_eur * rates[dst]
    return converted, effective_date


# ── CrewAI tool wrapper ───────────────────────────────────────────────


def create_currency_tools(agent_id: str) -> list:
    """Create currency tools for a CrewAI agent. Returns [] if crewai not installed."""
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        logger.debug("currency_tools: crewai not installed")
        return []

    class _ConvertInput(BaseModel):
        amount: float = Field(description="Amount of money to convert")
        from_currency: str = Field(description="Source currency code (e.g. USD, EUR, GBP, JPY)")
        to_currency: str = Field(description="Target currency code (e.g. USD, EUR, GBP, JPY)")
        target_date: Optional[str] = Field(
            default=None,
            description=(
                "ISO date YYYY-MM-DD for historical conversion. "
                "Defaults to today. Weekends/holidays fall back to last business day."
            ),
        )

    class CurrencyConvertTool(BaseTool):
        name: str = "currency_convert"
        description: str = (
            "Convert money between currencies using European Central Bank (ECB) "
            "reference rates. Supports ~30 major currencies (EUR, USD, GBP, JPY, CHF, "
            "CNY, etc.). Pass an ISO date (YYYY-MM-DD) for historical conversion; "
            "defaults to today. ECB does not publish on weekends/holidays — those "
            "fall back to the most recent business day."
        )
        args_schema: Type[BaseModel] = _ConvertInput

        def _run(
            self,
            amount: float,
            from_currency: str,
            to_currency: str,
            target_date: Optional[str] = None,
        ) -> str:
            try:
                parsed_date: Optional[date] = None
                if target_date:
                    try:
                        parsed_date = date.fromisoformat(target_date.strip())
                    except ValueError:
                        return f"Invalid date '{target_date}'. Use ISO format YYYY-MM-DD."

                converted, effective = convert_currency(
                    Decimal(str(amount)),
                    from_currency,
                    to_currency,
                    parsed_date,
                )

                src = from_currency.strip().upper()
                dst = to_currency.strip().upper()
                requested = parsed_date.isoformat() if parsed_date else date.today().isoformat()
                fallback_note = (
                    f" (rates from {effective.isoformat()}, ECB did not publish on {requested})"
                    if parsed_date and effective != parsed_date
                    else f" (rates from {effective.isoformat()})"
                )

                rounded = converted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                rate = (converted / Decimal(str(amount))).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                ) if amount else Decimal("0")

                return (
                    f"{amount:,.2f} {src} = {rounded:,.2f} {dst}"
                    f"{fallback_note}\n"
                    f"Rate: 1 {src} = {rate} {dst}"
                )
            except ValueError as e:
                return f"Conversion error: {e}"
            except LookupError as e:
                return f"No rates available: {e}"
            except Exception as e:  # network / parse failures
                return f"Error fetching ECB rates: {str(e)[:300]}"

    class _RatesInput(BaseModel):
        target_date: Optional[str] = Field(
            default=None,
            description="ISO date YYYY-MM-DD. Defaults to today.",
        )
        currencies: Optional[str] = Field(
            default=None,
            description="Comma-separated currency codes to filter (e.g. 'USD,GBP,JPY'). Default: all.",
        )

    class CurrencyRatesTool(BaseTool):
        name: str = "currency_rates"
        description: str = (
            "Get ECB reference exchange rates for a given date. Returns rates "
            "quoted as 1 EUR = X currency. Use this to inspect available currencies "
            "or compare multiple currencies at once."
        )
        args_schema: Type[BaseModel] = _RatesInput

        def _run(
            self,
            target_date: Optional[str] = None,
            currencies: Optional[str] = None,
        ) -> str:
            try:
                parsed_date: Optional[date] = None
                if target_date:
                    try:
                        parsed_date = date.fromisoformat(target_date.strip())
                    except ValueError:
                        return f"Invalid date '{target_date}'. Use ISO format YYYY-MM-DD."

                effective, rates = get_ecb_rates(parsed_date)

                if currencies:
                    wanted = {c.strip().upper() for c in currencies.split(",") if c.strip()}
                    filtered = {k: v for k, v in rates.items() if k in wanted}
                    missing = wanted - filtered.keys()
                    if missing:
                        return (
                            f"Currencies not in ECB rates for {effective.isoformat()}: "
                            f"{', '.join(sorted(missing))}"
                        )
                    rates = filtered

                lines = [f"=== ECB rates ({effective.isoformat()}, base EUR) ==="]
                for cur in sorted(rates):
                    lines.append(f"  1 EUR = {rates[cur]} {cur}")
                return "\n".join(lines)
            except ValueError as e:
                return f"Error: {e}"
            except LookupError as e:
                return f"No rates available: {e}"
            except Exception as e:
                return f"Error fetching ECB rates: {str(e)[:300]}"

    return [CurrencyConvertTool(), CurrencyRatesTool()]


# ── Tool registry annotation (Phase 1a, passive) ────────────────────
try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="currency_convert",
        capabilities=["converts-currency", "fetches-finance"],
        description=(
            "Convert an amount between two currencies using ECB daily "
            "reference rates (cached). Returns the converted amount + "
            "the rate used. Use for any monetary calculation that "
            "crosses a currency boundary."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _currency_convert_registry_factory(agent_id: str = "coder"):
        tools = create_currency_tools(agent_id=agent_id)
        for t in tools:
            if t.name == "currency_convert":
                return t
        raise RuntimeError("currency_convert factory could not find tool")

    @register_tool(
        name="currency_rates",
        capabilities=["fetches-finance"],
        description=(
            "List current ECB exchange rates for one or more "
            "currencies. Useful as a reference / quick lookup; "
            "for actual conversion use `currency_convert`."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _currency_rates_registry_factory(agent_id: str = "coder"):
        tools = create_currency_tools(agent_id=agent_id)
        for t in tools:
            if t.name == "currency_rates":
                return t
        raise RuntimeError("currency_rates factory could not find tool")
except ImportError:
    pass
