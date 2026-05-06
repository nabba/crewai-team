"""sections — section specs for the dossier report.

Each :class:`SectionSpec` defines:
  * ``key`` — slug used in TOC / fact-check telemetry
  * ``title`` — what the typesetter renders as the section header
  * ``slice_fields`` — which dossier field names are relevant
  * ``render_facts`` — how to format the slice for the LLM prompt
  * ``prose_template`` — the instructions handed to the LLM

The discipline is: the LLM only sees the slice's facts.  It cannot
reach into the broader dossier.  This is what prevents
cross-contamination (e.g. the workforce section accidentally
fabricating revenue numbers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.dossier.schema import CompanyDossier, DossierField


@dataclass(frozen=True)
class SectionSpec:
    """Static metadata for one report section."""

    key: str
    title: str
    slice_fields: tuple[str, ...]
    instructions: str
    """LLM-facing instructions describing the section's voice + scope."""
    target_word_count: int = 250
    """Soft target for the section's prose length."""


# ── Slice rendering ──────────────────────────────────────────────────


def render_slice(dossier: CompanyDossier, fields: tuple[str, ...]) -> str:
    """Format a dossier slice as a fact list for the prompt.

    Format:

        [field_name] = value (source: adapter, confidence: HIGH, as_of: 2024-12-31)
                            note: "Revenues, FY2024 10-K"

    A field that is NOT_DISCLOSED is rendered as
        [field_name] = NOT_DISCLOSED (reason: ...)

    UNRESOLVED fields are rendered as
        [field_name] = UNAVAILABLE (no source produced a value)

    so the LLM has explicit signal about what is missing rather than
    inferring absence by silence.
    """
    lines: list[str] = []
    for fname in fields:
        if not hasattr(dossier, fname):
            continue
        value = getattr(dossier, fname)
        if not isinstance(value, DossierField):
            continue
        lines.append(_render_one(fname, value))
    return "\n".join(lines)


def _render_one(name: str, dfield: DossierField) -> str:
    """Render a single field for the LLM prompt."""
    if dfield.is_known:
        rendered = dfield.render_value()
        src = dfield.source.adapter if dfield.source else "unknown"
        as_of = (
            f", as_of: {dfield.as_of.isoformat()}" if dfield.as_of else ""
        )
        line = (
            f"[{name}] = {rendered} "
            f"(source: {src}, confidence: {dfield.confidence.value}{as_of})"
        )
        if dfield.source and dfield.source.note:
            line += f"\n    note: {dfield.source.note}"
        if dfield.conflicts:
            for c in dfield.conflicts:
                line += (
                    f"\n    conflict: {c.value} "
                    f"({c.source.adapter}, {c.confidence.value})"
                )
        return line
    if dfield.status.value == "not_disclosed":
        return f"[{name}] = NOT_DISCLOSED (reason: {dfield.reason})"
    if dfield.status.value == "not_applicable":
        return f"[{name}] = NOT_APPLICABLE"
    return f"[{name}] = UNAVAILABLE (no source produced a value)"


# ── The catalog ──────────────────────────────────────────────────────


_BASE_RULES = """\
RULES (binding):
1. Use ONLY facts present in the dossier slice below.
2. Cite every numeric / date claim by referencing its bracketed name,
   e.g. "Revenue of $12.5B [revenue_usd]". Do not paraphrase numbers.
3. If a field is NOT_DISCLOSED, write "not disclosed" — do not estimate.
4. If a field is UNAVAILABLE, write "data unavailable" — do not invent
   plausible-looking values.
5. When sources conflict (the slice shows a "conflict:" line), name
   both sources rather than silently picking one.
6. Maintain investment-research voice: declarative, dispassionate, no
   marketing language. Avoid superlatives without explicit evidence.
7. Never refer to an analyst, opinion, or context not in the slice.
"""


SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        key="executive_summary",
        title="Executive Summary",
        slice_fields=(
            "legal_name", "description", "founded_on", "headquarters",
            "industry_codes", "employee_count", "revenue_usd", "ebitda_usd",
            "market_cap_usd", "total_funding_usd",
        ),
        instructions=(
            "Write a 5-7 sentence executive summary covering: what the "
            "company does, when and where it was founded, current scale "
            "(headcount + revenue), capital structure (market cap or "
            "total funding), and the single most important fact a "
            "decision-maker would want to know first.\n"
            + _BASE_RULES
        ),
        target_word_count=180,
    ),

    SectionSpec(
        key="history",
        title="Company History",
        slice_fields=(
            "founded_on", "founders", "milestones", "incorporated_in",
            "legal_name", "description", "headquarters",
        ),
        instructions=(
            "Write a chronological narrative of the company's history.\n"
            "Open with founding (date, founders, place of incorporation).\n"
            "Walk through major milestones IN ORDER. If milestone dates "
            "aren't in the slice, narrate by event without inventing dates.\n"
            "Close with the present-day legal entity name and HQ.\n\n"
            + _BASE_RULES
        ),
        target_word_count=350,
    ),

    SectionSpec(
        key="business_model",
        title="Business Model and Products",
        slice_fields=(
            "business_model", "products_services", "geographic_markets",
            "industry_codes", "description",
        ),
        instructions=(
            "Describe the company's revenue model: who pays, for what, "
            "in which geographies. If business_model is UNAVAILABLE but "
            "description provides hints, infer cautiously and label the "
            "inference. Cover product/service lines if known.\n\n"
            + _BASE_RULES
        ),
        target_word_count=300,
    ),

    SectionSpec(
        key="financials",
        title="Financial Profile",
        slice_fields=(
            "revenue_usd", "revenue_growth_yoy", "gross_profit_usd",
            "ebitda_usd", "net_income_usd", "fiscal_year_end",
            "market_cap_usd", "enterprise_value_usd", "pe_ratio",
            "ev_ebitda",
        ),
        instructions=(
            "Present the financial profile in TWO subsections:\n"
            "  * Operating performance (revenue, growth, gross profit, "
            "EBITDA, net income)\n"
            "  * Market valuation (market cap, enterprise value, P/E, "
            "EV/EBITDA), if applicable\n"
            "Always cite the fiscal year end. If the company is private "
            "and market valuation fields are UNAVAILABLE, omit the "
            "valuation subsection rather than fabricating.\n\n"
            + _BASE_RULES
        ),
        target_word_count=400,
    ),

    SectionSpec(
        key="customers_traffic",
        title="Customers, Users, and Traffic",
        slice_fields=(
            "customer_count", "monthly_active_users",
            "web_visits_monthly", "notable_customers",
            "description", "geographic_markets",
        ),
        instructions=(
            "Quantify the company's user base and reach: customer count, "
            "monthly active users, web visits, and notable named "
            "customers. If multiple are UNAVAILABLE, note that user "
            "metrics are not publicly disclosed and reference whichever "
            "geographies (geographic_markets) are known.\n\n"
            + _BASE_RULES
        ),
        target_word_count=250,
    ),

    SectionSpec(
        key="workforce",
        title="Workforce and Compensation",
        slice_fields=(
            "employee_count", "avg_salary_usd", "leadership",
            "headquarters",
        ),
        instructions=(
            "Report headcount and (when known) average compensation. "
            "List named leadership when in the slice. Do NOT estimate "
            "salaries — if avg_salary_usd is UNAVAILABLE, state that "
            "compensation data is not disclosed.\n\n"
            + _BASE_RULES
        ),
        target_word_count=200,
    ),

    SectionSpec(
        key="ownership_funding",
        title="Ownership and Funding",
        slice_fields=(
            "owners", "funding_rounds", "total_funding_usd",
            "market_cap_usd", "company_type",
        ),
        instructions=(
            "Describe the capital structure: ownership (cap-table for "
            "private companies, major shareholders for public), funding "
            "rounds in chronological order, total capital raised. If the "
            "company is public, note that public-float ownership is "
            "diversified rather than listing every 13F filer.\n\n"
            + _BASE_RULES
        ),
        target_word_count=300,
    ),

    SectionSpec(
        key="risks",
        title="Risks and Limitations",
        slice_fields=(),  # filled separately from coverage_report
        instructions=(
            "Summarise the limitations of THIS report: which fields had "
            "low-confidence sources, which were UNAVAILABLE, and which "
            "had conflicting source values. Do NOT speculate on company "
            "risks — only the data-quality risks of the report itself. "
            "Reference the coverage report attached below.\n\n"
            + _BASE_RULES
        ),
        target_word_count=200,
    ),
)


# ── Comparator section is separate (uses peer dossiers) ─────────────


def render_peer_slice(
    focal: CompanyDossier,
    peers: list[CompanyDossier],
    fields: tuple[str, ...] = (
        "revenue_usd", "ebitda_usd", "employee_count",
        "market_cap_usd", "ev_ebitda",
    ),
) -> str:
    """Render the focal company + each peer side-by-side as a table-of-facts."""
    if not peers:
        return (
            "PEER COMPARISON: peer set unavailable. The focal company "
            "lacks an industry classification (SIC/NAICS) sufficient "
            "for confident peer selection. The comparator section "
            "should be omitted from the prose."
        )
    lines = ["PEER COMPARISON SLICE:"]
    lines.append(f"\nFOCAL: {focal.ref.name} ({focal.ref.ticker or 'private'})")
    lines.append(render_slice(focal, fields))
    for i, peer in enumerate(peers, start=1):
        lines.append(
            f"\nPEER {i}: {peer.ref.name} ({peer.ref.ticker or 'private'})"
        )
        lines.append(render_slice(peer, fields))
    return "\n".join(lines)


COMPARATOR_SECTION = SectionSpec(
    key="comparator",
    title="Competitor Comparison",
    slice_fields=(),  # filled by render_peer_slice
    instructions=(
        "Compare the focal company to each named peer on revenue, "
        "EBITDA, headcount, and (when public) market cap and EV/EBITDA. "
        "Highlight where the focal sits in the distribution. NEVER "
        "invent peer financials not in the slice. If a metric is "
        "UNAVAILABLE for a peer, omit that comparison cell — do not "
        "infer.\n\n"
        + _BASE_RULES
    ),
    target_word_count=350,
)
