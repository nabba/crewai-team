"""schema — typed CompanyDossier with per-field provenance.

The schema is the contract between data collection (adapters), peer
selection, prose composition, and typesetting.  Every layer below the
contract ONLY knows about :class:`CompanyDossier`; nothing reaches
across — that is the discipline that keeps the prose layer from
inventing numbers.

Design notes
============

Why every field is wrapped in :class:`DossierField`
---------------------------------------------------
"Investment-grade" requires source-traceable assertions.  A bare
``revenue: float`` field can't distinguish "audited 10-K figure"
from "Crunchbase estimate" from "we don't know."  Wrapping the value
forces every consumer (composer, typesetter, fact-checker) to handle
all four cases — known / not-disclosed / not-applicable / unresolved
— uniformly.

Why ``FieldStatus`` instead of ``Optional[T]``
----------------------------------------------
``None`` is overloaded — it can mean "data exists but we couldn't get
it" or "doesn't apply to this company type."  The distinction matters
for the report (the latter is silent in the prose; the former is
flagged as a gap).  Explicit status > nullable.

Why values are JSON-friendly primitives + small containers
----------------------------------------------------------
The dossier is serialised to the epistemic ledger (one Claim per
populated field) and round-tripped via the typesetter.  Pydantic's
JSON-mode + the explicit ``as_jsonable`` / ``from_jsonable`` round-trip
keeps the boundary cheap and inspectable.

Reconciliation contract
-----------------------
When two adapters fill the same field with different values, the
collector keeps the higher-priority source and records the loser in
``DossierField.conflicts``.  The composer surfaces conflicts as
"Sources disagreed: Crunchbase reported X; LinkedIn estimated Y"
rather than silently picking.
"""
from __future__ import annotations

import enum
from datetime import date, datetime, timezone
from typing import Any, Generic, Iterable, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ── Enums ────────────────────────────────────────────────────────────


class FieldStatus(str, enum.Enum):
    """Why a field has (or lacks) a value.

    Composers must treat each status differently:

    * ``KNOWN``         — render the value, cite the source
    * ``NOT_DISCLOSED`` — render "not disclosed" with a confidence note
    * ``NOT_APPLICABLE``— omit silently (e.g. SEC filings for an EU private company)
    * ``UNRESOLVED``    — render "data unavailable" + reason; flag in appendix
    """

    KNOWN = "known"
    NOT_DISCLOSED = "not_disclosed"
    NOT_APPLICABLE = "not_applicable"
    UNRESOLVED = "unresolved"


class Confidence(str, enum.Enum):
    """Calibrated confidence buckets.

    Mapped to the epistemic ledger's evidence confidence as:

    * ``EXACT``      → 1.0  (issuer-reported in a regulatory filing)
    * ``HIGH``       → 0.9  (issuer-reported elsewhere; canonical registry)
    * ``MEDIUM``     → 0.7  (curated third-party DB; verifiable cross-source)
    * ``LOW``        → 0.4  (single web source, scraped, or estimated)
    * ``ESTIMATED``  → 0.3  (model inferred from indirect evidence)
    """

    EXACT = "exact"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

    def to_evidence_confidence(self) -> float:
        return {
            Confidence.EXACT: 1.0,
            Confidence.HIGH: 0.9,
            Confidence.MEDIUM: 0.7,
            Confidence.LOW: 0.4,
            Confidence.ESTIMATED: 0.3,
        }[self]


class CompanyType(str, enum.Enum):
    """High-level company shape that gates which adapters apply."""

    PUBLIC = "public"
    PRIVATE_VC = "private_vc"
    PRIVATE_MATURE = "private_mature"
    SUBSIDIARY = "subsidiary"
    UNKNOWN = "unknown"


# ── Core wrapper ─────────────────────────────────────────────────────


class Source(BaseModel):
    """Where a value came from.

    The ``adapter`` field is the name of the dossier adapter that
    produced the value (e.g. ``"sec_edgar"``, ``"companies_house"``).
    The ``url`` is the human-verifiable link the typesetter cites in
    the source appendix.  ``accessed_at`` makes the report auditable
    even years later when the URL has rotted.
    """

    model_config = ConfigDict(frozen=True)

    adapter: str
    url: str = ""
    document_id: str = ""  # e.g. SEC accession, Crunchbase UUID
    accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    note: str = ""  # short clarifier ("FY2024 10-K, page 47")


class DossierField(BaseModel, Generic[T]):
    """Typed value with provenance.

    The generic parameter ``T`` is the value type (str, int, float,
    date, list[Owner], …).  Pydantic v2 accepts the generic parameter
    via ``DossierField[int]``-style annotations on the parent model.

    Use the ``known() / not_disclosed() / unresolved() /
    not_applicable()`` factories rather than constructing directly —
    they enforce the status/value invariant that the composer relies on.
    """

    model_config = ConfigDict(frozen=False)  # collector mutates conflicts

    status: FieldStatus
    value: T | None = None
    source: Source | None = None
    confidence: Confidence = Confidence.LOW
    as_of: date | None = None  # the date the value is true *as of*
    reason: str = ""           # why UNRESOLVED / NOT_DISCLOSED
    conflicts: list["FieldConflict"] = Field(default_factory=list)

    # ── factories ────────────────────────────────────────────────────

    @classmethod
    def known(
        cls,
        value: T,
        *,
        source: Source,
        confidence: Confidence = Confidence.MEDIUM,
        as_of: date | None = None,
    ) -> "DossierField[T]":
        return cls(
            status=FieldStatus.KNOWN,
            value=value,
            source=source,
            confidence=confidence,
            as_of=as_of,
        )

    @classmethod
    def not_disclosed(
        cls, *, reason: str = "", source: Source | None = None,
    ) -> "DossierField[T]":
        return cls(
            status=FieldStatus.NOT_DISCLOSED,
            reason=reason or "issuer does not disclose",
            source=source,
        )

    @classmethod
    def not_applicable(cls, *, reason: str = "") -> "DossierField[T]":
        return cls(
            status=FieldStatus.NOT_APPLICABLE,
            reason=reason or "field does not apply to this company type",
        )

    @classmethod
    def unresolved(cls, *, reason: str = "") -> "DossierField[T]":
        return cls(
            status=FieldStatus.UNRESOLVED,
            reason=reason or "no source produced a value",
        )

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def is_known(self) -> bool:
        return self.status == FieldStatus.KNOWN

    def render_value(self) -> str:
        """User-facing rendering for the prose layer.

        Composers MUST use this rather than touching ``value`` directly
        so the four statuses render uniformly across sections.
        """
        if self.status == FieldStatus.KNOWN:
            return _format_value(self.value)
        if self.status == FieldStatus.NOT_DISCLOSED:
            return "not disclosed"
        if self.status == FieldStatus.NOT_APPLICABLE:
            return "not applicable"
        return "data unavailable"


class FieldConflict(BaseModel):
    """A losing value retained for transparency.

    When two adapters disagree, the collector picks the higher-priority
    one and stores the other here.  The composer surfaces conflicts in
    the prose ("Sources disagreed: Crunchbase reported X; LinkedIn
    estimated Y") rather than burying them.
    """

    model_config = ConfigDict(frozen=True)

    value: Any
    source: Source
    confidence: Confidence


# ── Sub-models ───────────────────────────────────────────────────────


class FundingRound(BaseModel):
    """One funding event (seed, Series A, debt, …).

    All amounts are in USD when known; if the adapter only has the
    native currency, the dossier collector converts via the existing
    ``currency_tools`` and stamps both into ``note``.
    """

    model_config = ConfigDict(frozen=True)

    round_type: str        # "seed", "series_a", "debt", "ipo", "secondary"
    amount_usd: float | None = None
    announced_on: date | None = None
    lead_investors: tuple[str, ...] = ()
    other_investors: tuple[str, ...] = ()
    valuation_usd: float | None = None
    note: str = ""


class Owner(BaseModel):
    """A shareholder / beneficial owner.

    For public companies this comes from 13F / DEF 14A.  For private
    companies it comes from cap-table-adjacent sources (Crunchbase
    investors list) or registry filings (Companies House PSC for UK).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str  # "founder", "vc", "pe", "strategic", "public_float", "individual"
    pct_ownership: float | None = None  # 0.0-1.0 — None when undisclosed
    note: str = ""


class PeerEntry(BaseModel):
    """A reference peer used in the comparator section.

    Selection criteria are recorded in ``selection_basis`` so the
    methodology is traceable in the report.  A weak basis ("same
    industry") is downweighted by the composer relative to a strong
    one ("same SIC + same revenue band + same geography").
    """

    model_config = ConfigDict(frozen=True)

    name: str
    ticker: str = ""
    selection_basis: tuple[str, ...] = ()


# ── The dossier ──────────────────────────────────────────────────────


class CompanyRef(BaseModel):
    """The minimum identity used to look the company up.

    Adapters may need ANY of these to find the company in their
    source:
      * ``ticker`` for SEC EDGAR / yfinance
      * ``website_domain`` for Crunchbase / SimilarWeb / Apollo
      * ``companies_house_number`` for the UK registry
      * ``wikidata_id`` for Wikidata
      * ``crunchbase_uuid`` after the first Crunchbase resolve

    The collector progressively enriches this ref as adapters discover
    new identifiers.
    """

    model_config = ConfigDict(frozen=False)

    name: str
    ticker: str = ""
    website_domain: str = ""
    companies_house_number: str = ""
    wikidata_id: str = ""
    crunchbase_uuid: str = ""
    country_hint: str = ""


class CompanyDossier(BaseModel):
    """The contract.

    Every report section reads from this model and ONLY this model.
    The prose composer cannot invent a fact: ``DossierField.is_known``
    is False → the prose says "not disclosed."

    Field organisation mirrors the report sections:
      * identity
      * history
      * business model
      * market & TAM
      * financials
      * customers / users / traffic
      * workforce & compensation
      * ownership & funding
      * competitor comparison (filled in Phase 4 — peer selection)
    """

    model_config = ConfigDict(frozen=False)

    # ── identity ─────────────────────────────────────────────────────
    ref: CompanyRef
    company_type: CompanyField_CompanyType = Field(
        default_factory=lambda: DossierField[CompanyType].unresolved(),
    )
    legal_name: CompanyField_str = Field(
        default_factory=lambda: DossierField[str].unresolved(),
    )
    description: CompanyField_str = Field(
        default_factory=lambda: DossierField[str].unresolved(),
    )
    headquarters: CompanyField_str = Field(
        default_factory=lambda: DossierField[str].unresolved(),
    )
    incorporated_in: CompanyField_str = Field(
        default_factory=lambda: DossierField[str].unresolved(),
    )
    industry_codes: CompanyField_StrTuple = Field(
        default_factory=lambda: DossierField[tuple[str, ...]].unresolved(),
    )
    website_url: CompanyField_str = Field(
        default_factory=lambda: DossierField[str].unresolved(),
    )

    # ── history ──────────────────────────────────────────────────────
    founded_on: CompanyField_date = Field(
        default_factory=lambda: DossierField[date].unresolved(),
    )
    founders: CompanyField_StrTuple = Field(
        default_factory=lambda: DossierField[tuple[str, ...]].unresolved(),
    )
    milestones: CompanyField_StrTuple = Field(
        default_factory=lambda: DossierField[tuple[str, ...]].unresolved(),
    )

    # ── business model ───────────────────────────────────────────────
    business_model: CompanyField_str = Field(
        default_factory=lambda: DossierField[str].unresolved(),
    )
    products_services: CompanyField_StrTuple = Field(
        default_factory=lambda: DossierField[tuple[str, ...]].unresolved(),
    )
    geographic_markets: CompanyField_StrTuple = Field(
        default_factory=lambda: DossierField[tuple[str, ...]].unresolved(),
    )

    # ── financials ───────────────────────────────────────────────────
    revenue_usd: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    revenue_growth_yoy: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    gross_profit_usd: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    ebitda_usd: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    net_income_usd: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    fiscal_year_end: CompanyField_str = Field(
        default_factory=lambda: DossierField[str].unresolved(),
    )

    # ── market data (public co.) ─────────────────────────────────────
    market_cap_usd: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    enterprise_value_usd: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    pe_ratio: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    ev_ebitda: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )

    # ── customers / users / traffic ──────────────────────────────────
    customer_count: CompanyField_int = Field(
        default_factory=lambda: DossierField[int].unresolved(),
    )
    monthly_active_users: CompanyField_int = Field(
        default_factory=lambda: DossierField[int].unresolved(),
    )
    web_visits_monthly: CompanyField_int = Field(
        default_factory=lambda: DossierField[int].unresolved(),
    )
    notable_customers: CompanyField_StrTuple = Field(
        default_factory=lambda: DossierField[tuple[str, ...]].unresolved(),
    )

    # ── workforce ────────────────────────────────────────────────────
    employee_count: CompanyField_int = Field(
        default_factory=lambda: DossierField[int].unresolved(),
    )
    avg_salary_usd: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )
    leadership: CompanyField_StrTuple = Field(
        default_factory=lambda: DossierField[tuple[str, ...]].unresolved(),
    )

    # ── ownership & funding ──────────────────────────────────────────
    owners: CompanyField_OwnerTuple = Field(
        default_factory=lambda: DossierField[tuple[Owner, ...]].unresolved(),
    )
    funding_rounds: CompanyField_FundingTuple = Field(
        default_factory=lambda: DossierField[tuple[FundingRound, ...]].unresolved(),
    )
    total_funding_usd: CompanyField_float = Field(
        default_factory=lambda: DossierField[float].unresolved(),
    )

    # ── peers (filled in Phase 4) ────────────────────────────────────
    peers: list["PeerEntry"] = Field(default_factory=list)

    # ── meta ─────────────────────────────────────────────────────────
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    coverage_report: dict[str, Any] = Field(default_factory=dict)
    """Summary of which adapters fired, which fields are missing, why.
       Populated by the collector; rendered into the source appendix."""

    # ── helpers ──────────────────────────────────────────────────────

    def known_field_count(self) -> int:
        """How many fields have a KNOWN status — used for coverage scoring."""
        count = 0
        for fname, _ in self.iter_fields():
            value = getattr(self, fname)
            if isinstance(value, DossierField) and value.is_known:
                count += 1
        return count

    def total_field_count(self) -> int:
        """Total number of DossierField slots (denominator for coverage %)."""
        return sum(1 for _ in self.iter_fields())

    def coverage_pct(self) -> float:
        total = self.total_field_count()
        return (self.known_field_count() / total) if total else 0.0

    def iter_fields(self) -> Iterable[tuple[str, DossierField]]:
        """Yield every (name, DossierField) on the dossier.

        Skips ``ref``, ``peers``, ``generated_at``, ``coverage_report``
        — those aren't typed dossier fields.
        """
        for name, info in type(self).model_fields.items():
            value = getattr(self, name)
            if isinstance(value, DossierField):
                yield name, value


# ── Concrete generic aliases for pydantic v2 ─────────────────────────
#
# Pydantic v2 needs concrete type annotations on each field; the
# ``DossierField[T]`` shorthand only works in user code because we
# pre-declare the parameterizations here.  This keeps the schema
# readable while satisfying pydantic's strict type system.

CompanyField_str = DossierField[str]
CompanyField_int = DossierField[int]
CompanyField_float = DossierField[float]
CompanyField_date = DossierField[date]
CompanyField_StrTuple = DossierField[tuple[str, ...]]
CompanyField_FundingTuple = DossierField[tuple[FundingRound, ...]]
CompanyField_OwnerTuple = DossierField[tuple[Owner, ...]]
CompanyField_CompanyType = DossierField[CompanyType]


# ── Value formatting ─────────────────────────────────────────────────


def _format_value(value: Any) -> str:
    """Stable string rendering used by ``DossierField.render_value``.

    Centralised so the typesetter, composer, fact-checker, and
    Signal-friendly summariser all render values identically.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        if abs(value) >= 1e6:
            return f"${value/1e6:.1f}M"
        if abs(value) >= 1e3:
            return f"${value/1e3:.1f}K"
        return f"{value:.2f}"
    if isinstance(value, int):
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"{value/1_000:.1f}K"
        return f"{value:,}"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (tuple, list)):
        return ", ".join(_format_value(v) for v in value)
    if isinstance(value, BaseModel):
        # Best-effort short rendering for sub-models; the prose layer
        # has dedicated formatters for FundingRound / Owner.
        return value.model_dump_json()
    return str(value)


# ── Reconciliation ───────────────────────────────────────────────────


def merge_field(
    *,
    existing: DossierField,
    new_value: Any,
    source: Source,
    confidence: Confidence,
    as_of: date | None,
    source_priority: Mapping[str, int],
) -> DossierField:
    """Decide whether ``new_value`` replaces ``existing``.

    Higher source priority wins.  When priority ties, higher confidence
    wins.  When both tie, the existing value wins (stability — the
    first source to fill a field "anchors" it absent stronger evidence).
    The losing value is recorded in ``conflicts`` regardless of which
    one wins, so the composer can surface disagreement.

    ``source_priority`` is the same map the collector uses, e.g.::

        {"sec_edgar": 100, "companies_house": 95, "crunchbase": 80,
         "wikidata": 60, "wikipedia": 40, "web": 20}
    """
    new_priority = source_priority.get(source.adapter, 0)
    if not existing.is_known:
        return existing.model_copy(update={
            "status": FieldStatus.KNOWN,
            "value": new_value,
            "source": source,
            "confidence": confidence,
            "as_of": as_of,
        })

    existing_priority = (
        source_priority.get(existing.source.adapter, 0)
        if existing.source else 0
    )

    new_wins = (
        new_priority > existing_priority
        or (
            new_priority == existing_priority
            and confidence.to_evidence_confidence()
            > existing.confidence.to_evidence_confidence()
        )
    )
    loser_value = existing.value if new_wins else new_value
    loser_source = existing.source if new_wins else source
    loser_conf = existing.confidence if new_wins else confidence
    if loser_source is None:
        return existing  # nothing to record; keep the winner

    conflict = FieldConflict(
        value=loser_value, source=loser_source, confidence=loser_conf,
    )
    if new_wins:
        return existing.model_copy(update={
            "value": new_value,
            "source": source,
            "confidence": confidence,
            "as_of": as_of,
            "conflicts": [*existing.conflicts, conflict],
        })
    return existing.model_copy(update={
        "conflicts": [*existing.conflicts, conflict],
    })


# Forward-ref resolution (PeerEntry is defined above CompanyDossier).
CompanyDossier.model_rebuild()
DossierField.model_rebuild()
