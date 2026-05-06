# Company Dossier subsystem

The reference for how the system produces an investment-grade,
source-attributed company dossier (10–15 page PDF) in one shot from a
natural-language request like *"build me an investment-grade overview
of Spotify (SPOT)"*.

Distinct from the open-ended `financial` crew (analyst chat, ad-hoc
ratios, DCF requests). The dossier subsystem is a **deterministic
pipeline** with the LLM scoped to prose composition only — it cannot
choose adapters, interpret raw numbers, or invent any fact.

---

## 1. Why this design exists

A typical "agent assembles a company report" flow has three failure modes
that compound:

1. **Hallucinated numbers.** Free-form LLM composition over scraped web
   snippets reliably invents revenue figures, headcount, founding dates.
   The fact looks plausible. The reader has no way to verify.
2. **No comparability.** Without a fixed schema the system might pull
   "revenue" for company A and "ARR" for company B and quietly compare
   them, producing a peer table that is mathematically wrong.
3. **No auditability.** When an investor asks "where did this number
   come from?", an open-ended pipeline can't answer.

The dossier flips the ordering: a typed structured `CompanyDossier` is
the **contract** between data and prose. Every populated field carries
its source URL, confidence band, and as-of date. The composition layer
is a strict projection — it cannot reach outside the dossier — and a
post-composition fact-check pass extracts every number and date from
the prose and verifies it against the dossier slice.

This pattern is borrowed from regulatory filing review: the financial
statements are the source of truth; the management discussion is a
projection over them; the auditor verifies that projection.

---

## 2. Architecture (one screen)

```
                  natural-language request ("build a dossier for X")
                              │
                              ▼
                  ┌─────────────────────────┐
                  │   identity parser       │   regex-based; falls
                  │ (pipeline.parse_…)      │   back gracefully
                  └────────────┬────────────┘
                               │  CompanyRef(name, ticker, …)
                               ▼
              ┌──────────────────────────────────┐
              │   collector.collect_dossier      │   parallel adapter
              │   • adapters in priority order   │   dispatch with
              │   • circuit breakers + timeouts  │   per-call timeout
              │   • iterative ref enrichment     │   + circuit breaker
              │   • merge_field reconciliation   │
              │   • emit each field as a Claim   │
              └────────────────┬─────────────────┘
                               │  CompanyDossier (typed + provenance)
                               ▼
              ┌──────────────────────────────────┐
              │   peers.select_peers             │   SIC/NAICS-based
              │   intersect-of-two-methods       │   peer selection
              └────────────────┬─────────────────┘
                               │  list[PeerEntry]
                               ▼  (collect_dossier per peer, parallel)
              ┌──────────────────────────────────┐
              │   compose.compose_report         │   section-by-section
              │   • slice = only relevant fields │   LLM call with
              │   • strict citation prompt       │   strict-citation
              │   • per-section fact-check       │   discipline
              └────────────────┬─────────────────┘
                               │  ComposedReport(sections, warnings)
                               ▼
              ┌──────────────────────────────────┐
              │   typeset.render_pdf             │   ReportLab Platypus
              │   cover, TOC, sections, source   │   multi-page renderer
              │   appendix, coverage appendix    │
              └────────────────┬─────────────────┘
                               │
                               ▼
                       /app/workspace/output/dossier_<slug>_<date>.pdf
```

---

## 3. Modules

| File | Responsibility |
|---|---|
| [`app/dossier/schema.py`](../app/dossier/schema.py) | `CompanyDossier`, `DossierField[T]`, `Source`, `Confidence`, `merge_field`. Source-of-truth for the field shape. |
| [`app/dossier/adapters/_base.py`](../app/dossier/adapters/_base.py) | `DossierAdapter` protocol; registry; HTTP helpers; per-process cache. |
| [`app/dossier/adapters/sec_edgar.py`](../app/dossier/adapters/sec_edgar.py) | SEC EDGAR companyfacts (XBRL parsing) — revenue, net income, employees, EBITDA derived. |
| [`app/dossier/adapters/wikidata.py`](../app/dossier/adapters/wikidata.py) | Wikidata search-entity → SPARQL. Founding date, founders, HQ, ticker, ISIN, website. |
| [`app/dossier/adapters/wikipedia.py`](../app/dossier/adapters/wikipedia.py) | Wikipedia REST. Description (intro paragraph) + milestone hints from section headings. |
| [`app/dossier/adapters/yfinance_market.py`](../app/dossier/adapters/yfinance_market.py) | Live market data: market cap, P/E, EV/EBITDA. |
| [`app/dossier/adapters/companies_house.py`](../app/dossier/adapters/companies_house.py) | UK statutory registry: profile, PSC ownership, registered office. Requires `COMPANIES_HOUSE_API_KEY`. |
| [`app/dossier/adapters/web_fallback.py`](../app/dossier/adapters/web_fallback.py) | Last-resort website + description via existing `web_search` infrastructure. Lowest priority. |
| [`app/dossier/collector.py`](../app/dossier/collector.py) | Parallel adapter dispatch, circuit breakers, ref enrichment over 2 passes, emits each populated field as a `Claim` in the epistemic ledger. |
| [`app/dossier/peers.py`](../app/dossier/peers.py) | SIC-based peer-set selection (intersect-of-two-methods). Honest empty list when peer selection isn't reliable. |
| [`app/dossier/sections.py`](../app/dossier/sections.py) | Section catalog: 8 standard sections + comparator. Per-section slice fields + strict-citation rules. |
| [`app/dossier/compose.py`](../app/dossier/compose.py) | Section-by-section LLM composition. Fact-check pass extracts numeric tokens (currency, percent, year, comma-int) and validates against slice. Falls back to deterministic slice-echo composer when no LLM. |
| [`app/dossier/typeset.py`](../app/dossier/typeset.py) | ReportLab Platypus multi-page renderer. Reuses `pdf_compose._RL_PACK` cached imports. |
| [`app/dossier/pipeline.py`](../app/dossier/pipeline.py) | Top-level `build_dossier` entry point + identity parser. |
| [`app/dossier/tools.py`](../app/dossier/tools.py) | CrewAI `BaseTool` wrapper + `@register_tool` decoration for `tool_search` discovery. |
| [`app/crews/dossier_crew.py`](../app/crews/dossier_crew.py) | Standard `DossierCrew().run(...)` shape; registered as `"company_dossier"` in [`app/crews/registry.py`](../app/crews/registry.py). |

---

## 4. Provenance and reconciliation

Every `DossierField` carries the source that filled it:

```python
DossierField(
    status=FieldStatus.KNOWN,
    value=12_500_000_000.0,
    source=Source(
        adapter="sec_edgar",
        url="https://www.sec.gov/cgi-bin/browse-edgar?CIK=0001639920",
        note="Revenues, FY2024 10-K",
    ),
    confidence=Confidence.EXACT,
    as_of=date(2024, 12, 31),
    conflicts=[
        FieldConflict(
            value=11_000_000_000.0,
            source=Source(adapter="wikidata", url="…"),
            confidence=Confidence.MEDIUM,
        ),
    ],
)
```

When two adapters fill the same field with different values,
`schema.merge_field` picks the higher-priority source and records the
loser in `conflicts`. Source priority (numeric, higher wins) is built
from the adapter registry:

| Adapter | Priority | Why |
|---|---:|---|
| `sec_edgar` | 100 | Regulator-grade, audited |
| `companies_house` | 95 | Statutory registry |
| `yfinance_market` | 85 | Live exchange-derived; redistributor |
| `wikidata` | 60 | Curated third-party DB |
| `wikipedia` | 40 | Community-edited |
| `web_fallback` | 20 | Last resort |

`Confidence` is calibrated so it maps to the epistemic ledger's
evidence-confidence: `EXACT=1.0, HIGH=0.9, MEDIUM=0.7, LOW=0.4,
ESTIMATED=0.3`. Every populated field becomes a `Claim` (with
`Evidence` pointing at the source URL) in the per-task ledger when a
`task_id` is supplied.

The composition layer cannot bypass provenance: each `slice_fields`
entry renders as `[field_name] = value (source: adapter, confidence:
HIGH, as_of: 2024-12-31)`, and the strict prompt forbids inventing
facts not in the slice. The fact-check pass enforces this with regex
extraction over currency, percent, year, and comma-grouped integers.

---

## 5. Routing

The Commander dispatches dossier requests via two layers:

1. **Fast-path regex** in [`app/agents/commander/routing.py`](../app/agents/commander/routing.py)
   — matches `dossier`, `due diligence`, `investment-grade`, `company
   profile/overview/review/report/brief`, `investor brief/report`. Runs
   *before* the financial fast-path so the bare word "investment"
   doesn't steal dossier requests.
2. **LLM routing catalog** — `_CREW_BASE_PURPOSE["company_dossier"]`
   describes the crew so the LLM router picks it for less-canonical
   phrasings.

Both layers route to the `"company_dossier"` crew name registered in
[`app/crews/registry.py`](../app/crews/registry.py).

Regression-protected by [`tests/test_dossier_routing.py`](../tests/test_dossier_routing.py).

---

## 6. Coverage and limitations

The free-tier MVP (no paid API keys) covers:

- Public US/foreign-listed companies via SEC EDGAR + yfinance + Wikidata
- Public companies elsewhere via Wikidata + Wikipedia (limited)
- UK private companies via Companies House (requires
  `COMPANIES_HOUSE_API_KEY`)
- All companies via Wikipedia + web fallback (description, website)

Out of MVP scope (fields exist in the schema but stay `UNRESOLVED`
until a paid adapter is added):

- Funding rounds / cap table for VC-backed private companies
  (Crunchbase, PitchBook, Dealroom)
- Web traffic / MAU (SimilarWeb)
- Salary bands (Levels.fyi, Glassdoor)
- Mature private companies outside UK (OpenCorporates global tier)

Adding any of these is a single new file under `app/dossier/adapters/`
implementing the `DossierAdapter` protocol — see existing adapters as
templates.

---

## 7. How to invoke

**Via Commander / Signal** (auto-routed):
> "Build me an investment-grade overview of Spotify (SPOT)"
> "Due diligence on Tesla"
> "Company profile for Stripe"

**Via tool from any agent**:
```
tool_search("company dossier")
build_company_dossier({"query": "Spotify (SPOT)"})
```

**Programmatic**:
```python
from app.dossier.pipeline import build_dossier
build = build_dossier(query="Spotify (SPOT)")
print(build.summary())          # one-line summary
print(build.pdf_path)           # /app/workspace/output/dossier_….pdf
```

Override the output directory in dev/test environments via
`DOSSIER_OUTPUT_DIR=/some/path`. Production uses the standard
sandboxed `/app/workspace/output/`.

---

## 8. Tests

| File | Coverage |
|---|---|
| [`tests/test_dossier_schema.py`](../tests/test_dossier_schema.py) | `DossierField` factories, `merge_field` reconciliation, value rendering. |
| [`tests/test_dossier_collector.py`](../tests/test_dossier_collector.py) | Parallel collection with mocks, source-priority reconciliation, ref enrichment, error handling. |
| [`tests/test_dossier_compose.py`](../tests/test_dossier_compose.py) | Fact-check correctness (correctly-quoted vs invented numbers; no double-flag on overlap). |
| [`tests/test_dossier_pipeline.py`](../tests/test_dossier_pipeline.py) | End-to-end with mocks, identity parsing, progress callback. |
| [`tests/test_dossier_routing.py`](../tests/test_dossier_routing.py) | Commander routing — dossier-shaped queries land on `company_dossier`; financial queries still land on `financial`. |
