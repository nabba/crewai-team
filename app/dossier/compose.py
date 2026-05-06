"""compose — section-by-section LLM prose generation with fact-check.

Architecture
============
For each section spec:
  1. Build a slice (only the fields the section needs).
  2. Hand the slice + instructions to an LLM specialist call.
  3. Run the fact-check pass on the prose.
  4. Record the prose + warnings in a :class:`SectionOutput`.

Why not a single LLM call for the whole report?
-----------------------------------------------
Three reasons:
  * Token budget: a 10-15 page report exceeds typical specialist
    ``max_tokens=4096`` defaults; per-section calls each fit comfortably.
  * Containment: a hallucination in the financial section can't bleed
    into the workforce section because they don't share context.
  * Fact-check granularity: per-section warnings tell the user
    exactly which section has data-quality issues.

LLM integration
===============
Reuses :func:`app.llm_factory.create_specialist_llm` — the same
factory the writer / financial / research crews use.  A fallback
"plain-text echo" composer is shipped for environments where the
LLM stack isn't wired up (tests, dev sandbox).  This means the
typesetter can always render *something* — composition degrades to
"here are the facts in prose-ish form" rather than failing.

Fact-check pass
===============
After each section is generated, we extract every numeric token
(currency, percent, integer, year) and every date.  For each, we
check that an equivalent token appears in the section's slice.
Tokens that don't match anything in the slice are recorded as
warnings — the typesetter renders them in the "Risks and Limitations"
appendix so the user knows where to scrutinise.

This is a regex-level check, NOT semantic verification.  It catches
the most common failure (LLM invents "$15.2B revenue" when the slice
says $12.5B) without trying to be clever about paraphrasing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from app.dossier.schema import CompanyDossier, DossierField, _format_value
from app.dossier.sections import (
    COMPARATOR_SECTION,
    SECTIONS,
    SectionSpec,
    render_peer_slice,
    render_slice,
)

logger = logging.getLogger(__name__)


# ── Output envelope ──────────────────────────────────────────────────


@dataclass
class SectionOutput:
    """One composed section + telemetry."""

    key: str
    title: str
    prose: str
    fact_check_warnings: list[str] = field(default_factory=list)
    word_count: int = 0
    slice_text: str = ""
    """Full slice the LLM was given — useful for debugging and the
       source appendix."""


@dataclass
class ComposedReport:
    """All sections of a finished report."""

    company_name: str
    sections: list[SectionOutput] = field(default_factory=list)
    coverage_report: dict = field(default_factory=dict)


# ── Composer entry ───────────────────────────────────────────────────


def compose_report(
    focal_dossier: CompanyDossier,
    peer_dossiers: list[CompanyDossier] | None = None,
    *,
    llm_call: Callable[[str], str] | None = None,
) -> ComposedReport:
    """Compose every section of the report.

    Args:
        focal_dossier: The dossier produced by ``collector.collect_dossier``.
        peer_dossiers: Lite-dossiers for each peer.  When empty, the
            comparator section becomes a stub explaining why.
        llm_call: Override the default LLM specialist for testing.
            Signature: prompt → prose.

    Returns:
        A :class:`ComposedReport` containing every section + the
        upstream coverage report.
    """
    llm = llm_call or _default_llm_call
    out = ComposedReport(
        company_name=focal_dossier.ref.name,
        coverage_report=dict(focal_dossier.coverage_report),
    )

    # Standard sections.
    for spec in SECTIONS:
        section = _compose_one(spec, focal_dossier, llm)
        out.sections.append(section)

    # Comparator (special — uses peers).
    out.sections.append(_compose_comparator(
        COMPARATOR_SECTION, focal_dossier, peer_dossiers or [], llm,
    ))

    return out


def _compose_one(
    spec: SectionSpec,
    dossier: CompanyDossier,
    llm: Callable[[str], str],
) -> SectionOutput:
    if spec.key == "risks":
        slice_text = _render_risks_slice(dossier)
    else:
        slice_text = render_slice(dossier, spec.slice_fields)
    prompt = _build_prompt(spec, slice_text)
    prose = _safe_call_llm(llm, prompt)
    warnings = _fact_check(prose, dossier, spec.slice_fields)
    return SectionOutput(
        key=spec.key,
        title=spec.title,
        prose=prose,
        fact_check_warnings=warnings,
        word_count=len(prose.split()),
        slice_text=slice_text,
    )


def _compose_comparator(
    spec: SectionSpec,
    focal: CompanyDossier,
    peers: list[CompanyDossier],
    llm: Callable[[str], str],
) -> SectionOutput:
    slice_text = render_peer_slice(focal, peers)
    prompt = _build_prompt(spec, slice_text)
    prose = _safe_call_llm(llm, prompt)
    # Fact-check for comparator: extract numbers, check focal + every peer.
    all_dossiers = [focal] + peers
    warnings: list[str] = []
    for dossier in all_dossiers:
        warnings.extend(_fact_check(prose, dossier, spec.slice_fields or (
            "revenue_usd", "ebitda_usd", "employee_count",
            "market_cap_usd", "ev_ebitda",
        )))
    # Dedupe — the same warning may surface for the focal + peers.
    warnings = sorted(set(warnings))
    return SectionOutput(
        key=spec.key,
        title=spec.title,
        prose=prose,
        fact_check_warnings=warnings,
        word_count=len(prose.split()),
        slice_text=slice_text,
    )


# ── Prompt assembly ──────────────────────────────────────────────────


def _build_prompt(spec: SectionSpec, slice_text: str) -> str:
    """Assemble the per-section prompt."""
    return f"""You are composing one section of an investment-grade company \
report. Your output is the prose for the section "{spec.title}" — nothing else.

INSTRUCTIONS:
{spec.instructions}

TARGET LENGTH: ~{spec.target_word_count} words. A little under or over is fine.

DOSSIER SLICE (the only facts you may use):
{slice_text}

OUTPUT FORMAT:
Plain prose. No section header (the typesetter adds it). Use Markdown
for inline emphasis if needed.  No bullet lists unless the section
specifically calls for them.  Cite numeric / date facts with their
[bracketed_field_name] like the slice shows.
"""


# ── LLM integration ──────────────────────────────────────────────────


_LLM_INSTANCE = None


def _default_llm_call(prompt: str) -> str:
    """Use the existing specialist factory.  Wide ``max_tokens`` so a
    single section is comfortably within budget."""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        try:
            from app.llm_factory import create_specialist_llm
            _LLM_INSTANCE = create_specialist_llm(
                max_tokens=2400,  # ~1800 words ceiling
                role="writing",
            )
        except Exception as exc:
            logger.warning(
                "dossier.compose: specialist LLM unavailable (%s); "
                "falling back to slice-echo composer",
                exc,
            )
            return _slice_echo(prompt)
    try:
        return str(_LLM_INSTANCE.call(prompt)).strip()
    except Exception as exc:
        logger.warning("dossier.compose: LLM call failed: %s; "
                       "falling back to slice-echo", exc)
        return _slice_echo(prompt)


def _safe_call_llm(llm: Callable[[str], str], prompt: str) -> str:
    """Wrap LLM call so a single-section failure becomes a labelled
    placeholder rather than aborting the whole report."""
    try:
        result = llm(prompt) or ""
    except Exception as exc:
        logger.warning("dossier.compose: section LLM call raised %s", exc)
        return f"[section unavailable: LLM call failed — {type(exc).__name__}]"
    return result.strip() or "[section unavailable: empty LLM output]"


def _slice_echo(prompt: str) -> str:
    """Last-resort composer when no LLM is available.

    Extracts the slice from the prompt and renders it as plain
    paragraphs — gives the typesetter something to render without
    inventing facts.  Used in tests and broken-LLM environments.
    """
    marker = "DOSSIER SLICE (the only facts you may use):\n"
    if marker not in prompt:
        return ("[section unavailable: no LLM configured]")
    slice_text = prompt.split(marker, 1)[1].split("\n\nOUTPUT FORMAT:", 1)[0]
    # Render the slice as a "facts" paragraph — strip the bracketed
    # field names so it reads less robotic.
    lines = []
    for raw in slice_text.splitlines():
        if not raw.strip():
            continue
        # Convert "[name] = value (...)" to "name: value"
        m = re.match(r"\[(\w+)\]\s*=\s*(.+)", raw.strip())
        if m:
            name = m.group(1).replace("_", " ")
            value = m.group(2).split(" (", 1)[0]
            lines.append(f"{name}: {value}.")
    return " ".join(lines) or "[section unavailable: empty slice]"


# ── Risks slice (uses coverage_report rather than dossier fields) ───


def _render_risks_slice(dossier: CompanyDossier) -> str:
    """The risks section reads coverage telemetry, not field values."""
    cov = dossier.coverage_report or {}
    parts = ["COVERAGE REPORT (data-quality limitations):"]
    parts.append(f"  fields_filled: {cov.get('fields_filled', 0)}")
    parts.append(f"  fields_total: {cov.get('fields_total', 0)}")
    parts.append(f"  coverage_pct: {cov.get('coverage_pct', 0.0)}%")
    if cov.get("adapters_fired"):
        parts.append(f"  adapters_fired: {', '.join(cov['adapters_fired'])}")
    if cov.get("adapters_skipped"):
        parts.append("  adapters_skipped:")
        for k, v in (cov.get("adapters_skipped") or {}).items():
            parts.append(f"    - {k}: {v}")
    if cov.get("adapters_errored"):
        parts.append("  adapters_errored:")
        for k, v in (cov.get("adapters_errored") or {}).items():
            parts.append(f"    - {k}: {v}")
    # Surface fields with conflicts so the LLM can mention them.
    conflicts = [
        (name, dfield)
        for name, dfield in dossier.iter_fields()
        if dfield.is_known and dfield.conflicts
    ]
    if conflicts:
        parts.append("  fields_with_source_conflicts:")
        for name, dfield in conflicts:
            parts.append(
                f"    - {name}: {len(dfield.conflicts)} alternate value(s) "
                f"recorded"
            )
    # Surface low-confidence fields.
    low_conf = [
        name for name, dfield in dossier.iter_fields()
        if dfield.is_known and dfield.confidence.value in ("low", "estimated")
    ]
    if low_conf:
        parts.append(f"  low_confidence_fields: {', '.join(low_conf)}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# Fact check
# ══════════════════════════════════════════════════════════════════════


# Regex catalog: ordered from most-specific to most-generic so that a
# token matched as "$1.2B" isn't also matched as "1.2".
_TOKEN_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Specific currency suffixes first (longest match wins on overlap).
    ("currency_b", re.compile(r"\$\s*\d+(?:\.\d+)?\s*B")),
    ("currency_m", re.compile(r"\$\s*\d+(?:\.\d+)?\s*M")),
    ("currency_k", re.compile(r"\$\s*\d+(?:\.\d+)?\s*K")),
    ("currency_plain", re.compile(r"\$\s*[\d,]+(?:\.\d+)?")),
    # Counts with size suffix.
    ("count_k", re.compile(r"\b\d+(?:\.\d+)?\s*K\b")),
    ("count_m", re.compile(r"\b\d+(?:\.\d+)?\s*M\b")),
    ("count_b", re.compile(r"\b\d+(?:\.\d+)?\s*B\b")),
    # Percentages.
    ("percent", re.compile(r"\b\d+(?:\.\d+)?\s*%")),
    # Years (constrained 1900s-2099 to reduce noise from random 4-digit numbers).
    ("year", re.compile(r"\b(?:19|20)\d{2}\b")),
    # Comma-grouped integers (e.g. "7,400 employees", "$1,250,000").
    # Catches headcount and dollar amounts the LLM might paraphrase.
    ("comma_int", re.compile(r"\b\d{1,3}(?:,\d{3})+\b")),
]


def _normalise_token(tok: str) -> str:
    """Normalise a token for comparison: collapse whitespace, lowercase."""
    return re.sub(r"\s+", "", tok).lower()


def _fact_check(
    prose: str,
    dossier: CompanyDossier,
    slice_fields: tuple[str, ...],
) -> list[str]:
    """Extract numeric tokens from prose; flag those not in the slice.

    Returns a list of human-readable warnings.  Empty list when prose
    only quotes facts that appear in the slice.

    NOTE: this is a coarse check.  It catches the case where the LLM
    invented a number outright; it does NOT catch arithmetic errors
    (e.g. computing growth from two correctly-cited values).  Those
    require a dedicated numeric-claim verifier — out of scope for MVP.

    Overlap handling: ``finditer`` over multiple patterns can match
    overlapping spans (``$12.50`` inside ``$12.50B``).  We sort all
    matches by start offset and skip any whose span is contained in
    a previously-accepted match — the longer / more-specific token
    wins because the patterns are ordered specific→generic.
    """
    if not prose:
        return []
    # Build the corpus of "valid" tokens — every formatted value of
    # every relevant field, plus several derived forms so the regex
    # has acceptable recall against natural-language paraphrasing.
    valid_tokens: set[str] = set()
    for fname in slice_fields:
        if not hasattr(dossier, fname):
            continue
        dfield = getattr(dossier, fname)
        if not isinstance(dfield, DossierField):
            continue
        if not dfield.is_known:
            continue
        _add_valid_forms(valid_tokens, dfield.value)
        if dfield.as_of:
            valid_tokens.add(str(dfield.as_of.year))

    # Always-allow: the company's own ticker / identifying digits in
    # the ref show up legitimately in prose.
    if dossier.ref.ticker:
        valid_tokens.add(_normalise_token(dossier.ref.ticker))

    # Always allow the coverage report's percent / counts when they
    # are referenced in the risks section.
    cov = dossier.coverage_report or {}
    if "coverage_pct" in cov:
        valid_tokens.add(_normalise_token(f"{cov['coverage_pct']}%"))
        valid_tokens.add(_normalise_token(str(cov["coverage_pct"])))
    if "fields_filled" in cov:
        valid_tokens.add(_normalise_token(str(cov["fields_filled"])))
    if "fields_total" in cov:
        valid_tokens.add(_normalise_token(str(cov["fields_total"])))

    # Collect all matches across all patterns first; then prune
    # overlapping spans so $12.50 inside $12.50B doesn't get flagged.
    raw_matches: list[tuple[int, int, str, str]] = []  # (start, end, kind, tok)
    for kind, pattern in _TOKEN_PATTERNS:
        for match in pattern.finditer(prose):
            raw_matches.append((match.start(), match.end(), kind, match.group(0)))
    # Sort by (start asc, length desc) so the longest match at each
    # start position is considered first.
    raw_matches.sort(key=lambda t: (t[0], -(t[1] - t[0])))

    accepted_spans: list[tuple[int, int]] = []
    accepted: list[tuple[str, str]] = []  # (kind, token)
    for start, end, kind, tok in raw_matches:
        if any(s <= start and end <= e for s, e in accepted_spans):
            continue  # span already covered by a longer match
        accepted_spans.append((start, end))
        accepted.append((kind, tok))

    warnings: list[str] = []
    seen: set[str] = set()
    for kind, tok in accepted:
        norm = _normalise_token(tok)
        if norm in seen:
            continue
        seen.add(norm)
        if norm in valid_tokens:
            continue
        # Permissive percent match: "18.2%" should validate against
        # the bare number "18.2" appearing as a coverage figure or
        # similar.  Strip the suffix and re-check.
        if kind == "percent":
            stripped = norm.rstrip("%")
            if stripped in valid_tokens:
                continue
        warnings.append(
            f"unverified {kind!r} token in prose: {tok!r} "
            f"(no matching fact in slice)"
        )
    return warnings


def _add_valid_forms(valid_tokens: set[str], value) -> None:
    """Stamp every plausible textual rendering of ``value`` into
    ``valid_tokens``.

    Generated forms cover: the canonical ``_format_value`` output, raw
    digit strings, comma-grouped strings, and (for dates) the year.
    The recall here is intentionally generous — false-negative
    fact-checks are worse than false-positives because they let
    invented numbers slip through.
    """
    if value is None:
        return
    formatted = _format_value(value)
    valid_tokens.add(_normalise_token(formatted))
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        valid_tokens.add(_normalise_token(f"{value:.0f}"))
        valid_tokens.add(_normalise_token(f"{value:,.0f}"))
        # The "billion" / "million" forms in plain English.
        if abs(value) >= 1e9:
            valid_tokens.add(_normalise_token(f"${value/1e9:.2f}B"))
            valid_tokens.add(_normalise_token(f"${value/1e9:.1f}B"))
            valid_tokens.add(_normalise_token(f"{value/1e9:.2f}B"))
            valid_tokens.add(_normalise_token(f"{value/1e9:.1f}B"))
        if abs(value) >= 1e6:
            valid_tokens.add(_normalise_token(f"${value/1e6:.1f}M"))
            valid_tokens.add(_normalise_token(f"${value/1e6:.0f}M"))
            valid_tokens.add(_normalise_token(f"{value/1e6:.1f}M"))
        if abs(value) >= 1e3:
            valid_tokens.add(_normalise_token(f"{value/1e3:.1f}K"))
        return
    if hasattr(value, "year"):
        valid_tokens.add(str(value.year))
    if isinstance(value, (tuple, list)):
        for v in value:
            _add_valid_forms(valid_tokens, v)
