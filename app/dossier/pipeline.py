"""pipeline — top-level orchestration of the dossier build.

Combines collection + peer selection + composition + typesetting into
one entry point.  This is what the crew layer wraps; it's also the
right surface for direct programmatic use (tests, scripts, custom
integrations).

Identity resolution
===================
The pipeline accepts either a structured :class:`CompanyRef` (when the
caller already knows ticker / domain / etc.) or a natural-language
``query`` string.  When a query is supplied, the pipeline runs a
lightweight identity parser:

  * Regex-extract ticker patterns (``SPOT``, ``$SPOT``, ``(SPOT)``)
  * Pull anything that looks like a company name (the rest)
  * Optionally enrich via an LLM call when ``llm_for_identity`` is
    provided (the crew layer wires this up)

The parser is deliberately simple — incorrect identity at this stage
fails closed (collector returns mostly UNRESOLVED fields, composer
flags the gaps prominently).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.dossier.collector import collect_dossier
from app.dossier.compose import ComposedReport, compose_report
from app.dossier.peers import select_peers
from app.dossier.schema import CompanyDossier, CompanyRef
from app.dossier.typeset import render_pdf

logger = logging.getLogger(__name__)


# ── Result envelope ──────────────────────────────────────────────────


@dataclass
class DossierBuild:
    """Everything the pipeline produces in one run."""

    ref: CompanyRef
    dossier: CompanyDossier
    peers: list[CompanyDossier]
    report: ComposedReport
    pdf_path: Path
    warnings_total: int

    def summary(self) -> str:
        """Compact human-readable summary suitable for Signal / chat reply."""
        cov = self.dossier.coverage_report or {}
        adapters_fired = ", ".join(cov.get("adapters_fired", []) or ["(none)"])
        warning_str = (
            f" | {self.warnings_total} fact-check flag(s)"
            if self.warnings_total else ""
        )
        return (
            f"Dossier for {self.ref.name} — "
            f"{cov.get('fields_filled', 0)}/{cov.get('fields_total', 0)} "
            f"fields populated ({cov.get('coverage_pct', 0.0)}%); "
            f"sources: {adapters_fired}; "
            f"{len(self.report.sections)} sections, "
            f"{len(self.peers)} peer(s){warning_str}.\n"
            f"PDF: {self.pdf_path}"
        )


# ── Identity parser ──────────────────────────────────────────────────


_TICKER_PATTERNS = (
    re.compile(r"\(([A-Z]{1,5})\)"),                    # "(SPOT)"
    re.compile(r"\bticker[:\s]+([A-Z]{1,5})\b", re.I),  # "ticker: SPOT"
    re.compile(r"\$([A-Z]{1,5})\b"),                    # "$SPOT"
    re.compile(r"\b([A-Z]{2,5})\s+stock\b"),            # "SPOT stock"
    re.compile(r"\bNYSE:([A-Z]{1,5})\b"),                # "NYSE: SPOT"
    re.compile(r"\bNASDAQ:([A-Z]{1,5})\b"),
)


def parse_identity(query: str) -> CompanyRef:
    """Extract a :class:`CompanyRef` from natural-language text.

    Conservative: any extracted ticker is uppercased, the rest of the
    string (with the ticker pattern stripped) becomes the company name.
    Empty input yields an empty ref — the caller decides whether to
    fail or to demand more information.
    """
    if not query:
        return CompanyRef(name="")

    cleaned = query.strip()
    ticker = ""
    for pat in _TICKER_PATTERNS:
        match = pat.search(cleaned)
        if match:
            ticker = match.group(1).upper()
            cleaned = pat.sub("", cleaned).strip()
            break

    # Strip filler words around the company name.
    cleaned = re.sub(
        r"\b(build|create|generate|produce|make|write|prepare|do|please|"
        r"investment|company|dossier|report|profile|analysis|review|"
        r"a|an|the|for|on|of|about)\b",
        "", cleaned, flags=re.I,
    ).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")

    return CompanyRef(name=cleaned, ticker=ticker)


# ── Public entry point ───────────────────────────────────────────────


def build_dossier(
    *,
    query: str | None = None,
    ref: CompanyRef | None = None,
    task_id: str | None = None,
    output_path: str | None = None,
    include_peers: bool = True,
    max_peers: int = 5,
    llm_call: Callable[[str], str] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> DossierBuild:
    """Run the full pipeline for one company and return the build envelope.

    Args:
        query: Free-text query identifying the company; ignored if
            ``ref`` is provided.
        ref: Pre-built CompanyRef; takes precedence over ``query``.
        task_id: Passed through to the collector for ledger emission.
        output_path: Override the default PDF filename / location.
        include_peers: When False, skip peer collection entirely
            (useful for fast smoke runs).
        max_peers: Cap on number of peers to collect.
        llm_call: Override the composer's LLM call (test injection).
        progress_callback: Optional ``str -> None`` callback fired at
            each major step ("collecting…", "selecting peers…", etc).
            Used by the crew layer to stream progress over Signal.

    Returns:
        :class:`DossierBuild` envelope with dossier, peers, composed
        sections, output PDF path, and aggregated fact-check warning
        count.
    """
    if ref is None:
        if not query:
            raise ValueError("build_dossier requires either query or ref")
        ref = parse_identity(query)
    if not ref.name and not ref.ticker:
        raise ValueError(
            "could not resolve company identity from query "
            f"{query!r}; pass an explicit CompanyRef instead",
        )

    _emit(progress_callback,
          f"Collecting dossier for {ref.name or ref.ticker}…")
    dossier = collect_dossier(ref, task_id=task_id)

    peers: list[CompanyDossier] = []
    if include_peers:
        _emit(progress_callback, "Selecting peers…")
        peer_refs = select_peers(dossier, max_peers=max_peers)
        for peer in peer_refs:
            try:
                peer_ref = CompanyRef(name=peer.name, ticker=peer.ticker)
                peers.append(collect_dossier(peer_ref))
            except Exception as exc:
                logger.warning("dossier.pipeline: peer %s collect failed: %s",
                               peer.name, exc)

    _emit(progress_callback,
          f"Composing report ({_len_known_sections(dossier)} sections)…")
    report = compose_report(dossier, peer_dossiers=peers, llm_call=llm_call)

    _emit(progress_callback, "Typesetting PDF…")
    pdf_path = render_pdf(
        report, dossier, peer_dossiers=peers, output_path=output_path,
    )

    warnings_total = sum(len(s.fact_check_warnings) for s in report.sections)
    return DossierBuild(
        ref=ref, dossier=dossier, peers=peers,
        report=report, pdf_path=pdf_path,
        warnings_total=warnings_total,
    )


def _emit(cb: Callable[[str], None] | None, msg: str) -> None:
    if cb is None:
        return
    try:
        cb(msg)
    except Exception:
        logger.debug("dossier.pipeline: progress callback failed", exc_info=True)


def _len_known_sections(_dossier: CompanyDossier) -> int:
    """Section count is currently fixed; this helper is here so the
    progress message stays accurate if SECTIONS grows later."""
    from app.dossier.sections import SECTIONS
    return len(SECTIONS) + 1  # +1 for the comparator
