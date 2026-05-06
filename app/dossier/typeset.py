"""typeset — render a :class:`ComposedReport` to a multi-page PDF.

Implementation
==============
Uses ReportLab Platypus, which gives us flowables (Paragraph, Table,
PageBreak, Spacer) that compose into multi-page documents with
automatic pagination, header/footer, and embedded charts.

We deliberately do NOT route through the ad-hoc :mod:`pdf_compose`
script-runner.  pdf_compose is for one-off Python snippets that
produce simple PDFs (analyst chat, single-figure summaries); this
typesetter is a structured renderer that always emits the same
section order, the same TOC, the same source-appendix template.

We DO reuse pdf_compose's cached heavy imports (``_RL_PACK``,
``_MPL_PACK``) and ``_safe_output_path`` — there's no value in
re-importing matplotlib + reportlab at module load.

Output
======
A single PDF in ``/app/workspace/output/`` with the filename
``dossier_<slug>_<YYYYmmdd>.pdf``.  Returns the absolute path so the
caller can pass it to ``signal_send_attachment``.

What's rendered
===============
Page 1: Cover page (company name, date, system identifier).
Page 2: Table of contents.
Pages 3-N: Section prose, each starting on a new page.  Sections that
           had fact-check warnings get a "Data-quality flags" sidebar.
Pages N+1 onward: Comparator section (when peers were available).
Last 2-3 pages: Source appendix — every populated dossier field, its
                value, source URL, confidence, and as_of date.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.dossier.compose import ComposedReport, SectionOutput
from app.dossier.schema import CompanyDossier, DossierField

logger = logging.getLogger(__name__)


# ── Heavy import reuse ───────────────────────────────────────────────
#
# We pull the cached reportlab bundle from pdf_compose so we don't
# pay the import cost twice and so any future changes to the
# warn-shim machinery (the comment block in pdf_compose explains the
# pydantic monkey-patch issue) flow through automatically.

try:
    from app.tools.pdf_compose import _RL_PACK
except Exception:
    _RL_PACK = {}


def _resolve_output_path(user_path: str | None) -> Path:
    """Resolve an output path for the typesetter.

    Resolution order:
      1. ``user_path`` is absolute → write there directly.  The dossier
         crew passes controlled paths; this is not an agent-callable
         interface so path traversal isn't a concern at this layer.
      2. ``DOSSIER_OUTPUT_DIR`` env var → write inside it (useful for
         tests + dev shells where ``/app/workspace`` doesn't exist).
      3. ``user_path`` is a basename → delegate to
         ``pdf_compose._safe_output_path`` which clamps to the
         production sandbox ``/app/workspace/output/``.

    The production path is preserved as the default — operators don't
    need to set the env var unless they want a different location.
    """
    import os
    if user_path:
        p = Path(user_path).expanduser()
        if p.is_absolute():
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

    env_dir = os.environ.get("DOSSIER_OUTPUT_DIR", "").strip()
    if env_dir:
        out_dir = Path(env_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        name = Path(user_path or "report.pdf").name or "report.pdf"
        return out_dir / name

    # Production default: pdf_compose's safe-output sandbox.
    try:
        from app.tools.pdf_compose import _safe_output_path
        return _safe_output_path(user_path or "report.pdf")
    except Exception:
        # Last-resort: /tmp.  Reached only when pdf_compose can't be
        # imported (broken install) — keeps the typesetter operational.
        out_dir = Path("/tmp/dossier")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / Path(user_path or "report.pdf").name


# ── Style & rendering ────────────────────────────────────────────────


def _slug(name: str) -> str:
    """Slugify a company name for the output filename."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "report"


def _build_styles():
    """Build the paragraph styles used throughout the report.

    Centralised so the full document renders with consistent typography.
    """
    if not _RL_PACK:
        return None
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

    styles = getSampleStyleSheet()
    out = {
        "title": ParagraphStyle(
            "DossierTitle", parent=styles["Title"],
            fontSize=24, leading=28, spaceAfter=18, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "DossierSubtitle", parent=styles["Normal"],
            fontSize=14, leading=18, spaceAfter=12,
            alignment=TA_CENTER, textColor=colors.grey,
        ),
        "h1": ParagraphStyle(
            "DossierH1", parent=styles["Heading1"],
            fontSize=18, leading=22, spaceBefore=14, spaceAfter=8,
            textColor=colors.HexColor("#0B3D91"),
        ),
        "h2": ParagraphStyle(
            "DossierH2", parent=styles["Heading2"],
            fontSize=13, leading=16, spaceBefore=10, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "DossierBody", parent=styles["Normal"],
            fontSize=10.5, leading=14.5, spaceAfter=10,
            alignment=TA_JUSTIFY,
        ),
        "small": ParagraphStyle(
            "DossierSmall", parent=styles["Normal"],
            fontSize=8.5, leading=11, textColor=colors.grey,
        ),
        "warn": ParagraphStyle(
            "DossierWarn", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=colors.HexColor("#A0410B"),
            leftIndent=10, spaceAfter=8,
        ),
        "toc_entry": ParagraphStyle(
            "DossierTOC", parent=styles["Normal"],
            fontSize=11, leading=18,
        ),
    }
    return out


def _markdown_to_paragraph_html(md: str) -> str:
    """Convert dossier-flavour markdown to ReportLab paragraph HTML.

    The composer emits prose with ``[bracketed_field]`` citations, occasional
    ``**bold**`` / ``*italic*`` markers, and bullet lists.  ReportLab's
    Paragraph wraps lightweight HTML — we map the markdown to that
    subset (``<b>``, ``<i>``, ``<br/>``).
    """
    text = md or ""
    # Escape angle brackets for ReportLab Paragraph (which does HTML-ish parsing).
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold / italic.
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    # Render [field_name] citations as small grey superscript so the
    # source-trace pattern is preserved without disrupting reading.
    text = re.sub(
        r"\[(\w+)\]",
        r'<font size="7" color="#888888"><super>\1</super></font>',
        text,
    )
    # Convert blank-line separated paragraphs to <br/><br/> so a single
    # Paragraph flowable can render the whole block.
    text = re.sub(r"\n{2,}", "<br/><br/>", text)
    text = text.replace("\n", " ")
    return text.strip()


# ── Page header / footer ─────────────────────────────────────────────


def _page_furniture(
    canvas, doc, *, company_name: str, generated_at: datetime,
):
    """Header/footer drawn on every page after the cover."""
    if not _RL_PACK:
        return
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    page_num = canvas.getPageNumber()
    if page_num == 1:
        return  # cover page is bare
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        0.75 * inch, 10.7 * inch,
        f"{company_name} — Investment Dossier",
    )
    canvas.drawRightString(
        7.75 * inch, 10.7 * inch, generated_at.strftime("%Y-%m-%d"),
    )
    canvas.line(0.75 * inch, 10.65 * inch, 7.75 * inch, 10.65 * inch)
    canvas.drawCentredString(
        4.25 * inch, 0.5 * inch, f"— page {page_num} —",
    )
    canvas.restoreState()


# ── The render function ──────────────────────────────────────────────


def render_pdf(
    report: ComposedReport,
    dossier: CompanyDossier,
    peer_dossiers: list[CompanyDossier] | None = None,
    *,
    output_path: str | None = None,
) -> Path:
    """Render the report to a PDF and return its path.

    Args:
        report: The composed sections (with fact-check warnings).
        dossier: The full dossier — used for the source appendix.
        peer_dossiers: Peer dossiers — used in the comparator table.
        output_path: Override the default output filename.

    Raises:
        RuntimeError: If reportlab isn't installed (the typesetter
            cannot degrade gracefully — we don't have an HTML fallback
            in MVP).  Production environments are expected to have
            reportlab installed (it lives in the docker image
            alongside matplotlib).
    """
    if not _RL_PACK:
        raise RuntimeError(
            "reportlab not available — cannot render PDF. "
            "Install reportlab in the runtime image."
        )

    from reportlab.lib.units import inch
    from reportlab.platypus import (
        BaseDocTemplate, PageTemplate, Frame,
        Paragraph, Spacer, Table, TableStyle, PageBreak,
    )
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER

    styles = _build_styles()

    if output_path:
        path = _resolve_output_path(output_path)
    else:
        slug = _slug(report.company_name)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = _resolve_output_path(f"dossier_{slug}_{date_str}.pdf")

    page_size = LETTER
    margin = 0.75 * inch

    # Set up the document with a header/footer template.
    doc = BaseDocTemplate(
        str(path),
        pagesize=page_size,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin + 0.25 * inch, bottomMargin=margin,
        title=f"{report.company_name} — Investment Dossier",
        author="BotArmy Dossier Subsystem",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height, id="main",
    )
    generated_at = datetime.now(timezone.utc)
    doc.addPageTemplates([PageTemplate(
        id="default", frames=[frame],
        onPage=lambda c, d: _page_furniture(
            c, d, company_name=report.company_name, generated_at=generated_at,
        ),
    )])

    flowables: list = []

    # ── 1. Cover page ────────────────────────────────────────────────
    flowables.extend(_render_cover(report, generated_at, styles))
    flowables.append(PageBreak())

    # ── 2. Table of contents ─────────────────────────────────────────
    flowables.extend(_render_toc(report, styles))
    flowables.append(PageBreak())

    # ── 3. Sections ──────────────────────────────────────────────────
    for section in report.sections:
        flowables.extend(_render_section(section, styles))
        flowables.append(PageBreak())

    # ── 4. Source appendix ───────────────────────────────────────────
    flowables.extend(_render_source_appendix(dossier, styles))

    # ── 5. Coverage report appendix ──────────────────────────────────
    flowables.append(PageBreak())
    flowables.extend(_render_coverage_appendix(dossier, styles))

    doc.build(flowables)
    logger.info("dossier.typeset: wrote PDF to %s", path)
    return path


# ── Cover ────────────────────────────────────────────────────────────


def _render_cover(report: ComposedReport, generated_at: datetime, styles) -> list:
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.units import inch
    out: list = []
    out.append(Spacer(1, 2.0 * inch))
    out.append(Paragraph("Investment Dossier", styles["title"]))
    out.append(Paragraph(report.company_name, styles["title"]))
    out.append(Spacer(1, 0.3 * inch))
    out.append(Paragraph(
        f"Generated {generated_at.strftime('%B %d, %Y')}",
        styles["subtitle"],
    ))
    out.append(Spacer(1, 1.5 * inch))
    out.append(Paragraph(
        "This document is auto-generated from public-source data with "
        "explicit provenance. Every numeric or date claim in the prose "
        "is traceable to a source in the appendix. Coverage and "
        "data-quality limitations are summarised in the final section.",
        styles["body"],
    ))
    out.append(Spacer(1, 0.3 * inch))
    cov = report.coverage_report or {}
    out.append(Paragraph(
        f"Coverage: {cov.get('coverage_pct', 0.0)}% "
        f"({cov.get('fields_filled', 0)} of {cov.get('fields_total', 0)} "
        f"fields populated). "
        f"Sources fired: {', '.join(cov.get('adapters_fired', []) or ['(none)'])}.",
        styles["small"],
    ))
    return out


# ── TOC ──────────────────────────────────────────────────────────────


def _render_toc(report: ComposedReport, styles) -> list:
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.units import inch
    out: list = []
    out.append(Paragraph("Contents", styles["h1"]))
    out.append(Spacer(1, 0.2 * inch))
    for i, section in enumerate(report.sections, start=1):
        warn = ""
        if section.fact_check_warnings:
            warn = (
                f' <font size="7" color="#A0410B">'
                f'⚠ {len(section.fact_check_warnings)} flag(s)</font>'
            )
        out.append(Paragraph(
            f"{i}. {section.title}{warn}", styles["toc_entry"],
        ))
    out.append(Paragraph(
        f"{len(report.sections) + 1}. Source Appendix",
        styles["toc_entry"],
    ))
    out.append(Paragraph(
        f"{len(report.sections) + 2}. Coverage and Limitations",
        styles["toc_entry"],
    ))
    return out


# ── Sections ─────────────────────────────────────────────────────────


def _render_section(section: SectionOutput, styles) -> list:
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.units import inch
    out: list = []
    out.append(Paragraph(section.title, styles["h1"]))
    out.append(Spacer(1, 0.1 * inch))
    out.append(Paragraph(
        _markdown_to_paragraph_html(section.prose),
        styles["body"],
    ))
    if section.fact_check_warnings:
        out.append(Spacer(1, 0.15 * inch))
        out.append(Paragraph("Data-quality flags", styles["h2"]))
        for w in section.fact_check_warnings:
            out.append(Paragraph(
                f"⚠ {_markdown_to_paragraph_html(w)}",
                styles["warn"],
            ))
    return out


# ── Source appendix ──────────────────────────────────────────────────


def _render_source_appendix(dossier: CompanyDossier, styles) -> list:
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    out: list = []
    out.append(Paragraph("Source Appendix", styles["h1"]))
    out.append(Paragraph(
        "Every populated field with its value, source, confidence, and "
        "as-of date.  Use this to verify the report's assertions "
        "against primary sources.",
        styles["small"],
    ))
    out.append(Spacer(1, 0.15 * inch))

    rows: list[list[str]] = [
        ["Field", "Value", "Source", "Confidence", "As of"],
    ]
    for fname, dfield in dossier.iter_fields():
        if not dfield.is_known:
            continue
        rows.append([
            fname,
            _truncate(dfield.render_value(), 38),
            _truncate(
                f"{dfield.source.adapter} — {dfield.source.url or '(no URL)'}",
                52,
            ) if dfield.source else "(unknown)",
            dfield.confidence.value,
            dfield.as_of.isoformat() if dfield.as_of else "—",
        ])

    if len(rows) == 1:
        out.append(Paragraph("(no fields populated)", styles["body"]))
        return out

    col_widths = [1.3 * inch, 1.3 * inch, 2.6 * inch, 0.9 * inch, 0.8 * inch]
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    out.append(table)
    return out


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# ── Coverage / limitations appendix ──────────────────────────────────


def _render_coverage_appendix(dossier: CompanyDossier, styles) -> list:
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.units import inch
    cov = dossier.coverage_report or {}
    out: list = []
    out.append(Paragraph("Coverage and Limitations", styles["h1"]))
    out.append(Paragraph(
        f"<b>Coverage:</b> {cov.get('coverage_pct', 0.0)}% of fields "
        f"populated ({cov.get('fields_filled', 0)} of "
        f"{cov.get('fields_total', 0)})."
        f"<br/><b>Generated in:</b> {cov.get('elapsed_seconds', 0.0)}s.",
        styles["body"],
    ))
    out.append(Spacer(1, 0.1 * inch))
    if cov.get("adapters_fired"):
        out.append(Paragraph(
            "<b>Adapters fired:</b> " + ", ".join(cov["adapters_fired"]),
            styles["body"],
        ))
    if cov.get("adapters_skipped"):
        out.append(Paragraph(
            "<b>Adapters skipped (not configured or insufficient ref):</b>",
            styles["body"],
        ))
        for a, reason in cov["adapters_skipped"].items():
            out.append(Paragraph(f"&nbsp;&nbsp;• {a} — {reason}",
                                 styles["small"]))
    if cov.get("adapters_errored"):
        out.append(Paragraph(
            "<b>Adapters errored:</b>", styles["body"],
        ))
        for a, reason in cov["adapters_errored"].items():
            out.append(Paragraph(f"&nbsp;&nbsp;• {a} — {reason}",
                                 styles["small"]))
    if cov.get("adapters_tripped"):
        out.append(Paragraph(
            "<b>Circuit-broken adapters (consecutive failures):</b>",
            styles["body"],
        ))
        for a, reason in cov["adapters_tripped"].items():
            out.append(Paragraph(f"&nbsp;&nbsp;• {a} — {reason}",
                                 styles["small"]))
    return out
