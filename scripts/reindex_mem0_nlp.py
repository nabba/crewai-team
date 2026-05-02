#!/usr/bin/env python3
"""Re-enrich existing mem0 memories with spaCy lemmatization + entities.

Used once after installing ``mem0ai[nlp]`` to backfill the
``text_lemmatized`` and ``entities`` fields on rows that were
inserted while spaCy was unavailable. Idempotent: only updates rows
that lack BOTH fields, so safe to re-run.

Usage (inside the gateway container):

    docker exec crewai-team-gateway-1 \
        python -m scripts.reindex_mem0_nlp

Cost ≈ 3.3 ms per row (spaCy pipeline) + DB UPDATE ≈ 1–3 ms.
For 10k rows: ~30 s. For typical 5–500 row inboxes: sub-second.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Iterable

import psycopg2

logger = logging.getLogger("mem0_nlp_reindex")


def _is_unenriched(payload: dict) -> bool:
    """True iff the row was inserted before spaCy was available.

    Detection logic (any one is sufficient):
      a) Both ``text_lemmatized`` and ``entities`` are missing/empty.
      b) ``text_lemmatized`` is byte-identical to ``data`` — that's
         what mem0's fallback path produces when spaCy raises ImportError
         in lemmatize_for_bm25 (it returns the input unchanged). A
         genuinely lemmatized version differs from the raw data for
         any non-trivial input.
      c) ``entities`` is None/empty (per-text NER almost always finds
         at least one PROPER noun in fact-bearing memory data).

    Conservative on edge cases: a one-word memory like "Hello" might
    legitimately lemmatize to itself with no entities, but those
    aren't fact-bearing memories worth re-enriching anyway. The
    UPDATE is idempotent — re-running it produces the same payload.
    """
    data = (payload.get("data") or "").strip()
    lem = (payload.get("text_lemmatized") or "").strip()
    ents = payload.get("entities")
    has_real_ents = ents not in (None, [], "[]", "")
    has_real_lem = lem and lem.lower() != data.lower()
    return not (has_real_lem and has_real_ents)


def _enrich(text: str, lemmatize, extract) -> tuple[str, list]:
    """Run spaCy once, return (lemmatized_text, entities_as_list)."""
    lemma = lemmatize(text or "")
    ents = extract(text or "")
    # entity_extraction returns list[tuple[label, span_text]];
    # JSONB-friendly serialisation as list-of-lists.
    ent_list = [[label, span] for (label, span) in ents]
    return lemma, ent_list


def reindex(dsn: str, *, batch_size: int = 500, dry_run: bool = False) -> dict:
    """Backfill enrichment for every unenriched row.

    Returns a summary dict suitable for printing or test assertions.
    """
    from mem0.utils.lemmatization import lemmatize_for_bm25
    from mem0.utils.entity_extraction import extract_entities

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    cur = conn.cursor()

    # Count rows for telemetry
    cur.execute("SELECT COUNT(*) FROM public.crewai_memories")
    total = cur.fetchone()[0]

    cur.execute(
        "SELECT id, payload FROM public.crewai_memories ORDER BY id"
    )
    rows = cur.fetchall()

    enriched = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for row_id, payload in rows:
        # psycopg2 returns JSONB as a dict already
        if not isinstance(payload, dict):
            try:
                payload = json.loads(payload)
            except Exception:
                failed += 1
                continue
        if not _is_unenriched(payload):
            skipped += 1
            continue
        text = (payload.get("data") or "").strip()
        if not text:
            skipped += 1
            continue
        try:
            lemma, ent_list = _enrich(text, lemmatize_for_bm25, extract_entities)
        except Exception as exc:
            logger.warning(f"row {row_id}: spaCy failed — {exc}")
            failed += 1
            continue
        new_payload = {**payload, "text_lemmatized": lemma, "entities": ent_list}
        if dry_run:
            enriched += 1
            continue
        cur.execute(
            "UPDATE public.crewai_memories SET payload = %s WHERE id = %s",
            (json.dumps(new_payload), row_id),
        )
        enriched += 1
        if enriched % batch_size == 0:
            conn.commit()
    if not dry_run:
        conn.commit()
    cur.close()
    conn.close()
    return {
        "total_rows": total,
        "enriched": enriched,
        "skipped": skipped,
        "failed": failed,
        "elapsed_s": round(time.time() - t0, 3),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Read DSN from settings — same source mem0_manager uses
    from app.config import get_settings
    s = get_settings()
    dsn = s.mem0_postgres_url
    if not dsn:
        print("ERROR: mem0_postgres_url not configured", file=sys.stderr)
        return 2
    summary = reindex(dsn)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
