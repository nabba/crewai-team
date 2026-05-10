# Archived change requests — wiki_index_reconciler validation failures

**Archived**: 2026-05-10T13:22:54.035407+00:00
**Count**: 355 change-request JSON files
**Pattern**: requestor=`wiki_index_reconciler`, path=`wiki/index.md`, status=`rejected`,
  decision_reason contains "outside the repo's allowed roots"

## Why these existed

The wiki-index reconciler files a CR whenever it detects drift between the
live `wiki/index.md` and the canonical content computed from on-disk pages.
A latent bug in `_compute_master_index_content` writes today's date into
the body line `Total pages: N | Last updated: YYYY-MM-DD`, while the
reconciler's hash-comparison normalizer only stripped the frontmatter
`updated_at:` field — so canonical (computed with the pin date 2000-01-01)
hashed differently from live (today's date) on every single run, producing
false-positive drift detections.

The validator then rejected each CR because `wiki/` was not in the
`_ALLOWED_ROOT_PREFIXES` list. Both bugs compounded: drift detector
produced fresh CRs on every run; validator rejected each one immediately;
the operator's CR queue piled up with 355 unactionable rejections.

## Why archive (not delete)

Audit-trail discipline. The hash-chained audit log in `audit.jsonl`
references these records; deleting them would create dangling references.
Archiving keeps them addressable forever while clearing them from the
active queue surface.

## What prevents recurrence

Three independent layers shipped on 2026-05-10:

1. **Validator** (`app/change_requests/validator.py`): added `wiki/` to
   `_ALLOWED_ROOT_PREFIXES`. TIER_IMMUTABLE check still applies.

2. **Drift detector** (`app/memory/wiki_index_reconciler.py`):
   `_normalize_for_hashing` now strips both the frontmatter `updated_at:`
   line AND the body `Total pages: N | Last updated: YYYY-MM-DD` line.
   Tests pin the regression: `test_no_drift_ignores_body_last_updated_date`.

3. **Defensive dedup** (`app/memory/wiki_index_reconciler.py`):
   `_existing_cr_blocks_filing` skips new filings when a non-terminal CR
   already exists for `wiki/index.md`, OR when a recent (≤7d) rejected
   CR has identical content. Honours operator decisions; prevents
   re-spam if anything else regresses.

## Forensic value

A single representative archived record is enough to understand what
happened. The remaining 354 are byte-identical
modulo timestamps and hashes; kept for completeness, not analysis.
