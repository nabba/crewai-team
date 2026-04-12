# raw/ — Unprocessed Source Material

This directory holds raw source documents before they are ingested into the wiki.

## Conventions

- Subdirectories are organized by topic: `creative/`, `firecrawl/`, `philosophical/`, `research/`, `transcripts/`, `ventures/`.
- Files here are **read-only inputs** for the wiki ingest pipeline.
- Once a raw document has been ingested into a wiki page, the wiki page's `source` frontmatter field should reference the original path here.
- Do not edit raw files after ingest — if corrections are needed, update the wiki page and note the discrepancy.
- Large binary files (PDFs, images) may live here temporarily but should not be committed to git long-term.
