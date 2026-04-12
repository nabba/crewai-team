# Wiki Operations

## Ingest Workflow

1. Raw material is placed in `raw/{category}/`.
2. Agent reads raw material and synthesizes a wiki page draft.
3. Agent calls `WikiWriteTool` with action `create`, providing full frontmatter and body.
4. WikiWriteTool validates frontmatter, acquires lock, writes file, rebuilds section index, appends to log.
5. Agent verifies the page appears in the section index.

## Query Workflow

1. Agent calls `WikiSearchTool` with keywords and optional section filter.
2. Tool performs grep-based search across wiki pages, returns matching snippets with frontmatter metadata.
3. Agent calls `WikiReadTool` for full page content of relevant results.
4. Agent synthesizes answer from wiki content, citing page slugs.

## Update Workflow

1. Agent calls `WikiReadTool` to get current page content.
2. Agent calls `WikiWriteTool` with action `update`, providing the slug and new content.
3. Tool increments version, updates `updated_at`, rebuilds index, appends to log.

## Deprecation Workflow

1. Editor creates the replacement page first.
2. Editor calls `WikiWriteTool` with action `deprecate` on the old page, setting `deprecated_by`.
3. Tool sets `status: deprecated`, updates index, appends to log.

## Lint Workflow

1. Auditor (or any agent with lint permission) calls `WikiLintTool`.
2. Tool runs 8 health checks:
   - Frontmatter completeness (required fields present)
   - Orphan detection (pages not linked from any index)
   - Dead link detection (wikilinks pointing to non-existent pages)
   - Contradiction detection (conflicting claims on overlapping tags)
   - Staleness check (pages not updated in 90+ days)
   - Index consistency (section index page counts match actual files)
   - Bidirectional link check (A links B implies B links A)
   - Epistemic boundary check (high/verified confidence without proper source)
3. Tool returns a structured report of all issues found.
