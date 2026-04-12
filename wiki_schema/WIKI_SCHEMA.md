# Wiki Page Schema

Every wiki page is a Markdown file with YAML frontmatter.

## Required Frontmatter Fields

```yaml
---
title: "Human-readable page title"
section: meta | self | philosophy | plg | archibal | kaicart
created_at: "ISO-8601 timestamp"
updated_at: "ISO-8601 timestamp"
author: "agent-name"
status: draft | active | deprecated
confidence: low | medium | high | verified
tags: [list, of, tags]
related: [list, of, related-page-slugs]
source: "raw/path or URL or 'synthesis'"
supersedes: null | slug-of-deprecated-page
---
```

## Optional Fields

- `deprecated_by`: slug of the page that replaces this one (set when status becomes `deprecated`)
- `epistemic_note`: free-text caveat about confidence boundaries
- `version`: integer, incremented on each update

## File Naming

- Slug format: `kebab-case-title.md`
- Stored in: `wiki/{section}/{slug}.md`
- Index files: `wiki/{section}/index.md`

## Body Structure

1. H1 title (matches frontmatter title)
2. One-paragraph summary
3. Content sections (H2+)
4. `## Related Pages` section with wikilinks: `[[section/slug]]`
5. `## Sources` section citing raw material or external URLs

## Wikilinks

Internal references use `[[section/slug]]` syntax. All wikilinks must be bidirectional (if A links to B, B should link to A).
