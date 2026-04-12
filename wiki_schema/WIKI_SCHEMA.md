# Wiki Page Schema

Every wiki page is a Markdown file with YAML frontmatter.

## Required Frontmatter Fields

```yaml
---
title: "Human-readable page title"
section: meta | self | philosophy | plg | archibal | kaicart
created_at: "ISO-8601 timestamp"
updated_at: "ISO-8601 timestamp"
date: "YYYY-MM-DD"  # Dataview-compatible ISO date
author: "agent-name"
status: draft | active | deprecated
confidence: low | medium | high | verified
tags: [list, of, tags]
aliases: ["alternative name"]  # Dataview-compatible alternative names
related: [list, of, related-page-slugs]
relationships:  # Typed links (optional, list of {type, target})
  - type: supports | contradicts | supersedes | prerequisite | tested_by | refines | extends
    target: "section/slug"
source: "raw/path or URL or 'synthesis'"
supersedes: null | slug-of-deprecated-page
---
```

## Optional Fields

- `deprecated_by`: slug of the page that replaces this one
- `epistemic_note`: free-text caveat about confidence boundaries
- `version`: integer, auto-incremented on each update
- `federation`: cross-wiki reference metadata `{source_wiki, source_page, sync_status}`

## Relationship Types

| Type | Meaning |
|------|---------|
| `supports` | This page provides evidence supporting the target |
| `contradicts` | This page's claims conflict with the target |
| `supersedes` | This page replaces the target (target should be deprecated) |
| `prerequisite` | The target should be read before this page |
| `tested_by` | The target page contains tests/validation for this page's claims |
| `refines` | This page provides more detail on a subtopic of the target |
| `extends` | This page adds new dimensions to the target's analysis |

## File Naming

- Slug format: `kebab-case-title.md`
- Stored in: `wiki/{section}/{slug}.md`
- Index files: `wiki/{section}/index.md`

## Body Structure

1. H1 title (matches frontmatter title)
2. One-paragraph summary
3. Content sections (H2+)
4. `## Contradictions and Open Questions` (required, may be empty)
5. `## Related Pages` section with wikilinks: `[[section/slug]]`
6. `## Sources` section citing raw material or external URLs
7. `## Change History` (maintained by agents)

## Wikilinks

Internal references use `[[section/slug]]` syntax. All wikilinks must be bidirectional (if A links to B, B should link to A).

## Obsidian Compatibility

- YAML frontmatter is Obsidian Dataview compatible
- `aliases` field enables alternative name search in Obsidian
- `date` field (ISO date) works with Dataview date queries
- Open `wiki/` as Obsidian vault for graph visualization
- Graph color coding: meta=blue, self=purple, philosophy=yellow, plg=green, archibal=blue, kaicart=red
