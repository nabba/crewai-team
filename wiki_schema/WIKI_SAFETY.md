# Wiki Safety Constraints (DGM)

## Core Principles

1. **No silent overwrites.** Every update increments `version` and appends to `wiki/log.md`.
2. **No deletions.** Pages are deprecated, never deleted. Deprecated pages remain readable.
3. **Path traversal protection.** All file operations are sandboxed to the `wiki/` directory tree. Any path containing `..` or absolute paths outside the wiki root are rejected.
4. **Epistemic honesty.** Pages must declare `confidence` level. Claims beyond the agent's epistemic boundary must be flagged with `epistemic_note`.
5. **Contradiction detection.** WikiLintTool checks for pages in the same section with conflicting claims on overlapping tags.

## Forbidden Operations

- Writing to paths outside `wiki/`
- Deleting any `.md` file
- Modifying `wiki/log.md` except to append new entries
- Setting `confidence: verified` without a `source` that is a concrete reference (not `synthesis`)
- Bypassing file locks

## Audit Trail

Every write operation appends a timestamped entry to `wiki/log.md`:
```
| {timestamp} | {agent} | {action} | {section}/{slug} | {summary} |
```

## Staleness Policy

Pages not updated in 90+ days are flagged as potentially stale by WikiLintTool. Stale pages should be reviewed and either refreshed or deprecated.
