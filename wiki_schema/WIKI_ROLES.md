# Wiki Agent Permissions

## Role Matrix

| Role | Read | Write | Deprecate | Lint | Ingest |
|------|------|-------|-----------|------|--------|
| Researcher | yes | yes (own section + meta) | no | no | yes |
| Philosopher | yes | yes (philosophy + meta) | no | no | yes |
| Editor | yes | yes (all sections) | yes | yes | yes |
| Auditor | yes | no | no | yes | no |
| Any Agent | yes | no | no | no | no |

## Section Ownership

- **meta**: shared cross-venture knowledge; any writer can contribute
- **self**: agent self-knowledge; Editor-only or designated self-reflection agent
- **philosophy**: philosophical frameworks; Philosopher + Editor
- **plg / archibal / kaicart**: venture-specific; assigned venture agent + Editor

## Write Rules

1. An agent may only write to sections listed in its role.
2. All writes acquire a file lock (wiki/.locks/{slug}.lock) and release on completion.
3. Deprecation requires Editor role. Deprecated pages must set `deprecated_by` pointing to the replacement.
4. Index files are rebuilt automatically by WikiWriteTool after every write.

## Conflict Resolution

If two agents attempt to write the same page simultaneously, the second writer receives a lock-contention error and must retry after a brief delay.
