# Operator CLI (`python -m app.cli`)

Narrow operational + recovery + scripting surface for Andrus AI. Lives at `app/cli/`.

**Three jobs:**

1. **Substrate-level escape hatch.** When Signal is broken or the dashboard is
   sick, `python -m app.cli status --endpoint tailnet` still works.
2. **Consolidation** of scattered `python -m app.X` modules under one umbrella
   (`aai brainstorm`, `aai drill run X`, …).
3. **Scriptability** — pipe-friendly output that composes with `jq` / `grep` / `less`.

**Explicitly NOT** a chat surface. Signal + Discord + `/cp/chat` already cover
that. Adding a fourth chat surface would add audit-log discipline, sender_id
namespacing, conversation-history wiring, and Goodhart-guard plumbing without
closing a real gap.

---

## Install / alias

The CLI is invoked as a module. No build-system changes are needed
(`pyproject.toml` has no `[project]` section yet, so adding `[project.scripts]`
would be invasive). Operator-recommended shell alias:

```bash
# ~/.zshrc
alias aai='python -m app.cli'
```

Then everywhere below, `aai foo` ≡ `python -m app.cli foo`.

## Configuration

Resolution order (highest priority first):

1. CLI flag (`--endpoint`, `--bearer`)
2. Environment (`AAI_ENDPOINT`, `AAI_BEARER`, or `GATEWAY_SECRET`)
3. `~/.config/andrusai/config.toml`
4. Built-in defaults (endpoint=`http://localhost:3100`)

Example `~/.config/andrusai/config.toml`:

```toml
[default]
endpoint = "tailnet"

[endpoints]
tailnet = "http://andrus-macbook-pro-16.tail5b289b.ts.net:3100"
funnel  = "https://andrus-macbook-pro-16.tail5b289b.ts.net"

[auth]
bearer = "..."
```

Named aliases: `local` / `tailnet` / `funnel`. Anything containing `://` is
treated as a verbatim URL.

## Global flags

All four work *before* OR *after* the subverb (`aai --json status`
and `aai status --json` both parse):

| Flag | Effect |
|---|---|
| `--endpoint URL\|alias` | Override endpoint resolution |
| `--bearer TOKEN` | Override bearer token |
| `--json` | Emit structured JSON (jq-friendly) |
| `--quiet` | Suppress informational output; errors still go to stderr |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | User error (bad args, file not found, etc.) |
| 2 | Transport / auth error (network down, 401/403) |
| 3 | Gateway returned non-2xx (5xx, 4xx ≠ auth) |
| 130 | Ctrl-C |

## Subcommand inventory

### Recovery / diagnostic

```bash
aai status                        # mirror of /cp/monitor
aai status --endpoint tailnet     # when Signal+dashboard sick, reach via tailnet
aai healing run <monitor>         # force-probe e.g. disk_quota, tz_drift
aai logs tail -n 100              # tail workspace/errors.jsonl
aai logs tail --path /some/log -n 50
```

### Recall / inspection

```bash
aai recall "Helsinki ferry"             # conversation memory search
aai recall "..." --top-k 20 --json | jq .
aai briefing morning                    # compose + print daily briefing
aai briefing weekly --json              # for piping into something else
aai ledger tail -n 20                   # identity continuity ledger
aai ledger tail --kind resilience_drill --kind chromadb_corruption
aai threads list --status open
aai threads show <8-char-prefix>
```

### Files / notes / skills

```bash
aai files list                          # all root buckets (output/skills/notes)
aai files list --root notes
aai files send <path> --via signal --body "fyi"
aai files send <path> --via email
aai files send <path> --via discord

cat draft.md | aai notes save "thinking"    # writes workspace/notes/thinking.md
aai notes save "title" --body "..." --overwrite

aai skills list
aai skills show <name>
```

### Governance read-side

```bash
aai cr list                              # all change requests
aai cr list --state pending
aai cr show <request-id>                 # CR detail with diff
aai amendments list                      # Tier-3 amendments
```

> Write-side governance (approve/reject) is **deliberately not** in the CLI.
> Approvals live on devices with typed-phrase / Signal-reaction confirmation
> flows. The CLI reads governance state, never acts on it.

### Wrappers (existing `python -m` entries)

```bash
aai brainstorm                           # ≡ python -m app.brainstorm
aai brainstorm --with-agents 4 --resume
aai drill list                           # ≡ python -m app.resilience_drills list
aai drill run backup_restore --dry-run
aai bootstrap google                     # ≡ python -m app.google_workspace.bootstrap
aai bootstrap web-push
aai bootstrap browse
aai bootstrap warm-spare
aai advisory goodhart --window-days 30
```

For wrappers, global flags must come **before** the verb
(`aai --json brainstorm --list` works; `aai brainstorm --json --list` is
forwarded to the underlying module verbatim, which may not understand `--json`).

## Scripting recipes

```bash
# What identity drift happened this month?
aai ledger tail --kind identity_drift_acceleration --json -n 50 | jq '.[] | .summary'

# Pipe the morning briefing into Mail.
aai briefing morning | mail -s "AI briefing" $USER

# Bulk-check pending CRs.
aai cr list --state pending --json | jq -r '.[] | "\(.id) \(.path)"'

# Quick recall during a terminal session.
aai recall "person X" --top-k 5 --quiet
```

## Design choices worth knowing

1. **Stdlib only** (urllib, argparse, tomllib). No `httpx` / `click` / `typer`
   dependency. Reason: the CLI must work when the gateway venv is sick;
   pulling deps would couple the recovery surface to the thing it might be
   recovering.

2. **`argparse.SUPPRESS` on global flags** — load-bearing. argparse's
   `parents=` mechanism overwrites parent values with leaf defaults unless
   the parent uses `default=argparse.SUPPRESS`. The `test_json_flag_works_in_every_position`
   test in `tests/test_cli_dispatcher.py` permanently pins this.

3. **Lazy imports inside subcommands** so a broken `app.life_companion` can't
   break `aai status`.

4. **Write-side governance excluded** — see note above.

5. **No new HTTP endpoints** — v1 wraps what exists.

## Testing

```bash
python -m pytest tests/test_cli_dispatcher.py -v
```

40 tests covering global flag positioning, config resolution, transport error
mapping, local-only commands, help-text rendering, and TOML loading.

## When to add a subcommand

A new subverb is justified when it meets ONE of:

- Wraps an existing `python -m` entry point under a memorable name.
- Provides a read-only view over an existing HTTP endpoint that is
  meaningfully more useful from a terminal than from a browser.
- Closes a substrate-level recovery gap (something an operator would need
  when the dashboard is down).

Not justified:

- "It would be nicer to type." → use a shell alias instead.
- Anything write-side that affects shared state → keep on the dashboard.
- Multi-turn conversational flows → Signal / voice handle that better.
- Requires a new HTTP endpoint that doesn't otherwise exist → ship the
  endpoint first, then add the CLI subcommand.
