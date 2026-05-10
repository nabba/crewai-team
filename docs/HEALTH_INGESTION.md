# Health ingestion (§5.1) — operator guide

Personal-health data ingestion for the AndrusAI companion. Default-OFF;
no health data leaves the local system; the morning/evening/weekly
briefings include a one-section rollup once you opt in.

## What this is, briefly

  - `app/health/import_apple.py` parses Apple's `apple_health_export.zip`
    (or extracted `export.xml`) into typed records.
  - `app/health/store.py` writes per-kind JSONL at
    `workspace/health/<kind>.jsonl`.
  - `app/health/summary.py:summarise_window(days=7)` rolls up the data.
  - `app/health/anomaly.py:detect_anomalies()` flags outliers (recent
    3 days vs baseline 30 days, |z| ≥ 2).
  - `app/health/idle_job.py` runs once per ~24 h and surfaces results
    via the daily briefing.
  - `app/inbox/` watches `workspace/inbox/` and routes
    `apple_health_export.zip` to the importer automatically.

## How access to Apple Health is gained

There is **no programmatic way** for a process on a Mac/Linux host to
read iPhone HealthKit data. Apple's HealthKit API is iOS-only and
requires an iOS app with explicit user consent. The realistic path is
**user-mediated export**:

1. **iPhone Health app** → tap your profile picture (top-right) →
   scroll to the bottom → **"Export All Health Data"**.
2. iOS spends 5–30 minutes packaging the data (size scales with years
   and watch wear-time — typically 50–500 MB).
3. iOS opens a Share sheet. Pick a transport:
   - **AirDrop** to the Mac (fastest)
   - **Save to Files** → iCloud Drive → drag from Finder on the host
   - **Email to self** → save the attachment
4. **Drop the resulting `apple_health_export.zip`** into
   `workspace/inbox/` on the host.

That's it. Within seconds the inbox watcher picks the file up;
within ~24 h the daily briefing shows a 7-day rollup.

## Enabling it

Two env vars, both default `false`:

```bash
export INBOX_INGESTION_ENABLED=true     # turns on workspace/inbox/ watcher
export HEALTH_INGESTION_ENABLED=true    # turns on the importer + summary
```

Optional knobs:

```bash
# Override the JSONL store location (default: $WORKSPACE_ROOT/health)
export HEALTH_BASE_DIR=/path/to/health/store

# Override the inbox watch directory (default: $WORKSPACE_ROOT/inbox)
export INBOX_DIR=/path/to/inbox

# Where the inbox text handler drops .md/.txt (default: $WORKSPACE_ROOT/notes)
export INBOX_NOTES_DIR=/path/to/notes
```

Both subsystems register themselves as LIGHT idle jobs in
`app/companion/loop.py:get_idle_jobs()`. No service restart required —
the next time the idle scheduler ticks, they pick up the env change.

## Privacy guarantees

  - **Health data NEVER leaves the host.** No ChromaDB embedding (which
    would route through external models), no external API calls, no
    LLM inference over raw records. Only summary statistics
    (mean / total / z-score) reach the daily-briefing composer.
  - **Default-OFF.** Both env vars are `false` until you explicitly
    set them. The system never imports without consent.
  - **Append-only JSONL.** No deletion path; if you want to wipe the
    data, `rm -rf $WORKSPACE_ROOT/health/`.
  - **Idempotent re-import.** Dedupe key `(start_iso, source_version)`
    means re-importing the same export adds zero records.

## Data retention

The store grows unboundedly. For the typical user (one watch + one
phone, several years of data) expect:

  - heart_rate.jsonl: ~10 MB / year
  - steps.jsonl: ~5 MB / year
  - active_energy.jsonl: ~5 MB / year
  - sleep.jsonl: ~1 MB / year
  - workouts.jsonl: ~100 KB / year
  - body_mass.jsonl: ~10 KB / year

After ~5 years you'd be at ~100 MB total — still trivial. If you want
to prune older data, just truncate the JSONL files; the importer's
dedup is keyed on `(start_iso, source_version)` so re-import adds
nothing back.

## Verifying it works

After dropping an export and waiting for one inbox-tick:

```bash
# Should now contain 5–6 .jsonl files (one per record kind):
ls $WORKSPACE_ROOT/health/

# A processed-manifest for your zip:
ls $WORKSPACE_ROOT/inbox/.processed/

# The original zip moved to today's archive:
ls $WORKSPACE_ROOT/inbox/.archive/$(date +%Y-%m-%d)/
```

The next morning briefing will include a `❤️  Health (7d):` section
with steps/sleep/HR averages and any anomaly flags. If you don't see
it, check the gateway logs for `health summary:` lines and the
summary's `record_counts` for an unexpectedly empty kind.

## Failure modes you might see

| Symptom | Cause | Fix |
|---|---|---|
| Zip sits in inbox, manifest says `failed_zip` | Partial download | Re-export, ensure file finished transferring before drop |
| Zip processed but `records_written` is empty | Export was empty (no Health permissions on the source device) | Check Apple Health → Settings → "All sources" → toggle data sources |
| Briefing has no Health section | Less than 24 h since enabling, or no records exist | Wait one tick or check `$WORKSPACE_ROOT/health/` |
| `INBOX failure` Signal ping for a PDF/audio drop | Recognised but no handler wired | Expected — file stays in `inbox/` for you to handle manually |

## Composability with §5.4 (multi-modal inbox)

The inbox classifier identifies `apple_health_export*.zip` by **peeking
the zip index** for an `apple_health_export/export.xml` member. If you
rename the file, classification still works as long as the internal
structure is intact. A renamed zip with the wrong contents falls
through to "unknown" and gets a Signal ping rather than failing
silently.
