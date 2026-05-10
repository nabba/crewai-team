# Multi-modal inbox ingestion (§5.4) — operator guide

A single file-drop directory the user can use to feed the system
arbitrary content from any source. The watcher classifies each file,
routes to the right handler, archives on success, and surfaces
failures via Signal / Web Push.

Sister doc: [HEALTH_INGESTION.md](HEALTH_INGESTION.md) describes the
specific Apple Health pipeline this inbox routes to.

## What this is, briefly

The user drops a file into `workspace/inbox/`. Within seconds, an
idle-tick:

1. SHA-256 hashes the file (dedup against `.processed/<sha>.json`).
2. Skips the file if it was modified in the last 5 seconds (so a
   half-uploaded file isn't read prematurely).
3. Classifies by extension + magic-bytes peek.
4. Dispatches to the per-kind handler.
5. Archives the file on success at `.archive/<YYYY-MM-DD>/`.
6. Records a manifest at `.processed/<sha>.json` with the outcome.
7. Pings the user (Signal + Web Push) only when something needs
   attention.

The watcher is a LIGHT idle job — it costs sub-second CPU on every
tick where the inbox is empty.

## Enabling it

```bash
export INBOX_INGESTION_ENABLED=true
```

Optional knobs:

```bash
# Override the watch directory (default: $WORKSPACE_ROOT/inbox)
export INBOX_DIR=/path/to/inbox

# Override where the text handler drops .md/.txt files
# (default: $WORKSPACE_ROOT/notes — the React /cp/files view picks it up)
export INBOX_NOTES_DIR=/path/to/notes
```

No service restart needed; the next idle tick picks up the env change.

## Supported file kinds

The classifier table in [app/inbox/classifier.py](../app/inbox/classifier.py):

| Kind | Extensions | Magic-byte check | Handler |
|---|---|---|---|
| `apple_health_export` | `.zip` (with internal `apple_health_export/export.xml`) | zip-index peek | imports via §5.1 |
| `text` | `.txt`, `.md`, `.markdown` | none (trusted) | copies to `WORKSPACE_ROOT/notes/` |
| `audio` | `.m4a`, `.mp3`, `.wav`, `.ogg`, `.flac` | per-format | recognised, no handler yet |
| `image` | `.png`, `.jpg`, `.jpeg`, `.heic`, `.webp` | per-format | recognised, no handler yet |
| `pdf` | `.pdf` | `%PDF` | recognised, no handler yet |
| `csv` | `.csv` | none | recognised, no handler yet |
| `spreadsheet` | `.xlsx`, `.ods` | none | recognised, no handler yet |
| `unknown` | anything else | n/a | manifest written, file left in place |

"Recognised, no handler yet" means: the classifier identifies the kind
but no handler is wired. The file gets a `failed` manifest and the
user is notified. This is intentional — when a new handler ships, the
previously-failed files can be re-dropped (the dedup key clears once
the manifest is removed).

## Magic-byte signatures

The classifier rejects files whose extension doesn't match the actual
content. Per-format signatures (offsets in parentheses):

```
PNG  (0):  89 50 4E 47 0D 0A 1A 0A
JPG  (0):  FF D8 FF
WEBP (0):  52 49 46 46          ("RIFF")
HEIC (4):  ftypheic | ftypheix | ftyphevc | ftypmif1
PDF  (0):  25 50 44 46          ("%PDF")
MP3  (0):  49 44 33 ("ID3") | FF FB | FF F3 | FF F2
M4A  (4):  ftypM4A | ftypisom | ftypmp42
WAV  (0):  52 49 46 46          ("RIFF")
OGG  (0):  4F 67 67 53          ("OggS")
FLAC (0):  66 4C 61 43          ("fLaC")
```

A file with a `.png` extension but a JPEG magic byte falls through to
`unknown` and gets a Signal ping.

## Apple Health zip detection

Two-stage, robust against renames:

1. **Primary**: open the zip and look for an
   `apple_health_export/export.xml` (or `*/export.xml`) member. If
   found, classify as `apple_health_export`.
2. **Fallback** (zip unreadable, e.g. partial download): match the
   filename — `apple_health_export.zip` or
   `apple_health_export*.zip`. The importer's own `failed_zip` branch
   then catches the partial-download case with a specific reason.

This means the user can rename the export to anything and as long as
the zip's internal structure is intact, classification still works.

## Idempotency + dedup

Each file is hashed by content (SHA-256). The hash is the manifest key:
`workspace/inbox/.processed/<sha>.json`. This means:

* **Re-dropping the same bytes is a no-op.** The duplicate gets moved
  straight to the archive without re-running the handler.
* **Renamed files dedup correctly.** If you drop `data.zip` and then
  `data-renamed.zip` with identical content, the second is skipped.
* **Re-running after handler updates**: delete the manifest entry to
  force re-processing on the next drop.

## Failure surfacing

The scheduler's `_maybe_notify` fires a single Signal + Web Push when:

* An Apple Health import succeeded — notable enough to confirm.
* Any file failed (handler raised, manifest written with status
  `failed`).
* Any file's classification was `unknown` (no extension match).

Routine text drops are silent — push spam is worse than silent
success when the file is already visible in the React `/cp/files`
view.

The notification body lists up to 10 events:

```
Inbox — 2 need attention

✓ apple_health_export.zip: imported 73,481 records across 6 kinds
✗ vacation.pdf (pdf): no handler for kind='pdf'
? receipt.heic (image): magic bytes don't match
```

Click the push to open `/cp/files` (the inbox is one tab below).

## What ends up where

After a successful tick:

```
workspace/inbox/                              # always empty after a tick
workspace/inbox/.processed/<sha>.json         # manifest per file ever processed
workspace/inbox/.archive/2026-05-10/          # successful files (today)
workspace/inbox/.archive/2026-05-11/          # …grouped by drop date
workspace/notes/<filename>                    # text handler output (visible in /cp/files)
workspace/health/<kind>.jsonl                 # Apple Health output
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| File sits in inbox unprocessed | `INBOX_INGESTION_ENABLED=false` | Set env to true |
| File deferred indefinitely | Modified within last 5s on every tick | Stop modifying it; let the watcher catch it next tick |
| Successful Apple Health import but no Health section in briefing | `HEALTH_INGESTION_ENABLED=false` | Set env to true |
| Failed PDF / audio / image | No handler wired (expected) | Drop content elsewhere, or wire a handler in `HANDLER_REGISTRY` |
| Re-drop after handler update doesn't process | SHA matches a prior manifest | Delete `.processed/<sha>.json` |
| Inbox push every tick | Same failed file repeatedly | Move/delete the file; the manifest then keeps the dedup quiet |

## Adding a new handler

The handler signature is:

```python
Handler = Callable[[Path, FileClassification, Path], str]

def _handle_pdf(path: Path, classification: FileClassification, base: Path) -> str:
    """Process the PDF; return a one-line outcome for the manifest."""
    # ... do the work ...
    return f"extracted {n_pages} pages, indexed {n_chunks} chunks"
```

To wire it: add the entry in `HANDLER_REGISTRY` in [router.py](../app/inbox/router.py).
A handler that raises is caught — the file stays in place with
`status="failed"` and the user is pinged.

## Files

```
app/inbox/__init__.py            public API
app/inbox/classifier.py          extension + magic-bytes classifier
app/inbox/router.py              scan_and_route + handlers
app/inbox/scheduler.py           inbox-tick idle job + notify

tests/inbox/test_classifier.py   18 tests
tests/inbox/test_router.py       10 tests
tests/inbox/test_scheduler.py    6 tests

workspace/inbox/                 the user's drop directory
workspace/inbox/.processed/      per-file manifests
workspace/inbox/.archive/        successfully processed files (date-bucketed)
```

PROGRAM.md §34.2 is the canonical change-log entry.
