# Source ledger — 10-year resiliency mechanism

**PROGRAM §56 (2026-05-17)** — companion to §55 (which closed the
dual-writer corruption). §56 is the layer that guarantees **every
ChromaDB KB is automatically reconstructable** from a plain JSONL
file, even if the ChromaDB files are lost entirely.

## The one-line summary

> Every chromadb write is mirrored to `workspace/<kb>/.source_ledger.jsonl`
> (append-only, hash-chained). The KB becomes a derived artifact —
> lose the chromadb files, replay the ledger, get the KB back.

## Why this hits 10 years

| Failure class | What protects you |
|---|---|
| Dual-writer corruption | §55 (removed orphan service) — root cause gone |
| Single-process corruption | WAL mode + daily integrity check + auto-quarantine |
| Filesystem bit-rot | §56 bit_rot_scan extension watches the ledger SHA-256 baseline |
| Embedding model rotation | Ledger stores text not embeddings — replay always uses current model |
| ChromaDB major version | JSONL outlives any vendor — replay code targets `col.upsert(...)`, stable across versions |
| Local disk total loss | Q17.1 warm-spare hourly to operator's partner host (already operational) |
| Partner host also dead | Off-host uploads (S3 / Google Drive) — opt-in via runtime_settings |
| Ledger itself damaged | Hash chain catches tampering; restore from off-host |
| "I want this to work in 10 years" | Quarterly drill proves the recovery pipeline still functions end-to-end |

## How it composes with existing infrastructure

```
                                  ┌── store_team_decision()
                                  ├── episteme.add_documents()
                                  ├── experiential.add_entry()
                       writes ────┼── philosophy.add_documents()
                                  ├── aesthetics.add_pattern()
                                  ├── tensions.add_tension()
                                  └── knowledge_base.add_document()
                                          │
                                          ▼
                              chromadb (HNSW + SQLite)
                                          │
                                          ▼ DUAL-WRITE hook
                          workspace/<kb>/.source_ledger.jsonl
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
            Q17.1 warm-spare        §56 off-host (S3/GDrive)   §55 integrity
            (already hourly)        (opt-in; daily)            (boot + daily)
                  │                       │                       │
                  └───────── recovery sources ────────────────────┘
                                          │
                                          ▼
                          source_ledger.replay_kb(name)
                       — re-embeds with current model —
                                          │
                                          ▼
                              Fresh, fully-functional KB
```

## Subsystems shipped

| Subsystem | File | What it does |
|---|---|---|
| Core ledger | `app/memory/source_ledger.py` | Hash-chained append, read, replay, drift detection |
| Dual-write hook (memory KB) | `app/memory/chromadb_manager.py` | Mirrors every `store()` to the ledger |
| Dual-write hooks (other 5 KBs) | `app/{episteme,experiential,philosophy,aesthetics,tensions,knowledge_base}/vectorstore.py` | Same via `hook_collection_add` |
| Bootstrap + drift daemon | `app/memory/source_ledger_daemon.py` | Daily back-fill + drift detection + chain verification |
| Boot drift detection | `app/memory/chromadb_integrity.py` (§55 extended) | Catches drift at gateway start |
| S3 uploader | `app/memory/source_ledger_offhost/s3.py` | Daily incremental upload; date-keyed objects |
| GDrive uploader | `app/memory/source_ledger_offhost/gdrive.py` | Same shape via existing Google OAuth |
| Bit-rot extension | `app/healing/monitors/bit_rot_scan.py` (Q17.3 extended) | Watches every `.source_ledger.jsonl` |
| Quarterly drill | `app/resilience_drills/drills/source_ledger_replay.py` | Rebuilds a random KB to scratch + verifies |
| Tests | `tests/test_source_ledger.py` | 22 tests (21 pass + 1 skipped gateway-deps) |

## Ledger schema

Each line of `workspace/<kb>/.source_ledger.jsonl` is a JSON object:

```json
{
  "ts": 1747512345.123,
  "collection": "beliefs",
  "doc_id": "uuid-string",
  "text": "the indexed text verbatim",
  "metadata": {"agent": "researcher", "scope": "team"},
  "prev_hash": "<64-hex>",
  "hash": "<64-hex>"
}
```

Where `hash = sha256(prev_hash + canonical_json(everything_except_hash))`.
Genesis `prev_hash` is 64 zeros.

`canonical_json` = `json.dumps(..., sort_keys=True, separators=(",", ":"))` —
deterministic bytes regardless of dict insertion order.

## Operator runbook

### "Source-ledger drift" Signal alert

Auto-recovery already fired. The daemon detected the live KB had
fewer rows than the ledger, ran `replay_kb()`, and re-embedded the
missing rows. Check `workspace/<kb>/.source_ledger.jsonl` and the
KB count to confirm convergence:

```bash
docker exec crewai-team-gateway-1 python -c "
from app.memory.source_ledger import check_drift, list_kbs
for kb in list_kbs():
    print(kb, check_drift(kb).to_dict())
"
```

### "Source-ledger hash chain broken" Signal alert

**Critical** — the ledger itself has been damaged or tampered with.
Don't trust auto-recovery here.

Restore from off-host (most recent good copy):

```bash
# Q17.1 warm-spare (always-on)
rsync -av \
  andrus.raudsalu@andrus-macbook-pro-16.tail5b289b.ts.net:~/andrusai-mirror/<kb>/.source_ledger.jsonl \
  workspace/<kb>/.source_ledger.jsonl

# Or S3 (if enabled)
aws s3 cp s3://<bucket>/<host>/<kb>/source_ledger/$(date +%F).jsonl.gz - | \
  gunzip > workspace/<kb>/.source_ledger.jsonl

# Or Google Drive (if enabled) — operator manually downloads the file
```

After restore, re-run chain verification:

```bash
python -c "
from app.memory.source_ledger import verify_chain
print(verify_chain('<kb>').to_dict())
"
```

### "Rebuild a KB from scratch"

```bash
# Quarantine the live KB
mv workspace/<kb>/ workspace/<kb>.manual_$(date +%Y%m%d_%H%M%S)/

# Replay
python -c "
from app.memory.source_ledger import replay_kb
print(replay_kb('<kb>').to_dict())
"
```

The boot integrity scan automates this same flow on detected
quarantine.

### "Set up off-host backup"

S3 (works with AWS, Backblaze B2, Wasabi, MinIO, R2):

```bash
# Add to .env (gateway reads on next restart)
LEDGER_S3_BUCKET=my-bucket
LEDGER_S3_ACCESS_KEY_ID=...
LEDGER_S3_SECRET_ACCESS_KEY=...
LEDGER_S3_ENDPOINT_URL=https://...  # optional; AWS doesn't need it

# Toggle ON in /cp/settings (or via runtime_settings setter)
```

Google Drive (uses existing Google Workspace OAuth):

```bash
# Toggle ON in /cp/settings — no extra setup needed if you already
# have the Google tools working (gmail, calendar, etc.).
# The daemon creates a folder "AndrusAI-Ledgers" under My Drive.
```

Both uploaders are daily idle jobs. Object lineage:
`<host>/<kb>/source_ledger/YYYY-MM-DD.jsonl.gz`. Each day creates a
new object — append-only lineage. Recovery is "list objects, sort,
concatenate."

### Quarterly drill output

Check `/cp/settings → Resilience drills` or:

```bash
tail -20 workspace/resilience/drill_audit.jsonl | \
  grep source_ledger_replay
```

A passing drill emits a row like:

```json
{
  "drill_name": "source_ledger_replay",
  "status": "PASS",
  "detail": {
    "kb_name": "philosophy",
    "ledger_rows": 3026,
    "scratch_rows": 3026,
    "loss_pct": 0.0,
    "chain_verify": {"ok": true}
  }
}
```

## Runtime settings

| Key | Default | What it does |
|---|---|---|
| `chromadb_source_ledger_enabled` | ON | Master kill switch for §56 |
| `chromadb_ledger_bootstrap_enabled` | ON | Daily back-fill from chromadb |
| `chromadb_ledger_drift_replay_enabled` | ON | Auto-replay on detected drift |
| `chromadb_ledger_s3_upload_enabled` | OFF | S3 off-host (needs creds) |
| `chromadb_ledger_gdrive_upload_enabled` | OFF | Google Drive off-host (needs OAuth) |
| `drill_source_ledger_replay_enabled` | ON | Quarterly rebuild drill |

All flippable from `/cp/settings` without restart. Failure-OPEN
posture: missing settings file → defaults ON, protection stays in
place.

## What this does NOT protect against

- **Pre-§56 data**: rows added to chromadb before §56 shipped aren't
  in the ledger yet. The bootstrap daemon back-fills on idle —
  ~5 minutes after first gateway boot, then daily. Full coverage
  for a 4k-row KB takes ~3 days at the 5000-rows-per-pass cap.
- **Embedding-model rotation that breaks similarity, not retrieval**:
  if you swap to a fundamentally incompatible embedding model, the
  text in the ledger still re-embeds fine, but the *meaning* of
  similarity may shift. This is a feature, not a bug — re-embedding
  with the new model is exactly what you want.
- **Operator deletes the ledger on purpose**: there's no protection
  against deliberate deletion. The hash chain catches tampering
  WITHIN the ledger, not deletion of the whole file. Operator-
  initiated `rm` is undone by restoring from warm-spare / off-host.

## Re-open conditions

Don't re-audit §56 unless:

1. The drill starts failing.
2. The hash chain breaks on real data (not test scenarios).
3. Drift detection triggers a replay AND the replay doesn't converge.
4. Bootstrap goes >7 days without making progress on a non-trivial KB.
5. A new KB shows up that the dual-write hooks don't cover.

Routine verification: the quarterly drill + the daily daemon
+ bit_rot_scan + `/cp/settings → ChromaDB integrity` together cover
the system's normal observation surface. The operator doesn't need
to do anything for 10-year-resiliency-mode to keep working —
that's the point.
