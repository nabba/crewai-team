# Embedding-Model Migration

> PROGRAM §40 (2026-05-10) — Q3 Item 12.

## Why this exists

The system pins **all** ChromaDB collections and pgvector columns to
one embedding dimension:

```python
# app/memory/chromadb_manager.py
_EMBED_DIM = 768  # IMMUTABLE — pinned to Ollama nomic-embed-text
```

The constant is in TIER_IMMUTABLE. Changing it silently corrupts every
retrieval — vectors of incompatible dim simply don't match. So we don't
change it casually. We change it through a controlled, reversible
multi-phase migration that the framework in `app/memory/embedding_migration/`
makes safe and auditable.

## The shape of a migration

```
IDLE ─► PLANNED ─► DUAL_WRITE ─► BACKFILLING ─► SHADOW_READ ─► READY
                                                                  │
                                                                  ▼
                                                              CUTOVER  ──► APPLIED ──► STANDDOWN_COMPLETE
                                                                  │
                                                                  ▼ (any phase)
                                                              ABORTED
```

Five operator actions move the system through that graph:

1. **Author + adopt a plan** — declare source / target models and which
   collections + columns to migrate.
2. **Enable dual-write** — every new `store()` to a source collection
   ALSO writes to a target-dimension shadow collection. Reads remain
   on source.
3. **Run the backfill** — drain historical rows through the target
   embedder into the shadow collection. Idempotent (id-dedup).
4. **Enable shadow-read** — sampled live retrieves run the same query
   against the shadow; rolling NDCG@10 records the divergence.
5. **Propose cutover** — gated by Tier-3 amendment + the verifier (DR
   backup freshness, NDCG threshold, row-count parity, plus phase
   READY). On `mark_applied`, the post-apply hook atomically renames
   shadow → source and source → archive.

Each step is a separate runtime_settings toggle. Operators can pause
between any two phases — the framework is fully observational until the
master switch for the next phase flips.

## Master switches (React `/cp/settings`)

| Key | Default | Effect |
|-----|---------|--------|
| `embedding_migration_dual_write_enabled` | `false` | Source `store()` writes to shadow when phase in `{DUAL_WRITE, BACKFILLING, SHADOW_READ, READY}`. |
| `embedding_migration_shadow_read_enabled` | `false` | Source `retrieve()` samples shadow when phase in `{SHADOW_READ, READY}`. |
| `embedding_migration_cutover_enabled` | `false` | The cutover *unblock* signal. Cutover still requires phase=READY + a Tier-3 amendment that passes the eligibility gate. |

Plus the state-machine blob `embedding_migration_state` — operators do
not edit this; the state machine owns it.

## Plan format

```json
{
  "plan_id": "ollama-nomic-to-mxbai-2026-Q3",
  "source": {
    "provider": "ollama",
    "name": "nomic-embed-text",
    "dim": 768
  },
  "target": {
    "provider": "ollama",
    "name": "mxbai-embed-large",
    "dim": 1024
  },
  "targets": [
    {"kind": "chromadb", "kb": "memory", "collection": "team_shared"},
    {"kind": "chromadb", "kb": "philosophy", "collection": "philosophy"},
    {"kind": "chromadb", "kb": "knowledge", "collection": "knowledge"}
  ],
  "cutover_threshold_ndcg": 0.95,
  "cutover_min_shadow_queries": 1000,
  "standdown_retention_days": 30,
  "notes": "Move to mxbai for better retrieval quality on long docs."
}
```

Save to `workspace/embedding_migration/plan.json`. The framework
reads it on every operation.

## Operator runbook

### Phase 0 — dry-run

Before authoring a real plan, run the dry-run pipeline against a
sandbox collection:

```bash
python -m app.memory.embedding_migration.dry_run \
    --target-provider ollama \
    --target-model mxbai-embed-large \
    --target-dim 1024
```

Expected output is a JSON report with `"ok": true` after ~5-10 seconds.
If any step shows `"ok": false`, fix the underlying issue (most often:
target embedder not reachable) before authoring a real plan.

### Phase 1 — author + adopt

1. Author `workspace/embedding_migration/plan.json` (see above).
2. In a Python REPL (or via a small script), adopt the plan:

   ```python
   from app.memory.embedding_migration import state, plan
   p = plan.load_plan()
   print(p.display_summary())          # eyeball the plan
   state.adopt_plan(p.plan_id)         # IDLE → PLANNED
   state.transition("DUAL_WRITE", reason="operator")
   ```

3. Flip `embedding_migration_dual_write_enabled` in `/cp/settings`.

New writes will now go to both source and shadow.

### Phase 2 — backfill historical rows

For each collection in the plan:

```python
from app.memory.embedding_migration.dual_write import backfill_one_collection
summary = backfill_one_collection("team_shared", batch_size=200)
print(summary)
```

Backfill is idempotent — you can stop and resume freely.

When backfill completes for every collection:

```python
state.transition("BACKFILLING", reason="backfill complete")  # noop if already
state.transition("SHADOW_READ", reason="ready for shadow read")
```

Flip `embedding_migration_shadow_read_enabled` in `/cp/settings`.

### Phase 3 — let shadow-read converge

The framework samples a fraction (default 5%, env
`EMBED_MIGRATION_SHADOW_SAMPLE_RATE`) of live retrieves and records
NDCG@10 against the shadow. Watch the React migration card until:

  * `shadow_query_count >= plan.cutover_min_shadow_queries`
  * `mean NDCG@10 >= plan.cutover_threshold_ndcg`

Then advance:

```python
state.transition("READY", reason="convergence threshold met")
```

### Phase 4 — cutover

1. Run a fresh DR export (the verifier requires a tarball <7 days old):

   ```bash
   python -m app.dr.export_kbs
   ```

2. Verify everything passes:

   ```python
   from app.memory.embedding_migration import verify
   r = verify.verify()
   for c in r.checks:
       print(f"  [{'OK' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
   assert r.ok
   ```

3. Flip `tier3_amendment_enabled=true` + `embedding_migration_cutover_enabled=true`
   in `/cp/settings`.

4. File the Tier-3 amendment:

   ```python
   from app.memory.embedding_migration import cutover
   result = cutover.propose_cutover()
   print(result.to_dict())
   ```

5. The proposal goes through the standard Tier-3 lifecycle (PROPOSED →
   STAGED → COOLDOWN_OK → APPROVED → APPLIED → STABLE). After APPLIED:

   ```python
   apply_result = cutover.post_apply_hook()
   ```

   This swaps the chromadb collections and moves state → `APPLIED`.

### Phase 5 — stand-down

For `standdown_retention_days` after APPLIED, the archived source
collections (`<name>__archive_<timestamp>`) remain in place. If a
regression appears, the operator can roll back by reversing the swap.

After the retention window, an operator deletes the archive
collections manually:

```python
from app.memory import chromadb_manager
client = chromadb_manager.get_client()
client.delete_collection("team_shared__archive_20260911T060000Z")
state.transition("STANDDOWN_COMPLETE", reason="retention window elapsed")
```

## Failure modes + remediation

### Dry-run fails at `dual_write_seed`

Almost always means the target embedder isn't reachable. Check:

  * Is the target Ollama model pulled? `ollama list` should show it.
  * Is `target.base_url` correct in the plan?

### Verifier reports `shadow_row_match` failure

The shadow has drifted from the source row count by more than 1%.
Most common cause: the backfill didn't complete for some collections.
Re-run `backfill_one_collection` on the affected collection.

### Verifier reports `ndcg_threshold` failure

The shadow rankings diverge from the source too much. Three options:

  * **Wait longer** — the rolling window may not have stabilised.
  * **Lower the threshold** in the plan if the target model is
    intentionally different (e.g. multilingual).
  * **Abandon the plan** — `state.abort("metrics diverged")` and
    delete the shadow collections by hand.

### Cutover hook fails partway through swap

The hook is best-effort but not transactionally atomic — chromadb
has no rename API. If it fails partway:

  * The archive collection `<name>__archive_<ts>` has the old data.
  * The live collection might be empty or partially populated.
  * **Roll forward manually**: dump the shadow content with
    `app.memory.chromadb_rebuild --kb <kb> --collection <shadow_name> --dry-run`,
    then `--from-snapshot <path>` into the live name.

## Plan validator allowlist (Q3.1 — 2026-05-11)

The framework's chromadb dual-write path is correct only for the
`memory` KB today. Pgvector dual-write is declared in the plan
schema but **not yet wired**. To prevent silently shadowing data
into the wrong place, the plan validator refuses unsupported
targets at both save and load time:

  * `SUPPORTED_TARGET_KINDS = {"chromadb"}` — pgvector / Neo4j refused
  * `SUPPORTED_CHROMADB_KBS = {"memory"}` — other KBs refused

Broadening the allowlist requires implementing the corresponding
dual-write path AND verifying end-to-end. Plans that hand-edit the
on-disk JSON to bypass the validator are rejected at next load.

## What the framework does NOT do

* **Migrate pgvector columns.** The pgvector path is declared in the
  plan but the runtime hooks are not wired (yet). Pgvector tables are
  small enough that a one-shot `UPDATE … SET embedding = …` with
  downtime is fine for now. We'll wire this in a follow-up when the
  pgvector population is large enough to need a phased migration.
* **Run on its own schedule.** Every phase requires an explicit
  operator action. There is no "auto-advance" daemon — by design.
  Migration is a deliberate event, not a background optimization.
* **Optimise NDCG.** The metric is a CUTOVER GATE. The system never
  trains against it.

## Why the protocol is gated by Tier-3 amendment

`_EMBED_DIM` is in TIER_IMMUTABLE. Every Tier-3 file is in TIER_IMMUTABLE
because changing it has consequences beyond a single function — a
silent dim change rewrites the meaning of every existing embedding,
breaking every retrieval in the system. The Tier-3 amendment protocol
forces:

  * **Citation ≥30 chars** explaining the reason (cutover.py provides
    a programmatically-generated one referencing the plan).
  * **Eligibility check** — promotion history + rollback rate + alignment
    audit must all pass.
  * **Audit chain** — hash-linked record of who proposed, who staged,
    who approved.
  * **Self-quarantine list** — `_EMBED_DIM` is verified to be outside
    the quarantine (it isn't — it's TIER_IMMUTABLE but quarantine-
    exempt — so the propose call succeeds).

This is exactly the kind of change the Tier-3 protocol exists for.
The embedding migration is, in fact, **the first production consumer
of the previously-dormant Tier-3 amendment path**.

## See also

* `app/governance_amendment/protocol.py` — Tier-3 lifecycle
* `docs/TIER3_AMENDMENT.md` — operator runbook for the protocol
* `app/memory/chromadb_rebuild.py` — operator-runnable per-collection rebuild (failsafe)
* `docs/DR_DRILL.md` — the DR drill that the verifier requires to be ≤7 days old
