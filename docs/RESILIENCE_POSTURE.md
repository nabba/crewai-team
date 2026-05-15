# Resilience Posture

**Decision (2026-05-13, PROGRAM §44.0):**
The system commits to **good backup + fast bare-metal recovery**, not
high-availability. **Identity is data, not uptime.**

This decision is load-bearing. Future infrastructure proposals that
assume HA (replication, failover, leader election, hot standby) should
be evaluated against this commitment first.

## Why no HA

### The identity model is captured in data, not compute

The system's identity lives in durable stores:

| Identity element | Storage |
|---|---|
| SubIA integrity manifest | git + hash-chained file |
| Continuity ledger | append-only JSONL + Postgres mirror |
| Identity claims FIFO | JSON file |
| Affect trace | JSONL with archive rotation |
| Welfare audit | JSONL with archive rotation |
| Beliefs | Postgres + Neo4j + ChromaDB |
| Tool registry | Postgres snapshot + ChromaDB index |

Replicating **compute** (HA) doesn't replicate **identity** any better
than backup does. A complete restore from backup rebuilds the system's
identity. Two HA replicas don't add anything backup didn't already.

### HA introduces failure modes that don't help us

For a single-operator personal system:

- **Split-brain risk**: distributed consensus over Postgres + Neo4j +
  ChromaDB needs a quorum protocol. Setting that up has its own
  failure modes.
- **Real failure modes don't benefit from HA on the same host**:
  - Power outage → HA on same machine doesn't help
  - Disk failure → backup + restore is the answer, not HA
  - Software bug → both replicas have the bug
  - Cloud-provider outage → backup-to-alternate is the answer

### Cost / benefit doesn't justify HA at personal scale

- HA doubles infrastructure cost
- Personal use can tolerate ~30 minutes of downtime
- The operator has alternative channels (Signal works without the
  gateway; conversation history can resume after restore)

## What we DO invest in

The Q6 quarterly drills (`docs/RESILIENCE_DRILLS.md`) operationalize
this posture:

1. **`backup_restore` drill** — verifies recovery procedure works
2. **`embedding_migration` drill** — verifies substrate migration works
3. **`secret_rotation` drill** — verifies rekey procedure works
4. **`kill_the_gateway` drill** — empirically measures recovery time

Plus the existing healing-monitor infrastructure that detects
component-level degradation between drills.

## Off-host backup policy

**Dual-target**: both S3 and Google Drive.

Rationale for two targets:
- Single-cloud failure mode (S3 outage, account-locked, account-deleted)
  is real
- Google Drive is operator-accessible without infrastructure (a
  human can manually retrieve a tarball even if the system is gone)
- S3 supports versioning + lifecycle policies (cold storage)

**Cadence**: weekly off-host upload. Operator-managed via a separate
cron / scheduled task (not part of the gateway). The
`workspace/backups/dr/` directory is the local source-of-truth;
operator's sync script copies tarballs to both targets.

**Drill verification**: the `backup_restore` drill verifies the
most recent LOCAL tarball restores correctly. It does NOT verify
off-host integrity (that would require pulling from S3/GDrive on every
drill — expensive and adds dependency on those providers). The
operator's off-host sync should have its own integrity verification.

**Future enhancement** (not Q6 scope): a separate "off-host verify
drill" that pulls a random old tarball from S3 or Google Drive,
verifies SHA-256, and confirms the operator's off-host pipeline
hasn't silently broken.

## Recovery time target

**30 minutes** from gateway death to fully-restored operation. This
target is verified empirically by the `kill_the_gateway` drill. If
three consecutive drills exceed this target, escape condition #2
(below) triggers a posture re-review.

## Escape conditions

The posture decision is FIXED for v1. It re-opens for operator-driven
review under any of:

1. **Operator-facing SLA commitment to a third party**: if the system
   ever serves a non-Andrus consumer who depends on uptime, the
   personal-scale argument no longer holds.

2. **Recovery time exceeds 30 minutes for 3 consecutive
   `kill_the_gateway` drills**: empirical evidence that the recovery
   procedure has drifted past the acceptable window.

3. **Hard real-time consumer appears**: if a new subsystem (voice
   call, alarm system, etc.) requires sub-minute latency, the
   posture must be re-evaluated.

4. **Disk failure or unrecoverable corruption**: actual occurrence of
   the failure mode the posture is designed to handle. Even if the
   recovery succeeds, log it as evidence; if it happens twice in a
   year the posture re-opens.

When any escape condition fires, the operator writes a posture-revision
decision document and amends `app/resilience_drills/posture.py:Posture`
constants. That module is not TIER_IMMUTABLE; this document IS the
governance gate.

## What this posture does NOT mean

- **Backups are not optional.** The drill verifies the backup actually
  works. A backup that never restores is worse than no backup
  (false sense of security).
- **The system can be down for arbitrary time.** Operator notice +
  recovery + verification should happen within hours, not days.
- **No multi-host configuration is permitted.** It's fine to have
  the gateway on one host and an off-host backup target on another.
  The posture rejects ACTIVE replication, not distributed storage.

## See also

- `docs/RESILIENCE_DRILLS.md` — operator guide for the quarterly drills
- `docs/DR_DRILL.md` — backup-restore drill detail (pre-Q6 doc)
- `app/resilience_drills/posture.py` — programmatic exposure of these
  constants for guard checks
