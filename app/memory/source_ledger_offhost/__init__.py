"""
source_ledger_offhost — Off-host destinations for the per-KB source ledger.

PROGRAM §56 (2026-05-17). Composes with the existing Q17.1 warm-spare
replication (which already covers ``workspace/`` hourly to the
operator's partner host). These add two additional destinations:

  * ``s3.py``      — S3-compatible bucket (any provider: AWS, Backblaze
                     B2, MinIO, Wasabi). Activated via env vars.
  * ``gdrive.py``  — Google Drive via the existing Google Workspace
                     OAuth pipe.

Both are off by default. The point isn't "use both at once"; it's
"have multiple options so the operator can pick a strategy that
doesn't depend on a single vendor or a single physical location."

Daily idle job model
====================

Each destination uploads incrementally — only the new ledger lines
since the last successful upload. State is per-destination JSON at
``workspace/<kb>/.source_ledger_<dest>_state.json`` with the
last-uploaded byte offset. Each daily upload creates a NEW object
keyed by date::

  s3://<bucket>/<host>/<kb>/source_ledger/2026-05-17.jsonl.gz

This append-only object lineage is the bullet-proof shape: recovery
is "list objects, sort by date, concatenate". Accidentally deleting
one day's object loses one day; doesn't poison the rest.
"""
from app.memory.source_ledger_offhost.common import (
    OffhostUploadResult,
    canonical_object_key,
    incremental_payload,
    record_upload,
)

__all__ = [
    "OffhostUploadResult",
    "canonical_object_key",
    "incremental_payload",
    "record_upload",
]
