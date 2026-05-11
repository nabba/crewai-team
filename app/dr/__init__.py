"""Disaster-recovery primitives — portable export + import + boot drill.

PROGRAM §40 (2026-05-10) — Q3 Item 13.

Three modules:

  * ``export_kbs.py``  — produce a portable tarball with every KB's
                          ChromaDB content (as JSONL), key Postgres
                          tables (as JSONL), and canonical workspace
                          ledgers (affect, identity, audit). Excludes
                          secrets (``.env``, ``secrets/``,
                          ``google_token.json``, …).
  * ``import_kbs.py``  — read a portable tarball back into ephemeral
                          target directories. Used by the boot drill;
                          can also be used by operators to restore a
                          single KB into a sandbox for inspection.
  * ``boot_drill.py``  — end-to-end DR exercise: take latest export,
                          import into ephemeral target, run sanity
                          queries, write a drill report. Surfaces a
                          Signal alert if anything breaks.

Composition with the existing engine ``app/healing/db_backup.py``:

  * ``db_backup.py`` produces ``pg_dump``-style binary dumps from the
    live Postgres / Neo4j containers. Best for quick same-cluster
    restore.
  * ``app/dr/`` produces a **container-independent** tarball driven
    only by application APIs. The portable tarball is what the boot
    drill validates — i.e. "can a fresh laptop with only the tarball
    re-build a working KB?". This is the question the DR drill
    actually answers.

Both layers run on their own cadences and store under
``workspace/backups/`` without colliding.
"""
from __future__ import annotations

__all__ = ["export_kbs", "import_kbs", "boot_drill"]
