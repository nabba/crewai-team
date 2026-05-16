"""warm_spare — partner-host replication & failover primitives (Q17.1).

Operator-driven. Three primitives:

  * manifest.py     — periodic SHA-256 snapshot of identity-critical files.
  * replication.py  — generates the rsync command-line; never executes it.
  * failover.py     — 5-state machine + typed-phrase claim_canonical().

This subsystem is HOST-AGNOSTIC: it operates on a known on-disk layout
and provides primitives the operator can wire into their own rsync /
restic / btrfs-send loop. We never store partner-host credentials in
the gateway.
"""
from __future__ import annotations

from app.warm_spare.failover import (
    FailoverState,
    claim_canonical,
    current_state,
    demote,
    record_heartbeat,
)
from app.warm_spare.manifest import (
    ManifestEntry,
    build_manifest,
    load_manifest,
    save_manifest,
)
from app.warm_spare.replication import (
    build_rsync_command,
    get_partner_target,
    set_partner_target,
    write_recipe_file,
)

__all__ = [
    "FailoverState",
    "ManifestEntry",
    "build_manifest",
    "build_rsync_command",
    "claim_canonical",
    "current_state",
    "demote",
    "get_partner_target",
    "load_manifest",
    "record_heartbeat",
    "save_manifest",
    "set_partner_target",
    "write_recipe_file",
]
