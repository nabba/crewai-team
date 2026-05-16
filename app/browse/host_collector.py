"""Host-native browser-history collector.

Runs OUTSIDE the gateway container — directly on macOS, with Full
Disk Access granted to the Python process. Reads ``~/Library/...``
browser history SQLite files, drops blocklisted entries, and writes
canonical events to ``./workspace/browse/events/<day>.jsonl``. That
directory is volume-mounted into the gateway container, so the
gateway-side LLM topic extraction (Phase B) and interest_model
collector (Phase B.2) pick up the events automatically.

Why this is a host-native split
-------------------------------

The gateway runs in Docker. Docker on macOS can't see ``~/Library/``
unless every browser directory is bind-mounted (brittle) AND Docker
Desktop itself has FDA (must be re-granted on every Docker upgrade).
A small dedicated host process keeps FDA scoped to one auditable
binary the operator can revoke from System Settings at any time.

The split is clean by file boundary:

  * Host: runs the readers (SQLite → events JSONL) — this module.
  * Gateway: runs the LLM topic extraction over the JSONL and the
    interest_model collector, both of which read from the volume-
    mounted workspace directory.

Usage
-----

  python -m app.browse.host_collector \\
      --workspace /Users/andrus/BotArmy/crewai-team/workspace --once

  python -m app.browse.host_collector \\
      --workspace /Users/andrus/BotArmy/crewai-team/workspace --watch

``--watch`` loops at ``--interval`` seconds (default 1800 = 30 min,
matching the gateway-side cadence). Honors
``BROWSE_INGESTION_ENABLED`` — when off, logs once and exits 0.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path


logger = logging.getLogger("browse.host_collector")


def _setup_workspace(path: Path) -> None:
    """Pin ``WORKSPACE_ROOT`` at the volume-mounted directory so
    :mod:`app.browse.store` writes events the gateway will see.

    The store also honors ``BROWSE_BASE_DIR`` for finer-grained
    overrides, but pinning ``WORKSPACE_ROOT`` is the right default
    here — the gateway's continuity ledger, audit logs, and every
    other module that takes a base path from workspace gets the
    same resolution.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"workspace path does not exist: {path}")
    os.environ["WORKSPACE_ROOT"] = str(path)


def _one_pass() -> int:
    """Run one aggregator pass on the host. Returns ``written`` count.

    Imports are deferred until ``WORKSPACE_ROOT`` has been pinned so
    the store's default-base evaluation lands on the right directory.
    """
    from app.browse import store
    from app.browse.aggregator import run_one_pass
    if not store.enabled():
        logger.warning(
            "BROWSE_INGESTION_ENABLED is off; nothing to do. "
            "Set BROWSE_INGESTION_ENABLED=true in the host env to enable."
        )
        return 0
    result = run_one_pass()
    logger.info(
        "browse.host: status=%s events=%d written=%d skipped_blocklisted=%d errors=%d",
        result.status,
        result.total_events,
        result.written,
        result.total_skipped_blocklisted,
        len(result.errors),
    )
    for browser, profile, err in result.errors:
        logger.warning("browse.host error: %s/%s — %s", browser, profile, err)
    return result.written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Browser-history host collector (PROGRAM §50).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Path to the workspace directory (the volume-mounted dir "
        "the gateway reads from). E.g. /Users/<you>/BotArmy/crewai-team/workspace.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="One pass + exit.")
    mode.add_argument(
        "--watch",
        action="store_true",
        help="Loop forever; one pass every --interval seconds.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1800,
        help="Watch-mode pass interval in seconds (default 1800 = 30 min).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _setup_workspace(args.workspace)

    if args.once:
        _one_pass()
        return 0

    # --watch mode.
    logger.info("browse.host: watching at %ds cadence", args.interval)
    while True:
        try:
            _one_pass()
        except Exception:
            logger.exception("browse.host: pass raised; sleeping anyway")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("browse.host: interrupted; exiting")
            return 0


if __name__ == "__main__":
    sys.exit(main())
