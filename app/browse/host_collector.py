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

  WORKSPACE_ROOT=/Users/andrus/BotArmy/crewai-team/workspace \\
  BROWSE_INGESTION_ENABLED=true \\
  python -m app.browse.host_collector \\
      --workspace /Users/andrus/BotArmy/crewai-team/workspace --once

``--watch`` instead of ``--once`` loops at ``--interval`` seconds
(default 1800 = 30 min, matching the gateway-side cadence). Honors
``BROWSE_INGESTION_ENABLED`` — when off, logs once and exits 0.

WORKSPACE_ROOT requirement
--------------------------

The ``WORKSPACE_ROOT`` env var **must** be set BEFORE Python evaluates
``-m app.browse.host_collector``. Reason: importing ``app.browse``
chains through ``app.browse.idle_job → app.browse.store →
app.paths``, and ``app.paths`` reads ``WORKSPACE_ROOT`` at module-
import time. The ``--workspace`` flag is preserved for explicit
intent and is checked against the env var, but it cannot retroactively
re-route imports that already happened.

The launchd plist at ``scripts/browse_host_collector.plist`` sets it
correctly. If you're running this manually from a shell, export it
first.
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
    """Validate the workspace path and verify the env var matches it.

    The actual ``WORKSPACE_ROOT`` env var must already be set when
    ``-m app.browse.host_collector`` starts, because ``app.paths``
    reads it at module-import time and the package init chain
    triggers that import before ``main()`` runs. We can still set it
    here as a fallback for tooling that re-exec's into another
    Python process — but if it wasn't set going in, the in-process
    ``app.paths.WORKSPACE_ROOT`` is already wrong and we can't fix
    it retroactively. Fail loudly with an actionable message.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"workspace path does not exist: {path}")
    os.environ.setdefault("WORKSPACE_ROOT", str(path))

    # Late-binding check: if app.paths has already been imported with
    # a different WORKSPACE_ROOT (the bug we hit on first deploy),
    # bail out with a precise diagnosis. Don't try to write events
    # into the wrong place.
    from app import paths as _paths  # already imported via the chain
    if _paths.WORKSPACE_ROOT.resolve() != path:
        raise SystemExit(
            "WORKSPACE_ROOT mismatch: "
            f"app.paths sees {_paths.WORKSPACE_ROOT.resolve()}, "
            f"--workspace says {path}. Export WORKSPACE_ROOT="
            f"{path} BEFORE invoking `python -m app.browse.host_collector`. "
            "See module docstring for why `-m` makes the runtime set "
            "too late."
        )


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
