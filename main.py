"""Stale top-level entry point — DO NOT USE.

The real gateway is ``app.main:app``. The Dockerfile and run_host.py
both target it; this top-level ``main.py`` is an orphan from before
the May 2026 refactor and registers only a small subset of cron jobs
(no idle scheduler, no healing daemons, no MCP, no router mounts).

Importing or running this module is almost certainly a deploy
mistake. Fail loudly so the operator notices immediately rather than
booting a half-wired gateway that silently misses the idle scheduler,
the 22 healing monitors, the auditor bridge, the watchdog, and most
of the routers.

If you genuinely want the gateway, use one of:

    uvicorn app.main:app --host 0.0.0.0 --port 8765
    python run_host.py
"""

raise RuntimeError(
    "crewai-team/main.py is an orphan stub. "
    "Use `uvicorn app.main:app` (Dockerfile target) or `python run_host.py` "
    "for host-native dev. See module docstring for context."
)
