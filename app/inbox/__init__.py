"""Multi-modal file-drop ingestion (§5.4 — decade-class roadmap).

The user drops files into ``workspace/inbox/`` from any source —
AirDrop, scp, the React UI, an Apple Health export. This package
watches the drop, classifies each file, routes to the right handler,
and archives the result. One unified file-drop interface for the
personal-agent surface.

Design discipline
-----------------

  - **Default OFF.** ``INBOX_INGESTION_ENABLED`` defaults to ``false``.
    The user opts in once the right archive directory + handler set
    is in place.

  - **Pluggable handlers.** Each handler maps an extension set to a
    callable; the registry is the source of truth for "what we can
    process." Adding a new file type is one entry plus the handler.

  - **Hash-deduplication.** Each file's SHA-256 is recorded at
    ``workspace/inbox/.processed/<sha>.json``; re-dropping the same
    file is idempotent. The manifest records ``handler``, ``status``,
    ``processed_at``, ``moved_to`` so the operator can audit.

  - **Conservative archiving.** Successfully processed files move to
    ``workspace/inbox/.archive/<YYYY-MM-DD>/`` so the inbox itself
    stays clear. Failed files stay in place so the operator can see
    what didn't work.

  - **Failure-isolated.** A handler that raises is caught, the file
    is left in place, and the manifest records ``status="failed"``
    with the reason. Other files in the same tick continue.

Public API
----------

  * :func:`scan_and_route` — one tick of the watcher; returns a
    structured :class:`ScanResult` so the operator can audit.
  * :func:`get_idle_jobs` — registered with companion.loop; LIGHT.
"""
from app.inbox.classifier import FileClassification, classify_file
from app.inbox.router import HANDLER_REGISTRY, ScanResult, scan_and_route
from app.inbox.scheduler import get_idle_jobs

__all__ = [
    "FileClassification",
    "HANDLER_REGISTRY",
    "ScanResult",
    "classify_file",
    "get_idle_jobs",
    "scan_and_route",
]
