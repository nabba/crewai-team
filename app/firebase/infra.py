"""
firebase.infra — Shared Firestore infrastructure.

Core utilities used by all firebase.* modules:
  - _get_db(): lazy Firestore client initialization
  - _fire(): fire-and-forget thread pool submission
  - _now_iso(): UTC timestamp helper
  - _add_activity(): activity feed writer
  - _prune_activities(): activity feed cleanup

Extracted from firebase_reporter.py to enable decomposition
without circular imports.
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_db = None
_db_lock = threading.Lock()
_PROJECT_ID = "botarmy-ba0c9"

# Bounded thread pool — shared by all firebase modules
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="firebase")


def _get_db():
    """Lazy-initialise the Firestore client once per process."""
    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is not None:
            return _db
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
            if sa_path and os.path.exists(sa_path):
                cred = credentials.Certificate(sa_path)
            else:
                cred = credentials.ApplicationDefault()

            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {"projectId": _PROJECT_ID})

            _db = firestore.client()
            logger.info("firebase.infra: Firestore client initialised")
        except Exception:
            logger.warning("firebase.infra: Firestore unavailable — monitoring disabled", exc_info=True)
            _db = False
    return _db


def _now_iso() -> str:
    from app.utils import now_iso
    return now_iso()


def _fire(fn):
    """Run fn in the bounded thread pool so Firestore latency never blocks agents."""
    try:
        _executor.submit(fn)
    except RuntimeError:
        pass


def _add_activity(db, event_type: str, crew: str, detail: str, task_id: str = "") -> None:
    """Append an event to the rolling activity feed (last 50)."""
    try:
        import uuid as _uuid
        db.collection("activities").document(_uuid.uuid4().hex[:12]).set({
            "type": event_type,
            "crew": crew,
            "detail": detail[:300],
            "task_id": task_id,
            "ts": _now_iso(),
        })
        _prune_activities(db)
    except Exception:
        pass


def _prune_activities(db, max_items: int = 50) -> None:
    """Keep only the most recent max_items activities."""
    try:
        from google.cloud.firestore_v1 import Query
        docs = (
            db.collection("activities")
            .order_by("ts", direction=Query.DESCENDING)
            .offset(max_items)
            .limit(20)
            .stream()
        )
        for doc in docs:
            doc.reference.delete()
    except Exception:
        pass
