"""
firebase_reporter.py — Real-time status reporting to Firestore.

Writes agent activity, task state, and system health to Firebase so the
monitoring dashboard can display live status without polling the server.

Firestore schema:
  status/system          — overall health, uptime, last-seen
  crews/{name}           — per-crew status, current task, eta
  tasks/{id}             — individual task records with ETA tracking
  activities/{id}        — rolling activity feed (last 50 entries)
  schedule/jobs          — upcoming cron-scheduled work

All writes are fire-and-forget (non-blocking) so a Firestore outage
never degrades agent performance.
"""

import logging
import os
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_db = None
_db_lock = threading.Lock()
_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "botarmy-ba0c9")  # Configurable via env


def _get_db():
    """Lazy-initialise the Firestore client once per process with thread safety."""
    global _db
    if _db is not None:
        return _db
    
    with _db_lock:
        if _db is not None:  # Double-checked locking pattern
            return _db
            
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            # Prefer a service-account JSON file; fall back to Application Default Credentials
            sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
            if sa_path and os.path.exists(sa_path):
                cred = credentials.Certificate(sa_path)
            else:
                cred = credentials.ApplicationDefault()

            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {"projectId": _PROJECT_ID})

            _db = firestore.client()
            logger.info("firebase_reporter: Firestore client initialised")
        except Exception as exc:
            logger.warning(f"firebase_reporter: Firestore unavailable — monitoring disabled: {exc}")
            _db = False   # sentinel: skip all future attempts
    return _db

# ... rest of file remains the same ...