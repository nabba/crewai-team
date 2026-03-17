from collections import defaultdict
from datetime import datetime, timedelta, timezone
from app.config import get_settings
import logging
import threading

logger = logging.getLogger(__name__)
settings = get_settings()

# Rate limit: max 30 messages per 10 minutes per sender
_rate_buckets: dict[str, list[datetime]] = defaultdict(list)
_rate_lock = threading.Lock()  # Guard concurrent access from async thread pool
MAX_MESSAGES = 30
WINDOW_MINUTES = 10
_MAX_TRACKED_SENDERS = 1000  # Prevent memory exhaustion from spoofed senders


def _redact_number(number: str) -> str:
    """Redact phone number for safe logging: +3725100500 -> +372***0500"""
    if len(number) > 7:
        return number[:4] + "***" + number[-4:]
    return "***"


def is_authorized_sender(sender: str) -> bool:
    """Only the owner's number may send commands."""
    # Normalize: strip whitespace, ensure + prefix
    normalized_sender = sender.strip()
    normalized_owner = settings.signal_owner_number.strip()
    authorized = normalized_sender == normalized_owner
    if not authorized:
        logger.warning(f"Blocked unauthorized sender: {_redact_number(normalized_sender)}")
    return authorized


def is_within_rate_limit(sender: str) -> bool:
    """Prevent runaway loops or abuse."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=WINDOW_MINUTES)

    with _rate_lock:
        # Evict stale senders to prevent memory exhaustion
        if len(_rate_buckets) > _MAX_TRACKED_SENDERS:
            stale = [k for k, v in _rate_buckets.items() if not v or v[-1] < cutoff]
            for k in stale:
                del _rate_buckets[k]

        bucket = [t for t in _rate_buckets[sender] if t > cutoff]
        _rate_buckets[sender] = bucket
        if len(bucket) >= MAX_MESSAGES:
            logger.warning("Rate limit exceeded")
            return False
        _rate_buckets[sender].append(now)
        return True
