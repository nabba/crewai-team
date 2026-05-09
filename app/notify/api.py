"""
notify.api — fan-out delivery + scheduled-job decorator.

Single fire-and-forget call:

    notify("Self-improvement", "✓ done in 42s", url="/cp/ops")

Sends to:
  - Signal direct message to the configured owner number
  - All registered Web Push subscriptions (silently no-op if VAPID off)

Both channels swallow errors internally so the caller's completion path
isn't blocked by a transient delivery problem. Failures land in the gateway
log and the ``notify_failure`` audit event for diagnosis.

The decorator below is the primary surface: any scheduled job, idle worker,
or proactive scanner can be wrapped at registration time without touching
the function's body.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Default URL the Web Push notification deep-links to.
_DEFAULT_DEEP_LINK = "/cp/ops"


def notify(
    title: str,
    body: str = "",
    *,
    url: str = _DEFAULT_DEEP_LINK,
    tag: str = "andrusai",
    signal: bool = True,
    web_push: bool = True,
    metadata: Optional[dict] = None,
) -> dict[str, Any]:
    """Fire-and-forget completion ping. Returns a small delivery summary
    so callers (and tests) can assert what landed where.

    ``metadata`` is optional and used by the feedback-router (Phase B
    #3, 2026-05-09): when set, the Signal send timestamp is recorded
    in ``workspace/companion/notify_meta.jsonl`` against the metadata
    so a later reaction on the message can be correlated back to the
    skill / recipe / task that produced this ping. Typical keys::

        {"skill_id": "...", "recipe_id": "...",
         "task_id": "...", "idea_id": "...",
         "workspace_id": "..."}
    """
    delivered: dict[str, Any] = {"signal": False, "web_push_count": 0, "signal_ts": None}

    if signal:
        ts = _send_signal_with_ts(title, body)
        delivered["signal"] = ts is not None and ts > 0
        delivered["signal_ts"] = ts
        if metadata and ts:
            try:
                from app.companion.notify_meta import record as _record_meta
                _record_meta(ts, metadata)
            except Exception:
                logger.debug("notify: meta record failed", exc_info=True)
    if web_push:
        delivered["web_push_count"] = _send_web_push(title, body, url=url, tag=tag)

    return delivered


def notify_on_complete(
    label: Optional[str] = None,
    *,
    notify_on_failure_only: bool = False,
    url: str = _DEFAULT_DEEP_LINK,
    silent: bool = False,
) -> Callable[[F], F]:
    """Decorator: ping Signal + Web Push when the wrapped fn finishes.

    Args:
        label: Short job name shown in the notification ("Self-improvement",
            "Workspace sync"). Defaults to the function's qualname.
        notify_on_failure_only: When True, skip the success ping — useful
            for high-frequency jobs (heartbeat, hourly sync) where success
            spam would be annoying. Failures still notify.
        url: Deep link the notification opens in the PWA.
        silent: Master kill switch — disable notifications entirely on
            this job (still runs the function as normal).
    """

    def deco(fn: F) -> F:
        job_label = label or fn.__qualname__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                start = time.monotonic()
                error: Optional[BaseException] = None
                try:
                    return await fn(*args, **kwargs)
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    if not silent:
                        _emit_completion(
                            job_label, error, time.monotonic() - start,
                            notify_on_failure_only=notify_on_failure_only,
                            url=url,
                        )

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            error: Optional[BaseException] = None
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                if not silent:
                    _emit_completion(
                        job_label, error, time.monotonic() - start,
                        notify_on_failure_only=notify_on_failure_only,
                        url=url,
                    )

        return wrapper  # type: ignore[return-value]

    return deco


# ── Internals ─────────────────────────────────────────────────────────────

def _emit_completion(
    label: str,
    error: Optional[BaseException],
    elapsed_s: float,
    *,
    notify_on_failure_only: bool,
    url: str,
) -> None:
    """Build the one-line message and dispatch."""
    if error is None:
        if notify_on_failure_only:
            return  # success suppressed for this job
        title = f"{label}"
        body = f"✓ done in {_human_duration(elapsed_s)}"
    else:
        # Skip silent SystemExit / KeyboardInterrupt — the operator
        # initiated those, no need to alert.
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return
        title = f"{label}"
        body = (
            f"✗ failed: {type(error).__name__}: "
            f"{str(error)[:120]} (after {_human_duration(elapsed_s)})"
        )
    try:
        notify(title, body, url=url)
    except Exception:
        # Never propagate notification failures back to the wrapped fn.
        logger.debug("notify_on_complete: dispatch failed", exc_info=True)


def _human_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{int(seconds * 1000)}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _send_signal(title: str, body: str) -> bool:
    """Send a single short message; return True/False. Kept for back-compat."""
    ts = _send_signal_with_ts(title, body)
    return ts is not None and ts > 0


def _send_signal_with_ts(title: str, body: str) -> Optional[int]:
    """Send a single short message; return the Signal-cli timestamp.

    None on every kind of failure (no recipient, signal-cli down, etc.).
    The timestamp is what reactions later target — Phase B #3 records
    it against caller-supplied metadata for the feedback-router.
    """
    try:
        from app.config import get_settings
        from app.signal_client import send_message_blocking
    except Exception:
        return None
    s = get_settings()
    recipient = (s.signal_owner_number or "").strip()
    if not recipient:
        return None
    text = title if not body else f"{title}\n{body}"
    try:
        ts = send_message_blocking(recipient, text)
        return ts if ts and ts > 0 else None
    except Exception:
        logger.debug("notify._send_signal: send failed", exc_info=True)
        return None


def _send_web_push(title: str, body: str, *, url: str, tag: str) -> int:
    """Fan-out to every registered Web Push subscription. Returns count delivered."""
    try:
        from app.web_push import send_to_all
        return int(send_to_all(title=title, body=body, url=url, tag=tag) or 0)
    except Exception:
        logger.debug("notify._send_web_push: dispatch failed", exc_info=True)
        return 0


# Re-export for test convenience without exposing the helper as public.
def _is_async(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or asyncio.iscoroutinefunction(fn)
