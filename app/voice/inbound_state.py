"""
voice.inbound_state — short-lived "last inbound was voice" cache.

Maps a Signal sender to the timestamp of their most recent voice-note
inbound. The reply path checks this cache to decide whether to TTS the
outgoing message. The cache TTL is bounded so that a long pause followed
by a text message doesn't get answered with audio.

Pure in-memory; no persistence required — the worst-case effect of losing
state on restart is one missed voice reply, which the user can always ask
for again.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

# A voice inbound counts for at most this many seconds — beyond that, the
# user is no longer "in voice mode" and replies go back to text.
_VOICE_TTL_SECONDS = 5 * 60

# Cap on the number of senders we track. The Signal owner check upstream
# means in practice this map only ever has one entry, but we still cap to
# defend against a future multi-recipient deployment.
_MAX_SENDERS = 32

_lock = threading.Lock()
_state: "OrderedDict[str, float]" = OrderedDict()


def mark_voice_inbound(sender: str) -> None:
    """Record that ``sender`` just sent a voice note."""
    if not sender:
        return
    now = time.time()
    with _lock:
        _state[sender] = now
        _state.move_to_end(sender)
        while len(_state) > _MAX_SENDERS:
            _state.popitem(last=False)


def is_voice_active(sender: str) -> bool:
    """True if the sender's most recent voice inbound is still inside the TTL."""
    if not sender:
        return False
    now = time.time()
    with _lock:
        ts = _state.get(sender)
        if ts is None:
            return False
        if now - ts > _VOICE_TTL_SECONDS:
            del _state[sender]
            return False
        return True


def clear(sender: str) -> None:
    """Forget any voice flag for the sender (used after sending a voice reply
    so the next reply lands as text unless the user sends another voice note)."""
    if not sender:
        return
    with _lock:
        _state.pop(sender, None)
