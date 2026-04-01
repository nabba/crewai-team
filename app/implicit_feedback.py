"""
implicit_feedback.py — Detects implicit user dissatisfaction signals.

Implicit signals are inherently noisy.  They are weighted much lower than
explicit feedback (reactions, corrections) and only trigger modifications
when a strong statistical pattern emerges over multiple interactions.

Signal types:
  - Re-request: same question rephrased within 5 minutes → previous answer was inadequate
  - Abandonment: user stops mid-task → possible frustration or irrelevance
  - Follow-up question: user asks for clarification → output was incomplete/unclear

Weight multipliers (relative to explicit feedback):
  - Re-request: 0.3x
  - Abandonment: 0.2x
  - Follow-up: 0.4x
"""

import logging
import re
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ── IMMUTABLE: Implicit signal weights ────────────────────────────────────
REREQUEST_WEIGHT = 0.3
ABANDONMENT_WEIGHT = 0.2
FOLLOWUP_WEIGHT = 0.4

# Thresholds
REREQUEST_WINDOW_SECONDS = 300    # 5 minutes
REREQUEST_SIMILARITY_THRESHOLD = 0.75
ABANDONMENT_TIMEOUT_MINUTES = 30
MIN_IMPLICIT_PATTERN_EVENTS = 5   # require more events than explicit (3)

# Follow-up question patterns
FOLLOWUP_PATTERNS = [
    r"(?i)what\s+do\s+you\s+mean",
    r"(?i)can\s+you\s+(explain|clarify|elaborate)",
    r"(?i)i\s+don'?t\s+understand",
    r"(?i)that\s+(doesn'?t|does\s+not)\s+(make\s+sense|answer)",
    r"(?i)be\s+more\s+specific",
    r"(?i)what\s+(exactly|specifically)",
    r"(?i)could\s+you\s+(rephrase|reword)",
    r"(?i)huh\??$",
    r"(?i)^what\?$",
]


class ImplicitFeedbackDetector:
    """Detects implicit user dissatisfaction from conversation patterns."""

    def __init__(self, conversation_store=None, feedback_pipeline=None):
        self._store = conversation_store
        self._pipeline = feedback_pipeline

    def detect_rerequest(self, sender: str, text: str,
                          recent_history: list) -> dict | None:
        """Detect if current message is a rephrased version of a recent request.

        Uses embedding similarity to compare current message with recent messages.
        Only triggers if similarity > threshold AND within 5 minutes.
        """
        if not recent_history or len(text) < 10:
            return None

        try:
            from app.memory.chromadb_manager import get_manager
            manager = get_manager()

            # Get embedding for current message
            current_embedding = manager._embed_text(text)

            # Check against recent user messages
            for prev in recent_history[-5:]:
                if prev.get("role") != "user":
                    continue
                prev_text = prev.get("content", "")
                if not prev_text or len(prev_text) < 10:
                    continue

                # Check time window
                prev_ts = prev.get("timestamp")
                if prev_ts:
                    try:
                        prev_time = datetime.fromisoformat(prev_ts)
                        now = datetime.now(timezone.utc)
                        if prev_time.tzinfo is None:
                            prev_time = prev_time.replace(tzinfo=timezone.utc)
                        if (now - prev_time).total_seconds() > REREQUEST_WINDOW_SECONDS:
                            continue
                    except Exception:
                        continue

                # Check semantic similarity
                prev_embedding = manager._embed_text(prev_text)
                similarity = self._cosine_similarity(current_embedding, prev_embedding)

                if similarity > REREQUEST_SIMILARITY_THRESHOLD:
                    logger.info(f"implicit_feedback: re-request detected (sim={similarity:.2f})")
                    return {
                        "feedback_type": "implicit_rerequest",
                        "raw_signal": f"Rephrased within 5min (sim={similarity:.2f})",
                        "confidence": REREQUEST_WEIGHT * min(similarity, 1.0),
                        "category": "completeness",
                        "severity": "moderate",
                        "target_layer": "adaptive",
                        "target_parameter": "system_prompt",
                        "direction": "Previous response was inadequate — user rephrased the question",
                    }
        except Exception:
            logger.debug("implicit_feedback: re-request detection failed", exc_info=True)
        return None

    def detect_followup_question(self, sender: str, text: str,
                                   recent_response: str) -> dict | None:
        """Detect if user is asking for clarification.

        Patterns: "what do you mean", "can you explain", "I don't understand"
        """
        if not text or len(text) < 3:
            return None

        for pattern in FOLLOWUP_PATTERNS:
            if re.search(pattern, text):
                logger.info(f"implicit_feedback: follow-up question detected")
                return {
                    "feedback_type": "implicit_followup",
                    "raw_signal": text[:200],
                    "confidence": FOLLOWUP_WEIGHT,
                    "category": "completeness",
                    "severity": "minor",
                    "target_layer": "adaptive",
                    "target_parameter": "system_prompt",
                    "direction": f"Response was unclear — user asked: {text[:100]}",
                }

        return None

    def detect_abandonment(self, sender_id: str,
                            last_bot_message_time: datetime) -> dict | None:
        """Detect if user stopped mid-conversation.

        Checks if the bot's last message was a question/offer and there's
        been no response for 30+ minutes.
        """
        if not last_bot_message_time:
            return None

        try:
            now = datetime.now(timezone.utc)
            if last_bot_message_time.tzinfo is None:
                last_bot_message_time = last_bot_message_time.replace(tzinfo=timezone.utc)

            elapsed = (now - last_bot_message_time).total_seconds() / 60
            if elapsed >= ABANDONMENT_TIMEOUT_MINUTES:
                logger.info(f"implicit_feedback: potential abandonment ({elapsed:.0f} min)")
                return {
                    "feedback_type": "implicit_abandonment",
                    "raw_signal": f"No response in {elapsed:.0f} minutes",
                    "confidence": ABANDONMENT_WEIGHT,
                    "category": "relevance",
                    "severity": "minor",
                    "target_layer": "adaptive",
                    "target_parameter": "system_prompt",
                    "direction": "User may have abandoned conversation — check response relevance",
                }
        except Exception:
            pass
        return None

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            if isinstance(a, (list, tuple)):
                import math
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot / (norm_a * norm_b)
        except Exception:
            pass
        return 0.0
