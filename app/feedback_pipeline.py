"""
feedback_pipeline.py — Collects, classifies, and aggregates user feedback signals.

IMMUTABLE — this module is part of the infrastructure layer and must NOT be
modifiable by any agent or modification engine.  It is listed in
auto_deployer.py PROTECTED_FILES.

Feedback types:
  - Explicit positive:  👍 ❤️ 🎉 reactions on bot messages
  - Explicit negative:  👎 😡 😕 reactions on bot messages
  - Explicit correction: "No, I meant X" / "That's wrong" type messages
  - Explicit instruction: "Be more concise" / "Always do X" type messages
  - Implicit re-request: Same question rephrased within 5 minutes (Phase 6)
  - Implicit abandonment: User stops mid-conversation (Phase 6)
  - Implicit follow-up: User asks for clarification (Phase 6)

The classification prompt is defined as a constant in this module (immutable)
and cannot be changed by the modification engine.
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ── IMMUTABLE: Feedback classification prompt ─────────────────────────────
# This prompt is part of the Tier 3 (Immutable) layer.  It defines how
# user feedback is interpreted and classified.  Changes require a code
# deployment by the system operator.

FEEDBACK_CLASSIFICATION_PROMPT = """You are a feedback diagnosis engine for an AI agent system.
Given a user's feedback on an agent's output, classify the feedback into a structured signal.

Categories: accuracy, style, completeness, relevance, tool_choice, speed, safety
Severity: critical (factual error, safety issue), moderate (wrong approach), minor (preference, polish)
Target layer: adaptive (prompt/examples/style), protected (workflow/tools/roles)
Target parameter: system_prompt, few_shot_examples, style_params, knowledge_base, tool_defaults, workflow_graph, agent_roles, tool_permissions
Direction: A concise description of what should change
Confidence: 0.0-1.0 based on how clearly the feedback indicates the needed change

User's original request: {task_text}
Agent's response (summary): {response_summary}
User's feedback: {feedback_text}
Crew that handled the task: {crew_used}

Respond with a JSON object:
{{"category": "...", "severity": "...", "target_layer": "...", "target_parameter": "...", "target_role": "{target_role}", "direction": "...", "confidence": 0.0}}

Do not speculate beyond what the feedback explicitly or strongly implies.
If the feedback is ambiguous, set confidence below 0.3."""

# ── IMMUTABLE: Pattern aggregation thresholds ─────────────────────────────
PATTERN_TRIGGER_COUNT = 3           # events with same (category, target_parameter) in window
PATTERN_TRIGGER_WINDOW_DAYS = 7     # window for pattern aggregation
CRITICAL_IMMEDIATE_TRIGGER = True   # critical severity → immediate trigger
CONFLICTING_FLAG = True             # opposite directions → flag for human review
DECLINING_TREND_THRESHOLD = 5       # interactions with declining ratings → diagnostic sweep

# ── Emoji → feedback type mapping ─────────────────────────────────────────
POSITIVE_EMOJIS = {"👍", "❤️", "🎉", "✅", "💯", "🙏", "👏", "🔥"}
NEGATIVE_EMOJIS = {"👎", "😡", "😕", "❌", "😤", "🤦", "💩"}

# ── Correction detection patterns ─────────────────────────────────────────
CORRECTION_INDICATORS = [
    r"(?i)^no[,.]?\s",
    r"(?i)that'?s?\s+(wrong|incorrect|not right|not what)",
    r"(?i)i\s+(meant|mean|said|asked)\s",
    r"(?i)^actually[,.]?\s",
    r"(?i)^wrong[,.]?\s",
    r"(?i)please?\s+(don'?t|stop|never)\s",
    r"(?i)^not\s+what\s+i\s+(wanted|asked|meant)",
]

INSTRUCTION_INDICATORS = [
    r"(?i)(always|never|from\s+now\s+on|in\s+the\s+future)\s",
    r"(?i)^be\s+more\s+",
    r"(?i)^be\s+less\s+",
    r"(?i)^(shorter|longer|more\s+concise|more\s+detailed)",
    r"(?i)^(remember|keep\s+in\s+mind)\s+that\s",
]


class FeedbackPipeline:
    """Collects, classifies, and aggregates user feedback signals."""

    def __init__(self, db_url: str):
        """Connect to PostgreSQL feedback schema."""
        self._db_url = db_url
        self._engine = None
        self._initialized = False

    def _get_engine(self):
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(self._db_url, pool_size=3, max_overflow=2)
                self._initialized = True
            except Exception:
                logger.warning("feedback_pipeline: PostgreSQL unavailable", exc_info=True)
        return self._engine

    def _execute(self, query: str, params: dict = None) -> list:
        """Execute a SQL query and return results."""
        engine = self._get_engine()
        if not engine:
            return []
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                if result.returns_rows:
                    return [dict(row._mapping) for row in result]
                conn.commit()
                return []
        except Exception:
            logger.warning("feedback_pipeline: query failed", exc_info=True)
            return []

    # ── Response metadata recording ───────────────────────────────────────

    def record_response_metadata(self, msg_timestamp: int, sender_id: str,
                                  task_text: str, response_text: str,
                                  crew_used: str, prompt_versions: dict,
                                  model_used: str, task_id: int = 0) -> None:
        """Store metadata about every bot response for later feedback correlation."""
        self._execute(
            """INSERT INTO feedback.response_metadata
               (msg_timestamp, sender_id, task_text, response_text,
                crew_used, prompt_versions, model_used, task_id)
               VALUES (:ts, :sender, :task, :response, :crew, :versions, :model, :task_id)
               ON CONFLICT (msg_timestamp) DO NOTHING""",
            {
                "ts": msg_timestamp,
                "sender": sender_id,
                "task": (task_text or "")[:2000],
                "response": (response_text or "")[:2000],
                "crew": crew_used or "",
                "versions": json.dumps(prompt_versions or {}),
                "model": model_used or "",
                "task_id": task_id,
            }
        )

    def _lookup_response_metadata(self, target_timestamp: int) -> dict | None:
        """Look up response metadata by Signal message timestamp."""
        rows = self._execute(
            "SELECT * FROM feedback.response_metadata WHERE msg_timestamp = :ts",
            {"ts": target_timestamp}
        )
        return rows[0] if rows else None

    # ── Explicit feedback: emoji reactions ─────────────────────────────────

    def process_reaction(self, sender_id: str, emoji: str,
                         target_timestamp: int, is_remove: bool = False) -> dict | None:
        """Process a Signal emoji reaction on a bot message.

        Looks up which response the reaction targets, creates a feedback event.
        Returns the event dict or None if the reaction is irrelevant.
        """
        if is_remove:
            return None  # Reaction removed — ignore for now

        # Determine feedback type from emoji
        if emoji in POSITIVE_EMOJIS:
            feedback_type = "explicit_positive"
        elif emoji in NEGATIVE_EMOJIS:
            feedback_type = "explicit_negative"
        else:
            return None  # Neutral or unknown emoji

        # Look up what response this reaction targets
        metadata = self._lookup_response_metadata(target_timestamp)

        event = {
            "id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "feedback_type": feedback_type,
            "raw_signal": emoji,
            "original_task": metadata.get("task_text", "") if metadata else "",
            "original_response": metadata.get("response_text", "") if metadata else "",
            "crew_used": metadata.get("crew_used", "") if metadata else "",
            "prompt_version": 0,
            "model_used": metadata.get("model_used", "") if metadata else "",
        }

        # For negative reactions, classify the feedback
        if feedback_type == "explicit_negative" and metadata:
            diagnosis = self._classify_negative_reaction(metadata)
            if diagnosis:
                event.update(diagnosis)
            else:
                # Default diagnosis for unclassified negative reactions
                event["category"] = "accuracy"
                event["severity"] = "moderate"
                event["target_layer"] = "adaptive"
                event["target_parameter"] = "system_prompt"
                event["target_role"] = metadata.get("crew_used", "commander")
                event["direction"] = "User reacted negatively — review response quality"
                event["confidence"] = 0.4
        elif feedback_type == "explicit_positive" and metadata:
            event["category"] = "accuracy"
            event["severity"] = "minor"
            event["target_layer"] = "adaptive"
            event["target_parameter"] = "system_prompt"
            event["target_role"] = metadata.get("crew_used", "commander")
            event["direction"] = "User confirmed response quality — reinforce current approach"
            event["confidence"] = 0.6

        # Store the event
        self._store_event(event)
        logger.info(f"feedback_pipeline: {feedback_type} reaction ({emoji}) on message ts={target_timestamp}")
        return event

    def _classify_negative_reaction(self, metadata: dict) -> dict | None:
        """Use LLM to classify why a user reacted negatively."""
        try:
            from app.llm_factory import create_cheap_vetting_llm
            llm = create_cheap_vetting_llm()
            prompt = FEEDBACK_CLASSIFICATION_PROMPT.format(
                task_text=(metadata.get("task_text") or "")[:500],
                response_summary=(metadata.get("response_text") or "")[:500],
                feedback_text="User reacted with a negative emoji (👎) to this response",
                crew_used=metadata.get("crew_used", "unknown"),
                target_role=metadata.get("crew_used", "commander"),
            )
            raw = str(llm.call(prompt)).strip()
            # Extract JSON from response
            if "{" in raw:
                json_str = raw[raw.index("{"):raw.rindex("}") + 1]
                return json.loads(json_str)
        except Exception:
            logger.debug("feedback_pipeline: classification failed", exc_info=True)
        return None

    # ── Explicit feedback: corrections and instructions ────────────────────

    def process_correction(self, sender_id: str, text: str,
                           recent_task: str, recent_response: str,
                           crew_used: str, prompt_version: int,
                           model_used: str = "") -> dict | None:
        """Classify a natural language correction or instruction.

        Returns structured diagnosis or None if not a correction/instruction.
        """
        import re

        # Check if text matches correction or instruction patterns
        is_correction = any(re.search(p, text) for p in CORRECTION_INDICATORS)
        is_instruction = any(re.search(p, text) for p in INSTRUCTION_INDICATORS)

        if not is_correction and not is_instruction:
            return None

        feedback_type = "explicit_correction" if is_correction else "explicit_instruction"

        # Classify with LLM
        try:
            from app.llm_factory import create_cheap_vetting_llm
            llm = create_cheap_vetting_llm()
            prompt = FEEDBACK_CLASSIFICATION_PROMPT.format(
                task_text=(recent_task or "")[:500],
                response_summary=(recent_response or "")[:500],
                feedback_text=text[:500],
                crew_used=crew_used or "unknown",
                target_role=crew_used or "commander",
            )
            raw = str(llm.call(prompt)).strip()
            diagnosis = {}
            if "{" in raw:
                json_str = raw[raw.index("{"):raw.rindex("}") + 1]
                diagnosis = json.loads(json_str)
        except Exception:
            logger.debug("feedback_pipeline: correction classification failed", exc_info=True)
            diagnosis = {}

        event = {
            "id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "feedback_type": feedback_type,
            "raw_signal": text[:500],
            "original_task": (recent_task or "")[:2000],
            "original_response": (recent_response or "")[:2000],
            "crew_used": crew_used or "",
            "prompt_version": prompt_version,
            "model_used": model_used,
            "category": diagnosis.get("category", "style"),
            "severity": diagnosis.get("severity", "moderate"),
            "target_layer": diagnosis.get("target_layer", "adaptive"),
            "target_parameter": diagnosis.get("target_parameter", "system_prompt"),
            "target_role": diagnosis.get("target_role", crew_used or "commander"),
            "direction": diagnosis.get("direction", f"User correction: {text[:200]}"),
            "confidence": diagnosis.get("confidence", 0.6),
        }

        self._store_event(event)
        logger.info(f"feedback_pipeline: {feedback_type} from user — {event['direction'][:80]}")
        return event

    # ── Event storage ─────────────────────────────────────────────────────

    def _store_event(self, event: dict) -> None:
        """Store a feedback event in PostgreSQL."""
        self._execute(
            """INSERT INTO feedback.events
               (id, sender_id, feedback_type, raw_signal,
                category, severity, target_layer, target_parameter, target_role,
                direction, confidence,
                original_task, original_response, crew_used, prompt_version, model_used)
               VALUES (:id, :sender_id, :feedback_type, :raw_signal,
                       :category, :severity, :target_layer, :target_parameter, :target_role,
                       :direction, :confidence,
                       :original_task, :original_response, :crew_used, :prompt_version, :model_used)""",
            event
        )

    # ── Pattern aggregation ───────────────────────────────────────────────

    def aggregate_patterns(self) -> list[dict]:
        """Group unprocessed feedback events into actionable patterns.

        Rules (IMMUTABLE):
        - 3+ events with same (category, target_parameter) in 7 days → trigger
        - 1 critical-severity event → immediate trigger
        - Conflicting directions → flag for human review
        Returns list of newly triggered patterns.
        """
        triggered = []

        # 1. Immediate trigger for critical events
        critical_events = self._execute(
            """SELECT * FROM feedback.events
               WHERE processed = FALSE AND severity = 'critical'
               ORDER BY timestamp""",
        )
        for evt in critical_events:
            pattern = self._create_or_update_pattern(evt, immediate=True)
            if pattern:
                triggered.append(pattern)
            self._execute(
                "UPDATE feedback.events SET processed = TRUE, pattern_id = :pid WHERE id = :eid",
                {"pid": pattern["id"] if pattern else None, "eid": evt["id"]}
            )

        # 2. Aggregate remaining events by (category, target_parameter, target_role)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=PATTERN_TRIGGER_WINDOW_DAYS)).isoformat()
        groups = self._execute(
            """SELECT category, target_parameter, target_role,
                      COUNT(*) as cnt,
                      MIN(timestamp) as first_seen,
                      MAX(timestamp) as last_seen,
                      array_agg(direction) as directions,
                      array_agg(id) as event_ids
               FROM feedback.events
               WHERE processed = FALSE
                 AND severity != 'critical'
                 AND timestamp >= :cutoff
                 AND feedback_type LIKE 'explicit_%'
               GROUP BY category, target_parameter, target_role
               HAVING COUNT(*) >= :threshold""",
            {"cutoff": cutoff, "threshold": PATTERN_TRIGGER_COUNT}
        )

        for group in groups:
            # Check for conflicting directions
            directions = group.get("directions", [])
            if self._has_conflicting_directions(directions):
                pattern = self._create_pattern(
                    group, status="conflicting",
                    direction="Conflicting feedback — requires human review"
                )
            else:
                # Synthesize direction from constituent events
                direction = self._synthesize_direction(directions)
                pattern = self._create_pattern(group, status="triggered", direction=direction)

            if pattern:
                triggered.append(pattern)
                # Mark events as processed
                event_ids = group.get("event_ids", [])
                for eid in event_ids:
                    self._execute(
                        "UPDATE feedback.events SET processed = TRUE, pattern_id = :pid WHERE id = :eid",
                        {"pid": pattern["id"], "eid": eid}
                    )

        if triggered:
            logger.info(f"feedback_pipeline: {len(triggered)} patterns triggered")
        return triggered

    def _create_or_update_pattern(self, event: dict, immediate: bool = False) -> dict | None:
        """Create or update a pattern from a single event."""
        pattern_id = str(uuid.uuid4())
        status = "triggered" if immediate else "pending"
        now = datetime.now(timezone.utc).isoformat()

        self._execute(
            """INSERT INTO feedback.patterns
               (id, category, target_parameter, target_role, direction,
                event_count, first_seen, last_seen, status, triggered_at)
               VALUES (:id, :cat, :param, :role, :dir, 1, :now, :now, :status, :triggered)""",
            {
                "id": pattern_id,
                "cat": event.get("category", ""),
                "param": event.get("target_parameter", ""),
                "role": event.get("target_role", ""),
                "dir": event.get("direction", ""),
                "now": now,
                "status": status,
                "triggered": now if immediate else None,
            }
        )
        return {
            "id": pattern_id,
            "category": event.get("category"),
            "target_parameter": event.get("target_parameter"),
            "target_role": event.get("target_role"),
            "direction": event.get("direction"),
            "event_count": 1,
            "status": status,
        }

    def _create_pattern(self, group: dict, status: str, direction: str) -> dict | None:
        """Create a pattern from an aggregated group."""
        pattern_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self._execute(
            """INSERT INTO feedback.patterns
               (id, category, target_parameter, target_role, direction,
                event_count, first_seen, last_seen, status, triggered_at)
               VALUES (:id, :cat, :param, :role, :dir, :cnt, :first, :last, :status, :triggered)""",
            {
                "id": pattern_id,
                "cat": group.get("category", ""),
                "param": group.get("target_parameter", ""),
                "role": group.get("target_role", ""),
                "dir": direction,
                "cnt": group.get("cnt", 0),
                "first": group.get("first_seen"),
                "last": group.get("last_seen"),
                "status": status,
                "triggered": now if status == "triggered" else None,
            }
        )
        return {
            "id": pattern_id,
            "category": group.get("category"),
            "target_parameter": group.get("target_parameter"),
            "target_role": group.get("target_role"),
            "direction": direction,
            "event_count": group.get("cnt", 0),
            "status": status,
        }

    def _has_conflicting_directions(self, directions: list) -> bool:
        """Detect if feedback directions conflict (e.g., 'be more concise' vs 'be more detailed')."""
        if not directions or len(directions) < 2:
            return False
        # Simple heuristic: check for opposing keywords
        opposites = [
            ("more", "less"), ("concise", "detailed"), ("shorter", "longer"),
            ("formal", "casual"), ("verbose", "brief"), ("add", "remove"),
        ]
        lower_dirs = [d.lower() if isinstance(d, str) else "" for d in directions]
        for word_a, word_b in opposites:
            has_a = any(word_a in d for d in lower_dirs)
            has_b = any(word_b in d for d in lower_dirs)
            if has_a and has_b:
                return True
        return False

    def _synthesize_direction(self, directions: list) -> str:
        """Synthesize a single direction from multiple feedback events."""
        valid = [d for d in directions if isinstance(d, str) and d.strip()]
        if not valid:
            return "Multiple feedback events — direction unclear"
        if len(valid) == 1:
            return valid[0]
        # Take the most common direction (simple frequency)
        from collections import Counter
        counter = Counter(valid)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # ── Query methods for modification engine ──────────────────────────────

    def get_triggered_patterns(self) -> list[dict]:
        """Return patterns with status='triggered' for the modification engine."""
        return self._execute(
            """SELECT * FROM feedback.patterns
               WHERE status = 'triggered'
               ORDER BY last_seen DESC"""
        )

    def mark_pattern_resolved(self, pattern_id: str, modification_id: str) -> None:
        """Link a pattern to its modification attempt and mark it resolved."""
        self._execute(
            """UPDATE feedback.patterns
               SET status = 'resolved', resolved_at = now(), modification_id = :mid
               WHERE id = :pid""",
            {"pid": pattern_id, "mid": modification_id}
        )

    def get_recent_feedback_for_role(self, role: str, n: int = 10) -> list[dict]:
        """Get recent feedback events targeting a specific role."""
        return self._execute(
            """SELECT * FROM feedback.events
               WHERE target_role = :role
               ORDER BY timestamp DESC
               LIMIT :n""",
            {"role": role, "n": n}
        )

    def get_feedback_stats(self) -> dict:
        """Get summary statistics for dashboard/digest."""
        rows = self._execute(
            """SELECT feedback_type, COUNT(*) as cnt
               FROM feedback.events
               GROUP BY feedback_type"""
        )
        stats = {r["feedback_type"]: r["cnt"] for r in rows}

        pattern_rows = self._execute(
            """SELECT status, COUNT(*) as cnt
               FROM feedback.patterns
               GROUP BY status"""
        )
        stats["patterns"] = {r["status"]: r["cnt"] for r in pattern_rows}
        return stats
