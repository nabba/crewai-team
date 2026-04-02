import logging
import time as _time
from crewai import Task, Crew, Process
from app.agents.media_analyst import create_media_analyst
from app.sanitize import wrap_user_input
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.rate_throttle import start_request_tracking, stop_request_tracking
from app.self_heal import diagnose_and_fix
from app.memory.belief_state import update_belief
from app.benchmarks import record_metric
from app.llm_selector import difficulty_to_tier

logger = logging.getLogger(__name__)

# Static task template for media analysis.
MEDIA_TASK_TEMPLATE = """\
Analyze the following media content:

{user_input}

INSTRUCTIONS:
1. Identify the media type (video, image, audio, document, photo).
2. For YouTube videos: extract the transcript first, then summarize key points.
3. For images/photos: describe what you see, extract any text or data.
4. For audio/podcasts: summarize the discussion, note speakers and key quotes.
5. For documents: extract key data, tables, and findings.

Provide a structured analysis with:
- Summary of content
- Key findings or data points
- Notable details
- Source quality assessment

Keep the output concise — the user reads on a phone via Signal.

After completing analysis, use self_report to assess your confidence.
"""

# Concise template for simple media tasks (difficulty 1-3).
SIMPLE_MEDIA_TEMPLATE = """\
Analyze this media content and provide a concise summary:

{user_input}

Return a brief, direct summary in 2-5 sentences. Include key facts or data extracted.
Do NOT write a full report. Keep it phone-readable.
"""


class MediaCrew:
    def run(self, task_description: str, parent_task_id: str = None, difficulty: int = 5) -> str:
        """Run media analysis crew on the given task."""
        _start = _time.monotonic()
        from app.conversation_store import estimate_eta
        task_id = crew_started(
            "media", f"Media: {task_description[:100]}",
            eta_seconds=estimate_eta("media"), parent_task_id=parent_task_id,
        )
        start_request_tracking(task_id)
        update_belief("media_analyst", "working", current_task=task_description[:100])
        from app.llm_mode import get_mode
        force_tier = difficulty_to_tier(difficulty, get_mode())
        analyst = create_media_analyst(force_tier=force_tier)

        template = SIMPLE_MEDIA_TEMPLATE if difficulty <= 3 else MEDIA_TASK_TEMPLATE
        task = Task(
            description=template.format(user_input=wrap_user_input(task_description)),
            expected_output="Structured media analysis with key findings.",
            agent=analyst,
        )

        crew = Crew(
            agents=[analyst],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            result = crew.kickoff()
            result_str = str(result)

            tracker = stop_request_tracking()
            _tokens = tracker.total_tokens if tracker else 0
            _model = ", ".join(sorted(tracker.models_used)) if tracker and tracker.models_used else ""
            _cost = tracker.total_cost_usd if tracker else 0.0
            update_belief("media_analyst", "completed", current_task=task_description[:100])
            record_metric("task_completion_time", _time.monotonic() - _start, {"crew": "media"})
            crew_completed("media", task_id, result_str[:2000],
                           tokens_used=_tokens, model=_model, cost_usd=_cost)
            return result_str
        except Exception as exc:
            stop_request_tracking()
            update_belief("media_analyst", "failed", current_task=task_description[:100])
            crew_failed("media", task_id, str(exc)[:200])
            diagnose_and_fix("media", task_description, exc)
            raise
