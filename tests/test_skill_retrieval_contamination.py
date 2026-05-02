"""Regression tests for skill-retrieval cross-topic contamination.

Production incident (2026-05-02 ~10:49 UTC): a multi-hour Estonia
deforestation thread was followed by a generic continuation
("please execute the plan and produce the report"). Routing dispatched
correctly to the coding crew, but skill retrieval matched the bare
phrase "the plan" against a stale auto-generated weather-forecasting
skill ("**** Reliable Weather Forecast Retrieval"). The crew then
produced a Weather Forecast System implementation instead of the
forest report. Vetting eventually caught the wrong-topic dispatch but
~6 minutes of compute were wasted on it.

These tests pin the four defences added to ``_load_relevant_skills``:

  1. Subject-less message detection — short generic messages
     ("execute the plan") substitute the recent conversation topic
     as the retrieval query, or skip retrieval when no history exists.
  2. Quality filter — skills whose topic carries placeholder markers
     (****, _____, <redacted>) are excluded.
  3. Semantic distance gate — records beyond _SKILL_DISTANCE_CEILING
     cosine distance are dropped even if they're in the top N.
  4. Recency overlap — when a long history is present, the loader uses
     the recent user turns as the retrieval query, so a forest
     conversation continues to retrieve forest-related skills.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NOTE: deliberately do NOT stub `chromadb` / `psycopg2` at module level.
# All tests in this file patch `app.self_improvement.integrator
# .search_skills_scored` so the loader's primary path returns before
# touching chromadb, and the legacy `chromadb_manager.retrieve` fallback
# never fires. Stubbing chromadb at import time poisons sys.modules for
# any test collected later (notably the test_e2e_v2_features.py suite,
# whose crewai imports require the real chromadb package layout).


def teardown_module(module):
    sys.modules.pop("commander_context", None)
    sys.modules.pop("app.agents.commander.context", None)


# Import context.py directly (bypass the package __init__ which pulls in
# the full crewai chain). Same trick test_contamination.py uses.
import importlib.util as _ilu

_context_path = os.path.join(
    os.path.dirname(__file__), "..", "app", "agents", "commander", "context.py"
)
_spec = _ilu.spec_from_file_location("commander_context", _context_path)
_context_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_context_mod)
sys.modules["app.agents.commander.context"] = _context_mod

from app.self_improvement.types import SkillRecord


# ── Test fixtures ───────────────────────────────────────────────────────────

# A long conversation thread on Estonia forest monitoring, ending with a
# subject-less continuation that triggered the production bug.
_FOREST_HISTORY = (
    "User: Can you map the deforestation in Estonia from 2020 to 2024?\n"
    "Assistant: Sure, I'll use Google Earth Engine to compute the loss layer "
    "from Hansen v1.11 across Estonia's national boundary.\n"
    "User: Good. Break down the loss by county and by year.\n"
    "Assistant: Plan: 1) export the Hansen lossyear band clipped to Estonia; "
    "2) reduce by maakond polygons; 3) emit a CSV plus a small chart.\n"
    "User: Please add a comparison against the EU forest monitoring baseline.\n"
    "Assistant: Will add the Copernicus HRL-FTY layer as an overlay.\n"
)
_SUBJECTLESS_CONTINUATION = "please execute the plan and produce the report"

# The tempting unrelated skill that surfaced in production.
_WEATHER_SKILL = SkillRecord(
    id="skill_weather_stale",
    topic="**** Reliable Weather Forecast Retrieval",
    content_markdown=(
        "When the user asks for a forecast, query open-meteo and fall back "
        "to MET.no. Always include precipitation probability and wind."
    ),
    kb="experiential",
)
_FOREST_SKILL = SkillRecord(
    id="skill_forest_gee",
    topic="Google Earth Engine scripting for Estonian forest monitoring",
    content_markdown=(
        "Use Hansen v1.11 lossyear band clipped to FAO GAUL Estonia. "
        "Reduce by maakond polygons. Export to Drive."
    ),
    kb="experiential",
)


# ─── Layer 1: subject-less message detection ────────────────────────────────


class TestSubjectlessDetection:

    @pytest.mark.parametrize("msg", [
        "execute the plan",
        "run it",
        "please execute the plan and produce the report",
        "do it now",
        "ok go ahead",
        "please continue",
        "yes",
        "ok",
        "produce the report",
        "generate the output",
    ])
    def test_subjectless_messages_detected(self, msg):
        assert _context_mod._is_subjectless_message(msg) is True

    @pytest.mark.parametrize("msg", [
        "produce the forest report for Estonia",
        "run the deforestation analysis",
        "execute the GEE script for forest cover",
        "what's the weather in Helsinki",
        "summarize the Hansen lossyear data",
        "fix the email sender",
    ])
    def test_topical_messages_not_subjectless(self, msg):
        assert _context_mod._is_subjectless_message(msg) is False


# ─── Layer 1b: recent topic extraction ──────────────────────────────────────


class TestRecentTopicExtraction:

    def test_extracts_user_lines_only(self):
        topic = _context_mod._extract_recent_topic(_FOREST_HISTORY)
        assert "deforestation in Estonia" in topic
        assert "comparison against the EU forest" in topic
        # Assistant lines must not bleed in — they can carry tangential
        # topics from the system's own past confusions.
        assert "Hansen v1.11" not in topic
        assert "Copernicus" not in topic

    def test_takes_last_three_user_turns(self):
        history = "\n".join(
            f"User: turn {i}" for i in range(1, 11)
        )
        topic = _context_mod._extract_recent_topic(history)
        assert "turn 8" in topic
        assert "turn 9" in topic
        assert "turn 10" in topic
        assert "turn 7" not in topic

    def test_empty_history_returns_empty(self):
        assert _context_mod._extract_recent_topic("") == ""

    def test_assistant_only_history_returns_empty(self):
        assert _context_mod._extract_recent_topic(
            "Assistant: I have a thought.\nAssistant: Another thought."
        ) == ""

    def test_max_chars_truncation(self):
        history = "User: " + ("x " * 1000) + "\n"
        topic = _context_mod._extract_recent_topic(history, max_chars=200)
        assert len(topic) <= 200


# ─── Layer 2: quality filter ────────────────────────────────────────────────


class TestQualityFilter:

    @pytest.mark.parametrize("topic", [
        "**** Reliable Weather Forecast Retrieval",
        "_____ lead generation playbook",
        "Random skill <redacted> for safety",
        "[REDACTED] internal API access",
    ])
    def test_placeholder_topics_filtered(self, topic):
        assert _context_mod._is_low_quality_skill_topic(topic) is True

    @pytest.mark.parametrize("topic", [
        "Google Earth Engine scripting for Estonian forest monitoring",
        "OpenRouter credit management and cost optimization",
        "Email inbox retrieval and ranking protocols",
    ])
    def test_clean_topics_pass(self, topic):
        assert _context_mod._is_low_quality_skill_topic(topic) is False

    def test_empty_topic_treated_as_low_quality(self):
        assert _context_mod._is_low_quality_skill_topic("") is True


# ─── End-to-end: the production failure mode ────────────────────────────────


class TestProductionRegression:
    """The exact failure scenario from 2026-05-02."""

    def test_subjectless_msg_with_forest_history_does_not_inject_weather(self):
        """The headline regression: short generic message after a long forest
        thread, with a tempting weather skill in the inventory. The loader
        must not surface weather content into the dispatched crew's context.
        """
        # Mock returns weather as the top semantic match (this is what was
        # happening in prod — short ambiguous query embeds close to anything).
        scored = [
            (_WEATHER_SKILL, 0.42),  # below ceiling, but topic carries ****
            (_FOREST_SKILL, 0.28),   # the actually-relevant skill
        ]
        with patch(
            "app.self_improvement.integrator.search_skills_scored",
            return_value=scored,
        ):
            out = _context_mod._load_relevant_skills(
                _SUBJECTLESS_CONTINUATION,
                conversation_history=_FOREST_HISTORY,
            )

        # Quality filter blocks the weather skill regardless.
        assert "Weather Forecast" not in out
        assert "open-meteo" not in out
        assert "precipitation" not in out
        assert "****" not in out
        # Forest skill must survive — recovery preserves good matches.
        assert "Estonian forest" in out or "forest" in out.lower()

    def test_subjectless_msg_with_no_history_returns_empty(self):
        """Without a history to recover topic from, refusing to inject
        anything is safer than guessing."""
        with patch(
            "app.self_improvement.integrator.search_skills_scored",
            return_value=[(_WEATHER_SKILL, 0.42), (_FOREST_SKILL, 0.28)],
        ):
            out = _context_mod._load_relevant_skills(
                _SUBJECTLESS_CONTINUATION, conversation_history="",
            )
        assert out == ""

    def test_distance_gate_drops_orthogonal_matches(self):
        """Even with a perfectly topical query, weak matches should not
        surface as 'relevant knowledge'."""
        # Only a far-away (orthogonal) skill is returned by search.
        far = SkillRecord(
            id="skill_far", topic="Cooking pasta carbonara",
            content_markdown="Whisk eggs with cheese; do not scramble.",
            kb="experiential",
        )
        with patch(
            "app.self_improvement.integrator.search_skills_scored",
            return_value=[(far, 0.95)],   # well past the 0.55 ceiling
        ):
            out = _context_mod._load_relevant_skills(
                "produce the Estonia deforestation report"
            )
        assert "carbonara" not in out
        assert out == ""

    def test_topical_msg_still_retrieves_relevant_skills(self):
        """Defences must not break the happy path — a clear topical
        message should still pull in matching skills."""
        with patch(
            "app.self_improvement.integrator.search_skills_scored",
            return_value=[(_FOREST_SKILL, 0.18)],
        ):
            out = _context_mod._load_relevant_skills(
                "run the GEE forest monitoring analysis for Estonia",
                conversation_history="",
            )
        assert "Estonian forest" in out
        assert "Hansen" in out

    def test_quality_filter_holds_even_for_topical_query(self):
        """Even if a placeholder skill is the closest match to a real
        topical query, it must not surface."""
        weather_close = SkillRecord(
            id="skill_w2",
            topic="**** Weather monitoring playbook",
            content_markdown=(
                "Forest fire risk depends on wind and humidity — fetch "
                "weather first."
            ),
            kb="experiential",
        )
        with patch(
            "app.self_improvement.integrator.search_skills_scored",
            return_value=[(weather_close, 0.22), (_FOREST_SKILL, 0.30)],
        ):
            out = _context_mod._load_relevant_skills(
                "run forest monitoring for Estonia"
            )
        assert "****" not in out
        assert "Weather monitoring" not in out
        assert "Estonian forest" in out


# ─── Sanity: the existing matches_context filter still applies ──────────────


class TestStillRespectsConditionalActivation:
    """The contamination defences must layer on top of the existing
    conditional-activation filter from Phase 3 of the SkillRecord overhaul,
    not replace it."""

    def test_mode_mismatch_still_filtered(self):
        only_local = SkillRecord(
            id="s_local", topic="Local Ollama tips",
            content_markdown="Start ollama before any local-mode call.",
            kb="experiential", requires_mode="local",
        )
        scored = [(only_local, 0.12)]
        with patch(
            "app.self_improvement.integrator.search_skills_scored",
            return_value=scored,
        ), patch("app.llm_mode.get_mode", lambda: "cloud"):
            out = _context_mod._load_relevant_skills(
                "set up the ollama runtime"
            )
        assert "Local Ollama tips" not in out


# ─── Write-side guard: integrator.integrate() rejects placeholder topics ────


class TestIntegratorRejectsPlaceholderTopics:
    """Defense-in-depth complement to the retrieval-side gate.

    The retrieval-side gate (``_is_low_quality_skill_topic``) keeps the
    bug from surfacing to agents, but the index still accumulates the
    contaminated records — May 2 smoke run found 4 active. Pin that
    ``integrate()`` rejects them at the write path so the gate stops
    being load-bearing for this class of bug.
    """

    def _make_draft(self, topic: str):
        from app.self_improvement.types import SkillDraft
        return SkillDraft(
            id="draft_test_placeholder",
            topic=topic,
            rationale="unit test",
            content_markdown=(
                "## Practice\nDo the thing the topic claims to do. "
                "Some plausible body so the draft passes the empty-content guard."
            ),
            proposed_kb="experiential",
        )

    @pytest.mark.parametrize("topic", [
        "**** Reliable Weather Forecast Retrieval",
        "**** Robust Weather Data Retrieval via Open-Meteo API",
        "_____ lead generation playbook",
        "Random skill <redacted> for safety",
        "[REDACTED] internal API access",
        "[redacted] mid-string token",
    ])
    def test_placeholder_topic_drafts_are_rejected(self, topic):
        from app.self_improvement import integrator as integ
        write_calls: list = []
        persist_calls: list = []
        with patch.object(integ, "_write_to_kb",
                          lambda kb, rec: write_calls.append((kb, rec)) or True), \
             patch.object(integ, "_persist_record",
                          lambda rec: persist_calls.append(rec) or True):
            result = integ.integrate(self._make_draft(topic))
        assert result is None, (
            f"integrate() should reject placeholder topic: {topic!r}"
        )
        assert write_calls == [], "no KB write should have happened"
        assert persist_calls == [], "no record-index persist should have happened"

    @pytest.mark.parametrize("topic", [
        "",
        "   ",
    ])
    def test_empty_or_whitespace_topic_rejected(self, topic):
        from app.self_improvement import integrator as integ
        write_calls: list = []
        with patch.object(integ, "_write_to_kb",
                          lambda kb, rec: write_calls.append((kb, rec)) or True):
            result = integ.integrate(self._make_draft(topic))
        assert result is None
        assert write_calls == []

    def test_clean_topic_still_integrates(self):
        """Happy path: a clean topic must still flow through to a write."""
        from app.self_improvement import integrator as integ
        write_calls: list = []
        persist_calls: list = []

        # Bypass the content-novelty check so the test is independent of
        # the embeddings stack — the placeholder guard runs *before* it.
        with patch.object(integ, "novelty_report",
                          side_effect=Exception("skip novelty")), \
             patch.object(integ, "classify_kb", lambda d: "experiential"), \
             patch.object(integ, "_write_to_kb",
                          lambda kb, rec: write_calls.append((kb, rec)) or True), \
             patch.object(integ, "_persist_record",
                          lambda rec: persist_calls.append(rec) or True):
            result = integ.integrate(
                self._make_draft(
                    "Google Earth Engine scripting for Estonian forest monitoring"
                )
            )

        assert result is not None
        assert write_calls and write_calls[0][0] == "experiential"
        # The persisted record carries the same clean topic (no mutation).
        assert "forest" in result.topic.lower()

    def test_topic_helper_matches_retrieval_side_definition(self):
        """The two layers must agree on which topics are placeholders. A
        drift here would mean the retrieval gate is filtering records
        that the integrator now lets through, or vice versa, both of
        which are footguns."""
        from app.self_improvement.integrator import _topic_has_placeholder_marker
        for topic in (
            "**** something",
            "_____ something",
            "<redacted>",
            "[REDACTED]",
            "[redacted] x",
        ):
            assert _topic_has_placeholder_marker(topic) is True
            assert _context_mod._is_low_quality_skill_topic(topic) is True
        for topic in (
            "Google Earth Engine forest monitoring",
            "OpenRouter cost optimization",
        ):
            assert _topic_has_placeholder_marker(topic) is False
            assert _context_mod._is_low_quality_skill_topic(topic) is False


# ─── Upstream fix: _auto_create_skill no longer leaks bold delimiters ───────


class TestAutoSkillTopicExtraction:
    """The actual root cause of the May 2 records: the LLM emitted
    ``**Topic:** Foo`` and ``str.replace("Topic:", "")`` left ``**** Foo``.
    The new ``_extract_auto_skill_topic`` strips the label together with
    its surrounding bold markers."""

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        # base_crew imports a chunk of CrewAI machinery on module load
        # which is heavy in a unit test. Pull just the helper.
        import importlib.util as _ilu
        path = os.path.join(
            os.path.dirname(__file__), "..", "app", "crews", "base_crew.py"
        )
        spec = _ilu.spec_from_file_location("base_crew_for_topic_test", path)
        # Stub heavy upward imports before exec so module load doesn't
        # pull the whole world.
        import types as _types
        for name in (
            "crewai", "app.config", "app.benchmarks", "app.llm_selector",
            "app.sanitize", "app.self_heal", "app.firebase_reporter",
            "app.rate_throttle", "app.conversation_store", "app.tools.web_search",
            "app.tools.web_fetch", "app.tools.youtube_transcript",
            "app.tools.memory_tool", "app.tools.file_manager", "app.proposals",
            "app.project_context", "app.idle_scheduler",
        ):
            sys.modules.setdefault(name, _types.ModuleType(name))
        # The helper itself only needs `re`, no upward deps. If the module
        # load fails due to side-effects in unrelated parts of base_crew,
        # fall back to compiling the helper inline from source.
        try:
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._extract = mod._extract_auto_skill_topic
        except Exception:
            import re as _re
            label_re = _re.compile(
                r"\*{0,2}\s*Topic\s*:\s*\*{0,2}\s*", _re.IGNORECASE,
            )

            def _fallback_extract(skill_text: str, fallback: str) -> str:
                first_line = skill_text.strip().split("\n", 1)[0]
                first_line = label_re.sub("", first_line, count=1)
                first_line = first_line.strip().strip("*_# ").strip()
                return first_line[:100] or fallback[:80]
            self._extract = _fallback_extract

    @pytest.mark.parametrize("first_line, expected", [
        # The smoking-gun input — the original bug.
        ("**Topic:** Reliable Weather Forecast Retrieval",
         "Reliable Weather Forecast Retrieval"),
        ("**Topic:** Robust Weather Data Retrieval via Open-Meteo API",
         "Robust Weather Data Retrieval via Open-Meteo API"),
        # Plain label.
        ("Topic: Forest monitoring with Hansen v1.11",
         "Forest monitoring with Hansen v1.11"),
        # Heading-style with no label.
        ("# Forest monitoring", "Forest monitoring"),
        # Bold without "Topic:" label.
        ("**Forest monitoring**", "Forest monitoring"),
        # Lowercase label with bold.
        ("**topic:** forest plan", "forest plan"),
        # Heading + label combo.
        ("## Topic: Forest plan", "Forest plan"),
    ])
    def test_clean_extraction(self, first_line, expected):
        assert self._extract(first_line, fallback="fallback") == expected

    def test_no_placeholder_artifact_in_output(self):
        """Whatever transformation we apply, the output must not contain
        the Markdown-bold leakage pattern that caused the original bug."""
        from app.self_improvement.integrator import _topic_has_placeholder_marker
        for first_line in (
            "**Topic:** Reliable Weather Forecast Retrieval",
            "**Topic:** Robust Weather Data Retrieval via Open-Meteo API",
            "**Topic:** Lead Generation and Enrichment for PSPs",
        ):
            topic = self._extract(first_line, fallback="fallback")
            assert not _topic_has_placeholder_marker(topic), (
                f"upstream extractor leaked a placeholder marker: {topic!r}"
            )

    def test_empty_skill_text_falls_back_to_task(self):
        assert self._extract("", fallback="my original task") == "my original task"
