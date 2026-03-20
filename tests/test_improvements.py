"""Tests for Tier 1+2 improvements.

Tests verify:
  - safe_json_parse handles fences, errors, size limits
  - YouTube video ID extraction is strict
  - firebase_reporter uses bounded thread pool
  - RequestCostTracker has crew_name field
  - llm_benchmarks stores crew_name in request_costs
  - Metrics include error trend fields
  - Policy application is wired into all crews
  - Retrospective crew is not verbose in production
"""
import unittest


class TestSafeJsonParse(unittest.TestCase):
    """Verify safe_json_parse handles LLM output robustly."""

    def test_plain_json(self):
        from app.utils import safe_json_parse
        result, err = safe_json_parse('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})
        self.assertEqual(err, "")

    def test_markdown_fences(self):
        from app.utils import safe_json_parse
        result, err = safe_json_parse('```json\n{"a": 1}\n```')
        self.assertEqual(result, {"a": 1})
        self.assertEqual(err, "")

    def test_bare_fences(self):
        from app.utils import safe_json_parse
        result, err = safe_json_parse('```\n[1,2,3]\n```')
        self.assertEqual(result, [1, 2, 3])

    def test_invalid_json(self):
        from app.utils import safe_json_parse
        result, err = safe_json_parse('not json at all')
        self.assertIsNone(result)
        self.assertIn("JSON parse error", err)

    def test_empty_input(self):
        from app.utils import safe_json_parse
        result, err = safe_json_parse("")
        self.assertIsNone(result)
        self.assertIn("empty", err)

    def test_none_input(self):
        from app.utils import safe_json_parse
        result, err = safe_json_parse(None)
        self.assertIsNone(result)

    def test_size_limit(self):
        from app.utils import safe_json_parse
        huge = '{"x": "' + "a" * 200_000 + '"}'
        result, err = safe_json_parse(huge)
        self.assertIsNone(result)
        self.assertIn("too large", err)

    def test_custom_size_limit(self):
        from app.utils import safe_json_parse
        result, err = safe_json_parse('{"a": 1}', max_size=5)
        self.assertIsNone(result)
        self.assertIn("too large", err)

    def test_json_array(self):
        from app.utils import safe_json_parse
        result, err = safe_json_parse('["a", "b"]')
        self.assertEqual(result, ["a", "b"])


class TestYouTubeVideoIdExtraction(unittest.TestCase):
    """Verify video ID extraction is strict on character set.

    Uses source file checks since youtube_transcript_api may not be
    importable in all test environments.
    """

    def test_strict_character_class(self):
        """Regex should use [a-zA-Z0-9_-] not \\w (which allows non-ASCII)."""
        with open("app/tools/youtube_transcript.py") as f:
            source = f.read()
        self.assertIn("[a-zA-Z0-9_-]{11}", source)
        # Should NOT use \w for video ID matching (allows unicode)
        self.assertNotIn(r"[\w-]{11}", source)

    def test_shell_false_explicit(self):
        """subprocess.run must have shell=False for safety."""
        with open("app/tools/youtube_transcript.py") as f:
            source = f.read()
        self.assertIn("shell=False", source)

    def test_vtt_file_read_capped(self):
        """VTT file read must be capped to prevent OOM."""
        with open("app/tools/youtube_transcript.py") as f:
            source = f.read()
        self.assertIn("fh.read(1_000_000)", source)


class TestFirebaseReporterThreadPool(unittest.TestCase):
    """Verify firebase_reporter uses bounded thread pool."""

    def test_uses_thread_pool_executor(self):
        with open("app/firebase_reporter.py") as f:
            source = f.read()
        self.assertIn("ThreadPoolExecutor", source)
        self.assertIn("max_workers=4", source)

    def test_fire_uses_submit(self):
        with open("app/firebase_reporter.py") as f:
            source = f.read()
        self.assertIn("_executor.submit(fn)", source)
        # Should NOT use threading.Thread for _fire anymore
        self.assertNotIn("threading.Thread(target=fn", source)


class TestCostPerCrewTracking(unittest.TestCase):
    """Verify crew_name field exists and is persisted."""

    def test_tracker_has_crew_name(self):
        from app.rate_throttle import RequestCostTracker
        tracker = RequestCostTracker("test")
        self.assertEqual(tracker.crew_name, "")
        tracker.crew_name = "research"
        self.assertEqual(tracker.crew_name, "research")

    def test_record_request_cost_includes_crew(self):
        with open("app/llm_benchmarks.py") as f:
            source = f.read()
        self.assertIn("crew_name", source)
        self.assertIn("get_crew_cost_stats", source)

    def test_commander_sets_crew_name(self):
        with open("app/agents/commander.py") as f:
            source = f.read()
        self.assertIn("tracker.crew_name", source)

    def test_firebase_reports_crew_costs(self):
        with open("app/firebase_reporter.py") as f:
            source = f.read()
        self.assertIn("get_crew_cost_stats", source)
        self.assertIn("by_crew", source)


class TestErrorRateTrending(unittest.TestCase):
    """Verify error rate trending in metrics."""

    def test_metrics_include_trend(self):
        from app.metrics import compute_metrics
        m = compute_metrics()
        self.assertIn("error_rate_1h", m)
        self.assertIn("error_trend", m)
        self.assertIn(m["error_trend"], ("improving", "stable", "degrading"))

    def test_format_includes_trend(self):
        from app.metrics import format_metrics, compute_metrics
        text = format_metrics(compute_metrics())
        self.assertIn("last 1h:", text)
        # Should have a trend arrow
        self.assertTrue(
            any(arrow in text for arrow in ("↓", "↑", "→")),
            "Expected trend arrow in formatted metrics"
        )

    def test_error_trend_function(self):
        from app.metrics import _error_trend
        trend = _error_trend()
        self.assertIn(trend, ("improving", "stable", "degrading"))


class TestPolicyApplication(unittest.TestCase):
    """Verify policies are loaded and injected in all crews."""

    def test_research_crew_loads_policies(self):
        with open("app/crews/research_crew.py") as f:
            source = f.read()
        self.assertIn("load_relevant_policies", source)
        self.assertIn("policies_block", source)

    def test_coding_crew_loads_policies(self):
        with open("app/crews/coding_crew.py") as f:
            source = f.read()
        self.assertIn("load_relevant_policies", source)

    def test_writing_crew_loads_policies(self):
        with open("app/crews/writing_crew.py") as f:
            source = f.read()
        self.assertIn("load_relevant_policies", source)


class TestRetrospectiveNotVerbose(unittest.TestCase):
    """Verify retrospective crew is not verbose in production."""

    def test_verbose_false(self):
        with open("app/crews/retrospective_crew.py") as f:
            source = f.read()
        # The Crew instantiation should have verbose=False
        self.assertIn("verbose=False", source)
        # Should NOT have verbose=True
        self.assertNotIn("verbose=True", source)


class TestSafeJsonParseAdoption(unittest.TestCase):
    """Verify safe_json_parse is used in all LLM JSON parsing locations."""

    def test_evolution_uses_safe_parse(self):
        with open("app/evolution.py") as f:
            source = f.read()
        self.assertIn("safe_json_parse", source)
        # Should not have raw json.loads on LLM output
        self.assertNotIn("json.loads(raw_clean)", source)

    def test_auditor_uses_safe_parse(self):
        with open("app/auditor.py") as f:
            source = f.read()
        self.assertIn("safe_json_parse", source)

    def test_retrospective_uses_safe_parse(self):
        with open("app/crews/retrospective_crew.py") as f:
            source = f.read()
        self.assertIn("safe_json_parse", source)

    def test_commander_uses_safe_parse(self):
        with open("app/agents/commander.py") as f:
            source = f.read()
        self.assertIn("safe_json_parse", source)

    def test_research_crew_uses_safe_parse(self):
        with open("app/crews/research_crew.py") as f:
            source = f.read()
        self.assertIn("safe_json_parse", source)


if __name__ == "__main__":
    unittest.main()
