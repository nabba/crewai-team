"""Tests for output quality improvements — response cleaning, metadata stripping,
and research crew fast path for simple questions."""
import unittest
import re


class TestCleanResponse(unittest.TestCase):
    """Test the _clean_response function strips internal metadata."""

    def _clean(self, text):
        # _clean_response moved to postprocess.py when commander.py was
        # modularized into commander/{commands,context,execution,
        # orchestrator,postprocess,routing}.py.
        with open("app/agents/commander/postprocess.py") as f:
            source = f.read()
        assert "def _clean_response" in source
        from app.agents.commander.postprocess import _clean_response
        return _clean_response(text)

    def test_strips_critic_review(self):
        text = "Here is the answer.\n\n---\n\n**[Critic Review]**\nSome internal review text."
        result = self._clean(text)
        self.assertNotIn("Critic Review", result)
        self.assertIn("Here is the answer", result)

    def test_strips_proactive_notes(self):
        text = "Answer here.\n\n---\n\n**[Proactive Notes]**\n- Some proactive thing"
        result = self._clean(text)
        self.assertNotIn("Proactive", result)
        self.assertIn("Answer here", result)

    def test_strips_self_report(self):
        text = "Result.\n\n---\n\n**[Self Report]**\nConfidence: 0.8"
        result = self._clean(text)
        self.assertNotIn("Self Report", result)
        self.assertIn("Result", result)

    def test_strips_sub_agent_failure(self):
        text = "Partial results.\n\nNote: 2 sub-tasks failed."
        result = self._clean(text)
        self.assertNotIn("sub-tasks failed", result)
        self.assertIn("Partial results", result)

    def test_truncates_long_response(self):
        text = "A" * 2000
        result = self._clean(text)
        self.assertLessEqual(len(result), 1500)  # 1400 + truncation note
        self.assertIn("[Full response attached", result)

    def test_preserves_short_response(self):
        text = "Finland has 4.14 hectares of woodland per capita."
        result = self._clean(text)
        self.assertEqual(text, result)

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. " * 100  # ~1600 chars
        result = self._clean(text)
        self.assertIn("[Full response attached", result)
        # Should end at a sentence boundary, not mid-word
        before_truncation = result.split("\n\n[Full response attached")[0]
        self.assertTrue(before_truncation.endswith("."))

    def test_empty_input(self):
        self.assertEqual(self._clean(""), "")
        self.assertEqual(self._clean(None), None)


class TestResearchCrewFastPath(unittest.TestCase):
    """Verify research crew source has a fast path for simple questions."""

    def test_simple_research_template_exists(self):
        with open("app/crews/research_crew.py") as f:
            source = f.read()
        self.assertIn("SIMPLE_RESEARCH_TEMPLATE", source)
        self.assertIn("concisely and directly", source)

    def test_difficulty_gates_debate_and_critic(self):
        with open("app/crews/research_crew.py") as f:
            source = f.read()
        # Low difficulty should skip debate and critic
        self.assertIn("if difficulty <= 3:", source)
        self.assertIn("_run_simple", source)
        # Debate threshold was tightened (Apr 2026): now difficulty >= 8,
        # or >= 6 only when synthesis output is short (< 200 chars).
        # Previously triggered at >= 6 unconditionally.
        self.assertIn("difficulty >= 8", source)

    def test_critic_review_does_not_append_to_output(self):
        """Critic review must NOT append [Critic Review] to user output."""
        with open("app/crews/research_crew.py") as f:
            source = f.read()
        self.assertNotIn('result += f"\\n\\n---\\n\\n**[Critic Review]', source)

    def test_writing_crew_critic_does_not_append(self):
        with open("app/crews/writing_crew.py") as f:
            source = f.read()
        self.assertNotIn('result += f"\\n\\n---\\n\\n**[Critic Review]', source)

    def test_coding_crew_critic_does_not_append(self):
        with open("app/crews/coding_crew.py") as f:
            source = f.read()
        self.assertNotIn('result += f"\\n\\n---\\n\\n**[Critic Review]', source)


class TestRoutingPromptQuality(unittest.TestCase):
    """Verify routing prompt includes conciseness rules."""

    def test_routing_prompt_has_output_rules(self):
        # Routing prompt moved to commander/routing.py during the
        # commander.py modularization.
        with open("app/agents/commander/routing.py") as f:
            source = f.read()
        self.assertIn("CRITICAL OUTPUT RULES", source)
        self.assertIn("PHONE via Signal", source)
        self.assertIn("NOT a report", source)


class TestProactiveNotesNotInOutput(unittest.TestCase):
    """Proactive scan results should be logged, not sent to user."""

    def test_proactive_notes_not_appended(self):
        # Proactive-notes stripping moved to commander/postprocess.py;
        # the orchestrator no longer appends them to final_result. We
        # assert the postprocess STRIP_PATTERNS still includes the
        # Proactive Notes regex (i.e., we still scrub them if present).
        with open("app/agents/commander/postprocess.py") as f:
            postprocess_src = f.read()
        self.assertIn("Proactive Notes", postprocess_src)
        with open("app/agents/commander/orchestrator.py") as f:
            orchestrator_src = f.read()
        # The old pattern appended Proactive Notes — make sure that's gone.
        self.assertNotIn('final_result += (\n                    "\\n\\n---\\n\\n**[Proactive Notes]', orchestrator_src)


if __name__ == "__main__":
    unittest.main()
