"""Regression test for routing of execution-intent messages.

Background — 2026-05-02: in a forest-deforestation Signal thread the
user asked five times in a row to "execute the plan" / "run the
scripts" / "produce the report". Every request was routed to
``direct`` (the commander LLM answers itself) instead of to ``coding``
(the crew that actually has a sandbox + can run things). The LLM
then refused with "I cannot execute this pipeline / I can't run
code" — five times.

Root cause: the fast_route patterns in
``app/agents/commander/routing.py`` matched ``write|create|build|fix|
debug|refactor`` for code intent, but not ``execute|run|produce|
compile``. So the routing LLM's open-ended judgment took over and
chose direct.

Fix: add fast_route patterns that catch execution / report-production
intent and force-route to the coding crew at difficulty 7 (the level
that activates the full recovery-loop strategy set).
"""
from __future__ import annotations

import pytest

from app.agents.commander.routing import _try_fast_route


def _route(text: str):
    """Run the message through fast_route; return (crew, difficulty)
    or None if no rule matched."""
    decisions = _try_fast_route(text, has_attachments=False)
    if not decisions:
        return None
    return decisions[0]["crew"], decisions[0].get("difficulty")


class TestExecutionIntentRouting:

    def test_execute_the_plan_routes_to_coding(self):
        assert _route("please execute the plan") == ("coding", 7)
        assert _route("execute the plan") == ("coding", 7)

    def test_execute_in_sandbox_variants(self):
        assert _route("please execute the plan in your sandbox") == ("coding", 7)
        assert _route("execute the plan in the sandbox") == ("coding", 7)

    def test_run_the_scripts(self):
        assert _route("run the scripts") == ("coding", 7)
        assert _route("please run the script") == ("coding", 7)
        assert _route("run the analysis") == ("coding", 7)
        assert _route("run the pipeline") == ("coding", 7)

    def test_kick_off_workflow(self):
        assert _route("kick off the analysis") == ("coding", 7)
        assert _route("launch the pipeline") == ("coding", 7)

    def test_produce_compile_generate_report(self):
        assert _route("please produce the report") == ("coding", 7)
        assert _route("compile the report") == ("coding", 7)
        assert _route("generate me the maps") == ("coding", 7)
        assert _route("build the dataset") == ("coding", 7)

    def test_produce_with_articles(self):
        assert _route("produce a report") == ("coding", 7)
        assert _route("compile an output") == ("coding", 7)

    def test_produce_with_qualifiers(self):
        # ≤3 qualifier words allowed between article and noun.
        assert _route("produce a deforestation report") == ("coding", 7)
        assert _route("generate annual maps") == ("coding", 7)
        assert _route("compile the year-by-year results") == ("coding", 7)

    # ── Negative — must NOT over-match ──

    def test_does_not_match_status_questions(self):
        assert _route("what is the workspace") is None
        assert _route("how does evolution work") is None

    def test_does_not_match_research_questions(self):
        # "explain" → research, not coding
        crew, _ = _route("explain how Hansen GFC works")
        assert crew == "research"

    def test_does_not_match_create_summary(self):
        # The pre-existing writing pattern still wins for ``create
        # a summary`` (matched as writing intent, not coding) —
        # proving our new code-execution pattern doesn't over-claim
        # on bare ``create`` verbs that have writing-shaped objects.
        crew, _ = _route("create a summary of the meeting")
        assert crew == "writing"
