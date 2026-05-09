"""Routing for control_plane.tickets operations → PIM crew.

Pre-fix: "move the forest-age task to Eesti mets" fell through every
fast-route pattern, hit the LLM router, which sometimes picked the
research crew (no cp_tickets tools), and the agent then hallucinated
"no tasks found" because the local task-tools SQLite is a separate
DB from control_plane.tickets. PR #74 added the cp_*-ticket tools
to PIM; this PR makes sure the routing layer actually picks PIM.

Two pieces:
  1. Fast-route patterns for common ticket-ops phrasings
     (move/list/search/kanban) short-circuit to PIM without an LLM
     call.
  2. The PIM crew description in `_CREW_BASE_PURPOSE` mentions Kanban
     ticket ops so the LLM router (catch-all) also picks PIM when
     the fast-route misses.
"""
from __future__ import annotations

import pytest


def _route(text: str):
    """Run the fast-route classifier and return crew name or None.

    The PIM short-circuit returns difficulty=3 uniformly — we only
    assert on the crew name; the orchestrator picks the actual model
    tier downstream from heuristics that don't depend on this number.
    """
    from app.agents.commander.routing import _try_fast_route

    out = _try_fast_route(text, has_attachments=False)
    if not out:
        return None
    return out[0].get("crew")


class TestMoveTicketRoutes:
    """The exact 2026-05-09 phrasings that should now reach PIM."""

    def test_move_task_with_explicit_workspace(self) -> None:
        assert _route(
            "move the forest-age task from PLG to Eesti mets"
        ) == "pim"

    def test_move_task_simpler_phrasing(self) -> None:
        assert _route("move the task to Eesti mets") == "pim"

    def test_move_ticket_phrasing(self) -> None:
        assert _route(
            "move the ticket about forest age to Eesti mets"
        ) == "pim"

    def test_please_move_form(self) -> None:
        assert _route("please move that task to KaiCart") == "pim"

    def test_move_into_or_under(self) -> None:
        assert _route("move the task into Archibal") == "pim"
        assert _route("move ticket 42 under PLG") == "pim"


class TestListTicketsRoutes:

    def test_list_my_tickets(self) -> None:
        assert _route("list my tickets") == "pim"

    def test_list_tickets_in_workspace(self) -> None:
        assert _route("list all my tickets in PLG") == "pim"

    def test_show_my_tickets(self) -> None:
        assert _route("show my tickets") == "pim"

    def test_what_tickets_in_kanban(self) -> None:
        assert _route("show my kanban") == "pim"

    def test_what_is_on_my_kanban(self) -> None:
        assert _route("what is on my kanban") == "pim"


class TestSearchTicketsRoutes:

    def test_search_my_tickets(self) -> None:
        assert _route("search my tickets for forest") == "pim"

    def test_find_my_tickets(self) -> None:
        assert _route("find my tickets about deforestation") == "pim"


# ── Things that MUST NOT route to PIM ───────────────────────────────


class TestNoOverreach:
    """Make sure the new patterns don't hijack unrelated queries."""

    def test_move_a_file_does_not_route_to_pim(self) -> None:
        """'move' alone is not a PIM signal — the dual-signal
        (noun ∧ qualifier) gate requires a ticket/task noun too."""
        out = _route("move the forest.tif file to /data/")
        assert out != "pim"

    def test_research_question_unaffected(self) -> None:
        out = _route("what is the population of Estonia?")
        # research / direct / None — anything but pim is fine
        assert out != "pim"

    def test_coding_request_unaffected(self) -> None:
        out = _route("write a Python script that processes data")
        assert out != "pim"

    def test_search_the_web_not_pim(self) -> None:
        """'search' qualifier + no PIM noun — must not route to PIM."""
        out = _route("search the web for forest data")
        # Note: 'search' is a PIM qualifier, but there's no PIM noun
        # (email/inbox/ticket/...). The dual-signal gate rejects.
        assert out != "pim"


# ── LLM-router catalog ──────────────────────────────────────────────


class TestCrewCatalog:
    """The LLM-router catalog must mention Kanban so the LLM picks
    PIM for queries the fast-route doesn't catch."""

    def test_pim_purpose_mentions_kanban(self) -> None:
        from app.agents.commander.routing import _CREW_BASE_PURPOSE

        pim_purpose = _CREW_BASE_PURPOSE["pim"]
        assert "kanban" in pim_purpose.lower(), (
            "PIM crew description must mention 'Kanban' so the LLM "
            "router picks PIM for ticket-move queries"
        )
        assert "control_plane.tickets" in pim_purpose, (
            "PIM purpose must call out control_plane.tickets so the "
            "LLM doesn't conflate with the local SQLite tasks DB"
        )

    def test_pim_purpose_mentions_move_between_workspaces(self) -> None:
        from app.agents.commander.routing import _CREW_BASE_PURPOSE

        text = _CREW_BASE_PURPOSE["pim"].lower()
        assert "move" in text and "workspace" in text, (
            "PIM purpose should say 'move ... between workspaces' so "
            "the operator's natural phrasing routes here"
        )
