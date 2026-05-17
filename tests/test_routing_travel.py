"""Routing for travel / TripIt questions → PIM crew.

Pre-fix (2026-05-17): "what is address of my hotel in bucharest" and
"what are the next flights i must take" both fell through to the
``what|who|when|where`` fast-route → research crew, which doesn't
have the list_upcoming_flights / list_upcoming_trips tools wired
into its inventory.  Research crew correctly reported "missing tool"
on the first phrasing and silently failed on the second
("I'm sorry, but I couldn't find any information...").

Same pattern as the 2026-05-09 ticket-routing fix
(``test_routing_ticket_ops.py``): the noun gate gets the new tokens
(flights / hotels / trips / itineraries / reservations / bookings /
travels), the qualifier gate gets temporal anchors that pair with
those nouns (next / upcoming / future), the PIM short-circuit fires
in the dual-signal heuristic, and the LLM router's
``_CREW_BASE_PURPOSE`` description mentions travel for the
fast-route-misses path.
"""
from __future__ import annotations


def _route(text: str):
    """Run the fast-route classifier; return crew name or None."""
    from app.agents.commander.routing import _try_fast_route

    out = _try_fast_route(text, has_attachments=False)
    if not out:
        return None
    return out[0].get("crew")


class TestTravelQuestionsRouteToPim:
    """Live 2026-05-17 phrasings the operator actually tried."""

    def test_address_of_my_hotel(self) -> None:
        # The exact phrasing from the bug report.
        assert _route("what is address of my hotel in bucharest") == "pim"

    def test_next_flights_i_must_take(self) -> None:
        # The earlier phrasing from the same bug report.
        assert _route("what are the next flights i must take") == "pim"


class TestTravelShapesPimShortCircuit:
    """Sample of phrasings that should hit the PIM short-circuit."""

    def test_my_hotel(self) -> None:
        assert _route("where is my hotel") == "pim"

    def test_my_flight(self) -> None:
        assert _route("when is my flight") == "pim"

    def test_my_trip(self) -> None:
        assert _route("show me my trip details") == "pim"

    def test_my_itinerary(self) -> None:
        assert _route("what is my itinerary for next week") == "pim"

    def test_upcoming_trips(self) -> None:
        assert _route("list my upcoming trips") == "pim"

    def test_next_flight_with_my(self) -> None:
        assert _route("what is my next flight") == "pim"


class TestNonPimTravelStaysResearch:
    """Travel-topic questions WITHOUT a personal qualifier — these are
    research, not PIM, and must stay that way.  Pinned so the
    travel-noun expansion can't grow into a general flight-knowledge
    intercept by accident."""

    def test_general_flight_info(self) -> None:
        # No personal qualifier → research.  Don't intercept.
        assert _route("what airlines fly from tallinn to bucharest") != "pim"

    def test_general_hotel_info(self) -> None:
        assert _route("what hotels are popular in bucharest") != "pim"

    def test_research_on_flight_emissions(self) -> None:
        # The known weak case from the 2026-05-09 ticket-fix
        # comment ("research about email marketing at companies X,
        # Y, Z stays research") — same shape, different topic.
        assert _route(
            "research about flight emissions in european aviation"
        ) != "pim"
