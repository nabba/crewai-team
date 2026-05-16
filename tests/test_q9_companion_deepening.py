"""PROGRAM §46.6-§46.9 — Q9 companion-deepening tests.

Covers:

  §46.6 Q9.3 — travel module (TripIt iCal parse, segment kind
                detection, daily-briefing formatting, snapshot
                read/write)
  §46.7 Q9.4 — inbox handlers wired + youtube-link classifier upgrade
                + handler registry covers all KNOWN_KINDS
  §46.8 Q9.5 — calendar_invite + signal_send action handlers
                registered + validate vocabulary
  §46.9 Q9.6 — long-term goal quarterly review module + REST + slash
                + scheduler wiring

Each test is failure-isolated against the gateway-deps in the test
env (no crewai / no anthropic SDK / no Google client).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────
#   §46.6 — Q9.3 travel
# ─────────────────────────────────────────────────────────────────────


_SAMPLE_ICAL = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//TripIt//Calendar//EN
BEGIN:VEVENT
UID:trip-1-seg-1@tripit.com
SUMMARY:AY 123 Helsinki to London
LOCATION:Helsinki Airport (HEL)
DTSTART:20260620T080000Z
DTEND:20260620T100000Z
END:VEVENT
BEGIN:VEVENT
UID:trip-1-seg-2@tripit.com
SUMMARY:Tallink Megastar Helsinki - Tallinn
LOCATION:West Harbour, Helsinki
DTSTART:20260625T160000Z
DTEND:20260625T200000Z
END:VEVENT
BEGIN:VEVENT
UID:trip-1-hotel@tripit.com
SUMMARY:Radisson Blu Tallinn check-in
LOCATION:Tallinn
DTSTART:20260625T210000Z
DTEND:20260627T120000Z
END:VEVENT
END:VCALENDAR
"""


def test_travel_parses_ical_segments() -> None:
    from app.life_companion.travel import parse_ical
    segs = parse_ical(_SAMPLE_ICAL)
    assert len(segs) == 3
    # Sorted chronologically
    assert segs[0].starts_at < segs[1].starts_at < segs[2].starts_at
    # Kind detection
    assert segs[0].kind == "flight"
    assert segs[0].flight_number == "AY123"
    assert segs[1].kind == "ferry"
    assert segs[2].kind == "hotel"


def test_travel_normalize_dt_handles_three_forms() -> None:
    from app.life_companion.travel import _normalize_dt
    # UTC stamp
    assert _normalize_dt("20260620T080000Z").startswith("2026-06-20T08:00:00")
    # Floating (no Z)
    assert _normalize_dt("20260620T080000").startswith("2026-06-20T08:00:00")
    # Date-only
    assert _normalize_dt("20260620").startswith("2026-06-20T00:00:00")
    # Unknown form passes through
    assert _normalize_dt("garbage") == "garbage"


def test_travel_segment_kind_detection() -> None:
    from app.life_companion.travel import _detect_segment_kind
    kind, fn = _detect_segment_kind("BA 456 to Paris", "")
    assert kind == "flight"
    assert fn == "BA456"
    kind, _ = _detect_segment_kind("Eckerö Line departure", "")
    assert kind == "ferry"
    kind, _ = _detect_segment_kind("VR Intercity 123", "")
    assert kind == "train"
    kind, _ = _detect_segment_kind("Hertz car rental pickup", "")
    assert kind == "car"
    kind, _ = _detect_segment_kind("Just some random event", "")
    assert kind == "other"


def test_travel_upcoming_trips_filters_window(tmp_path, monkeypatch) -> None:
    """upcoming_trips reads the snapshot and filters by start window."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    snap = tmp_path / "life_companion" / "travel" / "tripit_trips.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    from datetime import timedelta as _td
    now = datetime.now(timezone.utc)
    future = (now + _td(hours=2)).replace(microsecond=0)
    snap.write_text(json.dumps([
        # future, in window
        {
            "summary": "F1", "location": "Helsinki",
            "starts_at": future.isoformat(),
            "ends_at": future.isoformat(),
            "uid": "u1", "kind": "flight", "flight_number": "AY1",
        },
        # past (excluded)
        {
            "summary": "F2", "location": "",
            "starts_at": "2020-01-01T00:00:00+00:00",
            "ends_at": "2020-01-01T01:00:00+00:00",
            "uid": "u2", "kind": "flight", "flight_number": "",
        },
    ]))
    from app.life_companion.travel import upcoming_trips
    out = upcoming_trips(window_days=30)
    summaries = {s.summary for s in out}
    assert "F1" in summaries
    assert "F2" not in summaries


def test_travel_format_for_briefing_empty_when_no_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from app.life_companion.travel import format_for_briefing
    assert format_for_briefing() == ""


def test_travel_run_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_MONITOR_ENABLED", "false")
    from app.life_companion.travel import run
    r = run()
    assert r["status"] == "skipped_disabled"


def test_travel_url_resolution_precedence(tmp_path, monkeypatch) -> None:
    """Q9.3 follow-up: ``runtime_settings.tripit_ical_url`` wins over
    ``TRIPIT_ICAL_URL`` env var. Same for the Aviationstack key."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    # Set env var (should LOSE to runtime_settings)
    monkeypatch.setenv("TRIPIT_ICAL_URL", "https://env.tripit.com/feed.ics")
    monkeypatch.setenv("AVIATIONSTACK_API_KEY", "envkey_envkey_envkey_envkey")
    # Configure runtime_settings via a stub that mimics the real
    # module's getters
    import sys
    from types import ModuleType
    fake = ModuleType("app.runtime_settings_fake")
    setattr(fake, "get_tripit_ical_url",
            lambda: "https://settings.tripit.com/feed.ics")
    setattr(fake, "get_aviationstack_api_key",
            lambda: "settings_settings_settings_token")
    sys.modules["app.runtime_settings"] = fake
    try:
        from app.life_companion.travel import (
            _get_aviationstack_key, _get_tripit_url,
        )
        assert _get_tripit_url() == "https://settings.tripit.com/feed.ics"
        assert _get_aviationstack_key() == "settings_settings_settings_token"
    finally:
        sys.modules.pop("app.runtime_settings", None)


def test_travel_url_falls_back_to_env_when_runtime_settings_empty(
    tmp_path, monkeypatch,
) -> None:
    """Empty runtime_settings → env-var fallback (back-compat)."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("TRIPIT_ICAL_URL", "https://env.tripit.com/feed.ics")
    monkeypatch.setenv("AVIATIONSTACK_API_KEY", "envkey_envkey_envkey")
    import sys
    from types import ModuleType
    fake = ModuleType("app.runtime_settings_fake")
    setattr(fake, "get_tripit_ical_url", lambda: "")
    setattr(fake, "get_aviationstack_api_key", lambda: "")
    sys.modules["app.runtime_settings"] = fake
    try:
        from app.life_companion.travel import (
            _get_aviationstack_key, _get_tripit_url,
        )
        assert _get_tripit_url() == "https://env.tripit.com/feed.ics"
        assert _get_aviationstack_key() == "envkey_envkey_envkey"
    finally:
        sys.modules.pop("app.runtime_settings", None)


def test_travel_card_mounted_in_settings_page() -> None:
    """Source-level wiring: TravelCard imported + mounted."""
    page = Path("dashboard-react/src/components/SettingsPage.tsx").read_text(
        encoding="utf-8",
    )
    assert "import { TravelCard }" in page
    assert "<TravelCard" in page
    card = Path("dashboard-react/src/components/TravelCard.tsx").read_text(
        encoding="utf-8",
    )
    assert "tripit_ical_url" in card
    assert "aviationstack_api_key" in card


def test_config_api_handles_travel_keys() -> None:
    src = Path("app/api/config_api.py").read_text(encoding="utf-8")
    assert "set_tripit_ical_url" in src
    assert "set_aviationstack_api_key" in src
    assert '"tripit_ical_url" in payload' in src
    assert '"aviationstack_api_key" in payload' in src


def test_verify_gateway_secret_is_mode_aware() -> None:
    """PROGRAM §46.12 — verify_gateway_secret must mirror
    require_gateway_auth's dev/prod split. In dev mode (no
    GATEWAY_AUTH_REQUIRED) it MUST pass through so React settings
    cards work without baking VITE_GATEWAY_SECRET into the JS
    bundle. In prod (GATEWAY_AUTH_REQUIRED=1) it MUST enforce."""
    src = Path("app/api/config_api.py").read_text(encoding="utf-8")
    # The function must read GATEWAY_AUTH_REQUIRED
    assert 'GATEWAY_AUTH_REQUIRED' in src
    # It must have the pass-through branch
    fn_start = src.find("def verify_gateway_secret(")
    fn_end = src.find("\ndef ", fn_start + 1)
    body = src[fn_start:fn_end]
    assert 'return True' in body
    assert "in (\"1\", \"true\"" in body or "'1', 'true'" in body


def test_settings_alias_endpoint_exists_and_mounted() -> None:
    """Latent-bug closure (PROGRAM §46.12): React cards all call
    /api/cp/settings, which previously had no handler. The alias
    router forwards to the canonical /config/runtime_settings."""
    alias_src = Path("app/control_plane/settings_alias_api.py").read_text(
        encoding="utf-8",
    )
    assert '@router.get("/settings")' in alias_src
    assert '@router.post("/settings")' in alias_src
    assert "set_runtime_settings_endpoint" in alias_src

    main_src = Path("app/main.py").read_text(encoding="utf-8")
    assert (
        "from app.control_plane.settings_alias_api import router as settings_alias_router"
        in main_src
    )
    assert "app.include_router(settings_alias_router)" in main_src


# ─────────────────────────────────────────────────────────────────────
#   §46.7 — Q9.4 inbox classifier + handler wiring
# ─────────────────────────────────────────────────────────────────────


def test_classifier_recognises_youtube_url_file(tmp_path) -> None:
    from app.inbox.classifier import classify_file
    f = tmp_path / "watch.url"
    f.write_text(
        "[InternetShortcut]\nURL=https://www.youtube.com/watch?v=abc123\n",
        encoding="utf-8",
    )
    c = classify_file(f)
    assert c.kind == "youtube_link"


def test_classifier_recognises_youtube_in_txt(tmp_path) -> None:
    from app.inbox.classifier import classify_file
    f = tmp_path / "link.txt"
    f.write_text("https://youtu.be/xyz789\n", encoding="utf-8")
    c = classify_file(f)
    assert c.kind == "youtube_link"


def test_classifier_treats_plain_txt_as_text(tmp_path) -> None:
    from app.inbox.classifier import classify_file
    f = tmp_path / "note.txt"
    f.write_text("Just a regular note, no URLs.\n", encoding="utf-8")
    c = classify_file(f)
    assert c.kind == "text"


def test_classifier_webloc_with_youtube_url(tmp_path) -> None:
    from app.inbox.classifier import classify_file
    f = tmp_path / "video.webloc"
    f.write_text(
        '<?xml version="1.0"?>\n<plist><dict>'
        '<key>URL</key><string>https://www.youtube.com/watch?v=abc</string>'
        '</dict></plist>\n',
        encoding="utf-8",
    )
    c = classify_file(f)
    assert c.kind == "youtube_link"


def test_inbox_handler_registry_covers_all_kinds() -> None:
    """Every KNOWN_KINDS entry except ``unknown`` has a real handler;
    none are stubbed as ``_handle_unsupported``."""
    from app.inbox.router import HANDLER_REGISTRY, _handle_unsupported
    from app.inbox.classifier import KNOWN_KINDS
    for kind in KNOWN_KINDS:
        if kind == "unknown":
            continue
        assert kind in HANDLER_REGISTRY, f"missing handler for {kind!r}"
        h = HANDLER_REGISTRY[kind]
        assert h is not _handle_unsupported, (
            f"{kind!r} still wired to _handle_unsupported"
        )


def test_inbox_youtube_handler_extracts_url_from_url_shortcut(tmp_path) -> None:
    """Source-level: the .url shortcut path uses configparser to
    extract the URL field."""
    src = Path("app/inbox/handlers/youtube_link.py").read_text(encoding="utf-8")
    assert "configparser" in src
    assert "InternetShortcut" in src
    assert "youtube" in src.lower()


def test_inbox_pdf_handler_routes_receipts_to_expense_ledger() -> None:
    """Source-level: the PDF handler appends to expenses.jsonl when
    the parsed JSON has kind=receipt."""
    src = Path("app/inbox/handlers/pdf_extract.py").read_text(encoding="utf-8")
    assert "expenses.jsonl" in src
    assert "kind=\"receipt\"" in src or "kind=='receipt'" in src or "'receipt'" in src


def test_inbox_image_handler_uses_anthropic_haiku_vision() -> None:
    src = Path("app/inbox/handlers/image_vision.py").read_text(encoding="utf-8")
    assert "claude-haiku" in src
    assert "media_type" in src
    assert "base64" in src


# ─────────────────────────────────────────────────────────────────────
#   §46.8 — Q9.5 action handlers
# ─────────────────────────────────────────────────────────────────────


def test_action_type_enum_includes_new_types() -> None:
    from app.action_requests.models import ActionType
    assert ActionType.CALENDAR_INVITE.value == "calendar_invite"
    assert ActionType.SIGNAL_SEND.value == "signal_send"


def test_calendar_invite_handler_validates_required_fields() -> None:
    from app.action_requests.handlers.calendar_invite import CalendarInviteHandler
    h = CalendarInviteHandler()
    # Missing summary
    ok, err = h.validate({"start_iso": "2026-06-01T10:00:00+00:00",
                          "end_iso": "2026-06-01T10:30:00+00:00"})
    assert not ok and "summary" in err
    # Bad start_iso
    ok, err = h.validate({"summary": "x", "start_iso": "bogus",
                          "end_iso": "2026-06-01T10:30:00+00:00"})
    assert not ok and "start_iso" in err
    # End before start
    ok, err = h.validate({"summary": "x",
                          "start_iso": "2026-06-01T10:30:00+00:00",
                          "end_iso": "2026-06-01T10:00:00+00:00"})
    assert not ok and "after start_iso" in err
    # Bad attendee
    ok, err = h.validate({"summary": "x",
                          "start_iso": "2026-06-01T10:00:00+00:00",
                          "end_iso": "2026-06-01T10:30:00+00:00",
                          "attendees": ["not-an-email"]})
    assert not ok and "attendee" in err
    # Good
    ok, err = h.validate({"summary": "Coffee",
                          "start_iso": "2026-06-01T10:00:00+00:00",
                          "end_iso": "2026-06-01T10:30:00+00:00",
                          "attendees": ["a@b.co"]})
    assert ok and err is None


def test_calendar_invite_renders_summary() -> None:
    from app.action_requests.handlers.calendar_invite import CalendarInviteHandler
    h = CalendarInviteHandler()
    out = h.render_summary({
        "summary": "Coffee with Ave",
        "start_iso": "2026-06-01T10:00:00+02:00",
        "location": "Cafe Esplanad",
        "attendees": ["ave@x.co"],
    })
    assert "Coffee with Ave" in out
    assert "Cafe Esplanad" in out
    assert "1 attendee" in out


def test_signal_send_handler_validates_required_fields() -> None:
    from app.action_requests.handlers.signal_send import SignalSendHandler
    h = SignalSendHandler()
    ok, err = h.validate({"recipient": "", "text": "hi"})
    assert not ok and "recipient" in err
    ok, err = h.validate({"recipient": "+358401234567", "text": ""})
    assert not ok and "text" in err
    ok, err = h.validate({"recipient": "+358401234567", "text": "hello"})
    assert ok and err is None


def test_signal_send_renders_summary_with_preview() -> None:
    from app.action_requests.handlers.signal_send import SignalSendHandler
    h = SignalSendHandler()
    out = h.render_summary({
        "recipient": "+358401234567",
        "text": "Long-ish body that should get truncated with an ellipsis when over 60 chars",
    })
    assert "Signal to" in out
    assert "+358401234567" in out
    assert "…" in out  # truncated


def test_action_handlers_register_at_import() -> None:
    """Both new handlers are registered (best-effort) when the
    handlers package imports."""
    src = Path("app/action_requests/handlers/__init__.py").read_text(encoding="utf-8")
    assert "calendar_invite" in src
    assert "signal_send" in src
    assert "CalendarInviteHandler" in src
    assert "SignalSendHandler" in src


# ─────────────────────────────────────────────────────────────────────
#   §46.9 — Q9.6 long-term goal review
# ─────────────────────────────────────────────────────────────────────


def test_quarter_label_format() -> None:
    from app.identity.long_term_goal_review import _current_quarter, _quarter_label
    # Synthetic now in Q2 (May)
    now = datetime(2026, 5, 16, tzinfo=timezone.utc)
    year, q, start, next_start = _current_quarter(now)
    assert year == 2026
    assert q == 2
    assert start.month == 4
    assert next_start.month == 7
    assert _quarter_label(year, q) == "2026_q2"


def test_quarter_label_q4_wraps_year() -> None:
    from app.identity.long_term_goal_review import _current_quarter
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    year, q, start, next_start = _current_quarter(now)
    assert q == 4
    assert start.month == 10
    assert next_start.year == 2027
    assert next_start.month == 1


def test_run_review_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LONG_TERM_GOAL_REVIEW_ENABLED", "false")
    from app.identity.long_term_goal_review import run_review
    r = run_review()
    assert r.status == "skipped_disabled"


def test_run_review_skips_when_recent(tmp_path, monkeypatch) -> None:
    """A recent last_run_at blocks the daily cadence guard (force=False)."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LONG_TERM_GOAL_REVIEW_ENABLED", "true")
    state_dir = tmp_path / "identity"
    state_dir.mkdir(parents=True, exist_ok=True)
    import time as _time
    (state_dir / "long_term_goal_review_state.json").write_text(
        json.dumps({"last_run_at": _time.time()}), encoding="utf-8",
    )
    from app.identity.long_term_goal_review import run_review
    r = run_review()
    assert r.status == "skipped_recent"


def test_run_review_writes_essay_when_forced(tmp_path, monkeypatch) -> None:
    """force=True bypasses cadence; with injected LLM call we write
    the essay file."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LONG_TERM_GOAL_REVIEW_ENABLED", "true")
    monkeypatch.setenv("LONG_TERM_GOAL_REVIEW_DIR", str(tmp_path / "reviews"))

    def fake_llm(system: str, user: str) -> str:
        return (
            "## What the stated long-term goals are\n\nGoal A, Goal B.\n\n"
            "## What measurable progress occurred this quarter\n\nN/A.\n"
        )

    from app.identity.long_term_goal_review import run_review
    r = run_review(force=True, llm_call=fake_llm)
    assert r.status == "wrote_review"
    assert r.quarter_label.startswith("20")
    assert "_q" in r.quarter_label
    dest = Path(r.written_to)
    assert dest.exists()
    body = dest.read_text(encoding="utf-8")
    assert "Goal A" in body


def test_list_recent_reviews_returns_newest_first(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LONG_TERM_GOAL_REVIEW_DIR", str(tmp_path))
    (tmp_path / "2026_q1.md").write_text("first")
    import time as _t
    _t.sleep(0.01)  # force distinct mtimes
    (tmp_path / "2026_q2.md").write_text("second")
    from app.identity.long_term_goal_review import list_recent_reviews
    rows = list_recent_reviews()
    assert rows[0]["quarter_label"] == "2026_q2"


def test_goals_rest_endpoints_exist() -> None:
    """Source-level: the /api/cp/goals routes are defined."""
    src = Path("app/control_plane/goals_api.py").read_text(encoding="utf-8")
    assert '@router.get("/state")' in src
    assert '@router.get("/reviews")' in src
    assert '@router.post("/review")' in src
    main_src = Path("app/main.py").read_text(encoding="utf-8")
    assert "from app.control_plane.goals_api import router as goals_router" in main_src


def test_goals_slash_command_handler_exists() -> None:
    src = Path("app/agents/commander/commands.py").read_text(encoding="utf-8")
    assert "def _handle_goals_command(" in src
    assert '"/goals review"' in src or "/goals review" in src
    reg_src = Path("app/agents/commander/command_registry.py").read_text(encoding="utf-8")
    assert "/goals review" in reg_src
    assert "/goals list-reviews" in reg_src


def test_goal_review_wired_into_identity_scheduler() -> None:
    src = Path("app/identity/scheduler.py").read_text(encoding="utf-8")
    assert "long_term_goal_review" in src
    assert "identity-long-term-goal-review" in src
