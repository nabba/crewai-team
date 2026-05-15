"""Travel monitor — PROGRAM §46.6 (Q9.3).

A passive surface that watches the operator's upcoming travel and
surfaces it through the existing personal-life channels (calendar
prep + daily briefing). No mutations; no auto-booking.

Two signal sources:

  1. **TripIt iCal feed** — TripIt offers a per-user ``.ics`` URL
     (Settings → Calendar Sync → "Copy to your calendar"). Operator
     sets ``TRIPIT_ICAL_URL`` env var; we fetch + parse on a daily
     cadence and keep a snapshot at ``workspace/life_companion/
     travel/tripit_trips.json``. NO OAuth dance.
  2. **Aviationstack flight status** — for the next departing flight
     within 24 h, we hit the free tier (100 calls/month) for live
     status (delay / gate / terminal). ``AVIATIONSTACK_API_KEY`` env
     var gates this; absent key = no live status, the trip data
     itself still surfaces.

Cross-cuts:

  * **calendar_prep** — when prepping a meeting in another city, the
    pre-meeting nudge appends a "🛫 You're traveling to <city>" line.
  * **daily_briefing** — gains a "🛫 Travel" section listing trips
    in the next 14 days with flight-status badges.
  * **routine_detector** — travel days are excluded from the routine
    DOW × hour-bucket clustering (a Helsinki commute that hits 0
    on Tuesdays because the operator was in Tallinn shouldn't read
    as "routine broken").

Master switches:

  * ``TRAVEL_MONITOR_ENABLED`` (default ``true``)
  * ``TRIPIT_ICAL_URL`` (env; absence = TripIt source disabled)
  * ``FLIGHT_TRACKING_ENABLED`` (default ``true`` only matters when
    the Aviationstack key is also set)
  * ``AVIATIONSTACK_API_KEY`` (env; absence = flight status disabled)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────


_RUN_CADENCE_S = 6 * 3600  # refresh TripIt + flight status every 6h
_DEFAULT_WINDOW_DAYS = 14
_STATE_FILE_NAME = "travel_state.json"
_TRIPS_FILE_NAME = "tripit_trips.json"
_FLIGHTS_FILE_NAME = "flight_status.json"
_HTTP_TIMEOUT_S = 15


def _enabled() -> bool:
    return os.getenv("TRAVEL_MONITOR_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _flight_tracking_enabled() -> bool:
    return os.getenv("FLIGHT_TRACKING_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _data_dir() -> Path:
    base = Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))
    d = base / "life_companion" / "travel"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Data model ────────────────────────────────────────────────────────


@dataclass
class TripSegment:
    """One leg of a trip (flight, ferry, train, hotel, car)."""

    summary: str               # raw iCal SUMMARY
    location: str               # iCal LOCATION (may be empty)
    starts_at: str              # ISO timestamp
    ends_at: str                # ISO timestamp
    uid: str                    # iCal UID (stable per segment)
    kind: str = "other"         # "flight" | "ferry" | "train" | "hotel" | "car" | "other"
    flight_number: str = ""     # parsed if kind=="flight"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlightStatus:
    """Aviationstack live-status snapshot."""

    flight_number: str
    departure_iata: str = ""
    arrival_iata: str = ""
    status: str = ""            # scheduled | active | landed | cancelled | etc
    delay_minutes: int = 0
    gate: str = ""
    terminal: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── TripIt ingestion ──────────────────────────────────────────────────


_FLIGHT_RE = re.compile(
    r"\b([A-Z]{2,3})\s*([0-9]{1,4})\b",  # IATA airline + flight num
)


def _detect_segment_kind(summary: str, location: str) -> tuple[str, str]:
    """Best-effort kind detection over the iCal SUMMARY."""
    s = (summary or "").lower()
    loc = (location or "").lower()
    if any(k in s for k in ("flight", " to ", "✈", "fly")):
        # Extract first flight-number-shaped token
        m = _FLIGHT_RE.search(summary or "")
        flight = (m.group(1) + m.group(2)) if m else ""
        return "flight", flight
    if any(k in s for k in ("ferry", "tallink", "viking line", "eckero", "eckerö", "stena")):
        return "ferry", ""
    if any(k in s for k in ("train", "rail", "vr ", "intercity")):
        return "train", ""
    if any(k in s for k in ("hotel", "hostel", "airbnb", "lodging", "check-in", "check in")):
        return "hotel", ""
    if any(k in s for k in ("car rental", "rental car", "hertz", "avis", "europcar")):
        return "car", ""
    return "other", ""


def parse_ical(ics_text: str) -> list[TripSegment]:
    """Parse a TripIt iCal body. Returns segments sorted by start time.

    Hand-rolled minimal parser (no icalendar dep) — TripIt's output is
    well-formed VEVENT blocks. We pluck SUMMARY / LOCATION / DTSTART /
    DTEND / UID per event.
    """
    segments: list[TripSegment] = []
    if not ics_text:
        return segments

    in_event = False
    event: dict[str, str] = {}
    # iCal can use line-continuation (a leading space/tab continues
    # the previous line). Unfold first.
    folded = []
    for raw in ics_text.splitlines():
        if not raw:
            continue
        if raw.startswith((" ", "\t")) and folded:
            folded[-1] += raw[1:]
        else:
            folded.append(raw)

    for line in folded:
        if line == "BEGIN:VEVENT":
            in_event = True
            event = {}
            continue
        if line == "END:VEVENT":
            in_event = False
            if "DTSTART" in event and "DTEND" in event:
                summary = event.get("SUMMARY", "").replace("\\n", " ").strip()
                location = event.get("LOCATION", "").replace("\\n", " ").strip()
                kind, flight_number = _detect_segment_kind(summary, location)
                segments.append(TripSegment(
                    summary=summary,
                    location=location,
                    starts_at=_normalize_dt(event["DTSTART"]),
                    ends_at=_normalize_dt(event["DTEND"]),
                    uid=event.get("UID", ""),
                    kind=kind,
                    flight_number=flight_number,
                ))
            continue
        if not in_event:
            continue
        # Property line: ``NAME[;params]:value``
        sep = line.find(":")
        if sep < 0:
            continue
        head, value = line[:sep], line[sep + 1:]
        prop = head.split(";", 1)[0].upper()
        if prop in ("SUMMARY", "LOCATION", "UID"):
            event[prop] = value
        elif prop == "DTSTART":
            event["DTSTART"] = value
        elif prop == "DTEND":
            event["DTEND"] = value

    segments.sort(key=lambda s: s.starts_at)
    return segments


def _normalize_dt(raw: str) -> str:
    """Convert iCal datetime forms to ISO 8601 UTC.

    Forms handled:
      ``20260520T140000Z`` → ``2026-05-20T14:00:00+00:00``
      ``20260520T140000`` (floating) → assume UTC
      ``20260520`` (date-only) → midnight UTC
    """
    if not raw:
        return ""
    raw = raw.strip()
    try:
        if raw.endswith("Z") and len(raw) == 16:
            dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc,
            )
        elif "T" in raw and len(raw) == 15:
            dt = datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc,
            )
        elif len(raw) == 8:
            dt = datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
        else:
            return raw  # unknown form — pass through
        return dt.isoformat()
    except (ValueError, TypeError):
        return raw


def _get_tripit_url() -> str:
    """Resolve the TripIt iCal feed URL.

    Resolution order:
      1. ``runtime_settings.get_tripit_ical_url()`` — operator-flippable
         via React /cp/settings → Travel card. Persists in
         ``workspace/runtime_settings.json``; no gateway restart needed.
      2. ``TRIPIT_ICAL_URL`` env var — backward-compat for the original
         shape; still works for operators on the env-var path.
    """
    try:
        from app.runtime_settings import get_tripit_ical_url
        url = (get_tripit_ical_url() or "").strip()
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("TRIPIT_ICAL_URL", "").strip()


def _get_aviationstack_key() -> str:
    """Resolve the Aviationstack key — runtime_settings first, env var
    fallback (mirrors :func:`_get_tripit_url`)."""
    try:
        from app.runtime_settings import get_aviationstack_api_key
        key = (get_aviationstack_api_key() or "").strip()
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("AVIATIONSTACK_API_KEY", "").strip()


def fetch_tripit() -> list[TripSegment]:
    """Fetch + parse the TripIt iCal feed. Returns [] when no URL
    configured or fetch fails (failure-isolated)."""
    url = _get_tripit_url()
    if not url:
        return []
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "AndrusAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return parse_ical(body)
    except (urllib.error.URLError, OSError, ValueError):
        logger.debug("travel: TripIt fetch failed", exc_info=True)
        return []


# ── Aviationstack flight status ───────────────────────────────────────


def fetch_flight_status(flight_number: str) -> FlightStatus | None:
    """Hit Aviationstack /v1/flights for one flight number. Returns
    None when the API key is absent OR the call fails OR no result.

    Free tier is HTTPS-only (paid is HTTP-only ironically; we go
    HTTPS). 100 calls/month — caller must avoid retrying.
    """
    if not _flight_tracking_enabled():
        return None
    key = _get_aviationstack_key()
    if not key or not flight_number:
        return None
    try:
        # Aviationstack expects IATA flight number (e.g. AY123)
        url = (
            f"https://api.aviationstack.com/v1/flights"
            f"?access_key={key}&flight_iata={flight_number}&limit=1"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "AndrusAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        logger.debug("travel: aviationstack fetch failed", exc_info=True)
        return None
    data = (payload or {}).get("data") or []
    if not data:
        return None
    row = data[0]
    dep = row.get("departure") or {}
    arr = row.get("arrival") or {}
    delay = dep.get("delay") or 0
    try:
        delay = int(delay)
    except (TypeError, ValueError):
        delay = 0
    return FlightStatus(
        flight_number=flight_number,
        departure_iata=str(dep.get("iata") or ""),
        arrival_iata=str(arr.get("iata") or ""),
        status=str(row.get("flight_status") or ""),
        delay_minutes=delay,
        gate=str(dep.get("gate") or ""),
        terminal=str(dep.get("terminal") or ""),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Public API ────────────────────────────────────────────────────────


def upcoming_trips(*, window_days: int = _DEFAULT_WINDOW_DAYS) -> list[TripSegment]:
    """Return segments whose start time is in the future and within
    ``window_days``. Source is the on-disk snapshot — caller does NOT
    block on the network."""
    snap = _data_dir() / _TRIPS_FILE_NAME
    if not snap.exists():
        return []
    try:
        raw = json.loads(snap.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=window_days)
    out: list[TripSegment] = []
    for r in raw:
        try:
            seg = TripSegment(**r)
        except TypeError:
            continue
        try:
            starts = datetime.fromisoformat(seg.starts_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if now <= starts <= end:
            out.append(seg)
    return out


def imminent_flight() -> tuple[TripSegment, FlightStatus | None] | None:
    """Return the next departing flight within 24 h + its status (or
    None if no flight is imminent)."""
    soon = now_plus(hours=24)
    for seg in upcoming_trips(window_days=1):
        if seg.kind != "flight" or not seg.flight_number:
            continue
        try:
            starts = datetime.fromisoformat(
                seg.starts_at.replace("Z", "+00:00"),
            )
        except ValueError:
            continue
        if starts > soon:
            break
        status = _read_flight_status(seg.flight_number)
        return seg, status
    return None


def now_plus(*, hours: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def _read_flight_status(flight_number: str) -> FlightStatus | None:
    snap = _data_dir() / _FLIGHTS_FILE_NAME
    if not snap.exists():
        return None
    try:
        rows = json.loads(snap.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(rows, dict):
        return None
    row = rows.get(flight_number)
    if not row:
        return None
    try:
        return FlightStatus(**row)
    except TypeError:
        return None


def is_traveling_at(dt_iso: str) -> tuple[bool, str]:
    """Return (True, location) when ``dt_iso`` falls inside ANY upcoming
    segment's window. Used by calendar_prep to flag out-of-city events.
    """
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False, ""
    for seg in upcoming_trips(window_days=_DEFAULT_WINDOW_DAYS):
        try:
            s = datetime.fromisoformat(seg.starts_at.replace("Z", "+00:00"))
            e = datetime.fromisoformat(seg.ends_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if s <= dt <= e:
            return True, seg.location or seg.summary
    return False, ""


# ── Idle-job runner ───────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One refresh cycle: pull TripIt, refresh flight status for the
    next 24 h's flights, persist snapshots.

    Cadence-checked internally so a chatty idle scheduler doesn't
    burn the Aviationstack quota.
    """
    if not _enabled():
        return {"status": "skipped_disabled"}

    state_path = _data_dir() / _STATE_FILE_NAME
    try:
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists() else {}
        )
    except Exception:
        state = {}

    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return {"status": "skipped_cadence"}

    state["last_run_at"] = now_ts

    # 1. TripIt — only if URL configured
    segments = fetch_tripit()
    if segments:
        snap = _data_dir() / _TRIPS_FILE_NAME
        try:
            tmp = snap.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps([s.to_dict() for s in segments], indent=2),
                encoding="utf-8",
            )
            tmp.replace(snap)
        except OSError:
            logger.debug("travel: trips snapshot write failed", exc_info=True)

    # 2. Flight status — only for flights departing in the next 24h.
    flight_status_map: dict[str, dict] = {}
    flights_path = _data_dir() / _FLIGHTS_FILE_NAME
    if flights_path.exists():
        try:
            existing = json.loads(flights_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                flight_status_map.update(existing)
        except Exception:
            pass

    imminent_flights = [
        s for s in upcoming_trips(window_days=2)
        if s.kind == "flight" and s.flight_number
    ]
    for seg in imminent_flights[:3]:  # cap at 3/cycle to spare the quota
        status = fetch_flight_status(seg.flight_number)
        if status is not None:
            flight_status_map[seg.flight_number] = status.to_dict()

    if flight_status_map:
        try:
            tmp = flights_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(flight_status_map, indent=2),
                encoding="utf-8",
            )
            tmp.replace(flights_path)
        except OSError:
            logger.debug("travel: flight snapshot write failed", exc_info=True)

    # Persist state
    try:
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(state_path)
    except OSError:
        pass

    return {
        "status": "ok",
        "segments_seen": len(segments),
        "flights_refreshed": len(flight_status_map),
    }


# ── Daily-briefing helper ─────────────────────────────────────────────


def format_for_briefing(*, window_days: int = _DEFAULT_WINDOW_DAYS) -> str:
    """Render upcoming trips as a markdown block for the daily
    briefing. Empty string when no trips (briefing skips the section)."""
    segs = upcoming_trips(window_days=window_days)
    if not segs:
        return ""
    lines = [f"### 🛫 Travel (next {window_days} days)"]
    for seg in segs[:8]:
        try:
            dt = datetime.fromisoformat(seg.starts_at.replace("Z", "+00:00"))
            when = dt.strftime("%a %b %d %H:%M UTC")
        except Exception:
            when = seg.starts_at[:16]
        icon = {
            "flight": "✈️", "ferry": "⛴", "train": "🚆",
            "hotel": "🏨", "car": "🚗",
        }.get(seg.kind, "🧳")
        line = f"  {icon} {when} — {seg.summary[:80]}"
        if seg.location:
            line += f" ({seg.location[:40]})"
        if seg.kind == "flight" and seg.flight_number:
            status = _read_flight_status(seg.flight_number)
            if status is not None:
                badge = status.status or "scheduled"
                if status.delay_minutes:
                    badge += f", delay {status.delay_minutes}m"
                if status.gate:
                    badge += f", gate {status.gate}"
                line += f"  [{badge}]"
        lines.append(line)
    return "\n".join(lines)
