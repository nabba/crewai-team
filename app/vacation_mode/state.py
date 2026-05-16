"""Vacation-mode state — operator-managed engagement + allowlist staging.

State lives in ``runtime_settings.vacation_mode_state`` (a JSON blob),
not in module constants. This is deliberate: the operator should be
able to flip vacation mode without a deploy, while the rest of the
auto-apply infrastructure (which has security-critical defaults baked
into source) stays compile-time.

Invariants
==========

  * **Staging is disjoint from engagement.** ``stage_allowlist`` is
    refused while engaged; ``engage`` is refused while not staged.
    This prevents "engage → expand allowlist → broaden auto-apply"
    paths.
  * **Engagement freezes the allowlist.** ``engage`` snapshots the
    current staged allowlist into the engagement state. Subsequent
    staging changes do NOT affect the in-flight engagement.
  * **Time-bounded.** Hard cap of 30 days per engagement. The sweep
    auto-disengages when ``until_ts`` passes.
  * **Identity-traced.** ``engaged_by`` records who engaged. Operator
    can audit later.

State shape (persisted JSON)
============================

::

    {
      "staged_allowlist": {
        "requestor_allowlist": ["agent_name_1", ...],
        "path_prefix_allowlist": ["app/companion/wiki/", ...],
        "max_diff_lines": 10
      },
      "engaged": false,
      "engagement": null  // or {engaged_at, until_ts, engaged_by,
                          //     reason, frozen_allowlist}
    }
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


MAX_DURATION_DAYS = 30
_DEFAULT_MAX_DIFF_LINES = 10
_MAX_REASON_CHARS = 200
_STATE_KEY = "vacation_mode_state"


class VacationModeError(Exception):
    """Raised for invalid operator-callable transitions."""


@dataclass
class VacationAllowlist:
    """What kinds of CRs can auto-apply during vacation."""
    requestor_allowlist: list[str] = field(default_factory=list)
    path_prefix_allowlist: list[str] = field(default_factory=list)
    max_diff_lines: int = _DEFAULT_MAX_DIFF_LINES

    def to_dict(self) -> dict[str, Any]:
        return {
            "requestor_allowlist": list(self.requestor_allowlist),
            "path_prefix_allowlist": list(self.path_prefix_allowlist),
            "max_diff_lines": int(self.max_diff_lines),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VacationAllowlist":
        return cls(
            requestor_allowlist=list(d.get("requestor_allowlist", []) or []),
            path_prefix_allowlist=list(d.get("path_prefix_allowlist", []) or []),
            max_diff_lines=int(d.get("max_diff_lines", _DEFAULT_MAX_DIFF_LINES)),
        )

    def is_empty(self) -> bool:
        return not (self.requestor_allowlist and self.path_prefix_allowlist)


@dataclass
class VacationEngagement:
    """A live engagement window. Allowlist is frozen at engagement
    time and never mutates during the window."""
    engaged_at: float
    until_ts: float
    engaged_by: str
    reason: str
    frozen_allowlist: VacationAllowlist

    def to_dict(self) -> dict[str, Any]:
        return {
            "engaged_at": float(self.engaged_at),
            "until_ts": float(self.until_ts),
            "engaged_by": str(self.engaged_by),
            "reason": str(self.reason),
            "frozen_allowlist": self.frozen_allowlist.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VacationEngagement":
        return cls(
            engaged_at=float(d["engaged_at"]),
            until_ts=float(d["until_ts"]),
            engaged_by=str(d.get("engaged_by", "unknown")),
            reason=str(d.get("reason", "")),
            frozen_allowlist=VacationAllowlist.from_dict(
                d.get("frozen_allowlist", {}) or {},
            ),
        )


@dataclass
class VacationState:
    """Top-level state stored in runtime_settings."""
    staged_allowlist: VacationAllowlist = field(default_factory=VacationAllowlist)
    engaged: bool = False
    engagement: Optional[VacationEngagement] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "staged_allowlist": self.staged_allowlist.to_dict(),
            "engaged": bool(self.engaged),
            "engagement": (
                self.engagement.to_dict() if self.engagement else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VacationState":
        return cls(
            staged_allowlist=VacationAllowlist.from_dict(
                d.get("staged_allowlist", {}) or {},
            ),
            engaged=bool(d.get("engaged", False)),
            engagement=(
                VacationEngagement.from_dict(d["engagement"])
                if d.get("engagement")
                else None
            ),
        )


# ── Persistence helpers ──────────────────────────────────────────────────


def _read_blob() -> dict[str, Any]:
    """Read the raw state blob from runtime_settings. Returns an empty
    dict on any error so callers can fall through to defaults."""
    try:
        from app.runtime_settings import _ensure_initialized
        return dict(_ensure_initialized().get(_STATE_KEY, {}) or {})
    except Exception:
        logger.debug("vacation_mode: read failed", exc_info=True)
        return {}


def _write_blob(blob: dict[str, Any]) -> None:
    """Persist the state blob to runtime_settings."""
    try:
        from app.runtime_settings import _update
        _update({_STATE_KEY: blob})
    except Exception:
        logger.warning("vacation_mode: write failed", exc_info=True)


def current_state() -> VacationState:
    """Return the current state (always usable, falls back to defaults
    on any persistence failure). Auto-expires a stale engagement.

    Auto-expiry routes through :func:`disengage` with
    ``auto_expired=True`` so the continuity-ledger event lands and
    the end-of-vacation digest gets composed. The call is reentrancy-
    guarded via a module-level flag so the recursive ``current_state``
    inside ``disengage`` doesn't loop."""
    global _IN_AUTO_EXPIRY
    state = VacationState.from_dict(_read_blob())
    if not state.engaged or state.engagement is None:
        return state
    if time.time() < state.engagement.until_ts:
        return state
    # Past until_ts — auto-expire.
    if _IN_AUTO_EXPIRY:
        # Reentrancy guard: ``disengage`` re-calls ``current_state``.
        # Return the not-yet-persisted state to break the loop.
        state.engaged = False
        state.engagement = None
        return state
    _IN_AUTO_EXPIRY = True
    try:
        logger.info("vacation_mode: auto-expiring stale engagement")
        return disengage(disengaged_by="auto_expire", auto_expired=True)
    finally:
        _IN_AUTO_EXPIRY = False


_IN_AUTO_EXPIRY = False


def is_active() -> bool:
    """True iff vacation mode is currently engaged and not auto-expired."""
    return current_state().engaged


def current_allowlist() -> VacationAllowlist:
    """The allowlist that applies right now.

    * If engaged: the FROZEN allowlist from engagement (immutable for
      the duration of the window).
    * If not engaged: the staged allowlist (operator-mutable).
    """
    state = current_state()
    if state.engaged and state.engagement is not None:
        return state.engagement.frozen_allowlist
    return state.staged_allowlist


# ── Operator-callable transitions ────────────────────────────────────────


def stage_allowlist(
    *,
    requestor_allowlist: list[str],
    path_prefix_allowlist: list[str],
    max_diff_lines: int = _DEFAULT_MAX_DIFF_LINES,
) -> VacationAllowlist:
    """Stage (or restage) the allowlist. Refused while engaged.

    Validates the inputs:
      * ``max_diff_lines`` ∈ [1, 50] — even §38.3 caps at 20; we go
        narrower by default but allow up to 50 if the operator wants
        slightly more headroom for, say, a wiki-update flurry.
      * Each path prefix must end with ``/`` to enforce the
        prefix-matching semantic (no whole-file allowlisting via
        bare paths — too easy to add an over-broad entry).
      * No path may match ``app/`` or ``tests/`` exactly (too broad)
        or be empty. At least one ``/`` past the root.
    """
    state = current_state()
    if state.engaged:
        raise VacationModeError(
            "cannot stage allowlist while vacation mode is engaged — "
            "disengage first, then stage, then re-engage"
        )
    if not isinstance(requestor_allowlist, list) or not all(
        isinstance(r, str) for r in requestor_allowlist
    ):
        raise VacationModeError("requestor_allowlist must be a list of str")
    if not isinstance(path_prefix_allowlist, list) or not all(
        isinstance(p, str) for p in path_prefix_allowlist
    ):
        raise VacationModeError("path_prefix_allowlist must be a list of str")
    if not isinstance(max_diff_lines, int) or not 1 <= max_diff_lines <= 50:
        raise VacationModeError("max_diff_lines must be int in [1, 50]")
    # Path-prefix sanity.
    for p in path_prefix_allowlist:
        if not p.endswith("/"):
            raise VacationModeError(
                f"path prefix {p!r} must end with '/' (prefix-match only)"
            )
        if p in ("app/", "tests/", "docs/", "wiki/", "deploy/"):
            raise VacationModeError(
                f"path prefix {p!r} is too broad; specify a subdirectory"
            )
        # Must have at least one slash past the root.
        head, _, _ = p.rstrip("/").rpartition("/")
        if not head:
            raise VacationModeError(
                f"path prefix {p!r} is a single-segment root; "
                f"specify a subdirectory"
            )
    new_allowlist = VacationAllowlist(
        requestor_allowlist=list(requestor_allowlist),
        path_prefix_allowlist=list(path_prefix_allowlist),
        max_diff_lines=max_diff_lines,
    )
    state.staged_allowlist = new_allowlist
    _write_blob(state.to_dict())
    logger.info(
        "vacation_mode: allowlist staged (requestors=%d, paths=%d, lines<=%d)",
        len(requestor_allowlist), len(path_prefix_allowlist), max_diff_lines,
    )
    return new_allowlist


def engage(
    *,
    until_ts: float,
    engaged_by: str,
    reason: str = "",
    now: Optional[float] = None,
) -> VacationEngagement:
    """Engage vacation mode. Refuses if:
      * already engaged
      * staged allowlist is empty (no requestor or path allowlisted)
      * ``until_ts`` <= now
      * duration > MAX_DURATION_DAYS
    """
    cur = float(now) if now is not None else time.time()
    state = current_state()
    if state.engaged:
        raise VacationModeError("vacation mode is already engaged")
    if state.staged_allowlist.is_empty():
        raise VacationModeError(
            "staged allowlist is empty; stage non-empty requestor + path "
            "lists before engaging"
        )
    if until_ts <= cur:
        raise VacationModeError("until_ts must be in the future")
    duration_days = (until_ts - cur) / 86400.0
    if duration_days > MAX_DURATION_DAYS:
        raise VacationModeError(
            f"duration {duration_days:.1f}d exceeds maximum "
            f"{MAX_DURATION_DAYS}d"
        )
    if not isinstance(engaged_by, str) or not engaged_by.strip():
        raise VacationModeError("engaged_by must be a non-empty operator id")
    reason = (reason or "").strip()[:_MAX_REASON_CHARS]
    engagement = VacationEngagement(
        engaged_at=cur,
        until_ts=float(until_ts),
        engaged_by=engaged_by.strip(),
        reason=reason,
        frozen_allowlist=VacationAllowlist.from_dict(
            state.staged_allowlist.to_dict()  # deep copy
        ),
    )
    state.engaged = True
    state.engagement = engagement
    _write_blob(state.to_dict())
    logger.warning(
        "vacation_mode: ENGAGED by %s until %s (%.1f days) — reason=%r",
        engagement.engaged_by,
        time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(engagement.until_ts)),
        duration_days,
        engagement.reason,
    )
    # PROGRAM §51 Q16 — emit continuity-ledger event so annual
    # reflection picks up vacation windows. Failure-isolated.
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="vacation_window",
            actor=engagement.engaged_by or "operator",
            summary=(
                f"vacation window engaged for {duration_days:.1f} day(s) — "
                f"{engagement.reason[:80] or 'no reason provided'}"
            ),
            detail={
                "event": "engage",
                "engaged_at": engagement.engaged_at,
                "until_ts": engagement.until_ts,
                "duration_days": round(duration_days, 2),
                "allowlist": engagement.frozen_allowlist.to_dict(),
            },
        )
    except Exception:
        logger.debug(
            "vacation_mode: ledger emit on engage failed", exc_info=True,
        )
    return engagement


def disengage(
    *,
    disengaged_by: str = "operator",
    auto_expired: bool = False,
    compose_digest: bool = True,
) -> VacationState:
    """Manually disengage vacation mode. Idempotent (no-op if already
    disengaged). Returns the post-state.

    Side effects when an engagement was live:
      * Emits ``vacation_window`` continuity-ledger event with
        ``event=disengage`` or ``event=auto_expire``.
      * Composes the end-of-vacation digest (unless
        ``compose_digest=False`` — useful for tests).

    Reads the blob directly (NOT via :func:`current_state`) so the
    auto-expiry reentrancy guard never short-circuits the prior-
    engagement capture.
    """
    state = VacationState.from_dict(_read_blob())
    prior_engagement = state.engagement
    if state.engaged:
        logger.warning(
            "vacation_mode: %s by %s",
            "AUTO-EXPIRED" if auto_expired else "DISENGAGED",
            disengaged_by,
        )
        # PROGRAM §51 Q16 — emit continuity-ledger event.
        if prior_engagement is not None:
            try:
                from app.identity.continuity_ledger import record_event
                duration_actual = max(
                    0.0,
                    (time.time() - prior_engagement.engaged_at) / 86400.0,
                )
                record_event(
                    kind="vacation_window",
                    actor=disengaged_by or "operator",
                    summary=(
                        f"vacation window {'auto-expired' if auto_expired else 'disengaged'} "
                        f"after {duration_actual:.2f} day(s) — "
                        f"{prior_engagement.reason[:80] or 'no reason'}"
                    ),
                    detail={
                        "event": "auto_expire" if auto_expired else "disengage",
                        "engaged_at": prior_engagement.engaged_at,
                        "until_ts": prior_engagement.until_ts,
                        "ended_at": time.time(),
                        "duration_days_actual": round(duration_actual, 3),
                    },
                )
            except Exception:
                logger.debug(
                    "vacation_mode: ledger emit on disengage failed",
                    exc_info=True,
                )
            # Compose digest (failure-isolated).
            if compose_digest:
                try:
                    from app.vacation_mode.digest import compose_digest as _compose
                    _compose(engagement=prior_engagement, ended_at=time.time())
                except Exception:
                    logger.debug(
                        "vacation_mode: digest composition failed",
                        exc_info=True,
                    )
    state.engaged = False
    state.engagement = None
    _write_blob(state.to_dict())
    return state
