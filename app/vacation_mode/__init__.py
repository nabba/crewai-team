"""Vacation mode — operator-unavailable autonomy with pre-staged allowlist.

PROGRAM §51 — Q16 Theme 3 (decade-resilience, operator-unavailable
autonomy). The defense lever to complement the ``operator_anomaly``
monitor (the observational piece).

Problem statement
=================

Almost every action in this system requires operator approval (CR
system, Tier-3 amendments, kill-the-gateway typed phrase, etc.). If
the operator is hospitalized for a month, the self-improvement axis
halts. The §38.3 auto-apply infrastructure was shipped dormant
(allowlists deliberately EMPTY per P1-P5 principles); vacation mode
is the staged, time-bounded path to enable a tightly-scoped subset of
auto-application during operator absence.

Design contract
===============

  1. **Time-bounded.** Engagement requires an explicit ``until_ts``.
     Auto-expires. Hard cap: 30 days per engagement.
  2. **Pre-staged.** Allowlist must be staged WHEN DISENGAGED. Adding
     to the allowlist while engaged is refused. (No chicken-and-egg
     "engage → expand allowlist → broaden auto-apply" attack vector.)
  3. **Default OFF.** Master switch + state both default OFF. No
     vacation mode runs unless an operator deliberately engages.
  4. **TIER_IMMUTABLE absolute.** Standard validator runs first;
     TIER_IMMUTABLE paths refused regardless of allowlist.
  5. **Tighter than §38.3 auto-apply.** Default line cap is 10 (vs
     §38.3's 20); requestor + path allowlists are operator-managed
     in runtime_settings rather than module constants.
  6. **Loud audit.** Every auto-apply during vacation emits a Signal
     alert (NOT arbitrated — bypasses suppression) plus an entry in
     ``workspace/vacation_mode/auto_apply_log.jsonl``.
  7. **Anomaly-aware.** Suspends on the operator_anomaly monitor's
     ``new_sender`` critical signal; auto-resumes when the dedup
     window expires (vacation-mode interpretation: "something is
     weird; let me wait until things look normal again").
  8. **Composes with auto-revert.** The existing 60-min rollback
     watcher in ``app/change_requests/auto_revert.py`` activates on
     every vacation-mode-approved CR.

What vacation mode is NOT
=========================

  * NOT a way to disable the operator gate permanently — it expires.
  * NOT a security-relaxation lever for TIER_IMMUTABLE — those are
    refused regardless.
  * NOT a substitute for the §38.3 auto-apply infrastructure — that
    pathway stays as-is; vacation mode is a separate, narrower path.
  * NOT a remote-management tool — the operator must be present to
    ENGAGE (typed-phrase confirmation through Signal or the React UI).

Files
=====

  * ``state.py``     — VacationState + persistence + engage/disengage
  * ``allowlist.py`` — VacationAllowlist + validate_vacation_apply
  * ``sweep.py``     — periodic scan + auto-approve of matching PENDING CRs
  * ``digest.py``    — end-of-vacation summary composer
"""
from __future__ import annotations

from app.vacation_mode.state import (
    VacationState,
    VacationAllowlist,
    is_active,
    current_state,
    current_allowlist,
    engage,
    disengage,
    stage_allowlist,
    VacationModeError,
    MAX_DURATION_DAYS,
)
from app.vacation_mode.allowlist import (
    validate_vacation_apply,
    VacationValidationResult,
)
from app.vacation_mode.sweep import (
    sweep_pending,
    SWEEP_INTERVAL_SECONDS,
    start_daemon,
)
from app.vacation_mode.digest import (
    compose_digest,
    list_digests,
    read_digest,
)

__all__ = [
    "VacationState",
    "VacationAllowlist",
    "VacationValidationResult",
    "VacationModeError",
    "is_active",
    "current_state",
    "current_allowlist",
    "engage",
    "disengage",
    "stage_allowlist",
    "validate_vacation_apply",
    "sweep_pending",
    "start_daemon",
    "compose_digest",
    "list_digests",
    "read_digest",
    "MAX_DURATION_DAYS",
    "SWEEP_INTERVAL_SECONDS",
]
