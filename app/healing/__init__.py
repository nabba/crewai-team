"""
app.healing — unified self-healing surface.

Three reactive surfaces, one package:

    error_diagnosis     per-exception remediation (was: app.self_heal)
    health_remediator   aggregate-health remediation (was: app.self_healer, IMMUTABLE)
    runbooks            anomaly-pattern remediation (operator-authored, IMMUTABLE)

Plus, since 2026-05-09:

    handlers            operational runbook handlers (eager-registered on import)
    monitors            proactive observability monitors (daemon thread)
    auditor_bridge      audit_journal → Signal (turns silent diagnoses into action)

Distinct from `app.self_improvement`, which is the proactive learning loop
(Gap Detector → Novelty → Learner → Integrator → Evaluator → Consolidator).
"""
import logging as _logging

from app.healing.error_diagnosis import (
    diagnose_and_fix,
    log_error,
    get_recent_errors,
    get_error_patterns,
)
from app.healing.health_remediator import SelfHealer
from app.healing.runbooks import (
    RunbookResult,
    maybe_run_runbook,
    register_runbook,
    runbooks_enabled,
    unregister_runbook,
)

# ── Eager wiring (2026-05-09) ─────────────────────────────────────────────
# Importing the submodules below triggers registration / daemon-start side
# effects. Each is wrapped so a single broken module never prevents the
# others from loading. The runbook framework already gates dispatch behind
# ``ERROR_RUNBOOKS_ENABLED``; monitors gate behind ``HEALING_MONITORS_ENABLED``;
# the auditor bridge gates behind ``HEALING_AUDITOR_BRIDGE_ENABLED``.

_log = _logging.getLogger(__name__)

try:
    from app.healing import handlers as _handlers  # noqa: F401  — side-effect import
except Exception:
    _log.warning("app.healing: handlers wiring failed", exc_info=True)

try:
    from app.healing import monitors as _monitors  # noqa: F401  — daemon start
except Exception:
    _log.warning("app.healing: monitors wiring failed", exc_info=True)

try:
    from app.healing import auditor_bridge as _auditor_bridge  # noqa: F401
except Exception:
    _log.warning("app.healing: auditor_bridge wiring failed", exc_info=True)

# Watchdog (Wave 2 #7, 2026-05-09) — re-spawns the daemons above when
# they die. Started LAST so the daemons it watches are already running
# at first reaper pass.
try:
    from app.healing import watchdog as _watchdog  # noqa: F401
except Exception:
    _log.warning("app.healing: watchdog wiring failed", exc_info=True)

__all__ = [
    "diagnose_and_fix",
    "log_error",
    "get_recent_errors",
    "get_error_patterns",
    "SelfHealer",
    "RunbookResult",
    "maybe_run_runbook",
    "register_runbook",
    "runbooks_enabled",
    "unregister_runbook",
]
