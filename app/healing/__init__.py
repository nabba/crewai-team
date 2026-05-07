"""
app.healing — unified self-healing surface.

Three reactive surfaces, one package:

    error_diagnosis     per-exception remediation (was: app.self_heal)
    health_remediator   aggregate-health remediation (was: app.self_healer, IMMUTABLE)
    runbooks            anomaly-pattern remediation (operator-authored, IMMUTABLE)

Distinct from `app.self_improvement`, which is the proactive learning loop
(Gap Detector → Novelty → Learner → Integrator → Evaluator → Consolidator).
"""

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
