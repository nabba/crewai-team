"""Architecture-request primitive — operator-gated subsystem proposals.

Where :mod:`app.change_requests` mediates agent-proposed *file* writes,
this module mediates agent-proposed *subsystem* additions. An agent
that recognises a class of capability the operator has not yet
designed proposes a package layout + integration sites + env switches
+ test plan; the operator approves the design at package granularity;
implementation then flows through the existing per-file change-request
gate.

Public surface:

  models           — :class:`ArchitectureRequest` + lifecycle states
  validator        — TIER_IMMUTABLE / package-layout / env-switch checks
  store            — per-record JSON + rolled-segment hash-chained audit
  lifecycle        — state-transition entry points
  scaffolder       — generates stub package skeleton in a staging area

Out of scope here (deferred to follow-up commits):
  - Signal reaction handler (mirrors change_requests/signal.py)
  - FastAPI surface under /api/cp/architecture-requests
  - React /cp/architecture-requests page
  - Auto-rollback monitor

Master switch: ``ARCHITECTURE_REQUESTS_ENABLED`` (default ``false``).
"""

from app.architecture_requests.models import (
    ArchitectureRequest,
    ArchStatus,
    DecisionSource,
    FileSpec,
    IntegrationPoint,
    VALID_INTEGRATION_KINDS,
)
from app.architecture_requests.validator import ValidationResult, validate

__all__ = [
    "ArchitectureRequest",
    "ArchStatus",
    "DecisionSource",
    "FileSpec",
    "IntegrationPoint",
    "VALID_INTEGRATION_KINDS",
    "ValidationResult",
    "validate",
]
