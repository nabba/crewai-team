"""self_model — positive self-knowledge layer (Q17.5).

Counterweight to Goodhart guard. agreement_ledger records every
proactive suggestion + operator response (accept/reject/ignore/
defer). Surfaces in daily briefing.
"""
from __future__ import annotations

from app.self_model.agreement_ledger import (
    AgreementResponse,
    record_response,
    record_suggestion,
    rolling_rate,
    summary_for_briefing,
)

__all__ = [
    "AgreementResponse",
    "record_response",
    "record_suggestion",
    "rolling_rate",
    "summary_for_briefing",
]
