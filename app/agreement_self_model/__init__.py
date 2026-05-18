"""agreement_self_model — positive self-knowledge layer (Q17.5).

Counterweight to Goodhart guard. agreement_ledger records every
proactive suggestion + operator response (accept/reject/ignore/
defer). Surfaces in daily briefing.

Renamed from ``app/self_model`` to break a shadow over the older
``app/self_model.py`` module (static dep-graph + capability map).
The on-disk data path (``workspace/self_model/agreement_ledger.jsonl``)
stays stable so prior recordings remain readable.
"""
from __future__ import annotations

from app.agreement_self_model.agreement_ledger import (
    AgreementResponse,
    briefing_section,
    record_response,
    record_suggestion,
    rolling_rate,
    summary_for_briefing,
)

__all__ = [
    "AgreementResponse",
    "briefing_section",
    "record_response",
    "record_suggestion",
    "rolling_rate",
    "summary_for_briefing",
]
