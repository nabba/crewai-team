"""
Request path — the synchronous Signal → Commander → Crews → Response pipeline.

This package provides a clean boundary around the request-handling control plane.
User messages arrive via Signal, get routed by Commander, dispatched to specialist
crews, vetted, and delivered back via Signal.

Components:
  - Commander: orchestrates routing + crew dispatch + response assembly
  - Crews: specialist agents (research, coding, writing, media)
  - Vetting: output quality/safety review before delivery
"""

from app.agents.commander import Commander
from app.vetting import vet_response

__all__ = ["Commander", "vet_response"]
