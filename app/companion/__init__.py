"""Companion Layer — per-workspace idle-time contemplation.

Phase 1 ships the loop skeleton (fairness scheduler + cost ledger + idle-job
self-registration). The cycle itself wires to Creative MAS in Phase 2.

Submodules:
  config    — durable per-workspace settings (read from CP projects.config_json).
  state     — runtime sidecar (vruntime, last_tick_at, daily cost).
  budget    — daily cost ledger with hard per-workspace caps.
  scheduler — CFS-style fairness selector with 12 h temporal floor.
  loop      — companion_tick + get_idle_jobs() for idle-scheduler registration.
"""

from app.companion.loop import get_idle_jobs, companion_tick

__all__ = ["get_idle_jobs", "companion_tick"]
