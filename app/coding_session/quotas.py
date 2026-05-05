"""Quota policy for coding sessions.

Pure logic — no I/O, no global state. The manager wires this up with
counters from the store + filesystem stat.

Defaults match `docs/CODING_SESSIONS.md` § 5. The values can be
overridden per-deployment via env vars (see ``QuotaConfig.from_env``)
without touching code.

The quota check returns a structured result so the agent gets a
specific reason on refusal — useful both for logging and for the
agent's own retry logic.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────


@dataclass(frozen=True)
class QuotaConfig:
    """All numeric limits in one place. Frozen so it's safe to share."""

    per_agent_active: int = 3
    system_active: int = 20
    per_session_disk_bytes: int = 100 * 1024 * 1024          # 100 MB
    system_disk_bytes: int = 5 * 1024 * 1024 * 1024          # 5 GB
    ttl_seconds: int = 30 * 60                               # 30 min hard
    idle_seconds: int = 10 * 60                              # 10 min idle
    run_wallclock_default_s: int = 120                       # default per-run
    run_wallclock_max_s: int = 600                           # ceiling per-run

    @classmethod
    def from_env(cls) -> "QuotaConfig":
        """Build a QuotaConfig from CODING_SESSION_* env vars; falls back
        to dataclass defaults for any var that isn't set or doesn't
        parse as a positive int."""
        return cls(
            per_agent_active=_int_env(
                "CODING_SESSION_PER_AGENT_ACTIVE", cls.per_agent_active,
            ),
            system_active=_int_env(
                "CODING_SESSION_SYSTEM_ACTIVE", cls.system_active,
            ),
            per_session_disk_bytes=_int_env(
                "CODING_SESSION_PER_SESSION_DISK_BYTES",
                cls.per_session_disk_bytes,
            ),
            system_disk_bytes=_int_env(
                "CODING_SESSION_SYSTEM_DISK_BYTES", cls.system_disk_bytes,
            ),
            ttl_seconds=_int_env(
                "CODING_SESSION_TTL_SECONDS", cls.ttl_seconds,
            ),
            idle_seconds=_int_env(
                "CODING_SESSION_IDLE_SECONDS", cls.idle_seconds,
            ),
            run_wallclock_default_s=_int_env(
                "CODING_SESSION_RUN_WALLCLOCK_DEFAULT_S",
                cls.run_wallclock_default_s,
            ),
            run_wallclock_max_s=_int_env(
                "CODING_SESSION_RUN_WALLCLOCK_MAX_S",
                cls.run_wallclock_max_s,
            ),
        )


def _int_env(key: str, default: int) -> int:
    """Parse a positive int env var; fall back to default on
    missing / invalid."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        v = int(raw)
        if v <= 0:
            return default
        return v
    except ValueError:
        logger.warning("quotas: %s=%r is not int; using default %d", key, raw, default)
        return default


# Module-level default. Cheap to construct; tests inject their own.
DEFAULTS = QuotaConfig()


# ── Result ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QuotaResult:
    ok: bool
    reason: str | None = None  # short, agent-friendly explanation

    @classmethod
    def allow(cls) -> "QuotaResult":
        return cls(ok=True, reason=None)

    @classmethod
    def deny(cls, reason: str) -> "QuotaResult":
        return cls(ok=False, reason=reason)


# ── Checks ──────────────────────────────────────────────────────────


def can_start_session(
    *,
    config: QuotaConfig,
    agent_active_count: int,
    system_active_count: int,
) -> QuotaResult:
    """Check whether a fresh session can be started right now."""
    if agent_active_count >= config.per_agent_active:
        return QuotaResult.deny(
            f"per-agent active session quota reached "
            f"({agent_active_count} ≥ {config.per_agent_active}). "
            f"Discard or submit an existing session first."
        )
    if system_active_count >= config.system_active:
        return QuotaResult.deny(
            f"system-wide active session quota reached "
            f"({system_active_count} ≥ {config.system_active}). "
            f"Wait for an existing session to finish."
        )
    return QuotaResult.allow()


def can_write_bytes(
    *,
    config: QuotaConfig,
    session_bytes_after_write: int,
    system_bytes_after_write: int,
) -> QuotaResult:
    """Check whether a write of N bytes can land. Caller computes the
    after-write totals (it knows the current size + the new content)."""
    if session_bytes_after_write > config.per_session_disk_bytes:
        return QuotaResult.deny(
            f"per-session disk quota would exceed "
            f"{config.per_session_disk_bytes} bytes "
            f"(would be {session_bytes_after_write})."
        )
    if system_bytes_after_write > config.system_disk_bytes:
        return QuotaResult.deny(
            f"total worktree-disk quota would exceed "
            f"{config.system_disk_bytes} bytes "
            f"(would be {system_bytes_after_write})."
        )
    return QuotaResult.allow()


def cap_run_timeout(*, config: QuotaConfig, requested_s: int | None) -> int:
    """Clamp a requested run timeout to (0, max_s]. None → default."""
    if requested_s is None:
        return config.run_wallclock_default_s
    if requested_s <= 0:
        return config.run_wallclock_default_s
    return min(requested_s, config.run_wallclock_max_s)
