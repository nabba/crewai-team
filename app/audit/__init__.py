"""Append-only audit storage with hash-chain integrity.

The :mod:`rolled_log` submodule exposes ``RolledLogStore``,
``RolledLogReader``, ``RolledLogVerifier`` — a decade-class durable
storage primitive for JSONL audit logs with size-triggered rotation
and tamper-evident hash chains across segment boundaries.

The :mod:`migration` submodule converts legacy single-file logs
(JSON list-of-dicts or plain JSONL) into rolled-segment form,
preserving forensic continuity via a sentinel boundary entry.
"""

from app.audit.rolled_log import (
    GENESIS,
    LEGACY_PREFIX,
    RolledLogReader,
    RolledLogStore,
    RolledLogVerifier,
    SegmentInfo,
    VerificationResult,
)

__all__ = [
    "GENESIS",
    "LEGACY_PREFIX",
    "RolledLogReader",
    "RolledLogStore",
    "RolledLogVerifier",
    "SegmentInfo",
    "VerificationResult",
]
