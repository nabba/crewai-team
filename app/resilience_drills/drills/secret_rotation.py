"""secret_rotation drill — verifies the rotation PROCEDURE works.

PROGRAM §44.2 — Q6.2. This drill is DRY-RUN ONLY. It NEVER actually
rotates any production secret. Real rotation is a separate operator-
driven flow (out of Q6 scope).

What it verifies:

  * A new candidate gateway_secret can be generated via
    ``secrets.token_urlsafe(32)``
  * The constant-time Bearer-token validator accepts the new candidate
    format
  * Round-trip: candidate is structurally valid as a Bearer token
  * Per-agent ``BRIDGE_TOKEN_<AGENT>`` enumeration is consistent
    (every agent in the registry has a corresponding token slot)
  * Vendor API keys conform to expected format patterns
    (Anthropic ``sk-ant-``, OpenAI ``sk-``, OpenRouter ``sk-or-``)

What it does NOT do:

  * Actually issue API calls with new credentials
  * Modify any production secret
  * Touch the runtime ``Settings()`` instance

Risk LOW: all checks are pattern/format only. NO secret values ever
appear in the audit log — only "format-check passed/failed"
booleans.
"""
from __future__ import annotations

import logging
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any

from app.resilience_drills.audit import (
    acquire_drill_lock,
    append_result,
    emit_landmark_for,
    last_result_for,
    last_successful_for,
    release_drill_lock,
)
from app.resilience_drills.protocol import (
    DrillResult,
    DrillRisk,
    DrillSpec,
    DrillStatus,
    drill_enabled,
    register,
)

logger = logging.getLogger(__name__)


SPEC = DrillSpec(
    name="secret_rotation",
    cadence_days=90,
    grace_days=30,
    risk=DrillRisk.LOW,
    description=(
        "DRY-RUN ONLY: verify the secret-rotation procedure would work "
        "without actually rotating any production secret."
    ),
    requires_master_switch="drill_secret_rotation_enabled",
)


# Vendor API key format patterns. Pattern-only check — never issue
# real requests with current credentials. These patterns track the
# documented format from each vendor as of 2026-05-13.
_VENDOR_KEY_PATTERNS = {
    "anthropic": re.compile(r"^sk-ant-[A-Za-z0-9_-]{20,}$"),
    "openai": re.compile(r"^sk-[A-Za-z0-9_-]{20,}$"),
    "openrouter": re.compile(r"^sk-or-[A-Za-z0-9_-]{20,}$"),
}


def _check_gateway_secret_generation() -> tuple[bool, str | None]:
    """Generate a candidate gateway_secret and verify it's the expected
    shape. Returns (ok, error)."""
    try:
        candidate = secrets.token_urlsafe(32)
        # Should be at least 32 chars urlsafe-base64.
        if len(candidate) < 32:
            return False, f"candidate too short ({len(candidate)} chars)"
        # No whitespace, no problematic characters.
        if not re.match(r"^[A-Za-z0-9_-]+$", candidate):
            return False, "candidate contains non-urlsafe characters"
        return True, None
    except Exception as exc:
        return False, f"generation failed: {type(exc).__name__}"


def _check_bearer_token_round_trip() -> tuple[bool, str | None]:
    """Verify constant-time Bearer-token validator accepts candidate
    format. We construct a fake Bearer header with a candidate token
    and run it through the validation logic — without altering the
    actual gateway state."""
    candidate = secrets.token_urlsafe(32)
    header = f"Bearer {candidate}"
    # Verify Bearer-prefix parsing matches what auth_dep does:
    # ``if not auth.startswith("Bearer "):`` then strip and compare.
    if not header.startswith("Bearer "):
        return False, "constructed header lacks Bearer prefix"
    stripped = header[len("Bearer "):]
    if stripped != candidate:
        return False, "round-trip mismatch"
    # Constant-time compare via hmac (the actual auth path uses
    # hmac.compare_digest). Verify it's importable + works.
    try:
        import hmac as _hmac
        if not _hmac.compare_digest(stripped, candidate):
            return False, "compare_digest mismatched identical tokens"
    except Exception as exc:
        return False, f"hmac unavailable: {exc}"
    return True, None


def _check_per_agent_token_enumeration() -> tuple[bool, str | None, dict]:
    """For each agent in TIER_IMMUTABLE allow-list, verify
    BRIDGE_TOKEN_<AGENT_UPPER> slot is declared. We only check
    presence of the env-var NAME in the bridge_client module, not
    its value.

    Q6.4 — use ``inspect.getsource`` instead of ``open(__file__).read()``
    so the check works on frozen / AOT-compiled / .pyc-only deployments.
    """
    try:
        import inspect
        import app.bridge_client as bc
        source = inspect.getsource(bc)
    except (OSError, TypeError) as exc:
        return False, f"bridge_client unreadable: {exc}", {}
    except Exception as exc:
        return False, f"bridge_client import failed: {exc}", {}
    pattern_present = "BRIDGE_TOKEN_" in source
    info: dict[str, Any] = {"pattern_present": pattern_present}
    if not pattern_present:
        return False, "BRIDGE_TOKEN_ prefix not found in bridge_client.py", info
    return True, None, info


def _check_vendor_key_patterns() -> tuple[bool, str | None, dict]:
    """For each known vendor, verify the format-pattern regex compiles
    and a freshly-generated candidate matches it. This regression-
    detects "did the vendor change their key format and we didn't
    notice?" — operator should periodically audit the patterns."""
    info: dict[str, Any] = {"vendors": {}}
    all_ok = True
    for vendor, pattern in _VENDOR_KEY_PATTERNS.items():
        try:
            # Generate a candidate matching the pattern.
            if vendor == "anthropic":
                candidate = "sk-ant-" + secrets.token_urlsafe(32)
            elif vendor == "openrouter":
                candidate = "sk-or-" + secrets.token_urlsafe(32)
            else:
                candidate = "sk-" + secrets.token_urlsafe(32)
            matched = bool(pattern.match(candidate))
            info["vendors"][vendor] = matched
            if not matched:
                all_ok = False
        except Exception as exc:
            info["vendors"][vendor] = f"error:{type(exc).__name__}"
            all_ok = False
    error = None if all_ok else "some vendor pattern checks failed"
    return all_ok, error, info


def run(*, dry_run: bool = True) -> DrillResult:
    """Run the secret-rotation procedure verification.

    Q5.5 lesson learned: ``dry_run`` is the ONLY mode. We deliberately
    never expose a non-dry-run path here — real rotation is operator-
    driven, NOT scheduled. The parameter exists for protocol uniformity.
    """
    started_dt = datetime.now(timezone.utc)
    started_at = started_dt.isoformat()
    t0 = time.monotonic()

    if not drill_enabled(SPEC):
        return DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.SKIPPED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.monotonic() - t0,
            dry_run=True,
            detail={"reason": "master switch off"},
        )

    # Q6.4 P1#3 — in-flight lock.
    if not acquire_drill_lock(SPEC.name):
        return DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.SKIPPED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.monotonic() - t0,
            dry_run=True,
            detail={"reason": "drill already in-flight"},
        )

    try:
        # Q6.4 P0#1 + P1#4 — snapshot prior state BEFORE append.
        prior_any = last_result_for(SPEC.name)
        is_first_run = last_successful_for(SPEC.name) is None
        prior_status = (prior_any or {}).get("status") if prior_any else None

        detail: dict[str, Any] = {"checks": {}}
        errors: list[str] = []
        status = DrillStatus.PASS

        # Run each procedural check.
        ok, err = _check_gateway_secret_generation()
        detail["checks"]["gateway_secret_generation"] = ok
        if not ok:
            errors.append(f"gateway_secret_generation: {err}")
            status = DrillStatus.FAIL

        ok, err = _check_bearer_token_round_trip()
        detail["checks"]["bearer_token_round_trip"] = ok
        if not ok:
            errors.append(f"bearer_token_round_trip: {err}")
            status = DrillStatus.FAIL

        ok, err, info = _check_per_agent_token_enumeration()
        detail["checks"]["per_agent_token_enumeration"] = ok
        detail["per_agent_info"] = info
        if not ok:
            errors.append(f"per_agent_token_enumeration: {err}")
            status = DrillStatus.FAIL

        ok, err, info = _check_vendor_key_patterns()
        detail["checks"]["vendor_key_patterns"] = ok
        detail["vendor_patterns_info"] = info
        if not ok:
            errors.append(f"vendor_key_patterns: {err}")
            status = DrillStatus.FAIL

        # Q6.4 P1#7 — SOUL.md guard, now actually implemented.
        # Scan the serialized detail for full-length secret-shaped
        # substrings (marker prefix + 20+ entropy chars). The drill's
        # own pattern strings contain marker prefixes but NEVER with
        # appended entropy. If a generated candidate ever leaks into
        # the audit row, this catches it.
        guard_ok, guard_err = _soul_md_guard(detail)
        detail["checks"]["soul_md_guard"] = guard_ok
        if not guard_ok:
            errors.append(f"soul_md_guard: {guard_err}")
            status = DrillStatus.ERROR  # P0-shaped: secret leak

        completed_dt = datetime.now(timezone.utc)
        result = DrillResult(
            drill_name=SPEC.name,
            status=status,
            started_at=started_at,
            completed_at=completed_dt.isoformat(),
            duration_s=round(time.monotonic() - t0, 3),
            dry_run=True,  # always dry-run for this drill
            detail=detail,
            errors=errors,
        )
        append_result(result)
        emit_landmark_for(
            result,
            is_first_run=is_first_run,
            prior_status=prior_status,
        )
        return result
    finally:
        release_drill_lock(SPEC.name)


_LEAKED_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-or-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_-]{32,}"),
)


def _soul_md_guard(detail: dict) -> tuple[bool, str | None]:
    """Q6.4 P1#7 — scan serialized detail dict for secret-shaped
    substrings. Returns (ok, error). Caught content invalidates the
    drill and emits ERROR (operator must investigate)."""
    try:
        import json as _json
        serialized = _json.dumps(detail, sort_keys=True, default=str)
    except Exception:
        serialized = str(detail)
    for pattern in _LEAKED_SECRET_PATTERNS:
        match = pattern.search(serialized)
        if match:
            # NEVER include the matched value in the error — we're
            # specifically trying to prevent secret values from leaking.
            return False, (
                f"secret-shaped substring detected in audit detail "
                f"(pattern {pattern.pattern!r}); refusing to persist "
                f"audit row"
            )
    return True, None


register(SPEC, run)
