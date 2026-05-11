"""Path validation for change requests.

The single rule: TIER_IMMUTABLE files are NEVER writable through
this path, regardless of human approval. The bounded vocabulary in
``app/auto_deployer.py::TIER_IMMUTABLE`` is the authoritative list.

Other constraints:
  * Path must be repo-relative (no absolute paths, no `..` traversal).
  * Path must point inside the repo tree (`app/`, `tests/`, `docs/`,
    `dashboard-react/`, `deploy/`, etc) — NOT into `workspace/`,
    which has its own write surface (`file_manager`).
  * Path must NOT match well-known sensitive patterns (`*.env*`,
    secrets/, .git/).
  * Content size must be reasonable (≤ 1 MB).

Validation is the single guard. Once a request passes here, it
goes to Signal for human approval. Once approved, it gets applied.
There is no post-approval validation that could reject a TIER_IMMUTABLE
write — the rule is "TIER_IMMUTABLE never reaches Signal."
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Maximum file content size we accept. Larger files are typically not
# legitimate single-shot agent edits (and would blow up Signal message
# size for the diff display).
_MAX_CONTENT_BYTES = 1_000_000  # 1 MB


# Roots inside the repo where change requests are allowed. Anything
# outside this set is rejected as "outside repo." Add new roots
# here when needed (e.g. a future top-level dir).
#
# ``wiki/`` is included because the wiki-index reconciler files CRs
# against ``wiki/index.md`` to land canonical-rebuild artefacts as
# permanent paper trails (it does NOT modify wiki content directly —
# the WikiWriteTool path remains the only writer for wiki pages).
# The TIER_IMMUTABLE check below still applies, so any wiki path
# placed in TIER_IMMUTABLE is still refused.
_ALLOWED_ROOT_PREFIXES: tuple[str, ...] = (
    "app/",
    "tests/",
    "docs/",
    "dashboard-react/",
    "deploy/",
    "scripts/",
    "host_bridge/",
    "wiki/",
)


# File-name patterns that are categorically rejected even at allowed
# roots (e.g. somebody's `.env.local` in `deploy/` would be a leak).
_BLOCKED_NAME_PATTERNS: tuple[str, ...] = (
    ".env",
    ".envrc",
    "secrets/",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
    ".git/",
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    is_tier_immutable: bool = False  # special-cased rejection


def _load_tier_immutable() -> frozenset[str]:
    """Lazy import of TIER_IMMUTABLE so the validator package doesn't
    drag auto_deployer into module-load time. Returns the canonical
    set of repo-relative paths that no agent path can modify."""
    try:
        from app.auto_deployer import TIER_IMMUTABLE
        return TIER_IMMUTABLE
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "validator: cannot load TIER_IMMUTABLE (%s) — failing closed",
            exc,
        )
        # Fail-closed: if we can't load the list, we refuse all writes.
        # Better to break the change-request flow than to allow an
        # unauthorized write.
        return frozenset({"*"})


def validate(
    *,
    path: str,
    new_content: str,
) -> ValidationResult:
    """Run every check; return the first failure or success.

    Args:
        path: Repo-relative path. e.g. ``"app/agents/pim_agent.py"``.
        new_content: Proposed file contents.

    Returns:
        ``ValidationResult(ok=True)`` if all checks pass.
        Otherwise ``ok=False`` with a human-readable ``reason`` and
        ``is_tier_immutable=True`` when the rejection was specifically
        because the path is in ``TIER_IMMUTABLE`` (so the caller can
        record it as TIER_IMMUTABLE_REFUSED, distinct from
        REJECTED).
    """
    # 1. Type sanity
    if not isinstance(path, str) or not path:
        return ValidationResult(ok=False, reason="path must be a non-empty string")
    if not isinstance(new_content, str):
        return ValidationResult(ok=False, reason="new_content must be a string")

    # 2. Size
    if len(new_content.encode("utf-8")) > _MAX_CONTENT_BYTES:
        return ValidationResult(
            ok=False,
            reason=f"new_content exceeds {_MAX_CONTENT_BYTES // 1024} KB cap",
        )

    # 3. Path traversal / absolute / weird
    norm = os.path.normpath(path)
    if norm != path:
        return ValidationResult(
            ok=False,
            reason=f"path is not normalized; got {path!r}, want {norm!r}",
        )
    if path.startswith("/") or path.startswith("\\"):
        return ValidationResult(ok=False, reason="path must be repo-relative, not absolute")
    if ".." in path.split("/"):
        return ValidationResult(ok=False, reason="path traversal (`..`) is forbidden")

    # 4. Allowed root prefix
    if not any(path.startswith(p) for p in _ALLOWED_ROOT_PREFIXES):
        return ValidationResult(
            ok=False,
            reason=(
                f"path {path!r} is outside the repo's allowed roots "
                f"({sorted(_ALLOWED_ROOT_PREFIXES)}); change-request flow "
                f"only writes under those roots"
            ),
        )

    # 5. Blocked-name patterns (independent of root)
    for pat in _BLOCKED_NAME_PATTERNS:
        if pat in path:
            return ValidationResult(
                ok=False,
                reason=f"path matches blocked pattern {pat!r} (likely sensitive)",
            )

    # 6. TIER_IMMUTABLE — the absolute rule
    tier_immutable = _load_tier_immutable()
    if path in tier_immutable:
        return ValidationResult(
            ok=False,
            reason=(
                f"path {path!r} is in TIER_IMMUTABLE — no agent path "
                f"can modify it, regardless of human approval. Operator "
                f"must edit directly via PR."
            ),
            is_tier_immutable=True,
        )

    return ValidationResult(ok=True)


def is_protected(path: str) -> bool:
    """Quick yes/no for "is this path TIER_IMMUTABLE?". Used by the
    React UI to disable approve/override buttons for protected paths."""
    return path in _load_tier_immutable()


# ── Auto-apply validation ────────────────────────────────────────────
#
# AUTO_APPLY-class change requests bypass the operator gate. The
# safety profile is intentionally narrow:
#
#   * Caller must be in the allowlist below — empty by default so
#     the capability is dormant until the operator explicitly opts a
#     specific runbook handler in.
#   * Patch is additive-only (no deleted lines from old_content).
#   * Net line delta ≤ ``_AUTO_APPLY_LINE_CAP``.
#   * Path is in the auto-apply path allowlist (also empty by
#     default — same dormant-by-default discipline).
#   * Path is NOT under any forbidden prefix (memory, KB, migrations,
#     souls, governance) — these are categorically refused even for
#     allowlisted callers because the consequences exceed the
#     auto-revert watcher's blast-radius guarantee.
#
# Bypassing the auto-apply criteria does NOT reject the CR — the
# ``create_request`` lifecycle gracefully downgrades the risk_class
# to STANDARD, sending the CR through the normal operator gate.
#
# ── Q1.4 (PROGRAM §40.4) — pattern-eligibility audit ─────────────────
#
# Recurring proposal: "auto-apply the schema-drift handlers
# (`_handle_numeric_overflow` + `_handle_missing_column` in
# ``app/healing/handlers/schema_drift.py``)." This has come up in
# Q1 planning rounds repeatedly. The answer is DELIBERATELY NO.
#
# Two independent disqualifiers:
#
#   1. **`migrations/` is in `_AUTO_APPLY_FORBIDDEN_PREFIXES`.** That
#      list categorically refuses auto-apply for ANY caller — schema
#      changes need eyeballs. Both schema-drift handlers write to
#      ``migrations/YYYYMMDD_HHMMSS_*.sql``.
#
#   2. **The handlers produce TODO scaffolds, not executable patches.**
#      ``_propose_widening_migration`` emits literal ``<TABLE>`` /
#      ``<COLUMN>`` placeholders that the operator MUST hand-edit
#      before running the migration. ``_propose_pending_migration_
#      marker`` writes an audit-trail marker whose docstring says
#      "Delete this file once the migration has run; it's only a
#      marker." Auto-applying either would land an unrunnable file
#      under ``migrations/`` AND require the operator to clean it up
#      anyway — strictly worse than the current operator-gated flow.
#
# This rationale is recorded here so future Q1 / Q-N audits don't
# re-propose it. If a future handler emits truly additive, executable,
# idempotent migration content (e.g. `ADD COLUMN IF NOT EXISTS` with
# concrete TABLE+COLUMN derived from the captured error context), it
# would still hit disqualifier #1 — the `migrations/` forbidden
# prefix would need to be revisited first, with explicit consideration
# of how the auto-revert watcher rolls back a schema change. (Spoiler:
# it doesn't, cleanly. That's the right reason for the prefix to
# stay forbidden.)

# Allowed callers (requestor agent_id). Empty by default.
# See the "pattern-eligibility audit" comment above for why this
# stays empty even for the seemingly-safe schema-drift handlers.
_AUTO_APPLY_ALLOWED_REQUESTORS: frozenset[str] = frozenset()

# Allowed target paths. Exact match OR prefix match (with trailing
# slash). Empty by default.
_AUTO_APPLY_ALLOWED_PATHS: tuple[str, ...] = ()

# Categorically forbidden prefixes — not even allowlisted callers
# can auto-apply changes here.
_AUTO_APPLY_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "app/memory/",      # embedding-dim invariants
    "app/souls/",       # identity-shaping artefacts
    "wiki/governance/", # constitution / governance docs
    "migrations/",      # schema migrations need eyeballs
    "deploy/",          # cluster topology
    "host_bridge/",     # the bridge that lets us write at all
)

# Net additional lines beyond the original. Net delta means:
#   added_lines - removed_lines
# We cap at 20 — anything larger needs operator review.
_AUTO_APPLY_LINE_CAP = 20


def _net_line_delta(old_content: str, new_content: str) -> tuple[int, int]:
    """Return (added_count, removed_count) line counts.

    Uses a minimal diff: lines present in new but not old count as
    added; lines present in old but not new count as removed. This is
    not a perfect diff (block moves register as both add and remove)
    — that's acceptable because moves indicate structural change that
    the operator gate should review.
    """
    old_lines = old_content.splitlines() if old_content else []
    new_lines = new_content.splitlines() if new_content else []
    if old_lines == new_lines:
        return 0, 0
    # Use difflib for accurate line accounting that handles reorderings.
    import difflib
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
    added = sum(
        1 for line in diff
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in diff
        if line.startswith("-") and not line.startswith("---")
    )
    return added, removed


def _matches_auto_apply_path(path: str) -> bool:
    """Whitelist check — caller path must match an allowed entry exactly
    OR be under an allowed prefix. Empty allowlist ⇒ no match."""
    if not _AUTO_APPLY_ALLOWED_PATHS:
        return False
    for allowed in _AUTO_APPLY_ALLOWED_PATHS:
        if path == allowed:
            return True
        if allowed.endswith("/") and path.startswith(allowed):
            return True
    return False


def validate_auto_apply(
    *,
    path: str,
    new_content: str,
    old_content: str,
    requestor: str,
) -> ValidationResult:
    """Strict validator for AUTO_APPLY-class change requests.

    Runs the standard ``validate(...)`` first — auto-apply NEVER
    relaxes any standard check. Then layers on the auto-apply
    constraints. Returns the first failure or success.
    """
    # 1. Standard validation must pass.
    standard = validate(path=path, new_content=new_content)
    if not standard.ok:
        return standard

    # 2. Forbidden-prefix check (categorical refusal).
    for forbidden in _AUTO_APPLY_FORBIDDEN_PREFIXES:
        if path.startswith(forbidden):
            return ValidationResult(
                ok=False,
                reason=(
                    f"path {path!r} is under {forbidden!r} — auto-apply "
                    f"is categorically forbidden for this prefix; route "
                    f"through the operator gate instead"
                ),
            )

    # 3. Requestor allowlist.
    if requestor not in _AUTO_APPLY_ALLOWED_REQUESTORS:
        return ValidationResult(
            ok=False,
            reason=(
                f"requestor {requestor!r} is not in the auto-apply "
                f"allowlist; capability is dormant until an operator "
                f"explicitly opts in (see crewai-team/docs/AUTO_APPLY.md)"
            ),
        )

    # 4. Path allowlist.
    if not _matches_auto_apply_path(path):
        return ValidationResult(
            ok=False,
            reason=(
                f"path {path!r} is not in the auto-apply path "
                f"allowlist; capability is dormant until an operator "
                f"explicitly opts the path in"
            ),
        )

    # 5. Line cap.
    added, removed = _net_line_delta(old_content, new_content)
    net = added - removed
    if net > _AUTO_APPLY_LINE_CAP:
        return ValidationResult(
            ok=False,
            reason=(
                f"net line delta {net} exceeds auto-apply cap "
                f"{_AUTO_APPLY_LINE_CAP} (added {added}, removed {removed})"
            ),
        )

    # 6. Additive-only.
    if removed > 0:
        return ValidationResult(
            ok=False,
            reason=(
                f"auto-apply requires additive-only patches; got "
                f"{removed} deleted line(s)"
            ),
        )

    return ValidationResult(ok=True)
