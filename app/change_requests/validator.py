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
_ALLOWED_ROOT_PREFIXES: tuple[str, ...] = (
    "app/",
    "tests/",
    "docs/",
    "dashboard-react/",
    "deploy/",
    "scripts/",
    "host_bridge/",
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
