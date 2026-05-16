"""Vacation-mode allowlist validation — narrower than §38.3 AUTO_APPLY.

Composes:
  * ``app.change_requests.validator.validate`` runs FIRST — TIER_IMMUTABLE
    + repo-root + sensitive-name + content-size all enforced.
  * Then the vacation-specific constraints below:
      - path matches a vacation path-prefix allowlist entry
      - requestor matches a vacation requestor allowlist entry
      - net line delta ≤ vacation max_diff_lines
      - patch is additive-only (no deletions; same as §38.3)
      - path is NOT under any of the FORBIDDEN_PREFIXES below (a
        super-set of §38.3's forbidden prefixes; we add more for the
        narrower vacation-mode policy)

Failure-isolated: never raises; returns a ``VacationValidationResult``
with ``ok=False`` + reason on any rejection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.vacation_mode.state import VacationAllowlist, current_allowlist

logger = logging.getLogger(__name__)


# Categorically forbidden — operator's allowlist cannot include these.
# Super-set of §38.3 ``_AUTO_APPLY_FORBIDDEN_PREFIXES``. New prefixes
# vs §38.3:
#   * app/governance_amendment/  — Tier-3 amendment protocol
#   * app/governance_ratchet/    — SAFETY/QUALITY minimum ratchet
#   * app/auto_deployer.py       — TIER_IMMUTABLE list home
#   * app/change_requests/       — the CR system itself
#   * app/vacation_mode/         — this module (no self-modification)
#   * app/subia/                 — consciousness modules
#   * .github/                   — CI/CD workflows
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    # §38.3 already-forbidden, retained:
    "app/memory/",
    "app/souls/",
    "wiki/governance/",
    "migrations/",
    "deploy/",
    "host_bridge/",
    # New, vacation-mode-specific:
    "app/governance_amendment/",
    "app/governance_ratchet/",
    "app/change_requests/",
    "app/vacation_mode/",
    "app/subia/",
    "app/auto_deployer.py",
    ".github/",
)


@dataclass(frozen=True)
class VacationValidationResult:
    ok: bool
    reason: Optional[str] = None
    is_tier_immutable: bool = False


def _net_line_delta(old_content: str, new_content: str) -> tuple[int, int]:
    """Same line-accounting as §38.3, copied here to keep allowlist
    self-contained (and to avoid importing module-private helpers
    from ``app.change_requests.validator``)."""
    import difflib
    old_lines = old_content.splitlines() if old_content else []
    new_lines = new_content.splitlines() if new_content else []
    if old_lines == new_lines:
        return 0, 0
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


def _matches_path_prefix(
    path: str,
    allowlist: VacationAllowlist,
) -> bool:
    """Path matches any vacation path-prefix allowlist entry. By
    construction, each entry ends with '/'."""
    for prefix in allowlist.path_prefix_allowlist:
        if path.startswith(prefix):
            return True
    return False


def validate_vacation_apply(
    *,
    path: str,
    new_content: str,
    old_content: str,
    requestor: str,
    allowlist: Optional[VacationAllowlist] = None,
) -> VacationValidationResult:
    """Vacation-specific auto-apply validator.

    Order:
      1. Standard ``validate()`` — TIER_IMMUTABLE + repo-root + sensitive
         name + content-size.
      2. Forbidden-prefix check (super-set of §38.3's).
      3. Requestor allowlist.
      4. Path prefix allowlist.
      5. Net line delta ≤ ``allowlist.max_diff_lines``.
      6. Additive-only (no deletions).
    """
    # 1. Standard validation runs first. Import is lazy so the
    # vacation_mode package doesn't drag change_requests at module load.
    try:
        from app.change_requests.validator import validate as cr_validate
    except Exception as exc:
        return VacationValidationResult(
            ok=False,
            reason=f"validator unavailable: {type(exc).__name__}",
        )
    standard = cr_validate(path=path, new_content=new_content)
    if not standard.ok:
        return VacationValidationResult(
            ok=False,
            reason=f"standard validation failed: {standard.reason}",
            is_tier_immutable=bool(standard.is_tier_immutable),
        )

    # 2. Forbidden-prefix check.
    for forbidden in FORBIDDEN_PREFIXES:
        if path.startswith(forbidden):
            return VacationValidationResult(
                ok=False,
                reason=(
                    f"path {path!r} is under forbidden prefix "
                    f"{forbidden!r}; not eligible for vacation auto-apply "
                    f"regardless of allowlist"
                ),
            )

    # 3. Allowlist (engaged-window frozen, or current staged).
    al = allowlist if allowlist is not None else current_allowlist()

    # 4. Requestor allowlist.
    if requestor not in set(al.requestor_allowlist):
        return VacationValidationResult(
            ok=False,
            reason=(
                f"requestor {requestor!r} is not in the vacation "
                f"requestor allowlist (have: "
                f"{sorted(al.requestor_allowlist)})"
            ),
        )

    # 5. Path allowlist.
    if not _matches_path_prefix(path, al):
        return VacationValidationResult(
            ok=False,
            reason=(
                f"path {path!r} matches no vacation path-prefix "
                f"allowlist entry (have: "
                f"{sorted(al.path_prefix_allowlist)})"
            ),
        )

    # 6. Line cap.
    added, removed = _net_line_delta(old_content, new_content)
    net = added - removed
    if net > al.max_diff_lines:
        return VacationValidationResult(
            ok=False,
            reason=(
                f"net line delta {net} exceeds vacation max "
                f"{al.max_diff_lines} (added {added}, removed {removed})"
            ),
        )

    # 7. Additive-only.
    if removed > 0:
        return VacationValidationResult(
            ok=False,
            reason=(
                f"vacation auto-apply requires additive-only patches; "
                f"got {removed} deleted line(s)"
            ),
        )

    return VacationValidationResult(ok=True)
