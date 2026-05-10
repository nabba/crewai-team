"""Validation for architecture requests.

Two layers of refusal:

1. **TIER_IMMUTABLE absolute** — if the proposed package_path or any
   file in file_layout collides with the canonical TIER_IMMUTABLE
   list, the request is refused at validate time. No human approval
   can override this. Same rule as :mod:`app.change_requests.validator`.

2. **Consciousness layer absolute** — proposals cannot place files
   under ``app/subia/`` or modify ``app/affect/goal_emitter.py``.
   The integrity manifest covers SubIA; goal_emitter is the sole
   writer to ``SelfState.current_goals`` (AE-1's STRONG anchor).
   These are refused even outside TIER_IMMUTABLE because they
   require Tier-3 amendment, not architecture-request review.

3. **Structural** — package_path normalised, file paths inside the
   package, integration kinds in the allowed vocabulary, env-switch
   names well-formed, all required fields populated.

Env-switch *collision* with existing ``.env`` entries is a softer
check: the validator surfaces collisions as warnings (returned in
``ValidationResult.warnings``) rather than refusals. The operator
sees the warning at decision time.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from app.architecture_requests.models import (
    ArchitectureRequest,
    VALID_INTEGRATION_KINDS,
)

logger = logging.getLogger(__name__)


# Roots inside the repo where new packages are allowed.
_ALLOWED_PACKAGE_ROOTS: tuple[str, ...] = (
    "app/",
    "tests/",
)

# Sentinels — touching any of these requires Tier-3 amendment, not
# architecture-request review.
_FORBIDDEN_PACKAGE_PREFIXES: tuple[str, ...] = (
    "app/subia/",
)

_FORBIDDEN_INDIVIDUAL_FILES: frozenset[str] = frozenset({
    "app/affect/goal_emitter.py",
})

_ENV_SWITCH_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_tier_immutable: bool = False


def _load_tier_immutable() -> frozenset[str]:
    """Lazy import so the validator doesn't drag auto_deployer at module load."""
    try:
        from app.auto_deployer import TIER_IMMUTABLE
        return TIER_IMMUTABLE
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "architecture_requests.validator: cannot load TIER_IMMUTABLE (%s) "
            "— failing closed",
            exc,
        )
        return frozenset({"*"})


def _normalise(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def _validate_package_path(package_path: str) -> tuple[list[str], bool]:
    """Return (errors, is_tier_immutable)."""
    errors: list[str] = []
    if not package_path:
        return ["package_path is empty"], False
    norm = _normalise(package_path).rstrip("/") + "/"
    if norm != package_path.rstrip("/") + "/":
        errors.append(
            f"package_path must be normalised; got {package_path!r}, want {norm!r}"
        )
    if package_path.startswith("/") or ".." in package_path.split("/"):
        errors.append("package_path must be repo-relative without traversal")
    if not any(package_path.startswith(r) for r in _ALLOWED_PACKAGE_ROOTS):
        errors.append(
            f"package_path {package_path!r} is outside the allowed roots "
            f"{sorted(_ALLOWED_PACKAGE_ROOTS)}"
        )
    for forbidden in _FORBIDDEN_PACKAGE_PREFIXES:
        if package_path.startswith(forbidden):
            errors.append(
                f"package_path {package_path!r} touches the consciousness layer "
                f"({forbidden}) — requires Tier-3 amendment, not architecture-request"
            )
    return errors, False


def _validate_file_layout(
    package_path: str,
    file_specs: list,
    tier_immutable: frozenset[str],
) -> tuple[list[str], bool]:
    """Return (errors, hit_tier_immutable)."""
    errors: list[str] = []
    hit_tier_immutable = False
    seen_paths: set[str] = set()
    for fs in file_specs:
        if fs.path in seen_paths:
            errors.append(f"file_layout has duplicate path {fs.path!r}")
        seen_paths.add(fs.path)
        if fs.path in tier_immutable:
            errors.append(
                f"file_layout entry {fs.path!r} is in TIER_IMMUTABLE — refused"
            )
            hit_tier_immutable = True
        if fs.path in _FORBIDDEN_INDIVIDUAL_FILES:
            errors.append(
                f"file_layout entry {fs.path!r} requires Tier-3 amendment, "
                f"not architecture-request"
            )
        if not any(fs.path.startswith(r) for r in _ALLOWED_PACKAGE_ROOTS):
            errors.append(
                f"file_layout entry {fs.path!r} is outside allowed roots"
            )
        if not fs.path.startswith(package_path):
            errors.append(
                f"file_layout entry {fs.path!r} is outside package_path {package_path!r}"
            )
        if not fs.purpose.strip():
            errors.append(f"file_layout entry {fs.path!r} missing purpose")
    return errors, hit_tier_immutable


def _validate_integration_points(
    integration_points: list,
    tier_immutable: frozenset[str],
) -> tuple[list[str], bool]:
    errors: list[str] = []
    hit_tier_immutable = False
    for ip in integration_points:
        if ip.kind not in VALID_INTEGRATION_KINDS:
            errors.append(
                f"integration_point kind {ip.kind!r} not in "
                f"{sorted(VALID_INTEGRATION_KINDS)}"
            )
        if ip.target_module in tier_immutable:
            # Architecture-requests cannot mutate TIER_IMMUTABLE files,
            # but reading them is fine. The kinds we accept (idle_job_
            # registration etc.) all imply a write to the target module.
            errors.append(
                f"integration_point target {ip.target_module!r} is "
                f"TIER_IMMUTABLE — operator must wire manually"
            )
            hit_tier_immutable = True
    return errors, hit_tier_immutable


def _validate_env_switches(
    env_switches: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Collision with existing env is a warning."""
    errors: list[str] = []
    warnings: list[str] = []
    for name in env_switches:
        if not _ENV_SWITCH_RE.match(name):
            errors.append(
                f"env switch {name!r} must be UPPER_SNAKE_CASE matching "
                f"{_ENV_SWITCH_RE.pattern}"
            )
        if name in os.environ:
            warnings.append(
                f"env switch {name!r} collides with an already-defined "
                f"environment variable; the new default will only apply "
                f"if the existing var is unset"
            )
    return errors, warnings


def validate(req: ArchitectureRequest) -> ValidationResult:
    """Run every check; aggregate the results."""
    errors: list[str] = []
    warnings: list[str] = []
    hit_tier_immutable = False

    # Required-field checks (cheap; do first).
    if not req.intent.strip():
        errors.append("intent must be non-empty")
    if len(req.intent) > 200:
        errors.append("intent must be ≤ 200 characters")
    if not req.motivation.strip():
        errors.append("motivation must be non-empty")
    if not req.test_plan.strip():
        errors.append("test_plan must be non-empty")
    if not req.requestor.strip():
        errors.append("requestor must be non-empty")

    # Package-path checks
    pp_errors, _ = _validate_package_path(req.package_path)
    errors.extend(pp_errors)

    tier_immutable = _load_tier_immutable()

    fl_errors, fl_tier = _validate_file_layout(
        req.package_path, req.file_layout, tier_immutable,
    )
    errors.extend(fl_errors)
    hit_tier_immutable = hit_tier_immutable or fl_tier

    if not req.file_layout:
        errors.append("file_layout cannot be empty")

    ip_errors, ip_tier = _validate_integration_points(
        req.integration_points, tier_immutable,
    )
    errors.extend(ip_errors)
    hit_tier_immutable = hit_tier_immutable or ip_tier

    env_errors, env_warnings = _validate_env_switches(req.env_switches)
    errors.extend(env_errors)
    warnings.extend(env_warnings)

    return ValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        is_tier_immutable=hit_tier_immutable,
    )


def is_protected_path(path: str) -> bool:
    """True iff the path requires Tier-3 amendment instead of architecture-request."""
    if path in _FORBIDDEN_INDIVIDUAL_FILES:
        return True
    if any(path.startswith(p) for p in _FORBIDDEN_PACKAGE_PREFIXES):
        return True
    return path in _load_tier_immutable()
