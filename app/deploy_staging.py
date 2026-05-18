"""
deploy_staging.py — Single staging contract for files headed to live deploy.

Closes the historical gap (audit 2026-05-18) where kept code mutations
and approved human-gate requests called schedule_deploy() without first
writing the mutation contents into APPLIED_CODE_DIR — the directory
auto_deployer.run_deploy() actually scans. The result was a silent no-op
("No files to deploy.") for any kept code mutation that did not also
flow through proposals.approve_proposal().

This module is NOT TIER_IMMUTABLE — operator-owned, but agent-modifiable
through the standard change-request gate. The safety guarantees are
defense-in-depth (path traversal + IMMUTABLE refusal) layered on top of
auto_deployer._validate_deploy_batch(), which remains the load-bearing
boundary check at deploy time.

Callers:
  - app/evolution.py:_trigger_code_auto_deploy — stages kept-experiment
    mutations BEFORE confidence classification, so both the HIGH path
    (schedule_deploy → run_deploy) and the BORDERLINE path
    (request_approval → owner reaction → schedule_deploy → run_deploy)
    find the files at deploy time.
  - app/proposals.py — already writes to APPLIED_CODE_DIR directly via
    shutil.copy2 (the working pattern this module extracts). Not yet
    refactored to use stage_for_deploy; semantically equivalent.

stage_for_deploy is content-mode (in-memory bytes); proposals.py uses
file-mode (shutil.copy2). Both write to the same APPLIED_CODE_DIR target.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.paths import APPLIED_CODE_DIR

logger = logging.getLogger(__name__)


def stage_for_deploy(rel_path: str, content: str, source: str) -> Path:
    """Stage a file for deploy by writing it to APPLIED_CODE_DIR/rel_path.

    Args:
        rel_path: Path relative to workspace (e.g. "app/agents/researcher.py").
                  Must not contain ".." or start with "/". Must not be a
                  TIER_IMMUTABLE file (defense in depth — auto_deployer
                  enforces this at deploy time too).
        content:  Full file contents. NOT truncated. The deploy boundary
                  needs the complete file because it runs AST validation
                  and constitutional-invariant checks against this content.
        source:   Provenance tag for the log line (e.g. "evolution-keep",
                  "proposal-42", "auditor"). Not used for safety; only for
                  audit trail.

    Returns:
        The absolute path of the staged file under APPLIED_CODE_DIR.

    Raises:
        ValueError if rel_path is unsafe (traversal / absolute /
                   IMMUTABLE / escapes APPLIED_CODE_DIR after resolution).
    """
    if not rel_path or ".." in rel_path or rel_path.startswith("/"):
        raise ValueError(f"stage_for_deploy: unsafe path: {rel_path!r}")

    # Defense in depth: refuse IMMUTABLE at stage time, not just deploy
    # time. The deploy boundary (_validate_deploy_batch) will refuse too,
    # but failing fast here gives a clear stack trace to the caller AND
    # prevents leaving an IMMUTABLE file in APPLIED_CODE_DIR for an
    # operator to mistakenly approve.
    try:
        from app.auto_deployer import get_protection_tier, ProtectionTier
        if get_protection_tier(rel_path) == ProtectionTier.IMMUTABLE:
            raise ValueError(
                f"stage_for_deploy: IMMUTABLE file cannot be staged: {rel_path}"
            )
    except ImportError:
        # auto_deployer should always be importable; if it isn't, the
        # deploy boundary won't run anyway, so failing here is safer than
        # silently staging.
        raise ValueError(
            f"stage_for_deploy: auto_deployer unavailable; refusing to stage {rel_path}"
        )

    dest = (APPLIED_CODE_DIR / rel_path).resolve()
    # Belt-and-suspenders path-traversal check after resolution. Catches
    # exotic cases (symlinks, oddly-named entries) that the surface
    # check above misses.
    try:
        dest.relative_to(APPLIED_CODE_DIR.resolve())
    except ValueError:
        raise ValueError(
            f"stage_for_deploy: path escapes APPLIED_CODE_DIR: {rel_path}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    logger.info(
        f"deploy_staging: staged {rel_path} "
        f"(source={source}, {len(content)} chars)"
    )
    return dest


def unstage(rel_path: str) -> None:
    """Remove a staged file. Idempotent; never raises.

    Used on experiment revert paths so a kept-then-rejected mutation
    does not leave dead bytes in APPLIED_CODE_DIR. Silently ignores
    unsafe paths and missing files.
    """
    if not rel_path or ".." in rel_path or rel_path.startswith("/"):
        return
    dest = APPLIED_CODE_DIR / rel_path
    if not dest.exists():
        return
    try:
        dest.unlink()
        logger.info(f"deploy_staging: unstaged {rel_path}")
    except OSError as exc:
        logger.debug(f"deploy_staging: unstage failed for {rel_path}: {exc}")
