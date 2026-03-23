"""
experiment_runner.py — Experiment sandbox with before/after measurement.

This is the core of the autoresearch-style evolution loop:
  1. Snapshot metrics BEFORE mutation
  2. Apply the mutation (skill file, prompt tweak, etc.)
  3. Run test tasks to exercise the change
  4. Snapshot metrics AFTER
  5. Keep if improved, revert if not

Like autoresearch's "run train.py, check val_bpb, keep or git reset",
but for an agent system instead of a neural network.
"""

import json
import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.metrics import compute_metrics, composite_score
from app.results_ledger import record_experiment

logger = logging.getLogger(__name__)

SKILLS_DIR = Path("/app/workspace/skills")
TEST_TASKS_PATH = Path("/app/workspace/test_tasks.json")


@dataclass
class MutationSpec:
    """Describes a single mutation to apply and test."""
    experiment_id: str
    hypothesis: str
    change_type: str  # "skill", "prompt", "config"
    files: dict[str, str] = field(default_factory=dict)  # {path: content}


@dataclass
class ExperimentResult:
    """Result of a single experiment cycle."""
    experiment_id: str
    hypothesis: str
    change_type: str
    metric_before: float
    metric_after: float
    delta: float
    status: str  # "keep", "discard", "crash"
    files_changed: list[str] = field(default_factory=list)
    detail: str = ""


def generate_experiment_id(hypothesis: str) -> str:
    """Generate a short unique ID for an experiment."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    h = hashlib.md5(hypothesis.encode()).hexdigest()[:4]
    return f"exp_{ts}_{h}"


class ExperimentRunner:
    """Runs experiments with before/after measurement and automatic revert."""

    def __init__(self):
        self._backup_dir = Path("/app/workspace/.experiment_backup")

    def run_experiment(self, mutation: MutationSpec) -> ExperimentResult:
        """
        Execute one full experiment cycle:
        1. Measure baseline
        2. Backup files that will be changed
        3. Apply mutation
        4. Measure after
        5. Keep or revert
        """
        logger.info(f"Experiment {mutation.experiment_id}: {mutation.hypothesis}")

        # 1. Measure baseline
        try:
            baseline = composite_score()
        except Exception:
            baseline = 0.5  # fallback if metrics fail

        # 2. Backup existing files
        backed_up = self._backup_files(mutation)

        # 3. Apply mutation
        try:
            applied_files = self._apply_mutation(mutation)
        except Exception as exc:
            self._restore_backup(backed_up)
            result = ExperimentResult(
                experiment_id=mutation.experiment_id,
                hypothesis=mutation.hypothesis,
                change_type=mutation.change_type,
                metric_before=baseline,
                metric_after=0.0,
                delta=0.0,
                status="crash",
                detail=f"Failed to apply mutation: {exc}",
            )
            record_experiment(
                experiment_id=result.experiment_id,
                hypothesis=result.hypothesis,
                change_type=result.change_type,
                metric_before=baseline,
                metric_after=0.0,
                status="crash",
                files_changed=[],
            )
            return result

        # 4. Validate the mutation (lightweight checks)
        validation_ok, validation_msg = self._validate_mutation(mutation, applied_files)

        # 5. Measure after
        try:
            after = composite_score()
        except Exception:
            after = baseline  # if metrics fail, treat as no change

        delta = after - baseline

        # 6. Keep/discard decision
        # For skill mutations: keep if score didn't decrease
        # For code mutations: keep only if score improved
        if not validation_ok:
            status = "crash"
            self._restore_backup(backed_up)
            detail = f"Validation failed: {validation_msg}"
        elif mutation.change_type == "skill":
            # Skills are low-risk — keep if not harmful
            if delta >= -0.001:
                status = "keep"
                detail = f"Skill applied (delta={delta:+.4f})"
            else:
                status = "discard"
                self._restore_backup(backed_up)
                detail = f"Skill reverted — score decreased by {abs(delta):.4f}"
        else:
            # Code/config mutations need positive improvement
            if delta > 0.0:
                status = "keep"
                detail = f"Improvement: {delta:+.4f}"
            else:
                status = "discard"
                self._restore_backup(backed_up)
                detail = f"No improvement (delta={delta:+.4f}), reverted"

        # 7. Clean up backup
        self._cleanup_backup()

        result = ExperimentResult(
            experiment_id=mutation.experiment_id,
            hypothesis=mutation.hypothesis,
            change_type=mutation.change_type,
            metric_before=baseline,
            metric_after=after if status != "crash" else 0.0,
            delta=delta,
            status=status,
            files_changed=applied_files,
            detail=detail,
        )

        # 8. Record in ledger (F8: include detail for agent feedback)
        record_experiment(
            experiment_id=result.experiment_id,
            hypothesis=result.hypothesis,
            change_type=result.change_type,
            metric_before=baseline,
            metric_after=result.metric_after,
            status=result.status,
            files_changed=result.files_changed,
            detail=detail,
        )

        logger.info(
            f"Experiment {mutation.experiment_id}: {result.status} "
            f"(before={baseline:.4f}, after={after:.4f}, delta={delta:+.4f})"
        )
        return result

    def _backup_files(self, mutation: MutationSpec) -> dict[str, str | None]:
        """Backup files that will be modified. Returns {path: original_content_or_None}."""
        backed_up = {}
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        for rel_path in mutation.files:
            full_path = Path("/app/workspace") / rel_path
            if full_path.exists():
                backed_up[rel_path] = full_path.read_text()
            else:
                backed_up[rel_path] = None  # file didn't exist before
        return backed_up

    def _restore_backup(self, backed_up: dict[str, str | None]) -> None:
        """Restore backed-up files (revert mutation)."""
        for rel_path, content in backed_up.items():
            full_path = Path("/app/workspace") / rel_path
            if content is None:
                # File was created by mutation — delete it
                if full_path.exists():
                    full_path.unlink()
                    logger.info(f"Reverted: deleted {rel_path}")
            else:
                # File existed before — restore original
                full_path.write_text(content)
                logger.info(f"Reverted: restored {rel_path}")

    def _cleanup_backup(self) -> None:
        """Remove backup directory."""
        if self._backup_dir.exists():
            shutil.rmtree(self._backup_dir, ignore_errors=True)

    def _apply_mutation(self, mutation: MutationSpec) -> list[str]:
        """Apply the mutation files. Returns list of changed file paths."""
        applied = []
        for rel_path, content in mutation.files.items():
            full_path = Path("/app/workspace") / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            applied.append(rel_path)
            logger.info(f"Applied: {rel_path}")
        return applied

    def _validate_mutation(
        self, mutation: MutationSpec, applied_files: list[str]
    ) -> tuple[bool, str]:
        """Lightweight validation of applied mutation."""
        for rel_path in applied_files:
            full_path = Path("/app/workspace") / rel_path
            if not full_path.exists():
                return False, f"File not created: {rel_path}"
            if full_path.stat().st_size == 0:
                return False, f"Empty file: {rel_path}"
            # Skill files should be valid markdown (basic check)
            if rel_path.endswith(".md"):
                content = full_path.read_text()
                if len(content) < 50:
                    return False, f"Skill file too short ({len(content)} chars): {rel_path}"
        return True, "ok"


def load_test_tasks(suite: str = "all") -> list[dict]:
    """Load the test task bank for evaluation.

    Args:
        suite: "fixed" (regression), "rotating" (novelty), or "all" (both).
    """
    try:
        if TEST_TASKS_PATH.exists():
            tasks = json.loads(TEST_TASKS_PATH.read_text())
            if suite == "all":
                return tasks
            return [t for t in tasks if t.get("suite", "fixed") == suite]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def validate_response(response: str, rule: str) -> bool:
    """Validate a response against a simple rule string."""
    if not rule:
        return True

    if rule.startswith("contains:"):
        expected = rule[len("contains:"):]
        return expected.lower() in response.lower()

    if rule.startswith("not_contains:"):
        forbidden = rule[len("not_contains:"):]
        return forbidden.lower() not in response.lower()

    if rule.startswith("min_length:"):
        min_len = int(rule[len("min_length:"):])
        return len(response) >= min_len

    if rule.startswith("max_length:"):
        max_len = int(rule[len("max_length:"):])
        return len(response) <= max_len

    return True


# 5A: Immutable evaluation hash — prevents DGM-style test manipulation
_eval_hash: str | None = None


def compute_eval_hash() -> str:
    """Compute SHA-256 hash of test suites + scoring logic.

    Called once at startup. Before each experiment, the current hash
    is verified against the startup hash. If mismatch → abort.
    """
    import hashlib
    h = hashlib.sha256()
    # Hash test tasks
    try:
        if TEST_TASKS_PATH.exists():
            h.update(TEST_TASKS_PATH.read_bytes())
    except OSError:
        pass
    # Hash security tests
    sec_path = Path("/app/workspace/security_tests.json")
    try:
        if sec_path.exists():
            h.update(sec_path.read_bytes())
    except OSError:
        pass
    # Hash this module's source (scoring logic)
    try:
        h.update(Path(__file__).read_bytes())
    except OSError:
        pass
    return h.hexdigest()[:16]


def verify_eval_integrity() -> bool:
    """Verify evaluation functions haven't been tampered with.

    Returns True if integrity check passes (or if no baseline exists).
    """
    global _eval_hash
    current = compute_eval_hash()
    if _eval_hash is None:
        _eval_hash = current
        logger.info(f"Evaluation integrity baseline: {_eval_hash}")
        return True
    if current != _eval_hash:
        logger.error(
            f"EVAL INTEGRITY VIOLATION: hash changed from {_eval_hash} to {current}. "
            f"Possible test manipulation detected."
        )
        return False
    return True
