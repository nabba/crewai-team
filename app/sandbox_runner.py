"""
sandbox_runner.py — Docker-isolated sandbox for testing code evolution mutations.

Replaces the in-place ExperimentRunner for CODE mutations with true process
isolation. A bad mutation that crashes the import system can't take down
the production gateway.

Architecture:
  - Fork-on-write: create a git branch, apply diff, mount into sandbox container
  - Sandbox container: network_mode=none (only Ollama via host gateway)
  - Pre-warmed container: keep a stopped container ready (~2s start vs ~15s cold)
  - Hard timeout: 5 minutes per evaluation cycle
  - Results returned via mounted workspace volume

Prompt mutations remain in-place (via prompt_registry) since they're safe
text swaps that can't crash the process.

IMMUTABLE — infrastructure-level module.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "crewai-sandbox:latest"
SANDBOX_TIMEOUT = 300  # 5 minutes hard limit
SANDBOX_MEMORY = "10g"
SANDBOX_CPUS = "4"
SANDBOX_WORKSPACE = Path("/app/workspace/sandbox-workspace")
EVAL_RESULT_FILE = "eval_result.json"

# Files whose hashes are verified before each sandbox run
EVAL_INTEGRITY_FILES = [
    "app/eval_sandbox.py",
    "app/safety_guardian.py",
    "app/feedback_pipeline.py",
    "app/vetting.py",
]


@dataclass
class SandboxResult:
    """Result from a sandboxed evolution cycle."""
    status: str  # "candidate", "discarded", "error", "timeout"
    branch: str = ""
    metrics: dict = field(default_factory=dict)
    change_description: str = ""
    requires_human_approval: bool = False
    error: str = ""
    duration_seconds: float = 0.0


class SandboxRunner:
    """Run code mutations in an isolated Docker container."""

    def __init__(self, base_branch: str = "main"):
        self.base_branch = base_branch
        self._eval_hashes = self._compute_eval_hashes()

    def _compute_eval_hashes(self) -> dict[str, str]:
        """Hash evaluation functions for tamper detection."""
        hashes = {}
        for f in EVAL_INTEGRITY_FILES:
            path = Path("/app") / f
            if path.exists():
                hashes[f] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes

    def verify_eval_integrity(self) -> bool:
        """Verify evaluation functions haven't been tampered with since init."""
        for f, expected in self._eval_hashes.items():
            path = Path("/app") / f
            if not path.exists():
                logger.error(f"sandbox: eval file missing: {f}")
                return False
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                logger.error(f"sandbox: INTEGRITY VIOLATION: {f}")
                return False
        return True

    def run_sandboxed_mutation(
        self,
        proposed_diff: str,
        change_description: str,
        files: dict[str, str] | None = None,
        yield_check: Callable[[], bool] | None = None,
    ) -> SandboxResult:
        """Execute one evolution cycle in a Docker sandbox.

        Args:
            proposed_diff: unified diff of proposed changes
            change_description: human-readable description
            files: optional dict of {path: content} to apply directly
            yield_check: callable that returns True if user task arrived

        Returns:
            SandboxResult with status, metrics, and approval requirement
        """
        start = time.monotonic()

        # Step 0: Verify evaluation integrity BEFORE anything
        if not self.verify_eval_integrity():
            return SandboxResult(
                status="error",
                error="Evaluation function integrity check failed — possible tampering",
                duration_seconds=time.monotonic() - start,
            )

        # Check for user task interruption
        if yield_check and yield_check():
            return SandboxResult(
                status="error", error="Yielded to user task",
                duration_seconds=time.monotonic() - start,
            )

        # Step 1: Prepare sandbox workspace
        SANDBOX_WORKSPACE.mkdir(parents=True, exist_ok=True)

        # Create a timestamped branch name
        branch_name = f"evolution/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        try:
            # Step 2: Apply proposed changes to sandbox workspace
            if files:
                self._apply_files(files)
            elif proposed_diff:
                applied = self._apply_diff(proposed_diff)
                if not applied:
                    return SandboxResult(
                        status="discarded",
                        error="Failed to apply proposed diff",
                        change_description=change_description,
                        duration_seconds=time.monotonic() - start,
                    )

            # Step 3: Run evaluation in sandbox container
            eval_result = self._run_in_sandbox(branch_name)

            # Step 4: Decision gate
            duration = time.monotonic() - start

            if eval_result.get("status") == "timeout":
                return SandboxResult(
                    status="timeout",
                    error="Evaluation exceeded time limit",
                    change_description=change_description,
                    duration_seconds=duration,
                )

            passed = eval_result.get("passed", False)
            safety_ok = not eval_result.get("safety_violations", False)
            metrics = eval_result.get("metrics", {})

            if passed and safety_ok:
                return SandboxResult(
                    status="candidate",
                    branch=branch_name,
                    metrics=metrics,
                    change_description=change_description,
                    requires_human_approval=self._requires_human_approval(
                        proposed_diff or json.dumps(files or {})
                    ),
                    duration_seconds=duration,
                )
            else:
                reason = "safety_violation" if not safety_ok else eval_result.get(
                    "failure_reason", "metrics_below_threshold"
                )
                return SandboxResult(
                    status="discarded",
                    metrics=metrics,
                    change_description=change_description,
                    error=reason,
                    duration_seconds=duration,
                )

        except Exception as e:
            return SandboxResult(
                status="error",
                error=str(e)[:500],
                change_description=change_description,
                duration_seconds=time.monotonic() - start,
            )
        finally:
            # Cleanup sandbox workspace
            self._cleanup_sandbox()

    def _apply_files(self, files: dict[str, str]) -> None:
        """Write proposed files into sandbox workspace."""
        for rel_path, content in files.items():
            full_path = SANDBOX_WORKSPACE / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

    def _apply_diff(self, diff: str) -> bool:
        """Apply a unified diff to sandbox workspace."""
        diff_path = SANDBOX_WORKSPACE / "proposed.patch"
        diff_path.write_text(diff)
        try:
            # Check if diff applies cleanly
            result = subprocess.run(
                ["git", "apply", "--check", str(diff_path)],
                capture_output=True, timeout=10,
                cwd=str(SANDBOX_WORKSPACE),
            )
            if result.returncode != 0:
                logger.warning(f"sandbox: diff check failed: {result.stderr.decode()[:200]}")
                return False

            # Apply the diff
            subprocess.run(
                ["git", "apply", str(diff_path)],
                capture_output=True, timeout=10,
                cwd=str(SANDBOX_WORKSPACE),
            )
            return True
        except Exception as e:
            logger.warning(f"sandbox: diff apply failed: {e}")
            return False

    def _run_in_sandbox(self, branch_name: str) -> dict:
        """Run evaluation suite inside sandbox container with hard timeout.

        The sandbox container:
        - Has no network access (network_mode=none) EXCEPT Ollama via host gateway
        - Mounts workspace read-only for application code
        - Mounts sandbox-workspace for proposed changes
        - Has hard timeout
        - Runs as non-root user
        """
        result_path = SANDBOX_WORKSPACE / EVAL_RESULT_FILE

        # Remove any stale result file
        result_path.unlink(missing_ok=True)

        try:
            # Build docker run command
            cmd = [
                "docker", "run", "--rm",
                "--name", f"sandbox-{branch_name.replace('/', '-')}",
                # Network isolation — only Ollama via host gateway
                "--network", "none",
                "--add-host", "host.docker.internal:host-gateway",
                # Resource limits
                "--memory", SANDBOX_MEMORY,
                "--cpus", SANDBOX_CPUS,
                # Security
                "--security-opt", "no-new-privileges:true",
                "--read-only",
                "--tmpfs", "/tmp:size=2g",
                # Mount application code (read-only)
                "-v", "/app:/app:ro",
                # Mount sandbox workspace (read-write for results)
                "-v", f"{SANDBOX_WORKSPACE}:/sandbox:rw",
                # Mount soul files read-only
                "-v", "/app/app/souls:/readonly/souls:ro",
                # Environment
                "-e", "SANDBOX_MODE=true",
                "-e", f"EVALUATION_TIMEOUT={SANDBOX_TIMEOUT}",
                "-e", "OLLAMA_HOST=host.docker.internal:11434",
                # Image
                SANDBOX_IMAGE,
                # Command: run evaluation
                "python", "-m", "app.sandbox_eval_runner",
                "--changes-dir", "/sandbox",
                "--output", f"/sandbox/{EVAL_RESULT_FILE}",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=SANDBOX_TIMEOUT + 30,  # +30s grace for container overhead
            )

            # Read results from mounted workspace
            if result_path.exists():
                return json.loads(result_path.read_text())
            else:
                stderr = result.stderr.decode()[:500] if result.stderr else ""
                return {
                    "passed": False,
                    "safety_violations": False,
                    "failure_reason": f"No result file produced. Exit={result.returncode}. {stderr}",
                }

        except subprocess.TimeoutExpired:
            # Kill the sandbox container
            try:
                subprocess.run(
                    ["docker", "kill", f"sandbox-{branch_name.replace('/', '-')}"],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass
            return {"status": "timeout", "passed": False, "safety_violations": False}

        except Exception as e:
            return {
                "passed": False,
                "safety_violations": False,
                "failure_reason": str(e)[:500],
            }

    def _requires_human_approval(self, content: str) -> bool:
        """Determine if a change crosses boundaries requiring human sign-off."""
        sensitive_paths = [
            "souls/", "SOUL.md",
            "agents/commander.py", "app/main.py",
            "config/", "app/config.py",
            "evolution/", "app/evolution.py",
            "app/sandbox_runner.py",
            "app/eval_sandbox.py",
            "app/safety_guardian.py",
            "app/auto_deployer.py",
            "app/vetting.py",
            "app/security.py",
            "tests/eval_",
            "docker-compose", "Dockerfile", "entrypoint",
        ]
        for path in sensitive_paths:
            if path in content:
                return True
        return False

    def _cleanup_sandbox(self) -> None:
        """Clean up sandbox workspace after evaluation."""
        try:
            if SANDBOX_WORKSPACE.exists():
                for item in SANDBOX_WORKSPACE.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
        except Exception:
            pass


# ── Pre-warmed container management ──────────────────────────────────────────


def build_sandbox_image(force: bool = False) -> bool:
    """Build the sandbox Docker image from Dockerfile.sandbox.

    Returns True if image was built or already exists.
    """
    if not force:
        # Check if image exists
        result = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return True

    dockerfile = Path("/app/Dockerfile.sandbox")
    if not dockerfile.exists():
        logger.warning("sandbox: Dockerfile.sandbox not found, using main image")
        return False

    result = subprocess.run(
        ["docker", "build", "-t", SANDBOX_IMAGE, "-f", "Dockerfile.sandbox", "."],
        capture_output=True, timeout=300,
        cwd="/app",
    )
    return result.returncode == 0


def prewarm_sandbox() -> bool:
    """Create a stopped sandbox container for fast starts.

    Pre-warmed start: ~2s. Cold build: ~15s.
    """
    try:
        # Remove any existing pre-warmed container
        subprocess.run(
            ["docker", "rm", "-f", "sandbox-warm"],
            capture_output=True, timeout=5,
        )

        # Create a new stopped container
        result = subprocess.run(
            ["docker", "create", "--name", "sandbox-warm",
             "--network", "none",
             "--memory", SANDBOX_MEMORY,
             "--cpus", SANDBOX_CPUS,
             SANDBOX_IMAGE],
            capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


# ── Parallel sandbox support ─────────────────────────────────────────────────


class ParallelSandboxRunner:
    """Run multiple evolution experiments in parallel sandboxes.

    On M4 Max (48GB): 2-3 concurrent sandboxes with 10GB each,
    leaving 18GB for Ollama models + system headroom.

    Each sandbox explores a different mutation strategy simultaneously.
    Best candidate across all parallel runs gets promoted.
    """

    def __init__(
        self,
        max_parallel: int = 2,
        base_branch: str = "main",
    ):
        self.max_parallel = max_parallel
        self.base_branch = base_branch
        self._runner = SandboxRunner(base_branch)

    def run_parallel_experiments(
        self,
        proposals: list[dict],
        yield_check: Callable[[], bool] | None = None,
    ) -> list[SandboxResult]:
        """Run multiple proposals in parallel sandboxes.

        Args:
            proposals: list of {diff: str, description: str, files: dict, strategy: str}
            yield_check: callable that returns True if user task arrived

        Returns:
            List of SandboxResult, one per proposal (best first)
        """
        import concurrent.futures

        # Cap at max_parallel
        active_proposals = proposals[:self.max_parallel]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = {}
            for i, proposal in enumerate(active_proposals):
                future = pool.submit(
                    self._run_one,
                    proposal,
                    f"parallel-{i}",
                    yield_check,
                )
                futures[future] = proposal

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=SANDBOX_TIMEOUT + 60)
                    results.append(result)
                except Exception as e:
                    results.append(SandboxResult(
                        status="error",
                        error=str(e)[:500],
                        change_description=futures[future].get("description", "unknown"),
                    ))

        # Sort: candidates first, then by metrics
        results.sort(key=lambda r: (
            0 if r.status == "candidate" else 1,
            -r.metrics.get("composite_score", 0),
        ))

        return results

    def _run_one(
        self,
        proposal: dict,
        suffix: str,
        yield_check: Callable[[], bool] | None,
    ) -> SandboxResult:
        """Run a single proposal in its own sandbox."""
        return self._runner.run_sandboxed_mutation(
            proposed_diff=proposal.get("diff", ""),
            change_description=proposal.get("description", ""),
            files=proposal.get("files"),
            yield_check=yield_check,
        )

    def select_best_candidate(self, results: list[SandboxResult]) -> SandboxResult | None:
        """Select the best candidate from parallel results.

        Strategy: pick the candidate with highest composite score.
        If no candidates, return None.
        """
        candidates = [r for r in results if r.status == "candidate"]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.metrics.get("composite_score", 0))
