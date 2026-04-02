"""
parallel_evolution.py — Parallel sandbox evolution with diverse archive.

Maintains a diverse archive of successful variants (stepping stones) and
explores multiple mutation strategies in parallel using 2-3 Docker sandbox
instances on M4 Max (48GB).

Inspired by DGM open-ended exploration: instead of a single linear evolution
path, maintain an archive of diverse successful configurations. Parent
selection favors novelty and under-explored strategies.

Architecture:
  - EvolutionArchive: stores diverse variants with novelty scoring
  - ParallelRunner: runs 2-3 sandbox instances simultaneously
  - Parent selection: tournament + novelty bonus + strategy diversity
  - Best candidate across all parallel runs gets promoted

Memory budget (M4 Max 48GB):
  2 sandboxes × 8GB = 16GB sandboxes
  Ollama models: ~20GB
  System + gateway: ~12GB headroom

IMMUTABLE — infrastructure-level module.
"""

import hashlib
import json
import logging
import math
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path("/app/workspace/evolution_archive")
SANDBOX_WORKSPACES = Path("/app/workspace/sandbox_workspaces")

# ── Archive entry ─────────────────────────────────────────────────────────────


@dataclass
class ArchiveEntry:
    """A successful agent configuration variant in the archive."""
    version_tag: str
    metrics: dict = field(default_factory=dict)
    change_description: str = ""
    parent_version: str = ""
    mutation_strategy: str = ""  # prompt_optimization | cascade_rebalance | rag_tuning | etc
    novelty_score: float = 0.0
    composite_score: float = 0.0
    created_at: str = ""
    times_selected_as_parent: int = 0
    child_success_count: int = 0
    child_total_count: int = 0

    def to_dict(self) -> dict:
        return {
            "version_tag": self.version_tag,
            "metrics": self.metrics,
            "change_description": self.change_description,
            "parent_version": self.parent_version,
            "mutation_strategy": self.mutation_strategy,
            "novelty_score": self.novelty_score,
            "composite_score": self.composite_score,
            "created_at": self.created_at,
            "times_selected_as_parent": self.times_selected_as_parent,
            "child_success_count": self.child_success_count,
            "child_total_count": self.child_total_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArchiveEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Evolution Archive ─────────────────────────────────────────────────────────


# IMMUTABLE: Novelty scoring weights
NOVELTY_RECENCY_WEIGHT = 0.3
NOVELTY_DIVERSITY_WEIGHT = 0.4
NOVELTY_UNDEREXPLORED_WEIGHT = 0.3

# UCB exploration constant for parent selection
UCB_C = 1.414


class EvolutionArchive:
    """Maintains a diverse archive of successful agent configurations.

    Supports novelty-based parent selection and strategy diversity tracking.
    """

    def __init__(self, archive_dir: Path = ARCHIVE_DIR):
        self._dir = archive_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[ArchiveEntry] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load archive from disk."""
        archive_file = self._dir / "archive.json"
        if archive_file.exists():
            try:
                data = json.loads(archive_file.read_text())
                self._entries = [ArchiveEntry.from_dict(e) for e in data]
            except Exception:
                logger.warning("parallel_evolution: failed to load archive", exc_info=True)

    def _save(self) -> None:
        """Persist archive to disk."""
        archive_file = self._dir / "archive.json"
        data = [e.to_dict() for e in self._entries]
        archive_file.write_text(json.dumps(data, indent=2))

    def add(self, entry: ArchiveEntry) -> None:
        """Add a successful variant to the archive."""
        if not entry.created_at:
            entry.created_at = datetime.now(timezone.utc).isoformat()
        entry.novelty_score = self.compute_novelty(entry.metrics)
        with self._lock:
            self._entries.append(entry)
            self._save()
        logger.info(f"parallel_evolution: archived {entry.version_tag} "
                    f"(novelty={entry.novelty_score:.3f}, strategy={entry.mutation_strategy})")

    def select_parent(self) -> Optional[ArchiveEntry]:
        """Select a parent variant using UCB1 with novelty bonus.

        Strategy: weighted selection favoring:
          - Higher novelty (explore diverse directions)
          - Recency (recent improvements are better starting points)
          - Under-explored mutation strategies
        """
        with self._lock:
            if not self._entries:
                return None

            total_selections = sum(e.times_selected_as_parent for e in self._entries) + 1
            best_score = -float("inf")
            best_entry = None

            for entry in self._entries:
                # UCB1 base: exploitation (composite score)
                exploit = entry.composite_score

                # Exploration bonus
                n = entry.times_selected_as_parent + 1
                explore = UCB_C * math.sqrt(math.log(total_selections) / n)

                # Novelty bonus
                novelty_bonus = entry.novelty_score * NOVELTY_DIVERSITY_WEIGHT

                # Recency bonus (decay over 30 days)
                try:
                    age_days = (datetime.now(timezone.utc) -
                                datetime.fromisoformat(entry.created_at)).days
                    recency = max(0, 1.0 - age_days / 30.0) * NOVELTY_RECENCY_WEIGHT
                except Exception:
                    recency = 0.0

                # Strategy diversity bonus
                strategy_count = sum(
                    1 for e in self._entries if e.mutation_strategy == entry.mutation_strategy
                )
                strategy_diversity = (1.0 / strategy_count) * NOVELTY_UNDEREXPLORED_WEIGHT

                score = exploit + explore + novelty_bonus + recency + strategy_diversity

                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry:
                best_entry.times_selected_as_parent += 1
                self._save()

            return best_entry

    def record_child_outcome(self, parent_tag: str, success: bool) -> None:
        """Record the outcome of a child mutation for parent statistics."""
        with self._lock:
            for entry in self._entries:
                if entry.version_tag == parent_tag:
                    entry.child_total_count += 1
                    if success:
                        entry.child_success_count += 1
                    self._save()
                    return

    def compute_novelty(self, candidate_metrics: dict) -> float:
        """How different is this candidate from existing archive entries?

        Uses L2 distance in normalized metric space. Higher = more novel.
        """
        if not self._entries or not candidate_metrics:
            return 1.0  # First entry is maximally novel

        # Metric dimensions to compare
        dims = ["task_completion", "user_alignment", "efficiency",
                "consistency", "generalization", "safety"]

        distances = []
        for entry in self._entries:
            if not entry.metrics:
                continue
            dist_sq = 0.0
            dim_count = 0
            for dim in dims:
                c_val = candidate_metrics.get(dim, 0.0)
                e_val = entry.metrics.get(dim, 0.0)
                dist_sq += (c_val - e_val) ** 2
                dim_count += 1
            if dim_count > 0:
                distances.append(math.sqrt(dist_sq / dim_count))

        if not distances:
            return 1.0

        # Novelty = average distance to k nearest neighbors (k=5)
        distances.sort()
        k = min(5, len(distances))
        return sum(distances[:k]) / k

    def get_strategy_distribution(self) -> dict[str, int]:
        """Return count of entries per mutation strategy."""
        dist: dict[str, int] = {}
        for entry in self._entries:
            s = entry.mutation_strategy or "unknown"
            dist[s] = dist.get(s, 0) + 1
        return dist

    def get_best_variants(self, n: int = 5) -> list[ArchiveEntry]:
        """Return top N variants by composite score."""
        return sorted(self._entries, key=lambda e: e.composite_score, reverse=True)[:n]

    def get_diverse_sample(self, n: int = 3) -> list[ArchiveEntry]:
        """Return N variants that are maximally diverse from each other."""
        if len(self._entries) <= n:
            return list(self._entries)

        # Greedy farthest-point sampling
        sample = [self._entries[0]]
        remaining = list(self._entries[1:])

        while len(sample) < n and remaining:
            best_dist = -1
            best_idx = 0
            for i, candidate in enumerate(remaining):
                min_dist = min(
                    self._pairwise_distance(candidate, s) for s in sample
                )
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            sample.append(remaining.pop(best_idx))

        return sample

    def _pairwise_distance(self, a: ArchiveEntry, b: ArchiveEntry) -> float:
        """L2 distance between two archive entries in metric space."""
        dims = ["task_completion", "user_alignment", "efficiency",
                "consistency", "generalization"]
        dist_sq = 0.0
        for dim in dims:
            v1 = a.metrics.get(dim, 0.0)
            v2 = b.metrics.get(dim, 0.0)
            dist_sq += (v1 - v2) ** 2
        return math.sqrt(dist_sq)

    @property
    def size(self) -> int:
        return len(self._entries)

    def format_report(self) -> str:
        """Generate human-readable archive report."""
        if not self._entries:
            return "Evolution archive: empty"

        lines = [
            f"🧬 Evolution Archive ({len(self._entries)} variants)",
            "",
            "Strategy Distribution:",
        ]
        for strategy, count in sorted(self.get_strategy_distribution().items()):
            lines.append(f"  {strategy}: {count}")

        lines.append("")
        lines.append("Top 5 Variants:")
        for entry in self.get_best_variants(5):
            child_rate = (
                f"{entry.child_success_count}/{entry.child_total_count}"
                if entry.child_total_count > 0 else "—"
            )
            lines.append(
                f"  {entry.version_tag}: score={entry.composite_score:.3f} "
                f"novelty={entry.novelty_score:.3f} children={child_rate} "
                f"strategy={entry.mutation_strategy}"
            )

        return "\n".join(lines)


# ── Parallel sandbox execution ────────────────────────────────────────────────


@dataclass
class ParallelCandidate:
    """A candidate from a parallel sandbox run."""
    sandbox_id: str
    mutation_strategy: str
    parent_tag: str
    diff: str = ""
    description: str = ""
    metrics: dict = field(default_factory=dict)
    passed: bool = False
    safety_violations: bool = False
    duration_seconds: float = 0.0
    error: str = ""


class ParallelEvolutionRunner:
    """Run multiple sandbox instances in parallel for diverse exploration.

    On M4 Max (48GB):
      2 sandboxes × 8GB = 16GB
      Ollama: ~20GB
      System: ~12GB headroom
    """

    # IMMUTABLE resource limits
    MAX_PARALLEL = 2  # Conservative for M4 Max
    MEMORY_PER_SANDBOX = "8g"
    CPUS_PER_SANDBOX = 3
    EVAL_TIMEOUT = 300  # 5 minutes

    def __init__(self, archive: EvolutionArchive | None = None):
        self._archive = archive or EvolutionArchive()
        self._sandbox_lock = threading.Lock()

    def run_parallel_cycle(
        self,
        proposals: list[dict],
        base_branch: str = "main",
    ) -> list[ParallelCandidate]:
        """Run multiple mutation proposals in parallel sandboxes.

        Each proposal dict: {strategy, diff, description, parent_tag}

        Returns list of candidates with metrics. The caller decides which
        (if any) to promote.
        """
        # Limit to MAX_PARALLEL
        proposals = proposals[:self.MAX_PARALLEL]
        if not proposals:
            return []

        SANDBOX_WORKSPACES.mkdir(parents=True, exist_ok=True)

        candidates: list[ParallelCandidate] = []

        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL) as executor:
            futures = {}
            for i, proposal in enumerate(proposals):
                sandbox_id = f"sandbox-{i}"
                future = executor.submit(
                    self._run_single_sandbox,
                    sandbox_id=sandbox_id,
                    proposal=proposal,
                    base_branch=base_branch,
                )
                futures[future] = sandbox_id

            for future in as_completed(futures, timeout=self.EVAL_TIMEOUT + 60):
                try:
                    candidate = future.result()
                    candidates.append(candidate)
                except Exception as e:
                    sid = futures[future]
                    candidates.append(ParallelCandidate(
                        sandbox_id=sid,
                        mutation_strategy="unknown",
                        parent_tag="",
                        error=str(e)[:500],
                    ))

        # Record outcomes in archive
        for candidate in candidates:
            if candidate.parent_tag:
                self._archive.record_child_outcome(
                    candidate.parent_tag, candidate.passed
                )

        return candidates

    def select_best_candidate(
        self, candidates: list[ParallelCandidate]
    ) -> Optional[ParallelCandidate]:
        """Select the best candidate from parallel runs.

        Criteria (in priority order):
          1. No safety violations
          2. Passed evaluation
          3. Highest composite score
        """
        valid = [c for c in candidates if c.passed and not c.safety_violations]
        if not valid:
            return None

        # Sort by composite score (sum of metric values)
        def _score(c: ParallelCandidate) -> float:
            return sum(c.metrics.get(d, 0.0) for d in [
                "task_completion", "user_alignment", "efficiency",
                "consistency", "generalization",
            ])

        valid.sort(key=_score, reverse=True)
        return valid[0]

    def _run_single_sandbox(
        self,
        sandbox_id: str,
        proposal: dict,
        base_branch: str,
    ) -> ParallelCandidate:
        """Run a single mutation in an isolated sandbox container."""
        start = time.monotonic()

        candidate = ParallelCandidate(
            sandbox_id=sandbox_id,
            mutation_strategy=proposal.get("strategy", "unknown"),
            parent_tag=proposal.get("parent_tag", ""),
            diff=proposal.get("diff", ""),
            description=proposal.get("description", ""),
        )

        workspace = SANDBOX_WORKSPACES / sandbox_id
        workspace.mkdir(parents=True, exist_ok=True)

        try:
            # Write the proposed diff to workspace
            diff_path = workspace / "proposed.patch"
            diff_path.write_text(proposal.get("diff", ""))

            # Write evaluation config
            eval_config = {
                "sandbox_id": sandbox_id,
                "strategy": proposal.get("strategy", ""),
                "description": proposal.get("description", ""),
                "timeout": self.EVAL_TIMEOUT,
            }
            (workspace / "eval_config.json").write_text(json.dumps(eval_config))

            # Hash evaluation functions for integrity check
            eval_hash = self._hash_eval_functions()
            (workspace / "eval_hash.txt").write_text(eval_hash)

            # Run sandbox container
            result = self._run_container(sandbox_id, workspace)

            # Parse results
            results_path = workspace / "eval_result.json"
            if results_path.exists():
                eval_result = json.loads(results_path.read_text())
                candidate.metrics = eval_result.get("metrics", {})
                candidate.passed = eval_result.get("passed", False)
                candidate.safety_violations = eval_result.get("safety_violations", False)
            else:
                candidate.error = result.get("error", "No evaluation result produced")

        except subprocess.TimeoutExpired:
            candidate.error = f"Timeout after {self.EVAL_TIMEOUT}s"
        except Exception as e:
            candidate.error = str(e)[:500]
        finally:
            candidate.duration_seconds = time.monotonic() - start
            # Cleanup workspace
            try:
                shutil.rmtree(workspace, ignore_errors=True)
            except Exception:
                pass

        return candidate

    def _run_container(self, sandbox_id: str, workspace: Path) -> dict:
        """Launch a sandbox Docker container for evaluation."""
        try:
            from app.config import get_settings
            settings = get_settings()
            sandbox_image = settings.sandbox_image

            cmd = [
                "docker", "run", "--rm",
                "--name", f"crewai-{sandbox_id}",
                "--network", "none",
                "--memory", self.MEMORY_PER_SANDBOX,
                f"--cpus={self.CPUS_PER_SANDBOX}",
                "--tmpfs", "/tmp:size=2g",
                "-v", f"{workspace}:/workspace",
                "-e", "SANDBOX_MODE=true",
                "-e", f"EVALUATION_TIMEOUT={self.EVAL_TIMEOUT}",
                "-e", f"OLLAMA_HOST={settings.local_llm_base_url}",
                sandbox_image,
                "python", "-m", "app.reference_tasks", "--run-suite",
                "--output", "/workspace/eval_result.json",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=self.EVAL_TIMEOUT + 30,
            )

            if result.returncode != 0:
                return {"error": result.stderr[:500]}
            return {"success": True}

        except FileNotFoundError:
            # Docker not available (running outside Docker, or Docker-in-Docker)
            # Fall back to in-process evaluation
            return self._fallback_eval(workspace)

    def _fallback_eval(self, workspace: Path) -> dict:
        """In-process evaluation fallback when Docker is unavailable."""
        try:
            from app.reference_tasks import ReferenceTaskSuite
            suite = ReferenceTaskSuite()
            results = suite.run_quick_check()
            (workspace / "eval_result.json").write_text(json.dumps(results))
            return {"success": True, "method": "in-process"}
        except Exception as e:
            return {"error": f"Fallback eval failed: {e}"}

    def _hash_eval_functions(self) -> str:
        """Hash evaluation functions to detect tampering."""
        eval_files = [
            "/app/app/reference_tasks.py",
            "/app/app/eval_sandbox.py",
            "/app/app/evolution_db/eval_sets.py",
        ]
        combined = ""
        for path in eval_files:
            try:
                combined += Path(path).read_text()
            except FileNotFoundError:
                pass
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    @property
    def archive(self) -> EvolutionArchive:
        return self._archive


# ── Module-level helpers ──────────────────────────────────────────────────────


_runner: ParallelEvolutionRunner | None = None
_runner_lock = threading.Lock()


def get_runner() -> ParallelEvolutionRunner:
    """Get or create the singleton parallel evolution runner."""
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = ParallelEvolutionRunner()
        return _runner


def run_parallel_evolution_cycle() -> dict:
    """Entry point for idle scheduler — run one parallel evolution cycle.

    1. Select diverse parents from archive
    2. Generate mutation proposals for each
    3. Run in parallel sandboxes
    4. Archive best result
    """
    from app.idle_scheduler import should_yield

    runner = get_runner()

    # Select diverse parents
    parents = runner.archive.get_diverse_sample(runner.MAX_PARALLEL)

    if not parents and runner.archive.size == 0:
        # Bootstrap: use current system as first archive entry
        try:
            from app.version_manifest import get_current_manifest
            manifest = get_current_manifest()
            if manifest:
                runner.archive.add(ArchiveEntry(
                    version_tag=manifest.get("version", "v0.0.0"),
                    metrics={},
                    change_description="Initial system state (bootstrap)",
                    mutation_strategy="baseline",
                    composite_score=0.5,
                ))
                parents = runner.archive.get_diverse_sample(1)
        except Exception:
            pass

    if not parents:
        return {"status": "no_parents", "message": "Archive empty, no parents to select"}

    if should_yield():
        return {"status": "yielded", "message": "User task arrived"}

    # Generate proposals (one per parent)
    proposals = []
    strategies = _suggest_strategies(runner.archive)

    for i, parent in enumerate(parents):
        strategy = strategies[i % len(strategies)] if strategies else "prompt_optimization"
        proposal = {
            "strategy": strategy,
            "parent_tag": parent.version_tag,
            "diff": "",  # Will be generated by evolution.py's proposal logic
            "description": f"Parallel mutation: {strategy} from {parent.version_tag}",
        }
        proposals.append(proposal)

    if should_yield():
        return {"status": "yielded", "message": "User task arrived"}

    # Run parallel
    candidates = runner.run_parallel_cycle(proposals)

    # Select best
    best = runner.select_best_candidate(candidates)

    result = {
        "status": "completed",
        "candidates_tested": len(candidates),
        "candidates_passed": sum(1 for c in candidates if c.passed),
        "best_candidate": None,
    }

    if best:
        # Archive the winner
        runner.archive.add(ArchiveEntry(
            version_tag=f"v-parallel-{int(time.time())}",
            metrics=best.metrics,
            change_description=best.description,
            parent_tag=best.parent_tag,
            mutation_strategy=best.mutation_strategy,
            composite_score=sum(best.metrics.get(d, 0) for d in [
                "task_completion", "user_alignment", "efficiency",
            ]),
        ))
        result["best_candidate"] = {
            "strategy": best.mutation_strategy,
            "metrics": best.metrics,
            "description": best.description,
        }

    return result


def _suggest_strategies(archive: EvolutionArchive) -> list[str]:
    """Suggest mutation strategies that are under-explored in the archive."""
    ALL_STRATEGIES = [
        "prompt_optimization",
        "cascade_rebalance",
        "rag_tuning",
        "few_shot_injection",
        "style_calibration",
        "knowledge_expansion",
        "tool_integration",
    ]

    dist = archive.get_strategy_distribution()
    # Sort by count (ascending) — least explored first
    scored = sorted(ALL_STRATEGIES, key=lambda s: dist.get(s, 0))
    return scored
