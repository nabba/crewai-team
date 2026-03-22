"""
variant_archive.py — DGM-inspired variant archive with genealogy tracking.

Each experiment is stored as a variant with a parent link, creating a tree
of evolutionary lineage. The archive supports diversity-aware selection
for the evolution agent's context (not just best-scoring variants, but
diverse branches that explored different directions).

Based on: Darwin Gödel Machine (Sakana AI / UBC, 2025) — archive
of agent variants with Darwinian selection.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Optional
from app.utils import now_iso

logger = logging.getLogger(__name__)

ARCHIVE_PATH = Path("/app/workspace/variant_archive.json")
_MAX_VARIANTS = 500


def _load() -> list[dict]:
    from app.utils import load_json_file
    return load_json_file(ARCHIVE_PATH, default=[])


def _save(variants: list[dict]) -> None:
    from app.utils import save_json_file
    save_json_file(ARCHIVE_PATH, variants, max_entries=_MAX_VARIANTS)


def add_variant(
    experiment_id: str,
    hypothesis: str,
    change_type: str,
    parent_id: str = "root",
    fitness_before: float = 0.0,
    fitness_after: float = 0.0,
    test_pass_rate: float = 0.0,
    status: str = "keep",
    files_changed: list[str] = None,
    mutation_summary: str = "",
) -> dict:
    """Add a new variant to the archive. Returns the variant dict."""
    variant = {
        "id": experiment_id,
        "parent_id": parent_id,
        "hypothesis": hypothesis[:500],
        "change_type": change_type,
        "fitness_before": round(fitness_before, 6),
        "fitness_after": round(fitness_after, 6),
        "delta": round(fitness_after - fitness_before, 6),
        "test_pass_rate": round(test_pass_rate, 4),
        "status": status,
        "files_changed": files_changed or [],
        "mutation_summary": mutation_summary[:300],
        "timestamp": now_iso(),
        "generation": 0,  # computed from parent chain
    }

    # Compute generation from parent
    archive = _load()
    parent = next((v for v in archive if v["id"] == parent_id), None)
    if parent:
        variant["generation"] = parent.get("generation", 0) + 1

    archive.append(variant)
    _save(archive)
    return variant


def get_lineage(variant_id: str) -> list[dict]:
    """Get the full ancestry chain of a variant (root → ... → variant)."""
    archive = _load()
    by_id = {v["id"]: v for v in archive}

    chain = []
    current = by_id.get(variant_id)
    while current:
        chain.append(current)
        parent_id = current.get("parent_id", "root")
        if parent_id == "root" or parent_id not in by_id:
            break
        current = by_id[parent_id]

    chain.reverse()
    return chain


def get_best_variants(n: int = 5, status_filter: str = "keep") -> list[dict]:
    """Get the top N variants by fitness score."""
    archive = _load()
    filtered = [v for v in archive if v.get("status") == status_filter]
    filtered.sort(key=lambda v: v.get("fitness_after", 0), reverse=True)
    return filtered[:n]


def get_diverse_sample(n: int = 5) -> list[dict]:
    """Get a diverse sample of variants across different branches.

    Uses hypothesis hash to ensure we pick from different evolutionary
    directions, not just the same branch.
    """
    archive = _load()
    if not archive:
        return []

    # Group by hypothesis hash prefix (first 4 chars = branch identifier)
    branches: dict[str, list[dict]] = {}
    for v in archive:
        h = hashlib.md5(v.get("hypothesis", "").encode()).hexdigest()[:4]
        branches.setdefault(h, []).append(v)

    # Pick the best variant from each branch, then take top N
    representatives = []
    for branch_variants in branches.values():
        best = max(branch_variants, key=lambda v: v.get("fitness_after", 0))
        representatives.append(best)

    representatives.sort(key=lambda v: v.get("fitness_after", 0), reverse=True)
    return representatives[:n]


def get_recent_variants(n: int = 10) -> list[dict]:
    """Get the N most recent variants."""
    archive = _load()
    return archive[-n:]


def get_last_kept_id() -> str:
    """Get the ID of the most recently kept variant (used as parent for next)."""
    archive = _load()
    for v in reversed(archive):
        if v.get("status") == "keep":
            return v["id"]
    return "root"


def get_drift_score() -> float:
    """Compute cumulative drift distance from the root.

    Counts total number of kept mutations. Higher = more evolved from baseline.
    """
    archive = _load()
    kept = [v for v in archive if v.get("status") == "keep"]
    return len(kept)


def format_archive_context(n: int = 8) -> str:
    """Format archive for evolution agent context."""
    best = get_best_variants(4)
    diverse = get_diverse_sample(4)
    recent = get_recent_variants(4)

    # Merge and deduplicate
    seen = set()
    all_variants = []
    for v in best + diverse + recent:
        if v["id"] not in seen:
            seen.add(v["id"])
            all_variants.append(v)

    if not all_variants:
        return "No experiments in archive yet."

    lines = ["## Variant Archive (DGM-style genealogy)\n"]
    for v in all_variants[:n]:
        parent = v.get("parent_id", "root")
        lines.append(
            f"  [{v['status']:7s}] gen={v.get('generation', 0)} "
            f"Δ={v['delta']:+.4f} test={v.get('test_pass_rate', 0):.0%} | "
            f"{v['hypothesis'][:60]} (parent: {parent[:12]})"
        )

    drift = get_drift_score()
    lines.append(f"\nDrift score: {drift} mutations from baseline")
    return "\n".join(lines)
