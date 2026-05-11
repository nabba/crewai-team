"""Pre-cutover invariants and human-readable readiness report.

PROGRAM §40 (2026-05-10) — Q3 Item 12.

Verification hierarchy (each must pass before cutover unblocks):

  1. **Plan loaded** — ``plan.json`` exists and is parseable.
  2. **Phase ready** — state machine is in ``READY``.
  3. **Counters healthy** — ``shadow_query_count >= cutover_min_shadow_queries``.
  4. **NDCG threshold met** — last rolling-window mean
     ``>= cutover_threshold_ndcg``.
  5. **Per-collection invariants** — for every chromadb target:
        a. Source + shadow both exist.
        b. Shadow row count is within ±1% of source row count.
        c. peek(1) on shadow returns rows of ``target.dim``.
  6. **DR backup fresh** — there exists a DR tarball under
     ``workspace/backups/dr/`` < 7 days old.

Returns a typed report. The ``ok`` field is the cutover gate.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.memory.embedding_migration import plan as plan_mod
from app.memory.embedding_migration import state as state_mod
from app.memory.embedding_migration.dual_write import shadow_collection_name

logger = logging.getLogger(__name__)


@dataclass
class VerifyCheck:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyReport:
    plan_id: str | None = None
    phase: str = ""
    checks: list[VerifyCheck] = field(default_factory=list)
    ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "phase": self.phase,
            "checks": [c.to_dict() for c in self.checks],
            "ok": self.ok,
        }


def _check_dr_freshness() -> VerifyCheck:
    dr_dir = Path("/app/workspace/backups/dr")
    if not dr_dir.exists():
        return VerifyCheck(
            "dr_backup_fresh", False,
            "no DR backup directory exists; run python -m app.dr.export_kbs",
        )
    candidates = sorted(
        dr_dir.glob("dr_*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return VerifyCheck(
            "dr_backup_fresh", False,
            "no DR tarball found; run python -m app.dr.export_kbs",
        )
    latest = candidates[0]
    age_s = time.time() - latest.stat().st_mtime
    age_days = age_s / 86400.0
    if age_days > 7:
        return VerifyCheck(
            "dr_backup_fresh", False,
            f"latest DR tarball is {age_days:.1f} days old (≥7d limit); "
            f"run python -m app.dr.boot_drill --export-fresh",
        )
    return VerifyCheck(
        "dr_backup_fresh", True,
        f"latest DR tarball: {latest.name} ({age_days:.1f}d old)",
    )


def _check_collection_invariants(plan) -> list[VerifyCheck]:
    checks: list[VerifyCheck] = []
    try:
        from app.memory import chromadb_manager
        client = chromadb_manager.get_client()
    except Exception as exc:
        return [VerifyCheck(
            "chromadb_client", False,
            f"cannot open chromadb client: {exc}",
        )]

    for target in plan.targets:
        if target.kind != "chromadb":
            continue
        col_name = target.collection or "?"
        shadow_name = shadow_collection_name(col_name, plan.plan_id)
        # Source exists?
        try:
            src = client.get_collection(col_name)
            src_count = src.count()
        except Exception as exc:
            checks.append(VerifyCheck(
                f"source_exists::{col_name}", False, f"source missing: {exc}",
            ))
            continue
        # Shadow exists?
        try:
            shadow = client.get_collection(shadow_name)
            shadow_count = shadow.count()
        except Exception as exc:
            checks.append(VerifyCheck(
                f"shadow_exists::{col_name}", False,
                f"shadow missing: {exc}",
            ))
            continue
        # Row count within ±1%?
        if src_count == 0:
            ratio_ok = shadow_count == 0
        else:
            delta = abs(src_count - shadow_count) / src_count
            ratio_ok = delta <= 0.01
        checks.append(VerifyCheck(
            f"shadow_row_match::{col_name}", ratio_ok,
            f"source={src_count} shadow={shadow_count}",
        ))
        # Shadow peek dim
        try:
            sample = shadow.peek(1)
            embs = sample.get("embeddings") if sample else None
            if embs and embs[0] and len(embs[0]) == plan.target.dim:
                checks.append(VerifyCheck(
                    f"shadow_dim::{col_name}", True,
                    f"observed dim={len(embs[0])}",
                ))
            else:
                got = len(embs[0]) if (embs and embs[0]) else 0
                checks.append(VerifyCheck(
                    f"shadow_dim::{col_name}", False,
                    f"expected={plan.target.dim} got={got}",
                ))
        except Exception as exc:
            checks.append(VerifyCheck(
                f"shadow_dim::{col_name}", False, f"peek failed: {exc}",
            ))

    return checks


def verify() -> VerifyReport:
    report = VerifyReport()

    plan = plan_mod.load_plan()
    if plan is None:
        report.checks.append(VerifyCheck(
            "plan_loaded", False, "no plan at workspace/embedding_migration/plan.json",
        ))
        report.ok = False
        return report
    report.plan_id = plan.plan_id
    report.checks.append(VerifyCheck("plan_loaded", True, plan.plan_id))

    cur_state = state_mod.get_state()
    report.phase = cur_state.phase
    if cur_state.phase != state_mod.PHASE_READY:
        report.checks.append(VerifyCheck(
            "phase_ready", False,
            f"phase {cur_state.phase}; need {state_mod.PHASE_READY}",
        ))
    else:
        report.checks.append(VerifyCheck("phase_ready", True, ""))

    counters = cur_state.counters
    if counters.shadow_query_count >= plan.cutover_min_shadow_queries:
        report.checks.append(VerifyCheck(
            "shadow_sample_size", True,
            f"{counters.shadow_query_count} ≥ {plan.cutover_min_shadow_queries}",
        ))
    else:
        report.checks.append(VerifyCheck(
            "shadow_sample_size", False,
            f"{counters.shadow_query_count} < "
            f"{plan.cutover_min_shadow_queries}",
        ))

    if (
        counters.last_ndcg_at_10 is not None
        and counters.last_ndcg_at_10 >= plan.cutover_threshold_ndcg
    ):
        report.checks.append(VerifyCheck(
            "ndcg_threshold", True,
            f"{counters.last_ndcg_at_10:.4f} ≥ {plan.cutover_threshold_ndcg}",
        ))
    else:
        report.checks.append(VerifyCheck(
            "ndcg_threshold", False,
            f"current={counters.last_ndcg_at_10}; "
            f"need ≥ {plan.cutover_threshold_ndcg}",
        ))

    report.checks.extend(_check_collection_invariants(plan))
    report.checks.append(_check_dr_freshness())

    report.ok = all(c.ok for c in report.checks)
    return report
