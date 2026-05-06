"""collector — orchestrate dossier adapters and merge results.

The collector is deterministic: given a :class:`CompanyRef`, it
queries every applicable adapter in parallel, merges the typed
results into a canonical :class:`CompanyDossier`, and emits each
populated field as a :class:`Claim` in the epistemic ledger.

No LLM is involved here.  The LLM enters the pipeline only at the
composition stage — by which point the dossier is frozen.

Concurrency contract
====================
* Adapters fire in parallel up to ``max_parallel`` (default 4).
* Each adapter call has a per-call wallclock cap (``per_call_timeout``,
  default 25s).  A timed-out adapter does not block the rest.
* A circuit breaker protects against an adapter that throws
  repeatedly: after ``max_consecutive_failures`` strikes the adapter
  is short-circuited for the rest of the call.
* Iterative enrichment: after each adapter completes, ref enrichments
  (e.g. SEC CIK discovered via Wikidata) are folded back so that
  later adapters benefit.  This is bounded to one extra pass to
  prevent loops.

Reuse
=====
* :func:`schema.merge_field` — picks winning value, records conflicts
* :class:`epistemic.Ledger` — every populated field becomes a Claim
* The circuit-breaker shape mirrors
  :class:`research_orchestrator._DomainBreaker` but lives here so we
  don't depend on private state of another module.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from app.dossier.adapters import (
    DossierAdapter,
    DossierAdapterResult,
    all_adapters,
    clear_cache,
    install_defaults,
    source_priority_map,
)
from app.dossier.schema import (
    CompanyDossier,
    CompanyRef,
    Confidence,
    DossierField,
    FieldStatus,
    Source,
    merge_field,
)

logger = logging.getLogger(__name__)


# ── Circuit breaker ──────────────────────────────────────────────────


@dataclass
class _AdapterBreaker:
    """Per-adapter consecutive-failure counter.

    Mirrors :class:`research_orchestrator._DomainBreaker` (private
    over there; we keep our own to avoid coupling).
    """

    max_consecutive_failures: int = 2
    failures: dict[str, int] = field(default_factory=dict)
    tripped: dict[str, str] = field(default_factory=dict)

    def record_success(self, adapter: str) -> None:
        self.failures[adapter] = 0

    def record_failure(self, adapter: str, reason: str) -> None:
        self.failures[adapter] = self.failures.get(adapter, 0) + 1
        if self.failures[adapter] >= self.max_consecutive_failures:
            self.tripped[adapter] = reason

    def is_tripped(self, adapter: str) -> bool:
        return adapter in self.tripped


# ── Result envelope ──────────────────────────────────────────────────


@dataclass
class CollectionReport:
    """Telemetry for one dossier collection run."""

    elapsed_seconds: float = 0.0
    adapters_fired: list[str] = field(default_factory=list)
    adapters_skipped: dict[str, str] = field(default_factory=dict)
    adapters_errored: dict[str, str] = field(default_factory=dict)
    adapters_tripped: dict[str, str] = field(default_factory=dict)
    fields_filled: int = 0
    fields_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "adapters_fired": list(self.adapters_fired),
            "adapters_skipped": dict(self.adapters_skipped),
            "adapters_errored": dict(self.adapters_errored),
            "adapters_tripped": dict(self.adapters_tripped),
            "fields_filled": self.fields_filled,
            "fields_total": self.fields_total,
            "coverage_pct": (
                round(self.fields_filled / self.fields_total * 100, 1)
                if self.fields_total else 0.0
            ),
        }


# ── Public entry point ───────────────────────────────────────────────


def collect_dossier(
    ref: CompanyRef,
    *,
    task_id: str | None = None,
    max_parallel: int = 4,
    per_call_timeout: float = 25.0,
    enrichment_passes: int = 2,
) -> CompanyDossier:
    """Build a complete :class:`CompanyDossier` for ``ref``.

    Args:
        ref: Identity of the company.  Mutated in place to absorb
            adapter ref-enrichments (CIK, Wikidata QID, etc.).
        task_id: When provided, every populated field is emitted as a
            :class:`Claim` in a per-task :class:`Ledger`.  None = skip
            ledger emission (useful for tests and isolated calls).
        max_parallel: Number of adapters firing concurrently.  Lower
            on slow networks; higher only if adapters are mostly cached.
        per_call_timeout: Hard wallclock cap per adapter call.  Wider
            than the orchestrator's 20s default because some adapters
            (SEC companyfacts) genuinely need 15-20s on a cold fetch.
        enrichment_passes: Re-runs after the first pass.  After each
            pass, adapters that previously couldn't_collect (because
            of missing identifiers) are re-checked.  Default 2 = first
            pass + one re-pass after enrichment.

    Returns:
        A populated dossier whose ``coverage_report`` field summarises
        which adapters fired, which fields are missing, and why.
    """
    install_defaults()
    clear_cache()

    started = time.monotonic()
    dossier = CompanyDossier(ref=ref)
    report = CollectionReport()
    breaker = _AdapterBreaker()

    fired_already: set[str] = set()
    for pass_num in range(enrichment_passes):
        eligible = [
            a for a in all_adapters()
            if a.name not in fired_already
            and not breaker.is_tripped(a.name)
            and a.is_configured()
            and a.can_collect(ref)
        ]
        if not eligible:
            break
        results = _run_parallel(
            eligible, ref,
            max_parallel=max_parallel,
            per_call_timeout=per_call_timeout,
            breaker=breaker,
        )

        # Merge results in priority order so the highest-priority adapter
        # anchors each field; lower-priority writes record conflicts.
        priority = source_priority_map()
        results.sort(
            key=lambda r: -priority.get(r.adapter_name, 0),
        )
        for result in results:
            if not result.is_ok():
                report.adapters_errored[result.adapter_name] = result.error
                continue
            report.adapters_fired.append(result.adapter_name)
            fired_already.add(result.adapter_name)
            _apply_result(dossier, result, source_priority=priority)
            _apply_ref_enrichment(ref, result.ref_enrichment)

        # Record adapters that we tried but were skipped at the gate.
        for adapter in eligible:
            if adapter.name in fired_already:
                continue
            if adapter.name in report.adapters_errored:
                continue
            report.adapters_skipped[adapter.name] = "did not return"

    # Adapters that never got eligible (no identity, not configured).
    for adapter in all_adapters():
        if adapter.name in fired_already:
            continue
        if adapter.name in report.adapters_skipped:
            continue
        if adapter.name in report.adapters_errored:
            continue
        if not adapter.is_configured():
            report.adapters_skipped[adapter.name] = "not configured"
        elif not adapter.can_collect(ref):
            report.adapters_skipped[adapter.name] = "ref lacks required identity"

    report.adapters_tripped = dict(breaker.tripped)
    report.elapsed_seconds = time.monotonic() - started
    report.fields_total = dossier.total_field_count()
    report.fields_filled = dossier.known_field_count()

    dossier.coverage_report = report.to_dict()

    # Optional: emit each populated field as a Claim in the ledger.
    if task_id:
        _emit_to_ledger(dossier, task_id=task_id)

    return dossier


# ── Internal helpers ─────────────────────────────────────────────────


def _run_parallel(
    adapters: list[DossierAdapter],
    ref: CompanyRef,
    *,
    max_parallel: int,
    per_call_timeout: float,
    breaker: _AdapterBreaker,
) -> list[DossierAdapterResult]:
    """Fire adapters concurrently, honouring per-call timeout + breaker."""
    out: list[DossierAdapterResult] = []
    if not adapters:
        return out
    with ThreadPoolExecutor(
        max_workers=min(max_parallel, len(adapters)),
        thread_name_prefix="dossier-adapter",
    ) as pool:
        futures: dict[Future, DossierAdapter] = {
            pool.submit(_safe_collect, adapter, ref): adapter
            for adapter in adapters
        }
        for future, adapter in futures.items():
            try:
                result = future.result(timeout=per_call_timeout)
            except TimeoutError:
                breaker.record_failure(adapter.name, "timeout")
                out.append(DossierAdapterResult(
                    adapter_name=adapter.name,
                    error=f"timeout after {per_call_timeout:.0f}s",
                ))
                continue
            except Exception as exc:
                breaker.record_failure(
                    adapter.name, f"{type(exc).__name__}: {str(exc)[:200]}",
                )
                out.append(DossierAdapterResult(
                    adapter_name=adapter.name,
                    error=f"{type(exc).__name__}: {exc}",
                ))
                continue
            if result.is_ok():
                breaker.record_success(adapter.name)
            else:
                breaker.record_failure(adapter.name, result.error)
            out.append(result)
    return out


def _safe_collect(adapter: DossierAdapter,
                  ref: CompanyRef) -> DossierAdapterResult:
    """Wrap ``adapter.collect`` so an adapter that violates the
    no-raise contract still produces a structured failure."""
    try:
        return adapter.collect(ref)
    except Exception as exc:
        return DossierAdapterResult(
            adapter_name=adapter.name,
            error=f"adapter raised {type(exc).__name__}: {str(exc)[:200]}",
        )


def _apply_result(
    dossier: CompanyDossier,
    result: DossierAdapterResult,
    *,
    source_priority: dict[str, int],
) -> None:
    """Merge one adapter result into the dossier."""
    schema_fields = type(dossier).model_fields
    for upd in result.fields:
        if upd.field_name not in schema_fields:
            logger.debug(
                "dossier: adapter %s tried to fill unknown field %r — ignoring",
                result.adapter_name, upd.field_name,
            )
            continue
        existing = getattr(dossier, upd.field_name)
        if not isinstance(existing, DossierField):
            continue  # not a typed dossier field; skip
        source = upd.materialize_source(
            adapter=result.adapter_name,
            url=result.base_url,
            document_id="",
        )
        merged = merge_field(
            existing=existing,
            new_value=upd.value,
            source=source,
            confidence=upd.confidence,
            as_of=upd.as_of,
            source_priority=source_priority,
        )
        setattr(dossier, upd.field_name, merged)


def _apply_ref_enrichment(ref: CompanyRef,
                          enrichment: dict[str, str]) -> None:
    """Fold an adapter's discovered identifiers back into the ref.

    Only fills empty fields — never overwrites an explicit caller value.
    """
    for key, value in (enrichment or {}).items():
        if not hasattr(ref, key):
            continue
        if not value:
            continue
        if not getattr(ref, key, ""):
            setattr(ref, key, value)


def _emit_to_ledger(dossier: CompanyDossier, *, task_id: str) -> None:
    """Emit each populated field as a Claim in the epistemic ledger.

    Best-effort — a ledger import / persistence failure does NOT
    invalidate the dossier.  The dossier remains the canonical
    in-memory artefact; the ledger is auxiliary auditability.
    """
    try:
        from app.epistemic.ledger import (
            Claim, Evidence, Ledger, Register, VerificationStatus,
        )
    except Exception:
        logger.debug("dossier: epistemic ledger unavailable; skipping emission")
        return

    try:
        ledger = Ledger(task_id=task_id)
    except Exception as exc:
        logger.debug("dossier: ledger init failed: %s", exc)
        return

    for field_name, dfield in dossier.iter_fields():
        if not dfield.is_known or dfield.source is None:
            continue
        try:
            statement = (
                f"{dossier.ref.name}.{field_name} = {dfield.render_value()}"
            )
            evidence = (Evidence(
                kind="tool_call",
                source_ref=dfield.source.url or dfield.source.adapter,
                excerpt=dfield.source.note or dfield.source.adapter,
                confidence=dfield.confidence.to_evidence_confidence(),
            ),)
            ledger.emit(Claim.new(
                task_id=task_id,
                agent_role="dossier_collector",
                statement=statement,
                status=(
                    VerificationStatus.VERIFIED
                    if dfield.confidence in (Confidence.EXACT, Confidence.HIGH)
                    else VerificationStatus.INFERRED
                ),
                register=Register.INTERNAL,
                evidence=evidence,
                load_bearing=True,
                tags=("dossier", dfield.source.adapter, field_name),
            ))
        except Exception:
            # Per-claim failure is local; keep going so other fields
            # still land.  We log at debug to avoid noise.
            logger.debug("dossier: ledger emit failed for %s", field_name,
                         exc_info=True)
