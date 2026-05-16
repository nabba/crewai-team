"""Dependency-radar — weekly inbound-dependency health scan.

PROGRAM §48 — Q13.2 (year-2+ resilience #2.3). Complements
:mod:`app.library_radar` (which discovers NEW libraries to ADD).
This radar audits CURRENTLY-INSTALLED dependencies along three axes:

  1. **Outdated** (pip list --outdated --format=json).
  2. **Insecure** (OSV.dev CVE scan, batch API).
  3. **Abandoned** (GitHub repos/{owner}/{repo} pushed_at >365d).

Routing (matches user §2.3 spec):

  * **Patch-level bump** → CR via :mod:`app.proposal_bridge` (7d
    cooldown, 3-per-pass rate limit, weighted GW publish, standard
    operator gate at /cp/changes).
  * **Minor-version bump** → CR via proposal_bridge BUT with a
    longer cooldown (14d) so the operator has more time to
    consider; aggregated digest in Signal.
  * **Major-version bump** → Signal alert ONLY (no CR). The
    operator decides whether to schedule a major-version
    migration drill.
  * **CVE with patched version available** → CR via proposal_bridge
    at HIGHEST priority (3d cooldown). Topic-keyed Signal alert
    referencing the CVE id.
  * **CVE with NO patched version** → Signal alert only;
    operator decides on workaround.
  * **Repo abandonment** (≥365d since last push) → Signal alert
    once per week per package; no CR (replacement requires
    architectural decision).

Lessons-learned integration: before staging a proposal, consult
:mod:`app.self_improvement.lessons_learned` for prior failed
bumps of the same package. If similarity ≥0.6, surface the prior
failure inline in the CR body so the operator sees the precedent.

Master switch: ``dependency_radar_enabled`` (default ON).
Cadence: weekly (Sunday 03:00 UTC by default).

Public surface:

  * :func:`app.dependency_radar.proposer.run_one_pass` — pure
    function the daemon calls. Returns a :class:`RadarResult`.
  * :func:`app.dependency_radar.proposer.start_daemon` — boot
    anchor (called from :mod:`app.healing.__init__`).
  * :func:`app.dependency_radar.proposer._gather_outdated` /
    :func:`_gather_cves` / :func:`_gather_abandonment` —
    individual collectors (injectable for tests).
"""
from app.dependency_radar.proposer import (
    RadarFinding,
    RadarResult,
    Severity,
    run_one_pass,
    start_daemon,
)

__all__ = [
    "RadarFinding",
    "RadarResult",
    "Severity",
    "run_one_pass",
    "start_daemon",
]
