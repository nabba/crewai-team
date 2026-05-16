# Dependency radar (§2.3)

**Status (2026-05-16):** Shipped at PROGRAM §48 — Q13.2.

A weekly HEAVY idle that audits CURRENTLY-INSTALLED Python
dependencies along three axes. Complements `app/library_radar/`
which discovers NEW libraries to ADD — the dependency radar is the
INBOUND axis (security + abandonment of what we already have).

## Three signals collected per pass

| Signal | Source | Failure mode |
|---|---|---|
| **Outdated** | `pip list --outdated --format=json` subprocess, 60s timeout | pip unavailable → empty list, skip |
| **CVE** | OSV.dev `/v1/querybatch` POST with all (package, version) tuples | network error → empty dict, skip |
| **Abandoned** | `pip show <pkg>` for Home-page; `api.github.com/repos/{owner}/{repo}` for `pushed_at` | rate-limit → empty dict, skip |

All collectors are independently failure-isolated: one signal's
failure never prevents the others from routing.

## Severity classification

`_classify_bump(current, latest)` returns one of:

  * `PATCH` (1.2.3 → 1.2.4) — major + minor agree
  * `MINOR` (1.2.3 → 1.3.0) — major agrees
  * `MAJOR` (1.x → 2.0) — major differs
  * Anything with non-parseable parts → `MAJOR` (operator review)

Then OSV findings override:

  * If CVE has a patched version → severity becomes `CVE`
  * If CVE has no patched version → severity becomes `CVE_NO_FIX`

Plus a derived severity:

  * Repo last-pushed > 365 days ago → `ABANDONED`

## Routing matrix

| Severity | Route | Cooldown |
|---|---|---|
| `PATCH` | `proposal_bridge.store.stage()` → standard operator-gated CR | 7d |
| `MINOR` | `proposal_bridge.store.stage()` → standard operator-gated CR | 14d |
| `MAJOR` | Signal alert ONLY (no CR) — operator schedules migration window | n/a |
| `CVE` (patched version available) | `proposal_bridge.store.stage()` at **highest priority** | 3d |
| `CVE_NO_FIX` | Signal alert ONLY (`dep_cve_nofix:<pkg>` topic) | n/a |
| `ABANDONED` | Signal alert ONLY (`dep_abandoned:<pkg>` topic, once/week dedup) | n/a |

Rate limit: `_MAX_PROPOSALS_PER_PASS = 3` (matches the
proposal_bridge default rate). Excess findings carry over to next
week's pass.

## CR body shape

Each CR proposes a single line change to `requirements.txt`:

```markdown
# Dependency bump proposal

**Package:** `requests`
**Current:** `2.31.0`
**Proposed:** `2.31.1`
**Severity:** patch
**CVEs:** GHSA-... (when applicable)

## What this proposes

Update `requirements.txt` to pin `requests==2.31.1`. The operator
can then `pip install -r requirements.txt` and run the test suite.

## Why this is auto-proposed

Detected by the weekly `dependency_radar` HEAVY idle. The radar
files patch-level + CVE-patch CRs through the standard operator-
gated change-request flow (this CR). Major-version bumps surface
as Signal alerts ONLY — those require a deliberate migration
window.

## Prior precedent
<inline if lessons_learned has a match for this package>

## Disclaimer

This proposal does NOT actually install the new version in any
environment. It is a markdown diff against `requirements.txt`.
The operator approves via /cp/changes; only then does the change
land.
```

## Lessons-learned integration

Before staging, the proposer queries `companion.lessons_learned.
check_against(...)` for prior failed bumps of the same package.
On similarity ≥0.6, the precedent is surfaced inline in the CR
body so the operator sees "we tried this last quarter; it broke X."

## Master switch + wiring

  * Runtime setting `dependency_radar_enabled` (default ON;
    React `/cp/settings` toggle).
  * Boot-anchored in `app/healing/__init__.py` (same pattern as
    `library_radar` + `proposal_bridge`).
  * `proposal_bridge.store._KNOWN_SOURCES` includes
    `"dependency_radar"` so the staging tree at
    `workspace/proposal_bridge/dependency_radar/` validates.
  * Daemon thread eager-starts at module import via gated
    `start_daemon()`; cadence 7 days, warm-up 120s.

## What this doesn't do

  * **No automatic installation.** The CR is a markdown diff; the
    operator runs `pip install` themselves.
  * **No SBOM tracking** (CycloneDX / SPDX). Out of scope for
    year-2 resilience; the three-axis signal set covers the
    operator's stated concerns.
  * **No transitive-dependency CVE check** in v1. OSV's batch API
    returns CVEs only for the top-level pins; transitive CVEs
    would require `pip-audit` integration. Deferred.
  * **No GitHub auth.** Unauthenticated rate limit (60/hr) is fine
    for ~100 top-level deps once a week.

## See also

  * `app/dependency_radar/proposer.py` — implementation (~700 LOC)
  * `app/proposal_bridge/` — the CR-emitting pipeline this composes with
  * `app/library_radar/` — the OUTBOUND discovery sibling
  * `tests/test_q13_resilience_year2.py` — 13 dependency-radar tests
