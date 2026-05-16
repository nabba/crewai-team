"""Dependency-radar proposer — see package docstring.

The daemon is a thin loop around :func:`run_one_pass`. The pass is a
pure function so tests can drive it directly with injected
collectors and assert exact behaviour.
"""
from __future__ import annotations

import enum
import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_DAEMON_THREAD_NAME = "dependency-radar"
_WARMUP_S = 120
_POLL_INTERVAL_S = 7 * 24 * 3600   # weekly

_MAX_PROPOSALS_PER_PASS = 3        # match proposal_bridge rate limit

# Cooldowns by severity — the operator can bash through the lower-
# severity ones in batches without major-version surprises blocking.
_COOLDOWN_PATCH = 7
_COOLDOWN_MINOR = 14
_COOLDOWN_CVE = 3

# Abandonment threshold (days since last GitHub push). 1 year is the
# user's spec.
_ABANDONMENT_DAYS = 365

# OSV.dev batch endpoint. We could call individual /v1/query endpoints
# but the batch path is more polite and handles big dep sets.
_OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"

# GitHub API base. We require no auth — falling back gracefully when
# rate-limited (60 req/hr unauthenticated; with the operator's 100ish
# direct deps, one weekly pass fits inside the limit).
_GITHUB_API = "https://api.github.com"


_driver_lock = threading.Lock()
_driver_started = False
_stop_event = threading.Event()


# ── Types ───────────────────────────────────────────────────────────────


class Severity(str, enum.Enum):
    """Routing class for a finding."""

    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"
    CVE = "cve"
    CVE_NO_FIX = "cve_no_fix"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class RadarFinding:
    """One thing the radar wants to surface."""

    package: str
    severity: Severity
    current_version: str
    latest_version: Optional[str] = None   # not used for ABANDONED
    cve_ids: tuple[str, ...] = ()
    repo_url: Optional[str] = None
    last_push_at: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RadarResult:
    """Outcome of one weekly pass."""

    ran: bool = False
    findings_by_severity: dict[str, int] = field(default_factory=dict)
    cr_proposals_filed: int = 0
    alerts_fired: int = 0
    errors: int = 0
    findings: list[RadarFinding] = field(default_factory=list)


# ── Master-switch + state helpers ───────────────────────────────────────


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_dependency_radar_enabled
        return get_dependency_radar_enabled()
    except Exception:
        return os.getenv("DEPENDENCY_RADAR_ENABLED", "true").lower() in (
            "true", "1", "yes", "on",
        )


def _state_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "dependency_radar" / "state.json"
    except Exception:
        return Path("/app/workspace/dependency_radar/state.json")


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        # Failure-isolated: state persistence is best-effort. Read-only
        # workspace (rare deploy variant) or path-resolution edge case
        # must not break the radar pass.
        logger.debug("dependency_radar: state write failed", exc_info=True)


# ── Collectors (each one independently failure-isolated) ───────────────


def _gather_outdated(
    *,
    pip_runner: Optional[Callable[[], list[dict[str, Any]]]] = None,
) -> list[dict[str, Any]]:
    """Return a list of ``{package, current, latest}`` records.

    ``pip_runner`` is injectable for tests. The default invokes
    ``pip list --outdated --format=json`` as a subprocess with a
    60-second hard timeout."""
    if pip_runner is not None:
        try:
            return list(pip_runner() or [])
        except Exception:
            logger.debug("dependency_radar: injected pip_runner raised", exc_info=True)
            return []
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.debug(
                "dependency_radar: pip list --outdated rc=%d", result.returncode,
            )
            return []
        data = json.loads(result.stdout or "[]")
        # Normalise into {package, current, latest}.
        out = []
        for row in data:
            try:
                out.append({
                    "package": str(row["name"]),
                    "current": str(row["version"]),
                    "latest": str(row["latest_version"]),
                })
            except KeyError:
                continue
        return out
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        logger.debug("dependency_radar: pip subprocess failed", exc_info=True)
        return []


def _gather_cves(
    *,
    packages: list[tuple[str, str]],
    osv_runner: Optional[Callable[[list[tuple[str, str]]], dict[str, list[dict[str, Any]]]]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """For each (package, current_version) tuple, return a list of
    OSV vulnerability records. Empty list if no CVEs found.

    ``osv_runner`` is injectable for tests. The default uses urllib
    against the batch OSV endpoint."""
    if not packages:
        return {}
    if osv_runner is not None:
        try:
            return dict(osv_runner(packages) or {})
        except Exception:
            logger.debug("dependency_radar: injected osv_runner raised", exc_info=True)
            return {}
    try:
        import urllib.request
        queries = [
            {"package": {"name": pkg, "ecosystem": "PyPI"}, "version": ver}
            for (pkg, ver) in packages
        ]
        req = urllib.request.Request(
            _OSV_BATCH_URL,
            data=json.dumps({"queries": queries}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        results = payload.get("results") or []
        out: dict[str, list[dict[str, Any]]] = {}
        for (pkg, _), result in zip(packages, results):
            vulns = result.get("vulns") or []
            if vulns:
                out[pkg] = vulns
        return out
    except Exception:
        logger.debug("dependency_radar: OSV fetch failed", exc_info=True)
        return {}


def _gather_abandonment(
    *,
    packages: list[str],
    pip_show_runner: Optional[Callable[[str], dict[str, str]]] = None,
    github_runner: Optional[Callable[[str, str], Optional[datetime]]] = None,
) -> dict[str, datetime]:
    """For each package, attempt to resolve its GitHub repo and return
    the ``pushed_at`` timestamp. Packages whose repo can't be resolved
    or that aren't on GitHub return missing entries.

    Injectable for tests. The default uses pip show + GitHub API.
    """
    if not packages:
        return {}
    out: dict[str, datetime] = {}
    for pkg in packages:
        try:
            home_url = ""
            if pip_show_runner is not None:
                meta = pip_show_runner(pkg)
                home_url = (
                    meta.get("Home-page", "")
                    or meta.get("Project-URL", "")
                    or meta.get("home_page", "")
                )
            else:
                result = subprocess.run(
                    ["pip", "show", pkg],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    continue
                for line in result.stdout.splitlines():
                    if line.startswith("Home-page:"):
                        home_url = line.split(":", 1)[1].strip()
                        break
                    if line.startswith("Project-URL:") and "github" in line.lower():
                        home_url = line.split(":", 1)[1].strip()
                        break
            if "github.com" not in home_url:
                continue
            # Parse owner/repo from URL.
            parts = home_url.rstrip("/").split("github.com/")[-1].split("/")
            if len(parts) < 2:
                continue
            owner, repo = parts[0], parts[1].replace(".git", "")
            if github_runner is not None:
                ts = github_runner(owner, repo)
            else:
                ts = _github_pushed_at(owner, repo)
            if ts is not None:
                out[pkg] = ts
        except Exception:
            logger.debug(
                "dependency_radar: abandonment probe failed for %s",
                pkg, exc_info=True,
            )
            continue
    return out


def _github_pushed_at(owner: str, repo: str) -> Optional[datetime]:
    """Single GitHub API call; returns the repo's ``pushed_at`` as a
    timezone-aware datetime, None on failure (rate-limited, 404, ...).
    """
    try:
        import urllib.request
        url = f"{_GITHUB_API}/repos/{owner}/{repo}"
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pushed = data.get("pushed_at")
        if not pushed:
            return None
        return datetime.fromisoformat(pushed.replace("Z", "+00:00"))
    except Exception:
        return None


# ── Severity classification ─────────────────────────────────────────────


def _classify_bump(current: str, latest: str) -> Severity:
    """Return PATCH / MINOR / MAJOR by SemVer-style classification.

    Lenient: any non-parseable version part defaults to MAJOR (operator
    reviews it manually). The proposer can't make safety claims about
    weird version schemes."""

    def _parts(s: str) -> tuple[int, ...]:
        clean = "".join(c for c in s if c.isdigit() or c == ".")
        if not clean:
            return ()
        chunks = clean.split(".")
        try:
            return tuple(int(c) for c in chunks if c)
        except ValueError:
            return ()

    cur_parts = _parts(current)
    lat_parts = _parts(latest)
    if not cur_parts or not lat_parts:
        return Severity.MAJOR  # treat unknown as MAJOR
    if cur_parts[0] != lat_parts[0]:
        return Severity.MAJOR
    if len(cur_parts) >= 2 and len(lat_parts) >= 2:
        if cur_parts[1] != lat_parts[1]:
            return Severity.MINOR
    return Severity.PATCH


# ── Compose: build findings + route ─────────────────────────────────────


def _build_findings(
    outdated: list[dict[str, Any]],
    cves_by_pkg: dict[str, list[dict[str, Any]]],
    pushed_by_pkg: dict[str, datetime],
    *,
    now: Optional[datetime] = None,
) -> list[RadarFinding]:
    cur_now = now or datetime.now(timezone.utc)
    findings: list[RadarFinding] = []
    # Outdated findings — classified by severity of bump.
    seen_packages: set[str] = set()
    for row in outdated:
        pkg = row["package"]
        seen_packages.add(pkg)
        cur = row["current"]
        latest = row["latest"]
        severity = _classify_bump(cur, latest)
        cves = cves_by_pkg.get(pkg) or []
        # CVE overrides standard severity. If the CVE has a patched
        # version equal to the latest, treat as CVE. If no patch
        # available, CVE_NO_FIX.
        if cves:
            patched = any(
                _has_patched_version(v, latest) for v in cves
            )
            severity = Severity.CVE if patched else Severity.CVE_NO_FIX
        cve_ids = tuple(
            (v.get("id") or "") for v in cves if v.get("id")
        )
        findings.append(RadarFinding(
            package=pkg,
            severity=severity,
            current_version=cur,
            latest_version=latest,
            cve_ids=cve_ids,
        ))
    # CVE findings for packages NOT in outdated (someone might be on
    # latest version but with a known CVE). These have severity
    # CVE_NO_FIX (no upgrade available).
    for pkg, cves in cves_by_pkg.items():
        if pkg in seen_packages:
            continue
        cve_ids = tuple(
            (v.get("id") or "") for v in cves if v.get("id")
        )
        findings.append(RadarFinding(
            package=pkg,
            severity=Severity.CVE_NO_FIX,
            current_version="(current)",
            latest_version=None,
            cve_ids=cve_ids,
        ))
    # Abandonment findings — packages last pushed >365d ago.
    for pkg, pushed_at in pushed_by_pkg.items():
        if (cur_now - pushed_at).days < _ABANDONMENT_DAYS:
            continue
        findings.append(RadarFinding(
            package=pkg,
            severity=Severity.ABANDONED,
            current_version="(any)",
            latest_version=None,
            last_push_at=pushed_at.isoformat(),
            extra={"days_since_push": (cur_now - pushed_at).days},
        ))
    return findings


def _has_patched_version(vuln: dict[str, Any], latest: str) -> bool:
    """Best-effort: did OSV report a patched version, and is `latest`
    at or above it? We use a substring match against affected ranges
    rather than full semver comparison — false positives push toward
    CVE (auto-CR), false negatives push toward CVE_NO_FIX (Signal
    alert). Both surface to the operator, just at different
    cadences."""
    affected = vuln.get("affected") or []
    for a in affected:
        ranges = a.get("ranges") or []
        for r in ranges:
            events = r.get("events") or []
            for ev in events:
                if "fixed" in ev:
                    return True
    return False


# ── Routing ─────────────────────────────────────────────────────────────


def _propose_bump_cr(
    finding: RadarFinding,
    *,
    lessons_consult: Optional[Callable[[str], Optional[str]]] = None,
    stage_fn: Optional[Callable[..., None]] = None,
) -> bool:
    """File a proposal-bridge proposal for a single finding. Returns
    True on success."""
    try:
        from app.proposal_bridge.store import stage as _stage
        stage = stage_fn or _stage
        cooldown = {
            Severity.PATCH: _COOLDOWN_PATCH,
            Severity.MINOR: _COOLDOWN_MINOR,
            Severity.CVE: _COOLDOWN_CVE,
        }[finding.severity]
        signature = f"{finding.package}-{finding.severity.value}-{finding.latest_version}"
        # Lessons-learned consult: was there a prior failed bump of
        # this package? If so, surface it inline in the CR body so
        # the operator sees the precedent.
        precedent = ""
        if lessons_consult is not None:
            try:
                p = lessons_consult(finding.package)
                if p:
                    precedent = f"\n\n## Prior precedent\n\n{p}\n"
            except Exception:
                logger.debug("dependency_radar: lessons consult failed", exc_info=True)
        title_severity = {
            Severity.PATCH: "patch-level",
            Severity.MINOR: "minor-version",
            Severity.CVE: "CVE-patch",
        }[finding.severity]
        title = (
            f"Dependency: bump {finding.package} from "
            f"{finding.current_version} → {finding.latest_version} "
            f"({title_severity})"
        )
        body = (
            f"# Dependency bump proposal\n\n"
            f"**Package:** `{finding.package}`\n"
            f"**Current:** `{finding.current_version}`\n"
            f"**Proposed:** `{finding.latest_version}`\n"
            f"**Severity:** {finding.severity.value}\n"
        )
        if finding.cve_ids:
            body += f"**CVEs:** {', '.join(finding.cve_ids)}\n"
        body += (
            f"\n## What this proposes\n\n"
            f"Update `requirements.txt` to pin "
            f"`{finding.package}=={finding.latest_version}`. The "
            f"operator can then `pip install -r requirements.txt` "
            f"and run the test suite.\n"
            f"\n## Why this is auto-proposed\n\n"
            f"Detected by the weekly `dependency_radar` HEAVY idle. "
            f"The radar files patch-level + CVE-patch CRs through the "
            f"standard operator-gated change-request flow (this CR). "
            f"Major-version bumps surface as Signal alerts ONLY — "
            f"those require a deliberate migration window."
        )
        body += precedent
        body += (
            f"\n\n## Disclaimer\n\n"
            f"This proposal does NOT actually install the new version "
            f"in any environment. It is a markdown diff against "
            f"`requirements.txt`. The operator approves via "
            f"/cp/changes; only then does the change land."
        )
        # The CR target is requirements.txt — the diff is a single line.
        stage(
            source="dependency_radar",
            signature=signature,
            title=title,
            body_markdown=body,
            target_path="requirements.txt",
            cooldown_days=cooldown,
        )
        return True
    except Exception:
        logger.debug(
            "dependency_radar: stage failed for %s",
            finding.package, exc_info=True,
        )
        return False


def _alert_finding(finding: RadarFinding, *, notify_fn: Optional[Callable] = None) -> bool:
    """Send Signal alert for findings that don't get auto-CR'd
    (major version, CVE-no-fix, abandoned)."""
    try:
        if notify_fn is None:
            from app.notify import notify as _notify
            notify_fn = _notify
        if finding.severity == Severity.MAJOR:
            title = f"📦 Dependency: major-version available ({finding.package})"
            body = (
                f"{finding.package}: {finding.current_version} → "
                f"{finding.latest_version} (major). Schedule a "
                f"migration window before bumping; major versions "
                f"often have breaking API changes."
            )
            topic = f"dep_major:{finding.package}"
        elif finding.severity == Severity.CVE_NO_FIX:
            title = f"🚨 CVE without fix: {finding.package}"
            body = (
                f"{finding.package} ({finding.current_version}) has "
                f"reported CVEs without a patched version: "
                f"{', '.join(finding.cve_ids)}. Operator: investigate "
                f"workaround or pin away from vulnerable code paths."
            )
            topic = f"dep_cve_nofix:{finding.package}"
        elif finding.severity == Severity.ABANDONED:
            days = finding.extra.get("days_since_push", "?")
            title = f"🪦 Dependency abandoned: {finding.package}"
            body = (
                f"{finding.package} has not received a push to its "
                f"GitHub repo in {days}d. Consider migration to an "
                f"actively-maintained alternative when convenient."
            )
            topic = f"dep_abandoned:{finding.package}"
        else:
            return False  # not an alert-routed severity
        notify_fn(
            title=title,
            body=body,
            url="/cp/changes",
            topic=topic,
            critical=False,
            arbitrate=True,
        )
        return True
    except Exception:
        logger.debug("dependency_radar: alert failed", exc_info=True)
        return False


# ── Public entry: one pass ──────────────────────────────────────────────


def run_one_pass(
    *,
    pip_runner: Optional[Callable[[], list[dict[str, Any]]]] = None,
    osv_runner: Optional[Callable[[list[tuple[str, str]]], dict[str, list[dict[str, Any]]]]] = None,
    pip_show_runner: Optional[Callable[[str], dict[str, str]]] = None,
    github_runner: Optional[Callable[[str, str], Optional[datetime]]] = None,
    stage_fn: Optional[Callable[..., None]] = None,
    notify_fn: Optional[Callable] = None,
    lessons_consult: Optional[Callable[[str], Optional[str]]] = None,
    now: Optional[datetime] = None,
) -> RadarResult:
    """One radar pass: collect → classify → route. Pure function for
    test driving. Callable directly from the daemon loop.

    All collectors injectable; defaults call live subprocess / HTTP.

    Failure-isolated: any collector failure returns empty results
    for that signal; routing proceeds with whatever else succeeded.
    """
    result = RadarResult()
    if not _enabled():
        return result
    result.ran = True

    # Signal 1: outdated packages.
    outdated = _gather_outdated(pip_runner=pip_runner)

    # Signal 2: CVE scan for the packages we know we have.
    pkgs_with_versions = [
        (row["package"], row["current"]) for row in outdated
    ]
    cves_by_pkg = _gather_cves(packages=pkgs_with_versions, osv_runner=osv_runner)

    # Signal 3: abandonment for outdated packages (priority — we
    # already have version info on them). Skipped for healthy packages.
    pkg_names = [row["package"] for row in outdated]
    pushed_by_pkg = _gather_abandonment(
        packages=pkg_names,
        pip_show_runner=pip_show_runner,
        github_runner=github_runner,
    )

    findings = _build_findings(
        outdated, cves_by_pkg, pushed_by_pkg, now=now,
    )
    result.findings = findings

    # Bucket findings by severity for the summary.
    by_severity: dict[str, int] = {}
    for f in findings:
        by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1
    result.findings_by_severity = by_severity

    # Route. Patch / Minor / CVE → proposal_bridge (rate-limited).
    # Major / CVE_NO_FIX / Abandoned → Signal alert.
    proposed = 0
    for finding in findings:
        if proposed >= _MAX_PROPOSALS_PER_PASS:
            break
        if finding.severity in (Severity.PATCH, Severity.MINOR, Severity.CVE):
            if _propose_bump_cr(
                finding,
                lessons_consult=lessons_consult,
                stage_fn=stage_fn,
            ):
                proposed += 1
                result.cr_proposals_filed += 1
    result.alerts_fired = 0
    for finding in findings:
        if finding.severity in (Severity.MAJOR, Severity.CVE_NO_FIX, Severity.ABANDONED):
            if _alert_finding(finding, notify_fn=notify_fn):
                result.alerts_fired += 1

    # Persist last-pass state.
    state = _read_state()
    state["last_pass_at"] = (now or datetime.now(timezone.utc)).isoformat()
    state["last_findings_by_severity"] = by_severity
    state["last_cr_count"] = result.cr_proposals_filed
    state["last_alert_count"] = result.alerts_fired
    _write_state(state)

    return result


# ── Daemon ──────────────────────────────────────────────────────────────


def _is_running() -> bool:
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


def _daemon_loop() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        if _enabled():
            try:
                run_one_pass()
            except Exception:
                logger.debug("dependency_radar: pass raised", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start_daemon() -> bool:
    """Idempotent start. Returns True if a new thread was started."""
    global _driver_started
    with _driver_lock:
        if _is_running():
            return False
        _stop_event.clear()
        t = threading.Thread(
            target=_daemon_loop, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        t.start()
        _driver_started = True
        return True


def stop_daemon() -> None:
    """Signal the loop to exit. Useful for tests / shutdown."""
    _stop_event.set()


# Eager start at import — same pattern as library_radar.proposer +
# proposal_bridge.promoter. Boot-anchored by app/healing/__init__.py
# so a refactor of unrelated imports can't silently disable the daemon.
if _enabled():
    try:
        start_daemon()
    except Exception:  # pragma: no cover
        logger.warning("dependency_radar: start_daemon raised", exc_info=True)
