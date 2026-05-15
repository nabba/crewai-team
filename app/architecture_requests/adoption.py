"""Adoption-metric tracker for completed architecture requests.

PROGRAM §45.1 — Q7.1. Closes a gap the original §32.5 ship deferred:
the architecture-request primitive let agents propose new subsystems
and the scaffolder wrote stubs to disk, but nothing measured whether
the new subsystem was actually USED after `COMPLETED`. A subsystem
could be approved, scaffolded, implemented (via CR fanout), and
then sit dead forever.

This module computes a normalized adoption score in [0, 1] over a
configurable window (default 30 days) post-`COMPLETED`. The four
signals are deliberately blunt:

  1. ``n_imports`` — how many `from <package> import` matches exist
      anywhere under ``app/`` (excluding the subsystem's own files)
  2. ``n_idle_runs`` — if the request declared an idle-job
      ``IntegrationPoint``, count audit-log entries from that job
  3. ``n_outputs`` — count files written under the subsystem's
      ``workspace/<name>/`` directory
  4. ``n_operator_interactions`` — REST hits or React interactions
      via the audit-log

Goodhart guard
==============

The score IS NOT used to GATE anything — only to TRIGGER operator
review. The auto-rollback monitor at ``app/healing/monitors/
architecture_adoption.py`` proposes a rollback CR when score is
low; the operator always decides. Score is observational signal,
never a metric the system optimizes against.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default window. The Q7 plan called for 30 days post-COMPLETED; the
# auto-rollback monitor adds a further 30-60d window before acting.
_DEFAULT_WINDOW_DAYS = 30

# Signal weights. Sum to 1.0. Tuned so a fully-adopted subsystem
# (imports + idle-running + outputs + operator clicks) scores ~1.0
# and a subsystem with NO signal of any kind scores 0.0.
_WEIGHTS = {
    "imports":               0.40,  # primary signal: code actually depends on it
    "idle_runs":             0.30,  # secondary: the scheduled work fires
    "outputs":               0.20,  # tertiary: it produces artefacts
    "operator_interactions": 0.10,  # quaternary: operator touches the surface
}

# Saturation points — values at or above these contribute the full
# weighted signal. Below, contribute proportionally. Above the
# saturation threshold further usage doesn't increase the score
# (prevents one heavily-used subsystem from inflating the average).
_SATURATION = {
    "imports":               5,    # 5 distinct callers = fully integrated
    "idle_runs":             10,   # 10 audit entries in 30 days = idle-running normally
    "outputs":               5,    # 5 output files in 30 days = producing artefacts
    "operator_interactions": 3,    # 3 operator touches in 30 days = surface in use
}

# The threshold under which the auto-rollback monitor flags a
# completed-30d subsystem for operator review. Operator override
# possible via the per-request `adoption_override` field on the
# ArchitectureRequest.
LOW_ADOPTION_THRESHOLD = 0.2


@dataclass(frozen=True)
class AdoptionReport:
    """One adoption-measurement pass over a completed request."""

    request_id: str
    package_path: str
    window_days: int
    completed_at: str
    measured_at: str

    n_imports: int
    n_idle_runs: int
    n_outputs: int
    n_operator_interactions: int

    score: float                      # normalized [0, 1]
    is_low_adoption: bool             # score < LOW_ADOPTION_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "package_path": self.package_path,
            "window_days": self.window_days,
            "completed_at": self.completed_at,
            "measured_at": self.measured_at,
            "n_imports": self.n_imports,
            "n_idle_runs": self.n_idle_runs,
            "n_outputs": self.n_outputs,
            "n_operator_interactions": self.n_operator_interactions,
            "score": round(self.score, 4),
            "is_low_adoption": self.is_low_adoption,
        }


def _saturate(value: int, key: str) -> float:
    """Saturating normalization. value >= _SATURATION[key] → 1.0."""
    sat = _SATURATION[key]
    if sat <= 0:
        return 0.0
    return min(1.0, value / sat)


def compute_score(
    n_imports: int,
    n_idle_runs: int,
    n_outputs: int,
    n_operator_interactions: int,
) -> float:
    """Pure function — saturating weighted sum. Public so tests can
    pin the formula without going through the file-system probe path."""
    return round(
        _WEIGHTS["imports"]               * _saturate(n_imports, "imports")
      + _WEIGHTS["idle_runs"]             * _saturate(n_idle_runs, "idle_runs")
      + _WEIGHTS["outputs"]               * _saturate(n_outputs, "outputs")
      + _WEIGHTS["operator_interactions"] * _saturate(n_operator_interactions, "operator_interactions"),
        4,
    )


# ── Signal probes ────────────────────────────────────────────────────────


def _count_imports(package_path: str, *, repo_root: Path) -> int:
    """Count `from <package> import` and `import <package>` matches
    under ``app/`` (excluding the package's own files).

    Uses ripgrep when available; falls back to a Python walk. Bounded
    by the package's own file count so a self-import doesn't inflate.
    """
    if not package_path or not package_path.startswith("app/"):
        return 0
    # Convert path → import name. ``app/foo_bar/`` → ``app.foo_bar``.
    import_name = package_path.strip("/").replace("/", ".")
    # Strip trailing module suffix if any
    if import_name.endswith(".py"):
        import_name = import_name[:-3]
    own_prefix = package_path.rstrip("/") + "/"
    pattern = rf"^(from|import)\s+{re.escape(import_name)}([\s.,]|$)"
    try:
        result = subprocess.run(
            ["rg", "--no-heading", "--type", "py", "-n", pattern, "app/"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode > 1:  # rg returns 1 for "no matches"
            raise RuntimeError("rg failed")
        n = 0
        for line in (result.stdout or "").splitlines():
            # Skip self-imports.
            if line.split(":", 1)[0].startswith(own_prefix):
                continue
            n += 1
        return n
    except Exception:
        pass
    # Fallback: pure-Python walk.
    n = 0
    try:
        compiled = re.compile(pattern)
        for path in (repo_root / "app").rglob("*.py"):
            rel = str(path.relative_to(repo_root))
            if rel.startswith(own_prefix):
                continue
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if compiled.match(line):
                        n += 1
            except (OSError, UnicodeDecodeError):
                continue
    except Exception:
        return 0
    return n


def _count_idle_runs(
    integration_points: list,
    *,
    since_iso: str,
    workspace_root: Path,
) -> int:
    """Count idle-job audit entries under the subsystem's job names.

    The idle scheduler records each fired job to
    ``workspace/observability/idle_jobs.jsonl``. We count entries
    whose ``job_name`` matches an integration-point job declaration."""
    job_names: set[str] = set()
    for ip in integration_points or []:
        # ip may be IntegrationPoint dataclass OR dict (when loaded from JSON).
        if hasattr(ip, "kind"):
            kind = ip.kind
            target = getattr(ip, "target", "")
        else:
            kind = ip.get("kind", "")
            target = ip.get("target", "")
        if kind == "idle_job" and target:
            job_names.add(str(target))
    if not job_names:
        return 0
    path = workspace_root / "observability" / "idle_jobs.jsonl"
    if not path.exists():
        return 0
    n = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = row.get("ts") or ""
                if ts < since_iso:
                    continue
                if (row.get("job_name") or "") in job_names:
                    n += 1
    except OSError:
        return 0
    return n


def _count_outputs(
    package_path: str,
    *,
    since_iso: str,
    workspace_root: Path,
) -> int:
    """Count files under workspace/<subsystem_name>/ written after
    completed_at. Subsystem name is the last path component."""
    if not package_path:
        return 0
    # `app/foo_bar/` → `foo_bar`
    name = package_path.rstrip("/").split("/")[-1]
    out_dir = workspace_root / name
    if not out_dir.exists() or not out_dir.is_dir():
        return 0
    try:
        cutoff_dt = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0
    cutoff_ts = cutoff_dt.timestamp()
    n = 0
    try:
        for entry in out_dir.rglob("*"):
            if not entry.is_file():
                continue
            try:
                if entry.stat().st_mtime >= cutoff_ts:
                    n += 1
            except OSError:
                continue
    except OSError:
        return 0
    return n


def _count_operator_interactions(
    request_id: str,
    package_path: str,
    *,
    since_iso: str,
    workspace_root: Path,
) -> int:
    """Best-effort: count audit-log entries that reference this
    request_id or the subsystem name. Audit-log path matches the
    existing convention in app/audit.py."""
    audit_path = workspace_root / "audit_log.jsonl"
    if not audit_path.exists():
        return 0
    name = (package_path or "").rstrip("/").split("/")[-1] if package_path else ""
    needles = {request_id}
    if name:
        needles.add(name)
    n = 0
    try:
        with audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = row.get("ts") or ""
                if ts < since_iso:
                    continue
                # Concatenated serialised needle search.
                blob = json.dumps(row, default=str)
                if any(needle in blob for needle in needles if needle):
                    n += 1
    except OSError:
        return 0
    return n


# ── Public API ────────────────────────────────────────────────────────────


def measure(
    request_id: str,
    *,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    repo_root: Path | None = None,
    workspace_root: Path | None = None,
) -> AdoptionReport | None:
    """Compute an AdoptionReport for one architecture request.

    Returns None when the request:
      - doesn't exist
      - isn't in COMPLETED state
      - completed less than ``window_days`` ago

    Failure-isolated: any probe error returns 0 for that signal (so
    the score is conservative — under-reporting adoption rather than
    over-reporting it)."""
    try:
        from app.architecture_requests.store import get as _get_request
        req = _get_request(request_id)
    except Exception:
        return None
    if req is None:
        return None
    completed_at = getattr(req, "completed_at", None)
    if not completed_at:
        return None
    # Use the request's stringified status to avoid coupling to enum.
    status = getattr(req, "status", None)
    status_val = getattr(status, "value", str(status)).lower()
    if status_val != "completed":
        return None
    try:
        completed_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    age = datetime.now(timezone.utc) - completed_dt
    if age < timedelta(days=window_days):
        return None

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    if workspace_root is None:
        try:
            from app.paths import WORKSPACE_ROOT
            workspace_root = Path(WORKSPACE_ROOT)
        except Exception:
            workspace_root = Path("/app/workspace")

    package_path = getattr(req, "package_path", "")
    since_iso = completed_dt.isoformat()
    integration_points = list(getattr(req, "integration_points", None) or [])

    n_imports               = _count_imports(package_path, repo_root=repo_root)
    n_idle_runs             = _count_idle_runs(
        integration_points, since_iso=since_iso, workspace_root=workspace_root,
    )
    n_outputs               = _count_outputs(
        package_path, since_iso=since_iso, workspace_root=workspace_root,
    )
    n_operator_interactions = _count_operator_interactions(
        request_id, package_path,
        since_iso=since_iso, workspace_root=workspace_root,
    )

    score = compute_score(
        n_imports, n_idle_runs, n_outputs, n_operator_interactions,
    )
    return AdoptionReport(
        request_id=request_id,
        package_path=package_path,
        window_days=window_days,
        completed_at=completed_at,
        measured_at=datetime.now(timezone.utc).isoformat(),
        n_imports=n_imports,
        n_idle_runs=n_idle_runs,
        n_outputs=n_outputs,
        n_operator_interactions=n_operator_interactions,
        score=score,
        is_low_adoption=score < LOW_ADOPTION_THRESHOLD,
    )


def list_completed_eligible_for_measurement(
    *,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    max_age_days: int = 60,
) -> list[str]:
    """Return request IDs in COMPLETED state with completed_at age
    in the window ``[window_days, max_age_days]``. The monitor checks
    these for low-adoption signal."""
    try:
        from app.architecture_requests.store import list_all
        all_reqs = list_all()
    except Exception:
        return []
    out: list[str] = []
    now = datetime.now(timezone.utc)
    lower = now - timedelta(days=max_age_days)
    upper = now - timedelta(days=window_days)
    for req in all_reqs:
        status = getattr(req, "status", None)
        if getattr(status, "value", str(status)).lower() != "completed":
            continue
        completed_at = getattr(req, "completed_at", None)
        if not completed_at:
            continue
        try:
            completed_dt = datetime.fromisoformat(
                completed_at.replace("Z", "+00:00"),
            )
        except (ValueError, TypeError):
            continue
        if lower <= completed_dt <= upper:
            out.append(req.id)
    return out
