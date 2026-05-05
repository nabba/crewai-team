"""Apply approved change requests.

Hot-apply path (after Signal 👍 or React operator approve):

  1. Write the file via the host bridge (``write_host_file``).
  2. Best-effort module reload — for ``app/agents/*.py`` and similar
     Python files, importlib.reload picks up the change without a
     gateway restart. For files imported once at boot, we log
     "restart needed" but the hot-apply still succeeds.
  3. Git operations via the bridge: branch, commit, push, open PR
     against main. The PR is the durable second-gate artifact —
     operator merges it after review (or the auto-merge CI lands
     it if/when configured).

Rollback path (after operator clicks Rollback in React):

  1. ``git revert <commit_sha>`` on a fresh branch
  2. Push the revert branch
  3. Hot-revert: write the original file content back via the bridge
  4. Module reload again
  5. Open a "Revert" PR. Operator merges to make the rollback durable.

All bridge calls are wrapped in try/except — failures are captured
in the apply_error field and surface to the operator via the React
UI / audit log.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from app.change_requests import lifecycle, store
from app.change_requests.models import ChangeRequest, Status

logger = logging.getLogger(__name__)


_HOST_REPO_PATH_ENV = "HOST_REPO_PATH"
_HOST_REPO_PATH_DEFAULT = "/Users/andrus/BotArmy/crewai-team"


def _host_repo_path() -> str:
    return os.environ.get(_HOST_REPO_PATH_ENV) or _HOST_REPO_PATH_DEFAULT


def _module_path_for(file_path: str) -> str | None:
    """Convert a Python file path to its dotted module path.
    Returns None for non-Python files."""
    if not file_path.endswith(".py"):
        return None
    if file_path.endswith("__init__.py"):
        # __init__.py → the parent package
        return file_path[:-len("/__init__.py")].replace("/", ".")
    return file_path[:-3].replace("/", ".")


def _get_bridge() -> Any | None:
    """Return a BridgeClient for the change-request agent_id, or
    None if unreachable."""
    try:
        from app.bridge_client import get_bridge
    except Exception as exc:  # noqa: BLE001
        logger.warning("apply: cannot import bridge_client (%s)", exc)
        return None
    bridge = get_bridge("change_requests")
    if bridge is None or not bridge.is_available():
        logger.warning("apply: bridge unavailable for change_requests agent")
        return None
    return bridge


# ── Apply ───────────────────────────────────────────────────────────


@dataclass
class ApplyResult:
    ok: bool
    git_branch: str | None = None
    git_commit_sha: str | None = None
    pr_url: str | None = None
    error: str | None = None
    module_reload_ok: bool = False
    module_reload_note: str | None = None


def apply_change(request_id: str) -> ApplyResult:
    """Hot-apply the approved change + open auto-PR.

    Idempotent on retry: if the file write succeeded but git ops
    failed, calling apply_change again retries from scratch (re-
    writes file, re-tries git). Operators can retry via the React
    UI's "Retry apply" button if APPLY_FAILED.
    """
    cr = store.get(request_id)
    if cr is None:
        return ApplyResult(ok=False, error=f"request {request_id!r} not found")
    if cr.status != Status.APPROVED:
        return ApplyResult(
            ok=False,
            error=f"can only apply APPROVED requests; current={cr.status.value}",
        )

    bridge = _get_bridge()
    if bridge is None:
        msg = "host bridge unreachable; cannot write file or run git"
        lifecycle.mark_apply_failed(request_id, error=msg)
        return ApplyResult(ok=False, error=msg)

    repo = _host_repo_path()
    abs_path = os.path.join(repo, cr.path)
    branch = f"auto/change_{cr.id}"

    # 1. Write the file
    try:
        result = bridge.write_file(abs_path, cr.new_content, create_dirs=True)
        if not result or result.get("ok") is False:
            err = (result or {}).get("error", "unknown write_file error")
            lifecycle.mark_apply_failed(request_id, error=f"write_file: {err}")
            return ApplyResult(ok=False, error=f"write_file: {err}")
    except Exception as exc:  # noqa: BLE001
        lifecycle.mark_apply_failed(request_id, error=f"write_file raised: {exc}")
        return ApplyResult(ok=False, error=f"write_file raised: {exc}")

    # 2. Best-effort module reload
    reload_ok, reload_note = _try_module_reload(cr.path)

    # 3. Git operations
    git_result = _run_git_auto_pr(
        bridge=bridge,
        repo=repo,
        branch=branch,
        path=cr.path,
        commit_message=_build_commit_message(cr),
        pr_title=_build_pr_title(cr),
        pr_body=_build_pr_body(cr),
    )

    if not git_result.ok:
        lifecycle.mark_apply_failed(request_id, error=git_result.error or "git ops failed")
        return ApplyResult(
            ok=False,
            error=git_result.error,
            module_reload_ok=reload_ok,
            module_reload_note=reload_note,
        )

    # 4. Mark applied
    lifecycle.mark_applied(
        request_id,
        git_branch=branch,
        git_commit_sha=git_result.commit_sha or "",
        pr_url=git_result.pr_url,
    )
    return ApplyResult(
        ok=True,
        git_branch=branch,
        git_commit_sha=git_result.commit_sha,
        pr_url=git_result.pr_url,
        module_reload_ok=reload_ok,
        module_reload_note=reload_note,
    )


def _try_module_reload(file_path: str) -> tuple[bool, str | None]:
    """Best-effort reload of the Python module for ``file_path``.
    Returns (ok, note). ok=False is non-fatal — the on-disk file
    is correct; a gateway restart will pick it up."""
    module_name = _module_path_for(file_path)
    if module_name is None:
        return True, "not a Python file; no reload needed"
    if module_name not in sys.modules:
        return True, f"module {module_name!r} not loaded; will load fresh on next import"
    try:
        importlib.reload(sys.modules[module_name])
        logger.info("apply: reloaded module %s", module_name)
        return True, f"reloaded {module_name}"
    except Exception as exc:  # noqa: BLE001
        return False, f"reload of {module_name} raised: {exc} — gateway restart recommended"


# ── Git auto-PR ─────────────────────────────────────────────────────


@dataclass
class GitResult:
    ok: bool
    commit_sha: str | None = None
    pr_url: str | None = None
    error: str | None = None


def _run_git_auto_pr(
    *,
    bridge,
    repo: str,
    branch: str,
    path: str,
    commit_message: str,
    pr_title: str,
    pr_body: str,
) -> GitResult:
    """Branch + add + commit + push + open PR. All via bridge.execute.
    Returns GitResult with the commit SHA + PR URL on success."""

    def _run(args: list[str], timeout: int = 30) -> dict[str, Any]:
        return bridge.execute(args, working_dir=repo, timeout=timeout) or {}

    # Snapshot main HEAD so we know what to revert to if needed
    try:
        head = _run(["git", "rev-parse", "main"]).get("stdout", "").strip()
    except Exception:
        head = ""

    # 1. Create branch from main (force in case branch already exists)
    try:
        # Reset any in-progress branch state; checkout main first
        _run(["git", "fetch", "origin", "main"], timeout=30)
        _run(["git", "checkout", "main"], timeout=10)
        _run(["git", "reset", "--hard", "origin/main"], timeout=10)
        _run(["git", "checkout", "-B", branch], timeout=10)
    except Exception as exc:  # noqa: BLE001
        return GitResult(ok=False, error=f"branch setup failed: {exc}")

    # 2. The file write happened earlier via bridge.write_file. Stage it.
    try:
        add_res = _run(["git", "add", path])
        if add_res.get("returncode", 0) != 0:
            return GitResult(
                ok=False,
                error=f"git add failed: {add_res.get('stderr', '')[:400]}",
            )
    except Exception as exc:  # noqa: BLE001
        return GitResult(ok=False, error=f"git add raised: {exc}")

    # 3. Commit
    try:
        commit_res = _run(["git", "commit", "-m", commit_message])
        if commit_res.get("returncode", 0) != 0:
            stderr = commit_res.get("stderr", "")
            if "nothing to commit" in stderr.lower():
                return GitResult(
                    ok=False,
                    error="nothing to commit — file content unchanged?",
                )
            return GitResult(
                ok=False,
                error=f"git commit failed: {stderr[:400]}",
            )
        sha = _run(["git", "rev-parse", "HEAD"]).get("stdout", "").strip()
    except Exception as exc:  # noqa: BLE001
        return GitResult(ok=False, error=f"git commit raised: {exc}")

    # 4. Push
    try:
        push_res = _run(["git", "push", "-u", "origin", branch], timeout=60)
        if push_res.get("returncode", 0) != 0:
            return GitResult(
                ok=False,
                commit_sha=sha,
                error=f"git push failed: {push_res.get('stderr', '')[:400]}",
            )
    except Exception as exc:  # noqa: BLE001
        return GitResult(ok=False, commit_sha=sha, error=f"git push raised: {exc}")

    # 5. Open PR via gh
    pr_url = None
    try:
        pr_res = _run([
            "gh", "pr", "create",
            "--base", "main",
            "--head", branch,
            "--title", pr_title,
            "--body", pr_body,
        ], timeout=30)
        out = pr_res.get("stdout", "").strip()
        # gh prints the PR URL as the last line of output
        for line in out.splitlines():
            if line.startswith("https://github.com/"):
                pr_url = line.strip()
                break
    except Exception as exc:  # noqa: BLE001
        # PR creation is the LAST step — if it fails we still succeed
        # the apply (the commit is on a branch and pushed; operator
        # can open the PR manually).
        logger.warning("apply: gh pr create failed (non-fatal): %s", exc)

    return GitResult(ok=True, commit_sha=sha, pr_url=pr_url)


def _build_commit_message(cr: ChangeRequest) -> str:
    return (
        f"auto-fix: {cr.path} (change_request {cr.id})\n\n"
        f"{cr.reason}\n\n"
        f"Requested by: {cr.requestor}\n"
        f"Decided via: {cr.decided_by.value if cr.decided_by else 'unknown'}\n"
        f"Change request id: {cr.id}\n"
        f"Audit: workspace/change_requests/{cr.id}.json\n"
    )


def _build_pr_title(cr: ChangeRequest) -> str:
    return f"auto-fix: {cr.path} ({cr.requestor})"


def _build_pr_body(cr: ChangeRequest) -> str:
    return (
        f"## Auto-generated change request\n\n"
        f"**Path:** `{cr.path}`\n"
        f"**Requested by:** `{cr.requestor}`\n"
        f"**Decided by:** `{cr.decided_by.value if cr.decided_by else 'unknown'}`\n"
        f"**Change request id:** `{cr.id}`\n\n"
        f"### Reason\n\n{cr.reason}\n\n"
        f"### Diff\n\n```diff\n{cr.diff[:6000]}\n```\n\n"
        f"---\n\n"
        f"This PR was opened automatically after the change "
        f"request was approved through the Signal/React human "
        f"gate. The hot-apply has already happened in the running "
        f"gateway. Merging this PR makes the change durable in main.\n\n"
        f"To roll back: use the React control plane "
        f"(`/cp/changes/{cr.id}`) → Rollback button. The system "
        f"will revert the commit, hot-revert the file, and open a "
        f"separate revert PR.\n\n"
        f"Audit trail: `workspace/change_requests/{cr.id}.json`\n"
        f"Audit log: `workspace/change_requests/audit.jsonl`\n"
    )


# ── Rollback ────────────────────────────────────────────────────────


def rollback_change(request_id: str, *, operator: str) -> ApplyResult:
    """Roll back an APPLIED change.

    Sequence:
      1. Read the original file content from the request's
         ``old_content`` field.
      2. Write it back via the bridge (hot-revert).
      3. Best-effort module reload.
      4. ``git revert <commit_sha>`` on a fresh branch + push +
         open revert PR.
    """
    cr = store.get(request_id)
    if cr is None:
        return ApplyResult(ok=False, error=f"request {request_id!r} not found")
    if not cr.is_rollbackable:
        return ApplyResult(
            ok=False,
            error=f"cannot rollback {request_id!r} in status {cr.status.value}",
        )

    bridge = _get_bridge()
    if bridge is None:
        return ApplyResult(ok=False, error="host bridge unreachable")

    repo = _host_repo_path()
    abs_path = os.path.join(repo, cr.path)

    # 1. Hot-revert: write original content back
    try:
        result = bridge.write_file(abs_path, cr.old_content, create_dirs=True)
        if not result or result.get("ok") is False:
            err = (result or {}).get("error", "unknown write_file error")
            return ApplyResult(ok=False, error=f"hot-revert write_file: {err}")
    except Exception as exc:  # noqa: BLE001
        return ApplyResult(ok=False, error=f"hot-revert raised: {exc}")

    # 2. Module reload
    reload_ok, reload_note = _try_module_reload(cr.path)

    # 3. Git revert + auto-PR
    revert_branch = f"auto/revert_{cr.id}"
    git_result = _run_git_revert_pr(
        bridge=bridge,
        repo=repo,
        original_commit_sha=cr.git_commit_sha or "",
        revert_branch=revert_branch,
        pr_title=f"auto-revert: {cr.path} (change_request {cr.id})",
        pr_body=_build_revert_pr_body(cr, operator),
    )

    if not git_result.ok:
        return ApplyResult(
            ok=False,
            error=f"hot-revert succeeded but git revert failed: {git_result.error}",
            module_reload_ok=reload_ok,
            module_reload_note=reload_note,
        )

    lifecycle.mark_rolled_back(
        request_id,
        operator=operator,
        rollback_commit_sha=git_result.commit_sha or "",
        rollback_pr_url=git_result.pr_url,
    )
    return ApplyResult(
        ok=True,
        git_branch=revert_branch,
        git_commit_sha=git_result.commit_sha,
        pr_url=git_result.pr_url,
        module_reload_ok=reload_ok,
        module_reload_note=reload_note,
    )


def _run_git_revert_pr(
    *,
    bridge,
    repo: str,
    original_commit_sha: str,
    revert_branch: str,
    pr_title: str,
    pr_body: str,
) -> GitResult:
    """Branch from main + git revert + push + open revert PR."""
    def _run(args, timeout=30):
        return bridge.execute(args, working_dir=repo, timeout=timeout) or {}

    if not original_commit_sha:
        return GitResult(
            ok=False,
            error="original commit sha unknown; manual git revert required",
        )

    try:
        _run(["git", "fetch", "origin", "main"], timeout=30)
        _run(["git", "checkout", "main"], timeout=10)
        _run(["git", "reset", "--hard", "origin/main"], timeout=10)
        _run(["git", "checkout", "-B", revert_branch], timeout=10)

        revert_res = _run(["git", "revert", "--no-edit", original_commit_sha])
        if revert_res.get("returncode", 0) != 0:
            return GitResult(
                ok=False,
                error=f"git revert failed: {revert_res.get('stderr', '')[:400]}",
            )
        sha = _run(["git", "rev-parse", "HEAD"]).get("stdout", "").strip()
        push_res = _run(["git", "push", "-u", "origin", revert_branch], timeout=60)
        if push_res.get("returncode", 0) != 0:
            return GitResult(
                ok=False,
                commit_sha=sha,
                error=f"git push failed: {push_res.get('stderr', '')[:400]}",
            )
    except Exception as exc:  # noqa: BLE001
        return GitResult(ok=False, error=f"git revert flow raised: {exc}")

    pr_url = None
    try:
        pr_res = _run([
            "gh", "pr", "create",
            "--base", "main",
            "--head", revert_branch,
            "--title", pr_title,
            "--body", pr_body,
        ], timeout=30)
        for line in (pr_res.get("stdout", "") or "").splitlines():
            if line.startswith("https://github.com/"):
                pr_url = line.strip()
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("rollback: gh pr create failed (non-fatal): %s", exc)

    return GitResult(ok=True, commit_sha=sha, pr_url=pr_url)


def _build_revert_pr_body(cr: ChangeRequest, operator: str) -> str:
    return (
        f"## Auto-generated revert\n\n"
        f"**Reverts:** {cr.pr_url or 'commit ' + (cr.git_commit_sha or '?')}\n"
        f"**Original change request:** `{cr.id}`\n"
        f"**Original path:** `{cr.path}`\n"
        f"**Rolled back by:** `{operator}`\n\n"
        f"### Original reason\n\n{cr.reason}\n\n"
        f"---\n\n"
        f"This is an auto-generated revert PR triggered from the "
        f"React control plane. The hot-revert has already happened "
        f"in the running gateway. Merging this PR makes the rollback "
        f"durable in main.\n"
    )
