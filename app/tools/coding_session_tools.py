"""coding_session_* — the 7 agent-callable tools for Phase 5.4.

Pattern of use::

    sid = coding_session_start(base="main", purpose="fix the import bug")
    coding_session_read(session_id=sid, path="app/agents/pim_agent.py")
    coding_session_write(
        session_id=sid,
        path="app/agents/pim_agent.py",
        content=NEW_CONTENT,
    )
    coding_session_run(
        session_id=sid,
        argv=["pytest", "tests/test_pim_agent.py", "-v"],
        timeout_s=60,
    )
    # iterate read/write/run until tests pass …
    coding_session_diff(session_id=sid)            # last self-review
    coding_session_submit(
        session_id=sid,
        reason="all tests pass; PIM crew constructs cleanly",
    )

Each tool returns a string (CrewAI convention) — formatted for the
agent's prompt to read. Errors return a string starting with one of
``ERROR:``, ``REFUSED:``, or ``QUOTA_EXCEEDED:`` so the agent can
pattern-match on the prefix.

Shared state: all seven tools talk to the same ``Manager`` singleton
via ``app.coding_session.runtime.get_manager()``. Tests inject a
test Manager via ``runtime.set_manager_for_tests(...)``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Type

logger = logging.getLogger(__name__)


# ── Shared formatting helpers ───────────────────────────────────────


_ERR_PREFIX = "ERROR:"
_REFUSED_PREFIX = "REFUSED:"
_QUOTA_PREFIX = "QUOTA_EXCEEDED:"


def _format_session(cs: Any, *, header: str) -> str:
    """Compact human-readable session summary for tool returns."""
    return (
        f"{header}\n"
        f"  session_id: {cs.id}\n"
        f"  status:     {cs.status.value}\n"
        f"  agent:      {cs.agent_id}\n"
        f"  base:       {cs.base} ({cs.base_sha[:8]})\n"
        f"  worktree:   {cs.worktree_path}\n"
        f"  expires_at: {cs.expires_at}\n"
        f"  files:      {len(cs.files_touched)} touched, "
        f"{cs.bytes_written} bytes; {cs.run_count} runs"
    )


# ── Tool factory ────────────────────────────────────────────────────


def _build_tool_classes() -> dict[str, type]:
    """Lazy-build the seven tool classes. CrewAI BaseTool + pydantic
    args_schema isn't safe to evaluate at import time (CrewAI may not
    have loaded yet); deferring keeps this module importable in test
    contexts that don't have the full dep tree."""
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    # ── 1. start ──────────────────────────────────────────────────

    class _StartInput(BaseModel):
        base: str = Field(
            default="main",
            description=(
                "Branch / tag / sha to base the worktree on. Almost "
                "always 'main'; can be a feature branch for stacked "
                "work."
            ),
        )
        purpose: str = Field(
            description=(
                "One-paragraph statement of what this session intends "
                "to do. Surfaces in the audit log AND becomes the "
                "prefix of every change request's reason field at "
                "submit time. Be specific: 'fix NameError in PIM "
                "agent — optional_tool_group import missing' beats "
                "'fix bug'."
            ),
        )

    class CodingSessionStartTool(BaseTool):
        name: str = "coding_session_start"
        description: str = (
            "Start a fresh coding session. Creates an ephemeral git "
            "worktree where you can read, write, and run pytest / "
            "lint / typecheck commands freely.\n\n"
            "Quotas: 3 active per agent, 30 min TTL, 100 MB worktree, "
            "120 s default per coding_session_run.\n\n"
            "Returns the session_id, which all other coding_session_* "
            "tools take as their first arg."
        )
        args_schema: Type[BaseModel] = _StartInput

        def _run(self, base: str = "main", purpose: str = "") -> str:
            from app.coding_session import (
                IllegalTransition, QuotaExceeded, runtime,
            )

            mgr = runtime.get_manager()
            try:
                cs = mgr.start(
                    agent_id=_resolve_agent_id(),
                    base=base,
                    purpose=purpose,
                    worktree_root=runtime.worktree_root(),
                )
            except QuotaExceeded as exc:
                return f"{_QUOTA_PREFIX} {exc}"
            except ValueError as exc:
                return f"{_REFUSED_PREFIX} {exc}"
            except RuntimeError as exc:
                return (
                    f"{_ERR_PREFIX} cannot create worktree: {exc}. "
                    "If the bridge is unreachable, the session was "
                    "not persisted — retry."
                )
            except IllegalTransition as exc:
                return f"{_ERR_PREFIX} {exc}"

            return _format_session(
                cs, header=f"Coding session {cs.id} started.",
            )

    # ── 2. read ───────────────────────────────────────────────────

    class _ReadInput(BaseModel):
        session_id: str = Field(description="Session id from coding_session_start.")
        path: str = Field(
            description=(
                "Repo-relative path to read inside the worktree. "
                "e.g. 'app/agents/pim_agent.py'."
            ),
        )

    class CodingSessionReadTool(BaseTool):
        name: str = "coding_session_read"
        description: str = (
            "Read a file inside the coding-session worktree. The agent "
            "can read both files it has modified AND unchanged files — "
            "useful for 'what does this helper actually look like' "
            "mid-fix.\n\n"
            "For reads outside the worktree (e.g. test fixtures "
            "elsewhere in the host repo), use read_host_file instead."
        )
        args_schema: Type[BaseModel] = _ReadInput

        def _run(self, session_id: str, path: str) -> str:
            from app.coding_session import runtime

            mgr = runtime.get_manager()
            cs = mgr.get(session_id)
            if cs is None:
                return f"{_ERR_PREFIX} session {session_id!r} not found"
            if not cs.is_active:
                return (
                    f"{_REFUSED_PREFIX} session is {cs.status.value} "
                    f"(not ACTIVE); reads denied"
                )

            try:
                content = mgr.backend.read_worktree_file(
                    worktree_path=cs.worktree_path, path=path,
                )
            except FileNotFoundError:
                return f"{_REFUSED_PREFIX} {path!r} not in worktree"
            except Exception as exc:  # noqa: BLE001
                return f"{_ERR_PREFIX} read failed: {exc}"

            mgr.touch(session_id)
            return content

    # ── 3. write ──────────────────────────────────────────────────

    class _WriteInput(BaseModel):
        session_id: str = Field(description="Session id from coding_session_start.")
        path: str = Field(
            description=(
                "Repo-relative path to write inside the worktree. "
                "Validated against the same allowed-roots / "
                "TIER_IMMUTABLE rules as the change-request system."
            ),
        )
        content: str = Field(
            description=(
                "Complete new file contents. Not a diff. Max 1 MB."
            ),
        )

    class CodingSessionWriteTool(BaseTool):
        name: str = "coding_session_write"
        description: str = (
            "Write a file inside the coding-session worktree. The "
            "write does NOT reach the live tree; only coding_session_"
            "submit moves changes through the human gate.\n\n"
            "Validated at write-time against the same allowed-roots / "
            "TIER_IMMUTABLE rules as the change-request system — so "
            "you learn early if your target is forbidden."
        )
        args_schema: Type[BaseModel] = _WriteInput

        def _run(self, session_id: str, path: str, content: str) -> str:
            from app.change_requests import validate as cr_validate
            from app.coding_session import (
                can_write_bytes, runtime,
            )
            from pathlib import Path

            mgr = runtime.get_manager()
            cs = mgr.get(session_id)
            if cs is None:
                return f"{_ERR_PREFIX} session {session_id!r} not found"
            if not cs.is_active:
                return (
                    f"{_REFUSED_PREFIX} session is {cs.status.value} "
                    f"(not ACTIVE)"
                )

            # Fast-fail validator check (the change-request validator
            # has the same rules; we re-check at submit too)
            v = cr_validate(path=path, new_content=content)
            if not v.ok:
                tag = "TIER_IMMUTABLE" if v.is_tier_immutable else "VALIDATOR"
                return (
                    f"{_REFUSED_PREFIX} [{tag}] {v.reason}. "
                    f"This file cannot be written via the agent path; "
                    f"the operator must edit directly."
                )

            # Disk quota
            content_size = len(content.encode("utf-8"))
            after_session = cs.bytes_written + content_size
            # System total: estimate from index counts (cheap upper bound)
            # — for a tighter check we could sum bytes_written across
            # active sessions, but since per-session is already the
            # primary cap, the system check is a soft outer bound.
            from app.coding_session import store as cs_store
            after_system = sum(
                s.bytes_written for s in cs_store.list_all()
                if s.id != cs.id
            ) + after_session

            quota = can_write_bytes(
                config=mgr.config,
                session_bytes_after_write=after_session,
                system_bytes_after_write=after_system,
            )
            if not quota.ok:
                return f"{_QUOTA_PREFIX} {quota.reason}"

            # Actually write inside the worktree
            full = Path(cs.worktree_path) / path
            try:
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content, encoding="utf-8")
            except OSError as exc:
                return f"{_ERR_PREFIX} filesystem write failed: {exc}"

            try:
                mgr.record_write(session_id, path, content_size)
            except Exception as exc:  # noqa: BLE001
                # Persistence failed but write landed; leave the file
                # and surface the error so the agent retries.
                return f"{_ERR_PREFIX} session record_write raised: {exc}"

            return (
                f"OK: wrote {content_size} bytes to {path!r} "
                f"(session {session_id})"
            )

    # ── 4. run ────────────────────────────────────────────────────

    class _RunInput(BaseModel):
        session_id: str = Field(description="Session id from coding_session_start.")
        argv: list[str] = Field(
            description=(
                "Command + args as a list, NOT a shell string. "
                "argv[0] must be in the runner's allowlist (pytest, "
                "python, ruff, mypy, eslint, npx, node, plus read-only "
                "git/gh, plus POSIX read tools). Shell metacharacters "
                "have no special meaning."
            ),
        )
        timeout_s: int = Field(
            default=120,
            description=(
                "Wallclock timeout in seconds. Capped at 600. "
                "Default 120 — enough for a focused pytest module."
            ),
        )

    class CodingSessionRunTool(BaseTool):
        name: str = "coding_session_run"
        description: str = (
            "Execute a command inside the coding-session worktree. "
            "The most important tool: this is the iteration loop.\n\n"
            "Allowlist: pytest, python, python3, ruff, mypy, eslint, "
            "npx, node, plus read-only git (status, diff, log, show, "
            "ls-files, rev-parse) and gh (pr view, issue view), plus "
            "POSIX reads (cat, ls, wc, head, tail, grep, rg, find).\n\n"
            "Sandbox: shell=False (argv is a list); no network; CPU "
            "rlimit; wallclock timeout; cwd locked to worktree; "
            "secrets stripped from env.\n\n"
            "Returns a JSON object with stdout, stderr, exit_code, "
            "elapsed_ms, and the truncated_*/timed_out flags."
        )
        args_schema: Type[BaseModel] = _RunInput

        def _run(
            self,
            session_id: str,
            argv: list[str],
            timeout_s: int = 120,
        ) -> str:
            from app.coding_session import (
                cap_run_timeout, run as runner_run, runtime,
            )

            mgr = runtime.get_manager()
            cs = mgr.get(session_id)
            if cs is None:
                return f"{_ERR_PREFIX} session {session_id!r} not found"
            if not cs.is_active:
                return (
                    f"{_REFUSED_PREFIX} session is {cs.status.value} "
                    f"(not ACTIVE)"
                )

            # Cap the timeout at the configured maximum
            timeout_s = cap_run_timeout(
                config=mgr.config, requested_s=timeout_s,
            )

            result = runner_run(
                argv=list(argv),
                cwd=cs.worktree_path,
                timeout_s=timeout_s,
            )

            try:
                mgr.record_run(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "coding_session_run: record_run failed: %s", exc,
                )

            return json.dumps(result.to_dict(), indent=2)

    # ── 5. diff ───────────────────────────────────────────────────

    class _DiffInput(BaseModel):
        session_id: str = Field(description="Session id from coding_session_start.")

    class CodingSessionDiffTool(BaseTool):
        name: str = "coding_session_diff"
        description: str = (
            "Show the cumulative diff for the worktree against its "
            "base sha. Useful for the agent's last-pass self-review "
            "before coding_session_submit."
        )
        args_schema: Type[BaseModel] = _DiffInput

        def _run(self, session_id: str) -> str:
            from app.coding_session import run as runner_run, runtime

            mgr = runtime.get_manager()
            cs = mgr.get(session_id)
            if cs is None:
                return f"{_ERR_PREFIX} session {session_id!r} not found"
            if not cs.is_active:
                return (
                    f"{_REFUSED_PREFIX} session is {cs.status.value} "
                    f"(not ACTIVE)"
                )

            # `git diff` against the base sha (HEAD inside the
            # detached worktree IS base_sha). Captures both modified
            # and untracked-as-added (with --no-index for added paths).
            # For simplicity we use the porcelain output via the
            # backend's already-tested list_changed_paths + targeted
            # `git diff` per file would be richer; for v1 we just emit
            # `git diff` against HEAD which covers M and D, plus a
            # note if there are A files.
            changes = mgr.backend.list_changed_paths(
                worktree_path=cs.worktree_path,
            )
            mgr.touch(session_id)

            # Use the runner's allowlist to fetch the diff itself —
            # `git diff` is on the read-only git list.
            diff_result = runner_run(
                argv=["git", "diff", "HEAD"],
                cwd=cs.worktree_path,
                timeout_s=30,
            )
            diff_text = diff_result.stdout or ""

            # Untracked (added) files don't show in `git diff HEAD`;
            # list them explicitly so the agent sees the full picture.
            added = [p for (p, k) in changes if k == "A"]
            if added:
                diff_text += "\n\n[Added files (untracked, no diff):]\n"
                diff_text += "\n".join(f"  + {p}" for p in added)

            if not diff_text.strip():
                return f"(clean worktree — no changes since {cs.base_sha[:8]})"
            return diff_text

    # ── 6. submit ─────────────────────────────────────────────────

    class _SubmitInput(BaseModel):
        session_id: str = Field(description="Session id from coding_session_start.")
        reason: str = Field(
            description=(
                "Operator-facing message explaining what was done + "
                "why. Appended to each change-request's reason field "
                "after the session's purpose. Include 'tests pass' / "
                "'verified XYZ' if you ran them — gives the operator "
                "confidence."
            ),
        )

    class CodingSessionSubmitTool(BaseTool):
        name: str = "coding_session_submit"
        description: str = (
            "Bundle the worktree's per-file diffs into change requests "
            "and finalize the session. THE ONLY ESCAPE HATCH from "
            "sandbox to production.\n\n"
            "Each modified file becomes its own change request — "
            "operator sees one Signal ASK per file with the diff and "
            "👍/👎 prompt. TIER_IMMUTABLE files are refused per-file; "
            "other files in the batch still submit normally.\n\n"
            "Worktree is destroyed after submit. Re-iteration requires "
            "a fresh coding_session_start.\n\n"
            "Returns a JSON summary with one row per file: "
            "{path, change_request_id, status, refusal_reason?}."
        )
        args_schema: Type[BaseModel] = _SubmitInput

        def _run(self, session_id: str, reason: str) -> str:
            from app.coding_session import (
                IllegalTransition, runtime, submit_session,
            )

            mgr = runtime.get_manager()
            try:
                _, results = submit_session(
                    session_id,
                    submit_reason=reason,
                    manager=mgr,
                )
            except IllegalTransition as exc:
                return f"{_ERR_PREFIX} {exc}"
            except Exception as exc:  # noqa: BLE001
                return f"{_ERR_PREFIX} submit failed: {exc}"

            return json.dumps({
                "session_id": session_id,
                "submitted": len([
                    r for r in results if r.change_request_id is not None
                    and r.refusal_reason is None
                ]),
                "refused": len([
                    r for r in results if r.refusal_reason is not None
                    or r.change_request_id is None
                ]),
                "results": [r.to_dict() for r in results],
            }, indent=2)

    # ── 7. discard ────────────────────────────────────────────────

    class _DiscardInput(BaseModel):
        session_id: str = Field(description="Session id from coding_session_start.")
        reason: str = Field(
            description=(
                "Why are you giving up? Captured for postmortem. "
                "'tests fail; root cause unclear' is more useful than "
                "'never mind'."
            ),
        )

    class CodingSessionDiscardTool(BaseTool):
        name: str = "coding_session_discard"
        description: str = (
            "Abandon the session without filing change requests. "
            "Worktree is destroyed; no production effect.\n\n"
            "Use this when iteration has stalled and you cannot reach "
            "a state worth submitting. The reason is logged for the "
            "operator to review."
        )
        args_schema: Type[BaseModel] = _DiscardInput

        def _run(self, session_id: str, reason: str) -> str:
            from app.coding_session import IllegalTransition, runtime

            mgr = runtime.get_manager()
            try:
                cs = mgr.discard(session_id, reason=reason)
            except IllegalTransition as exc:
                return f"{_ERR_PREFIX} {exc}"

            ok, err = mgr.remove_worktree(cs)
            if not ok:
                logger.warning(
                    "coding_session_discard: teardown failed: %s", err,
                )

            return _format_session(
                cs,
                header=f"Coding session {cs.id} discarded.",
            )

    # ── 8. evolve ─────────────────────────────────────────────────

    class _EvolveInput(BaseModel):
        session_id: str = Field(
            description="Session id from coding_session_start.",
        )
        initial_path: str = Field(
            description=(
                "Repo-relative path to the program to evolve. Must exist "
                "in the worktree. TIER_IMMUTABLE files and anything under "
                "app/subia/ are refused — those need Tier-3 amendment, "
                "not coding-session evolution."
            ),
        )
        evaluate_path: str = Field(
            description=(
                "Repo-relative path to a script that scores variants of "
                "initial_path. Higher score = better. ShinkaEvolve runs "
                "this against each variant to drive the search."
            ),
        )
        num_generations: int = Field(
            default=5,
            ge=1, le=20,
            description=(
                "Generations to run. Capped at 20 (MAX_GENERATIONS_INLINE) "
                "to keep this tool single-shot rather than a bulk cycle."
            ),
        )
        num_islands: int = Field(
            default=1,
            ge=1, le=3,
            description="Population islands. Capped at 3.",
        )
        max_cost_usd: float = Field(
            default=1.5,
            gt=0.0, le=5.0,
            description=(
                "Per-run dollar cap on LLM proposal cost. Capped at $5 "
                "(MAX_COST_USD_INLINE)."
            ),
        )

    class CodingSessionEvolveTool(BaseTool):
        name: str = "coding_session_evolve_solution"
        description: str = (
            "Run ShinkaEvolve population-based search inside the "
            "coding session — when a single-shot fix isn't working, "
            "this evolves variants of one file against one evaluator "
            "and returns the best diff (vs the initial program).\n\n"
            "Hard caps: 20 generations, 3 islands, $5 per run. Refused "
            "if the session isn't ACTIVE, if either path resolves "
            "outside the worktree, or if initial_path is in TIER_IMMUTABLE "
            "or under app/subia/ or is app/affect/goal_emitter.py.\n\n"
            "Returns a JSON object with status (improved / no_improvement "
            "/ refused / error / shinka_unavailable), baseline_score, "
            "best_score, delta, and the unified diff when improved. The "
            "diff is NOT applied — call coding_session_write + "
            "coding_session_submit to land it through the standard "
            "change-request gate."
        )
        args_schema: Type[BaseModel] = _EvolveInput

        def _run(
            self,
            session_id: str,
            initial_path: str,
            evaluate_path: str,
            num_generations: int = 5,
            num_islands: int = 1,
            max_cost_usd: float = 1.5,
        ) -> str:
            from app.coding_session.evolution_bridge import evolve_in_session

            try:
                result = evolve_in_session(
                    session_id=session_id,
                    initial_path=initial_path,
                    evaluate_path=evaluate_path,
                    num_generations=num_generations,
                    num_islands=num_islands,
                    max_cost_usd=max_cost_usd,
                )
            except Exception as exc:  # noqa: BLE001
                return f"{_ERR_PREFIX} evolve_in_session raised: {exc}"

            payload = {
                "status": result.status,
                "baseline_score": result.baseline_score,
                "best_score": result.best_score,
                "delta": result.delta,
                "generations_run": result.generations_run,
                "variants_evaluated": result.variants_evaluated,
                "duration_seconds": result.duration_seconds,
                "diff": result.diff,
                "error": result.error,
                "refusal_reason": result.refusal_reason,
            }
            if result.status == "refused":
                return f"{_REFUSED_PREFIX} {result.refusal_reason}"
            return json.dumps(payload, indent=2)

    return {
        "coding_session_start":           CodingSessionStartTool,
        "coding_session_read":            CodingSessionReadTool,
        "coding_session_write":           CodingSessionWriteTool,
        "coding_session_run":             CodingSessionRunTool,
        "coding_session_diff":            CodingSessionDiffTool,
        "coding_session_submit":          CodingSessionSubmitTool,
        "coding_session_discard":         CodingSessionDiscardTool,
        "coding_session_evolve_solution": CodingSessionEvolveTool,
    }


# ── Agent-id resolution stub ────────────────────────────────────────


def _resolve_agent_id() -> str:
    """Return the agent_id for the current invocation context.

    Phase 5.4-d uses a static stub. Phase 5.4-e wires the real
    agent_id from the BaseTool invocation context (the same plumb-
    through point that ``request_restricted_write``'s requestor
    field will eventually use). Until then, all coding sessions
    record ``"coder"`` as the agent — accurate for the only agent
    that has these tools in its inventory anyway.
    """
    return "coder"


# ── Module-level cache + factory ────────────────────────────────────


_TOOL_CLASSES: dict[str, type] | None = None


try:
    _TOOL_CLASSES = _build_tool_classes()
except Exception as exc:
    logger.debug(
        "coding_session_tools: deferred class build (%s)", exc,
    )


def create_coding_session_tools() -> list:
    """Factory for explicit injection (agents not using the registry).
    Returns one instance per tool, in the start/read/write/run/diff/
    submit/discard/evolve order."""
    global _TOOL_CLASSES
    if _TOOL_CLASSES is None:
        try:
            _TOOL_CLASSES = _build_tool_classes()
        except Exception as exc:
            logger.warning(
                "create_coding_session_tools: build failed: %s", exc,
            )
            return []
    return [_TOOL_CLASSES[name]() for name in (
        "coding_session_start",
        "coding_session_read",
        "coding_session_write",
        "coding_session_run",
        "coding_session_diff",
        "coding_session_submit",
        "coding_session_discard",
        "coding_session_evolve_solution",
    )]


# ── Tool registry annotations ───────────────────────────────────────


try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    _TOOL_REGISTRY_SPECS = {
        "coding_session_start": {
            "capabilities": ["writes-coding-session"],
            "description": (
                "Start a coding session: creates an ephemeral git "
                "worktree where you can read/write/run pytest before "
                "submission through the change-request human gate. "
                "Returns the session_id."
            ),
        },
        "coding_session_read": {
            "capabilities": ["reads-coding-session"],
            "description": (
                "Read a file inside the coding-session worktree."
            ),
        },
        "coding_session_write": {
            "capabilities": ["writes-coding-session"],
            "description": (
                "Write a file inside the coding-session worktree. "
                "Sandboxed; never reaches the live tree."
            ),
        },
        "coding_session_run": {
            "capabilities": ["runs-coding-session"],
            "description": (
                "Run pytest / lint / typecheck inside the coding-"
                "session worktree. Allowlist + CPU + wallclock + "
                "output cap; no network."
            ),
        },
        "coding_session_diff": {
            "capabilities": ["reads-coding-session"],
            "description": (
                "Show the worktree's cumulative diff against the "
                "base sha. Use before submit for self-review."
            ),
        },
        "coding_session_submit": {
            "capabilities": ["submits-coding-session"],
            "description": (
                "Bundle the worktree diff into change requests + "
                "finalize the session. The single escape hatch from "
                "sandbox to production."
            ),
        },
        "coding_session_discard": {
            "capabilities": ["reads-coding-session"],  # No write/run; closest
            "description": (
                "Abandon the session without submission. Worktree "
                "destroyed; no production effect."
            ),
        },
        "coding_session_evolve_solution": {
            "capabilities": ["runs-coding-session"],
            "description": (
                "Run ShinkaEvolve population search on one file in the "
                "coding-session worktree; return the best diff. Hard-"
                "capped at 20 generations / 3 islands / $5. Refused on "
                "TIER_IMMUTABLE / app/subia/ / goal_emitter paths. The "
                "diff is NOT applied — submit through the standard "
                "change-request gate."
            ),
        },
    }

    def _make_factory(tool_name: str):
        def factory():
            tools = create_coding_session_tools()
            for t in tools:
                if t.name == tool_name:
                    return t
            raise RuntimeError(
                f"coding_session_tools: build returned no {tool_name!r}"
            )
        factory.__name__ = f"_{tool_name}_factory"
        return factory

    for _name, _spec in _TOOL_REGISTRY_SPECS.items():
        register_tool(
            name=_name,
            capabilities=_spec["capabilities"],
            description=_spec["description"],
            tier=Tier.PRODUCTION,
            lifecycle=Lifecycle.SINGLETON,
        )(_make_factory(_name))

except ImportError:
    # Tool registry not available (e.g. in a unit-test process that
    # doesn't load the full app). The tools still work via
    # create_coding_session_tools(); they just won't be discoverable
    # via tool_search.
    pass
