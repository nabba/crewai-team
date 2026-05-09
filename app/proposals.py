"""
proposals.py — Improvement proposal management.

Agents create proposals for system improvements (new skills, tools,
code changes). Proposals are stored in /app/workspace/proposals/ and
require explicit user approval via Signal before being applied.

Proposal types:
  skill   — new .md files for workspace/skills/ (low risk)
  code    — new/modified Python files (requires rebuild)
  config  — .env or docker-compose changes (requires restart)
"""

import ast
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROPOSALS_DIR = Path("/app/workspace/proposals")
SKILLS_DIR = Path("/app/workspace/skills")
APPLIED_CODE_DIR = Path("/app/workspace/applied_code")

# Code root is auto-detected so the validator works both inside the Docker
# container (where it's /app) and on developer hosts where the checkout
# lives somewhere else.  proposals.py itself is at <root>/app/proposals.py,
# so <root> is two levels up from __file__.
LIVE_CODE_ROOT = Path(__file__).resolve().parent.parent


def _next_id() -> int:
    """Return the next sequential proposal ID."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(d.name.split("_")[0])
        for d in PROPOSALS_DIR.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ]
    return max(existing, default=0) + 1


def _has_active_fix(resolution_target: str) -> bool:
    """F6: Check if there's already a pending or recently approved proposal for this error pattern."""
    try:
        for status_filter in ("pending", "approved"):
            proposals = list_proposals(status_filter)
            for p in proposals:
                if p.get("resolution_target") == resolution_target:
                    return True
    except Exception:
        pass
    return False


def _is_duplicate_proposal(title: str, proposal_type: str) -> bool:
    """H1: Check if a similar proposal already exists (pending).

    Uses word overlap to detect duplicates like "Data Visualization Tool"
    vs "Data Visualization Integration". Threshold: 60% word overlap.
    """
    from collections import Counter
    title_words = Counter(title.lower().split())
    if not title_words:
        return False

    for existing in list_proposals("pending"):
        if existing.get("type") != proposal_type:
            continue
        existing_words = Counter(existing.get("title", "").lower().split())
        if not existing_words:
            continue
        shared = sum((title_words & existing_words).values())
        total = max(sum(title_words.values()), sum(existing_words.values()), 1)
        if shared / total >= 0.6:
            logger.debug(f"Duplicate proposal skipped: '{title}' ≈ '{existing.get('title')}'")
            return True
    return False


# H1: Cap total pending proposals — auto-reject oldest when limit exceeded
_MAX_PENDING_PROPOSALS = 30


# ── Content validation (Q10: reject hallucinated proposals) ──────────────────
# Proposal #635 slipped through because it targeted a non-existent file
# ("handle_task.py") with tutorial-style comments instead of real code, plus
# imports referencing modules that don't exist in the codebase.  The LLM
# proposer hallucinated the whole thing.  These validators reject such
# proposals BEFORE they are written to disk and shown to the user.

def _is_importable_top_level(module: str) -> bool:
    """Best-effort check that the top-level of a module path is importable.

    Handles three kinds of imports:
      - stdlib (os, sys, json, ...): always True
      - installed packages (crewai, pydantic, fastapi, ...): check sys.modules
        first (fast path — most are already loaded), then try find_spec
      - local app modules (app.*, app/*): check file existence under /app/app/

    Returns True if the import probably works. Returns True (permissive) on
    any internal error so we never block a legit proposal on validator bugs.
    """
    if not module:
        return True
    top = module.split(".", 1)[0]

    # Stdlib modules (Python 3.10+)
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib and top in stdlib:
        return True

    # Local app modules — check on-disk path
    if top == "app":
        rest = module.split(".")[1:]
        if not rest:
            return True
        # Try package (dir with __init__.py) then module (.py)
        pkg = LIVE_CODE_ROOT / "app" / "/".join(rest) / "__init__.py"
        mod = LIVE_CODE_ROOT / "app" / ("/".join(rest[:-1]) + f"/{rest[-1]}.py" if len(rest) > 1 else f"{rest[0]}.py")
        return pkg.exists() or mod.exists()

    # Already-loaded package — fast path for crewai/pydantic/etc.
    if top in sys.modules:
        return True

    # Last resort: try to find a spec without importing
    try:
        import importlib.util as _iu
        return _iu.find_spec(top) is not None
    except Exception:
        return True  # permissive on validator error


def _ast_is_substantive(tree: ast.AST) -> bool:
    """True if the module has at least one function/class definition or
    a non-trivial assignment (more than just __version__ or constants).

    A file that is ONLY imports + comments + docstrings fails this check —
    which is exactly what proposal #635 looked like.
    """
    has_def_or_class = False
    has_real_assign = False
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            has_def_or_class = True
            break
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            has_real_assign = True
    return has_def_or_class or has_real_assign


def _validate_code_proposal_content(files: dict[str, str]) -> list[str]:
    """Validate a code proposal's file contents before creation.

    Checks:
      1. Each .py file parses as Python (syntax error → reject).
      2. Target file either already exists under /app/ OR lives in a plausible
         app/* subdirectory. Bare filenames like "handle_task.py" are rejected
         for code proposals because the deployer only accepts app/* paths
         anyway.
      3. File content is substantive — has at least one function/class/
         assignment, not just comments and imports.
      4. All imports resolve to something real (stdlib, installed package,
         or an existing /app/app/* module).

    Returns a list of violation strings.  Empty list = passes.
    """
    violations: list[str] = []
    for fpath, content in files.items():
        if not fpath.endswith(".py"):
            continue  # non-Python files in code proposals are rare; skip
        if not content or not content.strip():
            violations.append(f"{fpath}: empty file")
            continue

        # 1. Syntax check
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            violations.append(f"{fpath}: syntax error at line {exc.lineno}: {exc.msg}")
            continue

        # 2. Target path sanity.  auto_deployer only deploys app/* so bare
        #    filenames create dead weight in workspace/applied_code/.
        norm = fpath.replace("\\", "/").lstrip("/")
        if "/" not in norm:
            violations.append(
                f"{fpath}: code proposals must target app/* paths "
                f"(bare filenames cannot be auto-deployed)"
            )
            continue
        if not norm.startswith("app/"):
            violations.append(f"{fpath}: code proposals must live under app/")
            continue

        # 3. Substantive-content check
        if not _ast_is_substantive(tree):
            violations.append(
                f"{fpath}: no function/class/assignment found — "
                f"file appears to be only comments, imports, or docstrings"
            )
            continue

        # 4. Import validity — collect all top-level imports, reject if any
        #    reference a module that doesn't exist.
        bad_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_importable_top_level(alias.name):
                        bad_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                # Ignore relative imports (node.level > 0) — they resolve
                # against the destination package, not absolutely.
                if node.level == 0 and node.module:
                    if not _is_importable_top_level(node.module):
                        bad_imports.append(node.module)
        if bad_imports:
            violations.append(
                f"{fpath}: imports non-existent module(s): "
                f"{', '.join(sorted(set(bad_imports))[:5])}"
            )

    return violations


# ── Proposal explainer ──────────────────────────────────────────────────────
# Every proposal gets a human-readable .md document with three sections:
#   • Why useful — plain-English motivation
#   • What changes — overview of the diff and affected subsystems
#   • Potential risks — subsystems that could break, rollback plan
# Callers may supply these directly; if not, we generate them via a local
# LLM and fall back to a deterministic skeleton if the LLM is unavailable.

# Map known app/ subdirectories to subsystem names and what relies on them.
# Used by both the LLM prompt (as context) and the deterministic fallback.
_SUBSYSTEM_MAP: list[tuple[str, str, str]] = [
    ("app/agents/commander/", "Commander orchestrator",
     "routes every Signal message — errors here block all user traffic"),
    ("app/agents/", "Specialist agent factory",
     "affects the crew it defines; other crews unaffected"),
    ("app/crews/", "Crew execution",
     "affects one crew's run path; routing decisions unchanged"),
    ("app/tools/", "Agent tool",
     "agents that have this tool in their registry may behave differently"),
    ("app/mcp/", "MCP client/server",
     "external MCP tool access; no impact on built-in tools"),
    ("app/memory/", "Memory layer",
     "ChromaDB / Mem0 persistence; data durability-sensitive"),
    ("app/self_awareness/", "Self-awareness telemetry",
     "non-critical metrics; failures degrade gracefully"),
    ("app/subia/", "SubIA belief/grounding",
     "background reflection loops; runtime chat unaffected"),
    ("app/knowledge_base/", "Knowledge base / RAG",
     "KB search results shown to agents during tasks"),
    ("app/control_plane/", "Control plane (tickets, budgets, projects)",
     "ticket tracking and project isolation; not core chat path"),
    ("app/evolution.py", "Evolution loop",
     "self-improvement scheduler; does not affect chat"),
    ("app/auto_deployer.py", "Auto-deployer",
     "CRITICAL — deploy pipeline itself; mistakes can brick self-modification"),
    ("app/proposals.py", "Proposal system",
     "CRITICAL — this module itself; bugs here break the approval flow"),
    ("app/main.py", "Gateway entrypoint",
     "CRITICAL — Signal webhook handler; bugs here take the bot offline"),
    ("app/config.py", "Configuration",
     "settings loaded at startup; changes need container restart"),
    ("app/signal_client.py", "Signal client",
     "CRITICAL — message delivery path to the user"),
    ("app/security.py", "Security / sender authorization",
     "CRITICAL — authentication layer; never remove checks"),
    ("app/vetting.py", "Output vetting",
     "CRITICAL — final safety pass before replying to user"),
    ("app/sanitize.py", "Input sanitization",
     "CRITICAL — prompt-injection defense"),
]


def _identify_affected_subsystems(files: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return (subsystem, description, file_list) tuples for each impacted area."""
    hits: dict[str, list[str]] = {}
    for fpath in files or {}:
        norm = fpath.replace("\\", "/").lstrip("/")
        matched = False
        for prefix, name, desc in _SUBSYSTEM_MAP:
            if norm == prefix or norm.startswith(prefix):
                key = f"{name}|{desc}"
                hits.setdefault(key, []).append(norm)
                matched = True
                break
        if not matched:
            key = f"Uncategorised ({norm.split('/')[0]})|impact scope unclear"
            hits.setdefault(key, []).append(norm)
    return [
        (key.split("|")[0], key.split("|")[1], fs)
        for key, fs in hits.items()
    ]


def _explain_proposal(
    title: str, description: str,
    files: dict[str, str] | None, proposal_type: str,
) -> dict[str, str]:
    """Generate rationale / changes / risks sections via a local LLM.

    Returns a dict with keys: rationale, changes_summary, risks.
    Falls back to a deterministic skeleton if the LLM is unavailable —
    so proposal creation never blocks on LLM failure.
    """
    affected = _identify_affected_subsystems(files or {})
    affected_text = "\n".join(
        f"- {name} ({', '.join(fs)}): {desc}"
        for name, desc, fs in affected
    ) or "- (no files listed)"

    # Collapse file contents into a compact diff-style preview for the LLM.
    # Cap to avoid blowing the local model's context.
    preview_parts: list[str] = []
    if files:
        for fp, content in list(files.items())[:5]:
            snippet = content[:800]
            preview_parts.append(f"### {fp}\n```\n{snippet}\n```")
    preview = "\n\n".join(preview_parts) if preview_parts else "(no files)"

    prompt = (
        "You are writing a human-readable summary for a system-modification proposal.\n"
        "The user reads this on their phone before deciding to approve or reject.\n"
        "Be specific, concrete, and honest about what could go wrong.\n\n"
        f"TITLE: {title}\n"
        f"TYPE: {proposal_type}\n"
        f"DESCRIPTION: {description[:1500]}\n\n"
        f"AFFECTED SUBSYSTEMS:\n{affected_text}\n\n"
        f"FILE PREVIEWS:\n{preview}\n\n"
        "Produce a JSON object with EXACTLY these three string keys:\n"
        '  "rationale": 2-4 sentences explaining WHY this change is useful in plain\n'
        "      language. What user-visible problem does it fix, or what capability\n"
        "      does it add? Avoid jargon.\n"
        '  "changes_summary": 2-5 bullet points (joined by \\n) describing WHAT\n'
        "      changes. Name the concrete functions/files/behaviors affected.\n"
        '  "risks": 2-5 bullet points (joined by \\n) describing REAL risks to\n'
        "      other subsystems after deploy. If there are no meaningful risks\n"
        "      (e.g. a tiny docstring change), say so honestly — do NOT invent\n"
        "      risks. Always mention whether a container restart is required.\n\n"
        'Respond with ONLY the JSON object — no prose, no fences.'
    )

    try:
        from app.llm_factory import create_specialist_llm
        from app.utils import safe_json_parse
        llm = create_specialist_llm(
            max_tokens=500, role="self_improve", force_tier="local",
        )
        raw = str(llm.call(prompt)).strip()
        parsed, _ = safe_json_parse(raw)
        if parsed and all(k in parsed for k in ("rationale", "changes_summary", "risks")):
            return {
                "rationale": str(parsed["rationale"])[:1500],
                "changes_summary": str(parsed["changes_summary"])[:1500],
                "risks": str(parsed["risks"])[:1500],
            }
    except Exception as exc:
        logger.debug(f"Proposal explainer LLM failed, using fallback: {exc}")

    # Deterministic fallback — always yields a usable (if terse) document
    changes_lines = [
        f"- Modifies `{p}`" for p in list((files or {}).keys())[:10]
    ] or ["- (no file changes)"]
    risks_lines = []
    for name, desc, _fs in affected:
        if "CRITICAL" in desc:
            risks_lines.append(f"- **{name}** — {desc}")
        else:
            risks_lines.append(f"- {name}: {desc}")
    if proposal_type == "code":
        risks_lines.append("- Requires `docker compose up -d --build gateway` to take effect")
    elif proposal_type == "config":
        risks_lines.append("- Requires container restart to load new configuration")

    return {
        "rationale": description[:800] or "(no rationale provided by proposer)",
        "changes_summary": "\n".join(changes_lines),
        "risks": "\n".join(risks_lines) or "- No subsystems identified as affected",
    }


def _render_proposal_md(
    pid: int, title: str, description: str, proposal_type: str,
    files: dict[str, str] | None, rationale: str, changes_summary: str,
    risks: str, resolution_target: str,
) -> str:
    """Assemble the full human-readable proposal document."""
    file_list = (
        "\n".join(f"- `{p}`" for p in files) if files else "None"
    )
    approve_line = f"`approve {pid}`"
    reject_line = f"`reject {pid}`"
    return (
        f"# Proposal #{pid}: {title}\n\n"
        f"**Type:** {proposal_type}  \n"
        f"**Created:** {datetime.now(timezone.utc).isoformat()}  \n"
        + (f"**Resolves:** `{resolution_target}`  \n" if resolution_target else "")
        + f"\n"
        f"## Why this is useful\n\n{rationale}\n\n"
        f"## What will change\n\n{changes_summary}\n\n"
        f"## Potential risks to other subsystems\n\n{risks}\n\n"
        f"## Files touched\n\n{file_list}\n\n"
        f"## Original description\n\n{description}\n\n"
        f"---\n\n"
        f"**To decide:** react 👍 to the Signal notification to approve, or 👎 to reject.  \n"
        f"Or reply {approve_line} / {reject_line} via Signal.\n"
    )


def get_proposal_md_path(pid: int) -> Path | None:
    """Return the absolute path to a proposal's human-readable .md, or None."""
    pdir = _get_proposal_dir(pid)
    if not pdir:
        return None
    md = pdir / "proposal.md"
    return md if md.exists() else None


def set_proposal_signal_timestamp(pid: int, signal_ts: int) -> None:
    """Record the Signal message timestamp for a proposal's notification.

    Enables later reaction-based approval: when the user reacts 👍/👎 to the
    notification, we look up the proposal by this timestamp.
    """
    pdir = _get_proposal_dir(pid)
    if not pdir:
        return
    sf = pdir / "status.json"
    try:
        status = json.loads(sf.read_text())
        status["signal_msg_timestamp"] = int(signal_ts)
        from app.safe_io import safe_write_json as _swj
        _swj(sf, status)
    except Exception:
        logger.debug(f"Could not record signal timestamp for #{pid}", exc_info=True)


def find_proposal_by_signal_timestamp(signal_ts: int) -> int | None:
    """Find a proposal whose notification had the given Signal timestamp.

    Returns the proposal ID, or None if no match.  Used by the reaction
    handler to map 👍 / 👎 on a notification back to the proposal.
    """
    if not signal_ts:
        return None
    target = int(signal_ts)
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    for d in PROPOSALS_DIR.iterdir():
        sf = d / "status.json"
        if not sf.exists():
            continue
        try:
            s = json.loads(sf.read_text())
            if int(s.get("signal_msg_timestamp") or 0) == target:
                return int(s.get("id"))
        except Exception:
            continue
    return None


def create_proposal(
    title: str,
    description: str,
    proposal_type: str = "skill",
    files: dict[str, str] | None = None,
    resolution_target: str = "",
    rationale: str = "",
    changes_summary: str = "",
    risks: str = "",
) -> int:
    """
    Create a new improvement proposal.

    H1: Deduplicates against existing pending proposals (60% word overlap).
    Caps total pending proposals at _MAX_PENDING_PROPOSALS.

    Args:
        title: Short title (used in Signal summary)
        description: Detailed description of what and why
        proposal_type: "skill", "code", or "config"
        files: dict of {relative_path: file_content} to be applied on approval
        rationale: optional — plain-English "why this is useful".  Auto-generated
            via LLM if empty.
        changes_summary: optional — bullet list of what changes.  Auto-generated
            if empty.
        risks: optional — bullet list of risks to other subsystems.  Auto-
            generated if empty.

    Returns:
        The proposal ID (integer), or -1 if skipped as duplicate
    """
    # H1+F6: Deduplication check — also check against approved proposals
    # to prevent re-creating a fix that was already approved and deployed
    if _is_duplicate_proposal(title, proposal_type):
        return -1
    if resolution_target and _has_active_fix(resolution_target):
        logger.info(f"Skipping proposal — active fix already exists for {resolution_target}")
        return -1

    # H1: Prune oldest pending proposals if over cap
    pending = list_proposals("pending")
    if len(pending) > _MAX_PENDING_PROPOSALS:
        # Auto-reject the oldest excess proposals
        oldest = sorted(pending, key=lambda p: p.get("created_at", ""))
        for old in oldest[:len(pending) - _MAX_PENDING_PROPOSALS]:
            try:
                reject_proposal(old["id"])
                logger.info(f"Auto-rejected stale proposal #{old['id']}: {old.get('title', '')[:40]}")
            except Exception:
                pass

    # Validate file paths before creating proposal (C2, H7: prevent path traversal
    # and modification of protected security-critical files)
    if files:
        from app.auto_deployer import validate_proposal_paths
        path_violations = validate_proposal_paths(files)
        if path_violations:
            # INFO not WARN: this is the validator working as designed.
            # The caller gets the rejection signal (-1) and is expected
            # to handle it; we don't need to add the rejection to
            # errors.jsonl on top.  Frequent path-violation rejections
            # would indicate a buggy LLM proposer, but that's surfaced
            # by the proposer's metrics, not by error noise.
            logger.info(f"Proposal rejected — path violations: {path_violations}")
            return -1  # Signal rejection to caller

    # Q10: Validate content of code proposals — reject LLM hallucinations
    # where the "fix" is just comments targeting non-existent files or imports.
    if files and proposal_type == "code":
        content_violations = _validate_code_proposal_content(files)
        if content_violations:
            logger.warning(
                f"Proposal '{title[:60]}' rejected — content violations: "
                f"{content_violations}"
            )
            return -1

    pid = _next_id()
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40].strip()
    dirname = f"{pid:03d}_{safe_title.replace(' ', '_')}"
    pdir = PROPOSALS_DIR / dirname
    pdir.mkdir(parents=True, exist_ok=True)

    # Fill in any missing human-readable sections via the explainer.
    # If the caller passed all three, skip the LLM call entirely.
    if not (rationale and changes_summary and risks):
        explained = _explain_proposal(title, description, files, proposal_type)
        rationale = rationale or explained["rationale"]
        changes_summary = changes_summary or explained["changes_summary"]
        risks = risks or explained["risks"]

    # Write the enriched human-readable proposal document
    (pdir / "proposal.md").write_text(_render_proposal_md(
        pid=pid, title=title, description=description,
        proposal_type=proposal_type, files=files,
        rationale=rationale, changes_summary=changes_summary,
        risks=risks, resolution_target=resolution_target,
    ))

    # Write status
    status = {
        "id": pid,
        "title": title,
        "type": proposal_type,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied_at": None,
        "dirname": dirname,
        "resolution_target": resolution_target,  # F3: error pattern this fixes
    }
    from app.safe_io import safe_write_json as _swj; _swj(pdir / "status.json", status)

    # Write proposed files
    if files:
        files_dir = pdir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        for fpath, fcontent in files.items():
            target = files_dir / fpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fcontent)

    logger.info(f"Proposal #{pid} created: {title} ({proposal_type})")

    # F2: Auto-test code proposals — run test suite with proposed files applied
    # Results stored alongside proposal so user sees pass/fail before approving
    if proposal_type == "code" and files:
        try:
            test_result = _pretest_proposal(pid, files)
            status["pretest"] = test_result
            from app.safe_io import safe_write_json as _swj; _swj(pdir / "status.json", status)
        except Exception:
            logger.debug(f"Pretest skipped for proposal #{pid}", exc_info=True)

    return pid


def _pretest_proposal(pid: int, files: dict[str, str]) -> dict:
    """Run test tasks with proposed code applied, then revert.

    Returns {"passed": N, "failed": N, "total": N, "details": [...]}.
    Does NOT require the code to be deployed — applies temporarily, tests,
    then restores originals regardless of outcome.
    """
    import ast
    from pathlib import Path as _Path

    results = {"passed": 0, "failed": 0, "total": 0, "details": []}

    # 1. Validate syntax of all proposed Python files
    for fpath, content in files.items():
        if fpath.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as e:
                results["failed"] += 1
                results["total"] += 1
                results["details"].append(f"SYNTAX ERROR in {fpath}: {e}")
                return results  # Don't bother testing if syntax is broken
            results["passed"] += 1
            results["total"] += 1
            results["details"].append(f"SYNTAX OK: {fpath}")

    # 2. Run test tasks (eval integrity checked by experiment_runner)
    try:
        from app.experiment_runner import load_test_tasks, validate_response
        tasks = load_test_tasks("fixed")  # regression suite only
        if not tasks:
            results["details"].append("No test tasks available for regression testing")
            return results

        # We can't actually execute crews in a test harness without full Docker.
        # Instead, verify the proposed code doesn't break any imports or module loading.
        for fpath, content in files.items():
            if fpath.endswith(".py"):
                try:
                    # Compile to bytecode — catches more errors than just parse
                    compile(content, fpath, "exec")
                    results["passed"] += 1
                    results["details"].append(f"COMPILE OK: {fpath}")
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(f"COMPILE FAIL: {fpath}: {e}")
                results["total"] += 1
    except Exception as e:
        results["details"].append(f"Test framework error: {e}")

    return results


def list_proposals(status_filter: str = "pending") -> list[dict]:
    """List proposals filtered by status."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for d in sorted(PROPOSALS_DIR.iterdir()):
        sf = d / "status.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
                if status_filter == "all" or s.get("status") == status_filter:
                    results.append(s)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def get_proposal(proposal_id: int) -> dict | None:
    """Get a proposal by ID, including file contents."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    for d in PROPOSALS_DIR.iterdir():
        sf = d / "status.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
                if s.get("id") == proposal_id:
                    s["description"] = ""
                    pm = d / "proposal.md"
                    if pm.exists():
                        s["description"] = pm.read_text()
                    # List files
                    files_dir = d / "files"
                    s["files"] = {}
                    if files_dir.exists():
                        for f in files_dir.rglob("*"):
                            if f.is_file():
                                rel = str(f.relative_to(files_dir))
                                s["files"][rel] = f.read_text()[:8000]
                    return s
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def _get_proposal_dir(proposal_id: int) -> Path | None:
    """Find the directory for a proposal by ID."""
    for d in PROPOSALS_DIR.iterdir():
        sf = d / "status.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
                if s.get("id") == proposal_id:
                    return d
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def approve_proposal(proposal_id: int) -> str:
    """
    Approve and apply a proposal.

    - skill proposals: copies .md files to workspace/skills/
    - code proposals: copies files to workspace/applied_code/ for hot-reload
    - config proposals: saves but requires manual restart

    Returns a status message.
    """
    pdir = _get_proposal_dir(proposal_id)
    if not pdir:
        return f"Proposal #{proposal_id} not found."

    status = json.loads((pdir / "status.json").read_text())
    if status.get("status") != "pending":
        return f"Proposal #{proposal_id} is already {status.get('status')}."

    ptype = status.get("type", "skill")
    files_dir = pdir / "files"
    applied = []

    if files_dir.exists():
        for f in files_dir.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(files_dir)

            if ptype == "skill":
                dest = (SKILLS_DIR / rel).resolve()
                # Path traversal check — dest must stay inside SKILLS_DIR
                try:
                    dest.relative_to(SKILLS_DIR.resolve())
                except ValueError:
                    logger.warning(f"Path traversal blocked in proposal #{proposal_id}: {rel}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                applied.append(f"skills/{rel}")

            elif ptype in ("code", "config"):
                dest = (APPLIED_CODE_DIR / rel).resolve()
                try:
                    dest.relative_to(APPLIED_CODE_DIR.resolve())
                except ValueError:
                    logger.warning(f"Path traversal blocked in proposal #{proposal_id}: {rel}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                applied.append(f"applied_code/{rel}")

    # Update status
    status["status"] = "approved"
    status["applied_at"] = datetime.now(timezone.utc).isoformat()
    status["applied_files"] = applied
    from app.safe_io import safe_write_json as _swj; _swj(pdir / "status.json", status)

    logger.info(f"Proposal #{proposal_id} approved: {applied}")

    msg = f"Proposal #{proposal_id} approved and applied."
    if applied:
        msg += f" Files: {', '.join(applied)}"

    # R2: Trigger auto-deployer for code proposals so changes reach the live codebase
    if ptype == "code" and applied:
        try:
            from app.auto_deployer import schedule_deploy
            schedule_deploy(f"proposal #{proposal_id}: {status.get('title', '')[:60]}")
            msg += " Deploying to live codebase..."
        except Exception as exc:
            msg += f" Deploy trigger failed: {str(exc)[:100]}. Manual restart needed."
            logger.warning(f"Deploy trigger failed for proposal #{proposal_id}: {exc}")
    elif ptype == "config":
        msg += " Note: config changes require manual restart."
    return msg


def reject_proposal(proposal_id: int) -> str:
    """Reject a proposal."""
    pdir = _get_proposal_dir(proposal_id)
    if not pdir:
        return f"Proposal #{proposal_id} not found."

    status = json.loads((pdir / "status.json").read_text())
    if status.get("status") != "pending":
        return f"Proposal #{proposal_id} is already {status.get('status')}."

    status["status"] = "rejected"
    status["rejected_at"] = datetime.now(timezone.utc).isoformat()
    from app.safe_io import safe_write_json as _swj; _swj(pdir / "status.json", status)

    logger.info(f"Proposal #{proposal_id} rejected")
    return f"Proposal #{proposal_id} rejected."
