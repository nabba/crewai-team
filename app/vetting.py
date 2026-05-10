"""
vetting.py — Risk-based selective verification for agent output.

Replaces the binary vet-local/skip-API approach with a 4-tier system:
  "none"   — skip verification (direct answers, premium+easy)
  "schema" — format/sanity check only, no LLM call
  "cheap"  — quick yes/no via budget model (DeepSeek V3.2)
  "full"   — full Claude Sonnet review (current behavior for local output)

Risk level is derived from: crew_type + difficulty_score + model_tier.
All local model output still gets full vetting. Code always gets full vetting.
"""

import concurrent.futures as _cf
import logging
import os
import re
import threading
import time
from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Bounded LLM call (2026-04-25) ─────────────────────────────────────────
# Vetting must NEVER hang the request lifecycle.  Task 88 stalled for 17.7
# minutes because gpt-5.5 (openrouter) didn't return for the vetting call,
# and `llm.call()` had no timeout — the LiteLLM HTTP-read default is far
# longer than the soft-timeout that ultimately killed the task, and the
# orchestrator's `_vet_future.result(timeout=30)` couldn't fire because the
# proactive_scan path on the same thread was also blocked on an LLM.
#
# This pool serializes vetting LLM calls behind a hard wall-clock ceiling.
# When the ceiling fires we abandon the worker (it keeps spinning until
# the upstream connection drops / retries exhaust — that's fine, we don't
# need its output) and the caller falls back to the unvetted response.
_VET_LLM_TIMEOUT_S = float(os.getenv("VETTING_LLM_TIMEOUT_S", "90"))
_vet_call_pool = _cf.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="vet-llm-call",
)


class VettingTimeout(Exception):
    """Raised when a vetting LLM call exceeds its wall-clock budget."""


def _call_llm_bounded(llm, prompt: str, timeout_s: float | None = None) -> str:
    """Run ``llm.call(prompt)`` with a hard wall-clock ceiling.

    Returns the stripped string output.  Raises ``VettingTimeout`` if the
    call doesn't complete in time — callers must catch this and fall back
    to returning the unvetted response.
    """
    deadline = float(timeout_s if timeout_s is not None else _VET_LLM_TIMEOUT_S)
    fut = _vet_call_pool.submit(lambda: str(llm.call(prompt)).strip())
    try:
        return fut.result(timeout=deadline)
    except _cf.TimeoutError as exc:
        # Don't cancel the future — concurrent.futures can't actually kill
        # a thread, so cancellation here would be a lie.  The worker
        # eventually unwinds when LiteLLM gives up; we just stop waiting.
        raise VettingTimeout(
            f"vetting LLM call exceeded {deadline:.0f}s"
        ) from exc


# ── Phase 4: vetting outcomes feed llm_benchmarks ────────────────────────────
# The vetting gate is the richest production-quality signal the system has
# — a failure means the model generated something that didn't survive a
# deeper review. Feeding that back into the benchmarks table closes the
# quality half of the feedback loop opened in Phase 1 (latency + success
# on the tool-call path) and Phase 3 (external rankings).


def _record_vetting_outcome(
    generating_model: str | None,
    crew_name: str,
    passed: bool,
    elapsed_ms: int,
) -> None:
    """Emit a benchmarks row tagged with the canonical task type.

    ``generating_model`` is the model that *produced* the artefact being
    vetted, not the model doing the vetting. Resolved via
    ``llm_factory.get_last_model()`` at the call site when the caller
    doesn't pass one explicitly. Silently skipped if no model can be
    attributed to the artefact (pure-cache paths, early returns, etc.)
    """
    if not generating_model:
        return
    try:
        from app.llm_benchmarks import record
        from app.llm_catalog import canonical_task_type
        task_type = canonical_task_type(role=crew_name, crew_name=crew_name)
        record(generating_model, task_type, passed,
               latency_ms=max(0, int(elapsed_ms)), tokens=0)
    except Exception:
        logger.debug("vetting: failed to record outcome", exc_info=True)

# ── Cached LLM singletons for vetting (avoid re-creation per call) ───────────
_cheap_llm = None
_full_llm = None
_llm_lock = threading.Lock()


def _get_cheap_vetting_llm():
    global _cheap_llm
    if _cheap_llm is not None:
        return _cheap_llm
    with _llm_lock:
        if _cheap_llm is None:
            from app.llm_factory import create_cheap_vetting_llm
            _cheap_llm = create_cheap_vetting_llm()
    return _cheap_llm


def _get_full_vetting_llm():
    global _full_llm
    if _full_llm is not None:
        return _full_llm
    with _llm_lock:
        if _full_llm is None:
            from app.llm_factory import create_vetting_llm
            _full_llm = create_vetting_llm()
    return _full_llm

# ── Vetting prompts (reused from original for full verification) ──────────────

VETTING_PROMPTS = {
    "coding": """\
You are a senior software architect reviewing code from an AI model.

1. CORRECTNESS: Does the code do what was asked? Logic bugs?
2. SECURITY: Injection, path traversal, unsafe operations?
3. BEST PRACTICES: Clean code, error handling, no anti-patterns?
4. COMPLETENESS: Does it fully solve the task?

If the code is good, return it with minimal changes.
If it has bugs, FIX THEM and return corrected code.
If it's poor quality, REWRITE it properly.
DO NOT add disclaimers. Return the final clean response.

USER REQUEST:
{request}

MODEL OUTPUT:
{response}

Return the vetted response only.
""",
    "research": """\
You are a fact-checker reviewing research from an AI model.

1. ACCURACY: Flag anything factually wrong or unverifiable
2. SOURCES: Remove hallucinated URLs
3. COMPLETENESS: Does it answer what was asked?
4. FORMAT: Clean up for Signal (concise, structured, under 1500 chars)

USER REQUEST:
{request}

MODEL RESEARCH:
{response}

CRITICAL: Return the vetted response ONLY. Do NOT add any reviewer notes, disclaimers, warnings, caveats, or meta-commentary. No footnotes. No "Note:" blocks. The user sees your output directly.
""",
    "writing": """\
You are an editor reviewing content from an AI model.

1. QUALITY: Clear, professional language?
2. ACCURACY: Factual claims correct?
3. COMPLETENESS: Covers what was requested?
4. FORMAT: Appropriate for Signal (concise, well-structured, under 1500 chars)

USER REQUEST:
{request}

MODEL CONTENT:
{response}

CRITICAL: Return the polished version ONLY. Do NOT add any reviewer notes, disclaimers, warnings, caveats, or meta-commentary. No footnotes. No "Note:" blocks. The user sees your output directly.
""",
    "media": """\
You are a media analysis reviewer checking output from an AI media analyst.

1. ACCURACY: Are descriptions factual? No hallucinated visual/audio details?
2. DATA: Are extracted numbers, quotes, and timestamps correct?
3. COMPLETENESS: Does it answer what was asked about the media?
4. FORMAT: Concise for Signal (under 1500 chars), structured clearly.

USER REQUEST:
{request}

MODEL ANALYSIS:
{response}

CRITICAL: Return the vetted analysis ONLY. Do NOT add any reviewer notes, disclaimers, warnings, caveats, or meta-commentary. No footnotes. The user sees your output directly.
""",
}

DEFAULT_VETTING_PROMPT = """\
You are a quality reviewer. An AI model produced this response.
Check for accuracy, completeness, and formatting. Fix any issues.

USER REQUEST:
{request}

MODEL RESPONSE:
{response}

CRITICAL: Return the clean response ONLY. Under 1500 chars for Signal. Do NOT add any reviewer notes, disclaimers, warnings, caveats, or meta-commentary. No footnotes. The user sees your output directly.
"""

CHEAP_VETTING_PROMPT = """\
Review this AI response for quality. Is it accurate, complete, and well-formatted?

USER REQUEST (brief): {request}

RESPONSE TO CHECK:
{response}

Reply with ONLY "PASS" if acceptable, or "FAIL: <one-line reason>" if not.
"""

# ── Failure patterns for schema check ─────────────────────────────────────────

_FAILURE_PATTERNS = [
    re.compile(r"^I (?:cannot|can't|am unable to|don't have)", re.IGNORECASE),
    re.compile(r"^(?:sorry|apologies|unfortunately),?\s+I", re.IGNORECASE),
    re.compile(r"^As an AI", re.IGNORECASE),
    re.compile(r"^\s*$"),  # empty
    # Meta-commentary: model describes what it will do instead of producing content
    re.compile(r"(?:moving forward|next,?)\s+I\s+will\b", re.IGNORECASE),
    re.compile(r"\bI\s+will\s+(?:now\s+)?(?:assess|evaluate|review|reflect|analyze|proceed)\b", re.IGNORECASE),
]


# ── Risk assessment ───────────────────────────────────────────────────────────

def assess_risk_level(
    crew_name: str, difficulty: int, model_tier: str,
) -> str:
    """Determine verification tier based on task context.

    Returns one of: "none", "schema", "cheap", "full".
    """
    # Direct answers need no verification
    if crew_name == "direct":
        return "none"

    # Code always gets full verification regardless of model tier
    if crew_name == "coding":
        return "full"

    # Local and free-tier model output always gets full verification
    if model_tier in ("local", "free"):
        return "full"

    # High difficulty always gets full verification
    if difficulty >= 8:
        return "full"

    # Easy tasks (difficulty <= 3) with premium or budget/mid models — trust them.
    # S5: Saves 2-4s per simple question by skipping the vetting LLM call.
    # Quality gate in commander already catches refusals and meta-commentary.
    if difficulty <= 3 and model_tier in ("premium", "budget", "mid"):
        return "none"

    # Budget/mid models on moderate writing/research — schema check only
    if model_tier in ("budget", "mid") and crew_name in ("writing", "research") and difficulty <= 5:
        return "schema"

    # Budget/mid models on moderate tasks — cheap LLM check
    if model_tier in ("budget", "mid") and difficulty <= 6:
        return "cheap"

    # Premium models on moderate tasks — schema check
    if model_tier == "premium" and difficulty <= 6:
        return "schema"

    # Default: full verification
    return "full"


# ── Verification implementations ──────────────────────────────────────────────

def _verify_schema(
    response: str, crew_name: str,
    generating_model: str | None = None,
) -> tuple[bool, str]:
    """Format and sanity check — no LLM call.

    Returns (passed, response). If failed, caller should escalate.
    Emits a benchmarks row for the generating model when known.
    """
    t_start = time.monotonic()
    text = response.strip()

    def _finish(ok: bool) -> tuple[bool, str]:
        _record_vetting_outcome(
            generating_model, crew_name, ok,
            int((time.monotonic() - t_start) * 1000),
        )
        return ok, response

    # Check for empty or near-empty
    if len(text) < 10:
        logger.info("vetting[schema]: failed — response too short")
        return _finish(False)

    # Check for known failure patterns
    for pattern in _FAILURE_PATTERNS:
        if pattern.match(text):
            logger.info(f"vetting[schema]: failed — matched failure pattern")
            return _finish(False)

    # Length sanity (Signal messages should be <1500 chars for writing/research)
    if crew_name in ("writing", "research") and len(text) > 4000:
        logger.info("vetting[schema]: warning — response exceeds 4000 chars, but passing")

    return _finish(True)


def _verify_cheap(
    user_request: str, response: str, crew_name: str,
    generating_model: str | None = None,
) -> tuple[bool, str]:
    """Quick yes/no check via budget model. Returns (passed, response).

    Records the verdict against ``generating_model`` — the model that
    *produced* the response, not the cheap judge doing the review.
    """
    t_start = time.monotonic()
    try:
        llm = _get_cheap_vetting_llm()
        prompt = CHEAP_VETTING_PROMPT.format(
            request=user_request[:400],
            response=response[:3000],
        )
        # Direct LLM call — no Agent/Task/Crew overhead.
        # Bounded so a hung budget model can't gate delivery.
        try:
            result = _call_llm_bounded(llm, prompt).upper()
        except VettingTimeout as _tmo:
            logger.warning(
                f"vetting[cheap]: {crew_name} LLM call timed out ({_tmo}); "
                f"escalating to full"
            )
            _record_vetting_outcome(
                generating_model, crew_name, False,
                int((time.monotonic() - t_start) * 1000),
            )
            return False, response

        passed = result.startswith("PASS")
        if passed:
            logger.info(f"vetting[cheap]: PASS for {crew_name}")
        else:
            logger.info(f"vetting[cheap]: FAIL for {crew_name}: {result[:100]}")
        _record_vetting_outcome(
            generating_model, crew_name, passed,
            int((time.monotonic() - t_start) * 1000),
        )
        return passed, response

    except Exception as exc:
        logger.warning(f"vetting[cheap]: error ({exc}), escalating to full")
        _record_vetting_outcome(
            generating_model, crew_name, False,
            int((time.monotonic() - t_start) * 1000),
        )
        return False, response


_FULL_VETTING_PROMPT = """\
You are a quality gate reviewing AI output before delivery to a user.

USER REQUEST:
{request}

AI RESPONSE TO REVIEW:
{response}

TASK: Check this response for factual errors, hallucinated URLs/data, \
and completeness. Reply with EXACTLY this JSON format:

{{"verdict": "PASS"}} — if the response is accurate and complete.

{{"verdict": "FAIL", "issues": ["issue1", "issue2"], "corrected": "..."}} — \
if there are factual errors. "corrected" must contain the ORIGINAL response \
with ONLY the incorrect parts fixed. Do NOT rewrite, rephrase, shorten, \
or add commentary. Preserve the original structure, data points, and sources. \
Only change what is factually wrong.

Reply with the JSON object ONLY — no markdown fences, no prose.
"""


def _verify_full(
    user_request: str, response: str, crew_name: str,
    generating_model: str | None = None,
) -> str:
    """Full Claude Sonnet verification — pass/fail with targeted corrections only.

    Changed from rewrite mode to pass/fail+correct (Q3): Sonnet returns the
    original response with only factual errors fixed, preserving all data
    points, sources, and structure from the original.

    Includes L4 conscience check for irreversible/high-impact actions.
    Records the verdict against ``generating_model``.
    """
    t_start = time.monotonic()
    passed = False
    try:
        llm = _get_full_vetting_llm()
        prompt = _FULL_VETTING_PROMPT.format(
            request=user_request[:800],
            response=response[:6000],
        )
        # Direct LLM call — no Agent/Task/Crew overhead.
        # Hard wall-clock ceiling: vetting MUST NOT hang the request
        # lifecycle.  See _call_llm_bounded docstring for the 2026-04-25
        # task-88 outage that motivated this guard.
        try:
            raw = _call_llm_bounded(llm, prompt)
        except VettingTimeout as _tmo:
            logger.warning(
                f"vetting[full]: {crew_name} LLM call timed out ({_tmo}); "
                f"returning unvetted response"
            )
            _set_last_verdict(True)  # treat as PASS to skip retry path
            _record_vetting_outcome(
                generating_model, crew_name, False,
                int((time.monotonic() - t_start) * 1000),
            )
            return response

        # Parse structured verdict
        from app.utils import safe_json_parse
        parsed, err = safe_json_parse(raw)

        captured_issues: list[str] = []
        if parsed and isinstance(parsed, dict):
            verdict = parsed.get("verdict", "").upper()
            if verdict == "PASS":
                logger.info(f"vetting[full]: {crew_name} PASSED")
                result = response  # return ORIGINAL unchanged
                passed = True
            elif verdict == "FAIL":
                issues = parsed.get("issues", [])
                corrected = parsed.get("corrected", "")
                logger.info(f"vetting[full]: {crew_name} FAILED: {issues}")
                # 2026-04-26: capture the structured issues list so the
                # orchestrator's retry path can build a targeted hint
                # ("specifically these rows are missing X, Y") instead of
                # the generic "produce a substantive answer" boilerplate.
                if isinstance(issues, list):
                    captured_issues = [str(i) for i in issues if i]
                # Cure C (2026-05-10): stash the failure on the task's
                # progress tracker so the watchdog's apology message
                # can name the specific reasons instead of generic
                # "please re-send a narrower question".
                try:
                    from app.observability.task_progress import (
                        record_failure_context,
                    )
                    detail = "; ".join(captured_issues[:3]) if captured_issues else f"crew={crew_name}"
                    record_failure_context("vetting_fail", detail)
                except Exception:
                    pass
                # Use corrected version if provided and substantive
                if corrected and len(corrected) > len(response) * 0.5:
                    result = corrected
                    # Treat a reliable correction as a conditional pass
                    # (the reviewer replaced factual errors in place).
                    passed = True
                else:
                    result = response  # corrections too aggressive, keep original
                    # Genuine failure — no usable correction.  passed stays False.
            else:
                logger.warning(f"vetting[full]: unexpected verdict '{verdict}', keeping original")
                result = response
        else:
            # Couldn't parse JSON — check if it's a plain PASS/FAIL
            if raw.upper().startswith("PASS"):
                logger.info(f"vetting[full]: {crew_name} PASSED (plain text)")
                result = response
                passed = True
            else:
                logger.warning(f"vetting[full]: unparseable response, keeping original")
                result = response
        # Surface the verdict + issues list on the thread-local so
        # vet_response_detailed() can propagate them to the
        # orchestrator's retry logic.
        _set_last_verdict(passed)
        _set_last_issues(captured_issues)

        # L4: Conscience check — flag irreversible actions
        conscience_ok, conscience_reason = _conscience_check(result)
        if not conscience_ok:
            logger.warning(f"vetting[conscience]: {conscience_reason}")
            result += f"\n\nNote: {conscience_reason}"

        _record_vetting_outcome(
            generating_model, crew_name, passed,
            int((time.monotonic() - t_start) * 1000),
        )
        return result

    except Exception as exc:
        logger.warning(f"vetting[full]: failed ({exc}), returning unvetted response")
        _record_vetting_outcome(
            generating_model, crew_name, False,
            int((time.monotonic() - t_start) * 1000),
        )

    return response


# ── L4: Conscience Check (rule-based, no LLM call) ───────────────────────────

# Keywords that suggest irreversible or high-impact actions
_IRREVERSIBLE_KEYWORDS = [
    "delete permanently", "drop table", "rm -rf", "truncate table",
    "revoke access", "format disk", "overwrite", "destroy",
    "DROP DATABASE", "TRUNCATE", "shred", "purge all",
]

def _conscience_check(response: str) -> tuple[bool, str]:
    """L4 constitutional conscience check — rule-based, no LLM call.

    Scans for irreversible action keywords and unqualified absolute language.
    Returns (passed, reason). If failed, caller should append a warning.
    """
    if not response:
        return True, ""

    response_lower = response.lower()

    # Check for irreversible action keywords
    for kw in _IRREVERSIBLE_KEYWORDS:
        if kw.lower() in response_lower:
            return False, (
                f"Irreversible action detected: '{kw}'. "
                "Per constitution: irreversible actions require extra scrutiny."
            )

    return True, ""


# ── Public API ────────────────────────────────────────────────────────────────

# Module-level thread-local verdict + issues capture.  vet_response()
# returns a plain string (back-compat with many callers) but also stores
# the pass/fail verdict and the structured issues list on per-thread
# variables so new callers can read them via vet_response_detailed()
# without changing the primary API.
_LAST_VETTING_PASSED = threading.local()
_LAST_VETTING_ISSUES = threading.local()


def _set_last_verdict(passed: bool) -> None:
    _LAST_VETTING_PASSED.value = bool(passed)


def _get_last_verdict() -> bool:
    return getattr(_LAST_VETTING_PASSED, "value", True)


def _set_last_issues(issues: list[str]) -> None:
    _LAST_VETTING_ISSUES.value = list(issues or [])


def _get_last_issues() -> list[str]:
    return list(getattr(_LAST_VETTING_ISSUES, "value", []) or [])


def vet_response_detailed(
    user_request: str,
    local_response: str,
    crew_name: str,
    difficulty: int = 5,
    model_tier: str = "unknown",
    generating_model: str | None = None,
) -> tuple[str, bool, list[str]]:
    """Like vet_response but also returns the verdict + issues list.

    Returns (vetted_text, passed, issues).
      * passed=True   → response PASSed unchanged or was mechanically
                        corrected; deliver as-is.
      * passed=False  → FAIL verdict fired; ``issues`` is the structured
                        list of complaints from the vetting LLM (e.g.
                        ["row 5 LinkedIn URL wrong", "rows 7-12 missing
                        Sales Leader names"]). The orchestrator uses
                        this to build a targeted retry hint instead of
                        generic boilerplate.

    Backward-compat: callers that only need (text, passed) can ignore
    the third element via tuple unpacking with ``*_``.
    """
    text = vet_response(
        user_request, local_response, crew_name,
        difficulty, model_tier, generating_model,
    )
    return text, _get_last_verdict(), _get_last_issues()


def vet_response(
    user_request: str,
    local_response: str,
    crew_name: str,
    difficulty: int = 5,
    model_tier: str = "unknown",
    generating_model: str | None = None,
) -> str:
    """
    Risk-based selective verification of agent output.

    Determines the appropriate verification level based on crew type,
    task difficulty, and model tier, then applies the corresponding check.

    ``generating_model`` is the catalog key of the model that produced
    ``local_response``. Each vetting stage records its pass/fail verdict
    against that model in the benchmarks table so the selector can learn
    from real quality outcomes. Defaults to
    ``llm_factory.get_last_model()`` when omitted.
    """
    settings = get_settings()

    if not settings.vetting_enabled:
        return local_response

    if not local_response or len(local_response.strip()) < 10:
        return local_response

    if generating_model is None:
        try:
            from app.llm_factory import get_last_model
            generating_model = get_last_model()
        except Exception:
            generating_model = None

    risk = assess_risk_level(crew_name, difficulty, model_tier)
    logger.info(
        f"vetting: crew={crew_name} difficulty={difficulty} tier={model_tier} → risk={risk}"
    )

    if risk == "none":
        _set_last_verdict(True)
        return local_response

    if risk == "schema":
        passed, result = _verify_schema(local_response, crew_name, generating_model)
        if passed:
            _set_last_verdict(True)
            return result
        # Schema failed → escalate to cheap
        logger.info("vetting: schema failed, escalating to cheap verification")
        passed, result = _verify_cheap(user_request, local_response, crew_name, generating_model)
        if passed:
            _set_last_verdict(True)
            return result
        # Cheap also failed → full verification
        logger.info("vetting: cheap failed, escalating to full verification")
        return _run_full_and_record(user_request, local_response, crew_name, generating_model)

    if risk == "cheap":
        passed, result = _verify_cheap(user_request, local_response, crew_name, generating_model)
        if passed:
            _set_last_verdict(True)
            return result
        # Cheap failed → escalate to full
        logger.info("vetting: cheap failed, escalating to full verification")
        return _run_full_and_record(user_request, local_response, crew_name, generating_model)

    # risk == "full"
    return _run_full_and_record(user_request, local_response, crew_name, generating_model)


def _run_full_and_record(
    user_request: str, local_response: str, crew_name: str,
    generating_model: str | None,
) -> str:
    """Wrapper around _verify_full that also stores the verdict in the
    thread-local so vet_response_detailed() can surface it.  We infer
    pass/fail from whether _verify_full returned the original text or a
    corrected version — which is imprecise, so _verify_full sets the
    verdict directly on the same thread-local via _set_last_verdict().
    Falls back to True on any setup error so callers don't retry
    unnecessarily."""
    # Reset before call so a previous request's verdict doesn't leak.
    _set_last_verdict(True)
    try:
        return _verify_full(user_request, local_response, crew_name, generating_model)
    except Exception:
        _set_last_verdict(True)  # on unexpected exception, don't trigger retry
        return local_response
