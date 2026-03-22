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

import logging
import re
import threading
from app.config import get_settings

logger = logging.getLogger(__name__)

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

def _verify_schema(response: str, crew_name: str) -> tuple[bool, str]:
    """Format and sanity check — no LLM call.

    Returns (passed, response). If failed, caller should escalate.
    """
    text = response.strip()

    # Check for empty or near-empty
    if len(text) < 10:
        logger.info("vetting[schema]: failed — response too short")
        return False, response

    # Check for known failure patterns
    for pattern in _FAILURE_PATTERNS:
        if pattern.match(text):
            logger.info(f"vetting[schema]: failed — matched failure pattern")
            return False, response

    # Length sanity (Signal messages should be <1500 chars for writing/research)
    if crew_name in ("writing", "research") and len(text) > 4000:
        logger.info("vetting[schema]: warning — response exceeds 4000 chars, but passing")

    return True, response


def _verify_cheap(user_request: str, response: str, crew_name: str) -> tuple[bool, str]:
    """Quick yes/no check via budget model. Returns (passed, response)."""
    try:
        llm = _get_cheap_vetting_llm()
        prompt = CHEAP_VETTING_PROMPT.format(
            request=user_request[:400],
            response=response[:3000],
        )
        # Direct LLM call — no Agent/Task/Crew overhead
        result = str(llm.call(prompt)).strip().upper()

        if result.startswith("PASS"):
            logger.info(f"vetting[cheap]: PASS for {crew_name}")
            return True, response
        else:
            logger.info(f"vetting[cheap]: FAIL for {crew_name}: {result[:100]}")
            return False, response

    except Exception as exc:
        logger.warning(f"vetting[cheap]: error ({exc}), escalating to full")
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


def _verify_full(user_request: str, response: str, crew_name: str) -> str:
    """Full Claude Sonnet verification — pass/fail with targeted corrections only.

    Changed from rewrite mode to pass/fail+correct (Q3): Sonnet returns the
    original response with only factual errors fixed, preserving all data
    points, sources, and structure from the original.

    Includes L4 conscience check for irreversible/high-impact actions.
    """
    try:
        llm = _get_full_vetting_llm()
        prompt = _FULL_VETTING_PROMPT.format(
            request=user_request[:800],
            response=response[:6000],
        )
        # Direct LLM call — no Agent/Task/Crew overhead
        raw = str(llm.call(prompt)).strip()

        # Parse structured verdict
        from app.utils import safe_json_parse
        parsed, err = safe_json_parse(raw)

        if parsed and isinstance(parsed, dict):
            verdict = parsed.get("verdict", "").upper()
            if verdict == "PASS":
                logger.info(f"vetting[full]: {crew_name} PASSED")
                result = response  # return ORIGINAL unchanged
            elif verdict == "FAIL":
                issues = parsed.get("issues", [])
                corrected = parsed.get("corrected", "")
                logger.info(f"vetting[full]: {crew_name} FAILED: {issues}")
                # Use corrected version if provided and substantive
                if corrected and len(corrected) > len(response) * 0.5:
                    result = corrected
                else:
                    result = response  # corrections too aggressive, keep original
            else:
                logger.warning(f"vetting[full]: unexpected verdict '{verdict}', keeping original")
                result = response
        else:
            # Couldn't parse JSON — check if it's a plain PASS/FAIL
            if raw.upper().startswith("PASS"):
                logger.info(f"vetting[full]: {crew_name} PASSED (plain text)")
                result = response
            else:
                logger.warning(f"vetting[full]: unparseable response, keeping original")
                result = response

        # L4: Conscience check — flag irreversible actions
        conscience_ok, conscience_reason = _conscience_check(result)
        if not conscience_ok:
            logger.warning(f"vetting[conscience]: {conscience_reason}")
            result += f"\n\nNote: {conscience_reason}"

        return result

    except Exception as exc:
        logger.warning(f"vetting[full]: failed ({exc}), returning unvetted response")

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

def vet_response(
    user_request: str,
    local_response: str,
    crew_name: str,
    difficulty: int = 5,
    model_tier: str = "unknown",
) -> str:
    """
    Risk-based selective verification of agent output.

    Determines the appropriate verification level based on crew type,
    task difficulty, and model tier, then applies the corresponding check.
    """
    settings = get_settings()

    if not settings.vetting_enabled:
        return local_response

    if not local_response or len(local_response.strip()) < 10:
        return local_response

    risk = assess_risk_level(crew_name, difficulty, model_tier)
    logger.info(
        f"vetting: crew={crew_name} difficulty={difficulty} tier={model_tier} → risk={risk}"
    )

    if risk == "none":
        return local_response

    if risk == "schema":
        passed, result = _verify_schema(local_response, crew_name)
        if passed:
            return result
        # Schema failed → escalate to cheap
        logger.info("vetting: schema failed, escalating to cheap verification")
        passed, result = _verify_cheap(user_request, local_response, crew_name)
        if passed:
            return result
        # Cheap also failed → full verification
        logger.info("vetting: cheap failed, escalating to full verification")
        return _verify_full(user_request, local_response, crew_name)

    if risk == "cheap":
        passed, result = _verify_cheap(user_request, local_response, crew_name)
        if passed:
            return result
        # Cheap failed → escalate to full
        logger.info("vetting: cheap failed, escalating to full verification")
        return _verify_full(user_request, local_response, crew_name)

    # risk == "full"
    return _verify_full(user_request, local_response, crew_name)
