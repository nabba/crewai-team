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
from crewai import Agent, Task, Crew, Process
from app.config import get_settings

logger = logging.getLogger(__name__)

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

Return the vetted response only — no meta-commentary.
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

Return the polished version only — no disclaimers.
""",
}

DEFAULT_VETTING_PROMPT = """\
You are a quality reviewer. An AI model produced this response.
Check for accuracy, completeness, and formatting. Fix any issues.
Return the clean response only — no disclaimers. Under 1500 chars for Signal.

USER REQUEST:
{request}

MODEL RESPONSE:
{response}
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

    # Local model output always gets full verification
    if model_tier == "local":
        return "full"

    # High difficulty always gets full verification
    if difficulty >= 8:
        return "full"

    # Premium models on easy tasks — trust them
    if model_tier == "premium" and difficulty <= 3:
        return "none"

    # Budget/mid models on easy writing/research — schema check only
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
        from app.llm_factory import create_cheap_vetting_llm
        llm = create_cheap_vetting_llm()

        agent = Agent(
            role="Quick Reviewer",
            goal="Quickly assess if a response is acceptable quality.",
            backstory="You do fast quality checks on AI output. Be brief.",
            llm=llm, tools=[], verbose=False,
        )

        task = Task(
            description=CHEAP_VETTING_PROMPT.format(
                request=user_request[:400],
                response=response[:3000],
            ),
            expected_output='Either "PASS" or "FAIL: reason"',
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = str(crew.kickoff()).strip().upper()

        if result.startswith("PASS"):
            logger.info(f"vetting[cheap]: PASS for {crew_name}")
            return True, response
        else:
            logger.info(f"vetting[cheap]: FAIL for {crew_name}: {result[:100]}")
            return False, response

    except Exception as exc:
        logger.warning(f"vetting[cheap]: error ({exc}), escalating to full")
        return False, response


def _verify_full(user_request: str, response: str, crew_name: str) -> str:
    """Full Claude Sonnet verification — the original vetting logic."""
    try:
        from app.llm_factory import create_vetting_llm
        llm = create_vetting_llm()
        prompt_template = VETTING_PROMPTS.get(crew_name, DEFAULT_VETTING_PROMPT)

        agent = Agent(
            role="Quality Reviewer",
            goal="Ensure response quality, accuracy, and security before delivery.",
            backstory=(
                "You are the final quality gate. You review output from AI "
                "models before it reaches the user. You catch bugs, hallucinations, "
                "and security issues. Be concise — the output goes to Signal."
            ),
            llm=llm, tools=[], verbose=False,
        )

        task = Task(
            description=prompt_template.format(
                request=user_request[:800],
                response=response[:6000],
            ),
            expected_output="A clean, vetted response ready for the user.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        vetted = str(crew.kickoff()).strip()
        if vetted and len(vetted) > 20:
            logger.info(f"vetting[full]: {crew_name} vetted ({len(response)}→{len(vetted)} chars)")
            return vetted

    except Exception as exc:
        logger.warning(f"vetting[full]: failed ({exc}), returning unvetted response")

    return response


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
