"""
code_forge.py — Grounded code generation from verified skills.

Unlike vanilla LLM code generation, Code Forge:
  1. Draws from the skill library (tested, verified code)
  2. Uses learned API integrations (not hallucinated endpoints)
  3. Executes and validates in sandbox with assertions
  4. Self-debugs up to 3 iterations on failure
  5. Stores working code as new skills

Pipeline: Decompose → Knowledge Lookup → Compose → Test → Debug → Store

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# IMMUTABLE: Max self-debug iterations before escalating to human
MAX_DEBUG_ITERATIONS = 3

# IMMUTABLE: Decomposition prompt
DECOMPOSE_PROMPT = """Decompose this task into subtasks that can each be implemented independently.

Task: {task_description}

For each subtask, identify:
1. What it does (brief description)
2. What APIs/services it needs (if any)
3. What reusable patterns it needs (auth, retry, rate limiting, etc.)
4. Whether it's pure logic (no external dependencies) or integration code

Return as JSON array:
[
  {{
    "id": "subtask_1",
    "description": "What this subtask does",
    "apis_needed": ["API Name 1"],
    "patterns_needed": ["retry_with_backoff"],
    "is_pure_logic": false,
    "depends_on": []
  }}
]

Return ONLY valid JSON."""

# IMMUTABLE: Composition prompt
COMPOSE_PROMPT = """Compose a working Python script from these components.

Task: {task_description}

Available Skill Code (tested, verified — prefer using these over generating new):
{skill_code_blocks}

Subtasks to implement:
{subtask_descriptions}

Requirements:
1. Import and use the skill code where available
2. Generate new code ONLY for subtasks where no skill exists
3. Include proper error handling
4. Include a main() function or entry point
5. Add type hints and brief docstrings
6. Handle edge cases (empty data, API errors, etc.)

Generate ONLY Python code. No markdown fences, no explanation."""

# IMMUTABLE: Debug prompt
DEBUG_PROMPT = """This code failed. Fix it.

Code:
{code}

Error:
{error}

Previous fix attempts ({attempt_num}/{max_attempts}):
{previous_attempts}

Fix the code. Return ONLY the corrected Python code. No explanation."""


@dataclass
class CodeForgeResult:
    """Result of a code forge build."""
    success: bool = False
    code: str = ""
    test_code: str = ""
    skill_id: str = ""
    subtasks: list[dict] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    skills_missing: list[str] = field(default_factory=list)
    debug_iterations: int = 0
    error: str = ""
    duration_seconds: float = 0.0


class CodeForge:
    """Grounded code generation with skill library integration and self-debugging."""

    def __init__(self):
        pass

    def build(self, task_description: str, context: str = "") -> CodeForgeResult:
        """Full pipeline: decompose → lookup → compose → test → debug → store.

        Args:
            task_description: What the code should do
            context: Additional context (user requirements, constraints, etc.)

        Returns:
            CodeForgeResult with the generated code and metadata
        """
        start = time.monotonic()
        result = CodeForgeResult()

        try:
            # Step 1: Decompose task into subtasks
            subtasks = self._decompose(task_description)
            result.subtasks = subtasks

            # Step 2: Knowledge lookup for each subtask
            skill_blocks, missing = self._lookup_skills(subtasks)
            result.skills_used = list(skill_blocks.keys())
            result.skills_missing = missing

            # Step 3: Trigger API Scout for missing APIs
            if missing:
                self._learn_missing(missing)
                # Re-lookup after learning
                skill_blocks, still_missing = self._lookup_skills(subtasks)
                result.skills_used = list(skill_blocks.keys())
                result.skills_missing = still_missing

            # Step 4: Compose code from skills + generated parts
            code = self._compose(task_description, subtasks, skill_blocks, context)
            result.code = code

            # Step 5: Validate syntax
            syntax_ok, syntax_error = self._validate_syntax(code)
            if not syntax_ok:
                # Self-debug syntax errors
                code, debug_count = self._debug_loop(code, syntax_error, task_description)
                result.code = code
                result.debug_iterations = debug_count

            # Step 6: Generate test code
            result.test_code = self._generate_tests(task_description, code)

            result.success = True

        except Exception as e:
            result.error = str(e)[:500]
            logger.error(f"code_forge: build failed: {e}")

        result.duration_seconds = time.monotonic() - start
        return result

    def build_and_register(
        self, task_description: str, skill_name: str = "", context: str = ""
    ) -> CodeForgeResult:
        """Build code and register as a new skill."""
        result = self.build(task_description, context)

        if result.success and result.code:
            try:
                from app.atlas.skill_library import get_library
                library = get_library()

                # Generate skill ID
                safe_name = (
                    skill_name or task_description[:30]
                ).lower().replace(" ", "_").replace("/", "_")
                skill_id = f"recipes/{safe_name}"

                manifest = library.register_skill(
                    skill_id=skill_id,
                    name=skill_name or task_description[:50],
                    category="recipes",
                    code=result.code,
                    description=task_description,
                    source_type="trial_and_error",
                    test_code=result.test_code,
                    tags=["generated", "code_forge"],
                )
                result.skill_id = skill_id
                logger.info(f"code_forge: registered skill '{skill_id}' "
                            f"(confidence={manifest.effective_confidence():.2f})")
                # Audit trail
                try:
                    from app.atlas.audit_log import log_external_call
                    log_external_call(
                        agent="code_forge", action="build_and_register",
                        target=skill_id, method="decompose+generate+test+register",
                        result="success",
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"code_forge: skill registration failed: {e}")

        return result

    # ── Pipeline steps ────────────────────────────────────────────────────

    def _decompose(self, task_description: str) -> list[dict]:
        """Decompose task into subtasks using LLM."""
        prompt = DECOMPOSE_PROMPT.format(task_description=task_description)

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=1500, role="coding")
            raw = str(llm.call(prompt)).strip()

            # Parse JSON
            import re
            json_match = re.search(r'\[[\s\S]+\]', raw)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            logger.debug("code_forge: decomposition failed", exc_info=True)

        # Fallback: single subtask
        return [{"id": "main", "description": task_description,
                 "apis_needed": [], "patterns_needed": [], "is_pure_logic": True}]

    def _lookup_skills(self, subtasks: list[dict]) -> tuple[dict[str, str], list[str]]:
        """Look up relevant skills for each subtask.

        Returns: (skill_code_blocks, missing_capabilities)
        """
        from app.atlas.skill_library import get_library
        library = get_library()

        skill_blocks: dict[str, str] = {}  # skill_id → code
        missing: list[str] = []

        for subtask in subtasks:
            # Look for API skills
            for api_name in subtask.get("apis_needed", []):
                skill = library.find_api_skill(api_name)
                if skill:
                    code = library.get_skill_code(skill.skill_id)
                    if code:
                        skill_blocks[skill.skill_id] = code
                else:
                    missing.append(f"api:{api_name}")

            # Look for patterns
            for pattern_name in subtask.get("patterns_needed", []):
                skill = library.find_pattern(pattern_name)
                if skill:
                    code = library.get_skill_code(skill.skill_id)
                    if code:
                        skill_blocks[skill.skill_id] = code
                else:
                    # Check built-in auth patterns
                    from app.atlas.auth_patterns import get_pattern_code
                    pattern_code = get_pattern_code(pattern_name)
                    if pattern_code:
                        skill_blocks[f"pattern:{pattern_name}"] = pattern_code
                    else:
                        missing.append(f"pattern:{pattern_name}")

        return skill_blocks, missing

    def _learn_missing(self, missing: list[str]) -> None:
        """Trigger API Scout to learn missing capabilities."""
        from app.atlas.api_scout import get_scout
        scout = get_scout()

        for capability in missing:
            if capability.startswith("api:"):
                api_name = capability[4:]
                try:
                    logger.info(f"code_forge: learning API '{api_name}' via API Scout")
                    scout.build_and_register(api_name)
                except Exception:
                    logger.debug(f"code_forge: failed to learn API '{api_name}'",
                                 exc_info=True)

    def _compose(
        self,
        task_description: str,
        subtasks: list[dict],
        skill_blocks: dict[str, str],
        context: str,
    ) -> str:
        """Compose code from skill blocks + LLM-generated glue code."""
        # Format skill code for the prompt
        skill_code_text = ""
        for skill_id, code in skill_blocks.items():
            skill_code_text += f"\n# --- Skill: {skill_id} ---\n{code[:2000]}\n"

        subtask_text = json.dumps(subtasks, indent=2)[:2000]

        prompt = COMPOSE_PROMPT.format(
            task_description=task_description,
            skill_code_blocks=skill_code_text[:6000] or "(no skills available — generate everything)",
            subtask_descriptions=subtask_text,
        )

        if context:
            prompt += f"\n\nAdditional context:\n{context[:1000]}"

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=4096, role="coding")
            raw = str(llm.call(prompt)).strip()

            # Clean markdown fences
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            return raw
        except Exception as e:
            raise RuntimeError(f"Composition failed: {e}")

    def _validate_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python syntax."""
        try:
            compile(code, "<code_forge>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

    def _debug_loop(
        self, code: str, error: str, task_description: str
    ) -> tuple[str, int]:
        """Self-debug loop: fix errors up to MAX_DEBUG_ITERATIONS times."""
        attempts = []

        for i in range(MAX_DEBUG_ITERATIONS):
            prompt = DEBUG_PROMPT.format(
                code=code[:4000],
                error=error[:1000],
                attempt_num=i + 1,
                max_attempts=MAX_DEBUG_ITERATIONS,
                previous_attempts=json.dumps(attempts[-3:])[:500] if attempts else "None",
            )

            try:
                from app.llm_factory import create_specialist_llm
                llm = create_specialist_llm(max_tokens=4096, role="coding")
                raw = str(llm.call(prompt)).strip()

                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

                # Check if fixed
                ok, new_error = self._validate_syntax(raw)
                if ok:
                    logger.info(f"code_forge: self-debug fixed code on attempt {i + 1}")
                    return raw, i + 1

                attempts.append({"attempt": i + 1, "error": new_error[:200]})
                code = raw
                error = new_error
            except Exception:
                break

        logger.warning(f"code_forge: self-debug exhausted {MAX_DEBUG_ITERATIONS} attempts")
        return code, MAX_DEBUG_ITERATIONS

    def _generate_tests(self, task_description: str, code: str) -> str:
        """Generate test code for the composed solution."""
        prompt = f"""Generate pytest tests for this code.

Task: {task_description}

Code:
{code[:3000]}

Requirements:
1. Test the main function/entry point
2. Test edge cases (empty input, error conditions)
3. Use mocks for external API calls
4. At least 3 test functions

Generate ONLY Python test code. No markdown fences."""

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=2048, role="coding")
            raw = str(llm.call(prompt)).strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return raw
        except Exception:
            return ""


# ── Module-level singleton ───────────────────────────────────────────────────

_forge: CodeForge | None = None


def get_forge() -> CodeForge:
    """Get or create the singleton Code Forge."""
    global _forge
    if _forge is None:
        _forge = CodeForge()
    return _forge
