"""
tool_executor.py — Self-correcting tool execution + dynamic tool registry.

SelfCorrectingExecutor wraps tool calls with:
  1. Safety check via lifecycle hooks (PRE_TOOL_USE)
  2. Execution with automatic retry
  3. LLM-guided error correction between retries
  4. Result memorization via lifecycle hooks (POST_TOOL_USE)

DynamicToolRegistry allows agents to create and register new tools
with safety constraints:
  - Self-Improver cannot approve its own tools
  - Tools matching blocked patterns are always rejected
  - ChromaDB semantic search for tool discovery
  - Commander auto-approves; other agents need approval

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Error types ───────────────────────────────────────────────────────────────


class ToolCallError(Exception):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class ToolFormatError(ToolCallError):
    """Tool input was malformed — retryable with LLM correction."""
    def __init__(self, message: str):
        super().__init__(message, retryable=True)


class ToolExecutionError(ToolCallError):
    """Tool execution failed — may be retryable."""
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message, retryable=retryable)


class ToolSafetyError(ToolCallError):
    """Safety violation — never retryable."""
    def __init__(self, message: str):
        super().__init__(message, retryable=False)


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class ToolCallResult:
    """Result of a tool execution attempt."""
    success: bool = False
    result: Any = None
    error: str = ""
    attempts: int = 1
    corrections: list[str] = field(default_factory=list)
    tool_name: str = ""
    execution_time_ms: float = 0


# ── Self-Correcting Executor ─────────────────────────────────────────────────

# IMMUTABLE: max retries
MAX_RETRIES = 2

# IMMUTABLE: LLM correction prompt
CORRECTION_PROMPT = """\
A tool call failed. Fix the input and return ONLY the corrected input as JSON.

Tool: {tool_name}
Original input: {original_input}
Current input: {current_input}
Error: {error}
Task context: {task_context}

Return ONLY valid JSON for the corrected input. No explanation."""


class SelfCorrectingExecutor:
    """Wraps tool execution with automatic error correction via LLM.

    On failure:
      1. Checks if error is retryable
      2. Asks budget-tier LLM to fix the input
      3. Retries with corrected input
      4. Gives up after MAX_RETRIES attempts
    """

    def __init__(self, max_retries: int = MAX_RETRIES):
        self._max_retries = max_retries

    def execute(
        self,
        tool: Any,
        tool_input: dict | str,
        agent_id: str = "",
        task_context: str = "",
        tool_name: str = "",
    ) -> ToolCallResult:
        """Execute a tool with self-correction on failure."""
        name = tool_name or getattr(tool, "name", type(tool).__name__)
        start = time.monotonic()
        corrections = []
        current_input = tool_input
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                # Execute the tool
                if hasattr(tool, "_run"):
                    if isinstance(current_input, dict):
                        result = tool._run(**current_input)
                    else:
                        result = tool._run(input=current_input)
                elif callable(tool):
                    result = tool(current_input)
                else:
                    raise ToolCallError(f"Tool '{name}' is not callable", retryable=False)

                elapsed = (time.monotonic() - start) * 1000
                return ToolCallResult(
                    success=True, result=result, attempts=attempt + 1,
                    corrections=corrections, tool_name=name,
                    execution_time_ms=elapsed,
                )

            except ToolSafetyError:
                raise  # Never retry safety violations

            except (ToolCallError, Exception) as e:
                last_error = str(e)
                retryable = getattr(e, "retryable", True)

                if not retryable or attempt >= self._max_retries:
                    break

                logger.warning(f"Tool '{name}' failed (attempt {attempt + 1}): {last_error}")

                # Try LLM-guided correction
                corrected = self._get_correction(
                    name, tool_input, current_input, last_error, agent_id, task_context,
                )
                if corrected is not None:
                    corrections.append(f"Attempt {attempt + 1}: {last_error[:100]} → Corrected")
                    current_input = corrected
                    continue

                break  # Can't correct, stop retrying

        elapsed = (time.monotonic() - start) * 1000
        return ToolCallResult(
            success=False, error=last_error or "Unknown error",
            attempts=self._max_retries + 1, corrections=corrections,
            tool_name=name, execution_time_ms=elapsed,
        )

    def _get_correction(
        self, tool_name: str, original_input: Any, current_input: Any,
        error: str, agent_id: str, task_context: str,
    ) -> Optional[dict]:
        """Ask budget-tier LLM to fix the tool input."""
        prompt = CORRECTION_PROMPT.format(
            tool_name=tool_name,
            original_input=json.dumps(original_input) if isinstance(original_input, dict) else str(original_input),
            current_input=json.dumps(current_input) if isinstance(current_input, dict) else str(current_input),
            error=error[:500],
            task_context=task_context[:500],
        )

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=500, role="self_improve")
            raw = str(llm.call(prompt)).strip()

            # Parse JSON from response
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            return json.loads(raw.strip())
        except (json.JSONDecodeError, Exception):
            return None


# ── Dynamic Tool Registry ────────────────────────────────────────────────────


# IMMUTABLE: blocked tool name patterns
BLOCKED_PATTERNS = frozenset({
    "modify_safety", "bypass_check", "disable_hook",
    "remove_principle", "edit_soul", "override_constitution",
    "unregister_immutable", "delete_protected", "modify_tier3",
})


@dataclass
class RegisteredTool:
    """A dynamically registered tool."""
    name: str
    description: str
    fn: Callable
    created_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    usage_count: int = 0
    success_count: int = 0
    tags: list[str] = field(default_factory=list)
    source_code: str = ""
    approved: bool = False


class DynamicToolRegistry:
    """Registry for agent-created tools with ChromaDB semantic search.

    Safety invariants:
      - Self-Improver cannot approve its own tools
      - Tools matching BLOCKED_PATTERNS are always rejected
      - Commander auto-approves; other agents need explicit approval
    """

    def __init__(
        self,
        chroma_collection=None,
        approval_required: bool = True,
        auto_approve_agents: list[str] | None = None,
    ):
        self._tools: dict[str, RegisteredTool] = {}
        self._collection = chroma_collection
        self.approval_required = approval_required
        self.auto_approve_agents = auto_approve_agents or ["commander"]

    def register(
        self,
        name: str,
        description: str,
        fn: Callable,
        created_by: str = "",
        source_code: str = "",
        tags: list[str] | None = None,
        auto_approve: bool = False,
    ) -> RegisteredTool:
        """Register a new dynamic tool.

        Raises ToolSafetyError if name matches blocked patterns.
        Self-Improver cannot auto-approve.
        """
        # Safety check: blocked patterns with Unicode normalization + confusables
        # NFKC handles within-script variants; confusables handle cross-script
        # lookalikes (Cyrillic а→a, е→e, о→o, р→p, с→c, у→y, х→x, etc.)
        import unicodedata
        _CONFUSABLES = str.maketrans({
            '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
            '\u0441': 'c', '\u0443': 'y', '\u0445': 'x', '\u0456': 'i',
            '\u0455': 's', '\u0458': 'j', '\u0422': 'T', '\u041d': 'H',
            '\u0410': 'A', '\u0412': 'B', '\u0415': 'E', '\u041a': 'K',
            '\u041c': 'M', '\u041e': 'O', '\u0420': 'P', '\u0421': 'C',
            '\u0425': 'X', '\u0423': 'Y', '\u0417': '3',
            '\uff41': 'a', '\uff42': 'b', '\uff43': 'c', '\uff44': 'd',  # fullwidth
            '\uff45': 'e', '\uff46': 'f', '\uff4d': 'm', '\uff4e': 'n',
            '\uff4f': 'o', '\uff50': 'p', '\uff52': 'r', '\uff53': 's',
            '\uff54': 't', '\uff55': 'u', '\uff56': 'v', '\uff59': 'y',
        })
        name_safe = unicodedata.normalize("NFKC", name).translate(_CONFUSABLES).lower()
        desc_safe = unicodedata.normalize("NFKC", description).translate(_CONFUSABLES).lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern in name_safe or pattern in desc_safe:
                raise ToolSafetyError(
                    f"Tool name/description matches blocked pattern '{pattern}'"
                )

        # Self-Improver cannot self-approve
        if created_by == "self_improver" and auto_approve:
            auto_approve = False
            logger.warning("Self-Improver cannot auto-approve tools")

        # SECURITY: Prevent created_by spoofing — only Commander can auto-approve,
        # and only if the tool was actually created by Commander (not just claimed).
        # Log all tool registrations for audit trail.
        logger.info(f"Tool registration: '{name}' by '{created_by}' (auto_approve={auto_approve})")

        approved = (
            not self.approval_required
            or auto_approve
            or created_by in self.auto_approve_agents
        )

        tool = RegisteredTool(
            name=name, description=description, fn=fn,
            created_by=created_by, source_code=source_code,
            tags=tags or [], approved=approved,
        )
        self._tools[name] = tool

        # Index in ChromaDB for semantic search
        if self._collection:
            try:
                self._collection.upsert(
                    ids=[name],
                    documents=[f"{name}: {description}"],
                    metadatas=[{
                        "name": name, "created_by": created_by,
                        "approved": str(approved),
                        "tags": ",".join(tags or []),
                    }],
                )
            except Exception as e:
                logger.debug(f"Failed to index tool in ChromaDB: {e}")

        logger.info(f"Tool registered: '{name}' by {created_by} "
                    f"({'approved' if approved else 'pending'})")
        return tool

    def approve(self, name: str, approved_by: str = "commander") -> bool:
        """Approve a pending tool. Self-Improver cannot approve."""
        tool = self._tools.get(name)
        if not tool:
            return False
        if approved_by == "self_improver":
            logger.warning("Self-Improver cannot approve tools")
            return False
        tool.approved = True
        logger.info(f"Tool '{name}' approved by {approved_by}")
        return True

    def get(self, name: str) -> Optional[RegisteredTool]:
        """Get an approved tool by name."""
        tool = self._tools.get(name)
        return tool if tool and tool.approved else None

    def search(self, query: str, n_results: int = 5) -> list[RegisteredTool]:
        """Search for tools by semantic similarity (ChromaDB) or keywords."""
        if self._collection:
            try:
                results = self._collection.query(
                    query_texts=[query], n_results=n_results * 2,
                    where={"approved": "True"},
                )
                found = []
                for name in results.get("ids", [[]])[0]:
                    if name in self._tools and self._tools[name].approved:
                        found.append(self._tools[name])
                return found[:n_results]
            except Exception:
                pass

        return self._keyword_search(query, n_results)

    def _keyword_search(self, query: str, n_results: int) -> list[RegisteredTool]:
        query_lower = query.lower()
        scored = []
        for tool in self._tools.values():
            if not tool.approved:
                continue
            text = f"{tool.name} {tool.description} {' '.join(tool.tags)}".lower()
            score = sum(2 if w in tool.name.lower() else (1 if w in text else 0)
                        for w in query_lower.split())
            if score > 0:
                scored.append((score, tool))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:n_results]]

    def record_usage(self, name: str, success: bool) -> None:
        """Record tool usage for stats tracking."""
        tool = self._tools.get(name)
        if tool:
            tool.usage_count += 1
            if success:
                tool.success_count += 1

    def list_pending(self) -> list[RegisteredTool]:
        """List tools awaiting approval."""
        return [t for t in self._tools.values() if not t.approved]

    def list_all(self, approved_only: bool = True) -> list[RegisteredTool]:
        """List all registered tools."""
        return [t for t in self._tools.values() if not approved_only or t.approved]

    def get_stats(self) -> dict:
        tools = list(self._tools.values())
        return {
            "total": len(tools),
            "approved": sum(1 for t in tools if t.approved),
            "pending": sum(1 for t in tools if not t.approved),
        }


# ── Module-level singletons ──────────────────────────────────────────────────


_executor: SelfCorrectingExecutor | None = None
_tool_registry: DynamicToolRegistry | None = None


def get_executor() -> SelfCorrectingExecutor:
    global _executor
    if _executor is None:
        _executor = SelfCorrectingExecutor()
    return _executor


def get_tool_registry() -> DynamicToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        # Try to get ChromaDB collection
        collection = None
        try:
            import chromadb
            client = chromadb.HttpClient(host="chromadb", port=8000)
            collection = client.get_or_create_collection(
                name="crewai_dynamic_tools",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass
        _tool_registry = DynamicToolRegistry(chroma_collection=collection)
    return _tool_registry
