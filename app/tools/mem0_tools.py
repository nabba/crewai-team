"""
Mem0-backed tools for CrewAI agents — persistent cross-session memory.

These tools give agents the ability to:
  - Store important facts that persist across sessions
  - Recall facts from previous sessions
  - Store conversations for automatic fact extraction
  - Search entity relationships via graph memory

Tools return empty/graceful messages if Mem0 is disabled or unavailable.
"""
from crewai.tools import BaseTool
from pydantic import Field
from app.memory.mem0_manager import (
    store_memory, search_memory, store_conversation, search_shared,
)


_MAX_TOOL_INPUT = 10_000  # bytes — matches mem0_manager limit


class Mem0StoreTool(BaseTool):
    name: str = "persist_fact"
    description: str = (
        "Store an important fact or finding in PERSISTENT memory that survives across sessions. "
        "Use this for key discoveries, user preferences, or important data you want to remember "
        "next time. Args: text (str) - the fact to remember."
    )
    agent_id: str = Field(default="")

    def _run(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return "Error: text is required."
        if len(text.encode('utf-8', errors='replace')) > _MAX_TOOL_INPUT:
            return f"Error: fact too large (max {_MAX_TOOL_INPUT} bytes)."
        result = store_memory(text, agent_id=self.agent_id or None)
        if result is None:
            return "Persistent memory unavailable — fact noted in session only."
        return f"Fact persisted: {text[:100]}..."


class Mem0SearchTool(BaseTool):
    name: str = "recall_facts"
    description: str = (
        "Search persistent memory for facts from previous sessions. "
        "Use this BEFORE starting research to check if you already know the answer "
        "from a previous session. Args: query (str) - what to search for, "
        "scope (str) - 'private' (your own facts), 'shared' (all agents), or 'all' (both)."
    )
    agent_id: str = Field(default="")

    def _run(self, query: str, scope: str = "all") -> str:
        if not query or not isinstance(query, str):
            return "Error: query is required."
        if scope not in ("private", "shared", "all"):
            scope = "all"
        results = []
        if scope in ("shared", "all"):
            results.extend(search_shared(query, n=5))
        if scope in ("private", "all") and self.agent_id:
            results.extend(search_memory(query, agent_id=self.agent_id, n=5))

        if not results:
            return "No relevant facts found in persistent memory."

        # Deduplicate and format
        seen = set()
        lines = []
        for r in results:
            fact = r.get("memory", "") if isinstance(r, dict) else str(r)
            if fact and fact not in seen:
                seen.add(fact)
                lines.append(f"- {fact}")
        return "\n".join(lines) if lines else "No relevant facts found."


class Mem0ConversationStoreTool(BaseTool):
    name: str = "persist_conversation"
    description: str = (
        "Store a conversation exchange for automatic fact extraction. "
        "Mem0 will use an LLM to extract key facts from the conversation "
        "and store them persistently. Use after completing a research session "
        "to capture what you learned. "
        "Args: user_message (str), assistant_response (str)."
    )
    agent_id: str = Field(default="")

    def _run(self, user_message: str, assistant_response: str) -> str:
        if not user_message or not assistant_response:
            return "Error: both user_message and assistant_response are required."
        # Truncate to prevent oversized extractions
        messages = [
            {"role": "user", "content": user_message[:5000]},
            {"role": "assistant", "content": assistant_response[:5000]},
        ]
        result = store_conversation(messages, agent_id=self.agent_id or None)
        if result is None:
            return "Persistent memory unavailable — conversation not extracted."
        return "Conversation facts extracted and persisted."


def create_mem0_tools(agent_name: str) -> list:
    """Factory to create Mem0 tools configured for a specific agent.

    Returns empty list if mem0 is disabled in settings.
    """
    from app.config import get_settings
    if not get_settings().mem0_enabled:
        return []
    return [
        Mem0StoreTool(agent_id=agent_name),
        Mem0SearchTool(agent_id=agent_name),
        Mem0ConversationStoreTool(agent_id=agent_name),
    ]
