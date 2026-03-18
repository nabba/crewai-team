from crewai.tools import BaseTool
from pydantic import Field
from app.memory.chromadb_manager import store, retrieve, store_team, retrieve_team


class MemoryStoreTool(BaseTool):
    name: str = "memory_store"
    description: str = (
        "Store information in team memory. "
        "Args: text (str) - the content to store, "
        "metadata (str) - optional comma-separated key=value pairs."
    )
    collection: str = Field(default="default")

    def _run(self, text: str, metadata: str = "") -> str:
        meta = {}
        if metadata:
            for pair in metadata.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    meta[k.strip()] = v.strip()
        store(self.collection, text, meta)
        return f"Stored in memory ({self.collection}): {text[:100]}..."


class MemoryRetrieveTool(BaseTool):
    name: str = "memory_retrieve"
    description: str = (
        "Retrieve relevant information from team memory. "
        "Args: query (str) - search query."
    )
    collection: str = Field(default="default")

    def _run(self, query: str, n_results: int = 5) -> str:
        results = retrieve(self.collection, query, n=n_results)
        if not results:
            return "No relevant memories found."
        return "\n\n---\n\n".join(results)


class TeamMemoryStoreTool(BaseTool):
    name: str = "team_memory_store"
    description: str = (
        "Store information in SHARED team memory accessible by ALL agents and crews. "
        "Use this when findings should be visible to other crews working in parallel. "
        "Args: text (str) - the content to store."
    )

    def _run(self, text: str, metadata: str = "") -> str:
        meta = {}
        if metadata:
            for pair in metadata.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    meta[k.strip()] = v.strip()
        store_team(text, meta)
        return f"Stored in shared team memory: {text[:100]}..."


class TeamMemoryRetrieveTool(BaseTool):
    name: str = "team_memory_retrieve"
    description: str = (
        "Retrieve information from SHARED team memory written by any agent or crew. "
        "Use this to find research or context from other parallel crews. "
        "Args: query (str) - search query."
    )

    def _run(self, query: str, n_results: int = 5) -> str:
        results = retrieve_team(query, n=n_results)
        if not results:
            return "No shared team memories found."
        return "\n\n---\n\n".join(results)


def create_memory_tools(collection: str = "default"):
    """Factory to create memory tools: per-crew pair + shared team pair."""
    return [
        MemoryStoreTool(collection=collection),
        MemoryRetrieveTool(collection=collection),
        TeamMemoryStoreTool(),
        TeamMemoryRetrieveTool(),
    ]
