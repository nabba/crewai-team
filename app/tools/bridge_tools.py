"""
bridge_tools.py — CrewAI tool wrappers for the Host Bridge.

Each tool is scoped to the calling agent's capability token.
The bridge enforces permissions at infrastructure level — these
tools just provide the CrewAI interface.

Usage:
    from app.tools.bridge_tools import create_bridge_tools
    tools = create_bridge_tools("commander")
    # Returns list of CrewAI BaseTool instances

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_bridge_tools(agent_id: str) -> list:
    """Create a set of CrewAI tools bound to an agent's bridge client.

    Returns empty list if bridge is not configured or unavailable.
    """
    try:
        from app.bridge_client import get_bridge
        bridge = get_bridge(agent_id)
        if not bridge:
            return []
        if not bridge.is_available():
            logger.debug(f"bridge_tools: bridge unavailable for {agent_id}")
            return []
    except Exception:
        return []

    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        return []

    # ── Tool definitions ──────────────────────────────────────────

    class _ReadFileInput(BaseModel):
        path: str = Field(description="Absolute path to the file on the host")

    class ReadHostFileTool(BaseTool):
        name: str = "read_host_file"
        description: str = (
            "Read a file from the host computer's filesystem. "
            "Path must be within your allowed directories."
        )
        args_schema: Type[BaseModel] = _ReadFileInput

        def _run(self, path: str) -> str:
            result = bridge.read_file(path)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"
            return result.get("content", "")

    class _WriteFileInput(BaseModel):
        path: str = Field(description="Absolute path where the file should be written")
        content: str = Field(description="Content to write")

    class WriteHostFileTool(BaseTool):
        name: str = "write_host_file"
        description: str = (
            "Write content to a file on the host filesystem. "
            "Path must be within your allowed directories."
        )
        args_schema: Type[BaseModel] = _WriteFileInput

        def _run(self, path: str, content: str) -> str:
            result = bridge.write_file(path, content, create_dirs=True)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"
            return f"Written {result.get('written', 0)} bytes to {result.get('path', path)}"

    class _ListFilesInput(BaseModel):
        path: str = Field(description="Directory path on the host")
        pattern: str = Field(default="*", description="Glob pattern")

    class ListHostFilesTool(BaseTool):
        name: str = "list_host_files"
        description: str = (
            "List files in a directory on the host computer."
        )
        args_schema: Type[BaseModel] = _ListFilesInput

        def _run(self, path: str, pattern: str = "*") -> str:
            result = bridge.list_files(path, pattern, recursive=True)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"
            files = result.get("files", [])[:50]
            return json.dumps(files, indent=2)

    class _HttpInput(BaseModel):
        url: str = Field(description="URL to request")
        method: str = Field(default="GET", description="HTTP method")

    class HttpFromHostTool(BaseTool):
        name: str = "http_from_host"
        description: str = (
            "Make an HTTP request from the host machine's network context. "
            "Useful for accessing LAN services and APIs behind firewalls."
        )
        args_schema: Type[BaseModel] = _HttpInput

        def _run(self, url: str, method: str = "GET") -> str:
            result = bridge.http_request(url, method)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"
            return f"Status: {result.get('status_code')}\n\n{result.get('body', '')[:5000]}"

    class _ExecuteInput(BaseModel):
        command: str = Field(description="Shell command to execute (space-separated)")

    class ExecuteOnHostTool(BaseTool):
        name: str = "execute_on_host"
        description: str = (
            "Execute a shell command on the host macOS system. "
            "The command passes through safety filters and may require human approval."
        )
        args_schema: Type[BaseModel] = _ExecuteInput

        def _run(self, command: str) -> str:
            cmd_parts = command.split()
            result = bridge.execute(cmd_parts)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"
            output = result.get("stdout", "")
            if result.get("stderr"):
                output += f"\n\nSTDERR:\n{result['stderr']}"
            return output

    class _InferenceInput(BaseModel):
        prompt: str = Field(description="Prompt to send to local LLM")
        model: str = Field(default="qwen3:30b-a3b", description="Ollama model name")

    class LocalInferenceTool(BaseTool):
        name: str = "local_inference"
        description: str = (
            "Run inference on the host's local LLM via Ollama (Metal GPU). "
            "Useful for tasks that should not leave the local machine."
        )
        args_schema: Type[BaseModel] = _InferenceInput

        def _run(self, prompt: str, model: str = "qwen3:30b-a3b") -> str:
            result = bridge.inference(prompt, model)
            if "error" in result:
                return f"Error: {result.get('detail', result['error'])}"
            return result.get("response", "")

    # Return all tools
    return [
        ReadHostFileTool(),
        WriteHostFileTool(),
        ListHostFilesTool(),
        HttpFromHostTool(),
        ExecuteOnHostTool(),
        LocalInferenceTool(),
    ]
