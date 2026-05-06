"""request_restricted_write — agent-callable change-request tool.

Phase 5.3a tool. Lets a coding agent (or any agent that has this
tool in its inventory) request a write to a restricted path
(``app/...``, ``tests/...``, etc.) by going through the Signal +
React human-gated change-request system.

Pattern of use::

    # 1. Read the current file (caller's responsibility — typically
    #    via read_host_file from the bridge)
    current = read_host_file(path="app/agents/pim_agent.py")

    # 2. Compute the new content (typically via an LLM rewrite)
    new = current.replace(
        "from crewai import Agent\n",
        "from crewai import Agent\n"
        "from app.agents._common import optional_tool_group\n",
    )

    # 3. Submit
    request_restricted_write(
        path="app/agents/pim_agent.py",
        new_content=new,
        old_content=current,
        reason=(
            "PIM crew fails with NameError: optional_tool_group "
            "is not defined. Adding the missing import."
        ),
    )

    # → Tool returns:
    #   "Change request <id> created (PENDING). Signal ASK sent
    #    to operator. Status will be PENDING until 👍/👎 in Signal
    #    or React. Polling not yet supported in this tool — expect
    #    next-step user message to confirm or reject."

The tool returns SYNCHRONOUSLY after submission. It does NOT block
waiting for the user's reaction — that happens out-of-band via
Signal reactions or the React control plane. Tool output tells the
agent "request submitted, await user decision in Signal."

If the user wants the agent to wait for the decision before
continuing, the agent's task description should explicitly say so;
this tool returns immediately by design (long-blocking calls
deadlock the agent's ReAct loop).

TIER_IMMUTABLE behavior
-----------------------
If the path is in ``TIER_IMMUTABLE`` (see ``app/auto_deployer.py``),
the tool refuses at request time WITHOUT sending a Signal message.
The agent gets back: "TIER_IMMUTABLE — cannot be modified by agent
path; operator must edit directly." This is the absolute rule —
even React-side operator override cannot bypass it. The user
agreed to this constraint when authorizing the change-request
workflow.
"""
from __future__ import annotations

import logging
from typing import Type

logger = logging.getLogger(__name__)


def _build_tool_class():
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _RequestRestrictedWriteInput(BaseModel):
        path: str = Field(
            description=(
                "Repo-relative path of the file to modify. e.g. "
                "'app/agents/pim_agent.py'. Must be under one of the "
                "allowed roots (app/, tests/, docs/, dashboard-react/, "
                "deploy/, scripts/, host_bridge/). TIER_IMMUTABLE "
                "files are refused regardless of human approval."
            ),
        )
        new_content: str = Field(
            description=(
                "The COMPLETE new file contents. Not a diff — the full "
                "file as it should look after the change. Max 1 MB."
            ),
        )
        old_content: str = Field(
            description=(
                "The CURRENT file contents (read via read_host_file or "
                "equivalent before computing new_content). Used to "
                "compute the diff for operator review and for rollback. "
                "Empty string is acceptable when the target file does "
                "not yet exist."
            ),
            default="",
        )
        reason: str = Field(
            description=(
                "One-paragraph explanation for the operator. What bug "
                "or improvement does this change address? Why is the "
                "specific edit correct? This appears verbatim in the "
                "Signal ASK message and the auto-PR description."
            ),
        )

    class RequestRestrictedWriteTool(BaseTool):
        name: str = "request_restricted_write"
        description: str = (
            "Request a write to a restricted path (e.g. app/agents/, "
            "tests/, docs/) via the human-gated change-request system. "
            "USE THIS to fix bugs in production code that the file_manager "
            "tool cannot reach (file_manager only writes to output/, "
            "skills/, proposals/).\n\n"
            "Flow: (1) read the current file (read_host_file), (2) "
            "compute the new content (full file, not a diff), (3) call "
            "this tool. The system sends the diff to the user via Signal "
            "with a 👍/👎 prompt. On 👍 the file is hot-applied + an "
            "auto-PR is opened against main; on 👎 the request is "
            "rejected. Operator can also approve/reject via the React "
            "control plane.\n\n"
            "TIER_IMMUTABLE files (security core, eval infrastructure, "
            "souls, governance, forge, epistemic) are REFUSED at request "
            "time — no human approval can override. Operator must edit "
            "directly via PR.\n\n"
            "This tool returns immediately after submission. It does NOT "
            "wait for the user's reaction — expect a follow-up user "
            "message with the decision. If you need to confirm the "
            "decision before proceeding, ask the user explicitly in "
            "your response."
        )
        args_schema: Type[BaseModel] = _RequestRestrictedWriteInput

        def _run(
            self,
            path: str,
            new_content: str,
            old_content: str = "",
            reason: str = "",
        ) -> str:
            from app.change_requests import (
                Status, create_request, send_ask,
            )

            # Resolve the requestor agent_id from context if available.
            # For now we use a generic "agent" — Phase 5.3b can wire in
            # the actual caller's agent_id from the BaseTool invocation
            # context.
            requestor = "agent"

            try:
                cr = create_request(
                    requestor=requestor,
                    path=path,
                    new_content=new_content,
                    old_content=old_content,
                    reason=reason or "(no reason provided)",
                )
            except Exception as exc:  # noqa: BLE001
                return f"request_restricted_write ERROR: {type(exc).__name__}: {exc}"

            if cr.status == Status.TIER_IMMUTABLE_REFUSED:
                return (
                    f"REFUSED: path {path!r} is in TIER_IMMUTABLE — no "
                    f"agent path can modify it, regardless of human "
                    f"approval. The operator must edit this file "
                    f"directly via a manual PR. Reason from validator: "
                    f"{cr.decision_reason}"
                )

            if cr.status == Status.REJECTED:
                # Validation failure (not TIER_IMMUTABLE)
                return (
                    f"REJECTED at validation: {cr.decision_reason}\n\n"
                    f"Common causes: path outside allowed roots; path "
                    f"traversal; sensitive file pattern; content too "
                    f"large (>1 MB)."
                )

            # PENDING — send the Signal ASK
            try:
                ts = send_ask(cr.id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "request_restricted_write: send_ask raised: %s", exc,
                )
                ts = None

            if ts is None:
                return (
                    f"Change request {cr.id} created (PENDING) but Signal "
                    f"ASK could NOT be sent (Signal owner not configured "
                    f"or signal-cli unreachable). The request is visible "
                    f"in the React control plane at "
                    f"/api/cp/changes/{cr.id} — operator can approve or "
                    f"reject there."
                )

            return (
                f"Change request {cr.id} created (PENDING). Signal ASK "
                f"sent to operator (msg ts={ts}). The operator will react "
                f"with 👍 to approve or 👎 to reject. The request is also "
                f"visible at /api/cp/changes/{cr.id} for React-side "
                f"override.\n\n"
                f"This tool returns immediately. Expect a subsequent user "
                f"message confirming or rejecting the change. If your "
                f"current task depends on the change being applied, ask "
                f"the user to react in Signal first."
            )

    return RequestRestrictedWriteTool


try:
    RequestRestrictedWriteTool = _build_tool_class()
except Exception as exc:
    logger.debug("restricted_write_tool: deferred class build (%s)", exc)
    RequestRestrictedWriteTool = None  # type: ignore[assignment]


def create_restricted_write_tools(agent_id: str = "default") -> list:
    """Factory for explicit injection (for agents not using the
    registry). Returns a 1-element list."""
    global RequestRestrictedWriteTool
    if RequestRestrictedWriteTool is None:
        try:
            RequestRestrictedWriteTool = _build_tool_class()
        except Exception:
            return []
    return [RequestRestrictedWriteTool()]


# ── Tool registry annotation ────────────────────────────────────────


try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="request_restricted_write",
        capabilities=["registers-tool"],  # closest existing tag —
                                          # Phase 5.4 may add a dedicated
                                          # 'requests-code-change' tag
        description=(
            "Request a write to a restricted path (e.g. app/agents/, "
            "tests/, docs/) via the human-gated change-request system. "
            "USE THIS to fix bugs in production code beyond file_manager's "
            "reach. Flow: read current file → compute new content → call "
            "this tool → user 👍/👎 in Signal → on 👍 file is hot-applied "
            "+ auto-PR opened. TIER_IMMUTABLE files refused at request "
            "time. Returns immediately; reaction comes out-of-band."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _restricted_write_registry_factory():
        tools = create_restricted_write_tools()
        if not tools:
            raise RuntimeError("restricted_write_tool: factory returned empty list")
        return tools[0]
except ImportError:
    pass
