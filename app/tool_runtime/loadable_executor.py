"""LoadableAgentExecutor — CrewAgentExecutor with mid-loop tool refresh.

CrewAI 1.14.x captures `openai_tools`, `available_functions`, and
`_tool_name_mapping` ONCE at the top of `_invoke_loop_native_tools`,
before the iteration `while True:` loop. That means mutating
`self.original_tools` mid-loop has no effect — the model never sees
the new tools.

This executor overrides that loop to recompute the captures on every
iteration, with cheap reuse when the binder reports clean. When new
tools have just loaded, it ALSO appends a synthetic user-turn message
announcing them, so the model knows what changed without us having
to touch the (cached) system prompt.

The override is a focused copy of crewai's _invoke_loop_native_tools
with three insertions marked `# [DYNA]`. Everything else is upstream
behavior, kept verbatim so future CrewAI updates merge cleanly.

Reference upstream:
  /usr/local/lib/python3.13/site-packages/crewai/agents/crew_agent_executor.py
  Method `_invoke_loop_native_tools` around L449-L555.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentFinish
from crewai.utilities.agent_utils import (
    aget_llm_response,
    convert_tools_to_openai_schema,
    enforce_rpm_limit,
    format_message_for_llm,
    get_llm_response,
    handle_context_length,
    handle_max_iterations_exceeded,
    handle_unknown_error,
    has_reached_max_iterations,
    is_context_length_exceeded,
)
from crewai.utilities.printer import Printer
from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    from crewai.llms.base_llm import BaseLLM

    from app.tool_runtime.binder import ToolBinder

PRINTER = Printer()
logger = logging.getLogger(__name__)


class LoadableAgentExecutor(CrewAgentExecutor):
    """Drop-in CrewAgentExecutor that re-renders openai_tools each
    iteration, driven by a ToolBinder.

    The binder is an *implementation* attribute (PrivateAttr) — set
    it via ``executor.binder = ...`` immediately after construction.
    Pydantic field-typing on a forward-referenced runtime object
    causes class-not-fully-defined errors; PrivateAttr sidesteps it.
    """

    _binder: Any = PrivateAttr(default=None)

    @property
    def binder(self) -> "ToolBinder | None":
        return self._binder

    @binder.setter
    def binder(self, value: "ToolBinder | None") -> None:
        self._binder = value

    def _invoke_loop_native_tools(self) -> AgentFinish:  # noqa: C901
        """Overridden version of CrewAgentExecutor._invoke_loop_native_tools.

        DIFF vs upstream:
          [DYNA-1] openai_tools / available_functions / mapping are
                   recomputed when binder.dirty (initial computation
                   counts as dirty).
          [DYNA-2] Before each LLM call, if there are pending-announce
                   tools, append a synthetic user message naming them
                   so the model knows the toolset just expanded.
          [DYNA-3] After tool dispatch, the iteration falls through
                   to the top of the loop where [DYNA-1] picks up any
                   loads that happened during the dispatch.
        """
        if not self.original_tools:
            return self._invoke_loop_native_no_tools()

        # [DYNA-1] Initial render. After this, dirty=False until a load.
        openai_tools, available_functions, self._tool_name_mapping = (
            convert_tools_to_openai_schema(self._current_tools_for_dispatch())
        )
        if self.binder is not None:
            self.binder.clear_dirty()

        while True:
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        None,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    self._show_logs(formatted_answer)
                    return formatted_answer

                # [DYNA-1] If binder dirty, re-render the schema before LLM call.
                if self.binder is not None and self.binder.dirty:
                    openai_tools, available_functions, self._tool_name_mapping = (
                        convert_tools_to_openai_schema(self._current_tools_for_dispatch())
                    )
                    self.binder.clear_dirty()

                # [DYNA-2] Announce newly-loaded tools to the model.
                if self.binder is not None:
                    pending = self.binder.consume_pending()
                    if pending:
                        announce = self._build_announcement(pending)
                        self.messages.append(format_message_for_llm(announce))

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = get_llm_response(
                    llm=cast("BaseLLM", self.llm),
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=PRINTER,
                    tools=openai_tools,
                    available_functions=None,
                    from_task=self.task,
                    from_agent=self.agent,
                    response_model=self.response_model,
                    executor_context=self,
                    verbose=self.agent.verbose,
                )

                if (
                    isinstance(answer, list)
                    and answer
                    and self._is_tool_call_list(answer)
                ):
                    tool_finish = self._handle_native_tool_calls(
                        answer, available_functions
                    )
                    # [DYNA-3] After dispatch, loop top picks up any new loads.
                    if tool_finish is not None:
                        return tool_finish
                    continue

                if isinstance(answer, str):
                    formatted_answer = AgentFinish(
                        thought="", output=answer, text=answer,
                    )
                    self._invoke_step_callback(formatted_answer)
                    self._append_message(answer)
                    self._show_logs(formatted_answer)
                    return formatted_answer

                if isinstance(answer, BaseModel):
                    output_json = answer.model_dump_json()
                    formatted_answer = AgentFinish(
                        thought="", output=answer, text=output_json,
                    )
                    self._invoke_step_callback(formatted_answer)
                    self._append_message(output_json)
                    self._show_logs(formatted_answer)
                    return formatted_answer

                formatted_answer = AgentFinish(
                    thought="", output=str(answer), text=str(answer),
                )
                self._invoke_step_callback(formatted_answer)
                self._append_message(str(answer))
                self._show_logs(formatted_answer)
                return formatted_answer

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(PRINTER, e, verbose=self.agent.verbose)
                raise e
            finally:
                self.iterations += 1

    async def _ainvoke_loop_native_tools(self) -> AgentFinish:  # noqa: C901
        """Async mirror of `_invoke_loop_native_tools`.

        Same DYNA-1/2/3 contract: re-render schemas when binder is dirty,
        announce loaded tools to the model, and let the iteration top
        re-pick-up loads that happened during dispatch.

        Reference upstream:
          crew_agent_executor.py:1260 (`_ainvoke_loop_native_tools`).
        """
        if not self.original_tools:
            return await self._ainvoke_loop_native_no_tools()

        # [DYNA-1] Initial render.
        openai_tools, available_functions, self._tool_name_mapping = (
            convert_tools_to_openai_schema(self._current_tools_for_dispatch())
        )
        if self.binder is not None:
            self.binder.clear_dirty()

        while True:
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        None,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    self._show_logs(formatted_answer)
                    return formatted_answer

                # [DYNA-1] Re-render on dirty.
                if self.binder is not None and self.binder.dirty:
                    openai_tools, available_functions, self._tool_name_mapping = (
                        convert_tools_to_openai_schema(self._current_tools_for_dispatch())
                    )
                    self.binder.clear_dirty()

                # [DYNA-2] Announce newly-loaded tools.
                if self.binder is not None:
                    pending = self.binder.consume_pending()
                    if pending:
                        announce = self._build_announcement(pending)
                        self.messages.append(format_message_for_llm(announce))

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = await aget_llm_response(
                    llm=cast("BaseLLM", self.llm),
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=PRINTER,
                    tools=openai_tools,
                    available_functions=None,
                    from_task=self.task,
                    from_agent=self.agent,
                    response_model=self.response_model,
                    executor_context=self,
                    verbose=self.agent.verbose,
                )

                if (
                    isinstance(answer, list)
                    and answer
                    and self._is_tool_call_list(answer)
                ):
                    # `_handle_native_tool_calls` is sync even in async mode (upstream).
                    tool_finish = self._handle_native_tool_calls(
                        answer, available_functions
                    )
                    # [DYNA-3] Loop top picks up any new loads.
                    if tool_finish is not None:
                        return tool_finish
                    continue

                if isinstance(answer, str):
                    formatted_answer = AgentFinish(
                        thought="", output=answer, text=answer,
                    )
                    await self._ainvoke_step_callback(formatted_answer)
                    self._append_message(answer)
                    self._show_logs(formatted_answer)
                    return formatted_answer

                if isinstance(answer, BaseModel):
                    output_json = answer.model_dump_json()
                    formatted_answer = AgentFinish(
                        thought="", output=answer, text=output_json,
                    )
                    await self._ainvoke_step_callback(formatted_answer)
                    self._append_message(output_json)
                    self._show_logs(formatted_answer)
                    return formatted_answer

                formatted_answer = AgentFinish(
                    thought="", output=str(answer), text=str(answer),
                )
                await self._ainvoke_step_callback(formatted_answer)
                self._append_message(str(answer))
                self._show_logs(formatted_answer)
                return formatted_answer

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(PRINTER, e, verbose=self.agent.verbose)
                raise e
            finally:
                self.iterations += 1

    # ── helpers ─────────────────────────────────────────────────

    def _current_tools_for_dispatch(self) -> list[Any]:
        """Tools to render this iteration. Prefer binder if attached;
        else fall back to upstream's original_tools list."""
        if self.binder is not None:
            return self.binder.tools
        return list(self.original_tools or [])

    def _build_announcement(self, names: list[str]) -> str:
        """Synthetic user message announcing newly-loaded tools.

        Lands in the turn region of the prompt — system prefix stays
        cacheable; only the (already-uncached) latest turn changes.
        """
        if len(names) == 1:
            return (
                f"[Tool registry] Tool `{names[0]}` is now available to you. "
                "Its schema appears in the tools list. Call it normally."
            )
        listing = ", ".join(f"`{n}`" for n in names)
        return (
            f"[Tool registry] {len(names)} tools just loaded: {listing}. "
            "Their schemas appear in the tools list. Call them normally."
        )
