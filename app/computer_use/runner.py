"""
computer_use.runner — vision-driven UI loop orchestrator.

Loop:
    1. Take a screenshot via the backend.
    2. Send the conversation (system prompt + history + screenshot) to
       Claude Haiku 4.5 with the computer_20250124 tool.
    3. For each tool_use block in the response:
         - check budget caps
         - execute via the backend
         - append a tool_result block to the next turn
    4. Repeat until the model returns no tool_use (i.e. it has finished or
       answered the user).
    5. Persist task cost + audit lifecycle event.

Hard caps enforced inside the loop:
    · MAX_STEPS_PER_TASK
    · MAX_USD_PER_TASK
    · monthly cap (read from runtime_settings via budget.get_monthly_cap_usd)

The Anthropic call is wrapped behind an injectable ``client_factory`` so
tests can hand in a fake. The backend is also injectable; the default is
Playwright headless Chromium.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Callable, Optional, Protocol

from app.config import get_anthropic_api_key
from app.computer_use.audit import log_step, log_lifecycle
from app.computer_use.budget import (
    BudgetExceeded, MAX_STEPS_PER_TASK, MAX_USD_PER_TASK,
    check_can_start, check_step_within_budget, estimate_cost_usd,
    record_task_cost,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_VIEWPORT = (1280, 800)
SYSTEM_PROMPT = (
    "You are a vision-driven UI agent helping the operator finish a single "
    "task in the browser. Take a screenshot first, then issue at most ONE "
    "tool call per turn. Stop and reply in plain text the moment the task "
    "is done or you need clarification — do not loop on the same action. "
    "Browser viewport is 1280x800 pixels."
)


class _Backend(Protocol):
    def start(self, *, start_url: str = ...) -> None: ...
    def close(self) -> None: ...
    def screenshot(self) -> bytes: ...
    def perform(self, action: dict[str, Any]) -> str: ...


class _Result:
    """Outcome of a single ``run_task`` invocation."""

    def __init__(
        self,
        *,
        success: bool,
        text: str,
        steps: int,
        cost_usd: float,
        refused_reason: str = "",
    ):
        self.success = success
        self.text = text
        self.steps = steps
        self.cost_usd = cost_usd
        self.refused_reason = refused_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "steps": self.steps,
            "cost_usd": round(self.cost_usd, 4),
            "refused_reason": self.refused_reason,
        }


def run_task(
    task: str,
    *,
    start_url: str = "about:blank",
    model: str = DEFAULT_MODEL,
    backend: Optional[_Backend] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    max_steps: int = MAX_STEPS_PER_TASK,
) -> _Result:
    """Run one vision-CU task end-to-end."""
    if not task or not task.strip():
        return _Result(success=False, text="", steps=0, cost_usd=0.0,
                       refused_reason="empty task")

    log_lifecycle("start", {"task": task[:240], "start_url": start_url})

    # Pre-flight budget check.
    try:
        check_can_start()
    except BudgetExceeded as exc:
        log_lifecycle("refuse", {"reason": str(exc), "scope": exc.scope})
        return _Result(
            success=False, text="", steps=0, cost_usd=0.0,
            refused_reason=str(exc),
        )

    # Resolve backend + client.
    own_backend = backend is None
    if backend is None:
        from app.computer_use.browser_backend import (
            PlaywrightBrowserBackend, BrowserNotAvailable,
        )
        backend = PlaywrightBrowserBackend(viewport=DEFAULT_VIEWPORT)
        try:
            backend.start(start_url=start_url)
        except BrowserNotAvailable as exc:
            log_lifecycle("refuse", {"reason": f"backend unavailable: {exc}"})
            return _Result(
                success=False, text="", steps=0, cost_usd=0.0,
                refused_reason=f"browser backend unavailable: {exc}",
            )

    client = (client_factory or _default_client_factory)()
    if client is None:
        if own_backend:
            backend.close()
        log_lifecycle("refuse", {"reason": "anthropic client unavailable"})
        return _Result(
            success=False, text="", steps=0, cost_usd=0.0,
            refused_reason="anthropic client unavailable",
        )

    # ── Loop ────────────────────────────────────────────────────────────
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    total_cost = 0.0
    steps = 0
    final_text = ""
    refused = ""

    try:
        while steps < max_steps:
            steps += 1
            try:
                check_step_within_budget(total_cost)
            except BudgetExceeded as exc:
                refused = str(exc)
                log_lifecycle("budget_exceeded",
                              {"scope": exc.scope, "spent": exc.spent, "cap": exc.cap})
                break

            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[_computer_use_tool_def(viewport=DEFAULT_VIEWPORT)],
                messages=messages,
                betas=["computer-use-2025-01-24"],
            )

            usage = _normalise_usage(getattr(response, "usage", None))
            cost = estimate_cost_usd(usage)
            total_cost += cost

            blocks = _normalise_content_blocks(getattr(response, "content", []))
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            text_blocks = [b for b in blocks if b.get("type") == "text"]

            # Append assistant turn verbatim — needed for the multi-turn
            # tool-use protocol.
            messages.append({"role": "assistant", "content": blocks})

            if not tool_uses:
                final_text = "\n".join(b.get("text", "") for b in text_blocks).strip()
                log_step(steps, "model_finalize", payload={},
                         result=final_text[:200], cost_usd=cost)
                break

            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                action = dict(tu.get("input") or {})
                try:
                    desc = backend.perform(action)
                except Exception as exc:
                    desc = f"backend_error: {exc}"
                shot = backend.screenshot()
                shot_b64 = base64.b64encode(shot).decode("ascii")
                log_step(
                    steps, action.get("action", "?"),
                    payload={k: v for k, v in action.items() if k != "image"},
                    result=desc, screenshot_kb=len(shot) // 1024,
                    cost_usd=cost,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.get("id"),
                    "content": [
                        {"type": "text", "text": desc},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": shot_b64,
                            },
                        },
                    ],
                })

            messages.append({"role": "user", "content": tool_results})

        success = bool(final_text) and not refused
        return _Result(
            success=success, text=final_text, steps=steps,
            cost_usd=total_cost, refused_reason=refused,
        )
    finally:
        try:
            record_task_cost(
                task_summary=task,
                cost_usd=total_cost,
                steps=steps,
                success=bool(final_text) and not refused,
                refused_reason=refused,
            )
        except Exception:
            logger.debug("computer_use: record_task_cost failed", exc_info=True)
        log_lifecycle("finish", {
            "steps": steps, "cost_usd": round(total_cost, 4),
            "success": bool(final_text) and not refused,
            "refused_reason": refused,
        })
        if own_backend:
            try:
                backend.close()
            except Exception:
                pass


# ── Helpers ────────────────────────────────────────────────────────────────

def _default_client_factory():
    """Construct an Anthropic client. Returns None if unavailable."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return None
    key = get_anthropic_api_key()
    if not key:
        return None
    try:
        return Anthropic(api_key=key)
    except Exception as exc:
        logger.warning(f"computer_use: failed to construct Anthropic client: {exc}")
        return None


def _computer_use_tool_def(viewport: tuple[int, int] = DEFAULT_VIEWPORT) -> dict[str, Any]:
    """Tool definition Anthropic accepts in the ``tools`` array."""
    return {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": viewport[0],
        "display_height_px": viewport[1],
        "display_number": 1,
    }


def _normalise_usage(usage: Any) -> dict[str, int]:
    """Convert SDK usage object/dict to a plain dict."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    out: dict[str, int] = {}
    for key in (
        "input_tokens", "output_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens",
    ):
        v = getattr(usage, key, None)
        if v is not None:
            out[key] = int(v)
    return out


def _normalise_content_blocks(blocks: Any) -> list[dict[str, Any]]:
    """Convert a list of SDK ContentBlock objects to plain dicts."""
    out: list[dict[str, Any]] = []
    for b in blocks or []:
        if isinstance(b, dict):
            out.append(b)
            continue
        kind = getattr(b, "type", None)
        if kind == "text":
            out.append({"type": "text", "text": getattr(b, "text", "")})
        elif kind == "tool_use":
            out.append({
                "type": "tool_use",
                "id": getattr(b, "id", None),
                "name": getattr(b, "name", None),
                "input": getattr(b, "input", {}) or {},
            })
        else:
            # Pass through unknown blocks so the assistant turn is intact.
            try:
                out.append(b.model_dump())
            except Exception:
                continue
    return out
