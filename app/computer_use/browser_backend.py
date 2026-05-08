"""
computer_use.browser_backend — Playwright implementation of the agent's
"computer". The model issues mouse/keyboard actions; this backend executes
them inside a headless Chromium and returns a screenshot.

Why a browser instead of a desktop VM:
  - works inside the existing Docker container (Playwright already a dep)
  - covers the realistic 80% of personal-agent CU tasks (web flows + SPAs)
  - keeps Phase 6 shippable without new infrastructure
A swap-in backend with a real X11/KasmVNC desktop is the next iteration.

Backend contract — anything implementing these four methods works:

    open(url) -> None
    screenshot() -> bytes (PNG)
    perform(action: dict) -> str
    close() -> None

`action` matches Anthropic's computer-use tool schema:
    {action: "screenshot"} | {action: "left_click", coordinate: [x, y]}
    | {action: "double_click", coordinate: [x, y]}
    | {action: "type", text: "..."} | {action: "key", text: "Return"}
    | {action: "scroll", coordinate: [x, y], scroll_direction: "down",
       scroll_amount: 3}
    | {action: "mouse_move", coordinate: [x, y]}

Returned strings are short result descriptions ("clicked", "scrolled 3").
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_VIEWPORT = (1280, 800)


class BrowserNotAvailable(RuntimeError):
    """Raised when Playwright isn't installed or Chromium failed to launch."""


class PlaywrightBrowserBackend:
    """Headless Chromium driven by Playwright.

    One instance per task. The runner constructs it, runs the loop, and
    calls ``close()`` regardless of outcome.
    """

    def __init__(self, viewport: tuple[int, int] = DEFAULT_VIEWPORT) -> None:
        self.viewport = viewport
        self._pw = None
        self._browser = None
        self._page = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, *, start_url: str = "about:blank") -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserNotAvailable(
                "playwright not installed (pip install playwright "
                "&& playwright install chromium)"
            ) from exc

        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True)
            context = self._browser.new_context(
                viewport={"width": self.viewport[0], "height": self.viewport[1]},
                ignore_https_errors=False,
            )
            self._page = context.new_page()
            if start_url:
                self._page.goto(start_url, timeout=15_000)
        except Exception as exc:
            self.close()
            raise BrowserNotAvailable(f"Chromium launch failed: {exc}") from exc

    def close(self) -> None:
        try:
            if self._page is not None:
                self._page.close()
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._pw = None

    # ── Operations ────────────────────────────────────────────────────────

    def screenshot(self) -> bytes:
        if self._page is None:
            raise BrowserNotAvailable("backend not started")
        return self._page.screenshot(type="png", full_page=False)

    def perform(self, action: dict[str, Any]) -> str:
        if self._page is None:
            raise BrowserNotAvailable("backend not started")
        kind = action.get("action") or action.get("type") or ""
        kind = kind.lower()

        if kind == "screenshot":
            return "screenshot"
        if kind in ("left_click", "click"):
            x, y = _coords(action)
            self._page.mouse.click(x, y)
            return f"left_click({x},{y})"
        if kind == "right_click":
            x, y = _coords(action)
            self._page.mouse.click(x, y, button="right")
            return f"right_click({x},{y})"
        if kind == "double_click":
            x, y = _coords(action)
            self._page.mouse.dblclick(x, y)
            return f"double_click({x},{y})"
        if kind == "mouse_move":
            x, y = _coords(action)
            self._page.mouse.move(x, y)
            return f"mouse_move({x},{y})"
        if kind == "type":
            text = action.get("text", "")
            self._page.keyboard.type(text, delay=15)
            return f"type({len(text)} chars)"
        if kind == "key":
            text = action.get("text", "") or action.get("key", "")
            # Anthropic's tool sends keys like "Return" / "ctrl+a" — map a
            # couple of common ones to Playwright's expected names.
            mapped = _map_key_combo(text)
            self._page.keyboard.press(mapped)
            return f"key({text!r})"
        if kind == "scroll":
            x, y = _coords(action)
            direction = (action.get("scroll_direction") or "down").lower()
            amount = int(action.get("scroll_amount", 3))
            dy = 100 * amount * (1 if direction == "down" else -1) if direction in ("down", "up") else 0
            dx = 100 * amount * (1 if direction == "right" else -1) if direction in ("left", "right") else 0
            self._page.mouse.move(x, y)
            self._page.mouse.wheel(dx, dy)
            return f"scroll({direction}, {amount})"
        if kind == "wait":
            ms = int(action.get("duration_ms", 500))
            self._page.wait_for_timeout(ms)
            return f"wait({ms}ms)"
        if kind == "goto":
            url = action.get("url", "")
            if url:
                self._page.goto(url, timeout=15_000)
                return f"goto({url})"
            return "goto(missing url)"
        return f"unknown action {kind!r}"


def _coords(action: dict[str, Any]) -> tuple[int, int]:
    coord = action.get("coordinate") or [0, 0]
    if isinstance(coord, dict):
        return int(coord.get("x", 0)), int(coord.get("y", 0))
    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
        return int(coord[0]), int(coord[1])
    return (int(action.get("x", 0)), int(action.get("y", 0)))


def _map_key_combo(s: str) -> str:
    """Translate model-emitted key names to Playwright's vocabulary."""
    if not s:
        return ""
    parts = s.split("+")
    aliases = {
        "ctrl": "Control", "control": "Control", "cmd": "Meta",
        "super": "Meta", "shift": "Shift", "alt": "Alt",
        "return": "Enter", "esc": "Escape",
    }
    mapped = ["+".join("Plus" if p == "+" else p for p in [aliases.get(p.lower(), p)]) for p in parts]
    return "+".join(mapped)
