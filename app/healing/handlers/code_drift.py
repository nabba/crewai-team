"""Runbooks C + G — code-drift alerts that need human-eye fixes.

Both fire on signatures that point at a real code bug requiring
inspection.

  * **C**: ``NameError: name 'cost_mode' is not defined`` —
    uninitialised local variable in self-improvement / evolution
    crews. Surfaces to Signal (operator decides where to insert the
    init); auto-CR would be wrong without source-level analysis of
    intent.
  * **G**: ``Anthropic API call failed: 'str' object has no
    attribute 'content'`` — response-shape mismatch when the client
    returned a bare string. *CR-gated*: files a change-request with
    a defensive ``coerce_response()`` helper module the operator
    can wire into the call path after approval.

Both handlers dedupe with a 24 h cooldown so a sustained spike doesn't
spam Signal.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    compute_signature,
    file_change_request,
    read_state_json,
    sample_contains,
    send_signal_alert,
    write_state_json,
)
from app.healing.runbooks import RunbookResult, register_runbook

logger = logging.getLogger(__name__)

_STATE_FILE = "code_drift_alerts.json"
_ALERT_COOLDOWN_S = 24 * 3600


def _alert_with_cooldown(*, key: str, runbook: str, body: str, anomaly: dict) -> RunbookResult:
    state = read_state_json(_STATE_FILE, {"alerts": {}})
    entry = state.setdefault("alerts", {}).setdefault(key, {"last_alert_at": 0, "count": 0})
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["last_seen"] = time.time()

    audit_event(
        f"code_drift_{key}",
        count=entry["count"],
        pattern_signature=anomaly.get("pattern_signature"),
        runbook=runbook,
    )

    cool = (time.time() - entry.get("last_alert_at", 0)) >= _ALERT_COOLDOWN_S
    if cool:
        entry["last_alert_at"] = time.time()
        write_state_json(_STATE_FILE, state)
        send_signal_alert(body, tag=runbook)
        return RunbookResult(
            name=runbook, success=True,
            detail=f"alerted (count={entry['count']})",
        )

    write_state_json(_STATE_FILE, state)
    return RunbookResult(
        name=runbook, success=True,
        detail=f"tracked (count={entry['count']}, alert cooled down)",
    )


# ── Runbook C — cost_mode NameError ───────────────────────────────────────

# The NameError appears under multiple loggers depending on which crew hits
# it. We register one signature per known logger; a sample-substring guard
# defends inside the handler.
_C_LOGGERS_AND_MESSAGES = [
    (
        "app.crews.self_improvement_crew",
        "Improvement scan failed: name 'cost_mode' is not defined",
    ),
    (
        "app.evolution",
        "Evolution session failed: name 'cost_mode' is not defined",
    ),
]


def _handle_cost_mode(anomaly: dict[str, Any]) -> RunbookResult:
    if not sample_contains(anomaly, "cost_mode", "is not defined"):
        return RunbookResult(
            name="cost_mode_undefined_alert", success=False,
            detail="sample mismatch", error="sample_mismatch",
        )
    return _alert_with_cooldown(
        key="cost_mode_undefined",
        runbook="cost_mode_undefined_alert",
        body=(
            "🐛 Self-heal: `NameError: cost_mode is not defined` is "
            "recurring (self_improvement / evolution crews). The variable "
            "is referenced before any assignment. One-line fix: "
            "`cost_mode = get_settings().cost_mode` at the top of the "
            "affected scope. See `workspace/self_heal/code_drift_alerts.json` "
            "for occurrences."
        ),
        anomaly=anomaly,
    )


# ── Runbook G — Anthropic 'str'.content (CR-gated) ────────────────────────

_G_LOGGER = "root"
_G_MESSAGE = "Anthropic API call failed: 'str' object has no attribute 'content'"
_G_SIGNATURE = compute_signature(_G_LOGGER, _G_MESSAGE)


_G_GUARD_MODULE_TEMPLATE = '''"""Defensive Anthropic response coercion.

Auto-proposed by self-heal runbook G after a recurring
``'str' object has no attribute 'content'`` error. The Anthropic /
LiteLLM client occasionally returns a bare string in place of a
``Message`` object — most likely upstream schema drift. This module
provides a coercion helper that operators can wire into the call
path so the existing parsers keep working.

Wiring (operator action after approving this CR):

    from app.llms.anthropic_response_guard import coerce_response

    response = client.messages.create(...)
    response = coerce_response(response)   # <-- add this line
    text = response.content[0].text        # original parser unchanged

The guard is intentionally minimal — surfaces a structured warning so
genuine upstream drift is visible in logs, but never crashes the call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _SyntheticContent:
    """Minimal stand-in for an Anthropic content block."""
    text: str
    type: str = "text"


@dataclass
class _SyntheticMessage:
    """Minimal stand-in for an Anthropic Message — exposes ``.content`` as
    a list of blocks (matching the SDK shape) so existing parsers work.
    """
    content: list

    @property
    def role(self) -> str:
        return "assistant"

    @property
    def stop_reason(self) -> str:
        return "synthetic_string_coercion"


def coerce_response(response):
    """Return ``response`` unchanged unless it's a bare string.

    Bare-string drift is logged at WARNING so the issue stays visible.
    The synthetic Message wrapper preserves the existing parser path.
    """
    if isinstance(response, str):
        logger.warning(
            "anthropic_response_guard: bare-string response coerced to "
            "Message (schema drift?) — len=%d preview=%r",
            len(response), response[:80],
        )
        return _SyntheticMessage(content=[_SyntheticContent(text=response)])
    return response
'''


def _handle_anthropic_str_content(anomaly: dict[str, Any]) -> RunbookResult:
    if not sample_contains(anomaly, "'str' object has no attribute 'content'"):
        return RunbookResult(
            name="anthropic_str_content_cr", success=False,
            detail="sample mismatch", error="sample_mismatch",
        )

    # Track + dedup by signature (single canonical message — one CR ever).
    state = read_state_json(_STATE_FILE, {"alerts": {}})
    entry = state.setdefault("alerts", {}).setdefault(
        "anthropic_str_content",
        {"count": 0, "cr_id": None, "last_alert_at": 0},
    )
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["last_seen"] = time.time()

    audit_event(
        "code_drift_anthropic_str_content",
        count=entry["count"],
        cr_id=entry.get("cr_id"),
        pattern_signature=anomaly.get("pattern_signature"),
    )

    # Already filed a CR? Re-alert on cooldown but don't refile.
    if entry.get("cr_id"):
        cool = (time.time() - entry.get("last_alert_at", 0)) >= _ALERT_COOLDOWN_S
        if cool:
            entry["last_alert_at"] = time.time()
            write_state_json(_STATE_FILE, state)
            send_signal_alert(
                f"🐛 Self-heal: Anthropic `'str'`-content drift still "
                f"firing ({entry['count']} total). CR `{entry['cr_id']}` "
                f"is open in /cp/changes — approve to land the response "
                f"guard module.",
                tag="anthropic_str_content_cr",
            )
        else:
            write_state_json(_STATE_FILE, state)
        return RunbookResult(
            name="anthropic_str_content_cr", success=True,
            detail=f"already proposed cr_id={entry['cr_id']}",
        )

    # First time: file a CR with the guard module.
    cr_id = file_change_request(
        path="app/llms/anthropic_response_guard.py",
        new_content=_G_GUARD_MODULE_TEMPLATE,
        old_content="",
        reason=(
            "Self-heal: Anthropic client returned a bare string in place "
            f"of a Message object ({entry['count']} occurrences). Proposed: "
            "ship a defensive `coerce_response()` helper. Operator must "
            "approve and then wire it into the call path."
        ),
        requestor="self_heal_handler",
    )

    if cr_id:
        entry["cr_id"] = cr_id
        entry["last_alert_at"] = time.time()
        write_state_json(_STATE_FILE, state)
        return RunbookResult(
            name="anthropic_str_content_cr", success=True,
            detail=f"filed CR {cr_id} for response guard",
        )

    # CR system unavailable → degrade to alert.
    entry["last_alert_at"] = time.time()
    write_state_json(_STATE_FILE, state)
    send_signal_alert(
        f"⚠️  Self-heal: Anthropic `'str'`-content drift "
        f"({entry['count']} occurrences) — couldn't auto-file CR. "
        f"Add a defensive `isinstance(resp, str)` branch in the "
        f"response parser. Trail in "
        f"`workspace/self_heal/code_drift_alerts.json`.",
        tag="anthropic_str_content_cr",
    )
    return RunbookResult(
        name="anthropic_str_content_cr", success=False,
        detail="CR filing failed; alerted instead", error="cr_failed",
    )


def register() -> None:
    """Register C (one signature per known logger) and G (single sig)."""
    for i, (logger_name, message) in enumerate(_C_LOGGERS_AND_MESSAGES):
        sig = compute_signature(logger_name, message)
        # Distinct names so dispatcher per-runbook stats don't blend.
        name = f"cost_mode_undefined_alert_v{i}" if i > 0 else "cost_mode_undefined_alert"
        register_runbook(name, sig, _handle_cost_mode)
    register_runbook(
        "anthropic_str_content_cr", _G_SIGNATURE, _handle_anthropic_str_content,
    )
