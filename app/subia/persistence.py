"""
subia.persistence — kernel serialization to/from wiki pages.

Round-trips the `SubjectivityKernel` dataclass through YAML-frontmatter
markdown as specified in SubIA Part II §19. Two artefacts:

  kernel-state.md   full kernel snapshot (YAML frontmatter + prose body)
  hot.md            compressed session-continuity buffer (<= 500 tokens)

The kernel dataclass itself stays pure data. This module contains all
the presentation concerns: markdown formatting, YAML frontmatter,
timestamp handling, size caps. Callers use:

    from app.subia.persistence import (
        serialize_kernel_to_markdown,
        load_kernel_from_markdown,
        generate_hot_md,
        apply_hot_md,
        save_kernel_state,
        load_kernel_state,
    )

save/load_kernel_state are the common case — they read/write from
paths.KERNEL_STATE and paths.HOT_MD atomically via safe_io.

Infrastructure-level. Not agent-modifiable. See PROGRAM.md Phase 4.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.paths import HOT_MD, KERNEL_STATE
from app.safe_io import safe_write
from app.subia.config import SUBIA_CONFIG
from app.subia.kernel import (
    Commitment,
    ConsolidationBuffer,
    HomeostaticState,
    MetaMonitorState,
    Prediction,
    SceneItem,
    SelfState,
    SocialModelEntry,
    SubjectivityKernel,
)

logger = logging.getLogger(__name__)


# Delimiter pattern for YAML frontmatter.
_FRONTMATTER_RE = re.compile(
    r"^---\n(?P<yaml>.*?)\n---\n(?P<body>.*)$", re.DOTALL,
)


# ── Top-level entry points ─────────────────────────────────────────

def save_kernel_state(kernel: SubjectivityKernel) -> None:
    """Persist the kernel to paths.KERNEL_STATE and paths.HOT_MD."""
    content = serialize_kernel_to_markdown(kernel)
    safe_write(KERNEL_STATE, content)
    safe_write(HOT_MD, generate_hot_md(kernel))


def load_kernel_state(path: Path | str | None = None) -> SubjectivityKernel:
    """Load a kernel from paths.KERNEL_STATE. Missing/corrupt file
    returns a fresh default kernel — never raises.
    """
    target = Path(path) if path else KERNEL_STATE
    if not target.exists():
        logger.debug("subia.persistence: no kernel-state at %s", target)
        return SubjectivityKernel()
    try:
        content = target.read_text(encoding="utf-8")
    except OSError:
        logger.exception("subia.persistence: failed to read %s", target)
        return SubjectivityKernel()
    return load_kernel_from_markdown(content)


# ── Serialization ──────────────────────────────────────────────────

def serialize_kernel_to_markdown(kernel: SubjectivityKernel) -> str:
    """Serialize kernel to YAML-frontmatter markdown.

    Format matches SubIA Part II §19.1. The frontmatter contains the
    complete dataclass payload as a compact JSON-in-YAML block so the
    round-trip is lossless. The body is human-readable prose
    summarizing what Andrus would want to see when browsing in Obsidian.
    """
    fm = _frontmatter_dict(kernel)
    return (
        _yaml_frontmatter(fm)
        + "\n"
        + _markdown_body(kernel)
        + "\n"
    )


def load_kernel_from_markdown(content: str) -> SubjectivityKernel:
    """Reverse of serialize_kernel_to_markdown.

    Only the frontmatter is parsed — the prose body is re-derivable.
    Malformed frontmatter falls back to a default kernel rather than
    raising, because a broken wiki page must not crash startup.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        logger.debug("subia.persistence: no frontmatter in kernel page")
        return SubjectivityKernel()
    yaml_text = match.group("yaml")
    payload = _parse_frontmatter(yaml_text)
    if not isinstance(payload, dict):
        return SubjectivityKernel()
    return _kernel_from_frontmatter(payload)


def generate_hot_md(kernel: SubjectivityKernel) -> str:
    """Compact session-continuity buffer — <=500 tokens target.

    Per Amendment C.2 this is the cheapest possible handoff between
    sessions. Records: last focal scene (max 3), unresolved findings,
    top homeostatic pressures, and a resume hint.
    """
    lines = [
        "---",
        f"title: \"SubIA session continuity buffer\"",
        f"updated_at: \"{datetime.now(timezone.utc).isoformat()}\"",
        f"loop_count: {kernel.loop_count}",
        f"session_id: \"{kernel.session_id}\"",
        "---",
        "",
        "# hot.md — session continuity",
        "",
        "## Last focal scene",
    ]
    focal = kernel.focal_scene()[:3]
    if focal:
        for item in focal:
            affect = getattr(item, "dominant_affect", "neutral")
            tag = f" ({affect})" if affect != "neutral" else ""
            lines.append(
                f"- {getattr(item, 'summary', '')[:80]}{tag}"
            )
    else:
        lines.append("- (scene empty)")

    # Unresolved
    commitments = [c for c in kernel.self_state.active_commitments
                   if getattr(c, "status", "active") == "active"]
    unknowns = kernel.meta_monitor.known_unknowns[:5]
    lines.extend(["", "## Unresolved"])
    if commitments:
        for c in commitments[:5]:
            deadline = getattr(c, "deadline", None)
            tag = f" (deadline: {deadline})" if deadline else ""
            lines.append(f"- commitment: {c.description[:80]}{tag}")
    for u in unknowns:
        lines.append(f"- known-unknown: {str(u)[:100]}")
    if not commitments and not unknowns:
        lines.append("- (none)")

    # Homeostatic pressures
    dev = {
        v: d for v, d in kernel.homeostasis.deviations.items()
        if abs(d) > SUBIA_CONFIG["HOMEOSTATIC_DEVIATION_THRESHOLD"]
    }
    lines.extend(["", "## Homeostatic pressures"])
    if dev:
        for var in sorted(dev, key=lambda v: abs(dev[v]), reverse=True)[:5]:
            lines.append(f"- {var}: {dev[var]:+.2f}")
    else:
        lines.append("- (all within equilibrium)")

    # Resume hint (cheap heuristic)
    lines.extend(["", "## Resume hint"])
    lines.append(_derive_resume_hint(kernel))
    return "\n".join(lines) + "\n"


def apply_hot_md(kernel: SubjectivityKernel, content: str) -> None:
    """Apply a session-continuity overlay to an existing kernel.

    hot.md is a presentation artefact, not a full state replacement.
    We read the loop_count from its frontmatter so the kernel knows
    where the previous session left off; everything else in the
    hot.md is informational for humans (and for Andrus via Obsidian).
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return
    payload = _parse_frontmatter(match.group("yaml"))
    if not isinstance(payload, dict):
        return
    try:
        loop_count = int(payload.get("loop_count", kernel.loop_count))
    except (TypeError, ValueError):
        return
    if loop_count > kernel.loop_count:
        kernel.loop_count = loop_count
    session = payload.get("session_id")
    if isinstance(session, str):
        kernel.session_id = session


# ── Frontmatter helpers (deterministic JSON-in-YAML) ──────────────

def _frontmatter_dict(kernel: SubjectivityKernel) -> dict[str, Any]:
    """Build the frontmatter payload. Compact JSON-in-YAML so the
    round-trip is lossless without needing a YAML library dependency.
    """
    return {
        "title": "SubIA Kernel State",
        "slug": "kernel-state",
        "section": "self",
        "page_type": "log-entry",
        "created_by": "subia-infrastructure",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "update_count": kernel.loop_count,
        "tags": ["subia", "kernel", "consciousness"],
        "status": "active",
        "ownership": {"owned_by": "self", "valued_as": "high"},
        "kernel_payload": _kernel_payload(kernel),
    }


def _kernel_payload(kernel: SubjectivityKernel) -> dict[str, Any]:
    """The lossless JSON-serializable dataclass payload."""
    return {
        "loop_count":    kernel.loop_count,
        "last_loop_at":  kernel.last_loop_at,
        "session_id":    kernel.session_id,
        "scene": [_scene_item_dict(i) for i in kernel.scene],
        "self_state":    _self_state_dict(kernel.self_state),
        "homeostasis":   _homeostasis_dict(kernel.homeostasis),
        "meta_monitor":  _meta_monitor_dict(kernel.meta_monitor),
        "predictions": [
            _prediction_dict(p) for p in kernel.predictions[-64:]
        ],
        "social_models": {
            k: _social_model_dict(v)
            for k, v in kernel.social_models.items()
        },
        "consolidation_buffer": _consolidation_buffer_dict(
            kernel.consolidation_buffer,
        ),
    }


def _scene_item_dict(i) -> dict:
    """Serialize a scene-list item to a JSON-safe dict.

    Duck-typed because `kernel.scene` is the union of two shapes:
    `SceneItem` (the canonical kernel type) and `WorkspaceItem` (gate
    internals — entered through PP-1 surprise routing in Step 8). Both
    are valid in-flight scene items; persistence treats them
    uniformly, mapping `WorkspaceItem.{item_id, content, source_agent,
    salience_score}` to the SceneItem schema. Float-monotonic
    `entered_at` from WorkspaceItem is replaced with the current ISO
    timestamp — the original monotonic value is meaningless after a
    process restart anyway.
    """
    # Identity (SceneItem.id | WorkspaceItem.item_id)
    item_id = getattr(i, "id", None) or getattr(i, "item_id", "") or ""
    source = (
        getattr(i, "source", None)
        or getattr(i, "source_agent", None)
        or getattr(i, "source_channel", "")
        or ""
    )
    content_ref = getattr(i, "content_ref", "") or ""
    summary_raw = (
        getattr(i, "summary", None)
        or getattr(i, "content", "")
        or ""
    )
    salience_val = (
        getattr(i, "salience", None)
        if getattr(i, "salience", None) is not None
        else getattr(i, "salience_score", 0.0)
    )
    entered_at = getattr(i, "entered_at", "")
    if isinstance(entered_at, (int, float)):
        # WorkspaceItem stores a float monotonic timestamp; convert.
        entered_at = datetime.now(timezone.utc).isoformat()
    elif not isinstance(entered_at, str):
        entered_at = ""

    metadata = getattr(i, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "id": str(item_id),
        "source": str(source),
        "content_ref": str(content_ref),
        "summary": str(summary_raw)[:200],
        "salience": round(float(salience_val or 0.0), 4),
        "entered_at": entered_at,
        "ownership": getattr(i, "ownership", "") or "self",
        "valence": round(float(getattr(i, "valence", 0.0) or 0.0), 4),
        "dominant_affect": (
            getattr(i, "dominant_affect", None)
            or metadata.get("affect", "")
            or "neutral"
        ),
        "conflicts_with": list(getattr(i, "conflicts_with", []) or []),
        "action_options": list(getattr(i, "action_options", []) or []),
        "tier": getattr(i, "tier", "") or "focal",
    }


def _self_state_dict(s: SelfState) -> dict:
    return {
        "identity": dict(s.identity),
        "active_commitments": [
            _commitment_dict(c) for c in s.active_commitments
        ],
        "capabilities": dict(s.capabilities),
        "limitations": dict(s.limitations),
        "current_goals": list(s.current_goals),
        "social_roles": dict(s.social_roles),
        "autobiographical_pointers": list(s.autobiographical_pointers),
        "agency_log": s.agency_log[-200:],  # cap
    }


def _commitment_dict(c: Commitment) -> dict:
    return {
        "id": c.id, "description": c.description[:300],
        "venture": c.venture, "created_at": c.created_at,
        "deadline": c.deadline, "status": c.status,
        "related_wiki_pages": list(c.related_wiki_pages),
        "homeostatic_impact": dict(c.homeostatic_impact),
    }


def _homeostatic_dict_safe(d: dict | None) -> dict:
    if not d:
        return {}
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}


def _homeostasis_dict(h: HomeostaticState) -> dict:
    return {
        "variables":   _homeostatic_dict_safe(h.variables),
        "set_points":  _homeostatic_dict_safe(h.set_points),
        "deviations":  _homeostatic_dict_safe(h.deviations),
        "restoration_queue": list(h.restoration_queue),
        "last_updated": h.last_updated,
    }


def _meta_monitor_dict(m: MetaMonitorState) -> dict:
    return {
        "confidence": round(m.confidence, 4),
        "uncertainty_sources": list(m.uncertainty_sources)[:20],
        "known_unknowns": list(m.known_unknowns)[:20],
        "attention_justification": dict(m.attention_justification),
        "active_prediction_mismatches": list(m.active_prediction_mismatches)[:20],
        "agent_conflicts": list(m.agent_conflicts)[:20],
        "missing_information": list(m.missing_information)[:20],
    }


def _prediction_dict(p: Prediction) -> dict:
    return {
        "id": p.id, "operation": p.operation[:200],
        "predicted_outcome": dict(p.predicted_outcome),
        "predicted_self_change": dict(p.predicted_self_change),
        "predicted_homeostatic_effect": _homeostatic_dict_safe(
            p.predicted_homeostatic_effect,
        ),
        "confidence": round(p.confidence, 4),
        "created_at": p.created_at, "resolved": p.resolved,
        "actual_outcome": dict(p.actual_outcome) if p.actual_outcome else None,
        "prediction_error": (
            round(p.prediction_error, 4)
            if p.prediction_error is not None else None
        ),
        "cached": p.cached,
    }


def _social_model_dict(sm: SocialModelEntry) -> dict:
    return {
        "entity_id": sm.entity_id, "entity_type": sm.entity_type,
        "inferred_focus": list(sm.inferred_focus),
        "inferred_expectations": list(sm.inferred_expectations),
        "inferred_priorities": list(sm.inferred_priorities),
        "trust_level": round(sm.trust_level, 4),
        "last_interaction": sm.last_interaction,
        "divergences": list(sm.divergences),
    }


def _consolidation_buffer_dict(cb: ConsolidationBuffer) -> dict:
    # Cap at 100 to prevent unbounded growth; drained by Phase 7.
    return {
        "pending_episodes": cb.pending_episodes[-100:],
        "pending_relations": cb.pending_relations[-100:],
        "pending_self_updates": cb.pending_self_updates[-100:],
        "pending_domain_updates": cb.pending_domain_updates[-100:],
    }


# ── Deserialization ────────────────────────────────────────────────

def _kernel_from_frontmatter(data: dict) -> SubjectivityKernel:
    payload = data.get("kernel_payload")
    if not isinstance(payload, dict):
        return SubjectivityKernel()

    kernel = SubjectivityKernel(
        loop_count=int(payload.get("loop_count", 0)),
        last_loop_at=str(payload.get("last_loop_at", "")),
        session_id=str(payload.get("session_id", "")),
    )

    # Scene
    for raw in payload.get("scene", []) or []:
        try:
            kernel.scene.append(SceneItem(
                id=str(raw.get("id", "")),
                source=str(raw.get("source", "")),
                content_ref=str(raw.get("content_ref", "")),
                summary=str(raw.get("summary", "")),
                salience=float(raw.get("salience", 0.0)),
                entered_at=str(raw.get("entered_at", "")),
                ownership=str(raw.get("ownership", "self")),
                valence=float(raw.get("valence", 0.0)),
                dominant_affect=str(raw.get("dominant_affect", "neutral")),
                conflicts_with=list(raw.get("conflicts_with", [])),
                action_options=list(raw.get("action_options", [])),
                tier=str(raw.get("tier", "focal")),
            ))
        except (TypeError, ValueError):
            continue

    # Self-state
    ss_raw = payload.get("self_state") or {}
    if isinstance(ss_raw, dict):
        kernel.self_state.identity = dict(ss_raw.get("identity", {}) or {})
        kernel.self_state.capabilities = dict(ss_raw.get("capabilities", {}) or {})
        kernel.self_state.limitations = dict(ss_raw.get("limitations", {}) or {})
        kernel.self_state.current_goals = list(ss_raw.get("current_goals", []) or [])
        kernel.self_state.social_roles = dict(ss_raw.get("social_roles", {}) or {})
        kernel.self_state.autobiographical_pointers = list(
            ss_raw.get("autobiographical_pointers", []) or []
        )
        kernel.self_state.agency_log = list(ss_raw.get("agency_log", []) or [])
        for cr in ss_raw.get("active_commitments", []) or []:
            try:
                kernel.self_state.active_commitments.append(Commitment(
                    id=str(cr.get("id", "")),
                    description=str(cr.get("description", "")),
                    venture=str(cr.get("venture", "")),
                    created_at=str(cr.get("created_at", "")),
                    deadline=cr.get("deadline"),
                    status=str(cr.get("status", "active")),
                    related_wiki_pages=list(cr.get("related_wiki_pages", []) or []),
                    homeostatic_impact=dict(cr.get("homeostatic_impact", {}) or {}),
                ))
            except (TypeError, ValueError):
                continue

    # Homeostasis
    h_raw = payload.get("homeostasis") or {}
    if isinstance(h_raw, dict):
        kernel.homeostasis = HomeostaticState(
            variables=_homeostatic_dict_safe(h_raw.get("variables")),
            set_points=_homeostatic_dict_safe(h_raw.get("set_points")),
            deviations=_homeostatic_dict_safe(h_raw.get("deviations")),
            restoration_queue=list(h_raw.get("restoration_queue", []) or []),
            last_updated=str(h_raw.get("last_updated", "")),
        )

    # Meta-monitor
    m_raw = payload.get("meta_monitor") or {}
    if isinstance(m_raw, dict):
        kernel.meta_monitor = MetaMonitorState(
            confidence=float(m_raw.get("confidence", 0.5)),
            uncertainty_sources=list(m_raw.get("uncertainty_sources", []) or []),
            known_unknowns=list(m_raw.get("known_unknowns", []) or []),
            attention_justification=dict(m_raw.get("attention_justification", {}) or {}),
            active_prediction_mismatches=list(
                m_raw.get("active_prediction_mismatches", []) or []
            ),
            agent_conflicts=list(m_raw.get("agent_conflicts", []) or []),
            missing_information=list(m_raw.get("missing_information", []) or []),
        )

    # Predictions
    for pr in payload.get("predictions", []) or []:
        try:
            kernel.predictions.append(Prediction(
                id=str(pr.get("id", "")),
                operation=str(pr.get("operation", "")),
                predicted_outcome=dict(pr.get("predicted_outcome", {}) or {}),
                predicted_self_change=dict(pr.get("predicted_self_change", {}) or {}),
                predicted_homeostatic_effect=_homeostatic_dict_safe(
                    pr.get("predicted_homeostatic_effect"),
                ),
                confidence=float(pr.get("confidence", 0.5)),
                created_at=str(pr.get("created_at", "")),
                resolved=bool(pr.get("resolved", False)),
                actual_outcome=(
                    dict(pr.get("actual_outcome"))
                    if pr.get("actual_outcome") else None
                ),
                prediction_error=(
                    float(pr["prediction_error"])
                    if pr.get("prediction_error") is not None else None
                ),
                cached=bool(pr.get("cached", False)),
            ))
        except (TypeError, ValueError):
            continue

    # Social models
    for entity_id, sm_raw in (payload.get("social_models") or {}).items():
        if not isinstance(sm_raw, dict):
            continue
        try:
            kernel.social_models[entity_id] = SocialModelEntry(
                entity_id=str(sm_raw.get("entity_id", entity_id)),
                entity_type=str(sm_raw.get("entity_type", "agent")),
                inferred_focus=list(sm_raw.get("inferred_focus", []) or []),
                inferred_expectations=list(sm_raw.get("inferred_expectations", []) or []),
                inferred_priorities=list(sm_raw.get("inferred_priorities", []) or []),
                trust_level=float(sm_raw.get("trust_level", 0.7)),
                last_interaction=str(sm_raw.get("last_interaction", "")),
                divergences=list(sm_raw.get("divergences", []) or []),
            )
        except (TypeError, ValueError):
            continue

    # Consolidation buffer
    cb_raw = payload.get("consolidation_buffer") or {}
    if isinstance(cb_raw, dict):
        kernel.consolidation_buffer = ConsolidationBuffer(
            pending_episodes=list(cb_raw.get("pending_episodes", []) or []),
            pending_relations=list(cb_raw.get("pending_relations", []) or []),
            pending_self_updates=list(cb_raw.get("pending_self_updates", []) or []),
            pending_domain_updates=list(cb_raw.get("pending_domain_updates", []) or []),
        )

    return kernel


# ── YAML / markdown plumbing ──────────────────────────────────────

def _yaml_frontmatter(payload: dict) -> str:
    """Minimal YAML emitter: we embed the whole payload as JSON under
    a single `kernel_payload` key to avoid a YAML dependency.

    Top-level scalar keys (title, slug, etc.) are emitted as plain
    YAML; the complex `kernel_payload` is emitted as a JSON-in-YAML
    block-scalar (|) so the round-trip is lossless.
    """
    lines = ["---"]
    for key, value in payload.items():
        if key == "kernel_payload":
            continue
        lines.append(_yaml_scalar(key, value))
    # Embed the payload as compact JSON — safe to re-parse via json.loads
    # without a YAML dependency. Key is prefixed with `json:` so a human
    # reader can tell this block is a JSON literal.
    lines.append("kernel_payload_json: |")
    json_text = json.dumps(payload["kernel_payload"], default=str,
                           separators=(",", ":"))
    # Indent each line by 2 spaces per YAML block-scalar rules.
    for ln in json_text.splitlines() or [json_text]:
        lines.append("  " + ln)
    lines.append("---")
    return "\n".join(lines)


def _yaml_scalar(key: str, value: Any) -> str:
    """Emit a minimal YAML scalar/list/dict. Strings are double-quoted
    when they contain : or # or start with whitespace; everything else
    is passed through. For lists and dicts we fall back to JSON-in-YAML.
    """
    if isinstance(value, bool):
        return f"{key}: {'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"{key}: {value}"
    if value is None:
        return f"{key}: null"
    if isinstance(value, list):
        # Inline flow sequence: [a, b, c]
        if all(isinstance(v, (int, float, bool)) or v is None for v in value):
            return f"{key}: [{', '.join(_yaml_inline_scalar(v) for v in value)}]"
        # Strings: quote each
        if all(isinstance(v, str) for v in value):
            return f"{key}: [{', '.join(_quote(v) for v in value)}]"
        return f"{key}: {json.dumps(value, default=str)}"
    if isinstance(value, dict):
        return f"{key}: {json.dumps(value, default=str)}"
    # String
    return f"{key}: {_quote(str(value))}"


def _yaml_inline_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def _quote(s: str) -> str:
    # Double-quote + escape inner quotes.
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _parse_frontmatter(yaml_text: str) -> dict:
    """Parse the minimal YAML we emit: we only need to find the
    `kernel_payload_json: |` block plus a few top-level scalars
    (updated_at, loop_count, session_id for hot.md).
    """
    result: dict[str, Any] = {}

    lines = yaml_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # block-scalar: "key: |"
        if line.strip().endswith(": |"):
            key = line.strip()[:-3].strip()
            i += 1
            block: list[str] = []
            while i < len(lines):
                next_line = lines[i]
                # block-scalar content is indented (at least 2 spaces)
                if next_line.startswith("  "):
                    block.append(next_line[2:])
                    i += 1
                    continue
                if next_line.strip() == "":
                    i += 1
                    continue
                break
            body = "\n".join(block)
            if key == "kernel_payload_json":
                try:
                    result["kernel_payload"] = json.loads(body)
                except json.JSONDecodeError:
                    logger.debug("subia.persistence: kernel_payload_json "
                                 "failed to parse; falling back to default")
            continue
        # simple "key: value"
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = _parse_yaml_value(value.strip())
        i += 1
    return result


def _parse_yaml_value(raw: str) -> Any:
    if raw == "" or raw == "null":
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


# ── Body formatting (human-readable) ──────────────────────────────

def _markdown_body(kernel: SubjectivityKernel) -> str:
    """The prose body Andrus sees when opening the wiki page."""
    lines = [
        "# SubIA Kernel State",
        "",
        "This page is the human-readable serialization of the system's",
        "subjective kernel state. It is auto-generated after each CIL",
        "loop iteration. See PROGRAM.md and docs for architectural",
        "context.",
        "",
        f"## Loop {kernel.loop_count} — {kernel.last_loop_at or 'never'}",
        "",
        "## Current focal scene",
    ]
    focal = kernel.focal_scene()
    if focal:
        for i, item in enumerate(focal, 1):
            affect = getattr(item, "dominant_affect", "neutral")
            # Salience: SceneItem.salience | WorkspaceItem.salience_score
            salience = round(
                float(getattr(item, "salience", None)
                      or getattr(item, "salience_score", 0.0) or 0.0),
                2,
            )
            tag = f" [{affect}]" if affect != "neutral" else ""
            # Summary: SceneItem.summary | WorkspaceItem.content
            summary_text = (
                getattr(item, "summary", None)
                or getattr(item, "content", "")
                or ""
            )
            lines.append(
                f"{i}. {str(summary_text)[:120]} — salience {salience}{tag}"
            )
    else:
        lines.append("(scene empty)")

    lines.extend(["", "## Self-state"])
    cmt = kernel.self_state.active_commitments
    active = [c for c in cmt if getattr(c, "status", "active") == "active"]
    lines.append(f"Active commitments: {len(active)}")
    lines.append(f"Current goals: {len(kernel.self_state.current_goals)}")
    lines.append(f"Agency-log entries: {len(kernel.self_state.agency_log)}")

    lines.extend(["", "## Homeostasis"])
    threshold = SUBIA_CONFIG["HOMEOSTATIC_DEVIATION_THRESHOLD"]
    dev = {v: d for v, d in kernel.homeostasis.deviations.items()
           if abs(d) > threshold}
    if dev:
        lines.append("Variables above deviation threshold:")
        for var in sorted(dev, key=lambda v: abs(dev[v]), reverse=True):
            lines.append(f"- {var}: {dev[var]:+.2f}")
    else:
        lines.append("All variables within equilibrium range.")

    lines.extend(["", "## Meta-monitor"])
    lines.append(f"Confidence: {kernel.meta_monitor.confidence:.2f}")
    lines.append(f"Known unknowns: {len(kernel.meta_monitor.known_unknowns)}")

    return "\n".join(lines)


def _derive_resume_hint(kernel: SubjectivityKernel) -> str:
    """One-line suggestion for where to pick up next session."""
    dev = kernel.homeostasis.deviations
    if dev:
        top = max(dev, key=lambda v: abs(dev[v]))
        if abs(dev[top]) > SUBIA_CONFIG["HOMEOSTATIC_DEVIATION_THRESHOLD"]:
            direction = "reduce" if dev[top] > 0 else "increase"
            return f"Start by attending to homeostatic pressure: {direction} {top}."
    unresolved = [
        c for c in kernel.self_state.active_commitments
        if getattr(c, "status", "active") == "active"
    ]
    if unresolved:
        return f"Resume active commitment: {unresolved[0].description[:80]}."
    return "System balanced — no urgent resume hint."
