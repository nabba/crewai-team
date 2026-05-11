"""
core.py — Affect core: V_t / A_t / C_t computation + attractor labeling.

Reads from the existing self-awareness substrate:
    valence          ← state.somatic.valence (Damasio somatic marker)
    arousal          ← hyper_model.free_energy_proxy + certainty variance + Δerror/Δt
    controllability  ← state.certainty.adjusted_certainty (+ optional reality precision)

The "constructed emotion" attractor (Barrett) is a discrete label derived from
the V/A/C tuple together with viability context — it does not reflect a fixed
basic-emotion module. The label is for human readability and audit; the
machinery downstream uses V/A/C floats, not the label.

Persistence:
    Every snapshot is appended to /app/workspace/affect/trace.jsonl
    (one JSON object per line, ts-ordered) for the daily reflection cycle.

Phase-1 note: this module does not yet bias action selection. Hooks
(see hooks.py) and llm_sampling extension (separate change) will read the
state produced here.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any

from app.affect.schemas import AffectState, ViabilityFrame, utc_now_iso
from app.affect.viability import compute_viability_frame
from app.utils.jsonl_retention import append_with_archive_rotate

logger = logging.getLogger(__name__)

from app.paths import AFFECT_TRACE as _TRACE_FILE  # noqa: E402  workspace-aware path
_TRACE_LOCK = threading.Lock()
# Cap: ~100k lines of dense trace ≈ 2 months at 1/min cadence. Older records
# rotate to workspace/affect/archive/<YYYY-MM>_trace.jsonl — preserved forever
# for HOT-1 / decentered-reflection / backward-counterfactual replay probes.
_TRACE_MAX_LINES = 100_000

# Rolling buffer for trend / arousal computation (in-process; trace.jsonl is durable).
_recent_states: deque[AffectState] = deque(maxlen=64)
_recent_total_errors: deque[float] = deque(maxlen=32)


# ── V_t ──────────────────────────────────────────────────────────────────────


def _compute_valence(internal_state: Any | None, frame: ViabilityFrame) -> tuple[float, str]:
    """V_t in [-1, 1]. Movement toward (+) or away from (-) viability."""
    # Primary: somatic valence from existing SomaticMarker.
    somatic_v = 0.0
    have_somatic = False
    if internal_state is not None and hasattr(internal_state, "somatic"):
        try:
            somatic_v = float(internal_state.somatic.valence)
            have_somatic = bool(getattr(internal_state.somatic, "intensity", 0.0) > 0.05)
        except Exception:
            pass

    # Secondary: viability-deficit pull. High E_t pulls valence negative.
    # E_t is in [0, 1] roughly; map to a signed pull in [-0.5, 0].
    deficit_pull = -min(0.5, frame.total_error)

    # Compose: weight somatic heavier when it has any intensity, else deficit dominates.
    if have_somatic:
        composed = 0.7 * somatic_v + 0.3 * deficit_pull
        source = "composite (somatic + deficit)"
    else:
        composed = deficit_pull
        source = "viability_deficit"

    return _clamp(composed, -1.0, 1.0), source


# ── A_t ──────────────────────────────────────────────────────────────────────


def _compute_arousal(internal_state: Any | None, frame: ViabilityFrame) -> tuple[float, str]:
    """A_t in [0, 1]. Urgency, uncertainty, rate of change."""
    # Primary: free-energy proxy from HyperModel.
    fe = 0.0
    have_fe = False
    if internal_state is not None:
        try:
            fe = float(getattr(internal_state, "free_energy_proxy", 0.0))
            have_fe = fe > 0.0
        except Exception:
            pass

    # Secondary: rate of change of E_t (sudden viability collapse → arousal spike).
    delta = 0.0
    if _recent_total_errors:
        prev = _recent_total_errors[-1]
        delta = max(0.0, frame.total_error - prev)

    # Tertiary: epistemic uncertainty (already a viability variable).
    eu = float(frame.values.get("epistemic_uncertainty", 0.3))

    # Compose: free-energy carries most weight when present.
    if have_fe:
        composed = 0.5 * _clamp(fe * 2.0) + 0.3 * _clamp(delta * 5.0) + 0.2 * eu
        source = "free_energy + Δerror + uncertainty"
    else:
        composed = 0.6 * _clamp(delta * 5.0) + 0.4 * eu
        source = "Δerror + uncertainty"

    return _clamp(composed, 0.0, 1.0), source


# ── C_t ──────────────────────────────────────────────────────────────────────


def _compute_controllability(internal_state: Any | None, frame: ViabilityFrame) -> tuple[float, str]:
    """C_t in [0, 1]. Expected ability to reduce E_t."""
    # Primary: adjusted certainty from CertaintyVector.
    if internal_state is not None and hasattr(internal_state, "certainty"):
        try:
            ac = float(internal_state.certainty.adjusted_certainty)
            return _clamp(ac, 0.0, 1.0), "certainty.adjusted_certainty"
        except Exception:
            pass

    # Fallback: 1 - E_t (low error implies relatively good control).
    return _clamp(1.0 - frame.total_error, 0.0, 1.0), "1 - E_t"


# ── Constructed-emotion attractor label ──────────────────────────────────────


def _label_attractor(v: float, a: float, c: float, frame: ViabilityFrame) -> str:
    """Discrete attractor label from V/A/C + viability context.

    These are interpretive labels for human readers and dashboards. They are
    NOT exposed to the LLM (per CLAUDE.md output-quality preference: no internal
    metadata in agent output). Constructed-emotion view: same V/A/C can take
    different labels depending on viability context.
    """
    # Strongly positive, low arousal → peace / contentment
    if v >= 0.4 and a < 0.35:
        if frame.values.get("ecological_connectedness", 0.0) > 0.7:
            return "oneness"
        if frame.total_error < 0.15:
            return "peace"
        return "contentment"

    # Positive, high arousal → excitement / SEEKING
    if v >= 0.3 and a >= 0.55:
        return "excitement"

    # Negative, high arousal, high controllability → focused effort / urgency
    if v < -0.2 and a >= 0.5 and c >= 0.5:
        return "urgency"

    # Negative, high arousal, low controllability → distress
    if v < -0.3 and a >= 0.5 and c < 0.4:
        return "distress"

    # Negative, low arousal → discouragement / fatigue
    if v < -0.2 and a < 0.4:
        if frame.values.get("compute_reserve", 1.0) < 0.3:
            return "depletion"
        return "discouragement"

    # Specific deficits map to specific attractors when V/A are mild.
    out_of_band = frame.out_of_band(tolerance=0.25)
    if "compute_reserve" in out_of_band and frame.values["compute_reserve"] < 0.3:
        return "hunger"
    if "attachment_security" in out_of_band and frame.values["attachment_security"] < 0.45:
        return "separation"
    if "novelty_pressure" in out_of_band:
        if frame.values["novelty_pressure"] < 0.2:
            return "boredom"
        if frame.values["novelty_pressure"] > 0.85:
            return "overwhelm"

    return "neutral"


# ── Public entry point ──────────────────────────────────────────────────────


def compute_affect(
    internal_state: Any | None = None,
    *,
    persist: bool = True,
) -> tuple[AffectState, ViabilityFrame]:
    """Compute the current AffectState. Returns (affect, viability_frame)."""
    frame = compute_viability_frame(internal_state)

    v, vs = _compute_valence(internal_state, frame)
    a, as_ = _compute_arousal(internal_state, frame)
    c, cs = _compute_controllability(internal_state, frame)
    attractor = _label_attractor(v, a, c, frame)

    state = AffectState(
        valence=v,
        arousal=a,
        controllability=c,
        valence_source=vs,
        arousal_source=as_,
        controllability_source=cs,
        attractor=attractor,
        internal_state_id=getattr(internal_state, "state_id", None),
        viability_frame_ts=frame.ts,
        ts=utc_now_iso(),
    )

    prev = _recent_states[-1] if _recent_states else None
    _recent_states.append(state)
    _recent_total_errors.append(frame.total_error)

    if persist:
        _append_trace(state, frame)
        try:
            from app.affect.salience import emit_if_salient
            emit_if_salient(state, frame, prev)
        except Exception:
            logger.debug("affect.core: salience hook failed", exc_info=True)

    return state, frame


def latest_affect() -> AffectState | None:
    """Most recent computed AffectState, or None if none yet."""
    if _recent_states:
        return _recent_states[-1]
    return None


def recent_affect(n: int = 32) -> list[AffectState]:
    """In-process recent AffectStates for trend computation by other modules."""
    return list(_recent_states)[-n:]


# ── Trace persistence ────────────────────────────────────────────────────────


def _append_trace(state: AffectState, frame: ViabilityFrame) -> None:
    """Append one line to trace.jsonl with bounded archive rotation.

    Locked so concurrent steps don't interleave. Once the live file exceeds
    ``_TRACE_MAX_LINES``, the oldest half rotates to
    ``workspace/affect/archive/<YYYY-MM>_trace.jsonl`` — history preserved
    indefinitely while the live file stays read-hot.
    """
    try:
        line = json.dumps({
            "affect": state.to_dict(),
            "viability": frame.to_dict(),
        }, default=str)
        with _TRACE_LOCK:
            append_with_archive_rotate(
                _TRACE_FILE, line, max_lines=_TRACE_MAX_LINES,
            )
    except Exception:
        logger.debug("affect.core: trace append failed", exc_info=True)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))
