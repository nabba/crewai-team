"""synthesis_pass — cross-subsystem creative blend pass (Q17.7).

Weekly idle daemon. Picks 2 random subsystem pairs from a curated
descriptor list, feeds them to concept_blend.blend_concepts, scores
results with novelty_wrap + aesthetic_score, persists candidates.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_RUNNING = False
_THREAD_NAME = "synthesis-pass-daemon"
_LOCK = threading.Lock()
_MAX_CANDIDATES_PER_PASS = 2
_CADENCE_DAYS = 7

SUBSYSTEM_DESCRIPTORS: list[tuple[str, str, list[str], list[str]]] = [
    ("tool_registry", "capability vocabulary + ChromaDB semantic search index",
     ["@register_tool decorator", "capability tags", "ChromaDB index"], ["registers", "queries", "tags"]),
    ("companion_tensions", "open questions operator left with the system, freshness decay",
     ["tension records", "freshness halflife", "auto-dormant"], ["surfaces", "decays", "ages"]),
    ("person_correlation", "four-level opt-in tracking of people in operator's modalities",
     ["presence", "centrality", "suggestions", "social graph"], ["tracks", "scores", "connects"]),
    ("self_heal_runbooks", "reactive anomaly-pattern remediation with operator gate",
     ["pattern signatures", "runbook handlers"], ["matches", "remediates", "files-CR"]),
    ("philosophy_panel", "multi-tradition perspective consult for high-stakes decisions",
     ["Stoic", "Buddhist", "Pragmatist"], ["consults", "weighs", "tensions"]),
    ("creative_brainstorm", "8-technique state-machine ideation with multi-agent rounds",
     ["SCAMPER", "Six Hats", "concept-blend"], ["generates", "diverges", "selects"]),
    ("analogy_index", "structural-pattern retrieval over wiki + episteme",
     ["structure signature", "cross-domain examples"], ["retrieves", "matches", "abstracts"]),
    ("long_term_threads", "5-state lifecycle for multi-month inquiry",
     ["OPEN/IN_PROGRESS/BLOCKED", "unblock_hints"], ["tracks", "unblocks", "distills"]),
    ("browse_ingestion", "browser history as 6th interest signal modality",
     ["URL canonicalization", "blocklist", "topic clustering"], ["observes", "redacts", "themes"]),
    ("vacation_mode", "time-bounded operator-pre-staged auto-approval allowlist",
     ["typed-phrase", "rate-limited", "auto-revert"], ["delegates", "rate-limits", "audits"]),
    ("annual_reflection", "yearly value-drift summary over continuity ledger",
     ["per-kind Counter", "drift summary"], ["aggregates", "reflects", "narrates"]),
    ("interest_model", "5-stream cross-modal interest profiling",
     ["topic clusters", "Jaccard stability"], ["profiles", "clusters", "tracks"]),
    ("agreement_ledger", "operator-response self-model: accept/reject/ignore rates",
     ["rolling rate", "by-category"], ["measures", "buckets", "reports"]),
    ("warm_spare", "partner-host replication with failover state machine",
     ["replication manifest", "heartbeat", "claim-canonical"], ["replicates", "monitors", "fails-over"]),
    ("kb_contradiction", "weekly probe for negation contradictions across claims",
     ["pairwise contradiction", "subject grouping"], ["samples", "detects", "surfaces"]),
    ("tier3_amendment", "10-state protocol for amending TIER_IMMUTABLE files",
     ["eligibility gate", "self-quarantine"], ["proposes", "stages", "applies"]),
    ("library_radar", "discovery + smoke-trial + adoption-CR for libraries",
     ["pip outdated", "OSV.dev CVE"], ["discovers", "trials", "adopts"]),
    ("paper_pipeline", "research-paper digest + codeable-paper scaffold",
     ["multi-source", "codeable flag"], ["fetches", "summarises", "scaffolds"]),
    ("decentered_reflection", "no-self reflective pass over affect trace",
     ["episode", "affect arc"], ["reflects", "decenters", "narrates"]),
    ("cross_modal_pattern", "convergence detector across 6 interest modalities",
     ["modality factor", "log-scaled volume"], ["correlates", "detects", "boosts"]),
]


@dataclass
class SynthesisCandidate:
    ts: str
    subsystem_a: str
    subsystem_b: str
    blend_label: str
    emergent_structure: list[str] = field(default_factory=list)
    follow_on_questions: list[str] = field(default_factory=list)
    novelty_verdict: str | None = None
    aesthetic_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _candidates_path() -> Path:
    return _workspace_root() / "creativity" / "synthesis_candidates.jsonl"


def _state_path() -> Path:
    return _workspace_root() / "creativity" / "synthesis_pass_state.json"


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_synthesis_pass_enabled
        return get_synthesis_pass_enabled()
    except Exception:
        return True


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run": 0, "n_total": 0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run": 0, "n_total": 0}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        logger.debug("synthesis_pass: state write failed", exc_info=True)


def _append_candidates(candidates: list[SynthesisCandidate]) -> None:
    p = _candidates_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "a", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c.to_dict(), sort_keys=True) + "\n")
    except OSError:
        logger.debug("synthesis_pass: candidate append failed", exc_info=True)


def _build_input_space(descriptor: tuple[str, str, list[str], list[str]]):
    from app.creativity.concept_blend import InputSpace
    name, desc, elements, relations = descriptor
    return InputSpace(label=f"{name}: {desc}", salient_elements=elements, salient_relations=relations)


def _score_candidate(blend, blend_text: str) -> tuple[str | None, float | None]:
    novelty = None
    aesthetic = None
    try:
        from app.creativity import novelty_wrap
        verdict = novelty_wrap.assess_brainstorm_idea(blend_text)
        novelty = verdict.get("verdict") if isinstance(verdict, dict) else getattr(verdict, "verdict", None)
    except Exception:
        logger.debug("synthesis_pass: novelty failed", exc_info=True)
    try:
        from app.creativity import aesthetic_score
        aesthetic = aesthetic_score.score(blend_text)
    except Exception:
        logger.debug("synthesis_pass: aesthetic failed", exc_info=True)
    return novelty, aesthetic


def _pick_pairs(rng: random.Random, n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < 50:
        attempts += 1
        i, j = rng.sample(range(len(SUBSYSTEM_DESCRIPTORS)), 2)
        key = tuple(sorted([i, j]))
        if key in seen:
            continue
        seen.add(key)
        out.append((i, j))
    return out


def run_one_pass(*, rng_seed: int | None = None, llm_call: Optional[Callable] = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_attempted": 0, "n_persisted": 0, "errors": 0, "candidates": []}
    if not _enabled():
        summary["skipped"] = True
        return summary
    try:
        from app.creativity.concept_blend import blend_concepts
    except Exception:
        summary["errors"] += 1
        return summary
    rng = random.Random(rng_seed if rng_seed is not None else int(time.time()))
    pairs = _pick_pairs(rng, _MAX_CANDIDATES_PER_PASS)
    summary["n_attempted"] = len(pairs)
    persisted: list[SynthesisCandidate] = []
    for i, j in pairs:
        a_desc = SUBSYSTEM_DESCRIPTORS[i]
        b_desc = SUBSYSTEM_DESCRIPTORS[j]
        try:
            space_a = _build_input_space(a_desc)
            space_b = _build_input_space(b_desc)
            blend = blend_concepts(space_a, space_b, llm_call=llm_call)
            if blend is None:
                summary["errors"] += 1
                continue
            blend_text = f"{getattr(blend, 'blend_label', '')}. {' '.join(getattr(blend, 'emergent_structure', []) or [])}"
            novelty, aesthetic = _score_candidate(blend, blend_text)
            persisted.append(SynthesisCandidate(
                ts=datetime.now(timezone.utc).isoformat(),
                subsystem_a=a_desc[0],
                subsystem_b=b_desc[0],
                blend_label=getattr(blend, "blend_label", "")[:280],
                emergent_structure=list(getattr(blend, "emergent_structure", []) or [])[:8],
                follow_on_questions=list(getattr(blend, "follow_on_questions", []) or [])[:5],
                novelty_verdict=novelty,
                aesthetic_score=aesthetic,
            ))
        except Exception:
            logger.debug("synthesis_pass: blend failed", exc_info=True)
            summary["errors"] += 1
    if persisted:
        _append_candidates(persisted)
        try:
            from app.identity.continuity_ledger import record_event
            record_event(kind="q17_landmark", actor="synthesis_pass",
                         summary=f"synthesis_pass produced {len(persisted)} candidate blend(s)",
                         detail={"subsystem": "synthesis_pass", "n": len(persisted)})
        except Exception:
            pass
    summary["n_persisted"] = len(persisted)
    summary["candidates"] = [c.to_dict() for c in persisted]
    state = _read_state()
    state["last_run"] = datetime.now(timezone.utc).timestamp()
    state["n_total"] = int(state.get("n_total") or 0) + len(persisted)
    _write_state(state)
    return summary


def recent_candidates(*, n: int = 10) -> list[dict[str, Any]]:
    p = _candidates_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    out.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return out[:n]


def briefing_section() -> str:
    cands = recent_candidates(n=3)
    if not cands:
        return ""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    cands = [c for c in cands if c.get("ts", "") >= cutoff]
    if not cands:
        return ""
    lines = ["💡 Synthesis candidates this week:"]
    for c in cands:
        lines.append(f"  • {c.get('subsystem_a')} × {c.get('subsystem_b')}: {c.get('blend_label')}")
        nv = c.get("novelty_verdict")
        score = c.get("aesthetic_score")
        if nv or score is not None:
            lines.append(f"      novelty={nv} aesthetic={score}")
    return "\n".join(lines)


def _cadence_due(state: dict[str, Any]) -> bool:
    last = float(state.get("last_run") or 0)
    return (time.time() - last) >= _CADENCE_DAYS * 24 * 3600


def _daemon_loop() -> None:
    while True:
        try:
            state = _read_state()
            if _enabled() and _cadence_due(state):
                run_one_pass()
        except Exception:
            logger.debug("synthesis_pass: daemon iter failed", exc_info=True)
        time.sleep(3600)


def start_daemon() -> None:
    """Spawn the weekly synthesis-pass daemon.

    Idempotent: re-entrancy guarded by ``_RUNNING`` AND a thread-name
    enumeration check, so multiple imports (production boot + isolated
    test loads) never spawn duplicate threads. Daemon flag set so the
    process exits cleanly without joining.
    """
    global _RUNNING
    with _LOCK:
        if _RUNNING:
            return
        already = any(t.name == _THREAD_NAME and t.is_alive() for t in threading.enumerate())
        if already:
            _RUNNING = True
            return
        t = threading.Thread(target=_daemon_loop, name=_THREAD_NAME, daemon=True)
        t.start()
        _RUNNING = True


# Side-effect-on-import — matches the convention used by 12 sibling
# observational daemons (handlers, monitors, watchdog, library_radar,
# proposal_bridge, dependency_radar, auto_revert, governance_notifier,
# capability_gap_analyzer, recipe_consolidation, source_ledger_daemon,
# local_only) anchored from app/healing/__init__.py. Each module
# self-starts on first import; the healing __init__ is the canonical
# boot-chain hub that imports them all. start_daemon() is idempotent,
# so duplicate test-isolation loads are safe.
start_daemon()
