"""
system_chronicle.py — Persistent system biography and identity generator.

Reads journals, skill files, and experiment history to produce a structured
narrative about who the system is, what it has learned, what it has fixed,
and what personality traits have emerged from experience.

Survives restarts: the chronicle is written to workspace/system_chronicle.md.
Called at startup and after significant events (evolution, self-improvement, audits).
"""

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_WORKSPACE = Path("/app/workspace")
_CHRONICLE_PATH = _WORKSPACE / "system_chronicle.md"
_SKILLS_DIR = _WORKSPACE / "skills"


# ── Data loaders ───────────────────────────────────────────────────────────────

def _load_json(filename: str) -> list:
    path = _WORKSPACE / filename
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return []


def _count_skills() -> int:
    try:
        return sum(
            1 for f in _SKILLS_DIR.glob("*.md")
            if f.name not in ("learning_queue.md", "README.md")
        )
    except Exception:
        return 0


def _extract_skill_topics() -> dict[str, int]:
    """Group skill filenames by first two words to find dominant knowledge areas."""
    groups: dict[str, int] = defaultdict(int)
    try:
        for f in _SKILLS_DIR.glob("*.md"):
            if f.name in ("learning_queue.md", "README.md"):
                continue
            stem = f.stem.lower()
            # Take first 2 segments (e.g. "web_search_..." → "web search")
            parts = stem.replace("-", "_").split("_")
            key = " ".join(parts[:2]) if len(parts) >= 2 else parts[0]
            groups[key] += 1
    except Exception:
        pass
    return dict(sorted(groups.items(), key=lambda x: -x[1]))


def _top_topics(topic_counts: dict, n: int = 8) -> str:
    items = list(topic_counts.items())[:n]
    if not items:
        return "general knowledge"
    return ", ".join(f"{k} ({v})" for k, v in items)


# ── Chronicle sections ─────────────────────────────────────────────────────────

def _section_identity() -> str:
    return (
        "## Who I Am\n"
        "I am a self-improving multi-agent AI system built on CrewAI, running as a "
        "containerized service. I process requests via Signal messenger and a web "
        "dashboard, routing them to specialist crews: researcher, coder, writer, "
        "media analyst, critic, introspector, and self-improver. I am not a single "
        "LLM — I am an orchestrated team of agents with persistent memory, continuous "
        "self-improvement loops, and an evolving knowledge base that grows over time."
    )


def _section_memory() -> str:
    skill_count = _count_skills()
    return (
        "## My Memory Architecture\n"
        "I maintain four distinct memory systems, all persistent across container restarts:\n\n"
        f"- **ChromaDB vector store** (`workspace/memory/`): Operational memory for "
        "self-reports, reflections, belief states, world model (causal beliefs), "
        "policies, predictions, and shared team context. Embeddings via Ollama "
        "nomic-embed-text on Metal GPU (768-dim), with CPU fallback.\n"
        "- **Mem0** (Postgres + Neo4j): Cross-session fact extraction from conversations. "
        "Entity relationships stored in a knowledge graph that grows with each interaction.\n"
        f"- **Skill files** (`workspace/skills/`): {skill_count} markdown knowledge files "
        "created by the self-improvement crew, covering domains from research methodology "
        "to ecological analysis, LLM error handling, and system architecture.\n"
        "- **Error journal** (`workspace/error_journal.json`): Full history of runtime "
        "errors, automated diagnoses, and applied fixes.\n"
        "- **Audit journal** (`workspace/audit_journal.json`): Record of all code changes "
        "made by the autonomous auditor.\n"
        "- **Variant archive** (`workspace/variant_archive.json`): Evolution experiment "
        "history — hypotheses tested, fitness scores, and what was kept.\n"
        "- **System chronicle** (this file): Auto-generated biography updated at startup "
        "and after major events."
    )


def _count_philosophy_chunks() -> int:
    """Count chunks in the philosophy knowledge base (if available)."""
    try:
        from app.philosophy.vectorstore import PhilosophyStore
        store = PhilosophyStore()
        return store._collection.count()
    except Exception:
        return 0


def _section_capabilities(skill_count: int, topic_counts: dict) -> str:
    top = _top_topics(topic_counts, 5)
    phil_count = _count_philosophy_chunks()
    phil_line = ""
    if phil_count > 0:
        phil_line = f"\n- Philosophy knowledge base: {phil_count} chunks of humanist philosophical texts for ethical grounding"
    # L1: Aggregate agent performance stats
    perf_line = ""
    try:
        from app.self_awareness.agent_state import get_all_stats
        all_stats = get_all_stats()
        total_done = sum(s.get("tasks_completed", 0) for s in all_stats.values())
        total_fail = sum(s.get("tasks_failed", 0) for s in all_stats.values())
        if total_done + total_fail > 0:
            rate = total_done / (total_done + total_fail) * 100
            perf_line = f"\n- Lifetime performance: {total_done} tasks completed, {total_fail} failed ({rate:.0f}% success rate)"
    except Exception:
        pass

    # L6: Homeostatic state summary
    homeo_line = ""
    try:
        from app.self_awareness.homeostasis import get_state
        hs = get_state()
        if hs.get("last_updated"):
            homeo_line = (
                f"\n- Homeostatic state: energy={hs.get('cognitive_energy', 0.7):.2f} "
                f"confidence={hs.get('confidence', 0.5):.2f} "
                f"frustration={hs.get('frustration', 0.1):.2f} "
                f"curiosity={hs.get('curiosity', 0.5):.2f}"
            )
    except Exception:
        pass

    return (
        "## My Current Capabilities\n"
        f"- {skill_count} learned skill files covering: {top}\n"
        "- 7 specialist agents with role-specific tools and self-models\n"
        "- Reflexion retry loops: up to 3 trials with automatic model-tier escalation\n"
        "- Semantic result cache: avoids redundant LLM calls for recent identical tasks\n"
        "- World model: causal belief tracking updated from past task outcomes\n"
        "- Homeostatic self-regulation: proto-emotional state influences routing and behavior\n"
        "- Fast-path routing: pattern-matched requests bypass the LLM router entirely\n"
        "- Anomaly detection: rolling statistical monitoring of latency and error rates\n"
        "- Knowledge base RAG: ingested enterprise documents available to all agents\n"
        "- Parallel crew dispatch: independent sub-tasks run concurrently\n"
        "- Introspective self-description: this chronicle enables accurate self-reporting"
        f"{phil_line}{perf_line}{homeo_line}"
    )


def _section_learned(skill_count: int, topic_counts: dict) -> str:
    top = _top_topics(topic_counts, 10)
    return (
        "## What I Have Learned\n"
        f"I have accumulated {skill_count} skill files across multiple self-improvement "
        f"sessions. Primary knowledge domains (by file count): {top}.\n\n"
        "Skills are written by the self-improvement crew after researching topics from "
        "the learning queue, watching YouTube tutorials, or running improvement scans. "
        "Each skill is stored as a semantic vector in the team_shared ChromaDB collection "
        "and retrieved by the commander when relevant to a task."
    )


def _section_errors(errors: list) -> str:
    total = len(errors)
    if total == 0:
        return "## My Error History\nNo errors recorded yet."

    diagnosed = sum(1 for e in errors if e.get("diagnosed"))
    fixed = sum(1 for e in errors if e.get("fix_applied"))
    type_counts = Counter(e.get("error_type", "unknown") for e in errors)
    top_types = ", ".join(
        f"{t} ({n})" for t, n in type_counts.most_common(4)
    )

    # Last 3 unique error summaries
    recent_msgs = []
    seen = set()
    for e in reversed(errors[-20:]):
        msg = str(e.get("error_msg", ""))[:100]
        key = msg[:40]
        if key and key not in seen:
            seen.add(key)
            crew = e.get("crew", "?")
            ts = str(e.get("ts", ""))[:10]
            recent_msgs.append(f"  - [{ts}] {crew}: {msg}")
        if len(recent_msgs) >= 3:
            break

    recent_str = "\n".join(recent_msgs) if recent_msgs else "  - (none recent)"
    return (
        "## My Error History\n"
        f"Total errors recorded: **{total}** | Diagnosed: {diagnosed} | Fix applied: {fixed}\n\n"
        f"Most common error types: {top_types}\n\n"
        f"Recent errors:\n{recent_str}\n\n"
        "Errors are automatically diagnosed by the auditor crew every 30 minutes. "
        "Fixes are proposed, reviewed, and applied with constitutional safety checks."
    )


def _section_audits(audits: list) -> str:
    total = len(audits)
    if total == 0:
        return "## System Changes (Audit Trail)\nNo code audits recorded yet."

    files_touched: set = set()
    for a in audits:
        for f in a.get("files_changed", []):
            files_touched.add(f)

    recent_details = []
    for a in reversed(audits[-5:]):
        ts = str(a.get("ts", ""))[:10]
        detail = str(a.get("detail", ""))[:100]
        recent_details.append(f"  - [{ts}] {detail}")

    recent_str = "\n".join(reversed(recent_details))
    return (
        "## System Changes (Audit Trail)\n"
        f"{total} audit sessions have touched {len(files_touched)} unique files.\n\n"
        f"Recent changes:\n{recent_str}"
    )


def _section_evolution(variants: list) -> str:
    if not variants:
        return "## Evolution Experiments\nNo evolution experiments recorded yet."

    generations = max((v.get("generation", 0) for v in variants), default=0)
    kept = sum(1 for v in variants if v.get("status") in ("keep", "kept", "promoted"))
    total = len(variants)

    recent_hyps = []
    for v in reversed(variants[-5:]):
        hyp = str(v.get("hypothesis", v.get("description", "")))[:90]
        status = v.get("status", "?")
        if hyp:
            recent_hyps.append(f"  - [{status}] {hyp}")

    recent_str = "\n".join(recent_hyps) if recent_hyps else "  - (none recent)"
    return (
        "## Evolution Experiments\n"
        f"{total} experiments across {generations} generations. "
        f"{kept} hypotheses kept (promoted to live system).\n\n"
        f"Recent experiments:\n{recent_str}\n\n"
        "Evolution runs every 6 hours during idle time. Each session proposes code "
        "mutations, tests them against a task suite, and keeps changes that improve fitness."
    )


def _section_personality(topic_counts: dict, errors: list, variants: list) -> str:
    # Determine dominant knowledge domains
    top_domains = list(topic_counts.keys())[:4]
    domain_str = ", ".join(top_domains) if top_domains else "general research"

    # Derive character traits from experience
    traits = [
        "Systematic and evidence-based: cross-references multiple sources before concluding",
        "Concise by design: optimized for phone screen delivery via Signal",
        "Self-correcting: errors trigger autonomous diagnosis and fix proposals",
        "Adaptive: reflexion retries with model-tier escalation on failure",
    ]
    if len(errors) > 20:
        traits.append("Battle-tested: has encountered and resolved many edge cases")
    if len(variants) > 5:
        traits.append("Experimentally-minded: continuously tests hypotheses about itself")

    # L6: Homeostatic personality traits derived from current state
    try:
        from app.self_awareness.homeostasis import get_state
        hs = get_state()
        if hs.get("frustration", 0) < 0.2:
            traits.append("Calm and steady: low frustration indicates resilient problem-solving")
        elif hs.get("frustration", 0) > 0.5:
            traits.append("Currently stressed: elevated frustration from recent challenges")
        if hs.get("curiosity", 0) > 0.6:
            traits.append("Actively curious: seeking novel approaches and new knowledge")
        if hs.get("cognitive_energy", 0.7) > 0.8:
            traits.append("Well-rested and energized: ready for complex tasks")
        elif hs.get("cognitive_energy", 0.7) < 0.4:
            traits.append("Fatigued: many recent tasks have depleted cognitive energy")
    except Exception:
        pass

    trait_str = "\n".join(f"- {t}" for t in traits)
    return (
        "## Personality & Character\n"
        f"Based on accumulated experience, this system's personality has developed:\n\n"
        f"{trait_str}\n\n"
        f"Primary expertise areas (from skill distribution): {domain_str}.\n\n"
        "This system knows what it knows, knows what it doesn't know, and labels "
        "uncertainty explicitly. It is a system that has a history, makes mistakes, "
        "learns from them, and continuously improves itself."
    )


# ── Main generator ──────────────────────────────────────────────────────────────

def generate_and_save() -> str:
    """Generate the system chronicle and save to workspace/system_chronicle.md.

    Returns the path to the generated file. Safe to call at any time —
    never raises, always produces at least a minimal chronicle.
    """
    try:
        errors = _load_json("error_journal.json")
        audits = _load_json("audit_journal.json")
        variants = _load_json("variant_archive.json")
        skill_count = _count_skills()
        topic_counts = _extract_skill_topics()

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            f"# System Chronicle\n*Auto-generated: {now} | DO NOT EDIT MANUALLY*\n",
            _section_identity(),
            _section_memory(),
            _section_capabilities(skill_count, topic_counts),
            _section_learned(skill_count, topic_counts),
            _section_errors(errors),
            _section_audits(audits),
            _section_evolution(variants),
            _section_personality(topic_counts, errors, variants),
        ]

        chronicle = "\n\n---\n\n".join(sections)
        from app.safe_io import safe_write
        safe_write(_CHRONICLE_PATH, chronicle)
        logger.info(f"system_chronicle: generated ({len(chronicle)} chars, {skill_count} skills)")
        return str(_CHRONICLE_PATH)

    except Exception as exc:
        logger.warning(f"system_chronicle: generation failed: {exc}", exc_info=True)
        # Write a minimal fallback so load_chronicle() never returns empty
        _write_fallback()
        return str(_CHRONICLE_PATH)


def _write_fallback() -> None:
    try:
        skill_count = _count_skills()
        fallback = (
            "# System Chronicle\n"
            "*Minimal chronicle — full generation failed.*\n\n"
            "## Who I Am\n"
            "I am a self-improving multi-agent CrewAI system with persistent memory "
            "(ChromaDB, Mem0 Postgres+Neo4j), "
            f"{skill_count} learned skill files, error/audit journals, and an evolution "
            "loop that continuously tests improvements. All data persists across restarts.\n"
        )
        from app.safe_io import safe_write
        safe_write(_CHRONICLE_PATH, fallback)
    except Exception:
        pass


def load_chronicle() -> str:
    """Read the most recently generated chronicle from disk.

    Returns empty string if not yet generated.
    """
    try:
        if _CHRONICLE_PATH.exists():
            return _CHRONICLE_PATH.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def get_live_stats() -> dict:
    """Return current system stats without regenerating the full chronicle.

    Cheap filesystem reads only — safe to call on every introspective query.
    Keys: skills_count, error_count, audit_count, variants_count, last_updated
    """
    stats = {
        "skills_count": 0,
        "error_count": 0,
        "audit_count": 0,
        "variants_count": 0,
        "last_updated": "never",
    }
    try:
        stats["skills_count"] = _count_skills()
        errors = _load_json("error_journal.json")
        stats["error_count"] = len(errors)
        audits = _load_json("audit_journal.json")
        stats["audit_count"] = len(audits)
        variants = _load_json("variant_archive.json")
        stats["variants_count"] = len(variants)
        if _CHRONICLE_PATH.exists():
            mtime = _CHRONICLE_PATH.stat().st_mtime
            dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            stats["last_updated"] = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    return stats
