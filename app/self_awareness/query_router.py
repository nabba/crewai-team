"""
query_router.py — Three-layer self-referential query detection.

Classifies user queries into:
    SELF_DIRECT      — "What are you?" / "Describe your architecture"
    SELF_OPERATION    — "How did you decide that?" / "What tools do you have?"
    SELF_REFLECTIVE   — "What are your weaknesses?" / "How have you evolved?"
    SELF_COMPARATIVE  — "How are you different from ChatGPT?"
    NOT_SELF          — Everything else → normal crew execution

Detection layers:
    1. Keyword patterns (regex, high precision)
    2. Pronoun + system-noun co-occurrence
    3. Semantic similarity against exemplar bank (ChromaDB)

IMMUTABLE — infrastructure-level module.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SelfRefType(str, Enum):
    SELF_DIRECT = "self_direct"
    SELF_OPERATION = "self_operation"
    SELF_REFLECTIVE = "self_reflective"
    SELF_COMPARATIVE = "self_comparative"
    NOT_SELF = "not_self"


@dataclass
class SelfRefClassification:
    query: str
    classification: SelfRefType
    confidence: float
    matched_signals: list[str]
    should_ground: bool

    @property
    def is_self_referential(self) -> bool:
        return self.classification != SelfRefType.NOT_SELF


# ── Layer 1: Keyword patterns ─────────────────────────────────────────────────

_DIRECT_PATTERNS = [
    r"\bwhat\s+are\s+you\b", r"\bwho\s+are\s+you\b",
    r"\btell\s+me\s+about\s+yourself\b", r"\bdescribe\s+yourself\b",
    r"\byour\s+(?:architecture|design|code|codebase|agents?|tools?|memory|config)\b",
    r"\bhow\s+(?:are\s+you|do\s+you)\s+(?:built|designed|structured|configured|work)\b",
    r"\byour\s+(?:name|version|capabilities|limitations?|purpose)\b",
    r"\bhow\s+many\s+agents?\b", r"\b(?:which|what)\s+(?:llm|model)s?\s+do\s+you\s+use\b",
    r"\byour\s+(?:llm|model)\s+cascade\b", r"\byour\s+(?:source\s+code|stack|setup)\b",
    r"\binspect\s+(?:yourself|your|the\s+system)\b", r"\byour\s+soul\b",
]

_OPERATION_PATTERNS = [
    r"\bwhy\s+did\s+you\s+(?:choose|decide|pick|select|use)\b",
    r"\bhow\s+did\s+you\s+(?:arrive|come|get|reach)\b",
    r"\bcan\s+you\s+(?:do|handle|process|manage)\b",
    r"\bare\s+you\s+(?:able|capable|equipped)\b",
    r"\byour\s+(?:last|recent|previous)\s+(?:task|execution|run|output)\b",
    r"\bwhat\s+(?:tools?|capabilities?)\s+do\s+you\s+have\b",
    r"\bhow\s+do\s+you\s+(?:remember|store|retrieve|think|reason|decide)\b",
]

_REFLECTIVE_PATTERNS = [
    r"\bwhat\s+are\s+your\s+(?:weaknesses?|strengths?|limitations?|flaws?)\b",
    r"\bhow\s+have\s+you\s+(?:evolved|changed|improved|grown)\b",
    r"\bwhat\s+would\s+you\s+(?:change|improve|fix)\s+about\s+yourself\b",
    r"\breflect\s+on\s+(?:yourself|your)\b", r"\bself[-_]?(?:assess|evaluate|critique)\b",
    r"\bare\s+you\s+(?:aware|conscious|sentient)\b", r"\byour\s+(?:identity|self[-_]?model)\b",
]

_COMPARATIVE_PATTERNS = [
    r"\bhow\s+(?:are\s+you|do\s+you)\s+(?:different|compare|stack\s+up)\b",
    r"\b(?:vs|versus|compared\s+to)\s+(?:chatgpt|gpt|claude|gemini|llama)\b",
    r"\bwhat\s+makes\s+you\s+(?:special|unique|different)\b",
    r"\bare\s+you\s+(?:better|worse|faster|smarter)\s+than\b",
]

_COMPILED = {
    SelfRefType.SELF_DIRECT: [re.compile(p, re.I) for p in _DIRECT_PATTERNS],
    SelfRefType.SELF_OPERATION: [re.compile(p, re.I) for p in _OPERATION_PATTERNS],
    SelfRefType.SELF_REFLECTIVE: [re.compile(p, re.I) for p in _REFLECTIVE_PATTERNS],
    SelfRefType.SELF_COMPARATIVE: [re.compile(p, re.I) for p in _COMPARATIVE_PATTERNS],
}

# ── Layer 2: Pronoun + system noun co-occurrence ──────────────────────────────

_SELF_PRONOUNS = re.compile(r"\b(?:you|your|yourself|you're|yours)\b", re.I)
_SYSTEM_NOUNS = re.compile(
    r"\b(?:agent|crew|tool|memory|model|llm|cascade|architecture|code|config|pipeline|"
    r"workflow|system|backend|database|embedding|vector|graph|commander|researcher|coder|"
    r"writer|self[-_]?improver|soul|principle|chromadb|mem0|neo4j|pgvector|ollama|"
    r"openrouter|crewai|homeostasis|self[-_]?awareness)\b", re.I,
)

# ── Layer 3: Semantic exemplars ───────────────────────────────────────────────

EXEMPLARS = [
    "What are you made of?", "Describe your architecture.", "How many agents do you have?",
    "What models do you use?", "Tell me about your memory system.",
    "How do you decide which LLM to use?", "What is your purpose?",
    "What tools are available to you?", "Explain how you work internally.",
    "How were you built?", "What can you do and what can't you do?",
    "Describe your agents and their roles.", "How do you store knowledge?",
    "How do you improve yourself?", "What are your safety constraints?",
    "Describe your LLM cascade strategy.", "What programming language are you written in?",
    "How does your evolution system work?", "What is your self-model?",
]

SIMILARITY_THRESHOLD = 0.55


# ── Router ────────────────────────────────────────────────────────────────────


class SelfRefRouter:
    """Three-layer self-referential query classifier."""

    def __init__(self, semantic_enabled: bool = True):
        self._semantic = semantic_enabled
        self._exemplar_col = None
        if semantic_enabled:
            self._init_exemplars()

    def _init_exemplars(self):
        try:
            import chromadb
            client = chromadb.HttpClient(host="chromadb", port=8000)
            self._exemplar_col = client.get_or_create_collection(
                "self_ref_exemplars", metadata={"hnsw:space": "cosine"},
            )
            self._exemplar_col.upsert(
                ids=[f"ex_{i}" for i in range(len(EXEMPLARS))],
                documents=EXEMPLARS,
            )
        except Exception:
            self._semantic = False

    def classify(self, query: str) -> SelfRefClassification:
        signals = []
        scores = {t: 0.0 for t in SelfRefType}

        # Layer 1: keyword patterns
        for ref_type, patterns in _COMPILED.items():
            for p in patterns:
                if p.search(query):
                    scores[ref_type] += 0.6
                    signals.append(f"keyword:{p.pattern[:30]}")
                    break

        # Layer 2: pronoun + system noun
        if _SELF_PRONOUNS.search(query) and _SYSTEM_NOUNS.search(query):
            best = max(
                [t for t in SelfRefType if t != SelfRefType.NOT_SELF],
                key=lambda t: scores[t],
            )
            if scores[best] > 0:
                scores[best] += 0.2
            else:
                scores[SelfRefType.SELF_DIRECT] += 0.4
            signals.append("pronoun+noun")

        # Layer 3: semantic similarity
        if self._semantic and self._exemplar_col:
            try:
                r = self._exemplar_col.query(query_texts=[query], n_results=3)
                if r["distances"] and r["distances"][0]:
                    sim = 1 - min(r["distances"][0]) / 2
                    if sim > SIMILARITY_THRESHOLD:
                        boost = min(sim * 0.5, 0.4)
                        best = max(
                            [t for t in SelfRefType if t != SelfRefType.NOT_SELF],
                            key=lambda t: scores[t],
                        )
                        if scores[best] > 0:
                            scores[best] += boost
                        else:
                            scores[SelfRefType.SELF_DIRECT] += boost
                        signals.append(f"semantic:{sim:.2f}")
            except Exception:
                pass

        # Decision
        best = max(scores, key=lambda t: scores[t])
        score = scores[best]

        if best == SelfRefType.NOT_SELF or score < 0.3:
            return SelfRefClassification(
                query=query, classification=SelfRefType.NOT_SELF,
                confidence=1.0 - max(scores[t] for t in SelfRefType if t != SelfRefType.NOT_SELF),
                matched_signals=signals, should_ground=False,
            )

        return SelfRefClassification(
            query=query, classification=best,
            confidence=min(score, 1.0),
            matched_signals=signals, should_ground=True,
        )
