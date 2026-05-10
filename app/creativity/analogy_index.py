"""Cross-domain analogy index — Hofstadter-style structural retrieval.

The premise (cf. Hofstadter, *Surfaces and Essences*): creativity *is*
analogy. The system has rich knowledge across many domains in its KBs;
what's missing is a primitive that, given a problem, retrieves
structurally-similar problems from *other* domains.

Examples of structural matches the index aims to surface:

  problem: "user keeps re-asking the same question after a month"
    structurally like:
      - control theory: feedback loop with delay
      - ecology: predator-prey dynamics with population lag
      - economics: stock-flow misperception
      - addiction recovery: relapse cycle

  problem: "pattern_learner can't find a runbook for new errors"
    structurally like:
      - immunology: novel antigen recognition
      - cybersecurity: zero-day signature detection
      - language acquisition: unknown-word inference

This module is the *primitive*. The index is a JSONL store of
``AnalogyEntry`` records keyed by structural signature; query is
embedding-cosine over the structure_description text. The brainstorm
subsystem and reverie engine consume :func:`query_analogies` to
surface unrelated-domain analogues.

Storage layout::

    workspace/creativity/analogy_index.jsonl
        one ``AnalogyEntry`` per line, append-only

The store is small (curated by operator + populated by future
indexers) — JSONL is fine; no need for ChromaDB unless we hit 10k+
entries. Hash-trick embedding (already in app/utils/hash_embedding)
is fast + deterministic + good enough for "any reasonable match".

Public surface:

  add_entry(entry)
  query_analogies(problem_text, top_k=5, min_similarity=0.15)
  list_all()

Master switch: ``ANALOGY_INDEX_ENABLED`` (default ``true``). When
false, queries return empty + writes are silently skipped — i.e.
brainstorm consumers degrade gracefully.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.utils.hash_embedding import cosine, embed

logger = logging.getLogger(__name__)


_DEFAULT_PATH = Path("/app/workspace/creativity/analogy_index.jsonl")
_path_override: Path | None = None
_LOCK = threading.RLock()


def _enabled() -> bool:
    return os.getenv("ANALOGY_INDEX_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _resolve_path() -> Path:
    return _path_override if _path_override else _DEFAULT_PATH


@dataclass(frozen=True)
class DomainExample:
    """One concrete instance of an abstract structure in a specific domain."""

    domain: str
    title: str
    summary: str
    citation: str = ""


@dataclass(frozen=True)
class AnalogyEntry:
    """One abstract structure cataloged with examples across domains."""

    id: str
    structure_signature: str  # short label — "feedback_loop_with_delay"
    structure_description: str  # paragraph — what the abstract pattern is
    domain_examples: list[DomainExample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "structure_signature": self.structure_signature,
            "structure_description": self.structure_description,
            "domain_examples": [asdict(d) for d in self.domain_examples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalogyEntry":
        examples = [
            DomainExample(
                domain=d["domain"],
                title=d["title"],
                summary=d["summary"],
                citation=d.get("citation", ""),
            )
            for d in data.get("domain_examples", [])
        ]
        return cls(
            id=data["id"],
            structure_signature=data["structure_signature"],
            structure_description=data["structure_description"],
            domain_examples=examples,
        )


@dataclass(frozen=True)
class AnalogyMatch:
    """A query result: the entry plus its similarity score."""

    entry: AnalogyEntry
    similarity: float


def add_entry(entry: AnalogyEntry, *, path: Path | str | None = None) -> bool:
    """Append an entry to the index. Idempotent — adding an entry with the
    same id as an existing one DOES write a new line (latest wins on read).
    Returns True on success.
    """
    if not _enabled():
        return False
    target = Path(path) if path else _resolve_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _LOCK, open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        return True
    except OSError as exc:
        logger.debug("analogy_index: append failed: %s", exc)
        return False


def list_all(*, path: Path | str | None = None) -> list[AnalogyEntry]:
    """Read every entry. Last-write-wins on duplicate id."""
    target = Path(path) if path else _resolve_path()
    if not target.exists():
        return []
    by_id: dict[str, AnalogyEntry] = {}
    try:
        with _LOCK, open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    entry = AnalogyEntry.from_dict(raw)
                except (KeyError, TypeError):
                    continue
                by_id[entry.id] = entry
    except OSError as exc:
        logger.debug("analogy_index: read failed: %s", exc)
        return []
    return list(by_id.values())


def query_analogies(
    problem_text: str,
    *,
    top_k: int = 5,
    min_similarity: float = 0.15,
    exclude_domains: set[str] | None = None,
    path: Path | str | None = None,
) -> list[AnalogyMatch]:
    """Return up to ``top_k`` analogy entries whose structure_description
    is most similar to ``problem_text``, scored by hash-trick cosine.

    ``exclude_domains`` filters out entries whose ALL domain_examples
    fall in the excluded set — useful when the caller already knows
    the problem's home domain and wants only OTHER-domain analogues.

    Returns matches above ``min_similarity`` only, sorted descending.
    """
    if not _enabled() or not problem_text.strip():
        return []

    entries = list_all(path=path)
    if not entries:
        return []

    exclude = exclude_domains or set()
    query_emb = embed(problem_text)
    matches: list[AnalogyMatch] = []
    for entry in entries:
        # Skip entries whose every example is in an excluded domain.
        if exclude and entry.domain_examples and all(
            ex.domain in exclude for ex in entry.domain_examples
        ):
            continue
        sim = cosine(query_emb, embed(entry.structure_description))
        if sim < min_similarity:
            continue
        matches.append(AnalogyMatch(entry=entry, similarity=sim))

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:top_k]


def _reset_for_tests(path: Path | None = None) -> None:
    global _path_override
    with _LOCK:
        _path_override = path
