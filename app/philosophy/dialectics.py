"""
philosophy/dialectics.py — Graph-based dialectical argument structure.

Encodes philosophical arguments as a directed graph in Neo4j:

    (Claim) -[:COUNTERED_BY]-> (CounterClaim) -[:SYNTHESIZED_INTO]-> (Synthesis)

This enables retrieval patterns that vector search alone cannot:
  - "Find the counter-argument to X"
  - "Show the dialectical chain for topic Y"
  - "What tensions exist between Stoic and Utilitarian views on Z?"

Uses Neo4j via the existing Mem0 connection config.  Gracefully degrades
to empty results if Neo4j is unavailable.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Neo4j connection ────────────────────────────────────────────────────────

_driver: Any | None = None
_driver_failed: bool = False


def _get_driver():
    """Get a Neo4j driver, reusing the Mem0 connection config."""
    global _driver, _driver_failed
    if _driver is not None:
        return _driver
    if _driver_failed:
        return None

    try:
        from app.config import get_settings
        s = get_settings()

        url = s.mem0_neo4j_url or ""
        user = s.mem0_neo4j_user or "neo4j"
        password = s.mem0_neo4j_password.get_secret_value()

        if not (url and password):
            _driver_failed = True
            logger.info("philosophy.dialectics: Neo4j URL/password not configured — dialectics disabled")
            return None

        import neo4j
        _driver = neo4j.GraphDatabase.driver(url, auth=(user, password))
        _driver.verify_connectivity()
        logger.info("philosophy.dialectics: connected to Neo4j at %s", url)
        return _driver
    except Exception as exc:
        _driver_failed = True
        logger.info("philosophy.dialectics: Neo4j unavailable — dialectics disabled: %s", exc)
        return None


def _node_id(text: str) -> str:
    """Deterministic short hash for a text passage."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Public API ──────────────────────────────────────────────────────────────

class DialecticalGraph:
    """Manages philosophical argument structures in Neo4j."""

    def store_argument(
        self,
        claim_text: str,
        counter_claim_text: str,
        synthesis_text: str | None = None,
        source: str = "",
        tradition_a: str = "",
        tradition_b: str = "",
    ) -> bool:
        """Store a claim → counter-claim (→ synthesis) relationship.

        Returns True if stored, False if Neo4j unavailable.
        """
        driver = _get_driver()
        if driver is None:
            return False

        claim_id = _node_id(claim_text)
        counter_id = _node_id(counter_claim_text)

        cypher = """
        MERGE (c:PhilClaim {id: $claim_id})
        SET c.text = $claim_text, c.source = $source, c.tradition = $tradition_a
        MERGE (cc:PhilCounterClaim {id: $counter_id})
        SET cc.text = $counter_text, cc.source = $source, cc.tradition = $tradition_b
        MERGE (c)-[:COUNTERED_BY]->(cc)
        """
        params: dict[str, Any] = {
            "claim_id": claim_id,
            "claim_text": claim_text[:2000],
            "counter_id": counter_id,
            "counter_text": counter_claim_text[:2000],
            "source": source,
            "tradition_a": tradition_a,
            "tradition_b": tradition_b,
        }

        if synthesis_text:
            synth_id = _node_id(synthesis_text)
            cypher += """
            MERGE (s:PhilSynthesis {id: $synth_id})
            SET s.text = $synth_text, s.source = $source
            MERGE (cc)-[:SYNTHESIZED_INTO]->(s)
            """
            params["synth_id"] = synth_id
            params["synth_text"] = synthesis_text[:2000]

        try:
            with driver.session() as session:
                session.run(cypher, params)
            return True
        except Exception as exc:
            logger.warning("dialectics.store_argument failed: %s", exc)
            return False

    def find_counter_arguments(
        self, claim_query: str, n: int = 3
    ) -> list[dict]:
        """Find counter-arguments to a claim.

        Strategy: vector-search the philosophy collection for the claim,
        then follow COUNTERED_BY edges in Neo4j to find counter-claims.
        """
        driver = _get_driver()
        if driver is None:
            return []

        # Step 1: Find matching claims via vector search.
        try:
            from app.philosophy.vectorstore import get_store
            store = get_store()
            matches = store.query(query_text=claim_query, n_results=5)
        except Exception:
            return []

        if not matches:
            return []

        # Build node IDs from matched texts.
        claim_ids = [_node_id(m["text"]) for m in matches]

        # Step 2: Follow graph edges.
        cypher = """
        MATCH (c:PhilClaim)-[:COUNTERED_BY]->(cc:PhilCounterClaim)
        WHERE c.id IN $claim_ids
        OPTIONAL MATCH (cc)-[:SYNTHESIZED_INTO]->(s:PhilSynthesis)
        RETURN cc.text AS counter_text, cc.tradition AS tradition,
               cc.source AS source, s.text AS synthesis_text
        LIMIT $limit
        """
        try:
            with driver.session() as session:
                result = session.run(cypher, {"claim_ids": claim_ids, "limit": n})
                return [
                    {
                        "counter_claim": record["counter_text"],
                        "tradition": record.get("tradition", ""),
                        "source": record.get("source", ""),
                        "synthesis": record.get("synthesis_text"),
                    }
                    for record in result
                ]
        except Exception as exc:
            logger.warning("dialectics.find_counter_arguments failed: %s", exc)
            return []

    def find_dialectical_chain(self, topic: str, n: int = 5) -> list[dict]:
        """Find claim → counter → synthesis chains related to a topic.

        Returns chains as dicts with claim, counter_claim, synthesis.
        """
        driver = _get_driver()
        if driver is None:
            return []

        try:
            from app.philosophy.vectorstore import get_store
            store = get_store()
            matches = store.query(query_text=topic, n_results=5)
        except Exception:
            return []

        if not matches:
            return []

        claim_ids = [_node_id(m["text"]) for m in matches]

        cypher = """
        MATCH (c:PhilClaim)-[:COUNTERED_BY]->(cc:PhilCounterClaim)
        WHERE c.id IN $claim_ids
        OPTIONAL MATCH (cc)-[:SYNTHESIZED_INTO]->(s:PhilSynthesis)
        RETURN c.text AS claim, c.tradition AS claim_tradition,
               cc.text AS counter_claim, cc.tradition AS counter_tradition,
               s.text AS synthesis
        LIMIT $limit
        """
        try:
            with driver.session() as session:
                result = session.run(cypher, {"claim_ids": claim_ids, "limit": n})
                return [
                    {
                        "claim": record["claim"],
                        "claim_tradition": record.get("claim_tradition", ""),
                        "counter_claim": record["counter_claim"],
                        "counter_tradition": record.get("counter_tradition", ""),
                        "synthesis": record.get("synthesis"),
                    }
                    for record in result
                ]
        except Exception as exc:
            logger.warning("dialectics.find_dialectical_chain failed: %s", exc)
            return []


# ── Singleton ───────────────────────────────────────────────────────────────
_graph: DialecticalGraph | None = None


def get_graph() -> DialecticalGraph:
    """Lazy-singleton accessor."""
    global _graph
    if _graph is None:
        _graph = DialecticalGraph()
    return _graph


# ── Decision panel (PROGRAM §43 — Q5.1) ─────────────────────────────────────
#
# `find_counter_arguments` and `find_dialectical_chain` above are
# *retrieval* primitives — they answer "what's in the KB?". The panel
# below is a *decision* surface — it answers "for THIS question, surface
# the perspective tensions across N traditions so the operator/system
# can see the unresolved spaces before deciding."
#
# The panel does not produce a single answer. It returns structured
# tensions. Synthesis is only included when the KB already contains one
# (claim → counter → synthesis chain materialised by `store_argument`).
# Otherwise the tension is left unresolved — and *that's the point*.
# Unresolved tensions can be filed into the Q4.1 tensions store via
# `app/sentience_experiments/panel_bridge.py:file_unresolved_tensions`.


@dataclass
class PerspectiveTension:
    """One tradition's stance on a question — claim, counter, optional synthesis."""

    tradition: str
    claim: str
    counter_claim: str | None
    synthesis: str | None
    source: str
    confidence: float          # KB-coverage proxy in [0.0, 1.0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PanelResult:
    """Aggregated panel reading on a single question."""

    question: str
    perspectives: list[PerspectiveTension] = field(default_factory=list)
    unresolved_tensions: list[str] = field(default_factory=list)
    coverage: float = 0.0       # fraction of requested traditions that returned material
    consulted_at: str = ""
    cache_hit: bool = False
    skipped_reason: str | None = None  # set when panel was skipped (disabled / capped / KB-empty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "perspectives": [p.to_dict() for p in self.perspectives],
            "unresolved_tensions": list(self.unresolved_tensions),
            "coverage": float(self.coverage),
            "consulted_at": self.consulted_at,
            "cache_hit": bool(self.cache_hit),
            "skipped_reason": self.skipped_reason,
        }


# Default traditions to consult when the caller passes ``None``. Chosen
# for broad ethical coverage rather than completeness; the panel auto-
# falls back to whichever subset the KB actually contains.
_DEFAULT_TRADITIONS: tuple[str, ...] = (
    "Stoicism",
    "Utilitarianism",
    "Virtue ethics",
    "Pragmatism",
    "Kantian",
)


# Cost cap — see plan §5. The panel is consulted at most once per
# (question, traditions) tuple per cache window; subsequent calls reuse
# the persisted result.
_PANEL_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def _panel_enabled() -> bool:
    """Master switch read. Default ON.

    Two ways to disable:
      * ``PHILOSOPHY_PANEL_ENABLED=false`` env var
      * runtime_settings.philosophy_panel_enabled = False
    """
    env = os.environ.get("PHILOSOPHY_PANEL_ENABLED")
    if env is not None and env.lower() in ("0", "false", "no", "off"):
        return False
    try:
        from app.runtime_settings import get_philosophy_panel_enabled
        return get_philosophy_panel_enabled()
    except Exception:
        return True


def _panel_cache_path() -> Path:
    """Cache file. Honors WORKSPACE_ROOT override; never raises."""
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "philosophy" / "panel_cache.jsonl"
    except Exception:
        return Path("/app/workspace/philosophy/panel_cache.jsonl")


def _cache_key(question: str, traditions: tuple[str, ...]) -> str:
    """Deterministic key over (normalized-question, sorted-traditions)."""
    norm_q = (question or "").strip().lower()
    norm_t = ",".join(sorted(t.strip().lower() for t in traditions if t))
    raw = f"{norm_q}\x00{norm_t}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_cache(key: str) -> PanelResult | None:
    """Read the most-recent cache entry for this key within TTL.

    Cache is an append-only JSONL — we walk it backwards. Cheap because
    the panel is consulted at low cadence (~1-per-amendment / day).
    Returns None on any failure."""
    path = _panel_cache_path()
    if not path.exists():
        return None
    cutoff_ts = time.time() - _PANEL_CACHE_TTL_SECONDS
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    # Walk newest-first.
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("key") != key:
            continue
        ts_str = row.get("consulted_at") or ""
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            continue
        if ts < cutoff_ts:
            return None  # cache entry too old; nothing newer for this key
        # Hit — rehydrate the result.
        try:
            payload = row["result"]
            return PanelResult(
                question=payload["question"],
                perspectives=[
                    PerspectiveTension(**p) for p in payload.get("perspectives", [])
                ],
                unresolved_tensions=list(payload.get("unresolved_tensions", [])),
                coverage=float(payload.get("coverage", 0.0)),
                consulted_at=payload.get("consulted_at", ""),
                cache_hit=True,
                skipped_reason=payload.get("skipped_reason"),
            )
        except (KeyError, TypeError):
            return None
    return None


def _write_cache(key: str, result: PanelResult) -> None:
    """Append the result to the cache. Failure-isolated."""
    path = _panel_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The cache_hit flag is a runtime annotation; don't persist it
        # as True (the read path always restores it as True on a hit).
        payload = result.to_dict()
        payload["cache_hit"] = False
        row = {"key": key, "consulted_at": result.consulted_at, "result": payload}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        logger.debug("dialectics.panel: cache write failed", exc_info=True)


def consult_panel(
    question: str,
    traditions: list[str] | None = None,
    *,
    max_perspectives: int = 5,
    use_cache: bool = True,
) -> PanelResult:
    """Consult a multi-tradition perspective panel on a question.

    PROGRAM §43 (Q5.1) — Decision-panel layer over the dialectical
    primitives. Returns structured tensions, never prose.

    Parameters
    ----------
    question : str
        The decision being weighed. e.g. "Should governance.py
        SAFETY_FLOOR be relaxed from 0.85 to 0.80?"
    traditions : list[str] | None
        Which traditions to consult. ``None`` → top-5 by KB coverage
        (defaults to _DEFAULT_TRADITIONS).
    max_perspectives : int
        Cap on perspectives returned. Bounds operator-surface noise.
    use_cache : bool
        When True (default) reuse persisted results within 7d TTL.

    Returns
    -------
    PanelResult
        Structured tensions. Always returns — never raises. Empty
        ``perspectives`` + non-None ``skipped_reason`` signals the
        panel could not run (disabled / KB empty / Neo4j down).

    Failure modes
    -------------
    * Master switch off → ``skipped_reason="disabled"``, empty perspectives
    * Neo4j unavailable → ``skipped_reason="neo4j_unavailable"``
    * Philosophy KB empty → ``skipped_reason="kb_empty"``
    * Any other exception → ``skipped_reason="error:<type>"``

    Never raises. Caller code must handle empty-result case (e.g. by
    treating absence-of-panel as no-additional-evidence rather than
    blocking the decision).
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    q = (question or "").strip()
    if not q:
        return PanelResult(
            question="",
            consulted_at=now_iso,
            skipped_reason="empty_question",
        )

    if not _panel_enabled():
        return PanelResult(
            question=q,
            consulted_at=now_iso,
            skipped_reason="disabled",
        )

    requested = tuple(t.strip() for t in (traditions or _DEFAULT_TRADITIONS) if t.strip())
    if not requested:
        requested = _DEFAULT_TRADITIONS

    # Cache lookup.
    key = _cache_key(q, requested)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    # Build the panel — one perspective per requested tradition.
    perspectives: list[PerspectiveTension] = []
    unresolved: list[str] = []
    try:
        graph = get_graph()
    except Exception:
        return PanelResult(
            question=q,
            consulted_at=now_iso,
            skipped_reason="error:graph_init",
        )

    # Probe via the existing dialectical chain (composes claim →
    # counter → synthesis). For each requested tradition, filter the
    # results to that tradition's stance — when none match, leave the
    # slot empty and mark the question as unresolved for that tradition.
    try:
        chains = graph.find_dialectical_chain(q, n=max(10, max_perspectives * 3))
    except Exception:
        chains = []

    if not chains:
        return PanelResult(
            question=q,
            consulted_at=now_iso,
            skipped_reason="kb_empty",
            coverage=0.0,
        )

    # Bucket chains by tradition (claim_tradition or counter_tradition match).
    available_traditions: set[str] = set()
    for c in chains:
        if c.get("claim_tradition"):
            available_traditions.add(c["claim_tradition"])
        if c.get("counter_tradition"):
            available_traditions.add(c["counter_tradition"])

    for trad in requested:
        matched = None
        trad_lower = trad.lower()
        for c in chains:
            ct = (c.get("claim_tradition") or "").lower()
            cct = (c.get("counter_tradition") or "").lower()
            if trad_lower in (ct, cct):
                matched = c
                break
        if matched is None:
            unresolved.append(f"{trad}: no stance found in KB for {q!r}")
            continue
        # Determine which side of the dialectic matches this tradition.
        if (matched.get("claim_tradition") or "").lower() == trad_lower:
            claim_text = matched.get("claim") or ""
            counter_text = matched.get("counter_claim") or None
        else:
            claim_text = matched.get("counter_claim") or ""
            counter_text = matched.get("claim") or None
        synth = matched.get("synthesis")
        if not synth:
            unresolved.append(
                f"{trad}: claim present but no synthesis with counter — "
                f"unresolved tension"
            )
        perspectives.append(PerspectiveTension(
            tradition=trad,
            claim=claim_text[:1000],
            counter_claim=(counter_text[:1000] if counter_text else None),
            synthesis=(synth[:1000] if synth else None),
            source=matched.get("source", ""),
            confidence=1.0,  # KB-present = confidence 1.0; absence handled above
        ))
        if len(perspectives) >= max_perspectives:
            break

    coverage = (
        len([p for p in perspectives if p.claim])
        / max(1, len(requested))
    )
    result = PanelResult(
        question=q,
        perspectives=perspectives,
        unresolved_tensions=unresolved,
        coverage=round(coverage, 3),
        consulted_at=now_iso,
        cache_hit=False,
    )
    if use_cache:
        _write_cache(key, result)
    return result


def format_panel_for_operator(panel: PanelResult, *, max_chars: int = 800) -> str:
    """Render a PanelResult as a compact markdown block for Signal
    alerts and operator review surfaces. Returns empty string when the
    panel was skipped or has nothing to surface — never clutters."""
    if not panel or panel.skipped_reason:
        return ""
    if not panel.perspectives and not panel.unresolved_tensions:
        return ""
    lines = [
        f"🧭 Philosophy panel (coverage {panel.coverage:.0%}):",
    ]
    for p in panel.perspectives:
        claim_short = (p.claim or "")[:120]
        synth_short = (p.synthesis or "")[:120] if p.synthesis else ""
        if synth_short:
            lines.append(f"   • {p.tradition}: {claim_short} → {synth_short}")
        else:
            lines.append(f"   • {p.tradition}: {claim_short} (no synthesis)")
    if panel.unresolved_tensions:
        lines.append("   Unresolved:")
        for u in panel.unresolved_tensions[:3]:
            lines.append(f"     – {u[:160]}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars - 3] + "..."
    return out
