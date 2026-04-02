"""
grounding.py — Grounding protocol for self-referential queries.

When a self-referential query is detected:
    1. Gather context from self-inspection tools + self_knowledge RAG
    2. Build a constrained system prompt forcing answers from actual data
    3. Post-process to detect ungrounded generic-AI claims

The key insight: without grounding, the LLM answers "What are you?" from
training data — it talks about AI *in general*. With grounding, it answers
from actual introspection of its own code, config, runtime, and history.

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.self_awareness.query_router import SelfRefClassification, SelfRefType

logger = logging.getLogger(__name__)


# ── IMMUTABLE: Grounding system prompt ────────────────────────────────────────

GROUNDING_PROMPT = """You are answering a question about yourself — your own architecture, code, configuration, capabilities, history, or design.

CRITICAL RULES:
1. Answer ONLY from the grounded context below — from actual inspection of your own codebase, config, runtime, and self-model.
2. Do NOT use general knowledge about AI systems, LLMs, or multi-agent frameworks.
3. Do NOT say "As an AI language model" or "As a large language model" — you are a specific system with specific architecture.
4. If the context doesn't cover the question, say: "I don't have introspective access to that aspect of myself yet."
5. State facts as direct self-knowledge: "I have five agents" not "According to my configuration..."
6. Prefix inferences beyond the context with [INFERENCE].
7. Use first person: "I am", "my architecture", "I use", "my agents".

YOUR IDENTITY:
{self_model_summary}

YOUR CURRENT STATE:
{runtime_state}

GROUNDED CONTEXT (from code/config inspection):
{grounded_context}

RELEVANT SELF-KNOWLEDGE (from codebase embeddings):
{rag_context}
"""

COMPARATIVE_ADDENDUM = """
For this comparative question, you MAY use general knowledge about the OTHER system.
But all claims about YOURSELF must come from the grounded context above.
Clearly distinguish self-knowledge (grounded) from knowledge about others (general).
"""

REFLECTIVE_ADDENDUM = """
You are reflecting on yourself. Draw on:
  - Grounded context for factual self-knowledge
  - Your values (SOUL.md / philosophical layer)
Be honest and specific. Ground reflection in actual data, not generic AI platitudes.
"""

# Phrases that indicate the LLM is using training-data self-descriptions
# instead of grounded self-knowledge
_GENERIC_PHRASES = [
    "as an ai language model", "as a large language model",
    "i don't have feelings", "i don't have consciousness",
    "i'm just a tool", "i was trained on", "my training data",
    "i don't have access to the internet", "as an artificial intelligence",
    "i'm a text-based ai", "i don't have personal experiences",
    "my knowledge cutoff",
]


# ── Context gathering ─────────────────────────────────────────────────────────


@dataclass
class GroundedContext:
    self_model: str = ""
    runtime_state: dict = field(default_factory=dict)
    tool_outputs: dict = field(default_factory=dict)
    rag_results: list = field(default_factory=list)
    classification: Optional[SelfRefClassification] = None


class GroundingProtocol:
    """Gathers grounded context and builds constrained prompts."""

    def gather_context(self, classification: SelfRefClassification) -> GroundedContext:
        """Run relevant inspection tools based on query type."""
        from app.self_awareness.inspect_tools import ALL_INSPECT_TOOLS

        ctx = GroundedContext(classification=classification)

        # Always get self-model
        try:
            data = ALL_INSPECT_TOOLS["inspect_self_model"]()
            ctx.self_model = data.get("content", "Self-model not yet generated.")[:2000]
        except Exception as e:
            ctx.self_model = f"Unavailable: {e}"

        # Always get runtime
        try:
            ctx.runtime_state = ALL_INSPECT_TOOLS["inspect_runtime"](section="process")
        except Exception:
            pass

        # Type-specific context
        q = classification.query.lower()

        if classification.classification in (SelfRefType.SELF_DIRECT, SelfRefType.SELF_REFLECTIVE):
            # Structural questions → agents, config, codebase
            try:
                ctx.tool_outputs["agents"] = ALL_INSPECT_TOOLS["inspect_agents"]()
            except Exception as e:
                ctx.tool_outputs["agents"] = f"Error: {e}"

            if any(k in q for k in ("config", "llm", "model", "cascade", "memory", "setup", "stack")):
                try:
                    ctx.tool_outputs["config"] = ALL_INSPECT_TOOLS["inspect_config"](section="all")
                except Exception as e:
                    ctx.tool_outputs["config"] = f"Error: {e}"

            if any(k in q for k in ("code", "architecture", "structure", "built", "design", "module")):
                try:
                    ctx.tool_outputs["codebase"] = ALL_INSPECT_TOOLS["inspect_codebase"](scope="summary")
                except Exception as e:
                    ctx.tool_outputs["codebase"] = f"Error: {e}"

            if any(k in q for k in ("memory", "remember", "store", "database", "vector", "knowledge")):
                try:
                    ctx.tool_outputs["memory"] = ALL_INSPECT_TOOLS["inspect_memory"](backend="all")
                except Exception as e:
                    ctx.tool_outputs["memory"] = f"Error: {e}"

        elif classification.classification == SelfRefType.SELF_OPERATION:
            # Operational questions → runtime, agents, recent tasks
            try:
                ctx.tool_outputs["runtime"] = ALL_INSPECT_TOOLS["inspect_runtime"](section="all")
            except Exception as e:
                ctx.tool_outputs["runtime"] = f"Error: {e}"
            try:
                ctx.tool_outputs["agents"] = ALL_INSPECT_TOOLS["inspect_agents"]()
            except Exception as e:
                ctx.tool_outputs["agents"] = f"Error: {e}"

        elif classification.classification == SelfRefType.SELF_COMPARATIVE:
            # Comparative → need our own structure info
            try:
                ctx.tool_outputs["agents"] = ALL_INSPECT_TOOLS["inspect_agents"]()
                ctx.tool_outputs["config"] = ALL_INSPECT_TOOLS["inspect_config"](section="summary")
            except Exception:
                pass

        # RAG from self_knowledge collection
        try:
            import chromadb
            client = chromadb.HttpClient(host="chromadb", port=8000)
            col = client.get_or_create_collection("self_knowledge")
            if col.count() > 0:
                results = col.query(query_texts=[classification.query], n_results=5)
                if results["documents"] and results["documents"][0]:
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                        ctx.rag_results.append({
                            "document": doc[:1000],
                            "metadata": meta,
                        })
        except Exception:
            pass

        return ctx

    def build_system_prompt(self, ctx: GroundedContext) -> str:
        """Build the constrained system prompt from gathered context."""
        # Format tool outputs
        grounded_parts = []
        for name, output in ctx.tool_outputs.items():
            text = output if isinstance(output, str) else json.dumps(output, indent=2, default=str)
            grounded_parts.append(f"--- {name} ---\n{text[:3000]}")
        grounded = "\n\n".join(grounded_parts) or "No tool context gathered."

        # Format RAG results
        rag_parts = []
        for r in ctx.rag_results:
            src = r.get("metadata", {}).get("source_file", "unknown")
            rag_parts.append(f"[Source: {src}]\n{r.get('document', '')[:1000]}")
        rag = "\n\n".join(rag_parts) or "No codebase context found."

        prompt = GROUNDING_PROMPT.format(
            self_model_summary=ctx.self_model[:2000] or "Not yet generated.",
            runtime_state=json.dumps(ctx.runtime_state, indent=2, default=str)[:500],
            grounded_context=grounded,
            rag_context=rag,
        )

        # Add type-specific addendum
        if ctx.classification:
            if ctx.classification.classification == SelfRefType.SELF_COMPARATIVE:
                prompt += COMPARATIVE_ADDENDUM
            elif ctx.classification.classification == SelfRefType.SELF_REFLECTIVE:
                prompt += REFLECTIVE_ADDENDUM

        return prompt

    def post_process(self, response: str) -> dict:
        """Detect ungrounded generic-AI claims in the response."""
        ungrounded = [p for p in _GENERIC_PHRASES if p in response.lower()]
        return {
            "text": response,
            "ungrounded_detected": ungrounded,
            "grounded": len(ungrounded) == 0,
        }
