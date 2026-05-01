"""CodeAnalyst + ComponentDiscovery — adapters over existing inspect_tools.

Both are THIN WRAPPERS: they consume the mature six-tool layer at
`app.subia.tsal.inspect_tools` and add only the missing TSAL
contributions (architectural-pattern detection, dependency graph,
Ollama probe, Wiki probe, Cascade profile builder).

Failure-mode policy: every external dependency is wrapped; missing
backends produce empty profiles with `available=False`, never raise.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Code Analyst
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CodebaseProfile:
    project_root: str = ""
    total_modules: int = 0
    total_lines: int = 0
    total_classes: int = 0
    total_functions: int = 0
    packages: list = field(default_factory=list)
    modules: list = field(default_factory=list)
    tools: list = field(default_factory=list)         # discovered registered tools
    agents: list = field(default_factory=list)        # discovered agent roles
    dependencies: dict = field(default_factory=dict)  # module → [imports]
    patterns_detected: list = field(default_factory=list)
    config_files: list = field(default_factory=list)
    analyzed_at: str = ""


class CodeAnalyst:
    """Adapter over inspect_tools.inspect_codebase + inspect_agents."""

    def __init__(
        self,
        inspect_codebase_fn: Optional[Callable[[str], dict]] = None,
        inspect_agents_fn: Optional[Callable[[], dict]] = None,
        project_root: str = ".",
    ) -> None:
        if inspect_codebase_fn is None or inspect_agents_fn is None:
            try:
                from app.subia.tsal import inspect_tools as _it
                inspect_codebase_fn = inspect_codebase_fn or _it.inspect_codebase
                inspect_agents_fn = inspect_agents_fn or _it.inspect_agents
            except Exception:
                inspect_codebase_fn = inspect_codebase_fn or (lambda scope="full": {})
                inspect_agents_fn = inspect_agents_fn or (lambda: {})
        self._inspect_codebase = inspect_codebase_fn
        self._inspect_agents = inspect_agents_fn
        self.project_root = project_root

    def analyze(self) -> CodebaseProfile:
        prof = CodebaseProfile(
            project_root=str(self.project_root),
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            cb = self._inspect_codebase(scope="full") or {}
        except Exception:
            cb = {}
        prof.total_modules = int(cb.get("total_modules", 0))
        prof.total_lines = int(cb.get("total_lines", 0))
        prof.total_classes = int(cb.get("total_classes", 0))
        prof.total_functions = int(cb.get("total_functions", 0))
        prof.packages = list(cb.get("packages", []))
        prof.modules = list(cb.get("modules", []))
        prof.dependencies = self._build_dependencies(prof.modules)
        prof.tools = self._discover_tools(prof.modules)
        prof.patterns_detected = self._detect_patterns(prof)
        try:
            ag = self._inspect_agents() or {}
        except Exception:
            ag = {}
        prof.agents = list(ag.get("agents", []) or ag.get("roles", []))
        return prof

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _build_dependencies(modules: list) -> dict:
        """Module name → [imports]. Tolerates summary-mode (strings)."""
        deps: dict[str, list] = {}
        for m in modules:
            if isinstance(m, dict):
                name = m.get("path", "")
                imports = m.get("imports", []) or []
                if name:
                    deps[name] = imports
        return deps

    @staticmethod
    def _discover_tools(modules: list) -> list:
        """Find classes inheriting from BaseTool."""
        tools: list = []
        for m in modules:
            if not isinstance(m, dict):
                continue
            for cls in m.get("classes", []) or []:
                # AST class entries can be plain names (strings) when
                # inspect_codebase returns summary form.
                if isinstance(cls, str):
                    if "Tool" in cls and cls != "BaseTool":
                        tools.append({"name": cls, "module": m.get("path", "")})
        return tools

    @staticmethod
    def _detect_patterns(profile: CodebaseProfile) -> list:
        names = [m.get("path", "") if isinstance(m, dict) else str(m)
                 for m in profile.modules]
        patterns = []
        if any("subia" in n for n in names):
            n_subia = sum(1 for n in names if "subia/" in n)
            patterns.append(f"SubIA kernel ({n_subia} modules)")
        if any("cascade" in n or "/llm/" in n for n in names):
            patterns.append("multi-tier LLM cascade")
        if any("hook" in n or "lifecycle" in n for n in names):
            patterns.append("lifecycle hooks")
        if any("safety" in n or "dgm" in n or "guardian" in n for n in names):
            patterns.append("DGM safety invariant")
        if any("subia/connections/" in n for n in names):
            patterns.append("inter-system bridges (SIA Part II)")
        if any("subia/idle/" in n for n in names):
            patterns.append("idle-time consciousness work scheduler")
        if any("subia/probes/" in n for n in names):
            patterns.append("auto-regenerated consciousness scorecard")
        if any("subia/tsal/" in n for n in names):
            patterns.append("technical self-awareness layer (TSAL)")
        if any("wiki" in n for n in names):
            patterns.append("wiki-native knowledge subsystem")
        return patterns


# ─────────────────────────────────────────────────────────────────────
# Component Discovery
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ChromaDBState:
    available: bool = False
    collections: list = field(default_factory=list)
    total_documents: int = 0


@dataclass
class Neo4jState:
    available: bool = False
    node_count: int = 0
    relation_count: int = 0
    relation_types: list = field(default_factory=list)
    node_labels: list = field(default_factory=list)


@dataclass
class Mem0State:
    curated_available: bool = False
    full_available: bool = False
    curated_episode_count: int = 0
    full_record_count: int = 0


@dataclass
class OllamaState:
    available: bool = False
    models_installed: list = field(default_factory=list)
    model_loaded: str = ""


@dataclass
class WikiState:
    total_pages: int = 0
    pages_by_section: dict = field(default_factory=dict)
    disk_usage_mb: float = 0.0
    wiki_root: str = ""


@dataclass
class CascadeProfile:
    tiers: list = field(default_factory=list)
    current_default_tier: str = ""


@dataclass
class ComponentInventory:
    chromadb: ChromaDBState = field(default_factory=ChromaDBState)
    neo4j: Neo4jState = field(default_factory=Neo4jState)
    mem0: Mem0State = field(default_factory=Mem0State)
    ollama: OllamaState = field(default_factory=OllamaState)
    wiki: WikiState = field(default_factory=WikiState)
    cascade: CascadeProfile = field(default_factory=CascadeProfile)
    firecrawl_available: bool = False
    subia_active: bool = False
    discovered_at: str = ""


class ComponentDiscovery:
    """Adapter over inspect_tools.inspect_memory + new Ollama/Wiki probes."""

    def __init__(
        self,
        inspect_memory_fn: Optional[Callable[[str], dict]] = None,
        ollama_lister: Optional[Callable[[], list]] = None,
        ollama_loaded_detector: Optional[Callable[[], str]] = None,
        wiki_root: str = "wiki",
        cascade_tier_config: Optional[list] = None,
    ) -> None:
        if inspect_memory_fn is None:
            try:
                from app.subia.tsal import inspect_tools as _it
                inspect_memory_fn = _it.inspect_memory
            except Exception:
                inspect_memory_fn = lambda backend="all": {}
        self._inspect_memory = inspect_memory_fn
        self._ollama_lister = ollama_lister or self._default_ollama_lister
        self._ollama_loaded_detector = ollama_loaded_detector or (lambda: "")
        self._wiki_root = wiki_root
        self._cascade_tier_config = cascade_tier_config or []

    def discover(self) -> ComponentInventory:
        inv = ComponentInventory(
            discovered_at=datetime.now(timezone.utc).isoformat()
        )
        try:
            mem = self._inspect_memory(backend="all") or {}
        except Exception:
            mem = {}
        # ChromaDB
        cdb = mem.get("chromadb", {}) or {}
        if "error" not in cdb:
            inv.chromadb.available = True
            details = cdb.get("details", {}) or {}
            for name, info in details.items():
                count = (info or {}).get("count", 0)
                inv.chromadb.collections.append({"name": name, "count": count})
                inv.chromadb.total_documents += int(count or 0)
        # Neo4j
        n4 = mem.get("neo4j", {}) or {}
        if "error" not in n4 and "nodes" in n4:
            inv.neo4j.available = True
            inv.neo4j.node_count = int(n4.get("nodes", 0))
            inv.neo4j.relation_count = int(n4.get("relationships", 0))
        # Mem0 lives on PostgreSQL — surface availability via pg
        pg = mem.get("postgresql", {}) or {}
        if "error" not in pg and "tables" in pg:
            inv.mem0.curated_available = any(
                "memories" in t for t in (pg.get("table_list") or [])
            )
            inv.mem0.full_available = inv.mem0.curated_available

        # Ollama
        try:
            installed = self._ollama_lister() or []
            inv.ollama.available = True
            inv.ollama.models_installed = installed
            inv.ollama.model_loaded = self._ollama_loaded_detector() or ""
        except Exception:
            inv.ollama.available = False

        # Wiki
        inv.wiki = self._probe_wiki(self._wiki_root)

        # Cascade
        inv.cascade = self._build_cascade(inv.ollama, self._cascade_tier_config)
        return inv

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _default_ollama_lister() -> list:
        try:
            import subprocess
            r = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return []
            out: list = []
            for line in (r.stdout or "").strip().splitlines()[1:]:
                parts = line.split()
                if parts:
                    out.append({
                        "name": parts[0],
                        "size": parts[2] if len(parts) > 2 else "unknown",
                    })
            return out
        except Exception:
            return []

    @staticmethod
    def _probe_wiki(wiki_root: str) -> WikiState:
        state = WikiState(wiki_root=str(wiki_root))
        root = Path(wiki_root)
        if not root.exists():
            return state
        total_size = 0
        for section_dir in root.iterdir():
            if not section_dir.is_dir():
                continue
            pages = [p for p in section_dir.rglob("*.md")
                     if p.name != "index.md"]
            if pages:
                state.pages_by_section[section_dir.name] = len(pages)
                state.total_pages += len(pages)
                for p in pages:
                    try:
                        total_size += p.stat().st_size
                    except OSError:
                        pass
        state.disk_usage_mb = round(total_size / (1024 * 1024), 2)
        return state

    @staticmethod
    def _build_cascade(ollama: OllamaState, configured_tiers: list) -> CascadeProfile:
        prof = CascadeProfile()
        if ollama.available and ollama.models_installed:
            prof.tiers.append({
                "name": "tier_1_local",
                "provider": "ollama",
                "model": ollama.models_installed[0].get("name", "unknown"),
                "available": True,
                "cost_per_1k_tokens": 0.0,
                "latency_estimate_ms": 500,
            })
        for t in configured_tiers or []:
            prof.tiers.append({
                "name": t.get("name", "unknown"),
                "provider": t.get("provider", "unknown"),
                "model": t.get("model", "unknown"),
                "available": bool(t.get("api_key")),
                "cost_per_1k_tokens": t.get("cost", 0.0),
                "latency_estimate_ms": t.get("latency", 1000),
            })
        if prof.tiers:
            prof.current_default_tier = prof.tiers[0]["name"]
        return prof
