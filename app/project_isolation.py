"""
project_isolation.py — Per-venture memory/knowledge/config namespacing.

Prevents context bleed between PLG, Archibal, and KaiCart.

Each project gets its own:
    - Mem0 namespace (memory reads/writes scoped)
    - ChromaDB collection prefix (knowledge isolation)
    - Instruction overrides (project-specific agent behavior)
    - Variables (project-specific settings)
    - History (compressed conversation state per project)

Directory structure:
    workspace/projects/
    ├── plg/
    │   ├── instructions/       # Agent behavior overrides for PLG
    │   ├── knowledge/          # PLG-specific docs
    │   ├── skills/             # PLG-specific skills
    │   ├── variables.env
    │   └── config.yaml
    ├── archibal/
    └── kaicart/

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECTS_DIR = Path("/app/workspace/projects")

# ── Project config ────────────────────────────────────────────────────────────

@dataclass
class ProjectConfig:
    """Configuration for a single project/venture."""
    name: str
    display_name: str = ""
    description: str = ""
    mem0_namespace: str = ""
    chroma_prefix: str = ""
    enabled: bool = True
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name
        if not self.mem0_namespace:
            self.mem0_namespace = f"project_{self.name}"
        if not self.chroma_prefix:
            self.chroma_prefix = f"{self.name}_"

@dataclass
class ProjectContext:
    """Active project context — passed through the agent execution chain."""
    project: ProjectConfig
    instructions: dict[str, str] = field(default_factory=dict)
    variables: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.project.name

    @property
    def mem0_namespace(self) -> str:
        return self.project.mem0_namespace

    @property
    def chroma_collection(self) -> str:
        return f"{self.project.chroma_prefix}knowledge"

    @property
    def skills_collection(self) -> str:
        return f"{self.project.chroma_prefix}skills"

    def get_agent_instructions(self, agent_name: str) -> str | None:
        """Get project-specific instructions for an agent."""
        return self.instructions.get(agent_name) or self.instructions.get(agent_name.lower())

    def get_variable(self, key: str) -> str | None:
        return self.variables.get(key)

# ── IMMUTABLE: project detection keywords ─────────────────────────────────────
#
# 2026-05-09 lesson: PLG used to claim "estonia"/"baltic"/"latvia"/
# "lithuania" as keywords. Anything Estonia-related auto-routed to
# PLG even when the topic was clearly forest research. Geographic
# regions are not project signals — PLG is a ticketing platform that
# happens to operate in those countries. Tightened to PLG-specific
# brand / domain terms only.
#
# Eesti mets (added 2026-05-09) covers Estonian forest research,
# biodiversity analysis, and remote-sensing work (Hansen GFC,
# Landsat, Sentinel, Earth Engine). The keyword list is deliberately
# broader than the project name itself — "forest age distribution"
# should match without the user typing "eesti mets" verbatim.

PROJECT_KEYWORDS: dict[str, list[str]] = {
    "plg": [
        # Brand / product terms.  Tightened 2026-05-09 to drop bare
        # geographies (estonia/baltic/latvia/lithuania) — those silently
        # hijacked any task mentioning the region. The list below is
        # the live-events-ticketing vocabulary; broaden carefully.
        "plg", "piletilevi", "live nation", "iabilet", "protect group",
        # Ticketing-as-a-product
        "ticketing", "ticketing platform", "ticket", "tickets",
        "ticket sales", "ticket platform", "ticket inventory",
        "box office", "event ticket", "event ticketing",
        "event registration", "ticket pricing", "seat map",
        # Live-event surfaces
        "venue", "venues", "concert", "concerts", "tour",
        "festival", "live event", "live events", "promoter",
    ],
    "archibal": [
        # Brand
        "archibal",
        # Provenance / content-clearance core vocabulary
        "clearance", "content clearance", "rights clearance",
        "provenance", "ai provenance", "digital provenance",
        "content provenance", "media provenance",
        # The C2PA / Content Credentials standard family
        "c2pa", "content credentials", "content authenticity",
        "media authentication", "image authentication",
        "video authentication",
        # Cryptographic plumbing
        "pki", "public key infrastructure", "signing certificate",
        # AI-generated-content detection / labelling
        "ai detection", "ai-generated", "ai generated",
        "synthetic media", "generative content",
        "watermark", "watermarking", "content watermark",
        "deepfake", "deepfake detection",
        # Closely-related industry terms
        "trust score", "media authentication", "fact-checking pipeline",
    ],
    "kaicart": [
        # Brand + key partner platforms
        "kaicart", "tiktok", "tiktok shop", "shopee", "lazada", "shopify",
        # Asian / SEA e-commerce vocabulary (the user's framing)
        "asian e-commerce", "asia e-commerce",
        "southeast asia", "sea market", "sea region", "sea ecommerce",
        "asean e-commerce", "asean commerce",
        # SEA country names — distinct from Estonia/Baltic, low chance
        # of cross-claim with other projects
        "thai", "thailand", "vietnam", "vietnamese",
        "indonesia", "indonesian", "philippines", "filipino",
        "malaysia", "malaysian", "singapore",
        # Generic e-commerce vocabulary (KaiCart is the only ecom
        # project; safe to claim broadly)
        "e-commerce", "ecommerce", "online seller", "online store",
        "online marketplace", "marketplace", "small seller",
        "smb seller", "merchant", "merchants",
        "live commerce", "social commerce",
        # Operational vocabulary that lands here
        "fulfilment", "fulfillment", "drop-shipping", "dropshipping",
    ],
    # Project name uses a space (matches the Postgres CP record
    # "Eesti mets" — control_plane.projects.get_by_name is case-
    # insensitive). The project_isolation directory is
    # ``workspace/projects/eesti_mets/`` (underscore — dirs avoid
    # spaces); the config.yaml's ``name`` field is what registers
    # under this key.
    "eesti mets": [
        # Direct project / domain
        "eesti mets", "estonian forest", "estonia forest", "eesti metsa",
        "metsa", "rmk", "kaitseala",
        # Forest concepts (any of these alone is a strong signal)
        "forest", "forests", "forestry", "deforestation",
        "reforestation", "afforestation",
        "forest age", "forest health", "forest cover", "forest loss",
        "tree cover", "canopy",
        # Trees + species commonly studied in Estonian forests
        "spruce", "pine", "birch", "aspen", "alder",
        # Remote sensing — the toolchain Eesti mets analyses use
        "satellite", "landsat", "sentinel", "modis",
        "hansen", "global forest change", "gfc",
        "earth engine", "google earth engine", "gee",
        # Conservation / biodiversity intersection
        "biodiversity", "conservation", "wildlife corridor",
        "protected area", "national park", "nature reserve",
        "carbon stock", "forest carbon",
    ],
}

# ── Project Manager ───────────────────────────────────────────────────────────

class ProjectManager:
    """Manages project isolation across ventures.

    Usage:
        pm = ProjectManager()
        pm.scan_projects()

        # Auto-detect project from task
        project_name = pm.detect_project("Analyze PLG Baltic market data")
        ctx = pm.activate("plg")

        # Scope memory
        memories = pm.search_memory(ctx, "ticket pricing strategy")

        # Scope knowledge
        knowledge = pm.search_knowledge(ctx, "refund policy")

        # Agent instruction overrides
        commander_extra = ctx.get_agent_instructions("commander")
    """

    def __init__(self, projects_dir: Path = PROJECTS_DIR):
        self._dir = projects_dir
        self._projects: dict[str, ProjectConfig] = {}
        self._active: ProjectContext | None = None
        self._lock = threading.Lock()

    def scan_projects(self) -> list[str]:
        """Scan projects directory and load configurations."""
        if not self._dir.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            # Bootstrap default projects
            self._bootstrap_defaults()

        found = []
        for entry in sorted(self._dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            config = self._load_config(entry)
            if config:
                self._projects[config.name] = config
                found.append(config.name)

        logger.info(f"project_isolation: found {len(found)} projects: {found}")
        return found

    def _bootstrap_defaults(self) -> None:
        """Create default project directories for PLG, Archibal, KaiCart."""
        defaults = {
            "plg": {
                "display_name": "Protect Group (PLG)",
                "description": "Live entertainment ticketing across Baltic states",
            },
            "archibal": {
                "display_name": "Archibal",
                "description": "Content authenticity and digital provenance platform",
            },
            "kaicart": {
                "display_name": "KaiCart",
                "description": "TikTok Shop management for Thai SMB sellers",
            },
        }

        for name, meta in defaults.items():
            project_dir = self._dir / name
            for subdir in ("instructions", "knowledge", "skills"):
                (project_dir / subdir).mkdir(parents=True, exist_ok=True)

            # Write config.yaml
            config_path = project_dir / "config.yaml"
            if not config_path.exists():
                try:
                    import yaml
                    config_data = {"name": name, **meta}
                    config_path.write_text(
                        yaml.dump(config_data, default_flow_style=False)
                    )
                except ImportError:
                    # Fallback without yaml
                    config_path.write_text(
                        f"name: {name}\n"
                        f"display_name: {meta['display_name']}\n"
                        f"description: {meta['description']}\n"
                    )

            # Write empty variables.env
            vars_path = project_dir / "variables.env"
            if not vars_path.exists():
                vars_path.write_text("# Project variables\n")

        logger.info("project_isolation: bootstrapped default projects (plg, archibal, kaicart)")

    def _load_config(self, project_dir: Path) -> ProjectConfig | None:
        """Load project config from directory."""
        config_file = project_dir / "config.yaml"
        if config_file.exists():
            try:
                import yaml
                data = yaml.safe_load(config_file.read_text()) or {}
                return ProjectConfig(
                    name=data.get("name", project_dir.name),
                    display_name=data.get("display_name", project_dir.name),
                    description=data.get("description", ""),
                    settings=data.get("settings", {}),
                )
            except Exception as e:
                logger.warning(f"project_isolation: error loading {project_dir.name}: {e}")

        # Fallback: use directory name
        return ProjectConfig(name=project_dir.name)

    def _resolve_name(self, project_name: str) -> str | None:
        """Resolve a project name case-insensitively to its canonical key.

        Returns the canonical (stored) name, or None if no match.
        """
        if project_name in self._projects:
            return project_name
        # Case-insensitive fallback
        lowered = project_name.lower()
        for key in self._projects:
            if key.lower() == lowered:
                return key
        return None

    def activate(self, project_name: str) -> ProjectContext:
        """Activate a project, loading its instructions and variables.

        Name matching is case-insensitive — "PLG", "plg", and "Plg" all
        resolve to the same project.
        """
        canonical = self._resolve_name(project_name)
        if canonical is None:
            self.scan_projects()
            canonical = self._resolve_name(project_name)
        if canonical is None:
            raise ValueError(
                f"Unknown project: {project_name}. "
                f"Available: {list(self._projects.keys())}"
            )

        config = self._projects[canonical]
        project_dir = self._dir / canonical

        # Load agent instruction overrides
        instructions: dict[str, str] = {}
        inst_dir = project_dir / "instructions"
        if inst_dir.exists():
            for md in inst_dir.glob("*.md"):
                instructions[md.stem] = md.read_text(encoding="utf-8")

        # Load variables
        variables = self._parse_env(project_dir / "variables.env")

        ctx = ProjectContext(
            project=config, instructions=instructions, variables=variables,
        )

        with self._lock:
            self._active = ctx

        logger.info(f"project_isolation: activated '{canonical}' "
                    f"(ns={ctx.mem0_namespace}, instructions={len(instructions)})")
        return ctx

    def deactivate(self) -> None:
        """Deactivate the current project."""
        with self._lock:
            if self._active:
                logger.info(f"project_isolation: deactivated '{self._active.name}'")
            self._active = None

    @property
    def active(self) -> ProjectContext | None:
        with self._lock:
            return self._active

    def detect_project(self, task_description: str) -> str | None:
        """Heuristic project detection from task text."""
        task_lower = task_description.lower()
        scores: dict[str, int] = {}

        for proj, keywords in PROJECT_KEYWORDS.items():
            if proj in self._projects:
                score = sum(1 for kw in keywords if kw in task_lower)
                if score > 0:
                    scores[proj] = score

        if not scores:
            return None
        return max(scores, key=scores.get)

    def search_memory(self, ctx: ProjectContext, query: str, n: int = 5) -> list:
        """Search Mem0 memory scoped to a project namespace."""
        try:
            from app.memory.mem0_manager import Mem0Manager
            manager = Mem0Manager()
            # Use project namespace as agent_id for scoping
            return manager.search_memory(query, agent_id=ctx.mem0_namespace, limit=n)
        except Exception:
            return []

    def search_knowledge(self, ctx: ProjectContext, query: str, n: int = 3) -> dict:
        """Search project-scoped ChromaDB knowledge collection."""
        try:
            from app.memory.chromadb_manager import get_client, embed
            client = get_client()
            collection = client.get_or_create_collection(ctx.chroma_collection)
            return collection.query(query_embeddings=[embed(query)], n_results=n)
        except Exception:
            return {"documents": [[]], "distances": [[]]}

    def store_memory(self, ctx: ProjectContext, text: str, metadata: dict | None = None) -> None:
        """Store a memory scoped to a project namespace."""
        try:
            from app.memory.mem0_manager import Mem0Manager
            manager = Mem0Manager()
            manager.store_memory(text, agent_id=ctx.mem0_namespace)
        except Exception:
            pass

    def store_knowledge(self, ctx: ProjectContext, doc_id: str, content: str,
                        metadata: dict | None = None) -> None:
        """Store knowledge in a project-scoped ChromaDB collection."""
        try:
            from app.memory.chromadb_manager import get_client, embed
            client = get_client()
            collection = client.get_or_create_collection(ctx.chroma_collection)
            collection.upsert(
                ids=[doc_id], documents=[content],
                metadatas=[metadata or {}],
                embeddings=[embed(content)],
            )
        except Exception:
            pass

    def list_projects(self) -> list[dict]:
        """List all known projects."""
        return [
            {
                "name": p.name,
                "display_name": p.display_name,
                "description": p.description,
                "enabled": p.enabled,
            }
            for p in self._projects.values()
        ]

    def create_project(self, name: str, display_name: str = "",
                       description: str = "") -> ProjectConfig:
        """Create a new project."""
        project_dir = self._dir / name
        for subdir in ("instructions", "knowledge", "skills"):
            (project_dir / subdir).mkdir(parents=True, exist_ok=True)

        config = ProjectConfig(
            name=name, display_name=display_name or name,
            description=description,
        )

        try:
            import yaml
            with open(project_dir / "config.yaml", "w") as f:
                yaml.dump({
                    "name": config.name,
                    "display_name": config.display_name,
                    "description": config.description,
                }, f, default_flow_style=False)
        except ImportError:
            (project_dir / "config.yaml").write_text(
                f"name: {name}\ndisplay_name: {display_name or name}\n"
                f"description: {description}\n"
            )

        (project_dir / "variables.env").write_text("# Project variables\n")
        self._projects[name] = config

        # Auto-create per-business knowledge base collection.
        try:
            from app.knowledge_base.business_store import get_registry
            get_registry().create_store(name)
            logger.info(f"project_isolation: created KB collection for '{name}'")
        except Exception as e:
            logger.debug(f"project_isolation: KB creation for '{name}' deferred: {e}")

        logger.info(f"project_isolation: created project '{name}'")
        return config

    @staticmethod
    def _parse_env(path: Path) -> dict[str, str]:
        """Parse a simple key=value env file."""
        if not path.exists():
            return {}
        result = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip().strip('"').strip("'")
        return result

    def format_status(self) -> str:
        """Human-readable project status."""
        active = self._active
        lines = [
            f"📁 Project Isolation ({len(self._projects)} projects)",
            f"   Active: {active.name if active else 'none'}",
            "",
        ]
        for p in self._projects.values():
            marker = "→" if active and active.name == p.name else " "
            lines.append(f"   {marker} {p.display_name} ({p.name})")
            if p.description:
                lines.append(f"     {p.description[:80]}")
        return "\n".join(lines)

# ── Module-level singleton ───────────────────────────────────────────────────

_manager: ProjectManager | None = None

def get_manager() -> ProjectManager:
    """Get or create the singleton project manager."""
    global _manager
    if _manager is None:
        _manager = ProjectManager()
        _manager.scan_projects()
    return _manager
