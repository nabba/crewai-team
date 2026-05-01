"""subia.tsal — Phase 13: Technical Self-Awareness Layer.

AndrusAI knows what it IS technically through continuous self-inspection,
not static declaration. Five guarantees:

  1. Discovered (probing, not hand-written)
  2. Current (refresh schedule, stale-tracked)
  3. Wiki-native (writes wiki/self/ pages via injected WikiWriteTool)
  4. SubIA-wired (feeds self_state, homeostasis.overload, predictor)
  5. Actionable (Self-Improver consults it for evolution feasibility)

The package builds on the EXISTING `app.subia.tsal.inspect_tools`
six-tool inspection layer. TSAL adds what was missing:
  - Host hardware probing (HostProfile)
  - Resource monitoring with derived pressures
  - Wiki-native page generation
  - Operating-principles inference (Tier-1 LLM, weekly)
  - Evolution-feasibility checking (Self-Improver gate)
  - SubIA closed-loop wiring

Every external dependency (LLM, wiki writer) is injected — the package
is unit-testable without OpenRouter / wiki / external services.
"""
from .probers import HostProber, HostProfile, ResourceMonitor, ResourceState
from .inspectors import (
    CodeAnalyst, CodebaseProfile,
    ComponentDiscovery, ComponentInventory,
    CascadeProfile,
)
from .self_model import TechnicalSelfModel
from .generators import (
    PageGenerator,
    WikiWriterAdapter,
)
from .operating_principles import infer_operating_principles
from .evolution_feasibility import (
    EvolutionProposal,
    check_evolution_feasibility,
    FeasibilityReport,
)
from .refresh import register_tsal_jobs

__all__ = [
    "HostProber", "HostProfile", "ResourceMonitor", "ResourceState",
    "CodeAnalyst", "CodebaseProfile",
    "ComponentDiscovery", "ComponentInventory", "CascadeProfile",
    "TechnicalSelfModel",
    "PageGenerator", "WikiWriterAdapter",
    "infer_operating_principles",
    "EvolutionProposal", "check_evolution_feasibility", "FeasibilityReport",
    "register_tsal_jobs",
]
