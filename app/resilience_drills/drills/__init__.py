"""Drill implementations. Each module registers a DrillSpec + runner
at import. Importing this package brings all four drills into the
registry."""
from __future__ import annotations

# Import order does not matter — registration is idempotent.
from app.resilience_drills.drills import backup_restore  # noqa: F401
from app.resilience_drills.drills import embedding_migration  # noqa: F401
from app.resilience_drills.drills import secret_rotation  # noqa: F401
from app.resilience_drills.drills import kill_the_gateway  # noqa: F401
# PROGRAM §51 Q16 Theme 2 — vendor-independence drill (5th drill).
from app.resilience_drills.drills import vendor_independence  # noqa: F401
