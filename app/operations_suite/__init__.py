"""
operations_suite — Unified access to deployment and operational modules.

Provides a single import point for the 5 operations-related modules.

Usage:
    from app.operations_suite import create_manifest, restore_from_manifest
    from app.operations_suite import get_monitor, evaluate_health
    from app.operations_suite import SelfHealer
    from app.operations_suite import run_reference_suite
    from app.operations_suite import SandboxRunner
"""

# Version manifest (composite rollback)
from app.version_manifest import (
    create_manifest, get_current_manifest, restore_from_manifest,
    rollback_to_previous, list_manifests, cleanup_old_snapshots,
)

# Health monitoring
from app.health_monitor import (
    get_monitor, evaluate_health, record_interaction,
    HealthMonitor, InteractionMetrics, HealthAlert,
)

# Self-healing
from app.healing import SelfHealer

# Reference task suite
from app.reference_tasks import (
    REFERENCE_TASKS, run_reference_suite, verify_suite_integrity,
)

# Sandbox runner
from app.sandbox_runner import SandboxRunner
