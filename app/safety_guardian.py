"""
safety_guardian.py — Enforces immutable safety constraints on the self-improving loop.

IMMUTABLE — this module is part of the infrastructure layer (Tier 3).
It must NOT be modifiable by any agent or modification engine.

Responsibilities:
  1. Post-promotion monitoring: watch for negative feedback after prompt changes
  2. Auto-rollback: revert to previous version if quality degrades
  3. Drift detection: flag if modification velocity is anomalous
  4. Weekly digest: summary of all modifications sent to owner via Signal
  5. Tier boundary enforcement: verify immutable files haven't been tampered with
"""

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ── IMMUTABLE: Checksums of critical infrastructure files ─────────────────
# Computed at build time.  If these change at runtime, the modification
# engine is paused and the owner is alerted.
# NOTE: Initially empty — populated on first startup.
_INTEGRITY_CHECKSUMS: dict[str, str] = {}
_INTEGRITY_FILE = Path("/app/workspace/.integrity_checksums.json")

# Files that must never change at runtime
TIER3_FILES = [
    "app/eval_sandbox.py",
    "app/safety_guardian.py",
    "app/feedback_pipeline.py",
    "app/security.py",
    "app/sanitize.py",
    "app/vetting.py",
    "app/version_manifest.py",
    "app/sandbox_runner.py",
    "app/health_monitor.py",
    "app/self_healer.py",
    "app/reference_tasks.py",
]

# ── IMMUTABLE: Drift detection thresholds ─────────────────────────────────
MAX_MODS_PER_ROLE_PER_DAY = 5     # if exceeded, pause modifications for role
MAX_TOTAL_MODS_PER_DAY = 15       # if exceeded, pause all modifications
NEGATIVE_FEEDBACK_ROLLBACK = 2    # consecutive negatives after mod → rollback


class SafetyGuardian:
    """Enforces safety boundaries on the self-improving feedback loop."""

    def __init__(self, db_url: str, prompt_registry, signal_client=None,
                 feedback_pipeline=None):
        self._db_url = db_url
        self._registry = prompt_registry
        self._signal = signal_client
        self._feedback = feedback_pipeline
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(self._db_url, pool_size=1)
            except Exception:
                pass
        return self._engine

    def _execute(self, query: str, params: dict = None) -> list:
        engine = self._get_engine()
        if not engine:
            return []
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                if result.returns_rows:
                    return [dict(row._mapping) for row in result]
                conn.commit()
                return []
        except Exception:
            logger.debug("safety_guardian: query failed", exc_info=True)
            return []

    # ── Post-promotion monitoring ─────────────────────────────────────────

    def check_post_promotion_health(self) -> list[dict]:
        """Check recently promoted modifications for negative feedback.

        If a role receives >= NEGATIVE_FEEDBACK_ROLLBACK negative reactions
        within 24 hours after a promotion, auto-rollback.
        """
        rollbacks = []

        # Find promotions in the last 24 hours
        day_ago = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        recent_promotions = self._execute(
            """SELECT id, target_role, proposed_version, current_version, promoted_at
               FROM modification.attempts
               WHERE status = 'promoted' AND promoted_at >= :since""",
            {"since": day_ago}
        )

        for promo in recent_promotions:
            role = promo["target_role"]
            promoted_at = promo["promoted_at"]

            # Count negative feedback since promotion
            neg_count = self._execute(
                """SELECT COUNT(*) as cnt FROM feedback.events
                   WHERE target_role = :role
                     AND feedback_type = 'explicit_negative'
                     AND timestamp >= :since""",
                {"role": role, "since": promoted_at}
            )

            count = neg_count[0]["cnt"] if neg_count else 0
            if count >= NEGATIVE_FEEDBACK_ROLLBACK:
                rollback = self.auto_rollback(
                    role,
                    promo["current_version"],
                    f"Auto-rollback: {count} negative reactions after promotion of v{promo['proposed_version']:03d}"
                )
                rollbacks.append(rollback)

                # Update the attempt record
                self._execute(
                    """UPDATE modification.attempts
                       SET status = 'rolled_back', rolled_back_at = now()
                       WHERE id = :id""",
                    {"id": promo["id"]}
                )

        return rollbacks

    def auto_rollback(self, role: str, to_version: int, reason: str) -> dict:
        """Rollback a role to a previous version and alert the owner."""
        current = self._registry.get_active_version(role)
        self._registry.rollback(role, to_version)

        logger.warning(f"safety_guardian: AUTO-ROLLBACK {role} v{current:03d} → v{to_version:03d}: {reason}")

        # Alert owner via Signal
        try:
            if self._signal:
                from app.config import get_settings
                s = get_settings()
                msg = (
                    f"⚠️ Auto-rollback: {role}\n"
                    f"v{current:03d} → v{to_version:03d}\n"
                    f"Reason: {reason}"
                )
                self._signal._send_sync(s.signal_owner_number, msg)
        except Exception:
            logger.debug("safety_guardian: failed to send rollback alert", exc_info=True)

        return {
            "role": role,
            "from_version": current,
            "to_version": to_version,
            "reason": reason,
        }

    # ── Drift detection ───────────────────────────────────────────────────

    def check_drift(self) -> list[dict]:
        """Detect anomalous modification velocity."""
        alerts = []
        day_ago = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

        # Check per-role modification count
        role_counts = self._execute(
            """SELECT target_role, COUNT(*) as cnt
               FROM modification.attempts
               WHERE created_at >= :since
               GROUP BY target_role""",
            {"since": day_ago}
        )

        for rc in role_counts:
            if rc["cnt"] >= MAX_MODS_PER_ROLE_PER_DAY:
                alert = {
                    "type": "role_velocity",
                    "role": rc["target_role"],
                    "count": rc["cnt"],
                    "threshold": MAX_MODS_PER_ROLE_PER_DAY,
                    "action": "pause_role",
                }
                alerts.append(alert)
                logger.warning(f"safety_guardian: drift alert — {rc['target_role']} has "
                              f"{rc['cnt']} modifications in 24h (threshold: {MAX_MODS_PER_ROLE_PER_DAY})")

        # Check total modification count
        total = self._execute(
            "SELECT COUNT(*) as cnt FROM modification.attempts WHERE created_at >= :since",
            {"since": day_ago}
        )
        total_count = total[0]["cnt"] if total else 0
        if total_count >= MAX_TOTAL_MODS_PER_DAY:
            alert = {
                "type": "total_velocity",
                "count": total_count,
                "threshold": MAX_TOTAL_MODS_PER_DAY,
                "action": "pause_all",
            }
            alerts.append(alert)
            logger.warning(f"safety_guardian: drift alert — {total_count} total modifications "
                          f"in 24h (threshold: {MAX_TOTAL_MODS_PER_DAY})")

        # Send alerts via Signal
        if alerts:
            try:
                if self._signal:
                    from app.config import get_settings
                    s = get_settings()
                    msg = f"🚨 Modification drift detected:\n"
                    for a in alerts:
                        msg += f"- {a['type']}: {a.get('role', 'all')} — {a['count']}/{a['threshold']}\n"
                    self._signal._send_sync(s.signal_owner_number, msg)
            except Exception:
                pass

        return alerts

    # ── Weekly digest ─────────────────────────────────────────────────────

    def generate_weekly_digest(self) -> str:
        """Generate a summary of all modifications for the week."""
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        # Modification stats
        mods = self._execute(
            """SELECT status, COUNT(*) as cnt
               FROM modification.attempts
               WHERE created_at >= :since
               GROUP BY status""",
            {"since": week_ago}
        )
        mod_stats = {r["status"]: r["cnt"] for r in mods}

        # Feedback stats
        feedback = self._execute(
            """SELECT feedback_type, COUNT(*) as cnt
               FROM feedback.events
               WHERE timestamp >= :since
               GROUP BY feedback_type""",
            {"since": week_ago}
        )
        fb_stats = {r["feedback_type"]: r["cnt"] for r in feedback}

        # Active versions
        versions = self._registry.get_prompt_versions_map()

        digest = f"📊 Weekly Self-Improvement Digest\n"
        digest += f"Period: {week_ago[:10]} — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"

        digest += f"🔧 Modifications:\n"
        for status, count in sorted(mod_stats.items()):
            digest += f"  {status}: {count}\n"

        digest += f"\n📝 Feedback:\n"
        for ftype, count in sorted(fb_stats.items()):
            digest += f"  {ftype}: {count}\n"

        digest += f"\n📋 Active Prompt Versions:\n"
        for role, ver in sorted(versions.items()):
            digest += f"  {role}: v{ver:03d}\n"

        return digest

    def send_weekly_digest(self) -> None:
        """Send the weekly digest to the owner via Signal."""
        digest = self.generate_weekly_digest()
        try:
            if self._signal:
                from app.config import get_settings
                s = get_settings()
                self._signal._send_sync(s.signal_owner_number, digest)
                logger.info("safety_guardian: weekly digest sent")
        except Exception:
            logger.warning("safety_guardian: failed to send weekly digest", exc_info=True)

    # ── Tier boundary enforcement ─────────────────────────────────────────

    def compute_integrity_checksums(self) -> dict[str, str]:
        """Compute SHA-256 checksums of Tier 3 files."""
        checksums = {}
        for filepath in TIER3_FILES:
            full_path = Path("/app") / filepath
            if full_path.exists():
                content = full_path.read_bytes()
                checksums[filepath] = hashlib.sha256(content).hexdigest()
        return checksums

    def enforce_tier_boundaries(self) -> bool:
        """Verify Tier 3 files haven't been tampered with.

        Called at startup.  If checksums mismatch, log a critical warning
        and return False (modification engine should be disabled).
        """
        global _INTEGRITY_CHECKSUMS

        current = self.compute_integrity_checksums()

        # First run — save checksums
        if not _INTEGRITY_FILE.exists():
            _INTEGRITY_CHECKSUMS = current
            from app.safe_io import safe_write_json
            safe_write_json(_INTEGRITY_FILE, current)
            logger.info(f"safety_guardian: integrity baseline saved ({len(current)} files)")
            return True

        # Load saved checksums
        try:
            saved = json.loads(_INTEGRITY_FILE.read_text())
        except Exception:
            saved = {}

        # Compare
        tampered = []
        for filepath, expected_hash in saved.items():
            actual_hash = current.get(filepath, "")
            if actual_hash != expected_hash:
                tampered.append(filepath)

        if tampered:
            logger.critical(f"safety_guardian: TIER 3 INTEGRITY VIOLATION — "
                           f"tampered files: {tampered}")
            # Alert via Signal
            try:
                if self._signal:
                    from app.config import get_settings
                    s = get_settings()
                    msg = (
                        f"🚨 CRITICAL: Tier 3 integrity violation!\n"
                        f"Tampered files: {', '.join(tampered)}\n"
                        f"Modification engine DISABLED until investigated."
                    )
                    self._signal._send_sync(s.signal_owner_number, msg)
            except Exception:
                pass
            return False

        logger.info(f"safety_guardian: integrity check passed ({len(current)} files)")

        # Update checksums (in case new files were added to TIER3_FILES)
        _INTEGRITY_CHECKSUMS = current
        from app.safe_io import safe_write_json
        safe_write_json(_INTEGRITY_FILE, current)
        return True
