"""
skill_library.py — Upgraded skill management with manifests, test tracking, and confidence decay.

Skills are verified, executable Python code (or knowledge) stored with manifests
that track confidence, test results, dependencies, source attribution, and usage.

Replaces flat workspace/skills/*.md with a structured library:
  workspace/atlas/skills/
    ├── apis/{name}/           # API client code + tests
    ├── patterns/{name}/       # Reusable patterns (auth, retry, etc.)
    ├── recipes/{name}/        # Composed solutions
    └── learned/{source}/{name}/  # From YouTube or experimentation

Each skill directory contains:
  - manifest.json   (metadata, confidence, dependencies, test results)
  - code.py         (the executable skill code)
  - test_code.py    (test suite for the skill)
  - README.md       (human-readable description)

IMMUTABLE — infrastructure-level module.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILLS_DIR = Path("/app/workspace/atlas/skills")

# ── Skill categories ──────────────────────────────────────────────────────────

CATEGORIES = ("apis", "patterns", "recipes", "learned")

# ── Confidence decay parameters (IMMUTABLE) ──────────────────────────────────

# Confidence decays 0.5% per day since last verification
CONFIDENCE_DECAY_RATE = 0.005
# Minimum confidence floor — never drops below this from decay alone
CONFIDENCE_FLOOR = 0.30
# Boost per successful usage (capped at 0.10)
USAGE_SUCCESS_BOOST = 0.02
# Maximum boost from usage
MAX_USAGE_BOOST = 0.10
# Source reliability weights
SOURCE_WEIGHTS = {
    "openapi_spec": 0.95,
    "official_docs": 0.85,
    "official_sdk": 0.90,
    "youtube_tutorial": 0.60,
    "blog_post": 0.50,
    "trial_and_error": 0.70,
    "community_example": 0.55,
}


# ── Manifest schema ──────────────────────────────────────────────────────────


@dataclass
class SkillManifest:
    """Metadata for a single skill in the library."""
    skill_id: str = ""             # e.g. "apis/notion/client"
    name: str = ""                 # Human-readable name
    category: str = ""             # apis | patterns | recipes | learned
    version: str = "1.0.0"
    language: str = "python"
    description: str = ""

    # Confidence scoring
    confidence: float = 0.0        # 0.0 - 1.0, computed from source + tests + decay
    source_type: str = ""          # primary knowledge source
    source_urls: list[str] = field(default_factory=list)

    # Dependencies
    dependencies: list[str] = field(default_factory=list)  # pip packages
    auth_pattern: str = ""         # which auth pattern this uses (if API)

    # Test results
    tests_total: int = 0
    tests_passing: int = 0
    tests_failing: int = 0
    last_tested: str = ""          # ISO timestamp

    # Usage tracking
    usage_count: int = 0
    usage_success_count: int = 0
    last_used: str = ""

    # Lifecycle
    created_at: str = ""
    last_verified: str = ""
    code_hash: str = ""            # SHA-256 of code.py

    # Known limitations
    known_limitations: list[str] = field(default_factory=list)

    # Tags for search
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SkillManifest":
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def effective_confidence(self) -> float:
        """Compute current confidence with time decay and usage boost."""
        base = SOURCE_WEIGHTS.get(self.source_type, 0.50)

        # Test coverage boost
        if self.tests_total > 0 and self.tests_passing == self.tests_total:
            base += 0.10
        elif self.tests_total > 0 and self.tests_failing > 0:
            base -= 0.10

        # Time decay since last verification
        if self.last_verified:
            try:
                verified = datetime.fromisoformat(self.last_verified)
                days = (datetime.now(timezone.utc) - verified).days
                decay = min(days * CONFIDENCE_DECAY_RATE, base - CONFIDENCE_FLOOR)
                base -= max(0, decay)
            except Exception:
                pass

        # Usage success boost
        if self.usage_count > 0:
            success_rate = self.usage_success_count / self.usage_count
            boost = min(success_rate * USAGE_SUCCESS_BOOST * self.usage_count, MAX_USAGE_BOOST)
            base += boost

        return max(CONFIDENCE_FLOOR, min(1.0, base))


# ── Skill Library ────────────────────────────────────────────────────────────


class SkillLibrary:
    """Manages the structured skill library with manifests and confidence tracking."""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self._dir = skills_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        for cat in CATEGORIES:
            (self._dir / cat).mkdir(exist_ok=True)
        self._index: dict[str, SkillManifest] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Scan all skill directories and build in-memory index."""
        for manifest_path in self._dir.rglob("manifest.json"):
            try:
                data = json.loads(manifest_path.read_text())
                manifest = SkillManifest.from_dict(data)
                if manifest.skill_id:
                    manifest.confidence = manifest.effective_confidence()
                    self._index[manifest.skill_id] = manifest
            except Exception:
                logger.debug(f"skill_library: failed to load {manifest_path}", exc_info=True)

        logger.info(f"skill_library: indexed {len(self._index)} skills")

    def register_skill(
        self,
        skill_id: str,
        name: str,
        category: str,
        code: str,
        description: str = "",
        source_type: str = "trial_and_error",
        source_urls: list[str] | None = None,
        dependencies: list[str] | None = None,
        auth_pattern: str = "",
        test_code: str = "",
        tags: list[str] | None = None,
    ) -> SkillManifest:
        """Register a new skill in the library.

        Creates the skill directory with code, tests, manifest, and README.
        """
        if category not in CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {CATEGORIES}")

        skill_dir = self._dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:32]

        manifest = SkillManifest(
            skill_id=skill_id,
            name=name,
            category=category,
            description=description,
            source_type=source_type,
            source_urls=source_urls or [],
            dependencies=dependencies or [],
            auth_pattern=auth_pattern,
            tags=tags or [],
            created_at=now,
            last_verified=now,
            code_hash=code_hash,
        )

        # Write files
        (skill_dir / "code.py").write_text(code)
        from app.safe_io import safe_write_json; safe_write_json(skill_dir / "manifest.json", manifest.to_dict())

        if test_code:
            (skill_dir / "test_code.py").write_text(test_code)

        readme = f"# {name}\n\n{description}\n\n**Source:** {source_type}\n"
        if dependencies:
            readme += f"\n**Dependencies:** {', '.join(dependencies)}\n"
        (skill_dir / "README.md").write_text(readme)

        # Update index
        manifest.confidence = manifest.effective_confidence()
        self._index[skill_id] = manifest

        logger.info(f"skill_library: registered '{name}' as {skill_id} "
                     f"(confidence={manifest.confidence:.2f})")
        return manifest

    def get_skill(self, skill_id: str) -> Optional[SkillManifest]:
        """Look up a skill by ID."""
        return self._index.get(skill_id)

    def get_skill_code(self, skill_id: str) -> Optional[str]:
        """Read the code for a skill."""
        code_path = self._dir / skill_id / "code.py"
        if code_path.exists():
            return code_path.read_text()
        return None

    def search(
        self,
        query: str = "",
        category: str = "",
        min_confidence: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[SkillManifest]:
        """Search skills by query, category, confidence, or tags."""
        results = []
        query_lower = query.lower()
        tag_set = set(t.lower() for t in (tags or []))

        for manifest in self._index.values():
            # Filter by category
            if category and manifest.category != category:
                continue

            # Filter by confidence
            if manifest.effective_confidence() < min_confidence:
                continue

            # Filter by tags
            if tag_set and not tag_set.intersection(t.lower() for t in manifest.tags):
                continue

            # Text search across name, description, tags
            if query_lower:
                searchable = (
                    f"{manifest.name} {manifest.description} "
                    f"{' '.join(manifest.tags)} {manifest.skill_id}"
                ).lower()
                if query_lower not in searchable:
                    continue

            results.append(manifest)

        # Sort by confidence (highest first)
        results.sort(key=lambda m: m.effective_confidence(), reverse=True)
        return results

    def record_usage(self, skill_id: str, success: bool) -> None:
        """Record that a skill was used (success or failure)."""
        manifest = self._index.get(skill_id)
        if not manifest:
            return

        manifest.usage_count += 1
        if success:
            manifest.usage_success_count += 1
        manifest.last_used = datetime.now(timezone.utc).isoformat()
        manifest.confidence = manifest.effective_confidence()

        # Persist
        manifest_path = self._dir / skill_id / "manifest.json"
        if manifest_path.exists():
            from app.safe_io import safe_write_json; safe_write_json(manifest_path, manifest.to_dict())

    def record_test_results(
        self, skill_id: str, total: int, passing: int, failing: int
    ) -> None:
        """Record test results for a skill."""
        manifest = self._index.get(skill_id)
        if not manifest:
            return

        manifest.tests_total = total
        manifest.tests_passing = passing
        manifest.tests_failing = failing
        manifest.last_tested = datetime.now(timezone.utc).isoformat()
        manifest.last_verified = manifest.last_tested
        manifest.confidence = manifest.effective_confidence()

        # Persist
        manifest_path = self._dir / skill_id / "manifest.json"
        if manifest_path.exists():
            from app.safe_io import safe_write_json; safe_write_json(manifest_path, manifest.to_dict())

    def find_api_skill(self, api_name: str) -> Optional[SkillManifest]:
        """Find a tested API client skill by API name."""
        results = self.search(query=api_name, category="apis", min_confidence=0.5)
        return results[0] if results else None

    def find_pattern(self, pattern_name: str) -> Optional[SkillManifest]:
        """Find a reusable pattern by name."""
        results = self.search(query=pattern_name, category="patterns")
        return results[0] if results else None

    def get_stale_skills(self, max_age_days: int = 30) -> list[SkillManifest]:
        """Find skills that haven't been verified recently."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        stale = []
        for manifest in self._index.values():
            if manifest.last_verified:
                try:
                    verified = datetime.fromisoformat(manifest.last_verified)
                    if verified < cutoff:
                        stale.append(manifest)
                except Exception:
                    stale.append(manifest)
            else:
                stale.append(manifest)
        return stale

    def get_competence_summary(self) -> dict:
        """Summarize library competence for the competence tracker."""
        by_category: dict[str, list] = {}
        for manifest in self._index.values():
            cat = manifest.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                "skill_id": manifest.skill_id,
                "name": manifest.name,
                "confidence": manifest.effective_confidence(),
                "last_verified": manifest.last_verified,
                "usage_count": manifest.usage_count,
            })

        return {
            "total_skills": len(self._index),
            "by_category": by_category,
            "high_confidence": len([m for m in self._index.values()
                                    if m.effective_confidence() >= 0.8]),
            "stale": len(self.get_stale_skills()),
        }

    def list_skills(
        self,
        category: str = "",
        min_confidence: float = 0.0,
    ) -> list[SkillManifest]:
        """List all skills, optionally filtered by category and minimum confidence.

        Convenience wrapper around search() for callers expecting a list_skills API.
        """
        return self.search(category=category, min_confidence=min_confidence)

    def format_inventory(self, category: str = "") -> str:
        """Human-readable skill inventory."""
        skills = self.search(category=category) if category else list(self._index.values())
        if not skills:
            return "Skill library: empty"

        lines = [f"📚 Skill Library ({len(skills)} skills)", ""]

        by_cat: dict[str, list] = {}
        for s in skills:
            c = s.category or "uncategorized"
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(s)

        for cat, cat_skills in sorted(by_cat.items()):
            lines.append(f"  {cat}/ ({len(cat_skills)} skills)")
            for s in sorted(cat_skills, key=lambda x: x.effective_confidence(), reverse=True):
                conf = s.effective_confidence()
                status = "✅" if conf >= 0.8 else "⚠️" if conf >= 0.5 else "❌"
                lines.append(f"    {status} {s.name}  conf={conf:.2f}  uses={s.usage_count}")

        return "\n".join(lines)


# ── Module-level singleton ───────────────────────────────────────────────────

_library: SkillLibrary | None = None


def get_library() -> SkillLibrary:
    """Get or create the singleton skill library."""
    global _library
    if _library is None:
        _library = SkillLibrary()
    return _library
