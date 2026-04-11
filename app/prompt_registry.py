"""
prompt_registry.py — Versioned prompt storage with hot-reload.

Manages a git-tracked directory of versioned prompt files. Each role has a
directory with numbered versions (v001.md, v002.md, ...) and an active.txt
pointer that controls which version is live.

Architecture:
  - workspace/prompts/{role}/v{NNN}.md  — immutable once written
  - workspace/prompts/{role}/active.txt — points to active version number
  - workspace/prompts/_shared/{layer}/  — constitution, style, agents_protocol
  - workspace/prompts/few_shot_examples/ — versioned JSON arrays
  - workspace/prompts/style_params/     — versioned style configuration

The registry is part of the self-improving feedback loop's Tier 1 (Adaptive)
layer. The modification engine proposes new versions; the eval sandbox tests
them; promotion updates active.txt.

A generation counter tracks config changes so compose_backstory() can
invalidate its cache without polling the filesystem on every call.
"""

import json
import logging
import difflib
import shutil
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("/app/workspace/prompts")
SOULS_DIR = Path(__file__).parent / "souls"

# Roles that have per-role prompt files
AGENT_ROLES = [
    "commander", "researcher", "coder", "writer",
    "critic", "self_improver", "media_analyst",
]

# Shared layers (applied to all agents)
SHARED_LAYERS = ["constitution", "style", "agents_protocol"]

# ── Generation counter for cache invalidation ─────────────────────────────
# Bumped whenever a version is promoted.  loader.py checks this to decide
# whether to rebuild backstory from disk or serve cached.
_generation: int = 0


def bump_generation() -> int:
    """Increment the generation counter.  Returns the new value."""
    global _generation
    _generation += 1
    logger.info(f"prompt_registry: generation bumped to {_generation}")
    return _generation


def current_generation() -> int:
    """Return the current generation counter value."""
    return _generation


# ── Initialization ─────────────────────────────────────────────────────────

def init_registry() -> None:
    """Extract souls/*.md into workspace/prompts/ if not already present.

    Called once at startup.  If v001 already exists for a role, the existing
    file is preserved (may have been manually edited or promoted from a
    previous version).  This is idempotent.
    """
    logger.info("prompt_registry: initializing versioned prompt store")

    # Agent roles
    for role in AGENT_ROLES:
        role_dir = PROMPTS_DIR / role
        role_dir.mkdir(parents=True, exist_ok=True)
        v001 = role_dir / "v001.md"
        active_file = role_dir / "active.txt"

        if not v001.exists():
            source = SOULS_DIR / f"{role}.md"
            if source.exists():
                shutil.copy2(source, v001)
                logger.info(f"  {role}: extracted v001 from souls/{role}.md")
            else:
                v001.write_text(f"# {role}\n\nNo soul file found.\n")
                logger.warning(f"  {role}: no source soul file, created placeholder")

        if not active_file.exists():
            active_file.write_text("1")

    # Shared layers
    shared_dir = PROMPTS_DIR / "_shared"
    for layer in SHARED_LAYERS:
        layer_dir = shared_dir / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        v001 = layer_dir / "v001.md"
        active_file = layer_dir / "active.txt"

        if not v001.exists():
            source = SOULS_DIR / f"{layer}.md"
            if source.exists():
                shutil.copy2(source, v001)
                logger.info(f"  _shared/{layer}: extracted v001")
            else:
                v001.write_text(f"# {layer}\n\nNo source file found.\n")

        if not active_file.exists():
            active_file.write_text("1")

    # Few-shot examples store
    fse_dir = PROMPTS_DIR / "few_shot_examples"
    fse_dir.mkdir(parents=True, exist_ok=True)
    v001 = fse_dir / "v001.json"
    if not v001.exists():
        v001.write_text(json.dumps({
            "style_examples": [],
            "correction_examples": [],
            "domain_examples": [],
        }, indent=2))
    if not (fse_dir / "active.txt").exists():
        (fse_dir / "active.txt").write_text("1")

    # Style params store
    sp_dir = PROMPTS_DIR / "style_params"
    sp_dir.mkdir(parents=True, exist_ok=True)
    v001 = sp_dir / "v001.json"
    if not v001.exists():
        v001.write_text(json.dumps({
            "verbosity": "concise",
            "formality": "casual-professional",
            "max_signal_length": 1400,
            "citation_style": "inline",
            "code_block_preference": "fenced",
        }, indent=2))
    if not (sp_dir / "active.txt").exists():
        (sp_dir / "active.txt").write_text("1")

    logger.info(f"prompt_registry: initialized ({len(AGENT_ROLES)} roles, "
                f"{len(SHARED_LAYERS)} shared layers, few_shot_examples, style_params)")


# ── Version management ─────────────────────────────────────────────────────

def _resolve_dir(role: str) -> Path:
    """Resolve the directory for a given role or shared layer."""
    if role in AGENT_ROLES:
        return PROMPTS_DIR / role
    elif role in SHARED_LAYERS:
        return PROMPTS_DIR / "_shared" / role
    elif role in ("few_shot_examples", "style_params"):
        return PROMPTS_DIR / role
    else:
        raise ValueError(f"Unknown role/layer: {role}")


def _is_json_store(role: str) -> bool:
    """Check if this role uses JSON files instead of markdown."""
    return role in ("few_shot_examples", "style_params")


def _version_filename(version: int, role: str) -> str:
    ext = ".json" if _is_json_store(role) else ".md"
    return f"v{version:03d}{ext}"


def get_active_version(role: str) -> int:
    """Read the active version number for a role."""
    d = _resolve_dir(role)
    active_file = d / "active.txt"
    if active_file.exists():
        try:
            return int(active_file.read_text().strip())
        except (ValueError, OSError):
            pass
    return 1  # default to v001


def list_versions(role: str) -> list[int]:
    """List all version numbers for a role, sorted ascending."""
    d = _resolve_dir(role)
    ext = ".json" if _is_json_store(role) else ".md"
    versions = []
    for f in d.glob(f"v*{ext}"):
        try:
            num = int(f.stem[1:])  # "v003" → 3
            versions.append(num)
        except ValueError:
            continue
    return sorted(versions)


def get_prompt(role: str, version: int | None = None) -> str:
    """Read a specific version's content.  Defaults to active version."""
    if version is None:
        version = get_active_version(role)
    d = _resolve_dir(role)
    fname = _version_filename(version, role)
    filepath = d / fname
    if filepath.exists():
        return filepath.read_text().strip()
    logger.warning(f"prompt_registry: {role}/v{version:03d} not found")
    return ""


def get_active_prompt(role: str) -> str:
    """Shorthand for get_prompt(role) with active version."""
    return get_prompt(role)


def get_prompt_versions_map() -> dict[str, int]:
    """Return a dict of {role: active_version} for all roles + shared layers.

    Used by feedback pipeline to record which prompt versions were active
    when a response was generated.
    """
    versions = {}
    for role in AGENT_ROLES:
        versions[role] = get_active_version(role)
    for layer in SHARED_LAYERS:
        versions[layer] = get_active_version(layer)
    versions["few_shot_examples"] = get_active_version("few_shot_examples")
    versions["style_params"] = get_active_version("style_params")
    return versions


def propose_version(role: str, content: str, reason: str) -> int:
    """Write a new version file.  Does NOT promote it.

    Returns the new version number.
    """
    d = _resolve_dir(role)
    versions = list_versions(role)
    new_num = (max(versions) + 1) if versions else 1

    fname = _version_filename(new_num, role)
    filepath = d / fname
    from app.safe_io import safe_write
    safe_write(filepath, content)

    # Write a changelog entry
    changelog = d / "changelog.jsonl"
    entry = {
        "version": new_num,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "proposed",
    }
    with open(changelog, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"prompt_registry: proposed {role}/v{new_num:03d} — {reason[:80]}")
    return new_num


def promote_version(role: str, version: int) -> None:
    """Update active.txt to point to the given version.

    Bumps the generation counter so loader.py invalidates its cache.
    """
    d = _resolve_dir(role)
    fname = _version_filename(version, role)
    if not (d / fname).exists():
        raise FileNotFoundError(f"{role}/v{version:03d} does not exist")

    old_version = get_active_version(role)
    from app.safe_io import safe_write
    safe_write(d / "active.txt", str(version))

    # Update changelog
    changelog = d / "changelog.jsonl"
    entry = {
        "version": version,
        "reason": f"promoted (was v{old_version:03d})",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "promoted",
    }
    with open(changelog, "a") as f:
        f.write(json.dumps(entry) + "\n")

    bump_generation()
    logger.info(f"prompt_registry: promoted {role} v{old_version:03d} → v{version:03d}")


def rollback(role: str, to_version: int | None = None) -> int:
    """Rollback to previous or specific version.

    If to_version is None, rolls back to the version before the current one.
    Returns the version rolled back to.
    """
    current = get_active_version(role)
    if to_version is None:
        versions = list_versions(role)
        prev_versions = [v for v in versions if v < current]
        if not prev_versions:
            logger.warning(f"prompt_registry: no previous version to rollback {role}")
            return current
        to_version = max(prev_versions)

    promote_version(role, to_version)

    # Update changelog with rollback reason
    d = _resolve_dir(role)
    changelog = d / "changelog.jsonl"
    entry = {
        "version": to_version,
        "reason": f"rollback from v{current:03d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "rolled_back",
    }
    with open(changelog, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"prompt_registry: rolled back {role} v{current:03d} → v{to_version:03d}")
    return to_version


def get_diff(role: str, v_old: int, v_new: int) -> str:
    """Return a unified diff between two versions."""
    old_content = get_prompt(role, v_old)
    new_content = get_prompt(role, v_new)

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"{role}/v{v_old:03d}",
        tofile=f"{role}/v{v_new:03d}",
    )
    return "".join(diff)


# ── Style params and few-shot helpers ──────────────────────────────────────

def get_style_params() -> dict:
    """Load the active style parameters."""
    content = get_active_prompt("style_params")
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("prompt_registry: invalid style_params JSON")
        return {}


def get_few_shot_examples() -> dict:
    """Load the active few-shot examples."""
    content = get_active_prompt("few_shot_examples")
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("prompt_registry: invalid few_shot_examples JSON")
        return {}
