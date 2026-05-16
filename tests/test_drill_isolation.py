"""Drill isolation pinning tests.

Pins the safety guards that prevent the 2026-05-16 corruption
incident from recurring in restore-drill.sh and version-upgrade-drill.sh.
The sibling migration-drill.sh has its own pinning tests in
test_q13_resilience_year2.py — those guards are functionally identical
but use a different overlay file (narrower scope: postgres only).

The flaw being guarded against: `docker compose -p <project>` renames
containers and networks but reads bind-mount paths literally from
docker-compose.yml. Without an overlay, both the live stack and the
drill stack mount ./workspace/mem0_pgdata (and mem0_neo4j, and memory),
race on the data-dir lock, and the drill's partial restore corrupts
the live databases.
"""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _drill_script(name: str) -> Path:
    p = REPO_ROOT / "deploy" / "scripts" / name
    assert p.is_file(), f"missing drill script: {p}"
    return p


def test_drill_isolation_overlay_present() -> None:
    """The shared overlay must exist and remap all three data services
    (postgres + neo4j + chromadb) to ephemeral named volumes."""
    p = REPO_ROOT / "docker-compose.drill-isolation.yml"
    assert p.is_file(), (
        "docker-compose.drill-isolation.yml must exist alongside "
        "docker-compose.yml — both restore-drill.sh and "
        "version-upgrade-drill.sh reference it explicitly."
    )
    src = p.read_text()
    # Must define one named volume per data service.
    for vol in ("drill_pgdata", "drill_neo4j", "drill_chroma"):
        assert vol in src, (
            f"Overlay must define a drill-specific volume {vol}."
        )
    # Must use !override on volumes (not !reset). !reset clears the
    # parent list entirely — the drill container would start with NO
    # data dir and initdb would fail in confusing ways.
    assert "!override" in src, (
        "Overlay must use !override on volumes (not !reset — that "
        "erases the list instead of replacing it)."
    )
    # Must NOT reference the live bind-mount paths in any volume mount
    # context. (Header comment can describe them; the assertion below
    # focuses on what's load-bearing.)
    # The live mount sources are sufficiently unique substrings that
    # a regex-free check is fine here.
    body_after_services = src.split("services:", 1)[-1] if "services:" in src else src
    # Find the volumes: section under services (not the top-level one).
    # We just check that no live path appears in any volume line that
    # would actually mount into a container.
    for live_path in (
        "./workspace/mem0_pgdata",
        "./workspace/mem0_neo4j",
        "./workspace/memory:/",
    ):
        assert live_path not in body_after_services, (
            f"Overlay must not reference the live bind-mount {live_path}."
        )


def _assert_drill_uses_isolation(script: Path) -> None:
    src = script.read_text()
    # Must load the overlay file. The exact -f flag arrangement is up
    # to the script, but the overlay filename must appear.
    assert "docker-compose.drill-isolation.yml" in src, (
        f"{script.name} must load docker-compose.drill-isolation.yml. "
        "Without it the drill corrupts the live databases."
    )
    # Pre-flight check for all three live containers. The drill brings
    # up all three services, so all three live counterparts can race.
    for live in (
        "crewai-team-postgres-1",
        "crewai-team-neo4j-1",
        "crewai-team-chromadb-1",
    ):
        assert live in src, (
            f"{script.name} must pre-flight-check for {live} before "
            "starting its own stack."
        )


def test_restore_drill_uses_isolation() -> None:
    _assert_drill_uses_isolation(_drill_script("restore-drill.sh"))


def test_version_upgrade_drill_uses_isolation() -> None:
    _assert_drill_uses_isolation(_drill_script("version-upgrade-drill.sh"))


def test_restore_drill_executable() -> None:
    p = _drill_script("restore-drill.sh")
    assert os.access(p, os.X_OK), f"{p.name} must be executable"


def test_version_upgrade_drill_executable() -> None:
    p = _drill_script("version-upgrade-drill.sh")
    assert os.access(p, os.X_OK), f"{p.name} must be executable"


# ─────────────────────────────────────────────────────────────────────────
# Follow-up fixes (PR splitting out the two pre-existing bugs noted in
# the sibling-isolation PR body).
# ─────────────────────────────────────────────────────────────────────────


def test_compose_image_tags_are_overrideable() -> None:
    """docker-compose.yml must use ${IMAGE:-default} placeholders for
    the three data services, so version-upgrade-drill.sh's env-var
    image overrides actually take effect. The default values must
    preserve the previous hardcoded tags so live-stack behaviour is
    unchanged unless the operator explicitly sets the env vars."""
    p = REPO_ROOT / "docker-compose.yml"
    assert p.is_file()
    src = p.read_text()
    for placeholder in (
        "${POSTGRES_IMAGE:-pgvector/pgvector:pg16}",
        "${NEO4J_IMAGE:-neo4j:5-community}",
        "${CHROMA_IMAGE:-chromadb/chroma:0.5.23}",
    ):
        assert placeholder in src, (
            f"docker-compose.yml must contain {placeholder} so the "
            "version-upgrade drill's image override env vars actually "
            "flow through compose. The :-default form preserves live "
            "behaviour when no env var is set."
        )


def _assert_chromadb_restore_correctness(script: Path) -> None:
    src = script.read_text()
    # Must NOT hardcode the volume name — that produced a mismatch
    # against the overlay's `drill_chroma` namespacing and the
    # tarball ended up in an orphan volume.
    assert "${DRILL_PROJECT}_chroma:" not in src, (
        f"{script.name} must not hardcode `${{DRILL_PROJECT}}_chroma`. "
        "Resolve the volume name from the running container via "
        "`docker inspect ... --format ... .Destination /chroma/chroma`."
    )
    # Must derive the volume name from the running container.
    assert 'docker inspect "$CHR_CT"' in src and "/chroma/chroma" in src, (
        f"{script.name} must use docker inspect to get the chromadb "
        "volume name from the running container."
    )
    # Must use --strip-components=1 because the backup tar has a
    # leading `memory/` directory (created via `tar -czf -C ./workspace memory`),
    # and without stripping it the restored files end up at
    # /chroma/chroma/memory/<uuid>/... instead of /chroma/chroma/<uuid>/...
    # and chromadb silently came up empty.
    assert "--strip-components=1" in src, (
        f"{script.name} must use tar --strip-components=1 so the "
        "leading `memory/` directory in the backup tar is stripped "
        "and files land at /chroma/chroma/<uuid>/... where chromadb "
        "expects them."
    )


def test_restore_drill_chromadb_restore_is_correct() -> None:
    _assert_chromadb_restore_correctness(_drill_script("restore-drill.sh"))


def test_version_upgrade_drill_chromadb_restore_is_correct() -> None:
    _assert_chromadb_restore_correctness(_drill_script("version-upgrade-drill.sh"))
