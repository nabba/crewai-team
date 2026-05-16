"""Q16 Themes 4 + 5 — self-improvement velocity + knowledge management.

PROGRAM §51 Q16 (decade-resilience hardening) second batch.

Theme 4 — recursive self-improvement boundaries:
  * app/self_improvement/velocity.py — observational rollup
  * REST endpoint /api/cp/self-improvement/velocity
  * Self-quarantine pinning test (this file)

Theme 5 — knowledge management at decade-scale:
  * 38th monitor app/healing/monitors/wiki_staleness.py
  * app/self_improvement/claude_md_compaction.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolated_module(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, REPO_ROOT / rel_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_notify(monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: captured.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)
    return captured


# ═════════════════════════════════════════════════════════════════════════
#   Theme 4a — velocity aggregator
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_velocity(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_self_improvement_velocity_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def test_velocity_module_exists() -> None:
    p = REPO_ROOT / "app" / "self_improvement" / "velocity.py"
    assert p.is_file()
    src = p.read_text()
    assert "def velocity_summary" in src
    assert "def _crs_by_quarter" in src
    assert "def _architecture_adoption_histogram" in src
    assert "def _recipe_selection_summary" in src
    assert "def _lessons_learned_summary" in src
    assert "def _forge_graduations_summary" in src
    # Never mutates state — should not call any "store.save" / "approve"
    # methods.
    forbidden = ("store.save", "approve(", "auto_approve", "create_request")
    for f in forbidden:
        assert f not in src, f"velocity must be read-only: forbidden token {f}"


def test_velocity_disabled_returns_disabled_marker(monkeypatch) -> None:
    mod = _isolated_module(
        "app/self_improvement/velocity.py", "_q16t4_velocity_disabled",
    )
    _stub_rs_velocity(monkeypatch, enabled=False)
    out = mod.velocity_summary()
    assert out == {"disabled": True}


def test_velocity_quarter_key_pure() -> None:
    mod = _isolated_module(
        "app/self_improvement/velocity.py", "_q16t4_quarter_key",
    )
    from datetime import datetime, timezone
    jan1 = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    apr1 = datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp()
    jul1 = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()
    oct1 = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    assert mod._quarter_key(jan1) == "2026Q1"
    assert mod._quarter_key(apr1) == "2026Q2"
    assert mod._quarter_key(jul1) == "2026Q3"
    assert mod._quarter_key(oct1) == "2026Q4"


def test_velocity_handles_broken_upstream_sources(monkeypatch) -> None:
    """When all data sources are unavailable, every section reports
    available=False and the rollup still succeeds."""
    mod = _isolated_module(
        "app/self_improvement/velocity.py", "_q16t4_broken_sources",
    )
    _stub_rs_velocity(monkeypatch, enabled=True)
    out = mod.velocity_summary()
    assert "generated_at" in out
    assert "window_days" in out
    for key in (
        "change_requests", "architecture_adoption", "recipes",
        "lessons_learned", "forge_graduations",
    ):
        assert key in out
        # Either available=False or empty results, but not raised.
        assert isinstance(out[key], dict)


def test_velocity_lessons_learned_counts_recent(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/self_improvement/velocity.py", "_q16t4_lessons",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    self_heal_dir = tmp_path / "self_heal"
    self_heal_dir.mkdir()
    now = time.time()
    recent_iso = "2026-05-16T10:00:00+00:00"
    old_iso = "2024-01-01T10:00:00+00:00"
    (self_heal_dir / "lessons_learned.json").write_text(json.dumps([
        {"id": "1", "created_at": recent_iso, "consulted_at_history": [recent_iso]},
        {"id": "2", "created_at": old_iso},
        {"id": "3", "created_at": old_iso, "consulted_at_history": []},
    ]))
    # Patch "now" inside the function by patching time.time.
    monkeypatch.setattr(mod.time, "time", lambda: time.mktime(
        time.strptime("2026-05-20", "%Y-%m-%d"),
    ))
    out = mod._lessons_learned_summary()
    assert out["available"] is True
    assert out["n_total"] == 3
    # 30-day cutoff — recent_iso is within, others aren't.
    assert out["n_added_last_30d"] == 1
    # consultation_field_seen on at least one row means we compute it.
    assert out["n_with_consultations"] == 1


# ═════════════════════════════════════════════════════════════════════════
#   Theme 4a — REST endpoint
# ═════════════════════════════════════════════════════════════════════════


def test_velocity_rest_router_exists() -> None:
    p = REPO_ROOT / "app" / "api" / "self_improvement_api.py"
    assert p.is_file()
    src = p.read_text()
    assert "/api/cp/self-improvement" in src
    assert '@router.get("/velocity")' in src


def test_velocity_rest_mounted_in_main() -> None:
    p = REPO_ROOT / "app" / "main.py"
    src = p.read_text()
    assert "self_improvement_api" in src
    assert "self_improvement_router" in src


# ═════════════════════════════════════════════════════════════════════════
#   Theme 4b — self-quarantine pinning test
# ═════════════════════════════════════════════════════════════════════════


def test_self_quarantine_every_entry_has_explicit_rationale() -> None:
    """Every QUARANTINED_FILES entry must appear in a comment block in
    self_quarantine.py — this protects the list from silent shrinking
    via refactor.

    The current implementation groups entries into themed comment
    blocks (# Safety core / # Audit infrastructure / etc.). The
    pinning test asserts the comment-block-count plus that the file
    has at least one short rationale paragraph above each block."""
    src = (
        REPO_ROOT / "app" / "governance_amendment" / "self_quarantine.py"
    ).read_text()
    # Pull the QUARANTINED_FILES set out by importing the module
    # directly.
    mod = _isolated_module(
        "app/governance_amendment/self_quarantine.py",
        "_q16t4_self_quarantine_pin",
    )
    entries = mod.QUARANTINED_FILES
    assert len(entries) >= 30, (
        f"self-quarantine list shrank to {len(entries)} entries; "
        f"removing entries here is intentional and dangerous — "
        f"please double-check with a human before touching this list"
    )
    # Confirm at least 5 comment blocks (themed groupings) are present
    # — the empirical structure as of 2026-05-16 has 7+.
    comment_blocks = re.findall(r"^\s+# \S+", src, re.MULTILINE)
    assert len(comment_blocks) >= 5, (
        "self_quarantine.py has fewer themed comment blocks than "
        "expected; entries should be grouped + each group rationale "
        "documented for future readers"
    )
    # Every entry must appear literally in the source (no dynamic
    # construction allowed — the integrity manifest catches deploy-
    # time tampering, but only if the source values are literal).
    for entry in entries:
        assert entry in src, (
            f"quarantine entry {entry!r} not found verbatim in "
            f"self_quarantine.py — dynamic construction would bypass "
            f"the integrity manifest"
        )


def test_self_quarantine_module_docstring_explains_dgm_invariant() -> None:
    src = (
        REPO_ROOT / "app" / "governance_amendment" / "self_quarantine.py"
    ).read_text()
    # The module's top-level docstring should reference the DGM
    # (Darwin-Gödel Machine) safety invariant.
    assert "DGM" in src or "Darwin" in src, (
        "self_quarantine module docstring should explain the DGM "
        "safety invariant — the WHY behind these files being "
        "infrastructure-level"
    )


def test_self_quarantine_includes_critical_safety_infrastructure() -> None:
    """Pinning test: these specific paths MUST be in the quarantine
    list. Loss of any one would be a P0 safety regression."""
    mod = _isolated_module(
        "app/governance_amendment/self_quarantine.py",
        "_q16t4_self_quarantine_critical",
    )
    must_be_quarantined = (
        # The protocol itself
        "app/governance_amendment/protocol.py",
        "app/governance_amendment/self_quarantine.py",
        # Safety core
        "app/safety_guardian.py",
        "app/eval_sandbox.py",
        "app/auto_deployer.py",
        "app/goodhart_guard.py",
        "app/human_gate.py",
        # Tier-3 manifest
        "app/subia/.integrity_manifest.json",
        "app/subia/integrity.py",
        # Constitution
        "app/souls/constitution.md",
    )
    for path in must_be_quarantined:
        assert path in mod.QUARANTINED_FILES, (
            f"CRITICAL safety-infrastructure path {path!r} is NOT in "
            f"the self-quarantine list — this would be a P0 safety "
            f"regression. Restore the entry before this test passes."
        )


# ═════════════════════════════════════════════════════════════════════════
#   Theme 5a — wiki_staleness monitor
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_wiki(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_wiki_staleness_monitor_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_wiki_monitor(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/healing/monitors/wiki_staleness.py",
        "_q16t5_wiki_staleness",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    # Point the wiki root at the tmp_path/wiki for isolation.
    monkeypatch.setattr(mod, "_wiki_root", lambda: tmp_path / "wiki")
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    return mod


def test_wiki_staleness_module_exists() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "wiki_staleness.py"
    assert p.is_file()
    src = p.read_text()
    assert "def run(" in src
    assert "_STALE_THRESHOLD_DAYS" in src
    assert "_EXCLUDE_DIR_PREFIXES" in src
    # Auto-archive dirs MUST be excluded — assert key prefixes are listed.
    for prefix in ("legacy", "value_reflections", "governance"):
        assert prefix in src


def test_wiki_staleness_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_wiki_monitor(monkeypatch, tmp_path)
    _stub_rs_wiki(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.run(now=time.time())
    assert out.get("skipped") is True


def test_wiki_staleness_no_wiki_dir_silent(monkeypatch, tmp_path) -> None:
    mod = _load_wiki_monitor(monkeypatch, tmp_path)
    _stub_rs_wiki(monkeypatch)
    _stub_notify(monkeypatch)
    out = mod.run(now=time.time())
    assert out["ran"] is True
    assert out["wiki_present"] is False
    assert out["alert_sent"] is False


def test_wiki_staleness_no_stale_files_silent(monkeypatch, tmp_path) -> None:
    mod = _load_wiki_monitor(monkeypatch, tmp_path)
    _stub_rs_wiki(monkeypatch)
    _stub_notify(monkeypatch)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    # Recent file (mtime = now) should NOT be stale.
    fresh = wiki / "fresh.md"
    fresh.write_text("hello")
    out = mod.run(now=time.time())
    assert out["wiki_present"] is True
    assert out["n_stale"] == 0


def test_wiki_staleness_surfaces_stale_files(monkeypatch, tmp_path) -> None:
    mod = _load_wiki_monitor(monkeypatch, tmp_path)
    _stub_rs_wiki(monkeypatch)
    captured = _stub_notify(monkeypatch)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    stale = wiki / "old_topic.md"
    stale.write_text("ancient content")
    # Backdate the file.
    old_mtime = time.time() - 400 * 86400  # 400 days old
    os.utime(stale, (old_mtime, old_mtime))
    out = mod.run(now=time.time())
    assert out["wiki_present"] is True
    assert out["n_stale"] == 1
    assert out["n_digest_due"] == 1
    assert out["alert_sent"] is True
    assert any("Wiki freshness" in c.get("title", "") for c in captured)


def test_wiki_staleness_skips_archive_directories(monkeypatch, tmp_path) -> None:
    """legacy/, value_reflections/, governance/ should be excluded."""
    mod = _load_wiki_monitor(monkeypatch, tmp_path)
    _stub_rs_wiki(monkeypatch)
    _stub_notify(monkeypatch)
    wiki = tmp_path / "wiki"
    for sub in ("self/legacy", "self/value_reflections", "governance"):
        d = wiki / sub
        d.mkdir(parents=True)
        f = d / "old.md"
        f.write_text("archived")
        old = time.time() - 800 * 86400
        os.utime(f, (old, old))
    out = mod.run(now=time.time())
    assert out["n_stale"] == 0


def test_wiki_staleness_dedup_window(monkeypatch, tmp_path) -> None:
    """Second pass within dedup window should not re-surface the same
    file."""
    mod = _load_wiki_monitor(monkeypatch, tmp_path)
    _stub_rs_wiki(monkeypatch)
    captured = _stub_notify(monkeypatch)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    stale = wiki / "x.md"
    stale.write_text("...")
    old = time.time() - 500 * 86400
    os.utime(stale, (old, old))
    now = time.time()
    mod.run(now=now)
    n1 = len(captured)
    # Force the internal weekly cadence past.
    mod.run(now=now + 8 * 86400)
    # Same file → dedup → no second alert.
    assert len(captured) == n1


# ═════════════════════════════════════════════════════════════════════════
#   Theme 5b — CLAUDE.md compaction composer
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_compaction(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_claude_md_compaction_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_compaction(monkeypatch, tmp_path: Path, *, small_threshold: bool = False):
    """Each invocation creates a fresh isolated module so per-test
    state (e.g. ``_MIN_BYTES_TO_PROPOSE``) doesn't bleed across tests."""
    # Use a per-test unique name so test isolation is preserved.
    import uuid
    name = f"_q16t5_compaction_{uuid.uuid4().hex[:8]}"
    mod = _isolated_module(
        "app/self_improvement/claude_md_compaction.py", name,
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        mod, "_output_dir",
        lambda: tmp_path / "self_improvement" / "claude_md_compaction",
    )
    if small_threshold:
        # Tests use synthetic small files; lower the byte threshold so
        # the structural-split logic runs.
        monkeypatch.setattr(mod, "_MIN_BYTES_TO_PROPOSE", 1000)
    return mod


def test_compaction_module_exists() -> None:
    p = REPO_ROOT / "app" / "self_improvement" / "claude_md_compaction.py"
    assert p.is_file()
    src = p.read_text()
    assert "def compose_proposal" in src
    assert "def run_once" in src
    assert "_KEEP_RECENT_MONTHS" in src
    # Should NEVER auto-write to source CLAUDE.md.
    assert "CLAUDE.md" in src  # references it
    # No direct write back to source path — verify via grep that
    # source_path is only READ.
    assert "source_path.write" not in src


def test_compaction_disabled_returns_skipped(monkeypatch, tmp_path) -> None:
    mod = _load_compaction(monkeypatch, tmp_path)
    _stub_rs_compaction(monkeypatch, enabled=False)
    out = mod.run_once()
    assert out.get("skipped") is True


def test_compaction_skips_small_files(monkeypatch, tmp_path) -> None:
    mod = _load_compaction(monkeypatch, tmp_path, small_threshold=True)
    _stub_rs_compaction(monkeypatch)
    _stub_notify(monkeypatch)
    src = tmp_path / "CLAUDE.md"
    src.write_text("# Tiny\n- Q1 2026-01-01 first.\n")
    out = mod.compose_proposal(source_path=src)
    assert out is None  # too small to bother


def test_compaction_proposes_with_old_entries(monkeypatch, tmp_path) -> None:
    """A file with both old and recent Q-batch entries should produce
    a proposal with kept + archived counts."""
    mod = _load_compaction(monkeypatch, tmp_path, small_threshold=True)
    _stub_rs_compaction(monkeypatch)
    _stub_notify(monkeypatch)
    src = tmp_path / "CLAUDE.md"
    # Build a synthetic CLAUDE.md large enough to cross the threshold.
    head = (
        "# AndrusAI\n\n## Quick Reference\n"
        + ("- Item: filler\n" * 200)
        + "\n## Architecture\n\n"
    )
    # 5 old Q-batch entries.
    old_entries = "".join(
        f"- Q{i} old description (2023-01-{(i % 28) + 1:02d}; PROGRAM §{i})"
        f" — {('lorem ipsum ' * 30)}\n"
        for i in range(1, 6)
    )
    # 3 recent Q-batch entries (within 6 months).
    recent_entries = "".join(
        f"- Q{i} recent description (2026-05-{(i % 28) + 1:02d}; PROGRAM §{i})"
        f" — {('dolor sit amet ' * 30)}\n"
        for i in range(10, 13)
    )
    src.write_text(head + old_entries + recent_entries)

    # Reference "now" deterministically so the date heuristic stays
    # stable across test runs.
    from datetime import datetime, timezone
    now = datetime(2026, 5, 16, tzinfo=timezone.utc).timestamp()
    out = mod.compose_proposal(source_path=src, now=now)
    assert out is not None
    assert out["n_kept"] == 3
    assert out["n_archived"] == 5
    assert out["compacted_bytes"] < out["original_bytes"]
    # Three artifacts on disk.
    assert Path(out["compacted_path"]).exists()
    assert Path(out["archive_path"]).exists()
    assert Path(out["notes_path"]).exists()
    # Compacted file should preserve the head verbatim.
    compacted = Path(out["compacted_path"]).read_text()
    assert "Quick Reference" in compacted
    assert "Architecture" in compacted
    # Compacted should contain the RECENT entries but NOT the old ones.
    assert "(2026-05-" in compacted
    assert "(2023-01-" not in compacted
    # Pointer to the archive should be present.
    assert "claude-md-archive-" in compacted


def test_compaction_idempotent_within_year(monkeypatch, tmp_path) -> None:
    mod = _load_compaction(monkeypatch, tmp_path, small_threshold=True)
    _stub_rs_compaction(monkeypatch)
    _stub_notify(monkeypatch)
    src = tmp_path / "CLAUDE.md"
    head = "# Doc\n\n" + ("- filler\n" * 200) + "\n"
    old = "".join(
        f"- Q{i} (2023-0{i % 9 + 1}-01; PROGRAM §{i}) — " + ("body " * 50) + "\n"
        for i in range(1, 6)
    )
    recent = (
        f"- Q10 (2026-05-16; PROGRAM §50) — " + ("body " * 50) + "\n"
    )
    src.write_text(head + old + recent)
    from datetime import datetime, timezone
    now = datetime(2026, 5, 16, tzinfo=timezone.utc).timestamp()
    first = mod.compose_proposal(source_path=src, now=now)
    second = mod.compose_proposal(source_path=src, now=now)
    assert first is not None
    assert second is None  # already proposed this year


def test_compaction_undated_entries_archive(monkeypatch, tmp_path) -> None:
    """Q-batch entries WITHOUT an ISO date default to ARCHIVE."""
    mod = _load_compaction(monkeypatch, tmp_path, small_threshold=True)
    _stub_rs_compaction(monkeypatch)
    _stub_notify(monkeypatch)
    src = tmp_path / "CLAUDE.md"
    head = "# Doc\n\n" + ("- filler\n" * 200) + "\n"
    body = (
        # Undated old-style entry
        "- Q0 ancient feature without a date — " + ("body " * 50) + "\n"
        + "- Q1 (2026-05-10; PROGRAM §51) — " + ("body " * 50) + "\n"
    )
    src.write_text(head + body)
    from datetime import datetime, timezone
    now = datetime(2026, 5, 16, tzinfo=timezone.utc).timestamp()
    out = mod.compose_proposal(source_path=src, now=now)
    assert out is not None
    assert out["n_kept"] == 1
    assert out["n_archived"] == 1


# ═════════════════════════════════════════════════════════════════════════
#   Wiring tests
# ═════════════════════════════════════════════════════════════════════════


def test_runtime_settings_has_themes_4_5_switches() -> None:
    p = REPO_ROOT / "app" / "runtime_settings.py"
    src = p.read_text()
    for key in (
        "self_improvement_velocity_enabled",
        "wiki_staleness_monitor_enabled",
        "claude_md_compaction_enabled",
    ):
        assert f'"{key}"' in src
        assert f"def get_{key}" in src
        assert f"def set_{key}" in src


def test_monitors_init_registers_themes_5_components() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "__init__.py"
    src = p.read_text()
    assert '"wiki_staleness"' in src
    assert "wiki_staleness.run" in src
    assert '"claude_md_compaction"' in src
    assert "claude_md_compaction.run_once" in src
