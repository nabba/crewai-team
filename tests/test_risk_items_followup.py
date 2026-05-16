"""Tests for the four RISK items in the 2026-05-16 monitor audit.

The audit at docs/AUDIT_2026_05_16_DESTRUCTIVE_MONITORS.md identified
four auto-deleting monitors whose remediation shape (filter heuristic
→ unprompted destructive action) matched the same pattern that caused
today's two live-data incidents. This file pins the structural fixes.

  1. retention.run_worktrees       — validate worktree_path before rmtree
  2. retention.run_attachments     — refuse env-set paths outside safe prefixes
  3. lock_housekeeper              — explicit per-dir basename rule
  4. log_archival._purge_old_archives — retention floor + per-pass cap

Each fix is tested two ways: a source-level grep (works in any env) and
an isolated-load functional test that exercises the pure helper.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolated(rel: str, name: str):
    """Load a module under app/healing/monitors/ via spec_from_file_location.
    The monitors transitively import app.life_companion._common which
    needs pydantic_settings. Tests that exercise pure helpers stub out
    the import path."""
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / rel,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
# retention.run_worktrees — _validate_worktree_path
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def retention_mod(monkeypatch_module):
    """Load retention.py with the gateway-deps import stubbed."""
    # Stub _common before retention imports it.
    stub_common = type(sys)("stub_common")
    stub_common.audit_event = lambda *a, **k: None
    stub_common.background_enabled = lambda: True
    stub_common.read_state_json = lambda *a, **k: {}
    stub_common.send_signal_alert = lambda *a, **k: None
    stub_common.write_state_json = lambda *a, **k: None
    monkeypatch_module.setitem(sys.modules, "app.life_companion._common", stub_common)
    # Also need to stub the package itself.
    stub_pkg = type(sys)("stub_life_companion")
    stub_pkg._common = stub_common
    monkeypatch_module.setitem(sys.modules, "app.life_companion", stub_pkg)
    monkeypatch_module.setitem(sys.modules, "app", type(sys)("stub_app"))
    monkeypatch_module.setattr(sys.modules["app"], "life_companion", stub_pkg, raising=False)
    return _isolated("app/healing/monitors/retention.py", "retention_test")


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (pytest's default is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_validate_worktree_path_accepts_canonical(retention_mod, monkeypatch, tmp_path):
    """Happy path: absolute, inside worktree_root, basename = session_id,
    exactly one level deep → accepted."""
    monkeypatch.setenv("CODING_SESSION_WORKTREE_ROOT", str(tmp_path))
    (tmp_path / "sess-abc").mkdir()
    ok, reason = retention_mod._validate_worktree_path(
        str(tmp_path / "sess-abc"),
        expected_session_id="sess-abc",
    )
    assert ok, f"expected ok, got {reason}"


def test_validate_worktree_path_refuses_empty():
    """Empty path must refuse — would otherwise create an empty
    rmtree target that defaults to cwd."""
    mod = _isolated_for_pure_helpers()
    ok, reason = mod._validate_worktree_path("", expected_session_id="x")
    assert not ok
    assert "empty" in reason or "non_string" in reason


def test_validate_worktree_path_refuses_relative():
    """Relative paths resolve against cwd → could hit anywhere."""
    mod = _isolated_for_pure_helpers()
    ok, reason = mod._validate_worktree_path(
        "some/relative", expected_session_id="x",
    )
    assert not ok
    assert "relative" in reason


def test_validate_worktree_path_refuses_outside_root(monkeypatch, tmp_path):
    """Absolute path outside worktree_root must refuse — protects
    against a corrupted session JSON pointing at arbitrary host
    locations (e.g., /etc, /home/user, ...)."""
    mod = _isolated_for_pure_helpers()
    monkeypatch.setenv("CODING_SESSION_WORKTREE_ROOT", str(tmp_path / "root"))
    ok, reason = mod._validate_worktree_path(
        "/etc/passwd", expected_session_id="x",
    )
    assert not ok
    assert "outside_worktree_root" in reason


def test_validate_worktree_path_refuses_basename_mismatch(monkeypatch, tmp_path):
    """Defends against a session JSON whose worktree_path points at
    someone else's worktree dir (cross-session collision)."""
    mod = _isolated_for_pure_helpers()
    monkeypatch.setenv("CODING_SESSION_WORKTREE_ROOT", str(tmp_path))
    (tmp_path / "other-session").mkdir()
    ok, reason = mod._validate_worktree_path(
        str(tmp_path / "other-session"),
        expected_session_id="my-session",
    )
    assert not ok
    assert "basename_mismatch" in reason


def test_validate_worktree_path_refuses_nested(monkeypatch, tmp_path):
    """Two or more segments under worktree_root → refuse. Defends
    against worktree_path = '/tmp/agent-sessions/abc/def' style."""
    mod = _isolated_for_pure_helpers()
    monkeypatch.setenv("CODING_SESSION_WORKTREE_ROOT", str(tmp_path))
    (tmp_path / "abc" / "def").mkdir(parents=True)
    ok, reason = mod._validate_worktree_path(
        str(tmp_path / "abc" / "def"),
        expected_session_id="def",
    )
    assert not ok
    assert "depth" in reason


# ─────────────────────────────────────────────────────────────────────────
# retention.run_attachments — _is_attachments_dir_safe
# ─────────────────────────────────────────────────────────────────────────


def test_is_attachments_dir_safe_accepts_default():
    """Production default /app/attachments must be accepted."""
    mod = _isolated_for_pure_helpers()
    ok, _ = mod._is_attachments_dir_safe(Path("/app/attachments"))
    assert ok


def test_is_attachments_dir_safe_accepts_workspace_subdir():
    mod = _isolated_for_pure_helpers()
    ok, _ = mod._is_attachments_dir_safe(Path("/app/workspace/attachments"))
    assert ok


def test_is_attachments_dir_safe_refuses_root():
    """/ would let retention delete the entire FS root."""
    mod = _isolated_for_pure_helpers()
    ok, reason = mod._is_attachments_dir_safe(Path("/"))
    assert not ok
    assert "not_in_safe_prefix" in reason


def test_is_attachments_dir_safe_refuses_arbitrary_path():
    """/home/foo isn't on the safe-prefix list."""
    mod = _isolated_for_pure_helpers()
    ok, reason = mod._is_attachments_dir_safe(Path("/home/operator"))
    assert not ok
    assert "not_in_safe_prefix" in reason


def test_is_attachments_dir_safe_refuses_relative():
    mod = _isolated_for_pure_helpers()
    ok, reason = mod._is_attachments_dir_safe(Path("attachments"))
    assert not ok
    assert "relative_path" in reason


def test_is_attachments_dir_safe_refuses_suffix_attack():
    """Prefix-attack defense: /app/attachments-evil must not pass
    because the safe prefix check uses prefix + '/'."""
    mod = _isolated_for_pure_helpers()
    ok, reason = mod._is_attachments_dir_safe(Path("/app/attachments-evil"))
    assert not ok
    assert "not_in_safe_prefix" in reason


# ─────────────────────────────────────────────────────────────────────────
# lock_housekeeper — _basename_matches_rules
# ─────────────────────────────────────────────────────────────────────────


def _isolated_lock_housekeeper():
    # Stub the same gateway-deps imports.
    stub_common = type(sys)("stub_common")
    stub_common.audit_event = lambda *a, **k: None
    stub_common.background_enabled = lambda: True
    stub_common.read_state_json = lambda *a, **k: {}
    stub_common.send_signal_alert = lambda *a, **k: None
    stub_common.write_state_json = lambda *a, **k: None
    sys.modules.setdefault("app", type(sys)("stub_app"))
    sys.modules["app.life_companion"] = type(sys)("stub_lc")
    sys.modules["app.life_companion._common"] = stub_common
    return _isolated("app/healing/monitors/lock_housekeeper.py", "lh_test")


def test_lock_basename_matches_workspace_lock():
    mod = _isolated_lock_housekeeper()
    assert mod._basename_matches_rules(".workspace.lock", (".workspace.lock",))


def test_lock_basename_matches_wiki_slug_pattern():
    mod = _isolated_lock_housekeeper()
    # Wiki tools write per-slug lock files: <slug>.lock
    assert mod._basename_matches_rules("self.lock", ("*.lock",))
    assert mod._basename_matches_rules("hot.lock", ("*.lock",))


def test_lock_basename_refuses_unknown_in_workspace_root():
    """A `.lock`-suffixed file at workspace root that is NOT
    `.workspace.lock` must NOT match the rules — it might be a
    future subsystem's state file with a misleading suffix."""
    mod = _isolated_lock_housekeeper()
    assert not mod._basename_matches_rules(
        "random.lock", (".workspace.lock",),
    )
    assert not mod._basename_matches_rules(
        "some_other.lock", (".workspace.lock",),
    )


def test_lock_rules_table_present():
    """The per-dir rules table must exist with the three known
    lock-using subsystems."""
    mod = _isolated_lock_housekeeper()
    assert hasattr(mod, "_LOCK_RULES")
    dirs = {str(d) for d, _ in mod._LOCK_RULES}
    assert "/app/workspace" in dirs
    assert "/app/workspace/locks" in dirs
    assert "/app/workspace/dreams" in dirs


# ─────────────────────────────────────────────────────────────────────────
# log_archival — retention floor + per-pass cap
# ─────────────────────────────────────────────────────────────────────────


def test_log_archival_retention_floor_source():
    """The floor for LOG_ARCHIVE_RETENTION_DAYS must be ≥30 days post
    2026-05-16. 7 days was too aggressive — a forgotten env could nuke
    fresh archives before the operator noticed."""
    src = (REPO_ROOT / "app" / "healing" / "monitors" / "log_archival.py").read_text()
    assert "_MIN_RETENTION_DAYS = 30" in src
    assert "max(_MIN_RETENTION_DAYS, int(raw))" in src


def test_log_archival_per_pass_cap_source():
    """The purge must cap deletions per pass so a runaway purge can't
    nuke a large backlog in one go."""
    src = (REPO_ROOT / "app" / "healing" / "monitors" / "log_archival.py").read_text()
    assert "_PURGE_PER_PASS_CAP" in src
    # Cap should be applied within the purge function.
    purge_start = src.find("def _purge_old_archives")
    purge_end = src.find("\ndef ", purge_start + 1)
    purge_body = src[purge_start:purge_end]
    assert "cap_remaining" in purge_body
    assert "deferred_due_to_cap" in purge_body


def test_log_archival_purge_sorts_oldest_first_source():
    """When the cap engages, we must delete the OLDEST candidates
    (not whatever order iterdir returns first). Otherwise newer
    archives could disappear while older ones linger."""
    src = (REPO_ROOT / "app" / "healing" / "monitors" / "log_archival.py").read_text()
    purge_start = src.find("def _purge_old_archives")
    purge_end = src.find("\ndef ", purge_start + 1)
    purge_body = src[purge_start:purge_end]
    assert "candidates.sort" in purge_body
    assert "oldest" in purge_body.lower()


# ─────────────────────────────────────────────────────────────────────────
# Source-level discipline pins (work in any env)
# ─────────────────────────────────────────────────────────────────────────


def test_source_retention_validates_worktree_path():
    """retention.py must call _validate_worktree_path BEFORE rmtree."""
    src = (REPO_ROOT / "app" / "healing" / "monitors" / "retention.py").read_text()
    assert "_validate_worktree_path" in src
    # rmtree must be guarded by the validation result
    run_worktrees_start = src.find("def run_worktrees")
    run_worktrees_end = src.find("\ndef ", run_worktrees_start + 1)
    body = src[run_worktrees_start:run_worktrees_end]
    assert "is_valid, reason = _validate_worktree_path" in body
    assert "if not is_valid:" in body
    # The actual rmtree CALL (not the substring in a doc comment) must
    # come AFTER the validation. Use the specific call site as the
    # anchor instead of the bare substring "shutil.rmtree".
    call_marker = "shutil.rmtree(str(wt)"
    validation_marker = "is_valid, reason = _validate_worktree_path"
    assert call_marker in body
    assert body.index(validation_marker) < body.index(call_marker)


def test_source_retention_validates_attachments_dir():
    """retention.py must call _is_attachments_dir_safe BEFORE
    iterating files for deletion."""
    src = (REPO_ROOT / "app" / "healing" / "monitors" / "retention.py").read_text()
    assert "_is_attachments_dir_safe" in src
    run_att_start = src.find("def run_attachments")
    run_att_end = src.find("\ndef ", run_att_start + 1)
    body = src[run_att_start:run_att_end]
    assert "is_safe, reason = _is_attachments_dir_safe" in body
    assert "if not is_safe:" in body


def test_source_lock_housekeeper_uses_per_dir_rules():
    """lock_housekeeper.py must consult _LOCK_RULES + only auto-delete
    files matching their containing dir's pattern."""
    src = (REPO_ROOT / "app" / "healing" / "monitors" / "lock_housekeeper.py").read_text()
    assert "_LOCK_RULES" in src
    assert "_basename_matches_rules" in src
    # The candidate yield must include the matches_rule flag.
    assert "yield p, _basename_matches_rules" in src
    # run() must skip unknown-shape files (not auto-delete them).
    run_start = src.find("def run()")
    run_end = src.find("\n# ", run_start) if "\n# " in src[run_start:] else len(src)
    body = src[run_start:run_end]
    assert "skipped_unknown_shape" in body


# ─────────────────────────────────────────────────────────────────────────
# Helper for isolated-load tests above
# ─────────────────────────────────────────────────────────────────────────


def _isolated_for_pure_helpers():
    """Load retention.py with the gateway-deps imports stubbed.
    Used for tests that only exercise pure helpers like
    _validate_worktree_path and _is_attachments_dir_safe."""
    stub_common = type(sys)("stub_common")
    stub_common.audit_event = lambda *a, **k: None
    stub_common.background_enabled = lambda: True
    stub_common.read_state_json = lambda *a, **k: {}
    stub_common.send_signal_alert = lambda *a, **k: None
    stub_common.write_state_json = lambda *a, **k: None
    sys.modules.setdefault("app", type(sys)("stub_app"))
    sys.modules["app.life_companion"] = type(sys)("stub_lc")
    sys.modules["app.life_companion._common"] = stub_common
    return _isolated("app/healing/monitors/retention.py", "ret_test")
