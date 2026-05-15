"""PROGRAM §45.2 — Q7.2 schema-aware coding-session submit tests.

Covers:
  * detect_schema_changes — path-only detection
  * No false positives on routine Python files
  * Migration-name inference
  * Next-number computation continues existing convention
  * Stub renderer includes session id + paths + TODO
  * Submit pipeline emits a synthesized migration CR
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sm():
    return _load_isolated(
        "schema_migrations_q72",
        "app/coding_session/schema_migrations.py",
    )


# ─────────────────────────────────────────────────────────────────────────
#   Path-only detection — positive cases
# ─────────────────────────────────────────────────────────────────────────


def test_detect_control_plane_db(sm, tmp_path):
    """Edit to app/control_plane/db.py → hint produced."""
    hint = sm.detect_schema_changes(
        ["app/control_plane/db.py", "app/some_other.py"],
        migrations_dir=tmp_path,
    )
    assert hint is not None
    assert "app/control_plane/db.py" in hint.detected_paths
    assert hint.next_migration_number == 1
    assert hint.suggested_filename.endswith(".sql")
    assert "control_plane_db" in hint.inferred_name


def test_detect_migrations_dir_change(sm, tmp_path):
    """Direct edit of a migration file → hint produced."""
    hint = sm.detect_schema_changes(
        ["migrations/0042_existing.sql"],
        migrations_dir=tmp_path,
    )
    assert hint is not None
    assert "migrations/0042_existing.sql" in hint.detected_paths
    assert "amend_migration_0042_existing" in hint.inferred_name


def test_detect_multiple_schema_paths(sm, tmp_path):
    """Multiple schema-owning paths → all matched."""
    hint = sm.detect_schema_changes(
        [
            "app/control_plane/db.py",
            "app/affect/db.py",
            "app/agents/foo.py",  # not schema-owning
        ],
        migrations_dir=tmp_path,
    )
    assert hint is not None
    assert "app/control_plane/db.py" in hint.detected_paths
    assert "app/affect/db.py" in hint.detected_paths
    assert "app/agents/foo.py" not in hint.detected_paths


# ─────────────────────────────────────────────────────────────────────────
#   Negative cases (no false positives)
# ─────────────────────────────────────────────────────────────────────────


def test_no_hint_for_pure_python_change(sm, tmp_path):
    """Diff containing only routine .py files → no hint."""
    assert sm.detect_schema_changes(
        ["app/tools/web_search.py", "app/agents/commander/foo.py"],
        migrations_dir=tmp_path,
    ) is None


def test_no_hint_for_tests(sm, tmp_path):
    """Test files → no hint."""
    assert sm.detect_schema_changes(
        ["tests/test_foo.py"],
        migrations_dir=tmp_path,
    ) is None


def test_no_hint_for_empty_diff(sm, tmp_path):
    assert sm.detect_schema_changes([], migrations_dir=tmp_path) is None


def test_no_hint_for_python_mentioning_create_table_in_string(sm, tmp_path):
    """Path-only detection means a .py file containing 'CREATE TABLE'
    in a string literal doesn't trip the detector (this is the key
    discipline from operator decision: path-only for v1)."""
    assert sm.detect_schema_changes(
        ["app/some_module.py"],  # mentions CREATE TABLE in docstring
        migrations_dir=tmp_path,
    ) is None


# ─────────────────────────────────────────────────────────────────────────
#   Next-migration-number continues existing convention
# ─────────────────────────────────────────────────────────────────────────


def test_next_migration_number_continues_existing(sm, tmp_path):
    """Existing 0001, 0002, 0005 → next is 0006."""
    (tmp_path / "0001_init.sql").write_text("-- noop\n")
    (tmp_path / "0002_add_users.sql").write_text("-- noop\n")
    (tmp_path / "0005_add_index.sql").write_text("-- noop\n")
    assert sm._next_migration_number(tmp_path) == 6


def test_next_migration_number_when_empty(sm, tmp_path):
    assert sm._next_migration_number(tmp_path) == 1


def test_next_migration_number_ignores_non_numeric(sm, tmp_path):
    (tmp_path / "README.md").write_text("\n")
    (tmp_path / "0003_foo.sql").write_text("-- noop\n")
    assert sm._next_migration_number(tmp_path) == 4


def test_next_migration_number_missing_dir(sm, tmp_path):
    nonexistent = tmp_path / "nope"
    assert sm._next_migration_number(nonexistent) == 1


# ─────────────────────────────────────────────────────────────────────────
#   Migration name inference
# ─────────────────────────────────────────────────────────────────────────


def test_inferred_name_is_snake_case(sm):
    name = sm._infer_migration_name(["app/control_plane/db.py"])
    assert "control_plane_db" == name


def test_inferred_name_combines_multiple_paths(sm):
    name = sm._infer_migration_name([
        "app/affect/db.py", "app/budgets/db.py",
    ])
    assert "affect_db" in name and "budgets_db" in name


def test_inferred_name_capped_at_80_chars(sm):
    very_long = [f"app/very/long/path/component_{i}/db.py" for i in range(10)]
    name = sm._infer_migration_name(very_long)
    assert len(name) <= 80


def test_inferred_name_for_migration_amendment(sm):
    name = sm._infer_migration_name(["migrations/0042_some_existing.sql"])
    assert name.startswith("amend_migration_")


# ─────────────────────────────────────────────────────────────────────────
#   Stub renderer
# ─────────────────────────────────────────────────────────────────────────


def test_stub_includes_session_id_paths_and_todo(sm, tmp_path):
    hint = sm.detect_schema_changes(
        ["app/control_plane/db.py"], migrations_dir=tmp_path,
    )
    stub = sm.render_migration_stub(
        hint, session_id="cs_test_42",
        purpose="add user profile column",
    )
    assert "cs_test_42" in stub
    assert "app/control_plane/db.py" in stub
    assert "TODO" in stub
    assert "BEGIN" in stub and "COMMIT" in stub
    assert "PROGRAM §45.2 Q7.2" in stub


def test_stub_has_idempotency_hint(sm, tmp_path):
    hint = sm.detect_schema_changes(
        ["app/control_plane/db.py"], migrations_dir=tmp_path,
    )
    stub = sm.render_migration_stub(
        hint, session_id="x", purpose="y",
    )
    # The renderer's convention block reminds the operator about
    # idempotency.
    assert "idempotent" in stub.lower() or "IF NOT EXISTS" in stub


# ─────────────────────────────────────────────────────────────────────────
#   Submit pipeline integration
# ─────────────────────────────────────────────────────────────────────────


def test_submit_module_imports_schema_migrations():
    """Source-level: submit.py imports the schema-migrations module
    inside the submit_session function. The detect→render→synthesize
    flow is unit-tested above; this asserts the wiring exists."""
    src = Path("app/coding_session/submit.py").read_text()
    assert (
        "from app.coding_session.schema_migrations import" in src
        and "detect_schema_changes" in src
        and "render_migration_stub" in src
    )
    # The synthesized-file helper exists.
    assert "_submit_one_synthesized_file" in src
    # The hint check is INSIDE submit_session (not at module top).
    submit_idx = src.find("def submit_session(")
    hint_idx = src.find("detect_schema_changes(changed_paths_only)")
    assert submit_idx < hint_idx, "schema detection must run inside submit_session"


def test_synthesized_file_helper_passes_empty_old_content():
    """The synthesized-file helper builds a CR with empty old_content
    (since the migration file is brand-new and the operator hasn't
    written it yet)."""
    src = Path("app/coding_session/submit.py").read_text()
    helper_start = src.find("def _submit_one_synthesized_file(")
    helper_end = src.find("\ndef ", helper_start + 1)
    body = src[helper_start:helper_end]
    assert 'old_content=""' in body
    assert "synthesized" in body.lower()
