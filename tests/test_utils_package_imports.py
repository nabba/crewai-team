"""Regression: ``app.utils`` must export every legacy symbol from
the pre-package ``app/utils.py`` module.

Pre-fix shape (the operator-reported bug, 2026-05-10):

  React Evolution Monitor → Genealogy tab showed:
    cannot import name 'now_iso' from 'app.utils'
    (/app/app/utils/__init__.py)

  Two things existed at the same import path:
    - app/utils.py             (legacy module — defined now_iso etc.)
    - app/utils/__init__.py    (new package — docstring-only)

  Python prefers packages over modules with the same name, so
  ``from app.utils import now_iso`` failed at module-load time
  for every caller (variant_archive.py, firebase/infra.py, plus
  ~25 lazy callers inside try/except blocks).

Post-fix:
  app/utils.py is deleted; its contents are merged into
  app/utils/__init__.py. All existing imports
  (``from app.utils import now_iso``,
   ``from app.utils import safe_json_parse``,
   ``from app.utils import feed_parser``) continue to work.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Layout contracts ────────────────────────────────────────────────


class TestNoShadowedModule:
    """The legacy module file MUST NOT come back. If a future PR
    re-adds app/utils.py, Python will silently shadow the package
    again — same symptom as before."""

    def test_app_utils_py_does_not_exist(self) -> None:
        legacy = _REPO_ROOT / "app" / "utils.py"
        assert not legacy.exists(), (
            f"{legacy} re-introduced — Python would shadow the "
            f"app/utils/ package. Move the contents into "
            f"app/utils/__init__.py instead."
        )

    def test_app_utils_package_exists(self) -> None:
        pkg_init = _REPO_ROOT / "app" / "utils" / "__init__.py"
        assert pkg_init.exists()


# ── Public symbol contract ─────────────────────────────────────────


class TestLegacySymbolsImportable:
    """Every symbol the legacy module exported must be importable
    from app.utils now."""

    def test_now_iso_importable(self) -> None:
        from app.utils import now_iso
        result = now_iso()
        # ISO 8601 with timezone — should at least parse back as datetime.
        assert "T" in result, f"unexpected format: {result!r}"
        assert "+" in result or result.endswith("Z"), (
            f"timestamp must include UTC offset; got {result!r}"
        )

    def test_safe_json_parse_importable(self) -> None:
        from app.utils import safe_json_parse
        result, err = safe_json_parse('{"a": 1}')
        assert err == ""
        assert result == {"a": 1}

    def test_load_json_file_importable(self) -> None:
        from app.utils import load_json_file
        # default-on-missing behaviour
        out = load_json_file(Path("/does/not/exist.json"), default={})
        assert out == {}

    def test_save_json_file_importable(self, tmp_path: Path) -> None:
        from app.utils import save_json_file
        path = tmp_path / "test.json"
        ok = save_json_file(path, {"a": 1})
        assert ok is True
        assert path.exists()

    def test_truncate_importable(self) -> None:
        from app.utils import truncate
        assert truncate("hello world", max_len=5) == "hello"
        assert truncate("", max_len=10) == ""


# ── Submodule re-exports ───────────────────────────────────────────


class TestSubmoduleReexports:
    """The package's three sibling modules — feed_parser /
    hash_embedding / jsonl_retention — must remain importable both
    via the qualified path and as attribute access on the package."""

    def test_feed_parser_attribute_access(self) -> None:
        from app.utils import feed_parser
        assert feed_parser is not None

    def test_hash_embedding_attribute_access(self) -> None:
        from app.utils import hash_embedding
        assert hash_embedding is not None

    def test_jsonl_retention_attribute_access(self) -> None:
        from app.utils import jsonl_retention
        assert jsonl_retention is not None


# ── Top-level callers must load (the original failure mode) ────────


class TestTopLevelImportSitesLoad:
    """The bug surfaced because two callers had a TOP-LEVEL
    ``from app.utils import now_iso`` (not lazy / try-except).
    Confirm both modules now import cleanly."""

    def test_variant_archive_imports(self) -> None:
        # variant_archive.py:17 had: from app.utils import now_iso
        # at module top-level → broke at import time, breaking the
        # Evolution Monitor.
        import importlib
        # Reload to make sure we're testing the actual import path,
        # not a cached module.
        if "app.variant_archive" in __import__("sys").modules:
            importlib.reload(__import__("sys").modules["app.variant_archive"])
        else:
            import app.variant_archive  # noqa: F401

    def test_firebase_infra_module_loads(self) -> None:
        # firebase/infra.py:82 has `from app.utils import now_iso`
        # inside a function (lazy). The module itself must still
        # import cleanly so callers don't fail at attribute lookup.
        import app.firebase.infra  # noqa: F401
