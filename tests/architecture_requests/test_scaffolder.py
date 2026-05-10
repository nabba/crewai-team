"""Tests for app.architecture_requests.scaffolder."""

from __future__ import annotations

from pathlib import Path

from app.architecture_requests import scaffolder
from app.architecture_requests.models import FileSpec
from .conftest import make_request


def test_scaffold_writes_each_file_layout_entry(tmp_path: Path) -> None:
    req = make_request(
        file_layout=[
            FileSpec(path="app/inquiry/__init__.py", purpose="public surface"),
            FileSpec(path="app/inquiry/composer.py", purpose="runs the inquiry"),
            FileSpec(path="app/inquiry/questions.py", purpose="curated list"),
        ],
    )
    out = scaffolder.scaffold(req, base=tmp_path)
    assert out.exists()
    assert (out / "app/inquiry/__init__.py").exists()
    assert (out / "app/inquiry/composer.py").exists()
    assert (out / "app/inquiry/questions.py").exists()


def test_scaffold_writes_manifest(tmp_path: Path) -> None:
    req = make_request()
    out = scaffolder.scaffold(req, base=tmp_path)
    manifest = (out / "MANIFEST.md").read_text()
    assert req.intent in manifest
    assert req.package_path in manifest
    assert "## Test plan" in manifest
    for fs in req.file_layout:
        assert fs.path in manifest


def test_scaffold_uses_explicit_initial_stub_when_present(tmp_path: Path) -> None:
    req = make_request(
        file_layout=[
            FileSpec(
                path="app/inquiry/__init__.py",
                purpose="surface",
                initial_stub='"""custom stub content."""\n',
            ),
        ],
    )
    out = scaffolder.scaffold(req, base=tmp_path)
    content = (out / "app/inquiry/__init__.py").read_text()
    assert content == '"""custom stub content."""\n'


def test_default_init_stub_is_minimal_docstring(tmp_path: Path) -> None:
    req = make_request(
        file_layout=[
            FileSpec(path="app/inquiry/__init__.py", purpose="public surface"),
        ],
    )
    out = scaffolder.scaffold(req, base=tmp_path)
    content = (out / "app/inquiry/__init__.py").read_text()
    assert content.startswith('"""public surface"""')
    assert "raise NotImplementedError" not in content


def test_default_module_stub_raises_not_implemented(tmp_path: Path) -> None:
    req = make_request(
        file_layout=[
            FileSpec(path="app/inquiry/composer.py", purpose="runs the inquiry"),
        ],
    )
    out = scaffolder.scaffold(req, base=tmp_path)
    content = (out / "app/inquiry/composer.py").read_text()
    assert "runs the inquiry" in content
    assert "raise NotImplementedError" in content


def test_scaffold_is_idempotent(tmp_path: Path) -> None:
    req = make_request()
    out1 = scaffolder.scaffold(req, base=tmp_path)
    out2 = scaffolder.scaffold(req, base=tmp_path)
    assert out1 == out2
    # Files still present, no error.
    for fs in req.file_layout:
        assert (out1 / fs.path).exists()


def test_scaffold_paths_stay_within_staging_dir(tmp_path: Path) -> None:
    req = make_request()
    out = scaffolder.scaffold(req, base=tmp_path)
    # Every written file is under the staging dir; nothing escapes.
    for written in out.rglob("*"):
        if written.is_file():
            assert tmp_path in written.resolve().parents


def test_staging_dir_for_returns_canonical_path(tmp_path: Path) -> None:
    p = scaffolder.staging_dir_for("abc-123", base=tmp_path)
    assert p == tmp_path / "abc-123" / "scaffold"
