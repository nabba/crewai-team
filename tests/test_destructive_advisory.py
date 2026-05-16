"""Tests for the destructive_advisory guardrail.

The guardrail exists because two monitors emitted destructive
recommendations on 2026-05-16 and both produced live-data incidents.
The dataclass enforces (at construction time) that an advisory cannot
be emitted without:

  * a snapshot that already exists on disk before construction
  * a schema-verify command the operator can run BEFORE acting
  * an explicit undo command
  * a one-line declaration of the classification assumption

These tests pin those guarantees so the discipline cannot regress to
"alert with rm -rf but no snapshot" by accident.
"""
from __future__ import annotations

import importlib.util
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "destructive_advisory_test",
        REPO_ROOT / "app" / "healing" / "destructive_advisory.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["destructive_advisory_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def da():
    return _load_module()


# ─────────────────────────────────────────────────────────────────────────
# snapshot_paths
# ─────────────────────────────────────────────────────────────────────────


def test_snapshot_paths_creates_tarball(da, tmp_path):
    """Happy path: tar contents include the target file."""
    target = tmp_path / "kb" / "aaa-bbb-ccc"
    target.mkdir(parents=True)
    (target / "data.bin").write_bytes(b"payload" * 100)

    snap = da.snapshot_paths(
        [target], dest_dir=tmp_path / "snaps", label="test_snap",
    )
    assert snap is not None
    assert snap.is_file()
    assert snap.suffix == ".gz"
    # Contents include the target file.
    with tarfile.open(snap, "r:gz") as tf:
        names = tf.getnames()
    assert any(n.endswith("data.bin") for n in names)


def test_snapshot_paths_refuses_empty_targets(da, tmp_path):
    """Discipline: empty target list must NOT silently create an
    empty snapshot — that would let a caller satisfy the advisory's
    snapshot requirement without actually snapshotting anything."""
    snap = da.snapshot_paths([], dest_dir=tmp_path, label="empty")
    assert snap is None


def test_snapshot_paths_refuses_missing_targets(da, tmp_path):
    """If any target doesn't exist on disk, refuse — don't produce a
    partial snapshot."""
    snap = da.snapshot_paths(
        [tmp_path / "does-not-exist"],
        dest_dir=tmp_path / "snaps",
        label="missing",
    )
    assert snap is None


def test_snapshot_paths_sanitises_label(da, tmp_path):
    """Path-traversal / odd chars in label are stripped so a malicious
    caller can't write outside dest_dir."""
    target = tmp_path / "kb"
    target.mkdir()
    (target / "x").write_bytes(b"x")
    snap = da.snapshot_paths(
        [target],
        dest_dir=tmp_path / "snaps",
        label="../../etc/passwd",
    )
    assert snap is not None
    # Resolved path must still be inside dest_dir.
    assert (tmp_path / "snaps") in snap.resolve().parents


# ─────────────────────────────────────────────────────────────────────────
# DestructiveAdvisory construction
# ─────────────────────────────────────────────────────────────────────────


def _valid_kwargs(da, tmp_path):
    """Build a kwargs dict for a valid DestructiveAdvisory. Returns
    (kwargs, snapshot_path)."""
    target = tmp_path / "kb" / "uuid-aaa"
    target.mkdir(parents=True)
    (target / "data.bin").write_bytes(b"x" * 1000)
    snap = da.snapshot_paths(
        [target], dest_dir=tmp_path / "snaps", label="valid",
    )
    assert snap is not None
    return {
        "monitor_name": "test_monitor",
        "summary": "Test summary",
        "targets": [target],
        "snapshot_path": snap,
        "apply_command": "rm -rf <target>",
        "undo_command": f"tar -xzf {snap}",
        "verify_command": "echo verify",
        "schema_assumption": "test assumption",
    }, snap


def test_advisory_constructs_with_full_kwargs(da, tmp_path):
    kwargs, _ = _valid_kwargs(da, tmp_path)
    adv = da.DestructiveAdvisory(**kwargs)
    assert adv.monitor_name == "test_monitor"


def test_advisory_refuses_missing_snapshot(da, tmp_path):
    """The load-bearing discipline: snapshot_path MUST exist on disk
    at construction time. A monitor cannot emit an advisory without
    first taking the snapshot."""
    kwargs, _ = _valid_kwargs(da, tmp_path)
    kwargs["snapshot_path"] = tmp_path / "nope.tar.gz"
    with pytest.raises(ValueError, match="snapshot_path"):
        da.DestructiveAdvisory(**kwargs)


def test_advisory_refuses_tiny_snapshot(da, tmp_path):
    """A 0-byte file passes 'exists' but isn't a real snapshot. The
    dataclass catches this so a caller can't pre-touch the path and
    claim success."""
    kwargs, snap = _valid_kwargs(da, tmp_path)
    snap.write_bytes(b"")
    with pytest.raises(ValueError, match="suspiciously small"):
        da.DestructiveAdvisory(**kwargs)


@pytest.mark.parametrize("missing_field", [
    "monitor_name",
    "summary",
    "apply_command",
    "undo_command",
    "verify_command",
    "schema_assumption",
])
def test_advisory_refuses_empty_required_string(da, tmp_path, missing_field):
    """Every discipline field must be non-empty."""
    kwargs, _ = _valid_kwargs(da, tmp_path)
    kwargs[missing_field] = ""
    with pytest.raises(ValueError, match=missing_field):
        da.DestructiveAdvisory(**kwargs)


def test_advisory_refuses_empty_targets(da, tmp_path):
    kwargs, _ = _valid_kwargs(da, tmp_path)
    kwargs["targets"] = []
    with pytest.raises(ValueError, match="targets"):
        da.DestructiveAdvisory(**kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Format
# ─────────────────────────────────────────────────────────────────────────


def test_format_contains_all_discipline_fields(da, tmp_path):
    """The formatted body must visibly include: verify command, apply
    command, undo command, snapshot path, schema assumption. These
    are the four fields whose absence caused the 2026-05-16 incidents."""
    kwargs, snap = _valid_kwargs(da, tmp_path)
    kwargs["verify_command"] = "VERIFY-MARKER"
    kwargs["apply_command"] = "APPLY-MARKER"
    kwargs["undo_command"] = "UNDO-MARKER"
    kwargs["schema_assumption"] = "ASSUMPTION-MARKER"
    adv = da.DestructiveAdvisory(**kwargs)
    body = adv.format()
    assert "VERIFY-MARKER" in body
    assert "APPLY-MARKER" in body
    assert "UNDO-MARKER" in body
    assert "ASSUMPTION-MARKER" in body
    assert str(snap) in body
    # The body must put the verify step BEFORE the apply step — the
    # discipline is "verify before act".
    assert body.index("VERIFY-MARKER") < body.index("APPLY-MARKER")


def test_format_truncates_long_target_list(da, tmp_path):
    """With many targets, the body shows only a preview + count so
    Signal messages stay readable."""
    targets = []
    for i in range(20):
        t = tmp_path / "kb" / f"target-{i:03d}"
        t.mkdir(parents=True)
        (t / "x").write_bytes(b"x" * 100)
        targets.append(t)
    snap = da.snapshot_paths(
        targets, dest_dir=tmp_path / "snaps", label="bulk",
    )
    kwargs, _ = _valid_kwargs(da, tmp_path)
    kwargs["targets"] = targets
    kwargs["snapshot_path"] = snap
    adv = da.DestructiveAdvisory(**kwargs)
    body = adv.format()
    assert "20" in body  # total count
    assert "+17 more" in body  # 20 - 3 shown = 17


# ─────────────────────────────────────────────────────────────────────────
# emit
# ─────────────────────────────────────────────────────────────────────────


def test_emit_calls_send_fn_with_body_and_tag(da, tmp_path):
    kwargs, _ = _valid_kwargs(da, tmp_path)
    adv = da.DestructiveAdvisory(**kwargs)

    sent: list[dict] = []

    def fake_send(body, *, tag):
        sent.append({"body": body, "tag": tag})

    audits: list[dict] = []

    def fake_audit(event, **kw):
        audits.append({"event": event, **kw})

    da.emit(adv, send_fn=fake_send, audit_fn=fake_audit)
    assert len(sent) == 1
    assert sent[0]["tag"] == "destructive_advisory:test_monitor"
    assert "test_monitor" not in sent[0]["body"] or "Test summary" in sent[0]["body"]
    assert len(audits) == 1
    assert audits[0]["event"] == "destructive_advisory_emitted"
    assert audits[0]["monitor"] == "test_monitor"


def test_emit_uses_custom_tag(da, tmp_path):
    kwargs, _ = _valid_kwargs(da, tmp_path)
    adv = da.DestructiveAdvisory(**kwargs)
    sent: list[dict] = []
    da.emit(
        adv,
        send_fn=lambda body, *, tag: sent.append({"body": body, "tag": tag}),
        audit_fn=lambda event, **kw: None,
        tag="custom_tag",
    )
    assert sent[0]["tag"] == "custom_tag"


def test_emit_send_failure_does_not_block_audit(da, tmp_path):
    """If the alert send blows up, audit still records the attempt so
    the operator has a trace of what would have been emitted."""
    kwargs, _ = _valid_kwargs(da, tmp_path)
    adv = da.DestructiveAdvisory(**kwargs)
    audits: list[dict] = []

    def fake_send(body, *, tag):
        raise RuntimeError("simulated send failure")

    def fake_audit(event, **kw):
        audits.append({"event": event, **kw})

    # Must not raise.
    da.emit(adv, send_fn=fake_send, audit_fn=fake_audit)
    assert len(audits) == 1
