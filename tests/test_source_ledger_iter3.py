"""PROGRAM §56 iter-3 — race fix, metadata sentinel, state summary.

Covers:
  * Compaction tail-stabilization loop: rows appended during fold
    end up in the new live ledger, not just history.
  * append_update(new_metadata=None) preserves prior metadata on fold;
    explicit {} still clears.
  * Hash chain remains valid across update rows containing the
    no-change sentinel.
  * state_summary() returns expected shape per KB.
"""
from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def ledger_module(tmp_path, monkeypatch):
    import app.paths as paths
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path)
    import app.memory.source_ledger as sl
    importlib.reload(sl)
    return sl


def _make_kb(ws: Path, name: str) -> Path:
    kb = ws / name
    kb.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(kb / "chroma.sqlite3"))
    conn.execute("CREATE TABLE IF NOT EXISTS placeholder (id INTEGER)")
    conn.commit()
    conn.close()
    return kb


# ────────────────────────────────────────────────────────────────────
#   #1 — Compaction race fix
# ────────────────────────────────────────────────────────────────────


def test_compaction_picks_up_rows_appended_during_fold(ledger_module, tmp_path, monkeypatch):
    """Simulate a writer appending rows in the middle of compaction.

    Strategy: monkey-patch ``_fold_ledger_from`` to perform a side-
    effect append the first time it's called with start_offset=0.
    The tail-stabilization loop should detect the growth, re-fold,
    and include the tail rows in the rebuilt ledger.
    """
    kb = "memory"
    _make_kb(tmp_path, kb)
    # Seed: 150 adds (above MIN_ROWS) + 50 deletes so reduction
    # threshold is satisfied.
    for i in range(150):
        ledger_module.append_row(kb, "c", f"d{i}", f"v0-{i}", {})
    for i in range(50):
        ledger_module.append_delete(kb, "c", f"d{i}")

    # Patch _fold_ledger_from so the FIRST invocation (start_offset=0)
    # appends a new row to the live ledger after reading. Subsequent
    # invocations (tail passes) read normally.
    original = ledger_module._fold_ledger_from
    call_count = {"n": 0}

    def fold_with_concurrent_append(ledger, start_offset, base_state=None):
        result = original(ledger, start_offset, base_state=base_state)
        if call_count["n"] == 0:
            # Simulate writer appending during fold — runs AFTER the
            # initial read completes but BEFORE the compacted file
            # is written.
            ledger_module.append_row(kb, "c", "concurrent_doc", "appended during fold", {})
        call_count["n"] += 1
        return result

    monkeypatch.setattr(ledger_module, "_fold_ledger_from", fold_with_concurrent_append)
    result = ledger_module.compact_ledger(kb)
    assert result.ok, result.to_dict()

    # The concurrent doc must end up in the new live ledger, NOT just history.
    live_rows = list(ledger_module.read_all(kb))
    doc_ids = {(r.collection, r.doc_id) for r in live_rows}
    assert ("c", "concurrent_doc") in doc_ids, (
        "concurrent-append row missing from compacted live ledger — race not closed"
    )


def test_compaction_tail_loop_terminates_on_stable_ledger(ledger_module, tmp_path):
    """Without a concurrent writer, the tail-stabilization loop exits
    after one stat() call (no growth → no extra passes)."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(150):
        ledger_module.append_row(kb, "c", f"d{i}", f"v-{i}", {})
    for i in range(50):
        ledger_module.append_delete(kb, "c", f"d{i}")
    # No monkey-patching — should complete normally.
    result = ledger_module.compact_ledger(kb)
    assert result.ok
    # And the chain still verifies.
    assert ledger_module.verify_chain(kb).ok


def test_compaction_max_tail_passes_bound(ledger_module, tmp_path, monkeypatch):
    """Under a sustained write storm where every fold triggers more
    writes, the loop must terminate at _COMPACTION_MAX_TAIL_PASSES.
    This guarantees forward progress even at the cost of a few rows
    landing in history-only (the loop bails out and proceeds with
    swap)."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(150):
        ledger_module.append_row(kb, "c", f"d{i}", f"v-{i}", {})
    for i in range(50):
        ledger_module.append_delete(kb, "c", f"d{i}")

    # Append on every fold call.
    original = ledger_module._fold_ledger_from
    storm = {"n": 0}

    def fold_with_storm(ledger, start_offset, base_state=None):
        result = original(ledger, start_offset, base_state=base_state)
        ledger_module.append_row(kb, "c", f"storm_{storm['n']}", "x", {})
        storm["n"] += 1
        return result

    monkeypatch.setattr(ledger_module, "_fold_ledger_from", fold_with_storm)
    result = ledger_module.compact_ledger(kb)
    # Compaction MUST complete (not hang forever).
    assert result.ok
    # Verify chain on the post-compaction live ledger.
    assert ledger_module.verify_chain(kb).ok


# ────────────────────────────────────────────────────────────────────
#   #2 — Metadata None vs {} sentinel
# ────────────────────────────────────────────────────────────────────


def test_no_change_sentinel_preserves_prior_metadata_on_fold(ledger_module, tmp_path, monkeypatch):
    """Update with new_metadata=None must preserve the prior add's
    metadata when replay folds the chain — does NOT clear to empty."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "text", {"important": "value", "k": 1})
    ledger_module.append_update(kb, "c", "d1", new_text="updated text", new_metadata=None)

    # Use the public _apply_row_to_state via the inline fold logic
    # by walking read_all + applying.
    state: dict = {}
    for row in ledger_module.read_all(kb):
        ledger_module._apply_row_to_state(state, row)
    val = state.get(("c", "d1"))
    assert val is not None
    text, meta, _ = val
    assert text == "updated text"
    # Prior metadata preserved.
    assert meta == {"important": "value", "k": 1}


def test_explicit_empty_metadata_clears_on_fold(ledger_module, tmp_path):
    """Update with new_metadata={} must CLEAR metadata to empty —
    distinct from None."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "text", {"important": "value"})
    ledger_module.append_update(kb, "c", "d1", new_text="updated", new_metadata={})

    state: dict = {}
    for row in ledger_module.read_all(kb):
        ledger_module._apply_row_to_state(state, row)
    val = state.get(("c", "d1"))
    text, meta, _ = val
    assert text == "updated"
    assert meta == {}, f"explicit empty dict should clear, got {meta!r}"


def test_explicit_new_metadata_overwrites(ledger_module, tmp_path):
    """Sanity: explicit metadata dict overwrites prior."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "t", {"a": 1, "b": 2})
    ledger_module.append_update(kb, "c", "d1", new_metadata={"c": 3})

    state: dict = {}
    for row in ledger_module.read_all(kb):
        ledger_module._apply_row_to_state(state, row)
    val = state.get(("c", "d1"))
    _, meta, _ = val
    assert meta == {"c": 3}


def test_hash_chain_valid_across_no_change_updates(ledger_module, tmp_path):
    """Sentinel rows must hash-chain like any other update row.
    Tampering tests already cover regular updates; this pins the new
    sentinel path."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "v0", {"k": 1})
    ledger_module.append_update(kb, "c", "d1", new_text="v1", new_metadata=None)
    ledger_module.append_update(kb, "c", "d1", new_text="v2", new_metadata=None)
    result = ledger_module.verify_chain(kb)
    assert result.ok, result.to_dict()


def test_sentinel_is_not_user_data(ledger_module):
    """The sentinel detector must not match operator-supplied metadata
    that happens to contain the sentinel key but other keys too."""
    assert ledger_module._is_no_change_sentinel(
        {"__sl_no_change__": True}
    )
    # Sentinel key with additional operator data → NOT a sentinel.
    assert not ledger_module._is_no_change_sentinel(
        {"__sl_no_change__": True, "user_field": "value"}
    )
    # Wrong value type.
    assert not ledger_module._is_no_change_sentinel(
        {"__sl_no_change__": "yes"}
    )
    # Empty dict.
    assert not ledger_module._is_no_change_sentinel({})
    # None.
    assert not ledger_module._is_no_change_sentinel(None)


def test_replay_kb_uses_sentinel_correctly(ledger_module, tmp_path, monkeypatch):
    """End-to-end: replay must honor sentinel through the fold-back-
    to-add transformation."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "original", {"keep": "this"})
    ledger_module.append_update(kb, "c", "d1", new_text="modified", new_metadata=None)

    class _FakeCol:
        def __init__(self): self.upserts = []
        def upsert(self, *, ids, documents, metadatas, embeddings):
            self.upserts.append({"ids": ids, "documents": documents, "metadatas": metadatas})

    class _FakeClient:
        def __init__(self): self.cols = {}
        def get_or_create_collection(self, name):
            if name not in self.cols:
                self.cols[name] = _FakeCol()
            return self.cols[name]

    fake = _FakeClient()
    import sys
    class FakeMgr:
        def get_kb_client(self, name): return fake
        def embed(self, text): return [0.0] * 8
    monkeypatch.setitem(sys.modules, "app.memory.chromadb_manager", FakeMgr())

    result = ledger_module.replay_kb(kb)
    assert result.ok
    upserts = fake.cols["c"].upserts
    assert len(upserts) == 1
    assert upserts[0]["documents"] == ["modified"]
    # The metadata in the upsert must preserve "keep": "this".
    assert upserts[0]["metadatas"][0] == {"keep": "this"}


# ────────────────────────────────────────────────────────────────────
#   #3 — state_summary() shape
# ────────────────────────────────────────────────────────────────────


def test_state_summary_returns_per_kb_rows(ledger_module, tmp_path):
    kb1 = "memory"
    kb2 = "philosophy"
    _make_kb(tmp_path, kb1)
    _make_kb(tmp_path, kb2)
    for i in range(5):
        ledger_module.append_row(kb1, "c", f"d{i}", "text", {})
    for i in range(3):
        ledger_module.append_row(kb2, "c", f"d{i}", "text", {})

    summary = ledger_module.state_summary()
    names = {kb["name"] for kb in summary["kbs"]}
    assert {kb1, kb2} <= names

    by_name = {kb["name"]: kb for kb in summary["kbs"]}
    assert by_name[kb1]["ledger_rows"] == 5
    assert by_name[kb2]["ledger_rows"] == 3
    assert by_name[kb1]["chain_ok"] is True
    # No off-host state files exist → both destinations should be None.
    assert by_name[kb1]["offhost"] == {"s3": None, "gdrive": None}


def test_state_summary_reports_chain_break(ledger_module, tmp_path):
    """state_summary must surface chain integrity failures so the
    React card can show them in red."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(3):
        ledger_module.append_row(kb, "c", f"d{i}", "text", {})
    # Tamper row 1's text — chain breaks at row 1.
    path = ledger_module.ledger_path(kb)
    lines = path.read_text().strip().split("\n")
    bad = json.loads(lines[1])
    bad["text"] = "TAMPERED"
    lines[1] = json.dumps(bad, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    summary = ledger_module.state_summary()
    by_name = {kb["name"]: kb for kb in summary["kbs"]}
    assert by_name[kb]["chain_ok"] is False
    assert by_name[kb]["chain_first_bad_row"] == 1


def test_rest_endpoint_pinned():
    """Pin: the dashboard route file must declare the new endpoint.
    Without this, the React card has nothing to call."""
    src = Path("/Users/andrus/BotArmy/crewai-team/app/control_plane/dashboard_routes_budgets_costs.py")
    if not src.exists():
        src = Path("app/control_plane/dashboard_routes_budgets_costs.py")
    if not src.exists():
        pytest.skip("dashboard routes file not on canonical paths")
    text = src.read_text()
    assert '"/source-ledger/state"' in text, (
        "REST endpoint /api/cp/source-ledger/state has been removed — "
        "React SourceLedgerCard would silently 404."
    )
