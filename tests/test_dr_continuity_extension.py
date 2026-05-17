"""Tests for the DR continuity extension — productization plan T2.3.

Verifies that the DR bundle includes the three new continuity items
(SubIA state, runtime_settings, SubIA integrity manifest) and that
the boot drill surfaces the SubIA continuity check.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings  # noqa: E402
import app.config as config_mod  # noqa: E402

config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


def _load_export_kbs():
    """Load export_kbs in isolation so we don't trigger boot side-effects."""
    spec = importlib.util.spec_from_file_location(
        "dr_export_t23", str(Path(__file__).parent.parent / "app/dr/export_kbs.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestLedgerIncludes:
    """The new continuity items are part of the export allowlist."""

    def test_subia_in_includes(self):
        mod = _load_export_kbs()
        assert "subia/" in mod._LEDGER_INCLUDES

    def test_subia_integrity_manifest_in_includes(self):
        mod = _load_export_kbs()
        assert ".subia_integrity.json" in mod._LEDGER_INCLUDES

    def test_runtime_settings_in_includes(self):
        mod = _load_export_kbs()
        assert "runtime_settings.json" in mod._LEDGER_INCLUDES

    def test_identity_still_in_includes(self):
        """Regression: don't drop pre-existing entries."""
        mod = _load_export_kbs()
        assert "identity/" in mod._LEDGER_INCLUDES
        assert "affect/" in mod._LEDGER_INCLUDES


class TestSecretDenylistStillBlocks:
    """The new includes must not weaken the secret denylist."""

    def test_runtime_settings_with_secret_substring_blocked(self):
        mod = _load_export_kbs()
        # A path like 'runtime_settings.json.private_key' would be secret-shaped
        assert mod._is_secret_path("runtime_settings.private_key.bak")

    def test_subia_with_token_substring_blocked(self):
        mod = _load_export_kbs()
        assert mod._is_secret_path("subia/oauth_token.json")

    def test_plain_subia_state_not_blocked(self):
        mod = _load_export_kbs()
        assert not mod._is_secret_path("subia/self/kernel-state.md")
        assert not mod._is_secret_path("subia/workspace/hot.md")


class TestExportManifestCarriesSubiaIntegrity:
    """The manifest dataclass has the new subia_integrity_at_export field."""

    def test_manifest_has_subia_field(self):
        mod = _load_export_kbs()
        m = mod.ExportManifest()
        assert hasattr(m, "subia_integrity_at_export")
        assert isinstance(m.subia_integrity_at_export, dict)

    def test_manifest_serializes_subia_field(self):
        mod = _load_export_kbs()
        m = mod.ExportManifest()
        m.subia_integrity_at_export = {
            "ok": False, "has_drift": True, "n_files": 164,
            "n_mismatched": 1, "n_extra": 2, "n_missing": 0,
        }
        d = m.to_dict()
        assert "subia_integrity_at_export" in d
        assert d["subia_integrity_at_export"]["n_files"] == 164


class TestDrillReportCarriesSubiaContinuity:
    """The DrillReport dataclass carries the new subia_continuity field."""

    def test_report_has_subia_continuity_field(self):
        # Load boot_drill in isolation
        spec = importlib.util.spec_from_file_location(
            "dr_drill_t23", str(Path(__file__).parent.parent / "app/dr/boot_drill.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        r = mod.DrillReport()
        assert hasattr(r, "subia_continuity")
        assert isinstance(r.subia_continuity, dict)
        d = r.to_dict()
        assert "subia_continuity" in d


def _load_boot_drill():
    spec = importlib.util.spec_from_file_location(
        "dr_drill_numpy_pin",
        str(Path(__file__).parent.parent / "app/dr/boot_drill.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _StubCollection:
    def __init__(self, peek_result, count=0):
        self._peek_result = peek_result
        self._count = count

    def count(self):
        return self._count

    def peek(self, n):
        return self._peek_result

    def query(self, **_):
        return {"documents": [["doc"]]}


class _StubClient:
    def __init__(self, col):
        self._col = col

    def get_collection(self, name):
        return self._col


class TestPeekDoesNotRaiseOnNumpyEmptyArray2026_05_17_Regression:
    """Pin for the 2026-05-17 numpy-truth-value-ambiguous bug.

    ChromaDB ``Collection.peek()`` returns numpy arrays for the
    ``embeddings`` field. ``if embs:`` / ``if embs[0]:`` on a numpy
    array raises ``ValueError: The truth value of an empty array is
    ambiguous`` (or "with more than one element is ambiguous"). The
    drill must use length/size checks, never truthiness. Removing this
    pin would re-enable the alert storm we hit on 2026-05-17.
    """

    def test_peek_empty_collection_does_not_raise(self, monkeypatch):
        np = pytest.importorskip("numpy")
        mod = _load_boot_drill()
        # Empty collection: peek returns numpy arrays of shape (0,) or (0, dim).
        empty_peek = {"embeddings": np.array([], dtype=float).reshape(0, 0)}
        col = _StubCollection(peek_result=empty_peek, count=0)

        class _FakeChromadb:
            PersistentClient = staticmethod(lambda path: _StubClient(col))

        monkeypatch.setitem(sys.modules, "chromadb", _FakeChromadb)
        res = mod._drill_chromadb_collection(Path("/tmp/fake_kb"), "c", expected_rows=0)
        assert res.ok is True, f"empty collection should drill clean, got error: {res.error}"
        assert res.error is None
        assert res.peek_dim is None  # nothing to measure
        assert res.observed_rows == 0

    def test_peek_populated_collection_uses_size_not_truthiness(self, monkeypatch):
        np = pytest.importorskip("numpy")
        mod = _load_boot_drill()
        # Populated: peek returns numpy 2D array shape (1, 4). bool(np.array([...]))
        # with >1 element ALSO raises — so the fix must avoid truthiness here too.
        emb = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float)
        col = _StubCollection(peek_result={"embeddings": emb}, count=1)

        class _FakeChromadb:
            PersistentClient = staticmethod(lambda path: _StubClient(col))

        monkeypatch.setitem(sys.modules, "chromadb", _FakeChromadb)
        res = mod._drill_chromadb_collection(Path("/tmp/fake_kb"), "c", expected_rows=1)
        assert res.ok is True, f"populated collection should drill clean, got: {res.error}"
        assert res.peek_dim == 4
        assert res.smoke_retrieve_ok is True


class TestErrorsAreAggregatedFromCollectionResults2026_05_17_Regression:
    """Pin for the misleading "0 errors" alert text.

    Per-collection drill failures live on ``CollectionDrillResult.error``.
    Before 2026-05-17 they were never copied into ``DrillReport.errors``,
    so the alert template (`f"{len(report.errors)} errors"`) always read
    zero even when every collection failed. The aggregation lives in the
    drill main loop, not the result type, but the surface contract is:
    if any chromadb_results[*].ok is False AND that result has an error,
    the message must end up in report.errors. The simplest way to pin
    that contract without spinning up real chromadb is a static check on
    the boot_drill source.
    """

    def test_per_collection_errors_appended_to_report_errors(self):
        src = (Path(__file__).parent.parent / "app/dr/boot_drill.py").read_text()
        # The aggregation block follows the per-collection append. We
        # assert that within the drill loop, when res.ok is False, the
        # error is added to report.errors.
        assert "report.errors.append(" in src
        # And specifically that the chromadb path does it (not just the
        # outer chromadb_list error branch).
        idx = src.find("report.chromadb_results.append(res)")
        assert idx > 0
        window = src[idx : idx + 400]
        assert "report.errors.append" in window, (
            "per-collection chromadb errors must be aggregated into "
            "report.errors next to the chromadb_results append"
        )


class TestSendDrillAlertUsesArbiter2026_05_17_Regression:
    """Pin for the alert-storm dedup fix.

    Before 2026-05-17 `_send_drill_alert` called
    `life_companion._common.send_signal_alert` directly, bypassing the
    surface arbiter. A broken drill firing on every scheduler tick
    produced 50+ identical Signal alerts within an hour. The alert
    must now route through `app.notify.notify` with a stable topic and
    `arbitrate=True`.
    """

    def test_alert_routes_through_app_notify_notify(self):
        src = (Path(__file__).parent.parent / "app/dr/boot_drill.py").read_text()
        # Source must reference the deduped notify path.
        assert "from app.notify import notify" in src
        # And must not reuse the raw send_signal_alert path.
        assert "send_signal_alert" not in src, (
            "boot_drill must not bypass the notify arbiter — alert "
            "storms result. Use notify(arbitrate=True, topic=...) "
            "instead."
        )
        # Topic + arbitrate must both appear in the alert path.
        idx = src.find("def _send_drill_alert(")
        assert idx > 0
        body = src[idx : idx + 1500]
        assert "arbitrate=True" in body
        assert "topic=" in body
