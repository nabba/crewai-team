"""Tests for the remaining RISK-class items from the 2026-05-16 audit.

  1. chromadb_hygiene — consecutive-failure tracking + chronic alert
  2. db_vacuum       — consecutive-failure tracking + chronic alert
  3. tz_drift        — synthesised real diff in the CR (not empty)

Source-level pins for the integration sites + pure-function tests
for the new synth helper. All run without gateway-deps.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_common():
    stub = type(sys)("stub_common")
    stub.audit_event = lambda *a, **k: None
    stub.background_enabled = lambda: True
    stub.read_state_json = lambda *a, **k: {}
    stub.send_signal_alert = lambda *a, **k: None
    stub.write_state_json = lambda *a, **k: None
    sys.modules.setdefault("app", type(sys)("stub_app"))
    sys.modules["app.life_companion"] = type(sys)("stub_lc")
    sys.modules["app.life_companion._common"] = stub
    return stub


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
# (1) chromadb_hygiene — consecutive-failure tracking
# ─────────────────────────────────────────────────────────────────────────


def test_chromadb_hygiene_failure_threshold_constant_present():
    src = (REPO_ROOT / "app/healing/monitors/chromadb_hygiene.py").read_text()
    assert "_FAILURE_ALERT_THRESHOLD" in src
    # Threshold should be a small positive int (current value = 4)
    assert "_FAILURE_ALERT_THRESHOLD = 4" in src


def test_chromadb_hygiene_tracks_per_path_failures():
    """The state dict must accumulate failures per chroma.sqlite3 path
    (not a global counter — different KBs can lock independently)."""
    src = (REPO_ROOT / "app/healing/monitors/chromadb_hygiene.py").read_text()
    assert 'state.get("consecutive_failures")' in src
    # Must keyed by path string (per-file granularity).
    assert "consecutive[path]" in src or "consecutive[info" in src


def test_chromadb_hygiene_resets_on_success():
    """A successful VACUUM must reset the per-path counter. Without
    reset the alert would fire forever after one bad streak."""
    src = (REPO_ROOT / "app/healing/monitors/chromadb_hygiene.py").read_text()
    # The reset logic should pop on success.
    assert "consecutive.pop(path" in src


def test_chromadb_hygiene_alerts_only_at_threshold_boundary():
    """The chronic-failure alert must fire ONCE when consecutive
    reaches the threshold — not on every subsequent pass. Otherwise
    operator gets daily spam."""
    src = (REPO_ROOT / "app/healing/monitors/chromadb_hygiene.py").read_text()
    # Pattern: `if consecutive[path] == _FAILURE_ALERT_THRESHOLD`
    # (equal, not >=) so it fires once on crossing.
    assert "== _FAILURE_ALERT_THRESHOLD" in src


# ─────────────────────────────────────────────────────────────────────────
# (2) db_vacuum — consecutive-failure tracking
# ─────────────────────────────────────────────────────────────────────────


def test_db_vacuum_failure_threshold_constant_present():
    src = (REPO_ROOT / "app/healing/monitors/db_vacuum.py").read_text()
    assert "_FAILURE_ALERT_THRESHOLD" in src
    assert "_FAILURE_ALERT_THRESHOLD = 4" in src


def test_db_vacuum_resets_on_success():
    src = (REPO_ROOT / "app/healing/monitors/db_vacuum.py").read_text()
    # On success: reset to 0
    assert 'state["consecutive_failures"] = 0' in src


def test_db_vacuum_increments_on_failure():
    src = (REPO_ROOT / "app/healing/monitors/db_vacuum.py").read_text()
    # On failure: increment from prev_failures
    assert "prev_failures + 1" in src


def test_db_vacuum_alerts_once_at_threshold():
    src = (REPO_ROOT / "app/healing/monitors/db_vacuum.py").read_text()
    assert "newly_chronic" in src
    assert 'tag="db_vacuum_chronic_failure"' in src


# ─────────────────────────────────────────────────────────────────────────
# (3) tz_drift — real synthetic diff (not empty)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tz_drift_mod():
    """Load tz_drift.py with gateway-deps stubbed."""
    _stub_common()
    return _load("app/healing/monitors/tz_drift.py", "tz_drift_test")


def test_synth_helper_produces_real_diff(tz_drift_mod):
    """The synth helper must produce a non-empty diff against the
    current temporal_context.py — that's the whole point of the fix."""
    src = (REPO_ROOT / "app/temporal_context.py").read_text()
    new_src = tz_drift_mod._synthesize_zoneinfo_patch(src)
    assert new_src is not None
    assert new_src != src, (
        "synth helper must produce a different source than the input. "
        "Empty diff = the original RISK item (CR with no real patch)."
    )


def test_synth_helper_output_parses(tz_drift_mod):
    """The synthesized source must be valid Python — otherwise a
    well-meaning operator would approve a CR that ships broken
    code."""
    src = (REPO_ROOT / "app/temporal_context.py").read_text()
    new_src = tz_drift_mod._synthesize_zoneinfo_patch(src)
    assert new_src is not None
    ast.parse(new_src)


def test_synth_helper_adds_zoneinfo_import(tz_drift_mod):
    src = (REPO_ROOT / "app/temporal_context.py").read_text()
    new_src = tz_drift_mod._synthesize_zoneinfo_patch(src)
    assert new_src is not None
    assert "from zoneinfo import ZoneInfo" in new_src


def test_synth_helper_uses_zoneinfo_in_function(tz_drift_mod):
    src = (REPO_ROOT / "app/temporal_context.py").read_text()
    new_src = tz_drift_mod._synthesize_zoneinfo_patch(src)
    assert new_src is not None
    assert 'ZoneInfo("Europe/Helsinki")' in new_src


def test_synth_helper_refuses_when_function_missing(tz_drift_mod):
    """If temporal_context.py is refactored such that _helsinki_tz no
    longer has the expected signature, the synth helper must refuse
    rather than silently produce nonsense."""
    src = "# refactored away\nfrom datetime import datetime\n"
    result = tz_drift_mod._synthesize_zoneinfo_patch(src)
    assert result is None


def test_synth_helper_refuses_when_import_missing(tz_drift_mod):
    """The synth needs to find the existing `from datetime import`
    line to anchor the new ZoneInfo import. If that line moves, the
    helper refuses rather than putting the import in the wrong place."""
    # Function present but no canonical datetime import line.
    src = (
        "from datetime import (\n"
        "    datetime, timedelta, timezone,\n"
        ")\n"
        "\n"
        "def _helsinki_tz() -> timezone:\n"
        "    return timezone.utc\n"
    )
    result = tz_drift_mod._synthesize_zoneinfo_patch(src)
    assert result is None


def test_tz_drift_cr_uses_real_diff_at_call_site():
    """Source-level pin: _propose_consolidation_cr must call
    _synthesize_zoneinfo_patch and pass the result as new_content
    (not pass src for both old + new — that was the original
    RISK shape)."""
    src = (REPO_ROOT / "app/healing/monitors/tz_drift.py").read_text()
    fn_start = src.find("def _propose_consolidation_cr")
    fn_end = src.find("\ndef ", fn_start + 1)
    body = src[fn_start:fn_end]
    assert "_synthesize_zoneinfo_patch" in body
    assert "new_content=new_src" in body
    # The earlier empty-diff shape (`new_content=src, old_content=src`)
    # must not reappear.
    assert "new_content=src,  # operator decides" not in body
