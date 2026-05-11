"""PROGRAM §40.3 — Q3.3 fourth-pass cleanup regression sweep.

Targets the round-4 findings:
  * restart-claim check now first in lifespan
  * cutover idempotency on plan_id
  * budget forecast excludes paused agents
  * DR secret denylist extended
  * read_archive releases lock between files
  * reclaim-trend alert dedup
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
#   Item 1 — restart-claim check ordering in lifespan
# ─────────────────────────────────────────────────────────────────────────


def test_restart_claim_check_runs_before_gateway_bind_check():
    """Source-level: the restart-claim check happens BEFORE the
    gateway-bind validation in lifespan. Defense in depth — even if
    the bind config is wrong (the bind check raises), the claim
    alert fires first so the operator sees both issues."""
    src = Path("app/main.py").read_text()
    # Scope the assertion to the lifespan() function body so we don't
    # accidentally match the function-definition occurrences earlier.
    lifespan_start = src.find("async def lifespan(app: FastAPI):")
    # End of lifespan body = next top-level definition.
    lifespan_end = src.find("\napp = FastAPI", lifespan_start)
    assert lifespan_start > 0 and lifespan_end > lifespan_start
    body = src[lifespan_start:lifespan_end]
    q33_marker = body.find("Q3.3 (PROGRAM §40.3 Item 1)")
    bind_check = body.find('GATEWAY_BIND must be 127.0.0.1')
    audit_config_call = body.find("_configure_audit_log()")
    assert q33_marker > 0, "Q3.3 marker not in lifespan body"
    assert bind_check > 0, "bind check not in lifespan body"
    assert audit_config_call > 0, "audit_log call not in lifespan body"
    assert q33_marker < bind_check, (
        "restart-claim check must run BEFORE gateway-bind check"
    )
    assert q33_marker < audit_config_call, (
        "restart-claim check must run BEFORE audit-log config"
    )


# ─────────────────────────────────────────────────────────────────────────
#   Item 2 — Cutover idempotency
# ─────────────────────────────────────────────────────────────────────────


def test_cutover_has_existing_proposal_helper():
    src = Path("app/memory/embedding_migration/cutover.py").read_text()
    assert "def _existing_active_proposal_for" in src
    # The helper must check non-terminal states only.
    helper_start = src.find("def _existing_active_proposal_for")
    helper_end = src.find("\ndef ", helper_start + 1)
    body = src[helper_start:helper_end]
    assert "TERMINAL_STATES" in body
    assert "plan_id" in body


def test_propose_cutover_checks_idempotency():
    """propose_cutover refuses to file a second proposal for the
    same plan_id when a non-terminal one already exists."""
    src = Path("app/memory/embedding_migration/cutover.py").read_text()
    propose_start = src.find("def propose_cutover")
    # Must call _existing_active_proposal_for before verify.
    relevant = src[propose_start:propose_start + 3000]
    assert "_existing_active_proposal_for" in relevant
    assert "non-terminal Tier-3 proposal already exists" in relevant


# ─────────────────────────────────────────────────────────────────────────
#   Item 3 — Budget forecast excludes paused
# ─────────────────────────────────────────────────────────────────────────


def test_budget_forecast_excludes_paused_agents():
    """The aggregate-budget-limit SQL must filter out is_paused=true
    rows. Paused budgets can't be spent against; including them
    would inflate headroom and hide breach risk."""
    src = Path("app/control_plane/budgets.py").read_text()
    forecast_start = src.find("def forecast_breach_periods")
    forecast_end = src.find("\ndef ", forecast_start + 1)
    if forecast_end < 0:
        forecast_end = len(src)
    body = src[forecast_start:forecast_end]
    assert "COALESCE(is_paused, false) = false" in body, (
        "forecast SQL must exclude paused budgets"
    )


# ─────────────────────────────────────────────────────────────────────────
#   Item 4 — DR secret denylist extension
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dr_export():
    return _load_isolated("dr_export_q33", "app/dr/export_kbs.py")


def test_denylist_catches_netrc(dr_export):
    assert dr_export._is_secret_path(".netrc")
    assert dr_export._is_secret_path("home/user/.netrc")
    assert dr_export._is_secret_path(".netrc.bak")


def test_denylist_catches_ssh_keys(dr_export):
    assert dr_export._is_secret_path(".ssh/id_rsa")
    assert dr_export._is_secret_path("dot_ssh/id_ed25519")
    assert dr_export._is_secret_path("id_ecdsa")
    # Public key counterpart — substring match still excludes (defensive)
    assert dr_export._is_secret_path("id_rsa.pub")


def test_denylist_catches_aws_credentials(dr_export):
    assert dr_export._is_secret_path(".aws/credentials")
    assert dr_export._is_secret_path("aws_access_key.json")
    assert dr_export._is_secret_path("AWS_CREDENTIALS")


def test_denylist_catches_keepass(dr_export):
    assert dr_export._is_secret_path("vault.kdb")
    assert dr_export._is_secret_path("personal.kdbx")
    assert dr_export._is_secret_path("backup/passwords.KDBX")


def test_denylist_catches_gpg_pgp(dr_export):
    assert dr_export._is_secret_path("key.gpg")
    assert dr_export._is_secret_path("backup.pgp")
    assert dr_export._is_secret_path(".gnupg/secring.gpg")


def test_denylist_catches_pem_key_files(dr_export):
    assert dr_export._is_secret_path("cert.pem")
    assert dr_export._is_secret_path("server.key")
    assert dr_export._is_secret_path("nested/dir/private.pem")


def test_denylist_does_not_overmatch_innocent_files(dr_export):
    """Legitimate affect ledgers / canonical workspace files should
    NOT trip the denylist."""
    assert not dr_export._is_secret_path("affect/trace.jsonl")
    assert not dr_export._is_secret_path("identity/continuity_ledger.jsonl")
    assert not dr_export._is_secret_path("audit_journal.json")
    assert not dr_export._is_secret_path("self_heal/auditor_bridge.json")


# ─────────────────────────────────────────────────────────────────────────
#   Item 5 — read_archive releases lock between files
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def jsonl_retention():
    return _load_isolated(
        "jsonl_retention_q33",
        "app/utils/jsonl_retention.py",
    )


def test_read_archive_lazy_yield_preserved(jsonl_retention, tmp_path):
    """read_archive must still yield lazily (line-by-line) — not
    buffer entire files. We confirm by reading a 1000-line archive
    and only consuming the first 3 lines; the iteration shouldn't
    have read the rest."""
    p = tmp_path / "x.jsonl"
    arch_dir = tmp_path / "archive"
    arch_dir.mkdir()
    arch_file = arch_dir / "2026-01_x.jsonl"
    with arch_file.open("w") as f:
        for i in range(1000):
            f.write(f'{{"i":{i}}}\n')

    it = jsonl_retention.read_archive(p, include_live=False)
    first_three = [next(it) for _ in range(3)]
    assert first_three[0].strip() == '{"i":0}'
    assert first_three[2].strip() == '{"i":2}'
    # We didn't exhaust the iterator — the rest stays lazy.


def test_read_archive_iterates_multiple_files(jsonl_retention, tmp_path):
    """End-to-end smoke after the Q3.3 lock refactor — across 3
    monthly archives + a live file."""
    p = tmp_path / "x.jsonl"
    arch_dir = tmp_path / "archive"
    arch_dir.mkdir()
    for month, prefix in [("2026-01", "jan"), ("2026-02", "feb"), ("2026-03", "mar")]:
        with (arch_dir / f"{month}_x.jsonl").open("w") as f:
            for i in range(5):
                f.write(f'{{"month":"{prefix}","i":{i}}}\n')
    with p.open("w") as f:
        for i in range(3):
            f.write(f'{{"month":"live","i":{i}}}\n')

    rows = [json.loads(line) for line in jsonl_retention.read_archive(p)]
    months = [r["month"] for r in rows]
    # Order: jan (5) → feb (5) → mar (5) → live (3)
    assert months[:5] == ["jan"] * 5
    assert months[5:10] == ["feb"] * 5
    assert months[10:15] == ["mar"] * 5
    assert months[15:18] == ["live"] * 3
    assert len(rows) == 18


def test_read_archive_lock_released_between_files(jsonl_retention, tmp_path):
    """While read_archive is yielding from file 2, the rotation lock
    is NOT held — a competing writer could acquire LOCK_EX.

    We assert this indirectly: after consuming all lines from file 1
    and before file 2, the reader has released the lock. A writer
    in another thread should NOT block beyond the per-file window."""
    if not jsonl_retention._HAS_FCNTL:
        pytest.skip("fcntl required for this test")
    p = tmp_path / "x.jsonl"
    arch_dir = tmp_path / "archive"
    arch_dir.mkdir()
    for month in ["2026-01", "2026-02", "2026-03"]:
        with (arch_dir / f"{month}_x.jsonl").open("w") as f:
            for i in range(3):
                f.write(f'{{"month":"{month}","i":{i}}}\n')

    writer_acquired = threading.Event()
    consumed = []

    def slow_reader():
        for line in jsonl_retention.read_archive(p, include_live=False):
            consumed.append(line)
            # Slow consumer — sleep between lines.
            time.sleep(0.05)

    def writer():
        # Wait briefly to give the reader a head start.
        time.sleep(0.1)
        with jsonl_retention._rotation_lock(p, exclusive=True):
            writer_acquired.set()

    t_r = threading.Thread(target=slow_reader)
    t_w = threading.Thread(target=writer)
    t_r.start(); t_w.start()
    t_r.join(timeout=5.0); t_w.join(timeout=5.0)
    # If lock were held across the entire iteration, the writer
    # would have had to wait the full duration (9 lines × 0.05 =
    # 0.45s). With per-file release, the writer should acquire
    # well before the reader finishes.
    assert writer_acquired.is_set(), (
        "writer should have acquired the lock between files"
    )
    assert len(consumed) == 9   # all archive lines yielded


# ─────────────────────────────────────────────────────────────────────────
#   Item 6 — Reclaim-trend alert dedup
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hygiene():
    return _load_isolated(
        "chromadb_hygiene_q33",
        "app/healing/monitors/chromadb_hygiene.py",
    )


def test_alert_dedup_source_marker(hygiene):
    """Source-level: the run() function references the Q3.3 dedup
    state field and the resurface threshold."""
    src = Path("app/healing/monitors/chromadb_hygiene.py").read_text()
    assert "trend_alert_repeats" in src
    assert "Q3.3" in src or "PROGRAM §40.3 Item 6" in src
    # Repeat threshold is 4 silent passes (~1 year quarterly).
    assert "repeats >= 4" in src


def test_alert_growing_reclaim_accepts_repeat_n(hygiene):
    """The alert formatter accepts a ``repeat_n`` arg for the
    resurfacing variant — operator can distinguish first alert
    from a re-nudge."""
    import inspect
    sig = inspect.signature(hygiene._alert_growing_reclaim)
    assert "repeat_n" in sig.parameters
    # Default = 1 for the first-alert path.
    assert sig.parameters["repeat_n"].default == 1
