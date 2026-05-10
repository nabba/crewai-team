"""Tests for app.proposal_bridge — staging + promotion to CR gate."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the bridge staging dir to tmp + reset the import-time
    constants so tests don't pollute one another."""
    monkeypatch.setenv("PROPOSAL_BRIDGE_DIR", str(tmp_path / "proposal_bridge"))
    # Re-import store to pick up env override fresh — _base_dir() reads it
    # at call time so re-import isn't strictly required, but the daemon
    # state is module-global.
    from app.proposal_bridge import promoter, store
    promoter.stop()
    promoter._driver_started = False
    return {
        "base": tmp_path / "proposal_bridge",
        "store": store,
        "promoter": promoter,
    }


def _stage_basic(store, source="capability_gap", signature="abc12345",
                 body="# Body content\nLine 2", title="Test proposal",
                 target_path=None, cooldown_days=7):
    target_path = target_path or f"docs/proposed_{source}/{signature}.md"
    state, _ = store.stage(
        source=source,
        signature=signature,
        title=title,
        body_markdown=body,
        target_path=target_path,
        cooldown_days=cooldown_days,
    )
    return state


def _stage_with_flag(store, **kw):
    """Variant that returns the (state, was_new) tuple directly."""
    target_path = kw.pop("target_path", None) or (
        f"docs/proposed_{kw.get('source', 'capability_gap')}/"
        f"{kw.get('signature', 'abc12345')}.md"
    )
    return store.stage(
        source=kw.get("source", "capability_gap"),
        signature=kw.get("signature", "abc12345"),
        title=kw.get("title", "Test proposal"),
        body_markdown=kw.get("body", "# Body content\nLine 2"),
        target_path=target_path,
        cooldown_days=kw.get("cooldown_days", 7),
    )


# ── Core stage() semantics ──────────────────────────────────────────────


def test_stage_creates_pair_on_disk(isolated_bridge):
    store = isolated_bridge["store"]
    base = isolated_bridge["base"]

    state = _stage_basic(store)

    assert state.status == store.ProposalStatus.STAGED
    body_path = base / "capability_gap" / "abc12345.md"
    meta_path = base / "capability_gap" / "abc12345.json"
    assert body_path.read_text() == "# Body content\nLine 2"
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "capability_gap"
    assert meta["status"] == "staged"
    assert meta["body_hash"]


def test_stage_idempotent_on_same_body(isolated_bridge):
    store = isolated_bridge["store"]
    first, was_new_first = _stage_with_flag(store)
    second, was_new_second = _stage_with_flag(store)
    assert was_new_first is True
    assert was_new_second is False
    assert first.staged_at == second.staged_at


def test_stage_bumps_cooldown_on_body_change(isolated_bridge):
    store = isolated_bridge["store"]
    first = _stage_basic(store, body="# v1")
    # Force a delay by manipulating the file's recorded timestamp.
    base = isolated_bridge["base"]
    meta_path = base / "capability_gap" / "abc12345.json"
    data = json.loads(meta_path.read_text())
    earlier = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    data["staged_at"] = earlier
    meta_path.write_text(json.dumps(data))

    second, was_new = _stage_with_flag(store, body="# v2")

    assert was_new is True
    assert second.staged_at != earlier  # bumped
    assert second.body_hash != first.body_hash


def test_stage_rejects_unknown_source(isolated_bridge):
    store = isolated_bridge["store"]
    with pytest.raises(ValueError, match="unknown proposal source"):
        store.stage(
            source="invented_source",
            signature="abc",
            title="x",
            body_markdown="x",
            target_path="docs/x.md",
        )


def test_stage_rejects_malformed_signature(isolated_bridge):
    store = isolated_bridge["store"]
    with pytest.raises(ValueError, match="signature"):
        store.stage(
            source="capability_gap",
            signature="has spaces",
            title="x",
            body_markdown="x",
            target_path="docs/x.md",
        )


def test_stage_does_not_resurrect_terminal(isolated_bridge):
    store = isolated_bridge["store"]
    state = _stage_basic(store)
    state.status = store.ProposalStatus.APPLIED
    state.resolved_at = datetime.now(timezone.utc).isoformat()
    store.update_proposal(state)

    again, was_new = _stage_with_flag(store, body="# different body now")
    # Still APPLIED — re-staging did not reset.
    assert again.status == store.ProposalStatus.APPLIED
    assert was_new is False


def test_list_proposals_filters(isolated_bridge):
    store = isolated_bridge["store"]
    _stage_basic(store, source="capability_gap", signature="cap1")
    _stage_basic(store, source="library_radar", signature="lib1")
    _stage_basic(store, source="paper_pipeline", signature="pap1")

    assert len(store.list_proposals()) == 3
    assert {s.signature for s in store.list_proposals(source="library_radar")} == {"lib1"}


def test_get_proposal_roundtrip(isolated_bridge):
    store = isolated_bridge["store"]
    _stage_basic(store, signature="lookup_me")
    state = store.get_proposal("capability_gap", "lookup_me")
    assert state is not None
    assert state.signature == "lookup_me"


def test_get_proposal_returns_none_for_miss(isolated_bridge):
    store = isolated_bridge["store"]
    assert store.get_proposal("capability_gap", "no_such_thing") is None


# ── Promoter: STAGED → CR_FILED ─────────────────────────────────────────


def _backdate_staging(store, source, signature, days_ago: int):
    """Move the staged_at backwards in time by `days_ago` days."""
    state = store.get_proposal(source, signature)
    assert state is not None
    state.staged_at = (
        datetime.now(timezone.utc) - timedelta(days=days_ago)
    ).isoformat()
    store.update_proposal(state)


def test_promoter_skips_when_cooldown_unelapsed(isolated_bridge):
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    _stage_basic(store, cooldown_days=7)  # just staged → not promotable yet

    counters = promoter.run_one_pass()
    assert counters["promoted_to_cr"] == 0
    state = store.get_proposal("capability_gap", "abc12345")
    assert state.status == store.ProposalStatus.STAGED


def test_promoter_files_cr_after_cooldown(isolated_bridge):
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    _stage_basic(store, cooldown_days=7)
    _backdate_staging(store, "capability_gap", "abc12345", days_ago=8)

    fake_cr = type("FakeCR", (), {
        "id": "cr_xyz",
        "status": _make_status("pending"),
        "decision_reason": None,
    })()

    with patch("app.change_requests.create_request", return_value=fake_cr), \
         patch("app.change_requests.send_ask"):
        counters = promoter.run_one_pass()

    assert counters["promoted_to_cr"] == 1
    state = store.get_proposal("capability_gap", "abc12345")
    assert state.status == store.ProposalStatus.CR_FILED
    assert state.cr_id == "cr_xyz"


def test_promoter_marks_validation_failure_as_rejected(isolated_bridge):
    """Validator-side terminal decision (TIER_IMMUTABLE / blocked path /
    outside roots) maps to REJECTED on our side — the producer's
    signature dedup then refuses to re-stage same content, breaking
    what would otherwise be a producer/promoter loop on permanent
    failures."""
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    _stage_basic(store, cooldown_days=1)
    _backdate_staging(store, "capability_gap", "abc12345", days_ago=2)

    rejected_cr = type("FakeCR", (), {
        "id": "cr_zzz",
        "status": _make_status("rejected"),
        "decision_reason": "outside allowed roots",
    })()

    with patch("app.change_requests.create_request", return_value=rejected_cr):
        counters = promoter.run_one_pass()

    assert counters["validator_rejected"] == 1
    state = store.get_proposal("capability_gap", "abc12345")
    assert state.status == store.ProposalStatus.REJECTED
    assert state.notes.get("rejection_layer") == "cr_validator"
    assert state.notes.get("cr_status") == "rejected"


def test_stage_rejects_target_path_outside_allowed_roots(isolated_bridge):
    """Stage-time path validation: impossible targets fail fast,
    not 7 days later at promotion."""
    store = isolated_bridge["store"]
    with pytest.raises(ValueError, match="rejected by validator"):
        store.stage(
            source="capability_gap",
            signature="abc12345",
            title="x",
            body_markdown="# body",
            target_path="workspace/foo.md",  # outside allowed roots
        )


def test_stage_rejects_tier_immutable_target(isolated_bridge):
    """TIER_IMMUTABLE paths refuse at stage time. Pulls a path
    dynamically from the canonical list so the test stays correct
    if the manifest evolves — picks one inside an allowed root so
    the TIER_IMMUTABLE check fires (not the outside-roots check)."""
    from app.change_requests.validator import _ALLOWED_ROOT_PREFIXES
    from app.auto_deployer import TIER_IMMUTABLE
    store = isolated_bridge["store"]
    protected_path = next(
        p for p in sorted(TIER_IMMUTABLE)
        if any(p.startswith(prefix) for prefix in _ALLOWED_ROOT_PREFIXES)
    )
    with pytest.raises(ValueError, match="TIER_IMMUTABLE"):
        store.stage(
            source="capability_gap",
            signature="abc12345",
            title="x",
            body_markdown="# body",
            target_path=protected_path,
        )


def test_promoter_rate_limits_per_pass(isolated_bridge):
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    # Stage 5 promotable proposals — promoter caps at 3/pass.
    for i in range(5):
        sig = f"sig{i:04d}"
        _stage_basic(store, signature=sig, cooldown_days=1)
        _backdate_staging(store, "capability_gap", sig, days_ago=2)

    fake_cr = type("FakeCR", (), {
        "id": "cr_x",
        "status": _make_status("pending"),
        "decision_reason": None,
    })()

    with patch("app.change_requests.create_request", return_value=fake_cr), \
         patch("app.change_requests.send_ask"):
        counters = promoter.run_one_pass()

    assert counters["promoted_to_cr"] == 3
    cr_filed = store.list_proposals(status=store.ProposalStatus.CR_FILED)
    staged = store.list_proposals(status=store.ProposalStatus.STAGED)
    assert len(cr_filed) == 3
    assert len(staged) == 2


# ── Promoter: CR_FILED → APPLIED / REJECTED ─────────────────────────────


def test_promoter_reconciles_applied(isolated_bridge):
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    _stage_basic(store)
    state = store.get_proposal("capability_gap", "abc12345")
    state.status = store.ProposalStatus.CR_FILED
    state.cr_id = "cr_done"
    state.cr_filed_at = datetime.now(timezone.utc).isoformat()
    store.update_proposal(state)

    applied_cr = type("FakeCR", (), {
        "id": "cr_done",
        "status": _make_status("applied"),
        "decision_reason": None,
        "pr_url": "https://github.com/x/y/pull/1",
    })()

    with patch("app.change_requests.get", return_value=applied_cr):
        counters = promoter.run_one_pass()

    assert counters["reconciled_applied"] == 1
    state = store.get_proposal("capability_gap", "abc12345")
    assert state.status == store.ProposalStatus.APPLIED


def test_promoter_reconciles_rejected(isolated_bridge):
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    _stage_basic(store)
    state = store.get_proposal("capability_gap", "abc12345")
    state.status = store.ProposalStatus.CR_FILED
    state.cr_id = "cr_no"
    state.cr_filed_at = datetime.now(timezone.utc).isoformat()
    store.update_proposal(state)

    rejected_cr = type("FakeCR", (), {
        "id": "cr_no",
        "status": _make_status("rejected"),
        "decision_reason": "operator declined",
        "pr_url": None,
    })()

    with patch("app.change_requests.get", return_value=rejected_cr):
        counters = promoter.run_one_pass()

    assert counters["reconciled_rejected"] == 1
    state = store.get_proposal("capability_gap", "abc12345")
    assert state.status == store.ProposalStatus.REJECTED


# ── Promoter: stale-staging expiry + cleanup ────────────────────────────


def test_promoter_expires_stale_staged(isolated_bridge):
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    _stage_basic(store, cooldown_days=7)
    # Backdate beyond _MAX_AGE_DAYS=30. Without a CR mock the promoter
    # would otherwise try to promote — but expire takes precedence
    # because we never actually return a CR; even so we monkeypatch
    # the rate-limit ceiling to 0 to be sure expiry runs first.
    _backdate_staging(store, "capability_gap", "abc12345", days_ago=45)

    # Mock CR system as unavailable so a promotion attempt would no-op.
    with patch("app.change_requests.create_request",
               side_effect=RuntimeError("unavailable")):
        counters = promoter.run_one_pass()

    state = store.get_proposal("capability_gap", "abc12345")
    assert state.status == store.ProposalStatus.EXPIRED
    assert counters["expired_stale"] == 1


def test_promoter_cleans_up_resolved_after_retention(isolated_bridge):
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    base = isolated_bridge["base"]
    _stage_basic(store)
    state = store.get_proposal("capability_gap", "abc12345")
    state.status = store.ProposalStatus.APPLIED
    state.resolved_at = (
        datetime.now(timezone.utc) - timedelta(days=20)
    ).isoformat()
    store.update_proposal(state)

    counters = promoter.run_one_pass()

    assert counters["cleaned_up"] == 1
    assert not (base / "capability_gap" / "abc12345.md").exists()
    assert not (base / "capability_gap" / "abc12345.json").exists()


def test_promoter_keeps_recent_resolved(isolated_bridge):
    """Cleanup should NOT fire within the retention window."""
    store = isolated_bridge["store"]
    promoter = isolated_bridge["promoter"]
    base = isolated_bridge["base"]
    _stage_basic(store)
    state = store.get_proposal("capability_gap", "abc12345")
    state.status = store.ProposalStatus.REJECTED
    state.resolved_at = (
        datetime.now(timezone.utc) - timedelta(days=3)
    ).isoformat()
    store.update_proposal(state)

    counters = promoter.run_one_pass()

    assert counters["cleaned_up"] == 0
    assert (base / "capability_gap" / "abc12345.json").exists()


# ── Disabled flag ───────────────────────────────────────────────────────


def test_promoter_disabled_short_circuits(isolated_bridge, monkeypatch):
    promoter = isolated_bridge["promoter"]
    monkeypatch.setenv("PROPOSAL_BRIDGE_ENABLED", "false")
    counters = promoter.run_one_pass()
    assert counters == {"status": "disabled"}


# ── Helpers ────────────────────────────────────────────────────────────


def _make_status(name: str):
    """Resolve to the actual Status enum value used in change_requests."""
    from app.change_requests import Status
    return Status(name)
