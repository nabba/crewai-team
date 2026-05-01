"""Tests for app.epistemic.autotune.

Covers:
* analyze_bias_library — severity-downgrade and retirement-candidate paths
* analyze_verifier_registry — retirement on zero matches
* run_full_analysis — end-to-end with persistence
* YAML patch generation + in-place severity edit
* PR plan generation (dry-run only)
* Persistence layer (persist + list + status updates)
* API endpoints (GET / POST accept/reject / POST run)
"""
from __future__ import annotations

import os
import sys
import textwrap
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import MagicMock, patch

# ── Stubs ────────────────────────────────────────────────────────────
_mock_psycopg2 = MagicMock()
_mock_psycopg2.InterfaceError = type("InterfaceError", (Exception,), {})
_mock_psycopg2.OperationalError = type("OperationalError", (Exception,), {})
sys.modules.setdefault("psycopg2", _mock_psycopg2)
sys.modules.setdefault("psycopg2.pool", MagicMock())

for _mod in ("crewai", "crewai.tools", "langchain_anthropic", "docker"):
    if _mod not in sys.modules:
        m = types.ModuleType(_mod)
        if _mod == "crewai.tools":
            m.tool = lambda name: (lambda fn: fn)
        sys.modules[_mod] = m


from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.epistemic.api import router  # noqa: E402
from app.epistemic.autotune import (  # noqa: E402
    DEFAULT_WINDOW_DAYS,
    FORCE_PROCEED_RATE_TOO_STRICT,
    MIN_FIRES_FOR_SEVERITY_PROPOSAL,
    PEER_REVIEW_ALLOW_RATE_TOO_AGGRESSIVE,
    RETIREMENT_FIRE_FLOOR,
    ProposalApplyError,
    ProposalKind,
    TuningProposal,
    _hash_proposal,
    _replace_yaml_severity,
    _new_proposal,
    analyze_bias_library,
    analyze_verifier_registry,
    apply_proposal_to_disk,
    open_pr_for_proposal,
    run_full_analysis,
)
from app.epistemic import span_writer  # noqa: E402


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _enabled():
    return patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"})


def _disabled():
    return patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""})


def _make_proposal(**overrides) -> TuningProposal:
    defaults = dict(
        target_kind="bias",
        target_id="inference_as_fact",
        kind=ProposalKind.SEVERITY_DOWNGRADE,
        rationale="test",
        metric_evidence={"fires": 50},
        yaml_patch="# placeholder",
        confidence=0.7,
    )
    defaults.update(overrides)
    return _new_proposal(**defaults)


# ============================================================================
# analyze_bias_library — severity downgrade
# ============================================================================

class TestAnalyzeBiasLibraryDowngrade(unittest.TestCase):

    def test_high_force_proceed_rate_proposes_downgrade(self):
        # 60 fires, 30 overrides, 18 force_proceed → fp_rate=0.60.
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "inference_as_fact": {
                           "severity": "high",
                           "fires": 60,
                           "overrides": 30,
                           "force_proceed": 18,
                           "use_revision": 8,
                           "abandon": 4,
                           "peer_reviews_triggered": 0,
                           "peer_reviews_allowed": 0,
                           "peer_reviews_vetoed": 0,
                           "incidents_as_root_cause": 5,
                       },
                   }):
            proposals = analyze_bias_library(window_days=7)

        self.assertEqual(len(proposals), 1)
        p = proposals[0]
        self.assertEqual(p.target_id, "inference_as_fact")
        self.assertEqual(p.kind, ProposalKind.SEVERITY_DOWNGRADE)
        self.assertGreater(p.confidence, 0.6)
        self.assertIn("force-proceed override rate", p.rationale)
        # YAML patch should reference both old and new severity.
        self.assertIn("severity: high", p.yaml_patch)
        self.assertIn("severity: medium", p.yaml_patch)

    def test_well_calibrated_bias_no_proposal(self):
        # 50 fires, 10 overrides, 1 force_proceed → fp_rate=0.10. Above
        # GOOD threshold but below TOO_STRICT.
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "inference_as_fact": {
                           "severity": "high",
                           "fires": 50,
                           "overrides": 10,
                           "force_proceed": 1,
                           "use_revision": 8,
                           "abandon": 1,
                           "peer_reviews_triggered": 0,
                           "peer_reviews_allowed": 0,
                           "peer_reviews_vetoed": 0,
                           "incidents_as_root_cause": 0,
                       },
                   }):
            proposals = analyze_bias_library(window_days=7)
        self.assertEqual(proposals, [])

    def test_below_volume_floor_no_severity_proposal(self):
        # 5 fires (below MIN_FIRES_FOR_SEVERITY_PROPOSAL=20). High
        # force_proceed rate, but volume is too small.
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "inference_as_fact": {
                           "severity": "high",
                           "fires": 5,
                           "overrides": 5,
                           "force_proceed": 5,
                           "use_revision": 0, "abandon": 0,
                           "peer_reviews_triggered": 0,
                           "peer_reviews_allowed": 0,
                           "peer_reviews_vetoed": 0,
                           "incidents_as_root_cause": 0,
                       },
                   }):
            proposals = analyze_bias_library(window_days=7)
        # Above retirement floor (3); below severity floor (20). No proposal.
        self.assertEqual(proposals, [])

    def test_low_severity_cannot_be_downgraded_further(self):
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "x": {
                           "severity": "low",
                           "fires": 50,
                           "overrides": 30,
                           "force_proceed": 20,
                           "use_revision": 5, "abandon": 5,
                           "peer_reviews_triggered": 0,
                           "peer_reviews_allowed": 0,
                           "peer_reviews_vetoed": 0,
                           "incidents_as_root_cause": 0,
                       },
                   }):
            proposals = analyze_bias_library(window_days=7)
        # No severity-downgrade proposal — already at LOW.
        self.assertEqual(
            [p for p in proposals if p.kind == ProposalKind.SEVERITY_DOWNGRADE],
            [],
        )


# ============================================================================
# analyze_bias_library — retirement
# ============================================================================

class TestAnalyzeBiasLibraryRetirement(unittest.TestCase):

    def test_zero_fires_proposes_retirement(self):
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "anomaly_dismissal": {
                           "severity": "high",
                           "fires": 0,
                           "overrides": 0,
                           "force_proceed": 0,
                           "use_revision": 0, "abandon": 0,
                           "peer_reviews_triggered": 0,
                           "peer_reviews_allowed": 0,
                           "peer_reviews_vetoed": 0,
                           "incidents_as_root_cause": 0,
                       },
                   }):
            proposals = analyze_bias_library(window_days=7)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].kind, ProposalKind.RETIREMENT_CANDIDATE)
        self.assertGreaterEqual(proposals[0].confidence, 0.6)
        self.assertIn("retiring", proposals[0].rationale.lower())

    def test_low_fires_above_zero_lower_confidence(self):
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "x": {
                           "severity": "medium",
                           "fires": 2,
                           "overrides": 0,
                           "force_proceed": 0,
                           "use_revision": 0, "abandon": 0,
                           "peer_reviews_triggered": 0,
                           "peer_reviews_allowed": 0,
                           "peer_reviews_vetoed": 0,
                           "incidents_as_root_cause": 0,
                       },
                   }):
            proposals = analyze_bias_library(window_days=7)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].kind, ProposalKind.RETIREMENT_CANDIDATE)
        # Low confidence (0.4) when fires>0 but ≤ floor.
        self.assertAlmostEqual(proposals[0].confidence, 0.4, places=2)

    def test_retirement_skips_severity_proposal(self):
        # If retirement fires, severity proposal should NOT also fire on
        # the same bias (mutually exclusive — too little data for both).
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "x": {
                           "severity": "high",
                           "fires": 1,
                           "overrides": 1,
                           "force_proceed": 1,
                           "use_revision": 0, "abandon": 0,
                           "peer_reviews_triggered": 0,
                           "peer_reviews_allowed": 0,
                           "peer_reviews_vetoed": 0,
                           "incidents_as_root_cause": 0,
                       },
                   }):
            proposals = analyze_bias_library(window_days=7)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].kind, ProposalKind.RETIREMENT_CANDIDATE)


# ============================================================================
# analyze_bias_library — peer-review aggressive
# ============================================================================

class TestAnalyzeBiasLibraryAggressive(unittest.TestCase):

    def test_high_peer_review_allow_rate_proposes_downgrade(self):
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "destructive_without_recheck": {
                           "severity": "critical",
                           "fires": 50,
                           "overrides": 5,
                           "force_proceed": 1,  # below TOO_STRICT
                           "use_revision": 2, "abandon": 2,
                           "peer_reviews_triggered": 10,
                           "peer_reviews_allowed": 7,    # 70% allow-rate
                           "peer_reviews_vetoed": 3,
                           "incidents_as_root_cause": 0,
                       },
                   }):
            proposals = analyze_bias_library(window_days=7)
        # Aggressive-bias proposal (peer-review path) fires.
        kinds = {p.kind for p in proposals}
        self.assertIn(ProposalKind.SEVERITY_DOWNGRADE, kinds)
        downgrade = next(
            p for p in proposals
            if p.kind == ProposalKind.SEVERITY_DOWNGRADE
        )
        self.assertIn("peer-review", downgrade.rationale.lower())


# ============================================================================
# analyze_verifier_registry
# ============================================================================

class TestAnalyzeVerifierRegistry(unittest.TestCase):

    def test_zero_matches_proposes_retirement(self):
        # Mock returns 0 for every shape.
        with patch("app.epistemic.span_writer.verifier_match_counts",
                   return_value={}):
            proposals = analyze_verifier_registry(window_days=7)
        self.assertGreater(len(proposals), 0)
        # Every proposal is verifier_retirement kind.
        for p in proposals:
            self.assertEqual(p.kind, ProposalKind.VERIFIER_RETIREMENT)
            self.assertEqual(p.target_kind, "verifier")

    def test_positive_matches_no_proposal(self):
        # Mock returns positive counts for every shape; nothing proposed.
        from app.epistemic.verification import VERIFIER_REGISTRY
        all_matches = {s.id: 5 for s in VERIFIER_REGISTRY()}
        # span_writer returns by tool_head, but the analyzer maps. To
        # bypass mapping, patch the analyzer's call into span_writer.
        with patch("app.epistemic.autotune.verifier_match_counts",
                   return_value=all_matches, create=True):
            # The above patch path doesn't exist; alternate approach:
            # mock the underlying span_writer function directly.
            pass

        with patch("app.epistemic.span_writer.verifier_match_counts",
                   return_value=all_matches):
            proposals = analyze_verifier_registry(window_days=7)
        self.assertEqual(proposals, [])


# ============================================================================
# run_full_analysis
# ============================================================================

class TestRunFullAnalysis(unittest.TestCase):

    def test_combines_bias_and_verifier_proposals(self):
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={
                       "x": {
                           "severity": "medium",
                           "fires": 0,
                           "overrides": 0, "force_proceed": 0,
                           "use_revision": 0, "abandon": 0,
                           "peer_reviews_triggered": 0,
                           "peer_reviews_allowed": 0,
                           "peer_reviews_vetoed": 0,
                           "incidents_as_root_cause": 0,
                       },
                   }), \
             patch("app.epistemic.span_writer.verifier_match_counts",
                   return_value={}), \
             patch("app.epistemic.span_writer.persist_tuning_proposal"):
            proposals = run_full_analysis(window_days=7, persist=True)
        # 1 bias retirement + N verifier retirements (one per shape).
        self.assertGreaterEqual(len(proposals), 2)
        kinds = {p.kind for p in proposals}
        self.assertIn(ProposalKind.RETIREMENT_CANDIDATE, kinds)
        self.assertIn(ProposalKind.VERIFIER_RETIREMENT, kinds)

    def test_persist_off_does_not_call_writer(self):
        with patch("app.epistemic.autotune._compute_bias_metrics",
                   return_value={}), \
             patch("app.epistemic.span_writer.verifier_match_counts",
                   return_value={}), \
             patch("app.epistemic.span_writer.persist_tuning_proposal") as mock_persist:
            run_full_analysis(window_days=7, persist=False)
        mock_persist.assert_not_called()


# ============================================================================
# Hashing + idempotence
# ============================================================================

class TestProposalHashing(unittest.TestCase):

    def test_same_inputs_same_hash(self):
        h1 = _hash_proposal(
            "bias", "x", ProposalKind.SEVERITY_DOWNGRADE, "patch text",
        )
        h2 = _hash_proposal(
            "bias", "x", ProposalKind.SEVERITY_DOWNGRADE, "patch text",
        )
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_different_inputs_different_hash(self):
        h1 = _hash_proposal(
            "bias", "x", ProposalKind.SEVERITY_DOWNGRADE, "a",
        )
        h2 = _hash_proposal(
            "bias", "x", ProposalKind.SEVERITY_DOWNGRADE, "b",
        )
        self.assertNotEqual(h1, h2)

    def test_proposal_carries_hash(self):
        p = _make_proposal()
        self.assertTrue(p.content_hash)
        self.assertEqual(len(p.content_hash), 16)


# ============================================================================
# YAML patch + apply
# ============================================================================

class TestYamlPatch(unittest.TestCase):

    def test_replace_severity_basic(self):
        yaml_text = textwrap.dedent("""
            biases:
              - id: inference_as_fact
                severity: high
                detector: realtime
              - id: tool_laziness
                severity: medium
                detector: posthoc
        """)
        new = _replace_yaml_severity(yaml_text, "inference_as_fact", "medium")
        self.assertIn("severity: medium", new.split("inference_as_fact")[1].split("- id")[0])
        # Other entry untouched.
        self.assertIn("severity: medium", new.split("tool_laziness")[1])

    def test_apply_severity_change_writes_file(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "app" / "epistemic" / "data"
            data_dir.mkdir(parents=True)
            yaml_text = textwrap.dedent("""\
                biases:
                  - id: inference_as_fact
                    severity: high
                    detector: realtime
            """)
            (data_dir / "biases.yaml").write_text(yaml_text)

            patch_text = (
                "# inference_as_fact — severity adjustment\n"
                "# - id: inference_as_fact\n"
                "# -   severity: high\n"
                "# +   severity: medium\n"
            )
            proposal = _new_proposal(
                target_kind="bias",
                target_id="inference_as_fact",
                kind=ProposalKind.SEVERITY_DOWNGRADE,
                rationale="test",
                metric_evidence={},
                yaml_patch=patch_text,
                confidence=0.8,
            )
            written = apply_proposal_to_disk(proposal, repo_root=tmp_path)
            self.assertEqual(written, data_dir / "biases.yaml")
            new_text = (data_dir / "biases.yaml").read_text()
            self.assertIn("severity: medium", new_text)
            self.assertNotIn("severity: high", new_text)

    def test_apply_retirement_is_manual(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "app" / "epistemic" / "data").mkdir(parents=True)
            (tmp_path / "app" / "epistemic" / "data" / "biases.yaml").write_text("biases: []\n")
            proposal = _make_proposal(kind=ProposalKind.RETIREMENT_CANDIDATE)
            with self.assertRaises(ProposalApplyError):
                apply_proposal_to_disk(proposal, repo_root=tmp_path)

    def test_apply_verifier_retirement_is_manual(self):
        proposal = _make_proposal(
            target_kind="verifier",
            target_id="filesystem.is_symlink",
            kind=ProposalKind.VERIFIER_RETIREMENT,
        )
        with self.assertRaises(ProposalApplyError):
            apply_proposal_to_disk(proposal, repo_root=Path("/tmp/x"))


# ============================================================================
# PR plan generation
# ============================================================================

class TestPrPlan(unittest.TestCase):

    def test_dry_run_returns_command_sequence(self):
        proposal = _make_proposal()
        plan = open_pr_for_proposal(proposal, dry_run=True)
        self.assertIn("branch", plan)
        self.assertEqual(plan["branch"], f"autotune/{proposal.content_hash}")
        self.assertIn("commands", plan)
        self.assertFalse(plan["executed"])
        # Body cites the proposal id and rationale.
        self.assertIn(proposal.proposal_id, plan["body"])
        self.assertIn("Manual review required", plan["body"])

    def test_non_dry_run_refuses_to_execute(self):
        proposal = _make_proposal()
        with self.assertRaises(ProposalApplyError):
            open_pr_for_proposal(proposal, dry_run=False)


# ============================================================================
# Persistence (span_writer extensions)
# ============================================================================

class TestProposalPersistence(unittest.TestCase):

    def test_persist_no_op_when_disabled(self):
        with _disabled(), patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_tuning_proposal(_make_proposal())
            mock_exec.assert_not_called()

    def test_persist_upserts_on_content_hash(self):
        with _enabled(), patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_tuning_proposal(_make_proposal())
            mock_exec.assert_called_once()
            sql = mock_exec.call_args[0][0]
            self.assertIn(
                "INSERT INTO control_plane.epistemic_tuning_proposals", sql,
            )
            self.assertIn("ON CONFLICT (content_hash) DO UPDATE", sql)

    def test_list_filters_by_status(self):
        rows = [{
            "proposal_id": "prop_aa", "content_hash": "h1",
            "target_kind": "bias", "target_id": "x",
            "kind": "severity_downgrade",
            "rationale": "r", "metric_evidence": {},
            "yaml_patch": "p", "confidence": 0.7,
            "status": "proposed", "operator_note": "",
            "created_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        }]
        with _enabled(), patch.object(span_writer, "execute", return_value=rows):
            out = span_writer.list_tuning_proposals(status="proposed")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["proposal_id"], "prop_aa")
        self.assertEqual(out[0]["status"], "proposed")

    def test_update_status_validates(self):
        with _enabled(), patch.object(span_writer, "execute"):
            with self.assertRaises(ValueError):
                span_writer.update_tuning_proposal_status(
                    proposal_id="x", status="bogus",
                )

    def test_update_status_returns_true_on_success(self):
        with _enabled(), patch.object(span_writer, "execute"):
            ok = span_writer.update_tuning_proposal_status(
                proposal_id="prop_aa", status="accepted",
                operator_note="merged",
            )
        self.assertTrue(ok)


# ============================================================================
# Aggregation helpers (the joins that drive the analyzer)
# ============================================================================

class TestAggregationHelpers(unittest.TestCase):

    def test_bias_match_counts(self):
        rows = [
            {"bias_id": "inference_as_fact", "n": 12},
            {"bias_id": "tool_laziness", "n": 4},
        ]
        with _enabled(), patch.object(span_writer, "execute", return_value=rows):
            out = span_writer.bias_match_counts(window_days=7)
        self.assertEqual(out, {"inference_as_fact": 12, "tool_laziness": 4})

    def test_override_counts_by_bias(self):
        rows = [
            {"bias_id": "x", "user_action": "force_proceed", "n": 3},
            {"bias_id": "x", "user_action": "abandon", "n": 1},
            {"bias_id": "y", "user_action": "force_proceed", "n": 2},
        ]
        with _enabled(), patch.object(span_writer, "execute", return_value=rows):
            out = span_writer.override_counts_by_bias(window_days=7)
        self.assertEqual(out["x"]["total"], 4)
        self.assertEqual(out["x"]["force_proceed"], 3)
        self.assertEqual(out["x"]["abandon"], 1)
        self.assertEqual(out["y"]["total"], 2)

    def test_peer_review_counts_by_bias(self):
        rows = [
            {"bias_id": "x", "decision": "veto", "n": 2},
            {"bias_id": "x", "decision": "allow", "n": 3},
        ]
        with _enabled(), patch.object(span_writer, "execute", return_value=rows):
            out = span_writer.peer_review_counts_by_bias(window_days=7)
        self.assertEqual(out["x"]["total"], 5)
        self.assertEqual(out["x"]["allow"], 3)
        self.assertEqual(out["x"]["veto"], 2)


# ============================================================================
# API endpoints
# ============================================================================

class TestTuningAPI(unittest.TestCase):

    def test_list_proposals_default_filter(self):
        client = _build_client()
        with patch(
            "app.epistemic.api.list_tuning_proposals",
            return_value=[{
                "proposal_id": "prop_aa", "content_hash": "h",
                "target_kind": "bias", "target_id": "x",
                "kind": "severity_downgrade",
                "rationale": "r", "metric_evidence": {},
                "yaml_patch": "p", "confidence": 0.7,
                "status": "proposed", "operator_note": "",
                "created_at": "2026-05-01T00:00:00+00:00",
                "updated_at": "2026-05-01T00:00:00+00:00",
            }],
        ):
            resp = client.get("/epistemic/tuning/proposals")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["status_filter"], "proposed")

    def test_proposal_detail_404(self):
        client = _build_client()
        with patch(
            "app.epistemic.api.lookup_tuning_proposal", return_value=None,
        ):
            resp = client.get("/epistemic/tuning/proposals/prop_missing")
        self.assertEqual(resp.status_code, 404)

    def test_accept_marks_status(self):
        client = _build_client()
        with patch(
            "app.epistemic.api.update_tuning_proposal_status",
            return_value=True,
        ) as mock_update:
            resp = client.post(
                "/epistemic/tuning/proposals/prop_aa/accept",
                json={"operator_note": "merged in PR #42"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "accepted")
        mock_update.assert_called_once()
        kwargs = mock_update.call_args.kwargs
        self.assertEqual(kwargs["status"], "accepted")
        self.assertEqual(kwargs["operator_note"], "merged in PR #42")

    def test_reject_marks_status(self):
        client = _build_client()
        with patch(
            "app.epistemic.api.update_tuning_proposal_status",
            return_value=True,
        ):
            resp = client.post(
                "/epistemic/tuning/proposals/prop_aa/reject",
                json={"operator_note": "false positive"},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "rejected")

    def test_run_endpoint_triggers_analysis(self):
        client = _build_client()
        # Patch the analyzer to return a single proposal so we don't hit DB.
        with patch(
            "app.epistemic.autotune.run_full_analysis",
            return_value=[_make_proposal()],
        ):
            resp = client.post(
                "/epistemic/tuning/run", json={"window_days": 14},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["window_days"], 14)
        self.assertEqual(body["proposal_count"], 1)


if __name__ == "__main__":
    unittest.main()
