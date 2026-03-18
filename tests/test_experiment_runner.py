"""Tests for app.experiment_runner — experiment sandbox."""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock config
from tests.test_metrics import _FakeSettings
import app.config as config_mod
config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


class TestMutationSpec:
    def test_create(self):
        from app.experiment_runner import MutationSpec
        m = MutationSpec(
            experiment_id="exp_001",
            hypothesis="Test hypothesis",
            change_type="skill",
            files={"skills/test.md": "# Test Skill\n\nThis is a test skill file with enough content."},
        )
        assert m.experiment_id == "exp_001"
        assert m.change_type == "skill"
        assert len(m.files) == 1


class TestGenerateExperimentId:
    def test_format(self):
        from app.experiment_runner import generate_experiment_id
        eid = generate_experiment_id("test hypothesis")
        assert eid.startswith("exp_")
        parts = eid.split("_")
        assert len(parts) == 3

    def test_different_hypotheses_different_ids(self):
        from app.experiment_runner import generate_experiment_id
        id1 = generate_experiment_id("hypothesis A")
        id2 = generate_experiment_id("hypothesis B")
        assert id1.split("_")[2] != id2.split("_")[2]


class TestExperimentRunner:
    def _make_runner(self, tmp_path, monkeypatch, scores):
        """Helper to create an ExperimentRunner with mocked methods pointing to tmp_path."""
        import app.experiment_runner as runner_mod
        import app.results_ledger as ledger_mod

        workspace = tmp_path / "workspace"
        workspace.mkdir(exist_ok=True)
        (workspace / "skills").mkdir(exist_ok=True)
        monkeypatch.setattr(runner_mod, "SKILLS_DIR", workspace / "skills")
        monkeypatch.setattr(ledger_mod, "LEDGER_PATH", tmp_path / "results.tsv")

        from app.experiment_runner import ExperimentRunner

        er = ExperimentRunner()
        er._backup_dir = tmp_path / ".backup"

        # Mock composite_score to return controlled values
        call_count = [0]
        def mock_composite():
            call_count[0] += 1
            idx = min(call_count[0] - 1, len(scores) - 1)
            return scores[idx]
        monkeypatch.setattr("app.experiment_runner.composite_score", mock_composite)

        # Patch file operations to use tmp workspace
        def patched_apply(self, mut):
            applied = []
            for rel_path, content in mut.files.items():
                full_path = workspace / rel_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                applied.append(rel_path)
            return applied

        def patched_backup(self, mut):
            backed_up = {}
            for rel_path in mut.files:
                full_path = workspace / rel_path
                backed_up[rel_path] = full_path.read_text() if full_path.exists() else None
            return backed_up

        def patched_restore(self, backed_up):
            for rel_path, content in backed_up.items():
                full_path = workspace / rel_path
                if content is None and full_path.exists():
                    full_path.unlink()
                elif content is not None:
                    full_path.write_text(content)

        def patched_validate(self, mut, applied):
            # Validate against the tmp workspace
            for rel_path in applied:
                full_path = workspace / rel_path
                if not full_path.exists():
                    return False, f"File not created: {rel_path}"
                if full_path.stat().st_size == 0:
                    return False, f"Empty file: {rel_path}"
                if rel_path.endswith(".md"):
                    content = full_path.read_text()
                    if len(content) < 50:
                        return False, f"Skill file too short: {rel_path}"
            return True, "ok"

        monkeypatch.setattr(ExperimentRunner, "_apply_mutation", patched_apply)
        monkeypatch.setattr(ExperimentRunner, "_backup_files", patched_backup)
        monkeypatch.setattr(ExperimentRunner, "_restore_backup", patched_restore)
        monkeypatch.setattr(ExperimentRunner, "_validate_mutation", patched_validate)

        return er, workspace

    def test_discard_on_regression(self, tmp_path, monkeypatch):
        """Test that a skill mutation is discarded when score drops."""
        from app.experiment_runner import MutationSpec

        er, workspace = self._make_runner(tmp_path, monkeypatch, scores=[0.5, 0.48])

        mutation = MutationSpec(
            experiment_id="exp_test",
            hypothesis="Test skill",
            change_type="skill",
            files={"skills/test_skill.md": "# Test\n\nThis is a test skill with enough content to pass validation checks."},
        )

        result = er.run_experiment(mutation)
        assert result.status == "discard"
        assert result.delta < 0
        # File should be reverted (deleted since it didn't exist before)
        assert not (workspace / "skills" / "test_skill.md").exists()

    def test_keep_on_improvement(self, tmp_path, monkeypatch):
        """Test that an improving mutation is kept."""
        from app.experiment_runner import MutationSpec

        er, workspace = self._make_runner(tmp_path, monkeypatch, scores=[0.5, 0.55])

        mutation = MutationSpec(
            experiment_id="exp_test2",
            hypothesis="Improving skill",
            change_type="skill",
            files={"skills/good_skill.md": "# Good Skill\n\nThis skill improves the system significantly with useful content."},
        )

        result = er.run_experiment(mutation)
        assert result.status == "keep"
        assert result.delta > 0
        assert (workspace / "skills" / "good_skill.md").exists()

    def test_keep_skill_on_neutral_change(self, tmp_path, monkeypatch):
        """Skills with neutral impact (within tolerance) should be kept."""
        from app.experiment_runner import MutationSpec

        er, workspace = self._make_runner(tmp_path, monkeypatch, scores=[0.5, 0.4995])

        mutation = MutationSpec(
            experiment_id="exp_neutral",
            hypothesis="Neutral skill",
            change_type="skill",
            files={"skills/neutral.md": "# Neutral Skill\n\nThis skill doesn't change the score but adds knowledge to the team."},
        )

        result = er.run_experiment(mutation)
        assert result.status == "keep"  # within -0.001 tolerance

    def test_crash_on_apply_failure(self, tmp_path, monkeypatch):
        """Test that apply failures result in crash status."""
        import app.experiment_runner as runner_mod
        import app.results_ledger as ledger_mod
        from app.experiment_runner import ExperimentRunner, MutationSpec

        monkeypatch.setattr(ledger_mod, "LEDGER_PATH", tmp_path / "results.tsv")
        monkeypatch.setattr("app.experiment_runner.composite_score", lambda: 0.5)

        er = ExperimentRunner()
        er._backup_dir = tmp_path / ".backup"

        def failing_apply(self, mut):
            raise RuntimeError("Disk full")

        def noop_backup(self, mut):
            return {}

        def noop_restore(self, backed_up):
            pass

        monkeypatch.setattr(ExperimentRunner, "_apply_mutation", failing_apply)
        monkeypatch.setattr(ExperimentRunner, "_backup_files", noop_backup)
        monkeypatch.setattr(ExperimentRunner, "_restore_backup", noop_restore)

        mutation = MutationSpec(
            experiment_id="exp_crash",
            hypothesis="Crashing mutation",
            change_type="skill",
            files={"skills/crash.md": "content"},
        )

        result = er.run_experiment(mutation)
        assert result.status == "crash"
        assert "Disk full" in result.detail

    def test_results_recorded_in_ledger(self, tmp_path, monkeypatch):
        """Test that experiments are recorded in the results ledger."""
        import app.results_ledger as ledger_mod
        from app.experiment_runner import MutationSpec

        er, workspace = self._make_runner(tmp_path, monkeypatch, scores=[0.5, 0.55])

        mutation = MutationSpec(
            experiment_id="exp_ledger",
            hypothesis="Ledger test",
            change_type="skill",
            files={"skills/ledger_test.md": "# Ledger Test\n\nThis tests that results are recorded in the ledger properly."},
        )

        er.run_experiment(mutation)

        results = ledger_mod.get_recent_results(10)
        assert len(results) == 1
        assert results[0]["experiment_id"] == "exp_ledger"
        assert results[0]["status"] == "keep"


class TestValidateResponse:
    def test_contains_match(self):
        from app.experiment_runner import validate_response
        assert validate_response("The capital of France is Paris", "contains:Paris")
        assert not validate_response("The capital is Berlin", "contains:Paris")

    def test_contains_case_insensitive(self):
        from app.experiment_runner import validate_response
        assert validate_response("paris is great", "contains:Paris")

    def test_min_length(self):
        from app.experiment_runner import validate_response
        assert validate_response("x" * 100, "min_length:100")
        assert not validate_response("short", "min_length:100")

    def test_empty_rule(self):
        from app.experiment_runner import validate_response
        assert validate_response("anything", "")

    def test_no_rule(self):
        from app.experiment_runner import validate_response
        assert validate_response("anything", None)


class TestLoadTestTasks:
    def test_load_existing(self, tmp_path, monkeypatch):
        import app.experiment_runner as runner_mod
        test_file = tmp_path / "test_tasks.json"
        tasks = [{"id": "t1", "prompt": "hello", "type": "factual"}]
        test_file.write_text(json.dumps(tasks))
        monkeypatch.setattr(runner_mod, "TEST_TASKS_PATH", test_file)

        loaded = runner_mod.load_test_tasks()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "t1"

    def test_load_missing(self, tmp_path, monkeypatch):
        import app.experiment_runner as runner_mod
        monkeypatch.setattr(runner_mod, "TEST_TASKS_PATH", tmp_path / "nonexistent.json")
        assert runner_mod.load_test_tasks() == []
