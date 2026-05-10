"""Regression: artifact-producing tasks must deliver an actual file
(Cure B, 2026-05-10).

Pre-fix shape: the orchestrator's success contract was "the LLM
returned non-empty text". User asked for a graphic; coder returned
Python source code that *would*, if run, produce one; orchestrator
declared success-shaped; vetting (correctly) rejected with "this
doesn't deliver the requested graphic — only partial code that
cannot generate the image"; vetting can't *make* a graphic so it
just rejected; eventually the watchdog fired with "narrow your
question" 30 minutes later.

Post-fix:
  • ``classify_task`` detects artifact intent at task entry.
  • ``build_artifact_directive`` augments the crew task with
    "use coding_session_*, execute the code, emit ARTIFACT: <path>".
  • ``verify_artifacts`` post-run requires an existing non-empty
    file at the referenced path; raises ``ArtifactNotProduced``
    if not, which the orchestrator surfaces to the retry path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.commander.artifact_intent import (
    ArtifactNotProduced,
    TaskShape,
    build_artifact_directive,
    classify_task,
    extract_artifact_paths,
    get_active_task_shape,
    set_active_task_shape,
    verify_artifacts,
)


# ── Classifier ──────────────────────────────────────────────────────


class TestClassifyTask:

    def test_text_request_stays_text(self) -> None:
        shape = classify_task("What is the capital of France?")
        assert shape.kind == "text"
        assert shape.expected_extensions == ()
        assert not shape.is_artifact

    def test_make_a_graphic_is_artifact(self) -> None:
        shape = classify_task(
            "Please make a graphic about forest age distribution in Estonia."
        )
        assert shape.is_artifact
        assert ".png" in shape.expected_extensions
        assert "graphic" in shape.trigger

    def test_generate_a_chart_is_artifact(self) -> None:
        shape = classify_task("Generate a chart of monthly revenue.")
        assert shape.is_artifact
        assert ".png" in shape.expected_extensions

    def test_create_pdf_report_is_artifact(self) -> None:
        shape = classify_task("Create a PDF report on Q3 results.")
        assert shape.is_artifact
        assert ".pdf" in shape.expected_extensions

    def test_explicit_extension_wins(self) -> None:
        """Explicit '.png' / 'as PDF' is the highest-confidence signal."""
        shape = classify_task("Render this analysis as a .pdf")
        assert shape.is_artifact
        assert shape.expected_extensions == (".pdf",)
        assert "explicit extension" in shape.trigger

    def test_save_as_csv_is_artifact(self) -> None:
        shape = classify_task("Save the results as CSV.")
        assert shape.is_artifact
        assert ".csv" in shape.expected_extensions

    def test_make_slides_is_artifact(self) -> None:
        shape = classify_task("Make a slides deck about our 2025 launch.")
        assert shape.is_artifact
        assert ".pptx" in shape.expected_extensions

    def test_produce_image_is_artifact(self) -> None:
        shape = classify_task("Produce an image illustrating photosynthesis.")
        assert shape.is_artifact

    def test_no_verb_means_text(self) -> None:
        """'The chart is interesting' has the noun but no verb — not a request."""
        shape = classify_task("The chart in your last response is interesting.")
        assert shape.kind == "text", (
            f"missing verb should default to text; got {shape}"
        )

    def test_empty_input_is_text(self) -> None:
        assert classify_task("").kind == "text"
        assert classify_task(None).kind == "text"  # type: ignore[arg-type]


# ── Directive ───────────────────────────────────────────────────────


class TestBuildArtifactDirective:

    def test_text_shape_emits_empty_directive(self) -> None:
        shape = TaskShape(kind="text")
        assert build_artifact_directive(shape) == ""

    def test_artifact_shape_includes_coding_session_instruction(self) -> None:
        shape = TaskShape(
            kind="artifact",
            expected_extensions=(".png",),
            trigger="noun match: graphic",
        )
        directive = build_artifact_directive(shape)
        # Critical pieces operators expect:
        assert "coding_session_" in directive, (
            "directive must point at coding_session_* tools"
        )
        assert "ARTIFACT:" in directive, (
            "directive must specify the response format the verifier expects"
        )
        assert ".png" in directive, "directive must echo expected extensions"


# ── Path extractor ──────────────────────────────────────────────────


class TestExtractArtifactPaths:

    def test_artifact_marker(self) -> None:
        text = "Done.\nARTIFACT: workspace/output/forest.png\nRegards."
        paths = extract_artifact_paths(text, allowed_extensions=(".png",))
        assert paths == ["workspace/output/forest.png"]

    def test_backtick_path(self) -> None:
        text = "I saved it to `workspace/output/chart.png`."
        paths = extract_artifact_paths(text, allowed_extensions=(".png",))
        assert paths == ["workspace/output/chart.png"]

    def test_bare_path(self) -> None:
        text = "Result file is at workspace/output/chart.pdf and looks good."
        paths = extract_artifact_paths(text, allowed_extensions=(".pdf",))
        assert "workspace/output/chart.pdf" in paths

    def test_filters_outside_allowed_roots(self) -> None:
        """A path under /etc/ must not slip through, even if mentioned."""
        text = "ARTIFACT: /etc/passwd"
        paths = extract_artifact_paths(text)
        assert paths == [], f"path-traversal-style mention must be filtered; got {paths}"

    def test_filters_wrong_extension(self) -> None:
        text = "ARTIFACT: workspace/output/forest.exe"
        paths = extract_artifact_paths(text, allowed_extensions=(".png",))
        assert paths == []

    def test_dedups(self) -> None:
        text = (
            "ARTIFACT: workspace/output/forest.png\n"
            "Saved to `workspace/output/forest.png`."
        )
        paths = extract_artifact_paths(text, allowed_extensions=(".png",))
        assert paths == ["workspace/output/forest.png"]

    def test_jpeg_normalized_to_jpg(self) -> None:
        """Allowed=jpg should match .jpeg too."""
        text = "ARTIFACT: workspace/output/photo.jpeg"
        paths = extract_artifact_paths(text, allowed_extensions=(".jpg",))
        assert paths == ["workspace/output/photo.jpeg"]


# ── Verifier ────────────────────────────────────────────────────────


class TestVerifyArtifacts:

    def test_text_shape_is_no_op(self, tmp_path: Path) -> None:
        shape = TaskShape(kind="text")
        # Should not raise even with no artifact and no response.
        result = verify_artifacts(shape, "no artifact here", workspace_root=tmp_path)
        assert result == ""

    def test_artifact_present_returns_path(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "workspace" / "output"
        out_dir.mkdir(parents=True)
        artifact = out_dir / "forest.png"
        artifact.write_bytes(b"\x89PNG\r\n\x1a\nfake png bytes")

        shape = TaskShape(
            kind="artifact", expected_extensions=(".png",),
            trigger="graphic",
        )
        verified = verify_artifacts(
            shape,
            "Saved successfully.\nARTIFACT: workspace/output/forest.png",
            workspace_root=tmp_path,
        )
        assert verified == str(artifact)

    def test_no_path_in_response_raises(self, tmp_path: Path) -> None:
        shape = TaskShape(
            kind="artifact", expected_extensions=(".png",),
            trigger="graphic",
        )
        with pytest.raises(ArtifactNotProduced) as excinfo:
            verify_artifacts(
                shape, "Here is some Python code, no actual file.",
                workspace_root=tmp_path,
            )
        assert "no file path mentioned" in str(excinfo.value)

    def test_path_does_not_exist_raises(self, tmp_path: Path) -> None:
        shape = TaskShape(
            kind="artifact", expected_extensions=(".png",),
            trigger="graphic",
        )
        with pytest.raises(ArtifactNotProduced) as excinfo:
            verify_artifacts(
                shape,
                "ARTIFACT: workspace/output/nonexistent.png",
                workspace_root=tmp_path,
            )
        assert "does not exist" in str(excinfo.value)

    def test_path_is_empty_raises(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "workspace" / "output"
        out_dir.mkdir(parents=True)
        empty = out_dir / "empty.png"
        empty.write_bytes(b"")  # 0 bytes

        shape = TaskShape(
            kind="artifact", expected_extensions=(".png",),
            trigger="graphic",
        )
        with pytest.raises(ArtifactNotProduced) as excinfo:
            verify_artifacts(
                shape,
                "ARTIFACT: workspace/output/empty.png",
                workspace_root=tmp_path,
            )
        assert "empty" in str(excinfo.value)

    def test_exception_carries_attempted_paths(self, tmp_path: Path) -> None:
        """The exception must include the list of paths that were
        tried + per-path reasons so the retry path's diagnostic is
        precise (not generic "task failed")."""
        shape = TaskShape(
            kind="artifact", expected_extensions=(".png",),
            trigger="graphic",
        )
        with pytest.raises(ArtifactNotProduced) as excinfo:
            verify_artifacts(
                shape,
                "ARTIFACT: workspace/output/missing1.png\n"
                "Or: `workspace/output/missing2.png`",
                workspace_root=tmp_path,
            )
        exc = excinfo.value
        assert len(exc.attempted_paths) == 2
        # Each entry is (path, reason)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in exc.attempted_paths)


# ── ContextVar plumbing ────────────────────────────────────────────


class TestActiveTaskShapeContextVar:

    def setup_method(self) -> None:
        set_active_task_shape(None)

    def test_get_returns_set_value(self) -> None:
        shape = TaskShape(kind="artifact", expected_extensions=(".pdf",))
        set_active_task_shape(shape)
        assert get_active_task_shape() is shape

    def test_default_is_none(self) -> None:
        assert get_active_task_shape() is None


# ── Wired into orchestrator ───────────────────────────────────────


class TestWiredIntoOrchestrator:
    """Source-grep contracts: the orchestrator must invoke the
    classifier at handle entry and the verifier at run-crew end.
    Without these, the module is dead code."""

    def test_classify_called_in_handle_locked(self) -> None:
        from pathlib import Path
        import re
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "agents" / "commander" / "orchestrator.py"
        ).read_text(encoding="utf-8")
        # Slice the _handle_locked function — find its def, then the
        # NEXT method def at the same column-4 indent.  (The
        # ``_run_crew`` methods appear EARLIER in the file than
        # ``_handle_locked`` so we can't use them as the end marker.)
        m = re.search(r"\n    def _handle_locked\(", src)
        assert m is not None, "_handle_locked must exist"
        idx_start = m.start() + 1
        # Find the next method definition.
        next_def = re.search(
            r"\n    def [a-zA-Z_]+\(",
            src[idx_start + 30:],
        )
        idx_end = (
            idx_start + 30 + next_def.start()
            if next_def is not None else len(src)
        )
        body = src[idx_start:idx_end]
        assert "classify_task" in body, (
            "_handle_locked must call classify_task at entry"
        )
        assert "set_active_task_shape" in body, (
            "_handle_locked must propagate the shape via ContextVar"
        )

    def test_directive_injected_in_run_crew_inner(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "agents" / "commander" / "orchestrator.py"
        ).read_text(encoding="utf-8")
        idx_start = src.index("def _run_crew_inner(")
        body = src[idx_start:idx_start + 8000]
        assert "build_artifact_directive" in body, (
            "_run_crew_inner must augment crew_task with directive when artifact"
        )

    def test_verify_called_before_return_in_run_crew_inner(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "agents" / "commander" / "orchestrator.py"
        ).read_text(encoding="utf-8")
        idx_start = src.index("def _run_crew_inner(")
        # Slice through the verify integration block.
        body = src[idx_start:]
        # Find the first end-of-function marker (the next def at column 4).
        next_def = body.find("\n    def ", 50)
        body = body[:next_def] if next_def > 0 else body
        assert "verify_artifacts" in body, (
            "_run_crew_inner must verify the artifact before returning"
        )
        assert "ArtifactNotProduced" in body, (
            "_run_crew_inner must catch ArtifactNotProduced explicitly"
        )
