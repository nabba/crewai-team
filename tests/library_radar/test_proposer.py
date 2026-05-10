"""Tests for app.library_radar.proposer."""

from __future__ import annotations

import logging
from pathlib import Path

from app.library_radar import proposer


_RADAR_TEXT_FRAMEWORK = (
    "[frameworks] LangGraph 0.5: Stateful agent graph framework "
    "for multi-step reasoning. Action: evaluate vs CrewAI for "
    "long-horizon orchestration."
)
_RADAR_TEXT_TOOL = (
    "[tools] uv 0.4: Rust-based Python package manager replacing "
    "pip+venv. Action: trial in CI for build-time speedup."
)
_RADAR_TEXT_RESEARCH = (
    "[research] arxiv:2611.12345: Constitutional AI for boundary "
    "drift detection. Action: read + summarise."
)
_RADAR_TEXT_MODEL = (
    "[models] kimi-k2-0905: Trillion-param MoE. Action: benchmark "
    "for reasoning."
)


def _seed_requirements(tmp_path: Path, packages: list[str]) -> Path:
    p = tmp_path / "requirements.txt"
    p.write_text("\n".join(packages) + "\n")
    return p


# ── parsing ──────────────────────────────────────────────────────────────


def test_parse_radar_line_extracts_fields() -> None:
    d = proposer._parse_radar_line(_RADAR_TEXT_FRAMEWORK)
    assert d is not None
    assert d.category == "frameworks"
    assert "LangGraph 0.5" in d.title
    assert "Stateful agent graph" in d.summary
    assert "evaluate vs CrewAI" in d.action


def test_parse_radar_line_returns_none_for_unparseable() -> None:
    assert proposer._parse_radar_line("not a radar line") is None
    assert proposer._parse_radar_line("") is None


def test_extract_package_names_filters_stopwords() -> None:
    names = proposer._extract_package_names("the new framework supports python")
    # "the", "new", "framework", "python" are stopwords; "supports" is too short pure-alpha.
    assert "the" not in names
    assert "framework" not in names


def test_extract_package_names_keeps_compound_tokens() -> None:
    names = proposer._extract_package_names("langgraph and llm-agent and uv")
    assert "langgraph" in names
    assert "llm-agent" in names
    assert "uv" not in names  # too short pure-alpha → filtered


# ── filtering by category ────────────────────────────────────────────────


def test_filters_to_frameworks_and_tools_only(tmp_path: Path) -> None:
    out = proposer.run_one_pass(
        proposed_dir=tmp_path,
        discoveries=[
            _RADAR_TEXT_FRAMEWORK,
            _RADAR_TEXT_TOOL,
            _RADAR_TEXT_RESEARCH,
            _RADAR_TEXT_MODEL,
        ],
        requirements_path=_seed_requirements(tmp_path, []),
    )
    assert out["status"] == "ok"
    assert out["n_relevant"] == 2  # framework + tool, NOT research/models
    assert out["drafts_written"] == 2


# ── filtering against requirements.txt ───────────────────────────────────


def test_filters_out_already_pinned(tmp_path: Path) -> None:
    out = proposer.run_one_pass(
        proposed_dir=tmp_path,
        discoveries=[_RADAR_TEXT_FRAMEWORK],
        requirements_path=_seed_requirements(tmp_path, ["langgraph==0.5"]),
    )
    # langgraph already pinned → filtered out → all_already_pinned
    assert out["status"] == "all_already_pinned"
    assert out["drafts_written"] == 0


def test_requirements_parser_handles_extras_and_specifiers(tmp_path: Path) -> None:
    pinned = proposer._read_requirements(
        _seed_requirements(tmp_path, [
            "anthropic==0.97.0",
            "pydantic[dotenv]>=2.10",
            "fastapi  ~=0.110",
            "  # this is a comment",
            "",
            "-e .",
            "-r other.txt",
        ]),
    )
    assert pinned == {"anthropic", "pydantic", "fastapi"}


# ── output structure ─────────────────────────────────────────────────────


def test_writes_proposal_with_expected_sections(tmp_path: Path) -> None:
    proposer.run_one_pass(
        proposed_dir=tmp_path,
        discoveries=[_RADAR_TEXT_FRAMEWORK],
        requirements_path=_seed_requirements(tmp_path, []),
    )
    drafts = list(tmp_path.glob("*.md"))
    assert len(drafts) == 1
    text = drafts[0].read_text()
    assert "Library adoption proposal" in text
    assert "LangGraph" in text
    assert "## Summary" in text
    assert "## Adoption checklist" in text
    assert "## Operator action" in text
    assert "scope_tech_radar" in text


def test_dedups_by_signature(tmp_path: Path) -> None:
    first = proposer.run_one_pass(
        proposed_dir=tmp_path,
        discoveries=[_RADAR_TEXT_FRAMEWORK],
        requirements_path=_seed_requirements(tmp_path, []),
    )
    assert first["drafts_written"] == 1
    second = proposer.run_one_pass(
        proposed_dir=tmp_path,
        discoveries=[_RADAR_TEXT_FRAMEWORK],
        requirements_path=_seed_requirements(tmp_path, []),
    )
    assert second["drafts_written"] == 0
    assert second["drafts_skipped_dedup"] == 1


# ── empty + disabled paths ───────────────────────────────────────────────


def test_no_evidence_returns_status(tmp_path: Path) -> None:
    out = proposer.run_one_pass(
        proposed_dir=tmp_path,
        discoveries=[],
        requirements_path=_seed_requirements(tmp_path, []),
    )
    assert out["status"] == "no_evidence"


def test_no_relevant_when_only_models(tmp_path: Path) -> None:
    out = proposer.run_one_pass(
        proposed_dir=tmp_path,
        discoveries=[_RADAR_TEXT_MODEL, _RADAR_TEXT_RESEARCH],
        requirements_path=_seed_requirements(tmp_path, []),
    )
    assert out["status"] == "no_relevant"


def test_disabled_short_circuits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LIBRARY_RADAR_ENABLED", "false")
    out = proposer.run_one_pass(
        proposed_dir=tmp_path,
        discoveries=[_RADAR_TEXT_FRAMEWORK],
        requirements_path=_seed_requirements(tmp_path, []),
    )
    assert out["status"] == "disabled"


# ── daemon discipline ────────────────────────────────────────────────────


def test_disabled_short_circuits_start(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LIBRARY_RADAR_ENABLED", "false")
    assert proposer._enabled() is False
    with caplog.at_level(logging.INFO, logger="app.library_radar.proposer"):
        proposer.start()
    assert any("disabled via" in r.message for r in caplog.records)


def test_stop_sets_event() -> None:
    proposer.stop()
    assert proposer._stop_event.is_set()
