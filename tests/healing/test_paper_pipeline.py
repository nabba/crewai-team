"""Tests for ``app.episteme.paper_pipeline`` (Phase C #3)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.episteme import paper_pipeline
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(paper_pipeline, "_PROPOSALS_PATH",
                        tmp_path / "proposed_experiments.jsonl")
    monkeypatch.setattr(paper_pipeline, "_SEEN_PATH",
                        tmp_path / "papers_seen.json")

    sent: list[str] = []
    monkeypatch.setattr(
        "app.healing.handlers._common.send_signal_alert",
        lambda body, **kw: sent.append(body) or True,
    )
    monkeypatch.setattr(
        "app.healing.handlers._common.audit_event",
        lambda *a, **k: None,
    )
    yield tmp_path, sent


def _atom_xml(entries: list[dict]) -> str:
    parts = []
    for e in entries:
        parts.append(f"""
<entry>
  <id>{e['id']}</id>
  <title>{e['title']}</title>
  <summary>{e['abstract']}</summary>
  <published>{e['published']}</published>
  <category term="cs.AI"/>
</entry>
""")
    return "<feed>" + "".join(parts) + "</feed>"


def test_build_terms_falls_back_to_always_on(isolated, monkeypatch):
    from app.episteme import paper_pipeline
    monkeypatch.setattr("app.companion.interest_model.current_profile",
                        lambda: {"topics": []})
    terms = paper_pipeline._build_terms()
    assert "agent" in terms or "alignment" in terms


def test_parse_atom_drops_old_entries(isolated):
    from app.episteme import paper_pipeline
    now = datetime.now(timezone.utc)
    entries = [
        {"id": "http://arxiv.org/abs/2401.fresh",
         "title": "Fresh paper", "abstract": "A new approach.",
         "published": (now - timedelta(days=5)).isoformat()},
        {"id": "http://arxiv.org/abs/2301.old",
         "title": "Old paper", "abstract": "Outdated.",
         "published": (now - timedelta(days=400)).isoformat()},
    ]
    xml = _atom_xml(entries)
    out = paper_pipeline._parse_atom(xml, lookback_days=14)
    assert len(out) == 1
    assert out[0]["title"] == "Fresh paper"


def test_run_proposes_new_papers(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.episteme import paper_pipeline

    now = datetime.now(timezone.utc)
    fake_papers = [{
        "id": "http://arxiv.org/abs/2505.12345",
        "title": "Self-Improving Agents via Reflection",
        "abstract": "We show that autonomous agents can improve via reflection on past trajectories.",
        "published": (now - timedelta(days=2)).isoformat(),
        "categories": ["cs.AI"],
    }]
    monkeypatch.setattr(paper_pipeline, "_fetch_arxiv_atom",
                        lambda q, max_results: "<feed/>")
    monkeypatch.setattr(paper_pipeline, "_parse_atom",
                        lambda xml, lookback_days: fake_papers)
    monkeypatch.setattr(paper_pipeline, "_summarize",
                        lambda title, abs_: {
                            "summary": "Reflection helps agents learn.",
                            "implications": ["wire reflection into the affect loop"],
                            "experiment": "Run an A/B with reflection on/off for 7 days.",
                            "relevance": 0.92,
                        })
    monkeypatch.setattr(paper_pipeline, "_build_terms",
                        lambda: ["self-improvement", "reflection"])

    summary = paper_pipeline.run()
    assert summary["proposed"] == 1
    assert summary["alerted"] is True
    rows = (tmp_path / "proposed_experiments.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    proposal = json.loads(rows[0])
    assert proposal["title"].startswith("Self-Improving")
    assert proposal["relevance"] == pytest.approx(0.92)
    assert "Self-Improving" in sent[0]


def test_run_dedups_seen_papers(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.episteme import paper_pipeline

    # Pre-seed seen with the paper id.
    (tmp_path / "papers_seen.json").write_text(
        json.dumps(["http://arxiv.org/abs/2505.dup"])
    )
    fake_paper = {
        "id": "http://arxiv.org/abs/2505.dup",
        "title": "Duplicate", "abstract": "x",
        "published": datetime.now(timezone.utc).isoformat(),
        "categories": [],
    }
    monkeypatch.setattr(paper_pipeline, "_fetch_arxiv_atom",
                        lambda q, max_results: "")
    monkeypatch.setattr(paper_pipeline, "_parse_atom",
                        lambda xml, lookback_days: [fake_paper])
    monkeypatch.setattr(paper_pipeline, "_summarize",
                        lambda title, abs_: {"summary": "x", "implications": [],
                                              "experiment": "x", "relevance": 0.5})
    monkeypatch.setattr(paper_pipeline, "_build_terms",
                        lambda: ["foo"])

    summary = paper_pipeline.run()
    assert summary["proposed"] == 0
    assert sent == []


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("PAPER_PIPELINE_ENABLED", "0")
    from app.episteme import paper_pipeline
    summary = paper_pipeline.run()
    assert summary["ran"] is False
