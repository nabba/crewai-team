"""Tests for app.creativity.analogy_index."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.creativity.analogy_index import (
    AnalogyEntry,
    DomainExample,
    add_entry,
    list_all,
    query_analogies,
    _reset_for_tests,
)


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path):
    p = tmp_path / "analogy_index.jsonl"
    _reset_for_tests(p)
    yield p
    _reset_for_tests(None)


def _entry(
    eid: str,
    *,
    sig: str = "feedback_loop_with_delay",
    description: str = "feedback loop with delay between cause and effect",
    domains: list[tuple[str, str, str]] | None = None,
) -> AnalogyEntry:
    examples = [
        DomainExample(domain=d, title=t, summary=s)
        for d, t, s in (domains or [
            ("control_theory", "delayed thermostat", "fluctuates"),
            ("ecology", "predator-prey lag", "boom-bust"),
        ])
    ]
    return AnalogyEntry(
        id=eid,
        structure_signature=sig,
        structure_description=description,
        domain_examples=examples,
    )


# ── round-trip ─────────────────────────────────────────────────────────


def test_add_and_list_round_trip() -> None:
    e1 = _entry("e1")
    add_entry(e1)
    out = list_all()
    assert len(out) == 1
    assert out[0].id == "e1"
    assert len(out[0].domain_examples) == 2
    assert out[0].domain_examples[0].domain == "control_theory"


def test_duplicate_id_last_write_wins() -> None:
    e1 = _entry("dup", description="initial description")
    e2 = _entry("dup", description="updated description")
    add_entry(e1)
    add_entry(e2)
    out = list_all()
    assert len(out) == 1
    assert out[0].structure_description == "updated description"


def test_list_handles_missing_file() -> None:
    """Querying before any add yields an empty list."""
    assert list_all() == []


def test_list_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "broken.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '{"id": "ok", "structure_signature": "x", '
        '"structure_description": "y", "domain_examples": []}\n'
        "this is not JSON\n"
        '{"id": "ok2", "structure_signature": "x", '
        '"structure_description": "y", "domain_examples": []}\n',
    )
    out = list_all(path=p)
    assert {e.id for e in out} == {"ok", "ok2"}


# ── query ──────────────────────────────────────────────────────────────


def test_query_returns_top_matches() -> None:
    add_entry(_entry(
        "feedback",
        description="feedback loop with delay between cause and effect",
    ))
    add_entry(_entry(
        "antigen",
        sig="novel_antigen_recognition",
        description="recognising a novel pattern outside the existing signature library",
        domains=[("immunology", "novel antigen", "MHC class I"),
                 ("cybersecurity", "zero-day", "signature gap")],
    ))
    add_entry(_entry(
        "trash_can",
        sig="container_with_named_items",
        description="container with named addressable items",
    ))

    # Query is lexically close to the "feedback" entry's description,
    # since hash-trick embedding is token-overlap-based.
    matches = query_analogies(
        "feedback loop with delay between cause and effect signals",
        top_k=3,
    )
    assert len(matches) >= 1
    assert matches[0].entry.id == "feedback"
    # Top match has higher similarity than later matches.
    if len(matches) >= 2:
        assert matches[0].similarity >= matches[1].similarity


def test_query_below_min_similarity_filtered() -> None:
    add_entry(_entry("e1", description="quantum entanglement nonlocality"))
    matches = query_analogies(
        "compound interest formula",
        top_k=5,
        min_similarity=0.30,
    )
    assert matches == []


def test_query_caps_at_top_k() -> None:
    for i in range(20):
        add_entry(_entry(
            f"e{i}",
            sig=f"sig_{i}",
            description=f"feedback loop variant {i} with delay structure",
        ))
    matches = query_analogies(
        "feedback loop with delay",
        top_k=5,
        min_similarity=0.0,
    )
    assert len(matches) == 5


def test_exclude_domains_filters_home_domain_only_entries() -> None:
    """If every example of an entry is in the excluded set, that entry is
    skipped — operator wants ANALOGUES from OTHER domains."""
    add_entry(_entry(
        "in_domain",
        description="feedback loop in control theory only",
        domains=[("control_theory", "thermostat", "x")],
    ))
    add_entry(_entry(
        "cross_domain",
        description="feedback loop with delay in another domain",
        domains=[
            ("control_theory", "thermostat", "x"),
            ("ecology", "predator-prey", "y"),
        ],
    ))
    matches = query_analogies(
        "feedback loop with delay",
        exclude_domains={"control_theory"},
        min_similarity=0.0,
    )
    ids = {m.entry.id for m in matches}
    assert "in_domain" not in ids
    assert "cross_domain" in ids


def test_query_empty_input_returns_empty() -> None:
    add_entry(_entry("e1"))
    assert query_analogies("") == []
    assert query_analogies("   ") == []


# ── enable / disable ───────────────────────────────────────────────────


def test_disabled_skip_writes_and_queries(monkeypatch) -> None:
    monkeypatch.setenv("ANALOGY_INDEX_ENABLED", "false")
    e = _entry("e1")
    assert add_entry(e) is False
    # Even if the file existed, queries return empty when disabled.
    assert query_analogies("anything") == []
