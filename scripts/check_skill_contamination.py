#!/usr/bin/env python3
"""Skill-retrieval contamination check — manual follow-up to commit 9fb067e.

Run this ~2 weeks after the 4-layer fix landed (target: 2026-05-16) to
confirm cross-topic contamination has not resurfaced. Originally an
exploratory routine; pulled local because the hottest checks (live
chromadb stats, gateway docker logs) only work from the host.

Usage (from the host, full coverage):
    docker exec -it crewai-team-gateway-1 \
        python /app/scripts/check_skill_contamination.py

Or from the host venv (chromadb sections will be empty — the index lives
in the docker compose stack and is not reachable from outside it):
    .venv/bin/python scripts/check_skill_contamination.py

Env knobs:
    SINCE         ISO date — default 2026-05-02 (the day the fix shipped)
    SAMPLE_SIZE   how many recent user messages to score — default 30
    GATEWAY_NAME  docker container name — default crewai-team-gateway-1
"""

from __future__ import annotations

import collections
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── repo bootstrap ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SINCE = os.environ.get("SINCE", "2026-05-02")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "30"))
GATEWAY_NAME = os.environ.get("GATEWAY_NAME", "crewai-team-gateway-1")

CONV_DB = ROOT / "workspace" / "conversations.db"
SKILLS_DIR = ROOT / "workspace" / "skills"


def _hr(title: str) -> None:
    print(f"\n{'═' * 72}\n  {title}\n{'═' * 72}")


# ── 1. WRONG CREW signal (docker logs) ──────────────────────────────────────


def check_wrong_crew_log() -> int:
    """Count "WRONG CREW" warnings emitted by the gateway since SINCE.

    The signal is `logger.warning("Vetting FAILED ... WRONG CREW ...")`
    in app/agents/commander/orchestrator.py:3042. Only visible in docker
    logs (no journal entry — the retry path is recovery, not error).
    """
    _hr(f"1. WRONG CREW vetting rejections since {SINCE}")
    try:
        # `docker logs --since` accepts ISO date; fall back to a plain
        # cat-and-grep when --since is unsupported (older docker).
        out = subprocess.run(
            ["docker", "logs", GATEWAY_NAME, "--since", SINCE],
            capture_output=True, text=True, check=False, timeout=60,
        )
        log = (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        print(f"  docker logs unavailable: {exc}")
        return -1

    pattern = re.compile(r"Vetting FAILED.*WRONG CREW", re.I)
    matches = [ln for ln in log.splitlines() if pattern.search(ln)]
    print(f"  {len(matches)} WRONG CREW retries since {SINCE}")
    for ln in matches[-10:]:
        print(f"    {ln[:200]}")
    return len(matches)


# ── 2. Subject-less detection: false-positive sweep ────────────────────────


def check_subjectless_falsepos() -> int:
    """Cases where the loader returned empty for a subject-less message
    with no recoverable history. Most are legitimate (the user genuinely
    sent "ok" with nothing to inherit topic from), but a spike implies
    history is being dropped before reaching the loader.
    """
    _hr("2. Subject-less detection — empty-history events")
    try:
        out = subprocess.run(
            ["docker", "logs", GATEWAY_NAME, "--since", SINCE],
            capture_output=True, text=True, check=False, timeout=60,
        )
        log = (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        print(f"  docker logs unavailable: {exc}")
        return -1

    matches = [
        ln for ln in log.splitlines()
        if "subjectless message" in ln and "no conversation history" in ln
    ]
    print(f"  {len(matches)} no-history skips since {SINCE}")
    for ln in matches[-5:]:
        print(f"    {ln[:200]}")
    if len(matches) > 50:
        print("  ⚠ high count — investigate whether history is being dropped "
              "upstream of _load_relevant_skills")
    return len(matches)


# ── 3. Placeholder-marker skills still being created ───────────────────────


def check_placeholder_skills() -> int:
    """Inspect the SkillRecord index for active records whose topic
    carries placeholder markers. Their continued presence means the
    upstream Learner is still emitting them — the retrieval-side filter
    is masking, not curing.
    """
    _hr("3. Placeholder-marker skills in the active index")
    try:
        from app.self_improvement.integrator import list_records
        from app.agents.commander.context import _is_low_quality_skill_topic
    except Exception as exc:
        print(f"  cannot import index: {exc}")
        return -1

    try:
        recs = list_records(status="active", limit=2000)
    except Exception as exc:
        print(f"  list_records failed (chromadb reachable?): {exc}")
        return -1

    if not recs:
        print("  0 active records returned — likely running outside the "
              "gateway container (chromadb is not reachable from host). "
              "Re-run with `docker exec ... python /app/scripts/...`.")
        return -1

    bad = [r for r in recs if _is_low_quality_skill_topic(r.topic)]
    print(f"  {len(bad)} / {len(recs)} active records have placeholder markers")
    by_kb = collections.Counter(r.kb for r in bad)
    for kb, n in by_kb.most_common():
        print(f"    {kb}: {n}")
    for r in bad[:8]:
        print(f"    {r.created_at[:10]} {r.kb} :: {r.topic[:80]}")
    if bad:
        print("  → upstream Learner needs review; the gate is filtering them "
              "but they shouldn't be created in the first place")
    return len(bad)


# ── 4. Distance histogram on a sample of recent user messages ──────────────


def check_distance_distribution() -> dict[str, int]:
    """Run search_skills_scored against the live chromadb for SAMPLE_SIZE
    recent user messages. Histogram the top-1 cosine distance, see
    whether _SKILL_DISTANCE_CEILING (currently 0.55) is well-calibrated.
    """
    _hr(f"4. Distance distribution across {SAMPLE_SIZE} recent user msgs")
    if not CONV_DB.exists():
        print(f"  {CONV_DB} not found — skipping")
        return {}

    try:
        from app.agents.commander.context import _is_subjectless_message
        from app.self_improvement.integrator import search_skills_scored
    except Exception as exc:
        print(f"  cannot import retrieval modules: {exc}")
        return {}

    since_dt = datetime.fromisoformat(SINCE).replace(tzinfo=timezone.utc)
    rows: list[tuple[str, str]] = []
    with sqlite3.connect(str(CONV_DB)) as c:
        rows = list(c.execute(
            "SELECT ts, content FROM messages "
            "WHERE role='user' AND ts >= ? "
            "ORDER BY ts DESC LIMIT ?",
            (since_dt.isoformat(), SAMPLE_SIZE),
        ))
    if not rows:
        print(f"  no user messages since {SINCE}")
        return {}

    bins = collections.Counter()
    subjectless_count = 0
    scored_count = 0
    sample_lines: list[str] = []
    for ts, content in rows:
        text = (content or "").strip()
        if not text:
            continue
        if _is_subjectless_message(text):
            subjectless_count += 1
            continue
        try:
            scored = search_skills_scored(text, n=3)
        except Exception as exc:
            print(f"  scoring failed for one row: {exc}")
            continue
        if not scored:
            bins["empty"] += 1
            continue
        scored_count += 1
        top_dist = scored[0][1]
        if top_dist < 0.30:
            bins["[0.00,0.30)"] += 1
        elif top_dist < 0.40:
            bins["[0.30,0.40)"] += 1
        elif top_dist < 0.45:
            bins["[0.40,0.45)"] += 1
        elif top_dist < 0.55:
            bins["[0.45,0.55)"] += 1
        elif top_dist < 0.65:
            bins["[0.55,0.65)"] += 1
        else:
            bins["[0.65, ∞)"] += 1
        sample_lines.append(
            f"    d={top_dist:.3f}  msg={text[:50]:<52} → {scored[0][0].topic[:60]}"
        )

    print(f"  {subjectless_count} of {len(rows)} recent msgs were subject-less "
          f"(skipped — they wouldn't surface skills regardless)")
    if scored_count == 0 and bins.get("empty", 0) == len(rows) - subjectless_count:
        print("  ⚠ every scoring attempt returned an empty result. The "
              "chromadb index is likely unreachable from this process — "
              "re-run inside the gateway container for valid data.")
    print()
    print("  Top-1 cosine distance histogram (gate fires above 0.55):")
    bin_order = [
        "[0.00,0.30)", "[0.30,0.40)", "[0.40,0.45)",
        "[0.45,0.55)", "[0.55,0.65)", "[0.65, ∞)", "empty",
    ]
    for b in bin_order:
        n = bins.get(b, 0)
        bar = "█" * n
        print(f"    {b:<12} {n:>3}  {bar}")
    print()
    print("  sample (last 10):")
    for ln in sample_lines[-10:]:
        print(ln)

    # Recommendation
    in_band = bins["[0.20,0.30)"] if False else (
        bins["[0.00,0.30)"] + bins["[0.30,0.40)"] + bins["[0.40,0.45)"]
    )
    near_ceiling = bins["[0.45,0.55)"]
    over_ceiling = bins["[0.55,0.65)"] + bins["[0.65, ∞)"]
    total = in_band + near_ceiling + over_ceiling
    if total >= 10:
        if near_ceiling / total >= 0.30:
            print("  → many top-1 matches sit at 0.45–0.55. Consider tightening "
                  "_SKILL_DISTANCE_CEILING from 0.55 to 0.45.")
        elif (in_band - bins.get("[0.40,0.45)", 0)) / total >= 0.95:
            print("  → almost everything lands tightly under 0.40. Consider "
                  "relaxing _SKILL_DISTANCE_CEILING from 0.55 to 0.65 to "
                  "recover legitimate weaker matches.")
        else:
            print("  → distribution looks healthy — current 0.55 ceiling fine.")
    else:
        print("  → too few non-subject-less samples to recommend a change "
              f"({total} scored). Re-run with SAMPLE_SIZE=100.")
    return dict(bins)


# ── 5. Verdict ─────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Skill-retrieval contamination check — since {SINCE}")
    print(f"Repo: {ROOT}")
    print(f"Conv DB: {CONV_DB.exists()}, container: {GATEWAY_NAME}")

    wrong_crew = check_wrong_crew_log()
    falsepos = check_subjectless_falsepos()
    placeholders = check_placeholder_skills()
    bins = check_distance_distribution()

    _hr("Verdict")
    if wrong_crew == 0:
        print("  ✓ No WRONG CREW vetting retries since the fix shipped.")
    elif wrong_crew > 0:
        print(f"  ⚠ {wrong_crew} WRONG CREW retries — eyeball the lines above. "
              "Cross-topic contamination resembling the May 2026 incident "
              "would show as a coding/research dispatch with a topic "
              "unrelated to the recent conversation.")
    if placeholders > 0:
        print(f"  ⚠ {placeholders} placeholder-marker skills still in the index. "
              "Open an issue against the Learner / Integrator (the retrieval "
              "gate is filtering them but the upstream creation path leaks).")
    if falsepos > 50:
        print(f"  ⚠ {falsepos} subjectless-no-history events — verify that "
              "conversation_history is reaching _load_relevant_skills.")
    print()
    print("  If the distance histogram suggests a threshold change, edit:")
    print("    app/agents/commander/context.py :: _SKILL_DISTANCE_CEILING")
    print("    docs/MEMORY_ARCHITECTURE.md §6.7.1 (table row 3)")
    print("  and open a draft PR — do NOT push directly to main.")


if __name__ == "__main__":
    main()
