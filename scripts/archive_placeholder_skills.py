#!/usr/bin/env python3
"""Archive active SkillRecord entries whose topic carries a placeholder marker.

Companion to the upstream fix that stops the Learner / auto-skill path from
emitting these in the first place. Once the integrate() guard ships, the
index will not accumulate new placeholder records — but four already exist
as of the May 2 smoke run, including the two that triggered the
2026-05-02 cross-topic contamination incident.

Sets ``status="archived"`` on each matching record (no deletion); the
original ``provenance.gap_id`` / ``created_from_gap`` field is preserved
so the audit trail back to the originating gap stays intact.

Usage (from inside the gateway container so chromadb is reachable):
    docker exec -it crewai-team-gateway-1 \\
        python /app/scripts/archive_placeholder_skills.py            # dry-run
    docker exec -it crewai-team-gateway-1 \\
        python /app/scripts/archive_placeholder_skills.py --apply    # mutate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Persist status=archived on matching records. Default is dry-run.",
    )
    parser.add_argument(
        "--limit", type=int, default=2000,
        help="Max active records to scan (default: 2000)",
    )
    args = parser.parse_args()

    try:
        from app.self_improvement.integrator import (
            list_records, update_record, _topic_has_placeholder_marker,
        )
    except Exception as exc:
        print(f"ERROR: cannot import integrator: {exc}", file=sys.stderr)
        return 2

    try:
        active = list_records(status="active", limit=args.limit)
    except Exception as exc:
        print(f"ERROR: list_records failed (chromadb reachable?): {exc}",
              file=sys.stderr)
        return 2

    if not active:
        print(
            "No active records returned. Likely running outside the gateway "
            "container — chromadb is not reachable from the host. Re-run with "
            "`docker exec ... python /app/scripts/...`."
        )
        return 1

    bad = [r for r in active if _topic_has_placeholder_marker(r.topic)]
    print(f"Scanned {len(active)} active records. "
          f"{len(bad)} carry placeholder markers.")

    if not bad:
        return 0

    for r in bad:
        gap = r.provenance.get("gap_id", "")
        print(f"  {r.created_at[:10]} {r.kb:>13} :: {r.topic[:80]}"
              + (f"  (gap={gap})" if gap else ""))

    if not args.apply:
        print("\nDry-run only — re-run with --apply to archive.")
        return 0

    archived = 0
    for r in bad:
        r.status = "archived"
        ok = update_record(r)
        if ok:
            archived += 1
            print(f"  archived: {r.id}")
        else:
            print(f"  FAILED:   {r.id}", file=sys.stderr)

    print(f"\nArchived {archived} / {len(bad)} placeholder records.")
    return 0 if archived == len(bad) else 1


if __name__ == "__main__":
    sys.exit(main())
