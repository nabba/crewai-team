# Knowledge Maintenance

**Status (2026-05-16):** Q16 Theme 5 — decade-resilience hardening,
knowledge management at decade-scale. Two artifacts: wiki staleness
probe + CLAUDE.md compaction proposer.

## Why this exists

Documentation rots. The wiki accumulates pages over years; many of
them say things like "current tech stack" or "Andrus's role" that
decay silently. CLAUDE.md grows monotonically as Q-batches ship —
in 5 years it's unreadable.

The system already has:

  * `wiki_index_reconciler` — keeps the INDEX honest about
    structure.
  * `docs/claude-md-archive.md` — used as a one-off offload during
    Q1–Q15.

What was missing: a **cyclic** offload + a content-freshness surface.
This theme fills both gaps.

## Wiki staleness probe (38th healing monitor)

`app/healing/monitors/wiki_staleness.py`. Daily probe, weekly
internal cadence.

### What it does

  1. Walks `wiki/` for markdown files.
  2. For each, computes `age_days` from `mtime`.
  3. Stale = age > 365 days.
  4. Per-file 90-day dedup (each page surfaces at most quarterly).
  5. Once per week, sends a Signal digest naming the 10 stalest
     pages, with three operator-action suggestions:
       * `touch` after a review pass
       * move under `wiki/archive/`
       * remove if no longer relevant

### What it deliberately doesn't do

  * Auto-edit any wiki page. The wiki is the operator's narrative;
    refresh decisions are operator-only.
  * File CRs. The wiki-index reconciler already does that for
    structural reconciliation; content freshness is softer signal
    and shouldn't be CR-volume.

### Categorical exclusions

These directories are auto-archive-by-design and NEVER alert:

```
wiki/self/legacy/
wiki/self/value_reflections/
wiki/self/quarterly_reviews/
wiki/governance/
wiki/archive/
```

### Master switch

`wiki_staleness_monitor_enabled` (default ON).

## CLAUDE.md compaction proposer

`app/self_improvement/claude_md_compaction.py`. Daily probe with
internal annual cadence (guarded by `_already_proposed_this_year`).

### What it does

  1. Locates CLAUDE.md candidates (project root + parents + custom
     paths from `CLAUDE_MD_PATHS` env var).
  2. Splits each into HEAD (everything before the first Q-batch
     bullet) + KEEP (recent dated entries within 6 months) +
     ARCHIVE (older or undated entries).
  3. Composes three artifacts under
     `workspace/self_improvement/claude_md_compaction/<year>/`:
       * `CLAUDE.md.compacted.md` — proposed replacement
       * `CLAUDE.md.archive.md` — offloaded history
       * `CLAUDE.md.notes.md` — operator next-steps + counts
  4. Sends a Signal notification on first generation per year.

### Operator workflow

After receiving the Signal alert:

```sh
# Review the three files
ls workspace/self_improvement/claude_md_compaction/2027/

# If both look right:
cp workspace/.../CLAUDE.md.archive.md \
   crewai-team/docs/claude-md-archive-2026.md
cp workspace/.../CLAUDE.md.compacted.md /Users/andrus/BotArmy/CLAUDE.md

# If anything looks wrong:
rm -rf workspace/self_improvement/claude_md_compaction/2027/
# Next annual pass will retry.
```

### What this is NOT

  * NOT auto-applied. CLAUDE.md often sits OUTSIDE the git repo —
    the CR system writes into the repo, which is the wrong target.
  * NOT LLM-rewritten. Pure structural split; no paraphrase-drift
    risk.
  * NOT a CR producer. Proposal lands as plain files; operator
    picks up via Signal alert or directly from workspace/.

### Composition contract

  * **Head section** (everything before the first Q-batch bullet)
    is preserved verbatim.
  * Each **Q-batch entry**'s date is parsed from the ISO date
    in the bullet text (e.g. `2026-05-16`).
  * Entries with age ≤ 6 months: KEPT.
  * Entries older OR undated: ARCHIVED.
  * Undated entries default to ARCHIVE because in this codebase's
    convention, the older entries are the ones missing dates
    (predating the dated convention).
  * No paraphrasing — the proposal is a pure structural split.

### Master switch

`claude_md_compaction_enabled` (default ON).

## When the alerts surface: triage

### Wiki staleness digest

For each named page, the operator picks ONE:

  1. **Refresh.** Open the page, verify content is still accurate,
     update anything stale, save (mtime gets refreshed → next
     probe sees fresh).
  2. **Archive.** Move under `wiki/archive/<topic>/` (the
     categorical exclusion list covers `wiki/archive/`). Page
     stays around for reference but stops aging the digest.
  3. **Prune.** Delete if no longer relevant. The wiki-index
     reconciler picks up the deletion on its next pass.

### CLAUDE.md compaction proposal

Review the three artifact files. Common patterns:

  * **Looks right.** Two `cp` commands to apply. The annual
    cadence guard ensures the next pass won't re-propose for the
    same year, regardless of whether you applied.
  * **Missed an entry.** The proposed compaction split a multi-line
    Q-batch entry incorrectly. Manually fix the compacted output
    before applying, then `touch` the proposal directory so the
    composer knows you've reviewed it.
  * **Don't want to compact this year.** Delete the proposal
    directory; the next year's pass will retry on the new size.

## Composing with the rest of the system

Both modules are **observational** and **operator-driven**. They
compose with:

  * `wiki_index_reconciler` — structural (index) vs. content
    (freshness). Same pipeline target (wiki/), different signal.
  * `notify_suppression_review` — both wiki and CLAUDE.md digests
    flow through the arbiter; very rarely should they be
    suppressed (low signal volume), but the suppression review
    catches edge cases.
  * Continuity ledger — neither emits a dedicated event kind;
    `summarise_drift` doesn't need to see "operator refreshed a
    wiki page" as identity-shaping.

## Deliberately deferred

  * **React dashboard surface for wiki staleness.** Signal digest
    is sufficient.
  * **Per-page change-history tracking.** Out of scope; the wiki
    is operator-narrated, not version-controlled in the same way
    code is.
  * **Auto-CR-generation for "obvious" wiki pruning.** Too risky;
    operator narrative deserves an operator review.
  * **Auto-apply of CLAUDE.md compaction.** Requires writing
    outside the git repo, which violates the CR system's contract.
    Manual two-file copy is the right ergonomics.
