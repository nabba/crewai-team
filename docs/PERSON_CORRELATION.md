# Person Correlation Stack — PROGRAM §42 (Q4.2)

> Per-person modality tracking, optional centrality, optional prescriptive
> nudges, and an optional social graph with shortest-path / community
> detection / bridge identification / graph-driven suggestions. Four
> levels of opt-in with progressively higher Goodhart risk. Two
> typed-phrase gates on the most invasive levels. **Default OFF at every
> level — nothing tracks unless you explicitly turn it on.**

---

## Table of contents

1. [Stance — why this is risky](#stance)
2. [Four levels at a glance](#levels)
3. [Storage layout](#storage)
4. [Master switches](#master-switches)
5. [Level 1 — Presence](#l1)
6. [Level 2 — Centrality](#l2)
7. [Level 3 — Suggestions](#l3)
8. [Level 4 — Social graph](#l4)
9. [L4.1 — Shortest-path queries](#l41)
10. [L4.2 — Community detection](#l42)
11. [L4.3 — Bridges and cut-vertices](#l43)
12. [L4.4 — Graph-driven suggestions](#l44)
13. [Mute / forget / opt-out semantics](#mute)
14. [Goodhart-of-the-indicator safeguards](#goodhart)
15. [DR exclusion](#dr)
16. [Integration with existing systems](#integration)
17. [Operator surfaces — React + Signal](#surfaces)
18. [What this layer deliberately does NOT do](#not)

---

<a id="stance"></a>

## 1. Stance — why this is risky

Tracking which people show up in your inputs is the closest this system
ever gets to surveilling third parties. Even though every byte lives on
your host and never leaves the box (see [DR exclusion](#dr)), the
*existence* of a "people score" — let alone a "social graph" — opens four
distinct failure modes:

1. **Goodhart of the indicator.** If a friend sees they have a "centrality
   score of 0.62," every future interaction is suspect.
2. **Prescriptive drift.** A system that says "you haven't talked to Maria
   in 6 weeks — call her?" is one step from "you SHOULD call Maria."
3. **Structural-role disclosure.** Telling you "Alice is a bridge between
   your work and family clusters" turns observation into instruction.
4. **Off-host leakage.** A backup that includes the social graph is a
   single bad rsync from being on a vendor disk.

The four-level architecture is **specifically designed so that turning
on one level does not implicitly turn on the next**. Each level has its
own switch; L4 and L4.4 each have a typed-phrase gate to defeat
accidental clicks.

---

<a id="levels"></a>

## 2. Four levels at a glance

| Level | Name                         | Default | Gate              | What it does |
|-------|------------------------------|---------|-------------------|--------------|
| L1    | Presence                     | OFF     | toggle            | Counts per modality per person |
| L2    | Centrality                   | OFF     | toggle (needs L1) | One operator-chosen score in [0, 1] |
| L3    | Suggestions                  | OFF     | toggle (needs L1) | Dormancy + responsiveness nudges, ≤3/briefing |
| L4    | Social graph                 | OFF     | **typed phrase** + L1 | Co-appearance edges with 3-month decay |
| L4.1  | Shortest-path queries        | OFF     | toggle (needs L4) | Operator-initiated BFS, logged |
| L4.2  | Community detection          | OFF     | toggle (needs L4) | Label propagation + modularity |
| L4.3  | Bridges / cut-vertices       | OFF     | toggle (needs L4) | Tarjan's algorithm |
| L4.4  | Graph-driven suggestions     | OFF     | **typed phrase** + L4 | Cluster-dormancy, bridge-maintenance, weak-tie nudges; shares L3 rate limit |

**Higher numbers are more invasive, more prescriptive, or both.**

---

<a id="storage"></a>

## 3. Storage layout

All files live under `workspace/companion/`. Nothing leaves the host
unless you explicitly back it up — and the DR exporter excludes every
graph-derived file by default (see §15).

```
workspace/companion/
├── person_profile.json                    # L1 snapshot (current people)
├── person_history.jsonl                   # L1 append-only sighting log
├── person_mutes.json                      # L1+L3 per-person mutes
├── person_centrality.json                 # L2 current scores
├── person_suggestion_mutes.json           # L3 per-person suggestion mute
├── person_suggestions_emitted.jsonl       # L3+L4.4 audit log of fired nudges
├── social_graph.json                      # L4 current edge map
├── social_graph_pair_mutes.json           # L4 per-pair mute
├── social_graph_path_opt_outs.json        # L4.1 per-person path opt-out
├── social_graph_query_log.jsonl           # L4.1 transparency log
├── social_graph_communities.json          # L4.2 current label assignment
├── social_graph_dissolved_clusters.json   # L4.2 operator-hidden clusters
└── social_graph_structural.json           # L4.3 bridges + articulations
```

Decay:
* L1 profile drops people not seen within `person_decay_months` (default 12).
* L4 edges use a 3-month half-life; edges below weight 0.1 drop entirely.

---

<a id="master-switches"></a>

## 4. Master switches

All flags live in `app/runtime_settings.py`. All bool defaults are
`False` except `person_centrality_formula = "frequency"` (string enum)
and `person_decay_months = 12` (int).

| Flag                                                  | Type | Default     | What it gates |
|-------------------------------------------------------|------|-------------|---------------|
| `person_correlation_enabled`                          | bool | **False**   | L1 master |
| `person_centrality_enabled`                           | bool | **False**   | L2 |
| `person_centrality_formula`                           | str  | `frequency` | Which formula L2 uses |
| `person_decay_months`                                 | int  | 12          | L1 inactivity decay |
| `person_suggestions_enabled`                          | bool | **False**   | L3 master |
| `person_suggestions_dormancy_enabled`                 | bool | **False**   | L3 sub |
| `person_suggestions_responsiveness_enabled`           | bool | **False**   | L3 sub |
| `person_correlation_social_graph_enabled`             | bool | **False**   | L4 master (typed phrase) |
| `graph_shortest_path_enabled`                         | bool | **False**   | L4.1 |
| `graph_communities_enabled`                           | bool | **False**   | L4.2 |
| `graph_bridges_enabled`                               | bool | **False**   | L4.3 |
| `graph_suggestions_enabled`                           | bool | **False**   | L4.4 master (typed phrase) |
| `graph_suggestions_cluster_dormancy_enabled`          | bool | **False**   | L4.4 sub |
| `graph_suggestions_bridge_maintenance_enabled`        | bool | **False**   | L4.4 sub |
| `graph_suggestions_weak_tie_enabled`                  | bool | **False**   | L4.4 sub |

Flipping any of these from the React `/cp/settings` page is audited as
`runtime_settings_change`. The L4 and L4.4 toggles enforce a typed-phrase
confirmation (`ENABLE SOCIAL GRAPH` and `ENABLE GRAPH-DRIVEN
SUGGESTIONS` respectively) in `app/api/config_api.py`.

---

<a id="l1"></a>

## 5. Level 1 — Presence

**Module:** `app/companion/person_model.py`
**Idle job:** `person-model` (LIGHT, registered in `app/companion/loop.py`)

Sources:
* Gmail senders (read via existing gmail tool registry)
* Calendar attendees (Google Calendar via existing OAuth client)
* Conversation_store participants

**Not sources:**
* Ticket assignees (those are agent roles, not humans)
* Email body content (no NLP, no entity extraction)
* Attachment metadata

For each person, the model tracks:
* `person_id` — canonical email (lowercase, header-stripped)
* `display_name` — best-effort from RFC 5322 display name
* `first_seen` / `last_seen` ISO timestamps
* `occurrences_per_modality` — `{"emails": int, "calendar": int, "conversations": int}`
* `cooccurring_topics` — read-only join against `interest_profile` (no writes back)

The compiled profile lives in `workspace/companion/person_profile.json`;
every sighting is appended to `person_history.jsonl`. Decay is
profile-level: a person not seen in `person_decay_months` is dropped from
the snapshot. Their entries in `person_history.jsonl` are kept (audit
trail).

---

<a id="l2"></a>

## 6. Level 2 — Centrality

**Module:** `app/companion/person_centrality.py`
**Idle job:** runs as part of `person-model`

Three formulas, **operator-picked**, never learned:

| Formula            | Description                                                |
|--------------------|------------------------------------------------------------|
| `frequency`        | `total_occurrences / max_total_in_pool` (normalized)        |
| `recency_weighted` | exp-decay over time (30d half-life), then normalized        |
| `cross_modal`      | `min(modality_count/4, 1.0) × min(log10(total+1)/log10(20+1), 1.0)` |

Two critical Goodhart guards:

1. **The React surface lists by `last_seen`, never by score.** Scores
   are displayed but not used as a sort key. Sorting by score is the
   first step toward "optimize against centrality."
2. **The arbiter caps centrality's salience contribution at 0.15.** Even
   if Maria has a perfect 1.0, a person-related notification only gets
   +0.15 of salience push.

---

<a id="l3"></a>

## 7. Level 3 — Suggestions

**Module:** `app/companion/person_suggestions.py`
**Surfaces:** daily briefing "💬 Suggestions" section, `/person` Signal
command, React Suggestions sub-tab.

Two categories, each with its own switch:

* **Dormancy.** Person had high recency_weighted score, then stopped
  appearing for ≥30 days. Surfaces as: *"You haven't heard from {name}
  in {N} days — want to reach out?"*
* **Responsiveness.** Person sent you 3+ messages in the last 14 days
  with no reply. Surfaces as: *"{name} has reached out {N} times — anything
  pending?"*

Hard rules:

1. **Always phrased as questions.** Never imperatives.
2. **≤3 total suggestions per briefing.** Shared cap with L4.4 — graph
   nudges and person nudges fight for the same 3 slots.
3. **Dedupe by `person_id`** — one nudge per person per briefing, ever.
4. **Per-person opt-out.** `/person mute-suggestions <email>` silences
   nudges for that person without removing them from the profile.
5. **Every fired nudge is logged** to `person_suggestions_emitted.jsonl`
   so you can audit what the system has been pushing.

---

<a id="l4"></a>

## 8. Level 4 — Social graph

**Module:** `app/companion/social_graph.py`
**Idle job:** `social-graph` (LIGHT)
**Gate:** Operator must type `ENABLE SOCIAL GRAPH` in the React card.

The social graph is the **most invasive** subsystem in the entire
codebase that isn't Tier-3-immutable. It builds a co-appearance map:
for each pair `(A, B)` that appears together in a thread, calendar
event, or conversation, the graph tracks an edge weight.

* **Decay:** 3-month half-life. An edge that doesn't see any new
  co-appearance halves every 90 days. Edges below 0.1 are dropped.
* **Per-pair mute:** `/person mute-pair <a> <b>` zeros out and prevents
  resurrection.
* **Per-person path opt-out:** `/person opt-out-of-paths <email>`. Used
  by L4.1 — see below.
* **Forget-graph:** `/person forget-graph` deletes the entire
  `social_graph.*` family in one shot. There is no recovery.

The L4 master switch alone gates *building* the graph. Each sub-feature
(L4.1, L4.2, L4.3, L4.4) is independently switchable.

---

<a id="l41"></a>

## 9. L4.1 — Shortest-path queries

**Module:** `app/companion/graph_features/shortest_path.py`
**Switch:** `graph_shortest_path_enabled`

Operator-initiated only. `/person path-to <email>` or the React
SocialGraphCard "Path query" form. Returns a list of intermediate
people forming the shortest path.

Three guarantees:
1. **Opt-outs are honored at intermediate position.** A person who
   opted out of paths can be a source or target, but **never an
   intermediate hop** in a path the operator did not start from them.
2. **Every query is logged.** `social_graph_query_log.jsonl`
   records who was queried, by whom, and when. Available at the
   React graph query log surface.
3. **Hop cap (default 6).** Prevents pathological searches.

---

<a id="l42"></a>

## 10. L4.2 — Community detection

**Module:** `app/companion/graph_features/communities.py`
**Switch:** `graph_communities_enabled`

Pure-Python label propagation (no networkx dependency). Operator can
**dissolve** any cluster from the React surface — dissolved clusters
are filtered from all read surfaces (briefing, React, Signal
output).

* `_label_propagation(adj, seed=…)` — deterministic with seed
* `_compute_modularity(adj, labels)` — Newman's modularity
* Cluster IDs are random 8-char hex (no persona-leaking name)

Goodhart guard: clusters are surfaced for *operator awareness*, never
as a route into prescriptive nudges except via L4.4-cluster-dormancy
(which is itself a separate switch behind the second typed-phrase
gate).

---

<a id="l43"></a>

## 11. L4.3 — Bridges and cut-vertices

**Module:** `app/companion/graph_features/bridges.py`
**Switch:** `graph_bridges_enabled`

Tarjan's bridge-finding algorithm, **iterative implementation** (no
recursion depth concerns even on multi-thousand-node graphs).

Surfaces:
* The structural view in React lists bridges + cut-vertices, each with
  the explanatory caveat: *"A bridge is the only path between two
  groups. A cut-vertex's removal would partition the graph.
  Structural roles are not virtues — the algorithm sees structure,
  not friendship."*
* The arbiter consults `is_bridge_or_cut(person_id)` and applies a
  **capped** boost of ≤0.10 to messages from bridge/cut people.

---

<a id="l44"></a>

## 12. L4.4 — Graph-driven suggestions

**Module:** `app/companion/graph_features/graph_suggestions.py`
**Switch:** `graph_suggestions_enabled` (master, typed phrase
`ENABLE GRAPH-DRIVEN SUGGESTIONS`) + three per-category switches.

The three categories:

| Category               | Trigger                                                                | Example phrasing |
|------------------------|------------------------------------------------------------------------|------------------|
| `cluster_dormancy`     | Entire cluster of ≥3 people inactive ≥45 days                          | *"Your {cluster_label} cluster hasn't been active in {N} days — anything brewing?"* |
| `bridge_maintenance`   | Bridge person hasn't appeared in ≥30 days                              | *"You haven't heard from {name}, who connects two parts of your life — check in?"* |
| `weak_tie_dormant`     | Low-weight edge (< 0.3) that's been inactive ≥60 days but was once strong | *"Reconnect with {name}? You used to be in regular contact."* |

All three:
* Phrased as questions
* Subject to the **shared L3+L4.4 cap of 3 nudges per briefing**
* Logged to `person_suggestions_emitted.jsonl` like all L3 nudges
* Per-person opt-out (same `mute_suggestions_for` as L3)

---

<a id="mute"></a>

## 13. Mute / forget / opt-out semantics

The system distinguishes four operator actions on a person:

| Action               | Effect                                                                            | Reversible? |
|----------------------|-----------------------------------------------------------------------------------|-------------|
| `mute`               | Hidden from ALL surfaces but profile retained                                     | Yes (`unmute`) |
| `mute-suggestions`   | Person appears in profile but produces no L3 / L4.4 nudges                        | Yes |
| `opt-out-of-paths`   | Cannot be an intermediate hop in L4.1 paths (still appears as source/target)      | Yes |
| `forget`             | Profile entry deleted; mute also cleared; history entries kept for audit         | No (re-tracked on next sighting) |
| `forget-graph`       | Entire `social_graph.*` family wiped — graph + communities + structural + query log | No (rebuild requires re-enabling L4) |
| `forget-all`         | Wipe of L1 + all mutes; equivalent to disabling L1 fresh                          | No |

These map onto Signal slash commands under `/person` and to action
buttons on the React PeopleCard / SocialGraphCard.

---

<a id="goodhart"></a>

## 14. Goodhart-of-the-indicator safeguards

In order of severity:

1. **List sorts.** Person lists and graph edge lists sort by
   `last_seen`, never by `score` or `weight`. Sorting by score is the
   gateway to "optimize against the score."
2. **Capped salience contributions.** The arbiter's
   `_person_centrality_boost` caps at 0.15, `_bridge_boost` at 0.10.
   Even a perfect-1.0 score / strongest bridge moves the needle by
   less than one quarter of a hard-rule notification.
3. **Suggestion rate limit.** ≤3 per briefing, shared between L3 and
   L4.4. Per-person dedupe within a briefing. Per-person
   `mute-suggestions` is independently honored.
4. **All suggestions phrased as questions.** No imperatives. Hand-coded
   templates, not LLM-generated.
5. **Operator-visible audit log.** Every fired nudge appears in
   `person_suggestions_emitted.jsonl`; every L4.1 query appears in
   `social_graph_query_log.jsonl`. The React surfaces these.
6. **Operator-dissolvable clusters.** Any L4.2 cluster can be hidden
   permanently with one click.
7. **No body parsing.** The L1 source for emails is the `From:` header
   only. No NLP, no entity extraction, no LLM over message bodies.

---

<a id="dr"></a>

## 15. DR exclusion

`app/dr/export_kbs.py` enforces a `_PATH_DENY_FRAGMENTS` list. A single
fragment, `"social_graph"`, catches **all five** derived files:

```
workspace/companion/social_graph.json
workspace/companion/social_graph_pair_mutes.json
workspace/companion/social_graph_communities.json
workspace/companion/social_graph_structural.json
workspace/companion/social_graph_query_log.jsonl
```

Plus `social_graph_path_opt_outs.json` and
`social_graph_dissolved_clusters.json`.

`person_profile.json` is **NOT** excluded — it's analogous to the
existing `interest_profile.json` and ships with regular backups. If you
want it excluded too, add `"person_profile"` to the deny-fragments.

---

<a id="integration"></a>

## 16. Integration with existing systems

| System                         | Hook | What it does |
|--------------------------------|------|---------------|
| **`app/notify/arbiter.py`**     | `_person_centrality_boost` + `_bridge_boost` | Salience contributions (capped) |
| **`app/life_companion/daily_briefing.py`** | `_gather_people_insights` + `_gather_person_suggestions` | Two new briefing sections gated by master switches |
| **`app/companion/loop.py`**     | `get_idle_jobs()` extended | 3 new LIGHT idle jobs: `person-model`, `social-graph`, `graph-features` |
| **`app/companion/interest_model.py`** | Read-only consumer | Person profile reads topic co-occurrence from the interest model |
| **`app/control_plane/dashboard_api.py`** | New endpoints | 15 new endpoints under `/companion/people/*` and `/companion/social_graph/*` |
| **`app/agents/commander/commands.py`** | `/person` slash command | Full operator surface in Signal |
| **`app/dr/export_kbs.py`**      | `_PATH_DENY_FRAGMENTS` | Single-fragment exclusion (see §15) |

The integration is **observational and additive** — no existing
behaviors change when all switches are OFF.

---

<a id="surfaces"></a>

## 17. Operator surfaces — React + Signal

### React `/cp/settings`

The `PersonCorrelationCard` (mounted in `SettingsPage.tsx`) implements
progressive disclosure. L2/L3/L4 are visible only when L1 is on. L4.1
through L4.4 are visible only when L4 is on. L4 and L4.4 each show a
text input + Enable button — the button is disabled until the input
exactly matches the typed phrase. Warning banners turn red at L4 and
red-with-extra-emojis at L4.4.

### React `/cp/companion`

Three new sub-tabs:
* **People** — `PeopleCard`: tracked-people list with per-row mute /
  mute-suggestions / opt-out / forget buttons + top-level "Forget all"
* **Graph** — `SocialGraphCard`: edge list / communities view (with
  per-cluster dissolve) / structural view (bridges + cut-vertices) /
  path query form / DangerZone (forget-graph)
* **Suggestions** — `PersonSuggestionsCard`: list of recent nudges

### Signal `/person` command

All operations available without leaving Signal:

```
/person mute <email>
/person unmute <email>
/person mute-suggestions <email>
/person unmute-suggestions <email>
/person opt-out-of-paths <email>
/person forget <email>
/person forget-all
/person forget-graph
/person path-to <email>          # uses your own canonical email as source
```

---

<a id="not"></a>

## 18. What this layer deliberately does NOT do

* **No alias resolution.** `maria@old` and `maria@new` are separate
  people until the operator manually does `forget` on one and lets the
  other accrete.
* **No off-host data.** Nothing reaches a vendor API. The L4.1 BFS,
  the L4.2 label propagation, the L4.3 Tarjan's algorithm — all run in
  pure Python in-process.
* **No score-sorted UI lists.** Even when scores are visible.
* **No imperative suggestions.** Every nudge is a question.
* **No automatic L4 enablement.** Two operator actions required: toggle
  L1, then type the L4 phrase. Two more for L4.4.
* **No LLM reading of message bodies for person tracking.** Only the
  `From:` header for email; only the structured attendee list for
  calendar; only the participant list for conversations.
* **No memory write from this layer into Mem0 / Neo4j.** All
  person-correlation state is file-backed and operator-deletable in
  one shot.

---

## Appendix — Module map

```
app/companion/
├── person_model.py             # L1: presence, profile, mute, forget
├── person_centrality.py        # L2: 3 formulas, salience helper
├── person_suggestions.py       # L3 + L4.4 unified emission
├── social_graph.py             # L4: edge model, decay, query log
└── graph_features/
    ├── shortest_path.py        # L4.1
    ├── communities.py          # L4.2
    ├── bridges.py              # L4.3
    └── graph_suggestions.py    # L4.4 generators

app/control_plane/dashboard_api.py
  + 15 new endpoints under /companion/people/*, /companion/social_graph/*

app/agents/commander/commands.py
  + /person slash command

dashboard-react/src/components/
├── PersonCorrelationCard.tsx   # /cp/settings card
├── PeopleCard.tsx              # /cp/companion People sub-tab
├── SocialGraphCard.tsx         # /cp/companion Graph sub-tab
└── PersonSuggestionsCard.tsx   # /cp/companion Suggestions sub-tab
```

Tests: `tests/test_q4_2_person_correlation.py` — 29 tests across L1–L4
covering presence, centrality formulas, graph algorithms, gating,
typed-phrase enforcement, DR exclusion, arbiter caps, briefing
integration, and idle-job registration.
