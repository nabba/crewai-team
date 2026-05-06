# Brainstorm Subsystem

Interactive idea-generation sessions backed by a library of structured
techniques (SCAMPER, Six Hats, How-Might-We, Reverse, Crazy-8s, Rapid
Ideation, Starbursting). Each session can run **solo** (just the human)
or in **team mode**, where 1–4 high-creativity agents from the existing
creative-crew roster (researcher / writer / coder / critic) brainstorm
alongside the human. Sessions are surface-agnostic: the same store + the
same facilitator drives Signal, the CLI, and the React dashboard.

> **Status**: shipped May 2026. 120 tests across 7 test files
> (`test_brainstorm_*.py`). React UI verified in the browser preview.

---

## 1. Why this subsystem exists

Operator request: "I want the system to be able to conduct different
brainstorming and idea-creation techniques with me through Q/A sessions
and input by me, and then write a final report. I want also a possibility
where I can add 3–5 agents with high creativity enabled and run the
brainstorm as a joint effort of all of us."

The Creative MAS pipeline (`app/crews/creative_crew.py`) was already
running structured Diverge/Discuss/Converge cycles, but it was
non-interactive — the human gives a single task, the crew runs to
completion, the report comes back. This subsystem keeps the human in the
loop step-by-step, layered on top of (not duplicating) the creative
pipeline's high-creativity agent construction.

---

## 2. Shape

```
                      ┌─────────────────────┐
                      │   Surfaces          │
                      │   ─────────         │
                      │   • Signal          │
                      │   • Python CLI      │
                      │   • React (web)     │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │   Facilitator       │     surface-agnostic
                      │   (StepDelivery)    │     state-machine driver
                      └──────────┬──────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
   ┌──────────────┐    ┌──────────────────┐   ┌──────────────┐
   │  Techniques  │    │   Multi-agent    │   │    Store     │
   │  (state      │    │   gather_seed +  │   │  (JSON       │
   │   machines)  │    │   gather_react   │   │   files)     │
   └──────────────┘    └──────────────────┘   └──────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │  Writer agent      │  → workspace/output/
                       │  (final report)    │     brainstorm/<id>.md
                       └────────────────────┘
```

All three surfaces talk to the same `app.brainstorm.facilitator` API and
share the same `workspace/brainstorm/` JSON store, so sessions started in
one surface can be resumed in another (e.g. Signal → React) when the
sender ID matches. The web UI defaults its sender to
`signal_owner_number`, so by default it shares state with the operator's
Signal sessions.

---

## 3. Module layout

```
app/brainstorm/
├── techniques/          # Technique state machines (one file per technique)
│   ├── base.py          # Technique ABC + LinearTechnique + TechniqueState
│   ├── scamper.py
│   ├── six_hats.py
│   ├── how_might_we.py
│   ├── reverse.py
│   ├── crazy_8s.py
│   ├── rapid_ideation.py
│   └── starbursting.py
├── session.py           # BrainstormSession dataclass + serialization
├── store.py             # JSON-file persistence under workspace/brainstorm/
├── facilitator.py       # Drives session: start/respond/skip/pause/resume/finish
├── multi_agent.py       # Parallel seed + react gathering (team mode)
├── report.py            # Writer-agent handoff + deterministic fallback
├── signal_handler.py    # /brainstorm slash commands + active-session routing
├── api.py               # FastAPI router under /api/cp/brainstorm/
├── cli.py               # python -m app.brainstorm
└── __main__.py          # CLI entry point

dashboard-react/src/
├── api/brainstorm.ts            # React Query hooks
├── types/brainstorm.ts          # TS types mirroring api.py response shapes
├── components/BrainstormPage.tsx
└── components/brainstorm/
    ├── StartPanel.tsx           # Pick technique + topic + with-agents
    ├── SessionView.tsx          # Active session: prompt + transcript + input
    ├── SessionsList.tsx         # Sidebar of past sessions (resume / delete)
    ├── ReportView.tsx           # Final report markdown render
    ├── AgentRoundBlock.tsx      # One round (seed or react)
    └── AgentResponseCard.tsx    # One agent's contribution
```

---

## 4. Techniques

Each technique is a state machine: a fixed list of steps, each with a
prompt template that interpolates `{topic}`. The facilitator walks them
sequentially. The seven currently implemented:

| Name             | Steps | Description                                          |
|------------------|-------|------------------------------------------------------|
| `scamper`        | 7     | S-C-A-M-P-E-R transformation lenses                  |
| `six_hats`       | 7     | de Bono's six thinking hats (white/red/black/yellow/green) bracketed by blue (open / close) |
| `how_might_we`   | 10    | Reframe a problem as opportunity questions, then expand |
| `reverse`        | 6     | Ask how to *cause* the problem, then invert each failure |
| `crazy_8s`       | 10    | 8 ideas in 8 quick rounds (Design Sprint)            |
| `rapid_ideation` | 7     | Three quantity-bursts (obvious / constraint-flipped / different lens), then cluster + select |
| `starbursting`   | 8     | Generate questions (not answers) along Who/What/When/Where/Why/How |

Adding a new technique is a single file plus a registry entry in
`techniques/__init__.py`. The base class `LinearTechnique` handles all
state-machine plumbing; concrete techniques only declare a list of `Step`
objects.

---

## 5. Solo mode flow

```
start("scamper", "improve onboarding")
  → state_index = 0
  → first prompt: "S — Substitute…"
  ▼
respond("answer to S step")
  → record response, advance to step 1
  → next prompt: "C — Combine…"
  ▼
…
  ▼
respond("answer to R step")  ← last step
  → record response
  → next_prompt = None (state machine complete)
  → emit "All steps complete. Reply /brainstorm finish."
  ▼
finish()
  → call Writer agent (or fallback markdown if BRAINSTORM_DISABLE_WRITER=1)
  → write workspace/output/brainstorm/<id>.md
  → status = "complete"
```

`pause` / `resume` save state to disk; `skip` records `(skipped)` and
advances; `cancel` marks status `cancelled` and clears the active pointer.

---

## 6. Team mode flow (the joint-effort variant)

When the user picks `with_agents=N` (1–4), each step runs **two** extra
parallel rounds via `app.brainstorm.multi_agent`:

1. **Seed** — agents propose initial ideas for the upcoming step.
   Anti-conformity prompt; no peer awareness yet (each agent answers in
   parallel without seeing the others). Output is shown to the human
   *before* they type.
2. **React** — after the human types their answer, the same agents see
   the human's answer + each other's seeds, then react / extend / disagree.

```
              start(with_agents=3)
                      │
                      ▼
              gather_seed(step 0)         ← parallel: 3 agents
                      │
                      ▼
   show user: prompt + 3 seed cards
                      │
            user types answer
                      │
                      ▼
              gather_react(step 0)        ← parallel: 3 agents
                      │ (sees user answer + peer seeds)
                      ▼
              gather_seed(step 1)         ← parallel: 3 agents
                      │
                      ▼
   show user: react cards + next prompt + new seed cards
                      │
                     ...
```

### High-creativity agent construction

`multi_agent._build_creative_agent(role)` mirrors
`creative_crew._make_agent`:

| Role        | LLM tier  | Reasoning method        |
|-------------|-----------|-------------------------|
| researcher  | `local`   | step_back               |
| writer      | `mid`     | analogical_blending     |
| coder       | `budget`  | compositional_cot       |
| critic      | `premium` | contrastive             |

A `ThreadPoolExecutor` runs the roster in parallel (workers =
`min(4, len(roster))`). Per-agent failures are captured into
`AgentResponse.error` and the round continues with the rest. A whole-round
crash falls back to solo for that step.

### Cost & budget

For a 7-step technique (e.g. SCAMPER) with 4 agents, team mode dispatches
≈ 56 LLM calls per session (`7 × 2 phases × 4 agents`). The soft cap
defaults to `$0.50` per session, overridable via
`BRAINSTORM_TEAM_BUDGET_USD`. When the cap is hit, subsequent rounds
return empty and the session continues solo-style for the remainder.

---

## 7. The three surfaces

### Signal (`/brainstorm` slash command)

| Command                                          | Action                                  |
|--------------------------------------------------|-----------------------------------------|
| `/brainstorm`                                    | Show technique menu                     |
| `/brainstorm <tech> <topic>`                     | Start solo session                      |
| `/brainstorm <tech> with N agents <topic>`       | Start team session (N up to 4)          |
| `/brainstorm <tech> with agents <topic>`         | Same as above; defaults N to 4          |
| `/brainstorm status`                             | Current session progress                |
| `/brainstorm skip`                               | Skip current step                       |
| `/brainstorm pause`                              | Save and exit                           |
| `/brainstorm resume [session_id]`                | Continue paused session                 |
| `/brainstorm finish`                             | Generate the final report               |
| `/brainstorm cancel`                             | Discard active session                  |
| `/brainstorm list`                               | Past sessions for this user             |
| `/brainstorm help`                               | Command list                            |

After a session is started, plain (non-slash) Signal messages from the
sender are interpreted as answers to the current step. The hook lives in
`app/agents/commander/commands.py:try_command` and short-circuits before
any other slash-command parsing.

### CLI (`python -m app.brainstorm`)

```
python -m app.brainstorm                          # interactive picker
python -m app.brainstorm --technique scamper --topic "..."
python -m app.brainstorm --with-agents 3          # team mode
python -m app.brainstorm --resume                 # most-recent paused
python -m app.brainstorm --resume <ID>            # specific session
python -m app.brainstorm --list                   # past sessions
python -m app.brainstorm --techniques             # menu
python -m app.brainstorm --sender +15551112222    # share with Signal
```

In-session commands: `skip`, `pause`, `cancel`, `finish`, `status`. Empty
input re-shows the current prompt; Ctrl-D pauses and exits.

### React (`/cp/brainstorm`)

Tab in the dashboard sidebar. Layout:
- Left pane: start panel (technique grid + topic + agent-count buttons)
  OR active-session view (prompt + per-step seed/react cards + reply box).
- Right pane: list of past sessions with status badges; click to inspect
  / resume / delete.
- After `Finish`, the panel switches to a markdown report viewer.

The web sender defaults to `signal_owner_number` so React + Signal share
the session pool by default. Override via `BRAINSTORM_WEB_SENDER` or per
request via `?sender=…` query parameter.

---

## 8. HTTP API (used by the React surface)

Mounted at `/api/cp/brainstorm/`. Inherits the gateway-auth dependency
from `app.control_plane.auth_dep.require_gateway_auth`.

| Method   | Path                                | Purpose                                |
|----------|-------------------------------------|----------------------------------------|
| GET      | `/techniques`                       | Catalog                                |
| GET      | `/sessions?sender=&include_other_senders=` | List sessions                  |
| GET      | `/sessions/active?sender=`          | Sender's currently-active session       |
| GET      | `/sessions/{id}`                    | Full session detail                    |
| POST     | `/sessions`                         | Start a session (`technique`, `topic`, `with_agents`) |
| POST     | `/sessions/{id}/respond`            | Record a user answer                   |
| POST     | `/sessions/{id}/skip`               | Skip current step                      |
| POST     | `/sessions/{id}/pause`              | Pause                                  |
| POST     | `/sessions/{id}/resume`             | Resume                                 |
| POST     | `/sessions/{id}/cancel`             | Cancel                                 |
| POST     | `/sessions/{id}/finish?generate_report=` | Finish + generate report          |
| DELETE   | `/sessions/{id}`                    | Delete a session                       |

All write endpoints validate that `{id}` matches the sender's currently
active session — pause / resume / start route lifecycle changes.

`POST /sessions`, `/respond`, `/skip` and `/finish` are long-running in
team mode (10–60s per round). The endpoint blocks synchronously; the
React surface shows a "Sending…" / "Finishing…" state.

---

## 9. Persistence

Sessions live under `workspace/brainstorm/`:

```
workspace/brainstorm/
├── sessions/
│   └── <session_id>.json       # full session payload
└── active/
    └── <safe_sender>.txt        # pointer to current active session
```

Atomic writes via temp-file + rename, mirroring `app.companion.state`.
`BRAINSTORM_DIR` env var overrides the base directory (used by tests).
Reports are written to `workspace/output/brainstorm/<session_id>.md`
(override with `BRAINSTORM_OUTPUT_DIR`).

---

## 10. Final-report generation

`app.brainstorm.report.generate_report(session)` returns
`(markdown, file_path)`:

1. **Primary path** — wraps the technique summary + raw transcript +
   structured agent rounds into a Writer-agent task. The Writer is
   instantiated via the existing `app.agents.writer.create_writer()`
   factory (CrewAI Agent + Task + Crew kickoff). In team mode the prompt
   asks the Writer to attribute strong contributions by role.
2. **Fallback** — if the Writer-agent path fails (LLM unavailable, env
   missing, etc.) or `BRAINSTORM_DISABLE_WRITER=1` is set, a deterministic
   markdown rendering is produced from the structured summary. The user
   always gets a report.

The report file is saved under `workspace/output/brainstorm/<id>.md`.

---

## 11. Test layout

| File                                              | What it covers                                          |
|---------------------------------------------------|---------------------------------------------------------|
| `tests/test_brainstorm_techniques.py`             | All 7 state machines walk to completion; SCAMPER letter mapping; Six Hats blue-open/blue-close; Crazy-8s 8 idea steps; Starbursting 6 question rays |
| `tests/test_brainstorm_store.py`                  | JSON round-trip; active-pointer; sender isolation; delete clears active |
| `tests/test_brainstorm_facilitator.py`            | Solo lifecycle: start/respond/skip/pause/resume/cancel/finish; empty-input handling; report-generator failure fallback |
| `tests/test_brainstorm_multi_agent.py`            | Roster resolution; parallel seed/react with mocked gatherers; per-agent error capture; persistence of `agent_rounds`; failure isolation |
| `tests/test_brainstorm_signal.py`                 | Slash-command parser (`with N agents` / `with agents` / solo); active-session message routing; status with team participants |
| `tests/test_brainstorm_commands_integration.py`   | `try_command` hook in `commander/commands.py` claims `/brainstorm` slash + active-session messages |
| `tests/test_brainstorm_api.py`                    | FastAPI router via TestClient: lifecycle, sender isolation, team-mode start with mocked gatherers, validation                |

Run all 120 tests:

```sh
.venv/bin/python -m pytest tests/test_brainstorm_*.py
```

---

## 12. Configuration

| Env var                          | Default                              | Purpose                                                    |
|----------------------------------|--------------------------------------|------------------------------------------------------------|
| `BRAINSTORM_DIR`                 | `workspace/brainstorm`               | Session-store base directory                               |
| `BRAINSTORM_OUTPUT_DIR`          | `workspace/output/brainstorm`        | Final-report directory                                     |
| `BRAINSTORM_DISABLE_WRITER`      | `0`                                  | Set `1` to force deterministic report (skip the LLM)       |
| `BRAINSTORM_TEAM_BUDGET_USD`     | `0.50`                               | Soft per-session cap for multi-agent rounds                |
| `BRAINSTORM_WEB_SENDER`          | `<signal_owner_number>` if set, else `web:default` | Default sender for the React surface     |

---

## 13. Composition with the rest of the system

- **Creative MAS** (`app/crews/creative_crew.py`) — the brainstorm team
  mode reuses the same agent factories and `_TIER_BY_ROLE_CREATIVE` /
  `_REASONING_METHOD_BY_ROLE` mappings, but runs them in parallel
  (creative_crew runs sequentially) for interactive UX.
- **Commander routing** — `app/agents/commander/commands.py:try_command`
  hooks `try_handle` from `signal_handler.py` near the top so brainstorm
  claims its messages before any other slash-command parsing.
- **Companion layer** — independent. Brainstorm has its own JSON store
  to keep ephemeral state separate from companion's per-workspace
  ideation cycles.
- **Conversation store** — independent. The brainstorm transcript is
  kept on the session payload, not in `messages`. (Signal's normal
  message log still persists each user turn.)
- **TIER_IMMUTABLE / safety** — `app/brainstorm/` is not in the
  TIER_IMMUTABLE list. Adding new techniques or surface tweaks is a
  normal change. Touching `app/brainstorm/multi_agent.py`'s tier mapping
  is a soft constraint — it's expected to drift from creative_crew over
  time; tests will catch any breakage.

---

## 14. Known follow-ups

- **Streaming agent output**. Each agent currently blocks until
  `crew.kickoff()` returns. Streaming token-by-token would let the React
  UI show progressive seed/react output. Not done because crewai's
  `kickoff()` is synchronous; would need either a SSE wrapper or a
  per-agent task with its own LLM streaming.
- **Personas vs. existing roles**. The roster is fixed to the four
  creative-crew roles. A natural follow-up is "named personas" (Skeptic,
  Optimist, Builder, Outsider) tuned per technique. The bones are there
  — `resolve_roster([…])` already accepts arbitrary role names; the
  factory just needs persona definitions plumbed through.
- **In-session steering**. The user can't currently say "skip the next
  agent's seed" or "give me one more round of react before moving on" —
  every step is one round of seed + one round of react. A `/brainstorm
  more` or "react again" UX pattern would be a small extension.
