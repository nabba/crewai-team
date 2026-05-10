# React Dashboard — operator surface reference

`dashboard-react/` (Vite + React 19 + TanStack Query v5 + Tailwind v4)
served by `dashboard/server.mjs` on `:3100` from
`dashboard/serve-root/cp/`. The Vite dev server runs on `:3101` for
local work. Both proxy `/api/`, `/config/`, and the seven
root-mounted FastAPI prefixes (`/kb`, `/fiction`, `/philosophy`,
`/episteme`, `/epistemic`, `/experiential`, `/aesthetics`,
`/tensions`, `/affect`) through to the gateway on `:8765`.

This doc is a navigation reference — what each tab does, what
endpoint backs it, and where the data lives. For deep program
history see `PROGRAM.md` §26.

## Build

```bash
cd dashboard-react
npm run build       # tsc -b && vite build && postbuild syncs serve-root
```

The postbuild step (`rm -rf ../dashboard/serve-root/cp && cp -R
../dashboard/build/. ../dashboard/serve-root/cp/`) is what makes the
:3100 server pick up new code. If `tsc -b` fails the postbuild never
runs and :3100 silently serves stale assets — keep `tsc -b` clean.

## Tabs

| Tab | Path | Backend endpoints | Notes |
|---|---|---|---|
| Dashboard | `/cp/` | `/api/cp/projects`, `/api/cp/tickets/board`, `/api/cp/budgets`, `/api/cp/audit?limit=10`, `/api/cp/governance/pending`, `/api/cp/consciousness` | Top-level overview |
| Chat | `/cp/chat` | `/api/cp/chat/messages`, `POST /api/cp/chat/send`, `/api/cp/signal-commands` | Signal mirror — `Commander.handle()` dispatch path. Markdown rendering (GFM + math + code highlighting). Filterable command sidebar with 85 commands × 13 categories. See PROGRAM.md §35.1 |
| Monitor | `/cp/monitor` | `/api/cp/system-status` | Probes containers + messaging + internal subsystems + external services every 10 s. Credit-exhaustion errors surface a "Top up →" link to the matching provider billing page. PROGRAM.md §35.2 |
| Tickets | `/cp/tickets` | `/api/cp/tickets/board`, `PUT /api/cp/tickets/{id}` | Kanban with @dnd-kit drag-drop; drag-to-todo re-queues through Commander |
| Tasks | `/cp/tasks` | `/api/cp/tasks` | Crew Activity — reads `control_plane.crew_tasks` (Postgres). Was Firestore until §26.2 |
| Budgets | `/cp/budgets` | `/api/cp/budgets`, `POST /api/cp/budgets/override`, `POST /api/cp/budgets/pause` | Per-agent cards. Pause/Resume button toggles `is_paused` in `control_plane.budgets` |
| Audit | `/cp/audit` | `/api/cp/audit?limit=300` | "Only rows with cost" toggle filters to cost-bearing entries (typically `ticket.completed`) |
| Governance | `/cp/governance` | `/api/cp/governance/pending`, approve/reject endpoints | |
| Changes | `/cp/changes` | `/api/cp/changes` | Operator gate for `request_restricted_write` agent-callable tool. See `docs/CHANGE_REQUESTS.md` |
| Org Chart | `/cp/org-chart` | `/api/cp/org-chart` | Vertical indented layout (mobile-friendly); merges Postgres `org_chart` rows with the canonical `CREW_REGISTRY` |
| Costs | `/cp/costs` | `/api/cp/costs/daily`, `/api/cp/costs/by-crew`, `/api/cp/costs/by-agent`, `/api/cp/costs/by-internal-agent`, `/api/cp/tokens` | Three breakdown panels (§26.1). Cost-by-Crew = `request_costs.crew_name` from SQLite. Cost-by-Agent = mapped via `_CREW_TO_AGENT`. Cost-by-Internal-Agent = `control_plane.budgets` filtered to internal roles + idle-scheduler job names |
| Workspaces | `/cp/workspaces` | `/api/workspaces`, `/api/workspaces/{id}/items`, `/api/workspaces/meta` | Consciousness-gate creation atomically creates a CP project too |
| Affect | `/cp/affect` | `/affect/now`, `/affect/feed`, `/affect/welfare-config`, `/affect/welfare-audit`, `/affect/attachments`, `/affect/reference-panel`, `/affect/calibration` | Viability, V/A/C, welfare envelope, attachment OtherModels (per-user + per-peer-agent), L9 reference panel |
| Epistemic | `/cp/epistemic` | `/epistemic/now`, `/epistemic/feed`, `/epistemic/pushback`, `/epistemic/peer-review`, `/epistemic/calibration`, `/epistemic/overrides` | Claim ledger / pushback / overrides. **Distinct from `/episteme/`** (the research RAG KB) — the proxy whitelist contains both |
| Evolution | `/cp/evolution` | `/api/cp/evolution/summary`, `/api/cp/evolution/results`, `/api/cp/evolution/engine`, `/api/cp/evolution/variants` | Overview + Experiment History + Engine Analysis + Genealogy. History list is newest-first; chart deltas + score-trend run left→right oldest→newest |
| Ops | `/cp/ops` | `/api/cp/errors`, `/api/cp/anomalies`, `/api/cp/deploys`, `/api/cp/error_audit` | Lists are newest-first |
| LLMs | `/cp/llms` | `/api/cp/llms/*` (catalog, roles, discovery, promotions, pins, judges) + `/config/llm_mode` | Runtime mode pills (free / budget / balanced / quality / insane / anthropic) |
| Knowledge | `/cp/knowledge` | `/kb/*`, `/fiction/*`, `/philosophy/*`, `/episteme/*`, `/experiential/*`, `/aesthetics/*`, `/tensions/*` | Per-KB stats + upload |
| Notes | `/cp/notes` | `/api/cp/notes/*` | Obsidian-style viewer (wikilinks, callouts, math, mermaid, graph) |
| Wiki | `/cp/wiki` | `/api/cp/notes/file?root=wiki` | Read-only views over `wiki/` |
| Brainstorm | `/cp/brainstorm` | `/api/cp/brainstorm/*` | 7 techniques (SCAMPER, Six Hats, How-Might-We, Reverse, Crazy-8s, Rapid Ideation, Starbursting). See `docs/BRAINSTORM.md` |
| Forge | `/cp/forge` + `/cp/forge/settings` + `/cp/forge/compositions` + `/cp/forge/tools/{id}` | Tool Registry + Forge subsystem |
| Files | `/cp/files` | Personal-agent file API |
| Sessions | `/cp/coding-sessions` | `/api/cp/coding-sessions` | Read-only — actionable surface is `/cp/changes`. See `docs/CODING_SESSIONS.md` |
| Settings | `/cp/settings` | `/config/runtime_settings`, `/config/background_tasks`, `/config/governance_ratchet/*`, `/config/runbook_settings`, `/config/web_push/*` | Personal-agent toggles + ops kill switches (§26.7) |

## Conventions

* **Lists** run newest-first (top row = most recent). Audit feed,
  Tasks, Ops/Errors, Ops/Anomalies, Ops/Deploys, Evolution
  History, Evolution Variants — all newest-first.
* **Charts** run left→right = oldest→newest. CostCharts daily line
  reverses the backend's DESC payload before plotting; Evolution
  score-trend and delta-bars use ASC backend data unchanged.
* **Project selector** filters every project-aware endpoint via
  `?project_id=...`. "All projects" mode passes no filter and
  pulls aggregate. Pre-tagging-migration rows have `project_id IS
  NULL` and only show under "All projects".
* **Cost attribution flow**: every LLM call → `record_tokens()`
  (`app/llm_benchmarks.py`) → SQLite `token_usage` + `request_costs`
  (project + crew tagged via ContextVars) → `reconcile_actual_spend`
  (`app/control_plane/budgets.py`) → Postgres `control_plane.budgets`
  (project + agent_role tagged). Three Cost panels read from these
  two stores; Daily-Costs reads `control_plane.audit_log` filtered
  on `cost_usd IS NOT NULL` (only `ticket.completed` rows populate
  it).
* **No `unknown` agent bucket.** Background work runs inside
  `agent_scope(job_name)` set by
  `idle_scheduler._run_single_job`, so each idle job gets its own
  budgets row (`llm-discovery`, `fiction-ingest`, …). Historical
  pre-§26.1 unknowns were merged into a single `idle_scheduler`
  row by migration `022`.

## Auth

There are two independent auth layers — they don't replace each
other, they stack:

**1. Gateway secret (backend ↔ proxy)**

All `/api/cp/*` and `/config/*` mutating routes require
`Authorization: Bearer <gateway-secret>` when
`GATEWAY_AUTH_REQUIRED=1`. The dashboard server (`server.mjs`)
injects the header automatically using `GATEWAY_SECRET` from env or
the sibling `.env`. The Vite dev server does the same. Without it,
POSTs return 401 from the gateway.

**2. Dashboard auth (browser ↔ proxy)** — only required when the
proxy is exposed publicly via Tailscale Funnel or similar.

Set `DASHBOARD_USER` + `DASHBOARD_PASS` in `.env` to enable. The
proxy then accepts EITHER:

  * `Authorization: Basic ...` matching `DASHBOARD_USER:DASHBOARD_PASS`, OR
  * cookie `dashboard_auth=<HMAC>` set by visiting `/cp/login?token=<DASHBOARD_PASS>` once.

Loopback (`localhost` / `127.0.0.1` / `::1`) bypasses both and stays
unauthenticated, so laptop-localhost dev keeps working.

iOS PWA standalone mode does NOT always inherit Basic-Auth
keychain entries — use the cookie path:

```
https://<funnel-host>/cp/login?token=<DASHBOARD_PASS>
```

Server returns `302 /cp/` + a 1-year HttpOnly+Secure cookie. The
home-screen icon's standalone scope shares cookies with regular
Safari tabs, so subsequent launches authenticate transparently.

Rotating `DASHBOARD_PASS` in `.env` invalidates every existing
cookie automatically (cookie value is `HMAC(pass, "dashboard-auth-v1")`).

**Public HTTPS (Tailscale Funnel)**

`tailscale funnel --bg --https=443 http://localhost:3100` exposes
the dashboard on `https://<machine>.<tailnet>.ts.net/cp/` with a
real Let's-Encrypt cert. Prerequisite admin-console toggles:
HTTPS Certificates (DNS page) + Funnel ACL grant.

**Full step-by-step walkthrough** (Tailscale toggles → CLI →
dashboard auth env vars → iPhone home-screen install → Web Push
permission grant → troubleshooting): see
[`docs/PWA_SETUP.md`](./PWA_SETUP.md). PROGRAM.md §35.3 covers
the design rationale; PWA_SETUP.md is the operator recipe.

## Common gotchas

* **`/affect/` and `/epistemic/` 404s** — proxy whitelist in
  `dashboard/server.mjs` was missing both at one point. They are
  in now; if you add another root-mounted FastAPI router, add it
  there too.
* **`:3100` serves stale bundle** — usually means `tsc -b` errored
  and postbuild never ran. Run `npm run build` manually and read
  the output, or as a one-off bypass: `npx vite build && rm -rf
  ../dashboard/serve-root/cp && cp -R ../dashboard/build/.
  ../dashboard/serve-root/cp/`.
* **"Failed to load /config/<x>"** — the gateway wasn't rebuilt
  after a backend code change. `docker compose build gateway && up
  -d gateway`.

## Where the dashboard lives in source

```
dashboard-react/
  src/
    App.tsx                     # routes
    api/
      endpoints.ts              # central path registry
      queries.ts                # TanStack Query hooks (typed)
      client.ts                 # fetch wrapper + auth
    components/                 # one file per top-level tab + sub-components
      Dashboard.tsx
      ChatPage.tsx              # /cp/chat — Signal mirror + command catalogue
      MonitorPage.tsx           # /cp/monitor — system status + credit alerts
      KanbanBoard.tsx
      TasksPage.tsx
      BudgetDashboard.tsx
      AuditFeed.tsx
      CostCharts.tsx
      ConsciousnessIndicators.tsx
      OpsPage.tsx
      LlmsPage.tsx
      EvolutionMonitor.tsx
      WorkspacesPage.tsx
      OrgChart.tsx
      NotesPage.tsx
      WikiPage.tsx
      BrainstormPage.tsx
      SettingsPage.tsx
      ChangesPage.tsx
      affect/                   # AffectPage + sub-components
      epistemic/                # EpistemicPage + sub-components
    crews.ts                    # CREW_REGISTRY: 11 user + 4 internal
    context/                    # ProjectContext (active-project switcher)
dashboard/
  server.mjs                    # Node static server + reverse proxy
  build/                        # Vite output
  serve-root/cp/                # postbuild copy that :3100 serves
```
