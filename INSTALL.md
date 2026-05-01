# BotArmy — Installation Guide

One executable. Three deployment targets. Phased rollout — each phase is
usable on its own without the next one being done.

```
./install.sh                  # interactive local install (Mac/Linux)
./install.sh --target k8s     # deploy to current kubectl context (Phase 2)
./install.sh --target aws     # provision EKS + RDS + deploy (Phase 3, planned)
./install.sh --target gcp     # provision GKE + Cloud SQL + deploy (Phase 3, planned)
```

## What's shipped vs planned

| Phase | Target | Status | What it gives you |
| --- | --- | --- | --- |
| **1** | `--target local` | ✅ shipped | Full stack on a single Mac or Linux machine via Docker Compose |
| **2** | `--target k8s` | 🟡 chart scaffolded | Deploys gateway + memory stack to any existing cluster (no Signal/Ollama/sandbox yet) |
| **3** | `--target aws`, `--target gcp` | 📋 roadmap only | Provisions cluster + managed Postgres, then runs Phase 2 chart |

Details on each phase below.

---

## Phase 1 — Local install (Mac · Linux)

### One-shot install

From the repository root:

```bash
./install.sh
```

What happens:

1. **Detects** OS (macOS / Ubuntu / Debian / Fedora / Arch / Alpine).
2. **Installs prereqs** if missing: Docker, Compose plugin, Python 3.11+,
   openssl. Uses Homebrew on Mac, the native package manager on Linux. On
   Linux, runs Docker's official install script with `sudo`.
3. **Initialises `.env`** from `.env.example`. If `.env` already exists,
   merges in any new keys without touching existing values.
4. **Generates secrets** — `GATEWAY_SECRET`, `MEM0_POSTGRES_PASSWORD`,
   `MEM0_NEO4J_PASSWORD`. Strong random values via `openssl rand`.
5. **Prompts for required API keys** — `ANTHROPIC_API_KEY`,
   `OPENROUTER_API_KEY`, `BRAVE_API_KEY`. Optionally also Apollo, Proxycurl,
   Smithery, Composio.
6. **Probes host RAM** so the LLM registry sizes Ollama models correctly
   (Docker Desktop on Mac otherwise mis-reads RAM via the VM cgroup).
7. **Builds** the gateway image and the sandbox image (`crewai-sandbox:latest`).
8. **Pulls** Postgres, Neo4j, ChromaDB, docker-socket-proxy.
9. **Starts** the stack with `docker compose up -d`.
10. **Verifies** every service is healthy via per-service probes.

End-to-end: 5–15 minutes depending on your bandwidth (the gateway image build
is the long step).

### Non-interactive install (CI / fresh machines)

Pre-stage your secrets in a file, then:

```bash
cat > /tmp/botarmy.env <<EOF
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-v1-...
BRAVE_API_KEY=BSA...
APOLLO_API_KEY=...               # optional
EOF

./install.sh --non-interactive --config /tmp/botarmy.env --yes
```

Or pass keys via environment variables:

```bash
ANTHROPIC_API_KEY=sk-ant-... OPENROUTER_API_KEY=sk-or-v1-... BRAVE_API_KEY=BSA... \
    ./install.sh --non-interactive --yes
```

### All flags

```
--target {local|k8s|aws|gcp}    deployment target (default: local)
--non-interactive               no prompts; require all keys via --config or env
--config FILE                   read API keys from FILE (.env-format)
--skip-prereqs                  don't install Docker/Python — fail if missing
--no-build                      skip docker build (use existing images)
--dry-run                       print what would happen, change nothing
--verify                        only run health checks on a running stack
--uninstall                     stop containers (optionally wipe data)
--yes, -y                       assume "yes" on confirmations
--help, -h                      this help
```

### What's installed where

```
crewai-team/
├── install.sh                   ← single entrypoint
├── .env                         ← generated; chmod 600
├── scripts/
│   ├── install.sh               ← compat shim → ../install.sh
│   └── install/                 ← installer modules
│       ├── lib.sh                  shared helpers (logging, OS detect, env munging)
│       ├── prereqs.sh              Docker + Compose + Python install
│       ├── secrets.sh              .env init, secret generation, key prompts
│       ├── local.sh                Phase 1 flow (compose-based)
│       ├── k8s.sh                  Phase 2 dispatch (helm)
│       ├── verify.sh               health checks
│       └── uninstall.sh            teardown
├── deploy/
│   ├── k8s/                     ← Phase 2 Helm chart
│   └── terraform/               ← Phase 3 cloud roadmap
└── workspace/                   ← persistent data lives here
    ├── mem0_pgdata/                Postgres data (do not delete to keep memory)
    ├── mem0_neo4j/                 Neo4j data
    ├── memory/                     ChromaDB collections
    └── crewai_storage/             CrewAI long-term memory (SQLite)
```

### Re-running

The installer is idempotent. Re-running on an already-installed system:

* Will not regenerate existing secrets.
* Will not overwrite API keys you've already set.
* Will append any new keys that appeared in `.env.example` since last install.
* Will rebuild images and restart the stack.

### Health checking and uninstall

```bash
./install.sh --verify         # re-check health any time
./install.sh --uninstall      # stop containers; optionally wipe data
```

`--uninstall` always asks twice before removing data — this is the
unrecoverable step (your Mem0 + ChromaDB memory).

### What `install.sh` cannot do for you

| Step | Why | Workaround |
| --- | --- | --- |
| Provide your API keys | Anthropic / OpenRouter / Brave have to be created on their portals | Installer prompts, or pass via `--config` / env |
| Register a Signal phone number | Signal sends an SMS code, requires human | Run `signal-cli register` after install |
| Tailscale auth | Browser-based OAuth | `tailscale up` after install |
| Pull Ollama models | Optional + bandwidth-heavy (10–50 GB per model) | `ollama pull qwen3:30b-a3b` after install |

---

## Phase 2 — Kubernetes (current state: scaffolded)

The Helm chart in `deploy/k8s/` deploys the gateway + memory stack to any
existing cluster (kind, k3d, EKS, GKE, AKS, on-prem). It's reachable via:

```bash
./install.sh --target k8s
```

This currently:

* Creates namespace `botarmy` (override with `BOTARMY_NAMESPACE`).
* Creates a `botarmy-env` Secret from your local `.env` (or `--config FILE`).
* Runs `helm upgrade --install` with the chart.

### What works in Phase 2

* Gateway as a Deployment, with HPA-friendly resource limits.
* Postgres / Neo4j / ChromaDB as StatefulSets with PersistentVolumeClaims.
* NetworkPolicy locking down internal services to only the gateway.
* Optional Ingress with cert-manager TLS.
* **Bearer-token auth** on `/api/cp/*` and `/epistemic/*` mutating routes
  (chart sets `gateway.authRequired: "true"` by default in K8s mode). Token
  is `$GATEWAY_SECRET`, sourced from the auto-generated env Secret. Read
  routes, `/health`, and `/metrics` are unaffected. See
  [deploy/k8s/README.md](deploy/k8s/README.md#authenticating-to-the-gateway-phase-b)
  for the auth model + how to make authorized requests.

### What's intentionally NOT in Phase 2

* **Signal-cli** — host launchd service on Mac. Needs a sidecar with persistent
  volume + SMS verification dance to translate to k8s. Skipped for now.
* **Host Ollama** — the local stack relies on `host.docker.internal:11434`. In
  k8s, either run Ollama as its own Deployment (GPU node selector required for
  practical performance), or rely entirely on the OpenRouter cascade.
* **Docker-socket sandbox** — security model is built around a host docker
  socket proxy. K8s replacement would be a Job-spawning controller. Not
  implemented; sandbox tool is disabled in k8s mode.

### What needs human attention in the chart

See `deploy/k8s/README.md` for the full punch list. The biggest one:
**Neo4j auth** — the official image expects `NEO4J_AUTH=neo4j/<password>` as a
single string, but our secret stores just the password. Currently this won't
authenticate on first boot; needs a small entrypoint shim or use the
[neo4j-helm-charts](https://github.com/neo4j/helm-charts) project as the basis
instead.

---

## Phase 3 — Cloud bootstrap (planned)

`./install.sh --target aws` and `--target gcp` will provision the cluster +
managed Postgres, then run the Phase 2 Helm chart on top.

Status: design only. See `deploy/terraform/README.md` for the planned module
structure, what "fully automatic" can and cannot mean here, and cost
guardrails.

This phase is intentionally last. Until Phase 2 has been validated against a
real cluster (even a local kind/k3d), debugging cloud-bootstrap on top of an
unproven chart means debugging two layers at once.

---

## Troubleshooting

**`docker info` fails after install on Linux**

You were added to the `docker` group, but your current shell doesn't have it
yet. Log out and back in, or run `newgrp docker`.

**`MEM0_POSTGRES_PASSWORD must be set in .env`**

Compose's `:?` syntax fails fast when a required var is missing. Re-run
`./install.sh` — it auto-generates these. If you need to recover an existing
DB, set them manually in `.env` to match the old values.

**`pip install` fails inside the container**

Almost always a network issue or transient PyPI hiccup. Retry with
`docker compose build --no-cache gateway`.

**Health check timeout for `gateway HTTP`**

The gateway image is large; first start can take 60–120 s after the container
shows "running". Re-run `./install.sh --verify` after another minute. If it
still fails, check logs: `docker compose logs gateway --tail 100`.

**Apple Silicon emulation warnings**

The Postgres / Neo4j base images are linux/amd64. Docker Desktop emulates
under Rosetta 2; performance is fine for dev workloads. If you want native
arm64 images, swap to `pgvector/pgvector:pg16-bookworm` (multi-arch) and
`neo4j:5-community` (already multi-arch).
