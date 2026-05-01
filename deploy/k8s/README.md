# BotArmy on Kubernetes — Phase 2

This Helm chart deploys the gateway + memory stack (Postgres/pgvector, Neo4j,
ChromaDB) to any Kubernetes cluster. It's the path forward for cloud
deployment (EKS / GKE / AKS / on-prem).

## Status

| Component | State | Notes |
| --- | --- | --- |
| Gateway Deployment + Service + Ingress | ✅ scaffolded | Image must be built + pushed to a registry first |
| Postgres (pgvector) StatefulSet | ✅ scaffolded | Uses `pgvector/pgvector:pg16` |
| Neo4j StatefulSet | ⚠️ scaffolded, auth shim TODO | NEO4J_AUTH expects `neo4j/<password>` — needs entrypoint munging |
| ChromaDB StatefulSet | ✅ scaffolded | |
| NetworkPolicy (lock down internal) | ✅ scaffolded | Cluster must run a CNI that supports policies (Calico, Cilium) |
| Secrets via `botarmy-env` | ✅ wired | Created by `./install.sh --target k8s` from local .env |
| **Signal-cli interface** | ❌ skipped | Host-only on local; needs sidecar with persistent volume |
| **Host Ollama** | ❌ skipped | Use cloud-LLM tier only, OR run Ollama as a separate Deployment with GPU node selector |
| **Docker-socket sandbox** | ❌ skipped | Replace with Job-based sandbox (kaniko-style) — not in this chart |
| Self-hosted Firecrawl | ❌ skipped | Add as Phase 2c if needed |

## Image prereq

The k8s dispatcher does NOT build or push images — that's your job before
deploying. Pick a registry your cluster can pull from (Docker Hub for public
testing; ECR/GCR/private registry for production), then:

```bash
# Linux/amd64 — match what your cluster nodes can run.
# (Apple Silicon Macs need --platform linux/amd64, otherwise you push arm64
# and the pod fails with "exec format error".)
docker buildx build --platform linux/amd64 \
    -t YOUR_REGISTRY/botarmy-gateway:0.1.0 \
    --push .
```

Then point the chart at it via shell env vars (the dispatcher reads these):

```bash
export BOTARMY_IMAGE_REPOSITORY=YOUR_REGISTRY/botarmy-gateway
export BOTARMY_IMAGE_TAG=0.1.0
./install.sh --target k8s
```

If you forget, the gateway pod sits in `ImagePullBackOff` and the dispatcher
prints recovery steps. No data is lost — just push the image and re-run
`kubectl rollout restart` per the dispatcher's hint.

## Quick start

```bash
# 1. Build + push the gateway image (see "Image prereq" above)
export BOTARMY_IMAGE_REPOSITORY=YOUR_REGISTRY/botarmy-gateway
export BOTARMY_IMAGE_TAG=0.1.0
docker buildx build --platform linux/amd64 \
    -t "$BOTARMY_IMAGE_REPOSITORY:$BOTARMY_IMAGE_TAG" --push .

# 2. Make sure ./.env exists (run a local install once, or hand-craft it)
ls ../../.env

# 3. Deploy
cd ../..
./install.sh --target k8s

# Or invoke helm directly:
kubectl create namespace botarmy
kubectl -n botarmy create secret generic botarmy-env --from-env-file=.env
helm upgrade --install botarmy ./deploy/k8s \
    --namespace botarmy \
    --set image.repository="$BOTARMY_IMAGE_REPOSITORY" \
    --set image.tag="$BOTARMY_IMAGE_TAG"
```

## Production override example (`values-prod.yaml`)

```yaml
image:
  repository: 1234567890.dkr.ecr.us-east-1.amazonaws.com/botarmy-gateway
  tag: "0.1.0"

gateway:
  replicas: 2
  ingress:
    enabled: true
    host: bot.example.com
    className: nginx
    tls:
      enabled: true

postgres:
  persistence:
    storageClass: gp3            # or pd-ssd on GCP
    size: 50Gi

neo4j:
  persistence:
    storageClass: gp3
    size: 50Gi
```

## What's intentionally not here yet

These are tracked in `INSTALL.md` and require separate design work:

1. **Managed databases** — on AWS, prefer RDS Postgres with `vector` extension
   and Aura DB for Neo4j over running them in-cluster. Set
   `postgres.enabled: false` (TODO: add the toggle) and point the gateway env at
   the managed endpoints via the `botarmy-env` secret.
2. **External Secrets Operator** — replace the `botarmy-env` Secret with
   ExternalSecret + AWS Secrets Manager / GCP Secret Manager.
3. **Sandbox replacement** — the local stack uses a docker-socket proxy + a
   sandbox image. In k8s the equivalent is a Job-spawning controller that
   runs each sandboxed call as a short-lived Pod with a tight
   PodSecurityContext. Out of scope for the first chart pass.
4. **Signal interface** — would need a sidecar with persistent storage for
   signal-cli's profile, plus an init container to register the number. SMS
   verification still has to be done by a human, so true zero-touch isn't
   possible.
5. **Ollama** — if you want local LLM in cluster, add a Deployment with a GPU
   node selector. Without GPUs, stick to the OpenRouter cascade.

## Development tips

```bash
# Render templates without applying
helm template botarmy ./deploy/k8s --debug

# Lint
helm lint ./deploy/k8s

# Dry-run install
helm install botarmy ./deploy/k8s --dry-run --debug
```
