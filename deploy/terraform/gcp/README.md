# BotArmy on GCP — Phase 3

Provisions GKE Autopilot + Cloud SQL Postgres (with pgvector) + Artifact Registry +
Secret Manager + a Google-managed TLS cert, then deploys the BotArmy Helm chart.

```bash
cd crewai-team
./install.sh --target gcp
```

Takes 15–25 minutes for a cold start. Idempotent.

## What gets provisioned

| Resource | Purpose | Notes |
| --- | --- | --- |
| Custom VPC + subnet | Network | Secondary ranges for pods + services (required for IP aliasing) |
| Cloud NAT | Egress for private nodes | One NAT (cheapest) covers the whole region |
| Service Networking peering | Cloud SQL private IP | Required so the cluster can reach Cloud SQL without a public endpoint |
| GKE Autopilot cluster | Kubernetes | Zonal (cheapest) or regional (prod). Workload Identity on. |
| GCP Service Account `botarmy-gateway` | Pod identity | Bound to the gateway KSA via Workload Identity |
| Cloud SQL Postgres 16 | Mem0 store | `cloudsql.enable_pgvector = on` flag + `CREATE EXTENSION vector` |
| Artifact Registry | Container images | Region-local DOCKER repo with cleanup policy |
| Secret Manager | Env source of truth | `<cluster>-env` JSON blob; gateway SA has accessor role |
| Kubernetes Secret `botarmy-env` | envFrom for the chart | Same shape as local + AWS install |
| Static external IP + ManagedCertificate | Ingress | Only when `domain` is set |
| BotArmy Helm chart | The app | Deployed via the helm provider |

## Prerequisites

- A GCP project with billing enabled.
- `gcloud auth application-default login` so Terraform's google provider has credentials.
- Your principal needs `roles/owner` on the project for the first apply (creates IAM bindings,
  enables APIs, creates service accounts). For ongoing ops, `roles/editor` + a few specific
  roles is enough. Lock down later.
- A domain (optional). Without one, the gateway is reachable only via `kubectl port-forward`.

The dispatcher checks for `gcloud`, `terraform`, `kubectl`, `helm`, `docker` at start and
bails with install URLs if any are missing.

## Costs

Both rough monthly estimates in `europe-north1`. GCP bills more granularly than AWS so
real bills depend a lot on traffic.

| Tier | Compute (Autopilot) | DB | Network | Storage | **~$/month** |
| --- | --- | --- | --- | --- | --- |
| `cheapest` | ~$60 (BotArmy + addons) | db-g1-small ($25) | 1× LB + 1× NAT (~$30) | PD SSD 30 GiB ($5) | **~$120** |
| `prod`     | ~$140 (regional, replicas) | db-custom-2-7680 HA ($140) | 1× LB + regional NAT (~$60) | PD SSD 150 GiB ($25) | **~$365** |

Notes:
- **GKE Autopilot has no node group bills** — you pay per pod resource request × time.
  This is great when BotArmy is the only thing running; less great if you scale to dozens
  of services.
- **Zonal Autopilot's control plane is free**; regional adds ~$73/mo (same as EKS).
- **Cloud SQL HA roughly doubles** the DB bill (synchronous standby in another zone).
- **Egress is the surprise** — LLM API calls go to Anthropic / OpenRouter (egress to
  internet ~$0.12/GB). A heavy-use bot can add $5-20/mo of egress.

The dispatcher shows this estimate and requires explicit confirmation before `apply`.

## Configuration

Two ways:

**1. Auto-generated `terraform.auto.tfvars`** — if you have a working `.env`, the dispatcher
synthesizes one from it.

**2. Hand-written `terraform.tfvars`** — copy the example:

```bash
cd deploy/terraform/gcp
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars         # at minimum, set project_id
```

## Manual flow (if you don't want the dispatcher)

```bash
cd deploy/terraform/gcp

terraform init
terraform plan -out tfplan
terraform apply tfplan

# kubeconfig
$(terraform output -raw kubeconfig_command)

# Push gateway image
AR_URL=$(terraform output -raw artifact_registry_url)
gcloud auth configure-docker "${AR_URL%%/*}" --quiet
docker buildx build --platform linux/amd64 -t "${AR_URL}/gateway:latest" --push ../../..

# Roll the gateway pods
NS=$(terraform output -raw namespace)
kubectl -n "$NS" rollout restart deployment -l app.kubernetes.io/component=gateway
kubectl -n "$NS" rollout status   deployment -l app.kubernetes.io/component=gateway

# Verify
kubectl -n "$NS" get pods,svc,ingress
```

## Troubleshooting

**`postgresql_extension.vector ... could not connect`**

Same root cause as the AWS module. Cloud SQL is on a private IP only — the Terraform
host needs network reachability into the VPC. Options:

1. **From inside the project** (cleanest): run `terraform apply` from a Cloud Shell or a
   Compute Engine instance that lives in the same VPC.
2. **From a laptop** (most common): use `gcloud compute start-iap-tunnel` on a small bastion
   VM, OR install the Cloud SQL Auth Proxy and point the postgresql provider at
   `127.0.0.1`.
3. **Skip the Terraform extension creation**: set `--target=skip` to apply infrastructure
   only, then run `CREATE EXTENSION vector;` manually via Cloud Shell, then re-apply.

For v1 the dispatcher prints clearer instructions if the postgresql provider fails.

**Pods stuck in `Pending` on Autopilot**

Autopilot enforces resource requests/limits. Check `kubectl describe pod` for messages
like "no nodes match required affinity" or "spec.containers[*].resources.requests.cpu must
be specified". The chart's defaults are sized for Autopilot, but if you've overridden them
with very small requests you might have hit the platform's lower bounds.

**`Error: googleapi: ... must be reserved before connect`**

The Service Networking peering didn't finish before Cloud SQL tried to provision. The TF
graph normally serialises this, but sometimes API enablement is slow. Just re-run
`terraform apply`.

**`docker push` fails with "denied: Permission denied"**

You need to authenticate Docker against Artifact Registry:

```bash
gcloud auth configure-docker europe-north1-docker.pkg.dev --quiet
```

Replace `europe-north1` with your actual region.

## Hardening (TODO for production)

1. **External Secrets Operator** instead of inlined `kubernetes_secret` — same as AWS,
   lets you rotate values without re-applying.
2. **Private GKE endpoint only** — `enable_private_endpoint = true`, reach the API via
   IAP TCP forwarding or VPN.
3. **Cloud Armor on the LB** — `kubernetes.io/ingress.allow-http: "false"` plus a
   `BackendConfig` that references a Cloud Armor security policy.
4. **Customer-managed encryption keys (CMEK)** for Cloud SQL + GKE node disks.
5. **Backup retention** — `point_in_time_recovery_enabled` is on for prod tier, but the
   actual retained-WAL window is 7 days. Bump if you need longer.
6. **Workload Identity Federation** for CI — let GitHub Actions deploy without a
   long-lived service account key.

## Tear-down

```bash
cd deploy/terraform/gcp
terraform destroy
```

This removes everything **including the Cloud SQL instance** (deletion_protection is off
in this module — flip it on by hand for prod). Artifact Registry images survive unless
you delete the repo manually.

If `destroy` hangs:

```bash
kubectl -n botarmy delete ingress --all      # releases the LB + IP
kubectl -n botarmy delete pvc --all          # releases PD volumes
terraform destroy
```
