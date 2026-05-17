# Production Hardening Guide

Five layers of defense beyond the default `helm install`. Applied in
order of value-per-effort:

  1. **Gateway HTTP auth** (Bearer token) — block unauthenticated
     mutation of the dashboard / epistemic APIs. **Default ON in K8s.**
  2. **NetworkPolicy egress allow-list** — block exfiltration.
     **Default ON with a permissive HTTPS-only seed since 2026-05.**
  3. **External Secrets Operator (ESO)** — make K8s Secrets a thin
     read-through cache of cloud Secret Manager. Opt-in via the
     `use_external_secrets` Terraform variable.
  4. **etcd encryption-at-rest** — encrypt secret values at the
     control-plane storage layer.
  5. **Redis-backed inbound DLQ** — survive pod restarts and share
     load-shed buffer across replicas. Opt-in via `REDIS_DLQ_URL`.

## 1. Gateway HTTP auth (Phase B, default ON in K8s)

The gateway exposes two router prefixes that mutate state:
`/api/cp/*` (control-plane dashboard) and `/epistemic/*` (calibration
ledger / overrides / pushback). Inside K8s, the gateway binds on
`0.0.0.0`, so the only application-layer boundary is HTTP auth.

### How it works

`app/control_plane/auth_dep.py` exposes a `require_gateway_auth`
FastAPI dependency that both routers attach at construction time:

```python
router = APIRouter(prefix="/api/cp",
                   dependencies=[Depends(require_gateway_auth)])
```

The dependency is **dev-friendly**: when `GATEWAY_AUTH_REQUIRED` is
unset (laptop dev), it returns silently. When set to `1` / `true` /
`yes` / `on`, it enforces `Authorization: Bearer <secret>` via
constant-time `hmac.compare_digest()` against `get_gateway_secret()`
from `app/config.py`.

**Internal Python callers** of `record_override()`, `evaluate_promotion()`,
etc., DO NOT pass through this dependency — the auth boundary is
HTTP, not function calls. Self-evolution chain, SubIA hooks, and
idle-job consumers all keep their existing direct-call semantics.

### Helm wiring

`deploy/k8s/values.yaml` sets `gateway.authRequired: "true"` by
default. The gateway pod template injects `GATEWAY_AUTH_REQUIRED` from
this value. Override per-environment with
`helm install -f values-dev.yaml ...` if you need open API on a
specific dev cluster.

### React dashboard

The build needs `VITE_GATEWAY_SECRET` set so the API client can attach
`Authorization: Bearer ${VITE_GATEWAY_SECRET}` to every request. Helm
should pass this in via the dashboard image build args; on laptop dev
with auth disabled, leave it unset and requests go through unsigned.

### Verify

```bash
# 1. Without a token — should 401 in K8s, 200 in dev mode
kubectl exec -it deploy/botarmy-gateway -- \
  curl -m 5 -i http://localhost:8765/api/cp/projects

# 2. With the right token — should always 200
kubectl exec -it deploy/botarmy-gateway -- sh -c '
  curl -m 5 -i \
    -H "Authorization: Bearer $GATEWAY_SECRET" \
    http://localhost:8765/api/cp/projects
'
```

## 2. NetworkPolicy egress allow-list

Phase C1 ships a second NetworkPolicy template gated by
`networkPolicy.egressAllowlist.enabled`. It restricts the gateway
pod's outbound traffic to: kube-dns, internal pods, and the explicit
list in `networkPolicy.egressAllowlist.external`.

### Default state (2026-05+)

`enabled: true` with a permissive HTTPS-only seed
(`ipBlocks: ["0.0.0.0/0"], port: 443`). Net result: all HTTPS egress
allowed, all non-HTTPS traffic (telnet, raw sockets, plain HTTP,
DNS-over-X) dropped. This matches the laptop deploy's effective
behaviour while still preventing the most common exfiltration paths.

### Tighten further

Replace `0.0.0.0/0` with an in-cluster Squid (or HAProxy) proxy
ClusterIP, configure the proxy with FQDN ACLs, and the egress allow-
list narrows to a single CIDR:

```yaml
# values.yaml
networkPolicy:
  egressAllowlist:
    enabled: true
    fqdnSupport: false   # set true on Cilium / Calico-NetworkSet
    external:
      - ipBlocks: ["10.0.42.5/32"]   # Squid proxy ClusterIP
        ports:
          - { protocol: TCP, port: 3128 }
        comment: "egress proxy with FQDN ACLs"
```

### Opt out

For dev clusters that want zero egress restrictions, set
`networkPolicy.egressAllowlist.enabled: false`. The second
NetworkPolicy template is then not rendered.

### Verify

```bash
kubectl exec -it deploy/botarmy-gateway -- \
  curl -m 5 -sI https://api.anthropic.com   # should succeed (HTTPS allowed)

kubectl exec -it deploy/botarmy-gateway -- \
  curl -m 5 -sI http://untrusted.example.com   # should hang (port 80 blocked)
```

## 3. External Secrets Operator (ESO)

The default Terraform writes secret values into a `kubernetes_secret`
(base64, not encrypted). ESO replaces this with an `ExternalSecret`
that reads from cloud Secret Manager (AWS / GCP) on a refresh
interval — rotating a value in Secret Manager updates the K8s Secret
without `terraform apply`.

### Step 1 — install the ESO controller

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace
```

### Step 2 — flip the Terraform variable

The ESO opt-in is wired as a Terraform variable so the entire flip
from terraform-owned Secret to ESO-reconciled Secret is a one-line
change:

```hcl
# terraform.tfvars
use_external_secrets             = true
external_secret_refresh_interval = "5m"   # default; tune to taste
```

When `use_external_secrets = true`:

* `kubernetes_secret.botarmy_env` is **not** created (count=0).
* A `ClusterSecretStore` is created (provider: AWS Secrets Manager
  / GCP Secret Manager), authenticating via IRSA (AWS) or Workload
  Identity (GCP).
* An `ExternalSecret` is created that ESO reconciles into the same
  Secret name (`botarmy-env`) the Helm chart's `envFrom` references.

The chart needs **zero changes** — it already mounts a Secret called
`botarmy-env`; whether Terraform writes that Secret directly or ESO
reconciles it from Secret Manager is invisible from the chart's POV.

### Step 3 — bind the ESO service account (out-of-band)

The `ClusterSecretStore` references the `external-secrets:external-secrets`
service account. That account must be bound to a cloud identity that
can read the secret.

#### AWS — IRSA

```hcl
module "eso_irsa" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  role_name = "${local.name}-eso"
  attach_external_secrets_policy = true
  external_secrets_secrets_manager_arns = [aws_secretsmanager_secret.botarmy_env.arn]

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["external-secrets:external-secrets"]
    }
  }
}
```

#### GCP — Workload Identity

```hcl
resource "google_service_account" "eso" {
  account_id   = "${local.name}-eso"
  display_name = "External Secrets Operator"
}

resource "google_secret_manager_secret_iam_member" "eso_read" {
  secret_id = google_secret_manager_secret.botarmy_env.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.eso.email}"
}

resource "google_service_account_iam_member" "eso_workload_identity" {
  service_account_id = google_service_account.eso.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[external-secrets/external-secrets]"
}
```

### Verify
```bash
kubectl get externalsecret botarmy-env -n botarmy
# STATUS should report "SecretSynced" within the refresh interval.

kubectl get secret botarmy-env -n botarmy -o yaml
# data block should now be ESO-managed (annotations include external-secrets).
```

## 4. etcd encryption-at-rest

By default, EKS / GKE store K8s Secret values base64-encoded in etcd.
A cluster-admin (or anyone reading etcd) can decode them. Encryption-
at-rest with envelope encryption via cloud KMS makes the values
indecipherable without KMS access.

### AWS EKS (envelope encryption with KMS)

Add to `eks.tf`:

```hcl
resource "aws_kms_key" "eks" {
  description             = "${local.name} EKS envelope encryption"
  deletion_window_in_days = var.tier == "prod" ? 30 : 7
  enable_key_rotation     = true
}

# Pass to the EKS module:
module "eks" {
  # ...
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }
}
```

### GCP GKE (Application-layer Secrets Encryption)

Add to `gke.tf`:

```hcl
resource "google_kms_key_ring" "gke" {
  name     = "${local.name}-gke"
  location = var.region
}

resource "google_kms_crypto_key" "gke_secrets" {
  name     = "secrets"
  key_ring = google_kms_key_ring.gke.id
  rotation_period = "7776000s"   # 90d
}

# In google_container_cluster:
database_encryption {
  state    = "ENCRYPTED"
  key_name = google_kms_crypto_key.gke_secrets.id
}
```

### Verify

For EKS:
```bash
aws eks describe-cluster --name $CLUSTER --query 'cluster.encryptionConfig'
# Should show resources=["secrets"] and the KMS ARN.
```

For GKE:
```bash
gcloud container clusters describe $CLUSTER --zone $ZONE --format='get(databaseEncryption)'
# Should show state: ENCRYPTED and your key.
```

## 5. Redis-backed inbound DLQ (multi-pod)

`app/dead_letter_inbound.py` ships two backends behind the same
public API:

* **In-process deque** (default) — bounded by `_DEFAULT_CAPACITY`
  (200 messages), 30-min TTL. Lost on pod restart. Correct for
  single-pod laptop deploys and dev clusters.
* **Redis-backed list** — opt-in via `REDIS_DLQ_URL`. Multi-pod
  deployments share one queue; pod restarts no longer lose buffered
  messages.

### Enable

Set on the gateway pod (via Helm `extraEnv` or directly in the
secret):

```bash
REDIS_DLQ_URL=redis://botarmy-redis:6379/0
REDIS_DLQ_KEY=botarmy:dlq:inbound      # default — change to share with siblings
REDIS_DLQ_CAPACITY=1000                # default — LTRIM enforced
```

The module logs `dlq: Redis backend active (key=…, cap=…)` at first
operation. If Redis is unreachable at the time of an enqueue/drain, it
logs a warning and falls back to in-process for that operation,
reconnecting on the next.

### Verify

```bash
# Check active backend
kubectl exec -it deploy/botarmy-gateway -- \
  curl -s http://localhost:8765/api/cp/idle/jobs | jq .inbound_dlq
# Expected: { "backend": "redis", "redis_url_configured": true, ... }
```

### Rollout

Existing in-process queue contents are NOT migrated. Drain the queue
first (`/api/cp/idle/jobs` should report depth=0) before flipping
`REDIS_DLQ_URL` on. After the flip, every new load-shed message goes
to Redis and the in-process deque stays empty.

## Order of operations

1. **Gateway auth** is on by default in K8s — no operator action
   needed. Verify the React dashboard is built with
   `VITE_GATEWAY_SECRET` matching the cluster's `GATEWAY_SECRET`.
2. **NetworkPolicy egress** is on by default with a permissive seed.
   Tighten by replacing `0.0.0.0/0` with provider CIDRs or a Squid
   proxy in dev first; verify nothing you depend on is blocked. Watch
   for drops with
   `kubectl logs -n kube-system <cni-pod> | grep DROP`.
3. **ESO** — install the controller, then flip
   `use_external_secrets = true` in `terraform.tfvars` and apply.
   Verify `status: SecretSynced` on the `ExternalSecret`.
4. **etcd encryption** — enable at cluster create time (best). For an
   existing cluster, follow your provider's "in-place" path
   (EKS: re-encryption via `aws eks update-cluster-config`;
   GKE: rotation is automatic once `database_encryption` is set).
5. **Redis DLQ** — only required when going multi-pod. Deploy Redis
   (or use managed Redis), set `REDIS_DLQ_URL`, verify backend is
   `redis` via `/api/cp/idle/jobs`.

## Risk notes

- **Gateway auth** — when `GATEWAY_AUTH_REQUIRED=1` and the gateway
  secret is empty, the dependency returns 503 (`auth misconfigured`)
  rather than silently allowing requests. This is deliberate: a
  missing secret is operator misconfiguration, not a request bug.
- **NetworkPolicy egress** — the highest-risk lever. A missed
  allowlist entry breaks LLM calls or Firecrawl silently. Stage
  rollout: enable, watch logs, expand the list, re-deploy.
- **ESO** — adds a runtime dependency. If ESO crashes, secret refresh
  stops; existing K8s Secrets remain (since `creationPolicy: Owner`).
  Acceptable for our blast radius.
- **etcd encryption** — adds a KMS dependency. KMS outage prevents
  reading secrets, which prevents pods starting. Cloud KMS SLAs are
  generally three nines+; acceptable.
- **Redis DLQ** — adds a runtime dependency. If Redis is unreachable,
  the module silently falls back to per-pod in-process; no requests
  fail. The trade-off is that during a Redis outage, multi-pod
  deployments can briefly split the queue across pods (each pod
  buffers locally). Drain reads from BOTH backends.


## 6. Cloud-hardening profile (provision-time, GCP + AWS)

Added 2026-05-17/18. PROGRAM.md §57. Sits *above* layers 1-5 — those
are runtime/cluster hardening; this layer is provision-time defense
applied by the migrate wizard.

### Activating

The wizard ships with `hardening_profile = "strict"` as the default.
To downgrade or disable, flip in `/cp/settings` → **Cloud hardening**.

| Profile | Applied |
|---|---|
| `off`    | Pre-2026-05-17 behavior — no extra hardening |
| `basic`  | Shielded GKE nodes, Workload Identity, VPC flow logs, CMEK on data-at-rest, deletion_protection on prod tier |
| `strict` | `basic` + master-authorized-networks (Tailnet+IP allowlist), Cloud Armor / WAFv2, Binary Authorization (AUDIT), audit-log sinks, org policies / SCPs (when org_id set) |

### Surfaces

- **REST**: `GET /api/cp/migrate/hardening-preview` (auto-detected
  Tailnet/laptop IP/org), `POST /api/cp/migrate/bootstrap-project`,
  `POST /api/cp/migrate/bootstrap-aws-account`.
- **CLI**: `botarmy hardening preview`, `botarmy hardening
  refresh-allowed-cidrs --write` (lock-out break-glass),
  `botarmy hardening sign-image <image@sha256:...>`.
- **React**: `/cp/migrate` Step 3.5 Hardening card,
  `/cp/settings` → Cloud-hardening card.
- **Install scripts**: `scripts/install/{gcp,aws}_bootstrap.sh` (Stage
  0a), `scripts/install/cosign_setup.sh` (attestor pipeline).

### Operator docs

- **[CLOUD_LOCKOUT_RECOVERY.md](../docs/CLOUD_LOCKOUT_RECOVERY.md)** —
  5 recovery procedures, least-destructive first, for when
  master-authorized-networks / Cloud Armor / Binary Authorization /
  VPC-SC lock you out.
- **[COSIGN_ATTESTOR.md](../docs/COSIGN_ATTESTOR.md)** — image-signing
  setup; prerequisite for graduating `binauthz_mode` from AUDIT to
  ENFORCE.
- **[VPC_SC_PERIMETER.md](../docs/VPC_SC_PERIMETER.md)** — when to
  enable VPC Service Controls, prerequisites, dry-run → enforced
  rollout.

### Composition with layers 1-5

- **Layer 1 (Gateway HTTP auth)** still applies post-provision —
  `master_authorized_networks` blocks at the K8s API tier; gateway
  auth blocks at the application tier inside the cluster.
- **Layer 2 (NetworkPolicy)** unchanged.
- **Layer 3 (ESO)** unchanged; the hardening layer adds CMEK on
  Secret Manager so even if a secret leaks, it's encrypted with a key
  you control.
- **Layer 4 (etcd encryption-at-rest)** — `basic` profile makes this
  customer-managed via `database_encryption.key_name` referencing the
  CMEK keyring. The Helm-chart-level setting is unchanged.
- **Layer 5 (Redis DLQ)** unchanged.

### Failure-open posture

Every detection helper in `app/substrate/cloud_hardening.py` is
failure-isolated — a missing `tailscale` binary or a broken HTTPS probe
returns None, never an exception. Same for the hardening probes added
to `cloud_doctor.py`. The Binary Authorization policy stays
`ALWAYS_ALLOW` when an attestor isn't wired even at ENFORCE mode — the
operator never gets their cluster bricked by mis-configuration alone.
