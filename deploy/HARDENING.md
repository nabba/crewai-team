# Production Hardening Guide

Three layers of defense beyond the default `helm install`. Applied in
order of value-per-effort:

  1. **NetworkPolicy egress allow-list** — block exfiltration.
  2. **External Secrets Operator (ESO)** — make K8s Secrets a thin
     read-through cache of cloud Secret Manager.
  3. **etcd encryption-at-rest** — encrypt secret values at the
     control-plane storage layer.

## 1. NetworkPolicy egress allow-list

Phase C1 ships a second NetworkPolicy template gated by
`networkPolicy.egressAllowlist.enabled`. It restricts the gateway
pod's outbound traffic to: kube-dns, internal pods, and the explicit
list in `networkPolicy.egressAllowlist.external`.

### Enable
```yaml
# values.yaml
networkPolicy:
  egressAllowlist:
    enabled: true
    fqdnSupport: false   # set true on Cilium / Calico-NetworkSet
    external:
      # Easiest: deploy an HTTP egress proxy in-cluster (Squid with FQDN ACLs)
      # and direct ALL gateway egress through it. The list below is then a
      # single ipBlock for the proxy service.
      - ipBlocks: ["10.0.42.5/32"]   # proxy ClusterIP — example
        ports:
          - { protocol: TCP, port: 3128 }
        comment: "egress proxy with FQDN ACLs"
```

### Verify
```bash
kubectl exec -it deploy/botarmy-gateway -- \
  curl -m 5 -sI https://api.anthropic.com   # should succeed via proxy

kubectl exec -it deploy/botarmy-gateway -- \
  curl -m 5 -sI https://untrusted.example.com   # should hang/timeout
```

## 2. External Secrets Operator (ESO)

The default Terraform writes secret values into a `kubernetes_secret`
(base64, not encrypted). ESO replaces this with an `ExternalSecret`
that reads from cloud Secret Manager (AWS / GCP) on a refresh
interval — rotating a value in Secret Manager updates the K8s Secret
without `terraform apply`.

### Install ESO

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace
```

### Replace the kubernetes_secret resource

In `deploy/terraform/aws/secrets.tf`, **after enabling ESO**:

```hcl
# Replace the kubernetes_secret resource with:
resource "kubernetes_manifest" "botarmy_external_secret" {
  manifest = {
    apiVersion = "external-secrets.io/v1beta1"
    kind       = "ExternalSecret"
    metadata = {
      name      = "botarmy-env"
      namespace = kubernetes_namespace.botarmy.metadata[0].name
    }
    spec = {
      refreshInterval = "5m"
      secretStoreRef = {
        kind = "ClusterSecretStore"
        name = "aws-secrets-manager"
      }
      target = {
        name           = "botarmy-env"
        creationPolicy = "Owner"
      }
      dataFrom = [{
        extract = {
          key = aws_secretsmanager_secret.botarmy_env.name
        }
      }]
    }
  }
}
```

Plus the `ClusterSecretStore` (one-time, per-cluster):

```hcl
resource "kubernetes_manifest" "aws_secret_store" {
  manifest = {
    apiVersion = "external-secrets.io/v1beta1"
    kind       = "ClusterSecretStore"
    metadata   = { name = "aws-secrets-manager" }
    spec = {
      provider = {
        aws = {
          service = "SecretsManager"
          region  = var.region
          auth = {
            jwt = {
              serviceAccountRef = {
                name      = "external-secrets"
                namespace = "external-secrets"
              }
            }
          }
        }
      }
    }
  }
}
```

### IAM (AWS)

Bind the ESO service account to the IAM role that can read your secret:

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

### Verify
```bash
kubectl get externalsecret botarmy-env -n botarmy
# STATUS should report "SecretSynced" within the refresh interval.

kubectl get secret botarmy-env -n botarmy -o yaml
# data block should now be ESO-managed (annotations include external-secrets).
```

## 3. etcd encryption-at-rest

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

## Order of operations

1. Apply NetworkPolicy egress in a dev cluster first; verify nothing
   you depend on is blocked. Watch for missed allowlist entries with
   `kubectl logs -n kube-system <cni-pod> | grep DROP`.
2. Install ESO. Migrate one secret first, verify ``status: SecretSynced``,
   then convert the rest in one Terraform PR.
3. Enable etcd encryption at cluster create time (best). For an
   existing cluster, follow your provider's "in-place" path
   (EKS: re-encryption is operator-driven via `aws eks update-cluster-config`;
   GKE: rotation is automatic once `database_encryption` is set).

## Risk notes

- **NetworkPolicy egress** — the highest-risk lever. A missed
  allowlist entry breaks LLM calls or Firecrawl silently. Stage
  rollout: enable, watch logs, expand the list, re-deploy.
- **ESO** — adds a runtime dependency. If ESO crashes, secret refresh
  stops; existing K8s Secrets remain (since `creationPolicy: Owner`).
  Acceptable for our blast radius.
- **etcd encryption** — adds a KMS dependency. KMS outage prevents
  reading secrets, which prevents pods starting. Cloud KMS SLAs are
  generally three nines+; acceptable.
