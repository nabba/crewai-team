# BotArmy on AWS / GCP — Phase 3 roadmap

> **Status: design only.** This directory will hold Terraform modules to provision
> the cluster + managed dependencies; once those exist, the same Helm chart from
> `deploy/k8s/` deploys onto them. Nothing here is implemented yet.

## Why a separate phase?

Cloud bootstrap is fundamentally different from "install on a host":

* It provisions billable resources (cluster control plane, node group, RDS,
  load balancer) — typically $200–500/month minimum.
* It needs cloud credentials (`aws configure` / `gcloud auth`).
* It takes 15–30 minutes to come up.
* Mistakes are expensive — destroying a populated RDS instance loses memory.

Mixing this into Phase 1's `install.sh` would make the local installer feel
heavyweight. Instead, the entrypoint dispatches:

```
./install.sh --target aws        # → calls into deploy/terraform/aws/
./install.sh --target gcp        # → calls into deploy/terraform/gcp/
```

…each of which runs `terraform apply`, then re-uses the Phase 2 Helm chart.

## Planned module structure

```
deploy/terraform/
├── aws/
│   ├── main.tf              # EKS cluster + node group + VPC
│   ├── rds.tf               # Postgres 16 with pgvector extension
│   ├── secrets.tf           # AWS Secrets Manager + IAM for ESO
│   ├── networking.tf        # ALB ingress controller setup
│   └── outputs.tf           # cluster endpoint, DB connection string, ARNs
├── gcp/
│   ├── main.tf              # GKE Autopilot cluster
│   ├── cloudsql.tf          # Cloud SQL Postgres + pgvector
│   ├── secrets.tf           # GCP Secret Manager + Workload Identity
│   └── outputs.tf
└── modules/
    ├── neo4j-aura/          # optional: provision Neo4j AuraDB instead of in-cluster
    └── observability/       # CloudWatch / Cloud Logging exporters
```

## What "fully automatic" means here (and doesn't)

The `--target aws` flow can do:

* Provision VPC + EKS + node group
* Provision RDS Postgres with `vector` extension enabled
* Create AWS Secrets Manager entries from the local `.env`
* Install the External Secrets Operator + cert-manager via Helm
* Run the BotArmy Helm chart with overrides pointing at the managed DB
* Configure ALB ingress with ACM cert

The user must still:

* Have an AWS / GCP account with billing enabled
* Run `aws configure` (or set credentials in env / IAM role) **before** invoking
* Decide on a region, domain name, and node sizing (we'll prompt or accept flags)
* Approve the Terraform plan (we'll show the plan + cost estimate, then prompt)

## Cost guardrails

The installer will refuse to apply without showing an estimated monthly cost
(via `infracost` if installed, or a hardcoded ballpark from the chosen tier).
A `--cheapest` flag will pick the smallest viable instance types:
- AWS: `t3.medium` nodes, `db.t3.medium` RDS → ~$120/mo
- GCP: `e2-medium` nodes, `db-f1-micro` Cloud SQL → ~$90/mo

A `--prod` flag will pick reasonable production sizing (multi-AZ, larger nodes).

## When to build this

After Phase 2 (the Helm chart) has been validated against at least one real
cluster — even a local k3d / kind cluster. Bootstrapping cloud resources on top
of a chart that doesn't yet work just means debugging two layers at once.

Track progress in this directory. Adding `aws/main.tf` is the natural next step.
