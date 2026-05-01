# BotArmy on AWS — Phase 3

Provisions EKS + RDS Postgres (with pgvector) + ECR + ALB + secrets, then
deploys the BotArmy Helm chart on top. End-to-end, one command:

```bash
cd crewai-team
./install.sh --target aws
```

Takes 15–25 minutes for a cold start. Idempotent — re-running just no-ops if
nothing's changed.

## What gets provisioned

| Resource | Purpose | Module / Resource |
| --- | --- | --- |
| VPC (2 or 3 AZs) | Network | `terraform-aws-modules/vpc/aws` |
| EKS cluster | Kubernetes control plane | `terraform-aws-modules/eks/aws` |
| Managed node group | Worker nodes (t3.medium ×2 / m5.large ×3) | EKS module |
| RDS Postgres 16 + pgvector | Mem0 memory store | `aws_db_instance` (custom) |
| ECR repository | Gateway container image | `aws_ecr_repository` |
| AWS Secrets Manager entry | Source of truth for env vars | `aws_secretsmanager_secret` |
| Kubernetes Secret `botarmy-env` | Mounted into the gateway via envFrom | `kubernetes_secret` |
| AWS Load Balancer Controller | Ingress → ALB | Helm chart |
| cert-manager (optional) | TLS via Let's Encrypt when no ACM cert | Helm chart |
| BotArmy Helm chart | The actual application | `deploy/k8s/` |

## Prerequisites

You provide:

- An AWS account with billing enabled.
- `aws configure` set up (or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars).
- The IAM principal needs broad permissions: EKS, RDS, EC2 (VPC + subnets +
  security groups), ECR, IAM (for IRSA roles), Secrets Manager, ELB. The
  managed `AdministratorAccess` policy is the simplest fit for a personal lab.
  For production, scope this down.
- A domain (optional). Without one, the gateway is reachable only via
  `kubectl port-forward`.
- An ACM certificate for that domain (optional). If absent, cert-manager
  installs Let's Encrypt instead.

The installer provides:

- `terraform`, `aws`, `kubectl`, `helm`, `docker` are checked at start. None
  are auto-installed by `--target aws` (they're rarely missing on developer
  machines and auto-installing terraform across distros is messy).

## Costs

Two tiers, both rough monthly estimates in `eu-north-1`:

| Tier | Compute | DB | Network | Storage | **~$/month** |
| --- | --- | --- | --- | --- | --- |
| `cheapest` | EKS + 2× t3.medium ($133) | db.t4g.micro ($13) | 1× ALB + 1× NAT ($57) | gp3 70 GiB ($7) | **~$210** |
| `prod`     | EKS + 3× m5.large ($283) | db.m5.large multi-AZ ($260) | 1× ALB + 3× NAT ($122) | gp3 150 GiB ($15) | **~$680** |

Big caveats:

- **EKS control plane is $73/mo regardless** — it bills even when no workloads
  are running. If you're not using the cluster, `terraform destroy` it.
- **Data transfer can dominate** for traffic-heavy use. ALB charges per LCU,
  NAT gateway charges per GiB. A surprising amount of bot LLM traffic egresses
  to OpenRouter/Anthropic — assume a few extra dollars per heavy week.
- **RDS storage and backups bill separately** from the instance class.

The dispatcher (`./install.sh --target aws`) shows this cost preview and
requires explicit confirmation before `terraform apply`.

## Configuration

Two ways to configure:

**1. Auto-generated `terraform.auto.tfvars`** — if you have a working `.env`
from a local install, the dispatcher synthesizes a tfvars file from it
(API keys, model selectors). Fastest path; great for "let me try this on AWS
real quick."

**2. Hand-written `terraform.tfvars`** — copy the example:

```bash
cd deploy/terraform/aws
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars
```

Then run the installer. The hand-written file takes precedence over auto.

## Manual flow (if you don't want the dispatcher)

```bash
cd deploy/terraform/aws

terraform init
terraform plan -out tfplan
terraform apply tfplan

# Get outputs
terraform output -raw kubeconfig_command | bash    # ← updates your kubeconfig
ECR_URL=$(terraform output -raw ecr_repository_url)

# Build + push gateway image
aws ecr get-login-password --region $(terraform output -raw cluster_region) | \
    docker login --username AWS --password-stdin "${ECR_URL%/*}"
docker buildx build --platform linux/amd64 -t "${ECR_URL}:latest" --push ../../..

# Roll the gateway pods to pick up the new image
NS=$(terraform output -raw namespace)
kubectl -n "$NS" rollout restart deployment -l app.kubernetes.io/component=gateway
kubectl -n "$NS" rollout status   deployment -l app.kubernetes.io/component=gateway

# Verify
kubectl -n "$NS" get pods,svc,ingress
```

## Hardening (TODO for production)

What v1 doesn't do yet, in priority order:

1. **External Secrets Operator instead of inlined `kubernetes_secret`.** Today,
   rotating an API key requires `terraform apply`. ESO would let you update
   AWS Secrets Manager and have it sync into the cluster automatically.
2. **Private EKS endpoint only.** Currently the cluster API has a public
   endpoint (still IAM-gated). Real prod should set `cluster_endpoint_public_access = false`
   and reach the API via VPN or bastion.
3. **WAF on the ALB.** Add a `aws_wafv2_web_acl` and associate it.
4. **RDS encryption at rest with a customer KMS key** (currently the AWS-managed
   default key, which works but rotates on AWS's schedule).
5. **Backup automation** — RDS automated backups are on (1 day cheapest, 7 days
   prod), but you probably want longer retention + point-in-time restore for a
   real workload. Bump `backup_retention_period`.
6. **Multi-region failover** — out of scope; the BotArmy memory model assumes
   a single primary.
7. **Cost guardrails per workload** — set ResourceQuotas on the namespace,
   LimitRanges per pod, and consider an HPA on the gateway Deployment.

## Tear-down

```bash
cd deploy/terraform/aws
terraform destroy
```

This **does** wipe the RDS instance (skip_final_snapshot is true on cheapest,
false on prod — adjust if you want a snapshot). The Secrets Manager entry has a
0-day recovery window on cheapest, 7 days on prod. ECR images survive.

If `terraform destroy` hangs on a Helm release or a finalizer, the usual fix is:

```bash
kubectl -n botarmy delete ingress --all      # releases the ALB
kubectl -n botarmy delete pvc --all          # releases EBS volumes
terraform destroy
```

## Troubleshooting

**`Error: creating Helm Release ... timeout`**

The ALB controller can take 2–3 min to register before BotArmy's Ingress can
provision its ALB. Re-run `terraform apply`; it's idempotent.

**`Error: postgresql_extension.vector ... could not connect`**

Terraform tries to talk to RDS over the public internet to install pgvector.
RDS is in private subnets — this only works because the dispatcher runs from
your laptop, which routes through the NAT GW. On Linux behind a corporate
firewall this can fail. Workaround: `terraform apply -target=aws_db_instance.botarmy`,
then run `psql "$(terraform output -raw rds_endpoint)" -c 'CREATE EXTENSION vector'`
from a bastion / EC2 jumpbox in the same VPC, then re-run `terraform apply`.

**Pods stuck in `Pending`**

`kubectl describe pod` — usually one of:
- "0/N nodes available: insufficient memory" → bump `node_desired_size` or use
  bigger instance types.
- "no matching PV" → the EBS CSI driver IRSA isn't set up; check
  `kubectl -n kube-system logs -l app=ebs-csi-controller`.

**`docker buildx` fails with "no builder configured"**

```bash
docker buildx create --use
```

Then re-run.
