# Cloud lock-out recovery

When `hardening_profile=strict` is active, the GKE control-plane endpoint
sits behind a `master_authorized_networks` allowlist auto-populated by
the migrate wizard from your Tailnet CIDR + laptop public IP. Several
failure modes can lock you out:

| Failure | What you'll see |
| --- | --- |
| **Laptop IP changes** (home → café, mobile hotspot, VPN flip) | `kubectl get nodes` hangs / `terraform apply` says `dial tcp …: i/o timeout` |
| **Tailnet not reachable** mid-apply | wizard rendered an empty allowlist; control plane is open to any IAM-valid caller but you trust nobody specific |
| **Cloud Armor blocks your client** (rate-limit ban / false-positive WAF) | `403 Forbidden` from ingress, not from the K8s API itself |
| **Binary Authorization in ENFORCE** rejects your deploy | pods stuck in `BinaryAuthorization` phase; `kubectl describe pod` shows the rejection reason |

The recovery procedure ranks the least-destructive options first. Try
in order. Skip ahead only when the safer path can't apply.

---

## 1. Just refresh the allowlist (most common)

```bash
botarmy hardening refresh-allowed-cidrs --write
cd deploy/terraform/gcp && terraform apply \
    -target=google_container_cluster.botarmy \
    -var-file=$WORKSPACE_ROOT/migrations/<latest_run_id>/terraform.tfvars
```

The CLI re-detects your Tailnet CIDR + current laptop public IP, writes
the new `allowed_cidrs` block into the per-run tfvars, and the targeted
apply touches only the cluster's `master_authorized_networks_config` —
no cluster downtime, no destroy/recreate.

`--write` is required to mutate the tfvars in place; without it the CLI
prints the new HCL fragment for you to copy.

---

## 2. Open the master endpoint via gcloud (break-glass, ~2 min)

When the targeted apply also can't reach the API (e.g. the Tailnet
detection broke), bypass terraform entirely:

```bash
gcloud container clusters update botarmy \
    --location europe-north1 \
    --enable-master-authorized-networks \
    --master-authorized-networks "$(curl -s https://checkip.amazonaws.com)/32"
```

This punches your current public IP into the allowlist directly. Run
the targeted terraform apply afterwards to reconcile the tfvars state.

To remove the allowlist entirely (open the master to any IAM-valid
caller — **temporarily, while you fix the underlying problem**):

```bash
gcloud container clusters update botarmy \
    --location europe-north1 \
    --no-enable-master-authorized-networks
```

The cluster's IAM gate still applies — randos on the internet still
can't `kubectl` in. But this widens the attack surface to "any
authenticated Google account that happens to have the right project
binding," so re-enable the allowlist as soon as the underlying issue
is fixed.

---

## 3. Cloud Armor false-positive (rate-limit / WAF)

If `403 Forbidden` is coming from ingress (Cloud Armor) rather than
the cluster:

```bash
# Inspect Cloud Armor events for the last hour
gcloud logging read \
    'resource.type="http_load_balancer" AND jsonPayload.enforcedSecurityPolicy.name="botarmy-armor"' \
    --limit 20 --format json

# Temporarily disable the security policy (don't leave it off!)
gcloud compute backend-services update <backend-service-name> \
    --security-policy "" \
    --global

# Or — exempt your laptop IP from the rate-limit rule
gcloud compute security-policies rules update 1000 \
    --security-policy botarmy-armor \
    --action allow \
    --src-ip-ranges "$(curl -s https://checkip.amazonaws.com)/32"
```

Re-attach the policy once you've identified the offending rule. The
React Settings card lets you flip the whole `hardening_profile` to
`basic` (which omits Cloud Armor) if you need a clean dev environment.

---

## 4. Binary Authorization rejects deploys

If pods are stuck in `BinaryAuthorization` status:

```bash
# Inspect the rejection reason
kubectl get pods -n botarmy
kubectl describe pod -n botarmy <pod-name>
```

Easiest fix: flip `binauthz_mode` back to `AUDIT` in
[/cp/settings](/cp/settings) → Cloud Hardening card. This takes effect
on the next terraform apply (or you can edit the policy directly):

```bash
gcloud container binauthz policy import \
    /path/to/binauthz-policy-audit.yaml
```

Where the policy YAML has:

```yaml
defaultAdmissionRule:
  evaluationMode: ALWAYS_ALLOW
  enforcementMode: DRYRUN_AUDIT_LOG_ONLY
```

ENFORCE is only safe to flip after your image-signing pipeline (cosign
attestor) is wired AND a signed image has been verified. The Settings
card refuses without a typed-phrase confirmation for this reason.

---

## 5. Nuclear: destroy + re-apply

If recovery options 1-4 don't apply, the cluster can be rebuilt:

```bash
# Take a fresh DR snapshot first (CMEK keys are prevent_destroy so they
# survive; CloudSQL data also survives because it's separate from GKE)
botarmy backup --label pre-rebuild

# Destroy + apply
cd deploy/terraform/gcp
terraform destroy -auto-approve \
    -var-file=$WORKSPACE_ROOT/migrations/<latest_run_id>/terraform.tfvars

# Re-run the wizard
# (or use the CLI: botarmy migrate --to gcp --live --confirm "MIGRATE TO GCP")
```

This loses the GKE cluster + helm releases (~15 min to rebuild) but
preserves: KMS keys (CMEK survives via `prevent_destroy`), CloudSQL
instance (separate resource), Secret Manager secrets, Artifact
Registry images, audit-log GCS bucket + BigQuery dataset.

Cluster rebuild from a fresh state takes 10-15 min. Your data is
preserved by the CMEK-protected CloudSQL instance.

---

## Prevention

* **Pin your laptop to Tailnet.** Tailnet IPs (`100.64.0.0/10`) are
  stable across coffee shops, mobile hotspots, and VPN flips. The
  wizard auto-detects Tailnet and pins the whole CGNAT CIDR so any
  Tailnet device of yours stays in the allowlist.
* **Don't promote `binauthz_mode` to ENFORCE** until you have:
  1. A cosign keypair (`deploy/k8s/binauthz/cosign.pub`)
  2. An attestor configured in the project
  3. Your CI/CD signing every image push
  4. A test deploy that succeeded with a signed image
* **Run `botarmy hardening preview`** before starting a wizard run to
  sanity-check what allowlist + org_id the orchestrator will see.

## Related files

* `app/substrate/cloud_hardening.py` — detection + validation
* `app/control_plane/migrate_api.py::get_hardening_preview` — the
  React Step 3.5 card pulls from here
* `deploy/terraform/gcp/hardening.tf` — master_authorized_networks
  block + Binary Authorization policy
* `deploy/terraform/gcp/cloud_armor.tf` — security policy + rules
* `scripts/botarmy` → `hardening` subcommand
