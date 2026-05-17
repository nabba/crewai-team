# VPC Service Controls perimeter (GCP)

VPC-SC creates an org-level network perimeter around your project that
blocks data exfiltration through Google-managed APIs (Cloud Storage,
Secret Manager, CloudSQL, Artifact Registry, etc.). It's the GCP
equivalent of "the data physically cannot leave this perimeter even if
someone steals a service account key."

**Why opt-in even at `strict`:** misconfiguring VPC-SC locks you out of
your own resources just as fast as a bad `master_authorized_networks`
does. Real production engagement, where IAM alone isn't enough, is the
right use case; a single-operator setup running on a workstation can
live without it.

## Prerequisites

1. **Workspace org** — VPC-SC is org-level. Solo Gmail accounts can't
   apply. The migrate wizard auto-detects via
   `app.substrate.cloud_hardening.detect_org_id()`.
2. **Access policy** — each org has at most one. The wizard
   auto-detects via `detect_access_policy_id(org_id)`. If none
   exists, create one once:
   ```bash
   gcloud access-context-manager policies create \
       --organization $ORG_ID \
       --title "BotArmy access policy"
   ```
3. **Caller permissions** —
   `roles/accesscontextmanager.policyAdmin` at the org level.

## Enabling

1. In `/cp/settings` → Cloud hardening → **VPC Service Controls** →
   click `Enable` (typed-phrase `ENABLE VPC SC` — there's no recovery
   from a bad perimeter without the break-glass below).
2. Confirm the rendered tfvars block in the next wizard run includes
   `vpc_sc_enabled = true` + a populated `access_policy_id`.
3. Apply.

**The first apply runs in DRY-RUN mode.** The perimeter is provisioned
in `use_explicit_dry_run_spec = true`, which means the spec is logged
as "would have blocked" events in Cloud Audit Logs but never enforced.
This is the safe path: review the logs for ~1 week, see what would have
been refused, identify operator workflows that need ingress allowlists,
THEN flip to enforced mode.

## Promoting to enforced

After ~1 week of clean dry-run logs (no false positives blocking
legitimate operator actions):

1. `/cp/settings` → Cloud hardening → VPC Service Controls → toggle
   `Dry-run` OFF.
2. Re-run the wizard; the same configuration moves from `spec` (dry-run)
   to `status` (enforced) on the perimeter resource.

If the first enforced apply breaks an operator workflow, immediately
flip Dry-run back ON. The terraform apply takes 30s and reverts cleanly.

## Restricted services (default)

```
storage.googleapis.com
container.googleapis.com
secretmanager.googleapis.com
sqladmin.googleapis.com
artifactregistry.googleapis.com
cloudkms.googleapis.com
logging.googleapis.com
monitoring.googleapis.com
containeranalysis.googleapis.com
binaryauthorization.googleapis.com
```

Extend via `var.vpc_sc_restricted_services` in your tfvars. Adding new
services AFTER going to enforced needs another ~week of dry-run
observation for each addition.

## Access level

The auto-generated access level uses the SAME CIDR allow-list as
`master_authorized_networks` (Tailnet + laptop public IP, auto-detected).
So if you `botarmy hardening refresh-allowed-cidrs --write`, that
refresh also unblocks VPC-SC ingress — single source of truth.

## Lock-out break-glass

Documented in [CLOUD_LOCKOUT_RECOVERY.md](CLOUD_LOCKOUT_RECOVERY.md).
Most common path:

```bash
# Disable the perimeter via gcloud (operator must have policy-admin)
gcloud access-context-manager perimeters update \
    "accessPolicies/${ACCESS_POLICY_ID}/servicePerimeters/${PERIMETER}" \
    --policy "${ACCESS_POLICY_ID}" \
    --set-resources "" --set-restricted-services "" --set-access-levels ""
```

Or via the React Settings card: toggle Dry-run ON. Next apply makes the
perimeter observational again within 30s.

## Composition with other hardening

VPC-SC composes with — does not replace — the other layers:
* `master_authorized_networks` blocks at the K8s API tier (Layer 7).
* Cloud Armor blocks at the LB tier (Layer 7).
* VPC-SC blocks at the API perimeter tier (cross-project / public
  internet) — i.e. it stops a leaked credential from copying data
  OUT to a different project or to a personal Gmail.
* CMEK + audit-log sinks remain enforced regardless.

VPC-SC is the layer that handles "an attacker has valid IAM but is on
the wrong network."
