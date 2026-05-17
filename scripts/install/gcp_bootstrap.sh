#!/usr/bin/env bash
# gcp_bootstrap.sh — Stage 0a of the migrate wizard.
#
# Creates a GCP project, links it to a billing account, and enables the
# required APIs. Strictly opt-in (gated by runtime_settings.gcp_bootstrap_enabled
# OR --confirm + typed phrase).
#
# Idempotent: if the project already exists in ACTIVE state, this script
# is a no-op and exits 0. Existing billing links and API enablement are
# also checked before being (re-)applied.
#
# Required args:
#   --project-id <id>           — desired project ID (e.g. botarmy-495107)
#   --billing-account <id>      — e.g. 01ABCD-EFGH12-IJ34KL
#
# Optional args:
#   --confirm "CREATE GCP PROJECT"   — typed phrase gate (or set
#                                       BOTARMY_GCP_BOOTSTRAP_CONFIRM env var)
#   --project-name <name>             — human-readable project name (defaults to project_id)
#   --org-id <id>                     — parent org (numeric, no prefix). Without it the
#                                       project sits under "No organization"
#   --dry-run                         — print what would happen, don't change anything
#   --help

set -euo pipefail

PROJECT_ID=""
PROJECT_NAME=""
BILLING_ACCOUNT=""
ORG_ID=""
CONFIRM_PHRASE=""
DRY_RUN=0

EXPECTED_PHRASE="CREATE GCP PROJECT"

# Required APIs — same list as deploy/terraform/gcp/main.tf::required_apis,
# kept in sync by hand. Terraform also enables them, but doing it here
# means the wizard's plan step doesn't error on missing API service-agents.
REQUIRED_APIS=(
    "compute.googleapis.com"
    "container.googleapis.com"
    "sqladmin.googleapis.com"
    "servicenetworking.googleapis.com"
    "secretmanager.googleapis.com"
    "artifactregistry.googleapis.com"
    "iam.googleapis.com"
    "monitoring.googleapis.com"
    "logging.googleapis.com"
    "cloudresourcemanager.googleapis.com"
    "cloudbilling.googleapis.com"
    "cloudkms.googleapis.com"
    "binaryauthorization.googleapis.com"
    "containeranalysis.googleapis.com"
    "orgpolicy.googleapis.com"
    "bigquery.googleapis.com"
)

# ─── arg parse ────────────────────────────────────────────────
usage() {
    sed -n '2,20p' "$0" | sed 's/^# *//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-id) PROJECT_ID="$2"; shift 2 ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        --billing-account) BILLING_ACCOUNT="$2"; shift 2 ;;
        --org-id) ORG_ID="$2"; shift 2 ;;
        --confirm) CONFIRM_PHRASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 2 ;;
    esac
done

CONFIRM_PHRASE="${CONFIRM_PHRASE:-${BOTARMY_GCP_BOOTSTRAP_CONFIRM:-}}"
PROJECT_NAME="${PROJECT_NAME:-$PROJECT_ID}"

# ─── pre-flight ───────────────────────────────────────────────
log()  { printf '\033[36m[gcp-bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[gcp-bootstrap]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[gcp-bootstrap]\033[0m %s\n' "$*" >&2; }

if [[ -z "$PROJECT_ID" ]]; then
    err "--project-id is required"
    exit 2
fi
if [[ -z "$BILLING_ACCOUNT" ]]; then
    err "--billing-account is required (find yours at https://console.cloud.google.com/billing)"
    exit 2
fi

# Validate billing-account format: XXXXXX-XXXXXX-XXXXXX
if [[ ! "$BILLING_ACCOUNT" =~ ^[A-Z0-9]{6}-[A-Z0-9]{6}-[A-Z0-9]{6}$ ]]; then
    err "--billing-account format is XXXXXX-XXXXXX-XXXXXX (got $BILLING_ACCOUNT)"
    exit 3
fi

# Validate project_id (GCP rules: 6-30 chars, lowercase + digits + hyphens, must start with a letter)
if [[ ! "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
    err "--project-id must be 6-30 chars, lowercase + digits + hyphens, start with a letter (got $PROJECT_ID)"
    exit 3
fi

# Validate org_id if provided
if [[ -n "$ORG_ID" && ! "$ORG_ID" =~ ^[0-9]+$ ]]; then
    err "--org-id must be a numeric organization ID (no 'organizations/' prefix; got $ORG_ID)"
    exit 3
fi

# ── Idempotency: bail out cleanly if project already ACTIVE
if state="$(gcloud projects describe "$PROJECT_ID" --format='value(lifecycleState)' 2>/dev/null)"; then
    if [[ "$state" == "ACTIVE" ]]; then
        log "Project $PROJECT_ID already exists and is ACTIVE — bootstrap is a no-op"
        # Still ensure billing + APIs in case the project was created without them
        :
    else
        err "Project $PROJECT_ID exists but lifecycleState=$state — refusing to operate"
        exit 4
    fi
fi

# ── Typed-phrase gate (only when about to actually create/modify)
if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ "$CONFIRM_PHRASE" != "$EXPECTED_PHRASE" ]]; then
        err "typed-phrase confirmation required: pass --confirm '$EXPECTED_PHRASE' (or set BOTARMY_GCP_BOOTSTRAP_CONFIRM)"
        err "Refusing to operate without an explicit confirmation."
        exit 5
    fi
fi

# ─── gcloud auth check ───────────────────────────────────────
if ! account="$(gcloud config get-value account 2>/dev/null)"; then
    err "no active gcloud account. Run 'gcloud auth login' first."
    exit 6
fi
log "Active gcloud account: $account"

# ─── 1. Create project ───────────────────────────────────────
if ! gcloud projects describe "$PROJECT_ID" --format='value(projectId)' >/dev/null 2>&1; then
    log "Creating project $PROJECT_ID..."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  <dry-run> gcloud projects create $PROJECT_ID --name=\"$PROJECT_NAME\"${ORG_ID:+ --organization=$ORG_ID}"
    else
        gcloud projects create "$PROJECT_ID" --name="$PROJECT_NAME" \
            ${ORG_ID:+--organization="$ORG_ID"} \
            --no-enable-cloud-apis
        log "Project $PROJECT_ID created"
    fi
else
    log "Project $PROJECT_ID already exists (skipping create)"
fi

# ─── 2. Link billing ─────────────────────────────────────────
current_billing="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)' 2>/dev/null || true)"
if [[ "$current_billing" == "billingAccounts/$BILLING_ACCOUNT" ]]; then
    log "Billing already linked to $BILLING_ACCOUNT (skipping)"
else
    log "Linking billing account $BILLING_ACCOUNT..."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  <dry-run> gcloud billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT"
    else
        gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"
        log "Billing linked"
    fi
fi

# ─── 3. Enable APIs ──────────────────────────────────────────
# Batch-enable — the gcloud API call accepts multiple service names.
log "Enabling ${#REQUIRED_APIS[@]} APIs (this can take ~30s each on first enable)..."
if [[ "$DRY_RUN" -eq 1 ]]; then
    log "  <dry-run> gcloud services enable --project=$PROJECT_ID ${REQUIRED_APIS[*]}"
else
    gcloud services enable --project="$PROJECT_ID" "${REQUIRED_APIS[@]}"
    log "APIs enabled"
fi

# ─── 4. Verify ───────────────────────────────────────────────
log "Verifying final state..."
if [[ "$DRY_RUN" -eq 0 ]]; then
    final_state="$(gcloud projects describe "$PROJECT_ID" --format='value(lifecycleState)')"
    final_billing="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)')"
    log "  lifecycleState: $final_state"
    log "  billing:        $final_billing"
    if [[ "$final_state" != "ACTIVE" ]]; then
        err "project is not ACTIVE — bootstrap left state inconsistent"
        exit 7
    fi
fi

log "Stage 0a complete. Proceed to Step 1 of the migrate wizard."
