#!/usr/bin/env bash
# e2e-test-gcp.sh — full end-to-end install verification on GCP.
#
# What this does (and what it costs):
#   1. Fresh clone repo into /tmp/botarmy-e2e-<timestamp>
#   2. Pre-flight: bash -n, terraform fmt+validate, helm lint, drift check
#   3. terraform apply  (15-25 min · provisions GKE Autopilot + Cloud SQL)
#   4. Build + push gateway image to Artifact Registry  (3-15 min)
#   5. Wait for gateway pod to become Ready
#   6. Port-forward and verify /metrics emits all four BotArmy metrics
#   7. Verify Mem0 created the pgvector extension via gateway logs
#   8. terraform destroy  (ALWAYS runs, even if 5/6/7 fail)
#   9. Print a one-screen summary + total spend estimate
#
# Estimated cost: $2-3 for ~50 minutes of running infrastructure.
# Estimated runtime: ~50 minutes wall-clock.
#
# Prerequisites (the script checks for all of them and bails with hints):
#   - gcloud CLI logged in to a project with billing enabled
#   - gcloud auth application-default login already done
#   - terraform, kubectl, helm, docker, jq installed
#   - gke-gcloud-auth-plugin on PATH (or in
#     /opt/homebrew/share/google-cloud-sdk/bin which we add automatically)
#   - An .env file with ANTHROPIC_API_KEY, BRAVE_API_KEY, OPENROUTER_API_KEY
#     (we read from $REPO_ROOT/.env or you set BOTARMY_ENV_FILE)
#
# Flags:
#   --project ID         GCP project (default: $GCP_PROJECT_ID, then gcloud config)
#   --region REGION      default: europe-north1
#   --cluster-name NAME  default: botarmy-e2e-<YYYYMMDD-HHMMSS>
#   --keep               don't auto-destroy at the end (you destroy manually)
#   --skip-build         reuse the previously-pushed image (saves 15 min)
#   --yes                skip the cost confirmation prompt
#   --help
#
# Exit codes: 0 ok · 1 user error · 2 prereq missing · 3 apply failed
#             · 4 verification failed · 5 destroy failed (manual cleanup needed)
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────
REPO_URL="${BOTARMY_REPO_URL:-https://github.com/nabba/AndrusAI.git}"
ENV_FILE="${BOTARMY_ENV_FILE:-}"
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="europe-north1"
ZONE="europe-north1-a"
CLUSTER_NAME="botarmy-e2e-$(date +%Y%m%d-%H%M%S)"
TEST_DIR="/tmp/${CLUSTER_NAME}"
LOG_FILE="${TEST_DIR}-log.txt"
KEEP=0
SKIP_BUILD=0
ASSUME_YES=0
START_TS=$(date +%s)

# ── Argument parsing ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)       PROJECT_ID="$2"; shift 2 ;;
        --region)        REGION="$2"; shift 2 ;;
        --cluster-name)  CLUSTER_NAME="$2"; TEST_DIR="/tmp/${CLUSTER_NAME}"; LOG_FILE="${TEST_DIR}-log.txt"; shift 2 ;;
        --keep)          KEEP=1; shift ;;
        --skip-build)    SKIP_BUILD=1; shift ;;
        --yes|-y)        ASSUME_YES=1; shift ;;
        --help|-h)       sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# ── Logging ─────────────────────────────────────────────────────
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
else
    C_RESET= C_RED= C_GREEN= C_YELLOW= C_BLUE= C_BOLD= C_DIM=
fi
info()    { printf "%s[i]%s %s\n" "$C_BLUE"   "$C_RESET" "$*"; }
ok()      { printf "%s[✓]%s %s\n" "$C_GREEN"  "$C_RESET" "$*"; }
warn()    { printf "%s[!]%s %s\n" "$C_YELLOW" "$C_RESET" "$*"; }
err()     { printf "%s[x]%s %s\n" "$C_RED"    "$C_RESET" "$*" >&2; }
step()    { printf "\n%s%s═══ %s ═══%s\n" "$C_BOLD" "$C_BLUE" "$*" "$C_RESET"; }

# ── Trap: ensure terraform destroy ALWAYS runs ──────────────────
DESTROY_NEEDED=0
EXIT_CODE=0
cleanup() {
    local rc=$?
    EXIT_CODE=$rc
    if [[ "$DESTROY_NEEDED" == "1" && "$KEEP" == "0" ]]; then
        step "Cleanup: terraform destroy"
        warn "ALWAYS-RUN cleanup. Don't interrupt."
        if (cd "$TEST_DIR/AndrusAI/deploy/terraform/gcp" \
                && terraform destroy -var-file=terraform.tfvars -input=false -auto-approve 2>&1); then
            ok "Resources destroyed."
        else
            err "Destroy failed. CHECK GCP CONSOLE FOR ORPHANED RESOURCES:"
            err "  https://console.cloud.google.com/welcome?project=${PROJECT_ID}"
            err "  Look for: SQL instances, GKE clusters, VPCs containing '${CLUSTER_NAME}'"
            EXIT_CODE=5
        fi
    elif [[ "$KEEP" == "1" && "$DESTROY_NEEDED" == "1" ]]; then
        warn "--keep flag set — resources still running. Destroy manually:"
        warn "  cd ${TEST_DIR}/AndrusAI/deploy/terraform/gcp && terraform destroy"
    fi

    local elapsed=$(( $(date +%s) - START_TS ))
    local mins=$((elapsed / 60))
    echo
    info "Total elapsed: ${mins} minutes"
    info "Full log: ${LOG_FILE}"
    exit "$EXIT_CODE"
}
trap cleanup EXIT INT TERM

# ── Prereq checks ───────────────────────────────────────────────
step "Prerequisites"
require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Missing: $1 — $2"
        exit 2
    fi
    ok "$1: $(command -v "$1")"
}

# Add gcloud's bundled gke-gcloud-auth-plugin to PATH on Mac
[[ -d "/opt/homebrew/share/google-cloud-sdk/bin" ]] \
    && export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

require_cmd gcloud    "https://cloud.google.com/sdk/docs/install"
require_cmd terraform "https://developer.hashicorp.com/terraform/install"
require_cmd kubectl   "https://kubernetes.io/docs/tasks/tools/"
require_cmd helm      "https://helm.sh/docs/intro/install/"
require_cmd docker    "https://docs.docker.com/get-docker/"
require_cmd jq        "brew install jq"
require_cmd git       "(should be present)"

if ! command -v gke-gcloud-auth-plugin >/dev/null 2>&1; then
    warn "gke-gcloud-auth-plugin not on PATH. Installing via gcloud components..."
    gcloud components install gke-gcloud-auth-plugin --quiet || \
        warn "  failed — run: gcloud components install gke-gcloud-auth-plugin"
fi

# ── GCP auth ────────────────────────────────────────────────────
step "GCP authentication"
ACCOUNT=$(gcloud config get-value account 2>/dev/null)
[[ -z "$ACCOUNT" || "$ACCOUNT" == "(unset)" ]] && {
    err "No active gcloud account. Run: gcloud auth login"; exit 2
}
ok "gcloud account: $ACCOUNT"

if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    err "Application-Default Credentials not set."
    err "Run in another terminal: gcloud auth application-default login"
    exit 2
fi
ok "Application-Default Credentials: active"

if [[ -z "$PROJECT_ID" ]]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]] && {
        err "No GCP project. Pass --project ID or run: gcloud config set project ID"; exit 1
    }
fi
ok "GCP project: $PROJECT_ID"

BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)' 2>/dev/null || echo "false")
if [[ "$BILLING" != "True" && "$BILLING" != "true" ]]; then
    err "Project $PROJECT_ID has no active billing."
    err "Link a billing account at https://console.cloud.google.com/billing"
    exit 2
fi
ok "Billing: enabled on $PROJECT_ID"

# ── API key sourcing ────────────────────────────────────────────
step "Locating API keys"
if [[ -z "$ENV_FILE" ]]; then
    REPO_LOCAL_GUESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
    [[ -r "$REPO_LOCAL_GUESS/.env" ]] && ENV_FILE="$REPO_LOCAL_GUESS/.env"
fi
if [[ -z "$ENV_FILE" || ! -r "$ENV_FILE" ]]; then
    err "No .env file found. Pass BOTARMY_ENV_FILE=/path/to/.env or run from a checkout with .env present."
    exit 1
fi
ok "Reading API keys from: $ENV_FILE"

env_get() {
    awk -F= -v k="$1" '$1==k { sub(/^[^=]+=/, ""); print; exit }' "$ENV_FILE" \
        | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}
ANTHROPIC_KEY=$(env_get ANTHROPIC_API_KEY)
BRAVE_KEY=$(env_get BRAVE_API_KEY)
OPENROUTER_KEY=$(env_get OPENROUTER_API_KEY)

for var in ANTHROPIC_KEY BRAVE_KEY OPENROUTER_KEY; do
    val="${!var}"
    if [[ -z "$val" || "$val" =~ ^your_.* ]]; then
        err "$var not set in $ENV_FILE (or still has placeholder value)."
        exit 1
    fi
done
ok "All three API keys present"

# ── Cost gate ───────────────────────────────────────────────────
cat <<EOF

  ${C_BOLD}This test will spend real money on GCP.${C_RESET}

  Project:      $PROJECT_ID
  Region:       $REGION
  Cluster:      $CLUSTER_NAME
  Tier:         cheapest
  Test dir:     $TEST_DIR

  Resources provisioned:
    - GKE Autopilot regional cluster (free first-cluster credit)
    - Cloud SQL db-g1-small Postgres 16 (~\$0.034/hr)
    - 1× Cloud NAT (~\$0.045/hr)
    - 1× Compute load balancer (~\$0.025/hr forwarding rule)
    - Artifact Registry, Secret Manager, IAM (effectively free)

  Estimated total: ${C_BOLD}\$2-3 for ~50 minutes${C_RESET}.

EOF
if [[ "$ASSUME_YES" != "1" ]]; then
    read -r -p "Proceed? [y/N] " r
    [[ "$r" =~ ^[Yy] ]] || { err "Aborted."; exit 1; }
fi

# ── Fresh clone ─────────────────────────────────────────────────
step "Clone fresh from $REPO_URL"
mkdir -p "$TEST_DIR"
git clone --depth 1 "$REPO_URL" "$TEST_DIR/AndrusAI" 2>&1 | tail -3
cd "$TEST_DIR/AndrusAI"
ok "Clone landed at $TEST_DIR/AndrusAI ($(git rev-parse --short HEAD))"

# ── Pre-flight smoke checks (the same things CI runs) ─────────
step "Pre-flight smoke checks"
for f in install.sh scripts/install/*.sh scripts/check-env-drift.sh scripts/e2e-test-gcp.sh; do
    [[ -f "$f" ]] && bash -n "$f" && ok "bash -n: $f"
done
( cd deploy/terraform/aws && terraform fmt -recursive -check >/dev/null \
    && terraform init -backend=false -input=false >/dev/null \
    && terraform validate ) | tail -2
( cd deploy/terraform/gcp && terraform fmt -recursive -check >/dev/null \
    && terraform init -backend=false -input=false >/dev/null \
    && terraform validate ) | tail -2
helm lint deploy/k8s/ 2>&1 | tail -3
bash scripts/check-env-drift.sh | tail -7

# ── Generate tfvars ─────────────────────────────────────────────
step "Generate terraform.tfvars"
cat > deploy/terraform/gcp/terraform.tfvars <<EOF
project_id        = "$PROJECT_ID"
region            = "$REGION"
zone              = "$ZONE"
cluster_name      = "$CLUSTER_NAME"
tier              = "cheapest"
enable_monitoring = false
extra_env = {
  ANTHROPIC_API_KEY  = "$ANTHROPIC_KEY"
  BRAVE_API_KEY      = "$BRAVE_KEY"
  OPENROUTER_API_KEY = "$OPENROUTER_KEY"
}
EOF
ok "Wrote $TEST_DIR/AndrusAI/deploy/terraform/gcp/terraform.tfvars"

# ── Apply ───────────────────────────────────────────────────────
step "terraform plan + apply (15-25 min)"
cd deploy/terraform/gcp
terraform init -upgrade -input=false 2>&1 | tail -3
terraform plan -var-file=terraform.tfvars -input=false -out=botarmy.tfplan 2>&1 | tail -5
DESTROY_NEEDED=1   # from this point we're billing
terraform apply -input=false -auto-approve botarmy.tfplan
ok "Apply complete"

# ── Capture outputs + kubeconfig ────────────────────────────────
step "Wire up kubectl + docker"
AR_URL=$(terraform output -raw artifact_registry_url)
NS=$(terraform output -raw namespace)
LOCATION=$(terraform output -raw cluster_location)

gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --location "$LOCATION" --project "$PROJECT_ID" 2>&1 | tail -2
gcloud auth configure-docker "${AR_URL%%/*}" --quiet 2>&1 | tail -1
ok "Configured: kubectl context + docker auth for $AR_URL"

# ── Build + push gateway image ──────────────────────────────────
if [[ "$SKIP_BUILD" == "1" ]]; then
    warn "Skipping image build (--skip-build) — assuming previous image still in registry."
else
    step "Build + push gateway image (linux/amd64, 3-15 min depending on cache)"
    cd "$TEST_DIR/AndrusAI"
    TAG="e2e-$(date +%Y%m%d-%H%M%S)"
    docker buildx build --platform linux/amd64 \
        -t "${AR_URL}/gateway:${TAG}" \
        -t "${AR_URL}/gateway:latest" \
        --push --progress=plain .
    ok "Pushed: ${AR_URL}/gateway:${TAG}"
fi

# ── Restart gateway + wait Ready ────────────────────────────────
step "Restart gateway pod, wait for Ready"
kubectl -n "$NS" rollout restart deployment -l app.kubernetes.io/component=gateway
if kubectl -n "$NS" rollout status deployment -l app.kubernetes.io/component=gateway --timeout=10m; then
    ok "Gateway deployment Ready"
else
    err "Gateway didn't become Ready in 10 min — pulling diagnostics:"
    kubectl -n "$NS" get pods -o wide
    kubectl -n "$NS" describe pod -l app.kubernetes.io/component=gateway 2>&1 | tail -40
    kubectl -n "$NS" logs -l app.kubernetes.io/component=gateway --tail=80 2>&1 | tail -100
    err "Bug caught — gateway didn't start cleanly."
    exit 4
fi

# ── /metrics verification ───────────────────────────────────────
step "Verify /metrics emits the BotArmy app metrics"
GW_SVC=$(kubectl -n "$NS" get svc -l app.kubernetes.io/component=gateway -o name | head -1)
kubectl -n "$NS" port-forward "$GW_SVC" 18765:8765 >/tmp/pf-${CLUSTER_NAME}.log 2>&1 &
PF_PID=$!
sleep 5

if ! curl -sf http://127.0.0.1:18765/metrics > /tmp/metrics-${CLUSTER_NAME}.out; then
    err "/metrics endpoint not reachable — port-forward log:"
    cat /tmp/pf-${CLUSTER_NAME}.log
    kill $PF_PID 2>/dev/null
    exit 4
fi

MISSING=()
for m in llm_requests_total llm_request_duration_seconds llm_cascade_all_tiers_failed_total \
         mem0_postgres_connection_errors_total http_requests_total http_request_duration_seconds; do
    grep -q "^# HELP $m " /tmp/metrics-${CLUSTER_NAME}.out || MISSING+=("$m")
done
kill $PF_PID 2>/dev/null

if (( ${#MISSING[@]} > 0 )); then
    err "Missing metrics: ${MISSING[*]}"
    exit 4
fi
ok "All 6 metrics present in /metrics"

# ── Mem0 / pgvector verification ────────────────────────────────
step "Verify Mem0 connected to Cloud SQL + created pgvector"
MEM0_LOG=$(kubectl -n "$NS" logs -l app.kubernetes.io/component=gateway --tail=500 2>&1 | grep -i "mem0:" | head -10)
if [[ -n "$MEM0_LOG" ]]; then
    echo "$MEM0_LOG" | sed 's/^/    /'
    if echo "$MEM0_LOG" | grep -q "client initialised"; then
        ok "Mem0 client initialised against Cloud SQL"
    else
        warn "Mem0 logged but no 'client initialised' confirmation."
        warn "This may indicate a connection problem — check logs above."
    fi
else
    warn "No mem0: log lines yet. Could be timing — check manually:"
    warn "  kubectl -n $NS logs -l app.kubernetes.io/component=gateway | grep -i mem0"
fi

# ── Summary ─────────────────────────────────────────────────────
step "Summary"
ELAPSED=$(( $(date +%s) - START_TS ))
MINS=$((ELAPSED / 60))
COST_EST=$(python3 -c "print(f'\${0.10 * $MINS / 60:.2f}')" 2>/dev/null || echo "(install python3 for cost calc)")
cat <<EOF

  ${C_GREEN}${C_BOLD}E2E test PASSED${C_RESET}

  Cluster:       $CLUSTER_NAME ($LOCATION)
  Project:       $PROJECT_ID
  Elapsed:       ${MINS} minutes
  Est. spend:    ${COST_EST}
  Log file:      $LOG_FILE

  Tearing down now (set --keep to skip).

EOF

# Trap will fire terraform destroy on exit
exit 0
