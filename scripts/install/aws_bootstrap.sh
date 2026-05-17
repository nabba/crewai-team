#!/usr/bin/env bash
# aws_bootstrap.sh — Stage 0a of the AWS migrate wizard.
#
# Creates a new AWS Organizations member account, waits for its
# provisioning to complete, and prints the role-assume command the
# operator runs to switch into the new account before applying the
# AWS terraform module.
#
# Idempotent: if an account with the same email already exists under
# this Organization, this script returns its id without re-creating.
#
# Required args:
#   --email <email>              — root email for the new account
#                                   (must be unique across all AWS)
#   --account-name <name>        — human-readable account name
#
# Optional args:
#   --confirm "CREATE AWS ACCOUNT"   — typed-phrase gate (or via
#                                       BOTARMY_AWS_BOOTSTRAP_CONFIRM)
#   --role-name <name>                — IAM role provisioned in the new
#                                       account for the management account
#                                       to assume (default:
#                                       OrganizationAccountAccessRole)
#   --org-unit-id <ou-id>             — destination OU
#                                       (default: org root)
#   --dry-run                         — print actions, don't change state
#   --help
#
# Prerequisites:
#   * AWS Organizations enabled in the management account
#   * Caller credentials have organizations:CreateAccount and
#     organizations:DescribeCreateAccountStatus
#   * Caller is running as a principal in the management account
#     (member accounts cannot self-create siblings)

set -euo pipefail

EMAIL=""
ACCOUNT_NAME=""
ROLE_NAME="OrganizationAccountAccessRole"
OU_ID=""
CONFIRM_PHRASE=""
DRY_RUN=0
EXPECTED_PHRASE="CREATE AWS ACCOUNT"

usage() {
    sed -n '2,30p' "$0" | sed 's/^# *//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --email) EMAIL="$2"; shift 2 ;;
        --account-name) ACCOUNT_NAME="$2"; shift 2 ;;
        --role-name) ROLE_NAME="$2"; shift 2 ;;
        --org-unit-id) OU_ID="$2"; shift 2 ;;
        --confirm) CONFIRM_PHRASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 2 ;;
    esac
done

CONFIRM_PHRASE="${CONFIRM_PHRASE:-${BOTARMY_AWS_BOOTSTRAP_CONFIRM:-}}"
ACCOUNT_NAME="${ACCOUNT_NAME:-${EMAIL%%@*}}"

log()  { printf '\033[36m[aws-bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[aws-bootstrap]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[aws-bootstrap]\033[0m %s\n' "$*" >&2; }

# ── arg validation ────────────────────────────────────────────
if [[ -z "$EMAIL" ]]; then
    err "--email is required"
    exit 2
fi
if [[ ! "$EMAIL" =~ ^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$ ]]; then
    err "--email format looks invalid: $EMAIL"
    exit 3
fi
# IAM role names: alnum + +=,.@_- ; max 64
if [[ ! "$ROLE_NAME" =~ ^[A-Za-z0-9+=,.@_-]{1,64}$ ]]; then
    err "--role-name invalid (got $ROLE_NAME)"
    exit 3
fi
# OU ids: ou-XXXX-XXXXXXXX (or empty)
if [[ -n "$OU_ID" && ! "$OU_ID" =~ ^ou-[a-z0-9]{4,32}-[a-z0-9]{8,32}$ ]]; then
    err "--org-unit-id invalid (expected ou-XXXX-XXXXXXXX, got $OU_ID)"
    exit 3
fi

# ── typed-phrase gate (skipped for --dry-run) ────────────────
if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ "$CONFIRM_PHRASE" != "$EXPECTED_PHRASE" ]]; then
        err "typed-phrase confirmation required: pass --confirm '$EXPECTED_PHRASE' (or set BOTARMY_AWS_BOOTSTRAP_CONFIRM)"
        err "Refusing to operate without an explicit confirmation."
        exit 5
    fi
fi

# ── caller check ─────────────────────────────────────────────
if ! caller_arn="$(aws sts get-caller-identity --query Arn --output text 2>/dev/null)"; then
    err "no AWS credentials configured. Run 'aws configure' or set AWS_ACCESS_KEY_ID/SECRET."
    exit 6
fi
caller_account="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"
log "Caller: $caller_arn"
log "Caller account: $caller_account"

# Verify Organizations is enabled + caller is in the management account
org_master_account=""
if ! org_master_account="$(aws organizations describe-organization \
    --query 'Organization.MasterAccountId' --output text 2>&1)"; then
    err "AWS Organizations not enabled or caller lacks organizations:DescribeOrganization"
    err "Detail: $org_master_account"
    exit 7
fi
if [[ "$caller_account" != "$org_master_account" ]]; then
    err "caller account ($caller_account) is not the Organizations management account ($org_master_account)"
    err "Member accounts cannot create siblings — switch to the management account first."
    exit 7
fi
log "Organization management account confirmed: $org_master_account"

# ── idempotency: check if account with this email already exists ──
existing_id=""
if existing_id="$(aws organizations list-accounts \
    --query "Accounts[?Email=='$EMAIL'].Id | [0]" --output text 2>/dev/null)" \
    && [[ -n "$existing_id" && "$existing_id" != "None" ]]; then
    log "Account with email $EMAIL already exists: $existing_id (skipping create)"
    log "Assume into the new account with:"
    log "  aws sts assume-role \\"
    log "    --role-arn arn:aws:iam::$existing_id:role/$ROLE_NAME \\"
    log "    --role-session-name botarmy-bootstrap"
    echo "$existing_id"
    exit 0
fi

# ── create account ──────────────────────────────────────────
if [[ "$DRY_RUN" -eq 1 ]]; then
    log "<dry-run> aws organizations create-account \\"
    log "    --email $EMAIL \\"
    log "    --account-name \"$ACCOUNT_NAME\" \\"
    log "    --role-name $ROLE_NAME"
    log "Then poll DescribeCreateAccountStatus until SUCCEEDED, then move to OU $OU_ID if set."
    exit 0
fi

log "Creating account $ACCOUNT_NAME ($EMAIL)..."
create_status="$(aws organizations create-account \
    --email "$EMAIL" \
    --account-name "$ACCOUNT_NAME" \
    --role-name "$ROLE_NAME" \
    --query 'CreateAccountStatus.Id' --output text)"

if [[ -z "$create_status" || "$create_status" == "None" ]]; then
    err "create-account returned no status id"
    exit 8
fi
log "Account create request id: $create_status"

# ── poll for completion (typically ~30-60s) ─────────────────
log "Polling for completion (up to 5 min)..."
deadline=$(($(date +%s) + 300))
new_account_id=""
state="IN_PROGRESS"
while [[ "$state" == "IN_PROGRESS" ]] && [[ "$(date +%s)" -lt "$deadline" ]]; do
    sleep 10
    state_json="$(aws organizations describe-create-account-status \
        --create-account-request-id "$create_status" \
        --output json 2>&1)" || true
    state="$(echo "$state_json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("CreateAccountStatus",{}).get("State",""))' 2>/dev/null || echo UNKNOWN)"
    log "  state=$state"
    if [[ "$state" == "SUCCEEDED" ]]; then
        new_account_id="$(echo "$state_json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("CreateAccountStatus",{}).get("AccountId",""))')"
        break
    fi
    if [[ "$state" == "FAILED" ]]; then
        reason="$(echo "$state_json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("CreateAccountStatus",{}).get("FailureReason",""))')"
        err "create-account failed: $reason"
        exit 9
    fi
done

if [[ -z "$new_account_id" ]]; then
    err "create-account did not complete within 5 min — check the AWS console"
    exit 10
fi

log "Account created: $new_account_id"

# ── move to OU if requested ─────────────────────────────────
if [[ -n "$OU_ID" ]]; then
    # Need the current parent (org root) to move from
    parent_id="$(aws organizations list-parents \
        --child-id "$new_account_id" \
        --query 'Parents[0].Id' --output text)"
    if [[ "$parent_id" != "$OU_ID" ]]; then
        log "Moving account to OU $OU_ID..."
        aws organizations move-account \
            --account-id "$new_account_id" \
            --source-parent-id "$parent_id" \
            --destination-parent-id "$OU_ID"
        log "Moved to OU $OU_ID"
    fi
fi

# ── print operator next-step ────────────────────────────────
log "Bootstrap complete. To assume into the new account:"
log "  aws sts assume-role \\"
log "    --role-arn arn:aws:iam::$new_account_id:role/$ROLE_NAME \\"
log "    --role-session-name botarmy-bootstrap"
log "Then export the returned AccessKeyId/SecretAccessKey/SessionToken, and re-run"
log "  botarmy migrate --to aws --tier cheapest --live --confirm \"MIGRATE TO AWS\""

# Emit just the account id on the LAST line so callers can parse with `tail -1`
echo "$new_account_id"
