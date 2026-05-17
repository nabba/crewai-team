#!/usr/bin/env bash
# cosign_setup.sh — One-time bootstrap for the Binary Authorization
# attestor pipeline.
#
# Generates a cosign keypair, uploads the public key to a Container
# Analysis NOTE, and creates a Binary Authorization ATTESTOR bound to
# that note. After this runs once you can:
#   1. ``botarmy hardening sign-image <full-image-ref>`` to sign images
#   2. flip ``binauthz_mode`` to ENFORCE in /cp/settings
#
# Idempotent: re-runs do not regenerate the keypair, do not duplicate
# the note/attestor, and never overwrite ``cosign.key`` if it exists.
#
# Required args:
#   --project-id <id>           — GCP project the attestor lives in
#
# Optional args:
#   --attestor-name <name>      — defaults to ``botarmy-attestor``
#   --note-id <id>              — Container Analysis note id (defaults to
#                                  ``botarmy-attestor-note``)
#   --keypair-dir <dir>         — output dir; defaults to
#                                  ``deploy/k8s/binauthz/``
#   --password <pw>             — cosign key password (or interactive
#                                  prompt; never echoed to logs)
#   --confirm "SET UP COSIGN"   — typed-phrase gate (or
#                                  BOTARMY_COSIGN_SETUP_CONFIRM env)
#   --dry-run                   — print actions, don't change state
#   --help

set -euo pipefail

PROJECT_ID=""
ATTESTOR_NAME="botarmy-attestor"
NOTE_ID="botarmy-attestor-note"
KEYPAIR_DIR=""
PASSWORD=""
CONFIRM_PHRASE=""
DRY_RUN=0
EXPECTED_PHRASE="SET UP COSIGN"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

usage() {
    sed -n '2,30p' "$0" | sed 's/^# *//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-id) PROJECT_ID="$2"; shift 2 ;;
        --attestor-name) ATTESTOR_NAME="$2"; shift 2 ;;
        --note-id) NOTE_ID="$2"; shift 2 ;;
        --keypair-dir) KEYPAIR_DIR="$2"; shift 2 ;;
        --password) PASSWORD="$2"; shift 2 ;;
        --confirm) CONFIRM_PHRASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 2 ;;
    esac
done

CONFIRM_PHRASE="${CONFIRM_PHRASE:-${BOTARMY_COSIGN_SETUP_CONFIRM:-}}"
KEYPAIR_DIR="${KEYPAIR_DIR:-$REPO_ROOT/deploy/k8s/binauthz}"
PASSWORD="${PASSWORD:-${COSIGN_PASSWORD:-}}"

log()  { printf '\033[36m[cosign-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[cosign-setup]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[cosign-setup]\033[0m %s\n' "$*" >&2; }

# ── arg validation ────────────────────────────────────────────
if [[ -z "$PROJECT_ID" ]]; then
    err "--project-id is required"
    exit 2
fi
if [[ ! "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
    err "--project-id format invalid"
    exit 3
fi
# IDs: lowercase alnum + hyphens, sensible bounds
if [[ ! "$ATTESTOR_NAME" =~ ^[a-z][a-z0-9-]{2,62}[a-z0-9]$ ]]; then
    err "--attestor-name invalid"
    exit 3
fi
if [[ ! "$NOTE_ID" =~ ^[a-z][a-z0-9-]{2,62}[a-z0-9]$ ]]; then
    err "--note-id invalid"
    exit 3
fi

# ── typed-phrase gate (skipped for --dry-run) ────────────────
if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ "$CONFIRM_PHRASE" != "$EXPECTED_PHRASE" ]]; then
        err "typed-phrase confirmation required: pass --confirm '$EXPECTED_PHRASE' (or set BOTARMY_COSIGN_SETUP_CONFIRM)"
        exit 5
    fi
fi

# ── prereqs ──────────────────────────────────────────────────
if ! command -v cosign >/dev/null 2>&1; then
    err "cosign not installed. Install: https://docs.sigstore.dev/cosign/installation/"
    exit 6
fi
if ! command -v gcloud >/dev/null 2>&1; then
    err "gcloud not installed"
    exit 6
fi
if ! gcloud config get-value account >/dev/null 2>&1; then
    err "no active gcloud account"
    exit 6
fi

mkdir -p "$KEYPAIR_DIR"
chmod 700 "$KEYPAIR_DIR"

KEY_PRIV="$KEYPAIR_DIR/cosign.key"
KEY_PUB="$KEYPAIR_DIR/cosign.pub"

# ── 1. Generate keypair (idempotent: skip if exists) ─────────
if [[ -f "$KEY_PRIV" && -f "$KEY_PUB" ]]; then
    log "Keypair already exists at $KEYPAIR_DIR (skipping cosign generate-key-pair)"
else
    log "Generating cosign keypair under $KEYPAIR_DIR..."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  <dry-run> cd $KEYPAIR_DIR && cosign generate-key-pair"
    else
        (
            cd "$KEYPAIR_DIR"
            COSIGN_PASSWORD="$PASSWORD" cosign generate-key-pair
        )
        chmod 600 "$KEY_PRIV"
        chmod 644 "$KEY_PUB"
        log "Keypair created. Private key: $KEY_PRIV (mode 600)"
    fi
fi

# ── 2. Enable required APIs ──────────────────────────────────
log "Ensuring containeranalysis + binaryauthorization APIs..."
if [[ "$DRY_RUN" -eq 1 ]]; then
    log "  <dry-run> gcloud services enable --project=$PROJECT_ID containeranalysis.googleapis.com binaryauthorization.googleapis.com"
else
    gcloud services enable --project="$PROJECT_ID" \
        containeranalysis.googleapis.com binaryauthorization.googleapis.com >/dev/null
    log "APIs ensured"
fi

# ── 3. Create the Container Analysis NOTE (idempotent) ───────
NOTE_PAYLOAD="$(mktemp)"
trap "rm -f $NOTE_PAYLOAD" EXIT

cat > "$NOTE_PAYLOAD" <<EOF
{
  "name": "projects/$PROJECT_ID/notes/$NOTE_ID",
  "attestation": {
    "hint": {
      "human_readable_name": "BotArmy gateway image attestor"
    }
  }
}
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
    log "  <dry-run> create note projects/$PROJECT_ID/notes/$NOTE_ID"
else
    if gcloud container binauthz attestors describe "$ATTESTOR_NAME" \
        --project="$PROJECT_ID" >/dev/null 2>&1; then
        log "Attestor $ATTESTOR_NAME already exists — skipping note + attestor create"
    else
        # Note creation uses the Container Analysis REST API directly
        # (no gcloud subcommand exists for create note).
        log "Creating Container Analysis note $NOTE_ID..."
        gcloud_token="$(gcloud auth print-access-token)"
        curl -sf -X POST \
            -H "Authorization: Bearer $gcloud_token" \
            -H "Content-Type: application/json" \
            -H "x-goog-user-project: $PROJECT_ID" \
            "https://containeranalysis.googleapis.com/v1/projects/$PROJECT_ID/notes/?noteId=$NOTE_ID" \
            -d @"$NOTE_PAYLOAD" >/dev/null || {
            warn "note create returned non-zero (might already exist — continuing)"
        }
        log "Note created"

        # ── 4. Create the ATTESTOR ───────────────────────────
        log "Creating Binary Authorization attestor $ATTESTOR_NAME..."
        gcloud container binauthz attestors create "$ATTESTOR_NAME" \
            --attestation-authority-note="$NOTE_ID" \
            --attestation-authority-note-project="$PROJECT_ID" \
            --project="$PROJECT_ID" >/dev/null
        log "Attestor created"
    fi

    # ── 5. Attach the public key to the attestor ─────────────
    log "Attaching public key to attestor..."
    gcloud container binauthz attestors public-keys add \
        --attestor="$ATTESTOR_NAME" \
        --pkix-public-key-file="$KEY_PUB" \
        --pkix-public-key-algorithm=ECDSA_P256_SHA256 \
        --project="$PROJECT_ID" >/dev/null 2>&1 || {
        warn "public key attach returned non-zero (might already be attached — continuing)"
    }
    log "Public key attached"
fi

# ── 6. Print operator next-steps ────────────────────────────
log "Cosign setup complete."
log ""
log "Next steps:"
log "  1. Store $KEY_PRIV in a secure location (NOT in git — gitignore'd)."
log "  2. Sign your gateway image:"
log "       botarmy hardening sign-image <full-image-ref>"
log "  3. Verify the attestor sees signed images, then in /cp/settings:"
log "     Cloud hardening → Binary Authorization mode → ENFORCE"
log ""
log "Attestor ARN for terraform / policy reference:"
log "  projects/$PROJECT_ID/attestors/$ATTESTOR_NAME"

# Print the attestor full name on the last line so callers can parse it
echo "projects/$PROJECT_ID/attestors/$ATTESTOR_NAME"
