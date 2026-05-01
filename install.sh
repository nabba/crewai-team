#!/usr/bin/env bash
# ─── BotArmy installer ─────────────────────────────────────────
# One executable to set up BotArmy on Mac, Linux, or Kubernetes.
#
#   ./install.sh                       # interactive local install (auto-detect OS)
#   ./install.sh --target local        # explicit local install
#   ./install.sh --target k8s          # deploy to current kubectl context
#   ./install.sh --non-interactive     # read API keys from BOTARMY_CONFIG
#   ./install.sh --config ./my.env     # use a specific config file
#   ./install.sh --skip-prereqs        # don't try to install Docker/Python
#   ./install.sh --no-build            # skip docker build (use existing images)
#   ./install.sh --dry-run             # print what would happen, change nothing
#   ./install.sh --verify              # only run health checks on running stack
#   ./install.sh --uninstall           # stop containers, optionally wipe data
#   ./install.sh --help
#
# Exit codes: 0 ok · 1 user error · 2 prereq missing · 3 install failure
# ────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve script dir even when called via symlink
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
INSTALL_ROOT="$(cd -P "$(dirname "$SOURCE")" && pwd)"
export INSTALL_ROOT
export LIB_DIR="${INSTALL_ROOT}/scripts/install"

# shellcheck source=scripts/install/lib.sh
source "${LIB_DIR}/lib.sh"

# ─── Default options ────────────────────────────────────────────
TARGET="local"
INTERACTIVE=1
CONFIG_FILE=""
SKIP_PREREQS=0
NO_BUILD=0
DRY_RUN=0
ONLY_VERIFY=0
UNINSTALL=0
ASSUME_YES=0

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

# ─── Parse args ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)         TARGET="${2:?--target requires a value}"; shift 2 ;;
        --target=*)       TARGET="${1#*=}"; shift ;;
        --non-interactive) INTERACTIVE=0; shift ;;
        --config)         CONFIG_FILE="${2:?--config requires a path}"; shift 2 ;;
        --config=*)       CONFIG_FILE="${1#*=}"; shift ;;
        --skip-prereqs)   SKIP_PREREQS=1; shift ;;
        --no-build)       NO_BUILD=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --verify)         ONLY_VERIFY=1; shift ;;
        --uninstall)      UNINSTALL=1; shift ;;
        --yes|-y)         ASSUME_YES=1; shift ;;
        --help|-h)        usage ;;
        *)                err "Unknown option: $1 (try --help)"; exit 1 ;;
    esac
done

export INTERACTIVE CONFIG_FILE SKIP_PREREQS NO_BUILD DRY_RUN ASSUME_YES

# ─── Banner ────────────────────────────────────────────────────
banner() {
    cat <<'EOF'

  ╔═══════════════════════════════════════════════════════════╗
  ║              BotArmy — Automated Installer                ║
  ║   CrewAI multi-agent system · Mem0 · pgvector · Neo4j     ║
  ╚═══════════════════════════════════════════════════════════╝

EOF
}
banner

# ─── Branch on operation mode ──────────────────────────────────
if [[ "$ONLY_VERIFY" == "1" ]]; then
    info "Mode: verify only"
    source "${LIB_DIR}/verify.sh"
    run_health_checks
    exit $?
fi

if [[ "$UNINSTALL" == "1" ]]; then
    info "Mode: uninstall"
    source "${LIB_DIR}/uninstall.sh"
    run_uninstall
    exit $?
fi

# ─── Detect platform ───────────────────────────────────────────
detect_platform   # exports OS_FAMILY, OS_PRETTY, ARCH, PKG_MGR
info "Detected: ${OS_PRETTY} (${ARCH})"

# ─── Dispatch to target ────────────────────────────────────────
case "$TARGET" in
    local)
        info "Target: local docker-compose stack"
        source "${LIB_DIR}/local.sh"
        run_local_install
        ;;
    k8s|kubernetes)
        info "Target: Kubernetes (current kubectl context)"
        source "${LIB_DIR}/k8s.sh"
        run_k8s_install
        ;;
    aws)
        info "Target: AWS (EKS + RDS + ECR + ALB)"
        source "${LIB_DIR}/aws.sh"
        run_aws_install
        ;;
    gcp)
        info "Target: GCP (GKE Autopilot + Cloud SQL + Artifact Registry)"
        source "${LIB_DIR}/gcp.sh"
        run_gcp_install
        ;;
    *)
        err "Unknown --target: $TARGET (expected: local, k8s, aws, gcp)"
        exit 1
        ;;
esac

success "Done. Run './install.sh --verify' to re-check health any time."
