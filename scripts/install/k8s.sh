#!/usr/bin/env bash
# k8s.sh — Phase 2 entry point. Delegates to the Helm chart at deploy/k8s.
# Currently a thin wrapper; the chart itself is the substance.

run_k8s_install() {
    step "Kubernetes deployment"

    require_cmd kubectl "Install from https://kubernetes.io/docs/tasks/tools/"
    require_cmd helm    "Install from https://helm.sh/docs/intro/install/"

    local chart_dir="${INSTALL_ROOT}/deploy/k8s"
    if [[ ! -f "${chart_dir}/Chart.yaml" ]]; then
        err "Helm chart not found at ${chart_dir}."
        err "K8s deployment is Phase 2 — see ${chart_dir}/README.md for status."
        exit 3
    fi

    # Show context so the user knows where they're deploying
    local ctx; ctx="$(kubectl config current-context 2>/dev/null || echo '<none>')"
    info "Current kubectl context: $ctx"

    if [[ "${INTERACTIVE:-1}" == "1" ]]; then
        if ! confirm "Deploy to context '$ctx'?" "y"; then
            err "Aborted."
            exit 1
        fi
    fi

    # Namespace
    local ns="${BOTARMY_NAMESPACE:-botarmy}"
    if ! kubectl get ns "$ns" >/dev/null 2>&1; then
        run kubectl create namespace "$ns"
    fi

    # Secrets — read from .env or $CONFIG_FILE
    local env_source="${CONFIG_FILE:-${INSTALL_ROOT}/.env}"
    if [[ ! -r "$env_source" ]]; then
        err "No .env or --config file found. Run './install.sh --target local' first to generate one,"
        err "or pass --config /path/to/secrets.env."
        exit 1
    fi

    info "Creating/updating Kubernetes secret 'botarmy-env' from $env_source"
    run kubectl -n "$ns" create secret generic botarmy-env \
        --from-env-file="$env_source" \
        --dry-run=client -o yaml | run kubectl apply -f -

    # Helm install / upgrade
    info "Deploying Helm chart..."
    run helm upgrade --install botarmy "$chart_dir" \
        --namespace "$ns" \
        --create-namespace \
        --wait \
        --timeout 10m

    success "Helm deployment complete."
    info "Inspect with:  kubectl -n $ns get pods,svc"
    warn "K8s mode skips: Signal-cli, host Ollama, docker-socket sandbox."
    warn "See deploy/k8s/README.md for what's still TODO in this chart."
}
