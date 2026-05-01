#!/usr/bin/env bash
# k8s.sh — Phase 2 entry point. Deploys the BotArmy Helm chart at
# deploy/k8s/ to whatever cluster your kubectl context is pointing at.
#
# Unlike the AWS / GCP dispatchers, this one does NOT build + push a
# container image — that's your responsibility before running. The
# chart's gateway Deployment references whatever you set in
# `image.repository` + `image.tag`. See "Image prereq" in
# deploy/k8s/README.md.

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

    # ── Image existence check ──────────────────────────────────
    # The dispatcher does not push images — but if the user hasn't pre-
    # pushed one, the deploy will sit in ImagePullBackOff. Sanity-check
    # the values they're about to use and warn early.
    local image_repo="${BOTARMY_IMAGE_REPOSITORY:-}"
    local image_tag="${BOTARMY_IMAGE_TAG:-}"
    if [[ -z "$image_repo" ]]; then
        warn "BOTARMY_IMAGE_REPOSITORY not set — chart will use the default 'botarmy/gateway'."
        warn "If you haven't pushed an image to that repo, the gateway pod will fail to pull."
        warn "Set BOTARMY_IMAGE_REPOSITORY and BOTARMY_IMAGE_TAG in your shell to override."
    else
        info "Will set image.repository=$image_repo image.tag=${image_tag:-latest}"
    fi

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

    # ── Helm install / upgrade ─────────────────────────────────
    # We deliberately do NOT pass --wait. The gateway pod can't be Ready
    # until its image is pulled, and many users push their image AFTER
    # this command (or the image lives on a private registry that takes
    # a moment to authenticate). The rollout-status check below is the
    # right readiness gate — it can be re-run if a pull lags.
    #
    # Verified in the GCP e2e test on 2026-05-01: with --wait, the helm
    # release timed out at 5m on ImagePullBackOff even though everything
    # else was healthy.
    info "Deploying Helm chart..."
    local helm_args=(
        upgrade --install botarmy "$chart_dir"
        --namespace "$ns"
        --create-namespace
        --timeout 5m
    )
    if [[ -n "$image_repo" ]]; then
        helm_args+=(--set "image.repository=$image_repo")
    fi
    if [[ -n "$image_tag" ]]; then
        helm_args+=(--set "image.tag=$image_tag")
    fi
    run helm "${helm_args[@]}"

    # ── Rollout status check (with image-push grace period) ───
    # Give the gateway up to 5 minutes to become Ready. If the user
    # is pushing the image in parallel, this gives them slack. After
    # 5 min we surface the actual pod status rather than blocking.
    info "Waiting up to 5 minutes for the gateway Deployment to become Ready..."
    if run kubectl -n "$ns" rollout status deployment \
            -l app.kubernetes.io/component=gateway \
            --timeout=5m; then
        success "Gateway is Ready."
    else
        warn "Gateway didn't become Ready in 5 min."
        warn "If you haven't pushed your gateway image yet, do that now:"
        warn "  docker buildx build --platform linux/amd64 -t \$IMAGE --push ."
        warn "Then re-run:"
        warn "  kubectl -n $ns rollout restart deployment -l app.kubernetes.io/component=gateway"
        warn "  kubectl -n $ns rollout status   deployment -l app.kubernetes.io/component=gateway"
        warn ""
        warn "Current state:"
        kubectl -n "$ns" get pods 2>&1 | head -10
    fi

    success "Helm deployment complete."
    info "Inspect with:  kubectl -n $ns get pods,svc,ingress"
    warn "K8s mode skips: Signal-cli, host Ollama, docker-socket sandbox."
    warn "See deploy/k8s/README.md for what's still TODO in this chart."
}
