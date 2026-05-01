#!/usr/bin/env bash
# gcp.sh — Phase 3 GCP deployment.
#
# Mirrors aws.sh: validate prereqs → check auth → tfvars → plan/apply →
# kubeconfig → build+push image to Artifact Registry → roll gateway → done.

TF_DIR="${INSTALL_ROOT}/deploy/terraform/gcp"

run_gcp_install() {
    step "GCP deployment (Phase 3)"

    require_cmd gcloud    "Install: https://cloud.google.com/sdk/docs/install"
    require_cmd terraform "Install: https://developer.hashicorp.com/terraform/install"
    require_cmd kubectl   "Install: https://kubernetes.io/docs/tasks/tools/"
    require_cmd helm      "Install: https://helm.sh/docs/intro/install/"
    require_cmd docker    "Docker is needed to build + push the gateway image."

    # ── GCP auth ──────────────────────────────────────────────
    local active_account active_project
    if ! active_account="$(gcloud config get-value account 2>/dev/null)" || [[ -z "$active_account" || "$active_account" == "(unset)" ]]; then
        err "No active gcloud account. Run: gcloud auth login"
        exit 2
    fi
    if ! active_project="$(gcloud config get-value project 2>/dev/null)" || [[ -z "$active_project" || "$active_project" == "(unset)" ]]; then
        warn "No default project set. Will read from terraform.tfvars."
        active_project=""
    fi
    info "GCP account: $active_account"
    [[ -n "$active_project" ]] && info "GCP project (default): $active_project"

    # ADC for Terraform — required because the google provider uses ADC,
    # not the gcloud config directly.
    if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
        warn "Application-Default Credentials not set."
        if [[ "${INTERACTIVE:-1}" == "1" ]]; then
            info "Running: gcloud auth application-default login"
            run gcloud auth application-default login
        else
            err "ADC required for Terraform. Run: gcloud auth application-default login"
            exit 2
        fi
    fi

    # ── tfvars setup ──────────────────────────────────────────
    local tfvars_file
    if [[ -n "${CONFIG_FILE:-}" ]]; then
        tfvars_file="$CONFIG_FILE"
        info "Using tfvars: $tfvars_file"
    elif [[ -f "${TF_DIR}/terraform.tfvars" ]]; then
        tfvars_file="${TF_DIR}/terraform.tfvars"
        info "Using tfvars: $tfvars_file"
    else
        if [[ "${INTERACTIVE:-1}" == "0" ]]; then
            err "No terraform.tfvars and --non-interactive. Provide one via --config FILE."
            exit 1
        fi
        warn "No terraform.tfvars found — generating from .env + your gcloud config."
        tfvars_file="$(_seed_gcp_tfvars "$active_project")"
    fi

    # ── Cost preview ──────────────────────────────────────────
    _show_gcp_cost_estimate "$tfvars_file"
    if ! confirm "Provision GCP resources now? Real money will be spent." "n"; then
        err "Aborted."
        exit 1
    fi

    # ── terraform ─────────────────────────────────────────────
    local tf_args=(-var-file="$tfvars_file")

    info "Running: terraform init"
    run_in_dir "$TF_DIR" terraform init -upgrade

    info "Running: terraform plan"
    run_in_dir "$TF_DIR" terraform plan "${tf_args[@]}" -out=botarmy.tfplan

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        info "[dry-run] would run: terraform apply botarmy.tfplan"
        return 0
    fi

    info "Running: terraform apply  (15–25 minutes for GKE + Cloud SQL)"
    run_in_dir "$TF_DIR" terraform apply botarmy.tfplan

    # ── outputs + kubeconfig + image push ─────────────────────
    local cluster_name location project ar_url ns
    cluster_name="$(terraform -chdir="$TF_DIR" output -raw cluster_name)"
    location="$(terraform -chdir="$TF_DIR" output -raw cluster_location)"
    project="$(terraform -chdir="$TF_DIR" output -raw project_id)"
    ar_url="$(terraform -chdir="$TF_DIR" output -raw artifact_registry_url)"
    ns="$(terraform -chdir="$TF_DIR" output -raw namespace)"

    success "Cluster up: $cluster_name in $location ($project)"

    info "Updating local kubeconfig"
    run gcloud container clusters get-credentials "$cluster_name" --location "$location" --project "$project"

    if [[ "${NO_BUILD:-0}" == "1" ]]; then
        info "Skipping image build (--no-build) — make sure ${ar_url}/gateway:latest exists."
    else
        _build_and_push_gcp_image "$ar_url"
    fi

    info "Restarting gateway pods to pull the freshly pushed image"
    run kubectl -n "$ns" rollout restart deployment -l app.kubernetes.io/component=gateway || true
    run kubectl -n "$ns" rollout status   deployment -l app.kubernetes.io/component=gateway --timeout=5m || \
        warn "Gateway rollout didn't complete in 5 min — check 'kubectl -n $ns get pods'."

    _print_gcp_summary "$cluster_name" "$ns"
}

# ─── Helpers ───────────────────────────────────────────────────
_seed_gcp_tfvars() {
    local project="$1"
    local env_file="${INSTALL_ROOT}/.env"
    local out="${TF_DIR}/terraform.auto.tfvars"

    if [[ -z "$project" && "${INTERACTIVE:-1}" == "1" ]]; then
        read -r -p "GCP project_id: " project
    fi
    if [[ -z "$project" ]]; then
        err "GCP project_id is required."
        exit 1
    fi

    local region="europe-north1"
    local cluster_name="botarmy"
    local tier="cheapest"
    local enable_monitoring="true"

    if [[ "${INTERACTIVE:-1}" == "1" ]]; then
        read -r -p "GCP region [${region}]: " r;             region="${r:-$region}"
        read -r -p "Cluster name [${cluster_name}]: " r;     cluster_name="${r:-$cluster_name}"
        read -r -p "Tier (cheapest/prod) [${tier}]: " r;     tier="${r:-$tier}"
        read -r -p "Enable monitoring (true/false) [${enable_monitoring}]: " r; enable_monitoring="${r:-$enable_monitoring}"
    fi

    {
        printf 'project_id        = %q\n'     "$project"
        printf 'region            = %q\n'     "$region"
        printf 'zone              = %q\n'     "${region}-a"
        printf 'cluster_name      = %q\n'     "$cluster_name"
        printf 'tier              = %q\n'     "$tier"
        printf 'enable_monitoring = %s\n'     "$enable_monitoring"
        printf 'extra_env = {\n'
        if [[ -r "$env_file" ]]; then
            local k v
            for k in ANTHROPIC_API_KEY OPENROUTER_API_KEY BRAVE_API_KEY \
                     APOLLO_API_KEY PROXYCURL_API_KEY SMITHERY_API_KEY \
                     COMPOSIO_API_KEY COMMANDER_MODEL SPECIALIST_MODEL \
                     LLM_MODE COST_MODE VETTING_MODEL; do
                v="$(env_get "$env_file" "$k")"
                if [[ -n "$v" && ! "$v" =~ ^(your_.*_here|generate_a_.*_here|\+1XXXXXXXXXX)$ ]]; then
                    printf '  %-20s = %q\n' "$k" "$v"
                fi
            done
        fi
        printf '}\n'
    } > "$out"

    info "Wrote $out"
    echo "$out"
}

_show_gcp_cost_estimate() {
    local tfvars="$1"
    local tier="cheapest"
    if grep -qE '^\s*tier\s*=\s*"prod"' "$tfvars" 2>/dev/null; then
        tier="prod"
    fi
    cat <<EOF

  ${C_BOLD}Estimated monthly GCP spend (${tier} tier, europe-north1)${C_RESET}
EOF
    if [[ "$tier" == "prod" ]]; then
        cat <<'EOF'
    Autopilot (regional control plane)  ~$73
    Autopilot pod compute (HA replicas)  ~$140
    db-custom-2-7680 HA Cloud SQL        ~$140
    1× LB + regional NAT                  ~$60
    PD SSD storage (~150 GiB)             ~$25
    ──────────────────────────────────────────
    Total (very rough)                   ~$440 / month
EOF
    else
        cat <<'EOF'
    Autopilot zonal control plane        $0
    Autopilot pod compute (BotArmy)      ~$60
    db-g1-small Cloud SQL                ~$25
    1× LB + 1× NAT                        ~$30
    PD SSD storage (~30 GiB)              ~$5
    ──────────────────────────────────────────
    Total (very rough)                   ~$120 / month
EOF
    fi
    echo
    warn "Egress to Anthropic / OpenRouter etc. bills separately at ~\$0.12/GB and can surprise you."
    echo
}

_build_and_push_gcp_image() {
    local ar_url="$1"
    local registry="${ar_url%%/*}"
    local tag="${BOTARMY_IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"

    step "Building + pushing gateway image to Artifact Registry"
    info "Tag: ${tag}"

    info "Configuring docker auth for $registry"
    run gcloud auth configure-docker "$registry" --quiet

    info "Building image (linux/amd64 — Autopilot uses x86_64 nodes)"
    run docker buildx build --platform linux/amd64 \
        -t "${ar_url}/gateway:${tag}" \
        -t "${ar_url}/gateway:latest" \
        --push \
        "$INSTALL_ROOT"

    success "Pushed: ${ar_url}/gateway:${tag}"
    success "Pushed: ${ar_url}/gateway:latest"
}

_print_gcp_summary() {
    local cluster_name="$1" ns="$2"
    cat <<EOF

  ${C_GREEN}${C_BOLD}✓ BotArmy is deployed to GCP.${C_RESET}

  Cluster:    $cluster_name
  Namespace:  $ns
  kubeconfig: already set as your current context

  Inspect:
    kubectl -n $ns get pods,svc,ingress
    kubectl -n $ns logs -l app.kubernetes.io/component=gateway -f

  Access (port-forward):
    kubectl -n $ns port-forward svc/botarmy-${cluster_name}-gateway 8765:8765
    open http://localhost:8765

  Tear down (DESTRUCTIVE — wipes Cloud SQL + cluster):
    cd deploy/terraform/gcp && terraform destroy

EOF
}
