#!/usr/bin/env bash
# aws.sh — Phase 3 AWS deployment.
#
# Flow:
#   1. Validate prereqs: aws cli, terraform, docker, kubectl, helm.
#   2. Validate AWS auth + show the account ID we're about to bill.
#   3. Read tfvars (default: deploy/terraform/aws/terraform.tfvars).
#   4. terraform init / plan / apply (with confirmation gate).
#   5. Update local kubeconfig.
#   6. Login to ECR, build the gateway image, push it.
#   7. Bounce the gateway Deployment so it picks up the new image.
#   8. Verify with kubectl rollout status.
#
# Honors --dry-run, --yes, --non-interactive, --skip-prereqs, --no-build, --config.

TF_DIR="${INSTALL_ROOT}/deploy/terraform/aws"

run_aws_install() {
    step "AWS deployment (Phase 3)"

    # ── Prereq tools ───────────────────────────────────────────
    if [[ "${SKIP_PREREQS:-0}" == "1" ]]; then
        info "Skipping prereq install (--skip-prereqs)"
    fi
    require_cmd aws       "Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    require_cmd terraform "Install: https://developer.hashicorp.com/terraform/install"
    require_cmd kubectl   "Install: https://kubernetes.io/docs/tasks/tools/"
    require_cmd helm      "Install: https://helm.sh/docs/intro/install/"
    require_cmd docker    "Docker is needed to build + push the gateway image."

    # ── AWS auth ──────────────────────────────────────────────
    local aws_account aws_arn aws_region
    if ! aws_account="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"; then
        err "AWS credentials not configured. Run 'aws configure' or export AWS_ACCESS_KEY_ID/SECRET."
        exit 2
    fi
    aws_arn="$(aws sts get-caller-identity --query Arn --output text)"
    aws_region="$(aws configure get region 2>/dev/null || echo eu-north-1)"
    info "AWS account: $aws_account"
    info "AWS principal: $aws_arn"

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
        warn "No terraform.tfvars found — generating from .env + your AWS profile."
        tfvars_file="$(_seed_tfvars_from_env)"
    fi

    # ── Cost preview ──────────────────────────────────────────
    _show_cost_estimate "$tfvars_file"
    if ! confirm "Provision AWS resources now? Real money will be spent." "n"; then
        err "Aborted."
        exit 1
    fi

    # ── terraform init / plan / apply ─────────────────────────
    local tf_args=(-var-file="$tfvars_file")

    info "Running: terraform init"
    run_in_dir "$TF_DIR" terraform init -upgrade

    info "Running: terraform plan"
    run_in_dir "$TF_DIR" terraform plan "${tf_args[@]}" -out=botarmy.tfplan

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        info "[dry-run] would run: terraform apply botarmy.tfplan"
        info "[dry-run] stopping before any AWS resources are created."
        return 0
    fi

    info "Running: terraform apply  (this takes 15–25 minutes for EKS + RDS)"
    run_in_dir "$TF_DIR" terraform apply botarmy.tfplan

    # ── Capture outputs ───────────────────────────────────────
    local cluster_name ecr_repo region
    cluster_name="$(terraform -chdir="$TF_DIR" output -raw cluster_name)"
    ecr_repo="$(terraform -chdir="$TF_DIR" output -raw ecr_repository_url)"
    region="$(terraform -chdir="$TF_DIR" output -raw cluster_region)"

    success "Cluster up: $cluster_name in $region"

    # ── kubeconfig ────────────────────────────────────────────
    info "Updating local kubeconfig"
    run aws eks update-kubeconfig --region "$region" --name "$cluster_name"

    # ── Build + push the gateway image ────────────────────────
    if [[ "${NO_BUILD:-0}" == "1" ]]; then
        info "Skipping image build (--no-build) — make sure $ecr_repo:latest exists."
    else
        _build_and_push_image "$ecr_repo" "$region"
    fi

    # ── Roll the gateway to pick up the new image ─────────────
    local ns; ns="$(terraform -chdir="$TF_DIR" output -raw namespace)"
    info "Restarting gateway pods to pull the freshly pushed image"
    run kubectl -n "$ns" rollout restart deployment -l app.kubernetes.io/component=gateway || true
    run kubectl -n "$ns" rollout status   deployment -l app.kubernetes.io/component=gateway --timeout=5m || \
        warn "Gateway rollout didn't complete in 5 min — check 'kubectl -n $ns get pods'."

    # ── Final summary ─────────────────────────────────────────
    _print_aws_summary "$cluster_name" "$ns"
}

# ─── Helpers ───────────────────────────────────────────────────
run_in_dir() {
    local dir="$1"; shift
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf "%s[dry-run]%s (cd %s && %s)\n" "$C_DIM" "$C_RESET" "$dir" "$*"
        return 0
    fi
    ( cd "$dir" && "$@" )
}

_seed_tfvars_from_env() {
    # Build a minimal terraform.tfvars from .env + a couple of region prompts.
    # Returns the file path on stdout.
    local env_file="${INSTALL_ROOT}/.env"
    local out="${TF_DIR}/terraform.auto.tfvars"

    local region; region="$(aws configure get region 2>/dev/null || echo eu-north-1)"
    local cluster_name="botarmy"
    local tier="cheapest"

    if [[ "${INTERACTIVE:-1}" == "1" ]]; then
        read -r -p "AWS region [${region}]: " r; region="${r:-$region}"
        read -r -p "Cluster name [${cluster_name}]: " r; cluster_name="${r:-$cluster_name}"
        read -r -p "Tier (cheapest/prod) [${tier}]: " r; tier="${r:-$tier}"
    fi

    {
        printf 'region       = %q\n' "$region"
        printf 'cluster_name = %q\n' "$cluster_name"
        printf 'tier         = %q\n' "$tier"
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

_show_cost_estimate() {
    local tfvars="$1"
    local tier="cheapest"
    if grep -qE '^\s*tier\s*=\s*"prod"' "$tfvars" 2>/dev/null; then
        tier="prod"
    fi
    cat <<EOF

  ${C_BOLD}Estimated monthly AWS spend (${tier} tier, eu-north-1)${C_RESET}
EOF
    if [[ "$tier" == "prod" ]]; then
        cat <<'EOF'
    EKS control plane              ~$73
    3× m5.large nodes              ~$210
    db.m5.large multi-AZ RDS       ~$260
    1× ALB (internet-facing)        ~$22
    NAT GW × 3 + data              ~$100
    EBS gp3 storage (~150 GiB)      ~$15
    ─────────────────────────────────────
    Total (very rough)              ~$680 / month
EOF
    else
        cat <<'EOF'
    EKS control plane              ~$73
    2× t3.medium nodes              ~$60
    db.t4g.micro RDS                ~$13
    1× ALB (internet-facing)        ~$22
    NAT GW × 1 + data              ~$35
    EBS gp3 storage (~70 GiB)        ~$7
    ─────────────────────────────────────
    Total (very rough)              ~$210 / month
EOF
    fi
    echo
    warn "These are ballparks — your actual bill depends on traffic, data egress, and how long the cluster runs. EKS clusters that are 'down' still bill the control plane."
    echo
}

_build_and_push_image() {
    local ecr_repo="$1" region="$2"
    local registry="${ecr_repo%/*}"
    local tag="${BOTARMY_IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"

    step "Building + pushing gateway image to ECR"
    info "Tag: ${tag}"

    info "Logging into ECR..."
    run sh -c "aws ecr get-login-password --region '$region' | docker login --username AWS --password-stdin '$registry'"

    info "Building image..."
    # Always build for amd64 — EKS managed node groups default to x86_64. Mac
    # users on Apple Silicon need --platform to avoid pushing arm64 images that
    # won't run on the nodes.
    run docker buildx build --platform linux/amd64 \
        -t "${ecr_repo}:${tag}" \
        -t "${ecr_repo}:latest" \
        --push \
        "$INSTALL_ROOT"

    success "Pushed: ${ecr_repo}:${tag}"
    success "Pushed: ${ecr_repo}:latest"
}

_print_aws_summary() {
    local cluster_name="$1" ns="$2"
    cat <<EOF

  ${C_GREEN}${C_BOLD}✓ BotArmy is deployed to AWS.${C_RESET}

  Cluster:    $cluster_name
  Namespace:  $ns
  kubeconfig: already set as your current context

  Inspect:
    kubectl -n $ns get pods,svc,ingress
    kubectl -n $ns logs -l app.kubernetes.io/component=gateway -f

  Access (port-forward):
    kubectl -n $ns port-forward svc/botarmy-${cluster_name}-gateway 8765:8765
    open http://localhost:8765

  Tear down (DESTRUCTIVE — wipes RDS + cluster):
    cd deploy/terraform/aws && terraform destroy

EOF
}
