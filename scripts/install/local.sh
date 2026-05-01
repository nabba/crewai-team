#!/usr/bin/env bash
# local.sh — Phase 1 install: prereqs → env → docker compose → verify.
# Runs on Mac (Intel/ARM) and Linux (apt/dnf/pacman/apk).

# shellcheck source=prereqs.sh
source "${LIB_DIR}/prereqs.sh"
# shellcheck source=secrets.sh
source "${LIB_DIR}/secrets.sh"
# shellcheck source=verify.sh
source "${LIB_DIR}/verify.sh"

run_local_install() {
    cd "$INSTALL_ROOT"

    # ── 1. Prereqs ─────────────────────────────────────────────
    if [[ "${SKIP_PREREQS:-0}" == "1" ]]; then
        info "Skipping prereq install (--skip-prereqs)"
        require_cmd docker "Install Docker Desktop or run without --skip-prereqs."
        require_cmd python3 "Install Python 3.11+ or run without --skip-prereqs."
        # Still need HOST_PYTHON for the capacity probe step.
        HOST_PYTHON="$(_pick_python)"
        export HOST_PYTHON
    else
        ensure_all_prereqs
    fi

    # ── 2. Workspace dirs ──────────────────────────────────────
    step "Preparing workspace directories"
    local dirs=(
        workspace/output
        workspace/memory
        workspace/skills
        workspace/proposals
        workspace/applied_code
        workspace/crewai_storage
        workspace/mem0_pgdata
        workspace/mem0_neo4j
        wiki
    )
    for d in "${dirs[@]}"; do
        run mkdir -p "$d"
    done

    if [[ ! -f workspace/skills/learning_queue.md ]]; then
        info "Seeding learning_queue.md"
        if [[ "${DRY_RUN:-0}" != "1" ]]; then
            cat > workspace/skills/learning_queue.md <<'EOF'
# Learning Queue — add one topic per line
CrewAI multi-agent patterns
Python async programming best practices
EOF
        fi
    fi

    # ── 3. Secrets + API keys ──────────────────────────────────
    configure_secrets_and_keys

    # ── 4. Detect host capacity (Ollama RAM budgeting) ─────────
    step "Probing host capacity"
    if [[ -x "$INSTALL_ROOT/scripts/sync_host_capacity.py" ]] && [[ -n "${HOST_PYTHON:-}" ]]; then
        run "$HOST_PYTHON" "$INSTALL_ROOT/scripts/sync_host_capacity.py" \
            || warn "Capacity probe failed — registry scanner will use a conservative fallback."
    else
        warn "sync_host_capacity.py not found or no Python — skipping."
    fi

    # ── 5. Build images ────────────────────────────────────────
    if [[ "${NO_BUILD:-0}" == "1" ]]; then
        info "Skipping image build (--no-build)"
    else
        step "Building Docker images"
        # Sandbox image (used by the sandbox executor for safe code-running)
        if [[ -d sandbox && -f sandbox/Dockerfile ]]; then
            info "Building sandbox image: crewai-sandbox:latest"
            run docker build -t crewai-sandbox:latest sandbox/
        else
            warn "sandbox/Dockerfile missing — sandbox tool will not work."
        fi

        # Gateway image (the main app)
        info "Building gateway image (this is the long step — 5–10 min on first run)"
        run docker compose build gateway
    fi

    # ── 6. Pull external images in parallel ────────────────────
    step "Pulling external images"
    run docker compose pull --ignore-buildable || warn "Some image pulls failed — will retry at start."

    # ── 7. Start the stack ─────────────────────────────────────
    step "Starting services"
    run docker compose up -d
    info "Containers started. Waiting for health checks to settle..."

    # ── 8. Verify ──────────────────────────────────────────────
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        info "[dry-run] would run health checks"
    else
        sleep 3
        run_health_checks || {
            err "Health checks failed. Inspect with:  docker compose logs --tail=50"
            exit 3
        }
    fi

    # ── 9. Post-install hints ──────────────────────────────────
    print_post_install_summary
}

print_post_install_summary() {
    cat <<EOF

  ${C_GREEN}${C_BOLD}✓ BotArmy is up.${C_RESET}

  Gateway:   http://127.0.0.1:$(env_get "$ENV_FILE" GATEWAY_PORT || echo 8765)
  Postgres:  internal only (mem0 / pgvector)
  Neo4j:     internal only (mem0 entity graph)
  ChromaDB:  internal only (RAG corpus)

  Next steps (optional):
    1. Signal interface:
         brew install signal-cli   # Mac
         signal-cli -u +YOUR_NUMBER register
         (See README.md → Signal section.)

    2. Local LLM tier:
         curl -fsSL https://ollama.com/install.sh | sh
         ollama pull qwen3:30b-a3b

    3. Public access (optional):
         tailscale up && tailscale serve --bg 8765

  Useful commands:
    ./install.sh --verify         # re-check health
    docker compose logs -f gateway
    docker compose down           # stop (data preserved in ./workspace/)
    ./install.sh --uninstall      # full teardown

EOF
}
