#!/usr/bin/env bash
# uninstall.sh — tear down the stack. Default: stop containers, keep data.
# Pass --yes --force-destructive to also wipe ./workspace/{mem0_pgdata,mem0_neo4j,memory}.

run_uninstall() {
    cd "$INSTALL_ROOT" || return 1
    step "Stopping containers"

    if has docker && docker compose ps -q 2>/dev/null | grep -q .; then
        run docker compose down
        success "Containers stopped."
    else
        info "No running containers."
    fi

    # ── Image removal (optional) ───────────────────────────────
    if confirm "Remove built images (gateway, crewai-sandbox)?" "n"; then
        run docker image rm crewai-team-gateway crewai-sandbox:latest 2>/dev/null || true
    fi

    # ── Data wipe (destructive!) ───────────────────────────────
    echo
    warn "Persistent data lives in:"
    warn "  workspace/mem0_pgdata/   (Postgres + pgvector)"
    warn "  workspace/mem0_neo4j/    (Neo4j entity graph)"
    warn "  workspace/memory/        (ChromaDB RAG)"
    warn "  workspace/crewai_storage/ (CrewAI long-term memory)"
    echo

    if require_explicit_confirm "Wipe ALL persistent memory? This is irreversible."; then
        for d in mem0_pgdata mem0_neo4j memory crewai_storage; do
            if [[ -d "workspace/$d" ]]; then
                run rm -rf "workspace/$d"
                info "  removed workspace/$d"
            fi
        done
        success "Persistent data removed."
    else
        info "Data preserved. Re-run ./install.sh to restart with the same memory."
    fi

    # ── .env handling ──────────────────────────────────────────
    if [[ -f .env ]] && require_explicit_confirm "Remove .env (contains API keys + generated secrets)?"; then
        run rm -f .env
        info "  removed .env"
    fi
}
