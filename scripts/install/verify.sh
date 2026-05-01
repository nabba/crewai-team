#!/usr/bin/env bash
# verify.sh — health checks for the running stack.
# Returns 0 if everything is healthy, non-zero otherwise.

run_health_checks() {
    cd "$INSTALL_ROOT" || return 1
    step "Running health checks"

    local failures=0
    local gateway_port; gateway_port="$(env_get "${INSTALL_ROOT}/.env" GATEWAY_PORT || echo 8765)"

    # ── 1. Compose stack: all services up? ─────────────────────
    if ! docker compose ps --format '{{.Service}} {{.State}}' 2>/dev/null | grep -qE '(running|healthy)'; then
        err "No running compose services — did you run install?"
        return 1
    fi

    # ── 2. Per-service ping ────────────────────────────────────
    _check_service "docker-proxy" \
        "docker compose exec -T docker-proxy nc -z localhost 2375" 30 || failures=$((failures+1))

    _check_service "postgres (mem0)" \
        "docker compose exec -T postgres pg_isready -U mem0 -d mem0 -q" 60 || failures=$((failures+1))

    _check_service "neo4j" \
        "docker compose exec -T neo4j wget --quiet --spider http://localhost:7474" 60 || failures=$((failures+1))

    _check_service "chromadb" \
        "docker compose exec -T chromadb /bin/sh -c 'curl -sf http://localhost:8000/api/v1/heartbeat || curl -sf http://localhost:8000/api/v2/heartbeat'" 30 \
        || failures=$((failures+1))

    _check_service "gateway HTTP" \
        "curl -sf http://127.0.0.1:${gateway_port}/health" 60 \
        || _check_service "gateway HTTP (root)" "curl -sf http://127.0.0.1:${gateway_port}/" 30 \
        || failures=$((failures+1))

    # ── 3. Sandbox image present ───────────────────────────────
    if docker image inspect crewai-sandbox:latest >/dev/null 2>&1; then
        success "sandbox image: crewai-sandbox:latest present"
    else
        warn "sandbox image missing — code-execution tool will fail"
        failures=$((failures+1))
    fi

    # ── Summary ────────────────────────────────────────────────
    echo
    if [[ "$failures" -eq 0 ]]; then
        success "All health checks passed."
        return 0
    else
        err "$failures health check(s) failed."
        warn "Inspect with:  docker compose logs --tail=80"
        return 1
    fi
}

_check_service() {
    local name="$1" cmd="$2" timeout="${3:-30}"
    local elapsed=0 interval=2
    printf "  %s[?]%s %s " "$C_BLUE" "$C_RESET" "$name"
    while (( elapsed < timeout )); do
        if eval "$cmd" >/dev/null 2>&1; then
            printf "%s✓%s\n" "$C_GREEN" "$C_RESET"
            return 0
        fi
        printf "."
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    printf " %s✗ (timeout after ${timeout}s)%s\n" "$C_RED" "$C_RESET"
    return 1
}
