#!/usr/bin/env bash
# prereqs.sh — ensure Docker, Compose, Python 3.11+ are present.
# Sourced by local.sh. Honours --skip-prereqs and --dry-run.

# ─── Docker ─────────────────────────────────────────────────────
ensure_docker() {
    if has docker && docker info >/dev/null 2>&1; then
        local v; v="$(docker --version | sed -E 's/.*version ([^,]+),.*/\1/')"
        success "Docker present: $v"
        return 0
    fi

    if has docker && ! docker info >/dev/null 2>&1; then
        warn "Docker is installed but the daemon is not reachable."
        case "$OS_FAMILY" in
            mac)
                warn "Start Docker Desktop, then re-run."
                ;;
            linux)
                warn "Try: sudo systemctl start docker  (and: sudo usermod -aG docker \$USER)"
                ;;
        esac
        exit 2
    fi

    if [[ "${SKIP_PREREQS:-0}" == "1" ]]; then
        err "Docker not found and --skip-prereqs was passed."
        exit 2
    fi

    info "Docker not found. Attempting install for $OS_FAMILY ($PKG_MGR)..."
    case "$OS_FAMILY" in
        mac)
            if ! has brew; then
                err "Homebrew is required to auto-install Docker on macOS. Install from https://brew.sh first."
                exit 2
            fi
            run brew install --cask docker
            warn "Docker Desktop installed. Open it once to finish setup, then re-run this installer."
            exit 0
            ;;
        linux)
            install_docker_linux
            ;;
        *)
            err "Cannot auto-install Docker on $OS_FAMILY. Install manually and re-run."
            exit 2
            ;;
    esac
}

install_docker_linux() {
    # Use Docker's convenience script — it handles all major distros and pulls
    # the right repo. Not perfect for production hardening, fine for dev/lab.
    if ! has curl; then
        pkg_install curl || exit 2
    fi
    info "Running Docker's official install script (https://get.docker.com)..."
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf "[dry-run] would run: curl -fsSL https://get.docker.com | sh\n"
    else
        curl -fsSL https://get.docker.com | sudo_run sh
    fi

    # Add current user to docker group so we don't need sudo for compose
    if [[ "$EUID" -ne 0 ]]; then
        warn "Adding $USER to the 'docker' group (you'll need to log out/in for it to take effect)."
        sudo_run usermod -aG docker "$USER" || true
    fi

    # Start the daemon
    if has systemctl; then
        sudo_run systemctl enable --now docker || true
    fi

    if ! docker info >/dev/null 2>&1; then
        warn "Docker is installed but you may need to log out + back in for group membership to apply."
        warn "Or run the rest of this installer with sudo."
    fi
}

# ─── Docker Compose plugin ──────────────────────────────────────
ensure_compose() {
    if docker compose version >/dev/null 2>&1; then
        local v; v="$(docker compose version --short 2>/dev/null || echo "?")"
        success "Docker Compose plugin present: v$v"
        return 0
    fi

    if has docker-compose; then
        warn "Found legacy 'docker-compose' (v1) but not the v2 plugin. Upgrading recommended."
    fi

    info "Docker Compose v2 plugin missing. Attempting install..."
    case "$OS_FAMILY" in
        mac)
            # Comes with Docker Desktop; if missing, Desktop install is broken.
            err "Docker Compose should ship with Docker Desktop. Reinstall Docker Desktop."
            exit 2
            ;;
        linux)
            case "$PKG_MGR" in
                apt) sudo_run apt-get install -y docker-compose-plugin ;;
                dnf) sudo_run dnf install -y docker-compose-plugin ;;
                *)   warn "Install docker-compose-plugin manually for $PKG_MGR." ;;
            esac
            ;;
    esac

    if ! docker compose version >/dev/null 2>&1; then
        err "Docker Compose v2 still not available."
        exit 2
    fi
}

# ─── Python 3.11+ ───────────────────────────────────────────────
ensure_python() {
    # The container ships its own Python. The host only needs Python for:
    #   - sync_host_capacity.py (probe RAM, runs at install)
    #   - run_host.py (bare-metal mode, optional)
    #
    # We accept any Python 3.11+ on the host — strict version match isn't needed
    # since the host scripts have zero deps.

    local py; py="$(_pick_python)"
    if [[ -n "$py" ]]; then
        local v; v="$($py --version 2>&1 | awk '{print $2}')"
        success "Python present: $py ($v)"
        export HOST_PYTHON="$py"
        return 0
    fi

    if [[ "${SKIP_PREREQS:-0}" == "1" ]]; then
        err "Python 3.11+ not found and --skip-prereqs was passed."
        exit 2
    fi

    info "Python 3.11+ not found. Installing..."
    case "$OS_FAMILY" in
        mac)
            has brew || { err "Install Homebrew first: https://brew.sh"; exit 2; }
            run brew install python@3.11
            ;;
        linux)
            case "$PKG_MGR" in
                apt)    sudo_run apt-get install -y python3.11 python3.11-venv python3-pip || sudo_run apt-get install -y python3 python3-venv python3-pip ;;
                dnf)    sudo_run dnf install -y python3.11 python3-pip || sudo_run dnf install -y python3 python3-pip ;;
                pacman) sudo_run pacman -Sy --noconfirm python python-pip ;;
                apk)    sudo_run apk add --no-cache python3 py3-pip ;;
                *)      err "Install Python 3.11+ manually for $PKG_MGR."; exit 2 ;;
            esac
            ;;
    esac

    py="$(_pick_python)"
    if [[ -z "$py" ]]; then
        err "Python install reported success but no compatible interpreter found."
        exit 2
    fi
    export HOST_PYTHON="$py"
}

_pick_python() {
    # Echo the first Python 3.11+ on PATH, or empty.
    local candidates=(python3.13 python3.12 python3.11 python3 python)
    for c in "${candidates[@]}"; do
        if has "$c"; then
            local ver; ver="$($c -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
            local major minor
            major="${ver%%.*}"; minor="${ver#*.}"
            if [[ "$major" -gt 3 ]] || [[ "$major" -eq 3 && "$minor" -ge 11 ]]; then
                echo "$c"
                return 0
            fi
        fi
    done
}

# ─── Optional: openssl (for secret generation) ──────────────────
ensure_openssl() {
    if has openssl; then return 0; fi
    info "Installing openssl (used for secret generation)..."
    case "$OS_FAMILY" in
        mac)   has brew && run brew install openssl@3 || warn "Install openssl manually." ;;
        linux) pkg_install openssl || warn "Install openssl manually." ;;
    esac
    # Not fatal — gen_secret has /dev/urandom fallback.
}

# ─── Top-level: run all prereq checks ───────────────────────────
ensure_all_prereqs() {
    step "Checking prerequisites"
    ensure_openssl       # quick, non-fatal
    ensure_docker
    ensure_compose
    ensure_python
}
