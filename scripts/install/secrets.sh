#!/usr/bin/env bash
# secrets.sh — handle .env initialisation:
#   1. Copy .env.example → .env if missing.
#   2. Auto-generate strong values for any secret still set to a placeholder
#      (GATEWAY_SECRET, MEM0_POSTGRES_PASSWORD, MEM0_NEO4J_PASSWORD).
#   3. Pull required API keys from $CONFIG_FILE (--non-interactive) OR prompt.
#   4. Merge any new keys that appeared in .env.example since last install.
#
# Required keys (installer fails if not provided):
#   ANTHROPIC_API_KEY    — vetting layer + premium model fallback
#   OPENROUTER_API_KEY   — frontier-model cascade (DeepSeek/MiniMax/Kimi/GLM)
#   BRAVE_API_KEY        — web search (free tier)
#
# Optional keys (warn if missing, but proceed):
#   APOLLO_API_KEY, PROXYCURL_API_KEY, SMITHERY_API_KEY, COMPOSIO_API_KEY
#
# Signal-cli config (SIGNAL_BOT_NUMBER, SIGNAL_OWNER_NUMBER) is host-only and
# only validated for --target=local. K8s mode skips it.

ENV_FILE="${INSTALL_ROOT}/.env"
ENV_TEMPLATE="${INSTALL_ROOT}/.env.example"

# Required for compose to start at all (compose `:?` syntax errors otherwise).
SECRETS_TO_GENERATE=(
    "GATEWAY_SECRET:64"
    "MEM0_POSTGRES_PASSWORD:32"
    "MEM0_NEO4J_PASSWORD:32"
)

# Required user-supplied keys. Empty value or placeholder = missing.
REQUIRED_API_KEYS=(
    "ANTHROPIC_API_KEY"
    "OPENROUTER_API_KEY"
    "BRAVE_API_KEY"
)

OPTIONAL_API_KEYS=(
    "APOLLO_API_KEY"
    "PROXYCURL_API_KEY"
    "SMITHERY_API_KEY"
    "COMPOSIO_API_KEY"
)

PLACEHOLDER_PATTERN='^(your_.*_here|generate_a_64_char_random_string_here|\+1XXXXXXXXXX|change_?me|)$'

# ─── Step 1: ensure .env exists ─────────────────────────────────
ensure_env_file() {
    step "Initialising .env"

    if [[ ! -f "$ENV_TEMPLATE" ]]; then
        err ".env.example missing at $ENV_TEMPLATE — repo is incomplete."
        exit 3
    fi

    if [[ ! -f "$ENV_FILE" ]]; then
        info "Creating .env from template."
        run cp "$ENV_TEMPLATE" "$ENV_FILE"
    else
        info ".env already exists — will update missing values only."
        merge_new_template_keys
    fi

    # The compose file requires these but the template doesn't list them — patch.
    for kv in "${SECRETS_TO_GENERATE[@]}"; do
        local key="${kv%%:*}"
        if ! grep -qE "^${key}=" "$ENV_FILE" 2>/dev/null; then
            run sh -c "printf '%s=\n' '$key' >> '$ENV_FILE'"
        fi
    done
}

merge_new_template_keys() {
    # If .env.example has keys not in .env, append them with their default value.
    # Never overwrites an existing key.
    local key value
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" =~ ^# ]] && continue
        if ! grep -qE "^${key}=" "$ENV_FILE" 2>/dev/null; then
            info "  + adding new template key: $key"
            run sh -c "printf '%s=%s\n' '$key' '$value' >> '$ENV_FILE'"
        fi
    done < <(grep -E '^[A-Z_][A-Z0-9_]*=' "$ENV_TEMPLATE")
}

# ─── Step 2: generate secrets ───────────────────────────────────
generate_secrets() {
    step "Generating secrets"
    local key len current
    for kv in "${SECRETS_TO_GENERATE[@]}"; do
        key="${kv%%:*}"
        len="${kv##*:}"
        current="$(env_get "$ENV_FILE" "$key")"
        if [[ -z "$current" || "$current" =~ $PLACEHOLDER_PATTERN ]]; then
            local secret; secret="$(gen_password "$len")"
            env_set "$ENV_FILE" "$key" "$secret"
            success "Generated $key (${len} chars)"
        else
            info "$key already set, keeping existing value"
        fi
    done
}

# ─── Step 3: collect API keys ───────────────────────────────────
collect_api_keys() {
    step "API keys"

    # Source the config file if provided (overrides .env template values)
    local cfg_loaded=0
    if [[ -n "${CONFIG_FILE:-}" ]]; then
        if [[ ! -r "$CONFIG_FILE" ]]; then
            err "Config file not readable: $CONFIG_FILE"
            exit 1
        fi
        info "Loading API keys from: $CONFIG_FILE"
        cfg_loaded=1
    fi

    local missing=()
    for key in "${REQUIRED_API_KEYS[@]}"; do
        local value
        value="$(_resolve_key "$key" "$cfg_loaded")"
        if [[ -z "$value" || "$value" =~ $PLACEHOLDER_PATTERN ]]; then
            missing+=("$key")
        else
            env_set "$ENV_FILE" "$key" "$value"
            success "$key set"
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        if [[ "${INTERACTIVE:-1}" == "0" ]]; then
            err "Missing required keys (--non-interactive mode): ${missing[*]}"
            err "Provide them in --config FILE or via environment variables."
            exit 1
        fi
        warn "Missing required API keys: ${missing[*]}"
        echo
        for key in "${missing[@]}"; do
            _prompt_for_key "$key"
        done
    fi

    # Optional keys — silent unless interactive
    if [[ "${INTERACTIVE:-1}" == "1" && "$cfg_loaded" == "0" ]]; then
        if confirm "Configure optional integrations (Apollo, Proxycurl, Smithery, Composio)?" "n"; then
            for key in "${OPTIONAL_API_KEYS[@]}"; do
                _prompt_for_key "$key" "optional"
            done
        fi
    else
        for key in "${OPTIONAL_API_KEYS[@]}"; do
            local value; value="$(_resolve_key "$key" "$cfg_loaded")"
            [[ -n "$value" ]] && env_set "$ENV_FILE" "$key" "$value"
        done
    fi
}

_resolve_key() {
    # Try, in order: --config file → environment variable → existing .env value.
    # Echo the resolved value (or empty).
    local key="$1" use_cfg="$2"
    if [[ "$use_cfg" == "1" ]]; then
        local v; v="$(env_get "$CONFIG_FILE" "$key")"
        [[ -n "$v" ]] && { echo "$v"; return; }
    fi
    if [[ -n "${!key:-}" ]]; then
        echo "${!key}"
        return
    fi
    env_get "$ENV_FILE" "$key"
}

_prompt_for_key() {
    local key="$1" optional="${2:-}"
    local hint="" url=""
    case "$key" in
        ANTHROPIC_API_KEY)   hint="Anthropic API key (vetting + Opus fallback)";  url="https://console.anthropic.com/settings/keys" ;;
        OPENROUTER_API_KEY)  hint="OpenRouter API key (frontier-model cascade)";  url="https://openrouter.ai/keys" ;;
        BRAVE_API_KEY)       hint="Brave Search API key (free tier available)";   url="https://api.search.brave.com/" ;;
        APOLLO_API_KEY)      hint="Apollo.io API key (B2B contacts, optional)";   url="https://apollo.io/settings/api" ;;
        PROXYCURL_API_KEY)   hint="Proxycurl API key (LinkedIn lookup, optional)"; url="https://nubela.co/proxycurl/api-keys" ;;
        SMITHERY_API_KEY)    hint="Smithery API key (MCP server registry)";       url="https://smithery.ai/" ;;
        COMPOSIO_API_KEY)    hint="Composio API key (SaaS integrations)";         url="https://composio.dev/" ;;
        *)                   hint="$key" ;;
    esac
    printf "  %s%s%s\n" "$C_BOLD" "$hint" "$C_RESET"
    [[ -n "$url" ]] && printf "    %s%s%s\n" "$C_DIM" "$url" "$C_RESET"
    local value
    if [[ "$optional" == "optional" ]]; then
        read -r -p "    $key (leave blank to skip): " value
    else
        read -r -p "    $key: " value
    fi
    if [[ -n "$value" ]]; then
        env_set "$ENV_FILE" "$key" "$value"
    fi
}

# ─── Step 4: target-specific validation ─────────────────────────
validate_target_specific() {
    case "$TARGET" in
        local)
            # Signal-cli is host-only and OS-specific — warn, don't fail.
            local sig_bot; sig_bot="$(env_get "$ENV_FILE" "SIGNAL_BOT_NUMBER")"
            if [[ "$sig_bot" =~ \+1XXXXXXXXXX || -z "$sig_bot" ]]; then
                warn "SIGNAL_BOT_NUMBER not configured — Signal interface will be disabled."
                warn "Configure it later in .env if you want to chat via Signal."
            fi

            # SIGNAL_ATTACHMENT_PATH default is Mac-specific
            if [[ "$OS_FAMILY" == "linux" ]]; then
                local sap; sap="$(env_get "$ENV_FILE" "SIGNAL_ATTACHMENT_PATH")"
                if [[ "$sap" == */YOUR_USER/* ]]; then
                    local linux_default="${HOME}/.local/share/signal-cli/attachments"
                    info "Patching SIGNAL_ATTACHMENT_PATH for Linux: $linux_default"
                    env_set "$ENV_FILE" "SIGNAL_ATTACHMENT_PATH" "$linux_default"
                fi
            fi

            # SEC_EDGAR_USER_AGENT requires real email
            local sec_ua; sec_ua="$(env_get "$ENV_FILE" "SEC_EDGAR_USER_AGENT")"
            if [[ "$sec_ua" == *"your@email.com"* ]]; then
                warn "SEC_EDGAR_USER_AGENT contains placeholder email — financial filing tools will be rate-limited."
            fi
            ;;
        k8s|kubernetes)
            # In k8s, host integrations are out — make that explicit.
            info "K8s mode: Signal-cli, host Ollama, and docker-socket sandbox are skipped."
            ;;
    esac
}

# ─── Top-level entry ────────────────────────────────────────────
configure_secrets_and_keys() {
    ensure_env_file
    generate_secrets
    collect_api_keys
    validate_target_specific
    # Tighten perms — .env now contains real secrets
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        chmod 600 "$ENV_FILE" 2>/dev/null || true
    fi
}
