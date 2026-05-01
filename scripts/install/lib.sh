#!/usr/bin/env bash
# lib.sh — shared helpers for the BotArmy installer.
# Sourced by install.sh and the per-target scripts (local.sh, k8s.sh).
# Keep zero third-party dependencies — runs before pip/venv exist.

# ─── Colours ────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_DIM=$'\033[2m'
    C_BOLD=$'\033[1m'
else
    C_RESET="" C_RED="" C_GREEN="" C_YELLOW="" C_BLUE="" C_DIM="" C_BOLD=""
fi

info()    { printf "%s[i]%s %s\n" "$C_BLUE"   "$C_RESET" "$*"; }
success() { printf "%s[✓]%s %s\n" "$C_GREEN"  "$C_RESET" "$*"; }
warn()    { printf "%s[!]%s %s\n" "$C_YELLOW" "$C_RESET" "$*" >&2; }
err()     { printf "%s[x]%s %s\n" "$C_RED"    "$C_RESET" "$*" >&2; }
step()    { printf "\n%s%s== %s ==%s\n" "$C_BOLD" "$C_BLUE" "$*" "$C_RESET"; }
debug()   { [[ "${DEBUG:-0}" == "1" ]] && printf "%s[d] %s%s\n" "$C_DIM" "$*" "$C_RESET" >&2 || true; }

# ─── Run / dry-run wrapper ──────────────────────────────────────
# Use this for any state-changing command. Honors --dry-run.
run() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf "%s[dry-run]%s %s\n" "$C_DIM" "$C_RESET" "$*"
        return 0
    fi
    debug "+ $*"
    "$@"
}

# ─── Confirmation prompts ───────────────────────────────────────
confirm() {
    # confirm "Question?" [default-yn]
    local prompt="$1" default="${2:-y}" reply
    if [[ "${ASSUME_YES:-0}" == "1" ]]; then
        return 0
    fi
    if [[ "${INTERACTIVE:-1}" == "0" ]]; then
        # Non-interactive defaults to "yes" on confirmations (assumes user
        # accepted policy by passing --non-interactive). Destructive ops
        # should use require_explicit_confirm() instead.
        return 0
    fi
    local hint="[Y/n]"
    [[ "$default" == "n" ]] && hint="[y/N]"
    read -r -p "$prompt $hint " reply
    reply="${reply:-$default}"
    [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

require_explicit_confirm() {
    # Used for destructive ops (e.g. wiping volumes). Always prompts,
    # ignores ASSUME_YES unless --yes is given AND --force-destructive.
    local prompt="$1" reply
    if [[ "${FORCE_DESTRUCTIVE:-0}" == "1" && "${ASSUME_YES:-0}" == "1" ]]; then
        return 0
    fi
    read -r -p "$prompt Type 'yes' to continue: " reply
    [[ "$reply" == "yes" ]]
}

# ─── Platform detection ─────────────────────────────────────────
detect_platform() {
    local uname_s; uname_s="$(uname -s)"
    ARCH="$(uname -m)"

    case "$uname_s" in
        Darwin)
            OS_FAMILY="mac"
            local mac_ver; mac_ver="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
            OS_PRETTY="macOS ${mac_ver}"
            PKG_MGR="brew"
            ;;
        Linux)
            OS_FAMILY="linux"
            if [[ -r /etc/os-release ]]; then
                # shellcheck disable=SC1091
                . /etc/os-release
                OS_PRETTY="${PRETTY_NAME:-Linux}"
                case "${ID:-}${ID_LIKE:-}" in
                    *debian*|*ubuntu*) PKG_MGR="apt" ;;
                    *fedora*|*rhel*|*centos*|*rocky*|*almalinux*) PKG_MGR="dnf" ;;
                    *arch*)            PKG_MGR="pacman" ;;
                    *suse*)            PKG_MGR="zypper" ;;
                    *alpine*)          PKG_MGR="apk" ;;
                    *)                 PKG_MGR="" ;;
                esac
            else
                OS_PRETTY="Linux (unknown distro)"
                PKG_MGR=""
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            OS_FAMILY="windows"
            OS_PRETTY="Windows ($uname_s)"
            PKG_MGR=""
            err "Native Windows is not supported. Use WSL2 with an Ubuntu distro."
            exit 2
            ;;
        *)
            OS_FAMILY="unknown"
            OS_PRETTY="$uname_s"
            PKG_MGR=""
            warn "Unrecognised OS: $uname_s — installer will try its best."
            ;;
    esac

    export OS_FAMILY OS_PRETTY ARCH PKG_MGR
}

# ─── Command existence ──────────────────────────────────────────
has() { command -v "$1" >/dev/null 2>&1; }

require_cmd() {
    # require_cmd <name> [hint]
    if ! has "$1"; then
        err "Required command '$1' not found.${2:+ $2}"
        exit 2
    fi
}

# ─── Package installation (best-effort) ─────────────────────────
sudo_run() {
    # Run a command with sudo if we're not already root. No-op the sudo if
    # not available (caller should have already verified).
    if [[ "$EUID" -eq 0 ]]; then
        run "$@"
    elif has sudo; then
        run sudo "$@"
    else
        err "This step needs root, but 'sudo' is not available. Re-run as root or install sudo."
        return 3
    fi
}

pkg_install() {
    # pkg_install <package> [package...] — install via the detected manager.
    # Returns 0 on success, non-zero if no manager or install fails.
    case "$PKG_MGR" in
        brew)   run brew install "$@" ;;
        apt)    sudo_run apt-get update && sudo_run apt-get install -y "$@" ;;
        dnf)    sudo_run dnf install -y "$@" ;;
        pacman) sudo_run pacman -Sy --noconfirm "$@" ;;
        zypper) sudo_run zypper install -y "$@" ;;
        apk)    sudo_run apk add --no-cache "$@" ;;
        *)
            warn "No supported package manager detected; please install manually: $*"
            return 1
            ;;
    esac
}

# ─── Secret generation ──────────────────────────────────────────
gen_secret() {
    # gen_secret [length=64]
    local len="${1:-64}"
    if has openssl; then
        openssl rand -base64 "$((len * 3 / 4 + 1))" | tr -d '/+=\n' | head -c "$len"
    elif [[ -r /dev/urandom ]]; then
        LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c "$len"
    else
        err "Cannot generate secret: no openssl and no /dev/urandom."
        return 1
    fi
}

gen_password() {
    # Like gen_secret but avoids characters that fight with shells / URIs.
    local len="${1:-32}"
    if has openssl; then
        openssl rand -base64 "$((len * 3 / 4 + 1))" \
            | tr -d '/+=\n' \
            | tr -dc 'A-Za-z0-9' \
            | head -c "$len"
    else
        gen_secret "$len"
    fi
}

# ─── .env file editing (idempotent) ─────────────────────────────
env_get() {
    # env_get <file> <key> — print value (without surrounding quotes), or empty.
    local file="$1" key="$2"
    [[ -r "$file" ]] || return 0
    awk -F= -v k="$key" '$1==k { sub(/^[^=]+=/, ""); print; exit }' "$file" \
        | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

env_set() {
    # env_set <file> <key> <value> — set or update. Preserves comments.
    local file="$1" key="$2" value="$3"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf "%s[dry-run]%s would set %s in %s\n" "$C_DIM" "$C_RESET" "$key" "$file"
        return 0
    fi
    touch "$file"
    if grep -qE "^${key}=" "$file"; then
        # macOS sed needs '' for in-place; use a portable workaround
        local tmp; tmp="$(mktemp)"
        awk -v k="$key" -v v="$value" -F= '
            BEGIN { OFS="=" }
            $1==k { print k "=" v; replaced=1; next }
            { print }
        ' "$file" > "$tmp"
        mv "$tmp" "$file"
    else
        printf "%s=%s\n" "$key" "$value" >> "$file"
    fi
}

env_set_if_missing() {
    # Only set if the key is absent OR equals the placeholder template.
    local file="$1" key="$2" value="$3"
    local current; current="$(env_get "$file" "$key")"
    if [[ -z "$current" || "$current" =~ ^(your_.*_here|generate_a_64_char_random_string_here|change_?me)$ ]]; then
        env_set "$file" "$key" "$value"
        return 0
    fi
    return 1
}

# ─── Service health polling ─────────────────────────────────────
wait_for() {
    # wait_for <description> <max-seconds> <bash-command-that-returns-0-on-ready>
    local desc="$1" timeout="$2"
    shift 2
    local elapsed=0 interval=2
    printf "%s[i]%s waiting for %s " "$C_BLUE" "$C_RESET" "$desc"
    while (( elapsed < timeout )); do
        if "$@" >/dev/null 2>&1; then
            printf " %sready%s (%ds)\n" "$C_GREEN" "$C_RESET" "$elapsed"
            return 0
        fi
        printf "."
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    printf " %stimeout%s\n" "$C_RED" "$C_RESET"
    return 1
}

# ─── Self-test ──────────────────────────────────────────────────
# Sanity check: this lib was sourced, not executed.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    err "lib.sh is meant to be sourced, not executed directly."
    exit 1
fi
