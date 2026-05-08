#!/usr/bin/env bash
# install_voice.sh — set up local voice mode on the macOS host.
#
# Installs the binaries and downloads the model files that the local
# voice backend (app/voice/local.py) expects:
#
#   whisper-cpp        STT (whisper.cpp build with Apple-Silicon Metal)
#   piper-tts          TTS (Python wrapper that ships the binary)
#   ffmpeg             format conversion (used in follow-ups)
#
#   ~/whisper-models/ggml-large-v3.bin
#   ~/piper-voices/en_US-lessac-medium.onnx + .json
#   ~/piper-voices/et_EE-mart-medium.onnx + .json
#   ~/piper-voices/fi_FI-harri-medium.onnx + .json
#
# Run once on the host:
#   bash host_bridge/install_voice.sh
#
# Idempotent — re-runs are safe and skip steps that have already been done.

set -euo pipefail

WHISPER_MODEL_DIR="${WHISPER_MODEL_DIR:-$HOME/whisper-models}"
PIPER_VOICE_DIR="${PIPER_VOICE_DIR:-$HOME/piper-voices}"

WHISPER_MODEL_NAME="ggml-large-v3.bin"
WHISPER_MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${WHISPER_MODEL_NAME}"

# Piper voices, hosted on HuggingFace under rhasspy/piper-voices.
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_VOICES=(
  "en/en_US/lessac/medium/en_US-lessac-medium"
  "et/et_EE/mart/medium/et_EE-mart-medium"
  "fi/fi_FI/harri/medium/fi_FI-harri-medium"
)

step() { printf "\033[1;34m▸ %s\033[0m\n" "$*"; }
ok()   { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m! %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31m✗ %s\033[0m\n" "$*"; exit 1; }

# ── Homebrew binaries ────────────────────────────────────────────────
need_brew() {
  command -v brew >/dev/null 2>&1 || \
    fail "Homebrew not found. Install from https://brew.sh and re-run."
}

install_brew_pkg() {
  local pkg="$1"
  if brew list --formula "$pkg" >/dev/null 2>&1; then
    ok "$pkg already installed"
  else
    step "brew install $pkg"
    brew install "$pkg"
  fi
}

# ── whisper.cpp ──────────────────────────────────────────────────────
install_whisper() {
  install_brew_pkg whisper-cpp
  if ! command -v whisper-cli >/dev/null 2>&1; then
    fail "whisper-cli not on PATH after brew install — check 'brew doctor'"
  fi
  mkdir -p "$WHISPER_MODEL_DIR"
  local target="$WHISPER_MODEL_DIR/$WHISPER_MODEL_NAME"
  if [[ -f "$target" ]]; then
    ok "Whisper model already at $target"
  else
    step "Downloading whisper-large-v3 model (~3 GB)"
    curl --fail --location --progress-bar --output "$target" "$WHISPER_MODEL_URL" || {
      rm -f "$target"
      fail "Whisper model download failed"
    }
  fi
}

# ── Piper TTS ────────────────────────────────────────────────────────
install_piper() {
  if ! command -v piper >/dev/null 2>&1; then
    if command -v pipx >/dev/null 2>&1; then
      step "pipx install piper-tts"
      pipx install piper-tts
    else
      step "pip install --user piper-tts"
      python3 -m pip install --user piper-tts
    fi
  fi
  if ! command -v piper >/dev/null 2>&1; then
    fail "piper not on PATH after install — add ~/.local/bin to PATH or use pipx"
  fi
  ok "piper available at $(command -v piper)"

  mkdir -p "$PIPER_VOICE_DIR"
  for voice_path in "${PIPER_VOICES[@]}"; do
    local basename
    basename="$(basename "$voice_path")"
    local onnx="$PIPER_VOICE_DIR/$basename.onnx"
    local meta="$PIPER_VOICE_DIR/$basename.onnx.json"
    if [[ -f "$onnx" && -f "$meta" ]]; then
      ok "Piper voice $basename already present"
      continue
    fi
    step "Downloading Piper voice $basename"
    curl --fail --location --silent --output "$onnx" "$PIPER_BASE/$voice_path.onnx" || {
      rm -f "$onnx"; fail "Failed to fetch $basename.onnx"
    }
    curl --fail --location --silent --output "$meta" "$PIPER_BASE/$voice_path.onnx.json" || {
      rm -f "$onnx" "$meta"; fail "Failed to fetch $basename.onnx.json"
    }
  done
}

# ── ffmpeg (used by follow-up format-conversion patch) ───────────────
install_ffmpeg() {
  install_brew_pkg ffmpeg
}

# ── Smoke test ───────────────────────────────────────────────────────
smoke_test() {
  step "Smoke test: whisper-cli --help"
  whisper-cli --help >/dev/null 2>&1 || warn "whisper-cli --help failed"
  step "Smoke test: piper --help"
  piper --help >/dev/null 2>&1 || warn "piper --help failed"
  ok "Voice toolchain installed"
}

main() {
  need_brew
  install_whisper
  install_piper
  install_ffmpeg
  smoke_test
  echo
  echo "Next:"
  echo "  - Open the React Settings page (/cp/settings) and switch Voice mode to 'Local'."
  echo "  - Send a Signal voice note from your iPhone — the gateway will transcribe it."
  echo "  - The reply will come back as a Piper-synthesized voice attachment."
}

main "$@"
