#!/usr/bin/env bash
# Install the host-side gateway watchdog as a launchd LaunchAgent.
#
# The gateway runs background work on the same event loop that serves HTTP.
# A heavy idle-job burst (training scorer + ONNX download + sentience probes)
# can starve /signal/inbound for minutes after boot. The signal forwarder
# drops messages whose retry budget runs out inside that window. The
# in-container healing layer can't recover from a hung event loop because
# it lives inside the same process.
#
# This watchdog runs on the host, polls /health every 20s, and restarts the
# gateway container after 6 consecutive failures (~2 min hung). Alerts go
# via signal-cli directly so the operator hears about it.
#
# Logs:  workspace/healing/.gateway_watchdog.log
#
# Usage:
#   ./scripts/install_gateway_watchdog.sh install    # link + load
#   ./scripts/install_gateway_watchdog.sh start      # smoke test (kickstart now)
#   ./scripts/install_gateway_watchdog.sh restart    # reload the plist
#   ./scripts/install_gateway_watchdog.sh stop       # unload the agent
#   ./scripts/install_gateway_watchdog.sh uninstall  # unload + remove
#   ./scripts/install_gateway_watchdog.sh status     # is it loaded?

set -euo pipefail

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/gateway_watchdog.plist"
PLIST_DST="$HOME/Library/LaunchAgents/org.andrus.botarmy.gateway-watchdog.plist"
LABEL="org.andrus.botarmy.gateway-watchdog"
LOG_DIR="/Users/andrus/BotArmy/crewai-team/workspace/healing"

cmd="${1:-install}"

case "$cmd" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents"
    mkdir -p "$LOG_DIR"
    cp "$PLIST_SRC" "$PLIST_DST"
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
    echo "✓ Installed and loaded $LABEL"
    echo "  Polls:    http://127.0.0.1:8765/health every 20s"
    echo "  Trigger:  6 consecutive failures (~2 min hung) → restart gateway"
    echo "  Alerts:   Signal to +3725100500 via signal-cli on port 7583"
    echo "  Logs:     $LOG_DIR/.gateway_watchdog.log"
    ;;
  start)
    launchctl kickstart -p "gui/$(id -u)/$LABEL"
    echo "✓ Kickstarted $LABEL. Tail the log to watch:"
    echo "  tail -F $LOG_DIR/.gateway_watchdog.log"
    ;;
  restart)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
    echo "✓ Restarted $LABEL"
    ;;
  stop)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    echo "✓ Stopped $LABEL"
    ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "✓ Uninstalled $LABEL"
    ;;
  status)
    if launchctl list | grep -q "$LABEL"; then
      launchctl list | grep "$LABEL"
      echo "(PID / last_exit_code / label above; PID should be non-'-' since this is a daemon)"
    else
      echo "Not loaded."
    fi
    ;;
  *)
    echo "Usage: $0 {install|start|restart|stop|uninstall|status}"
    exit 1
    ;;
esac
