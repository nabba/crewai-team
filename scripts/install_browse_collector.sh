#!/usr/bin/env bash
# Install the browse host collector as a launchd LaunchAgent.
#
# PROGRAM §50 (Phase B+ turn-on). The gateway runs in Docker and can't
# read ~/Library/Safari/ etc. directly. This LaunchAgent runs as the
# logged-in macOS user, reads the host browser history SQLite files,
# and writes events into the same workspace directory the gateway
# mounts — so events flow end-to-end without any cross-OS quirks.
#
# Usage:
#   ./scripts/install_browse_collector.sh install   # link + load
#   ./scripts/install_browse_collector.sh start     # kick off once
#   ./scripts/install_browse_collector.sh stop      # stop the agent
#   ./scripts/install_browse_collector.sh uninstall # unload + remove
#   ./scripts/install_browse_collector.sh status    # is it loaded?

set -euo pipefail

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/browse_host_collector.plist"
PLIST_DST="$HOME/Library/LaunchAgents/org.andrus.botarmy.browse-collector.plist"
LABEL="org.andrus.botarmy.browse-collector"

cmd="${1:-install}"

case "$cmd" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DST"
    # bootstrap will fail if already loaded — accept both states.
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
    echo "✓ Installed and loaded $LABEL"
    echo "  Logs: ~/BotArmy/crewai-team/workspace/browse/.host_collector.log"
    echo ""
    echo "NEXT MANUAL STEP — grant Full Disk Access:"
    echo "  System Settings → Privacy & Security → Full Disk Access"
    echo "  Click '+' and add this Python interpreter:"
    echo "    $(readlink -f /Users/andrus/BotArmy/crewai-team/.venv/bin/python || echo /Users/andrus/BotArmy/crewai-team/.venv/bin/python)"
    echo "  Then run: $0 restart"
    ;;
  start)
    launchctl kickstart -p "gui/$(id -u)/$LABEL"
    echo "✓ Started one pass via launchd"
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
      echo "(PID / last_exit_code / label above)"
    else
      echo "Not loaded."
    fi
    ;;
  *)
    echo "Usage: $0 {install|start|restart|stop|uninstall|status}"
    exit 1
    ;;
esac
