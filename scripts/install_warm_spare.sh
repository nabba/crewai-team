#!/usr/bin/env bash
# Install the warm-spare rsync as a launchd LaunchAgent on macOS (Q17.1).

set -euo pipefail

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/warm_spare_host.plist"
PLIST_DST="$HOME/Library/LaunchAgents/org.andrus.botarmy.warm-spare-sync.plist"
LABEL="org.andrus.botarmy.warm-spare-sync"

cmd="${1:-install}"

case "$cmd" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DST"
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
    echo "✓ Installed and loaded $LABEL"
    echo "  Schedule: every hour at :00 local time"
    echo "  Logs:     ~/BotArmy/crewai-team/workspace/warm_spare/.sync.log"
    echo ""
    echo "Smoke test with one immediate pass:"
    echo "  $0 start"
    ;;
  start)
    launchctl kickstart -p "gui/$(id -u)/$LABEL"
    echo "✓ Kicked off one sync pass. Tail the log:"
    echo "  tail -F ~/BotArmy/crewai-team/workspace/warm_spare/.sync.log"
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
      echo "(PID / last_exit_code / label above; PID '-' is normal between runs)"
    else
      echo "Not loaded."
    fi
    ;;
  *)
    echo "Usage: $0 {install|start|restart|stop|uninstall|status}"
    exit 1
    ;;
esac
