#!/usr/bin/env bash
# Install the host-substrate-metrics collector as a launchd LaunchAgent.
#
# PROGRAM §51 Q16 Theme 1 follow-on (Q16.1 Item 6). Companion to
# scripts/host_substrate_metrics.sh + scripts/host_substrate_metrics.plist
# + app/healing/monitors/host_substrate_health.py.
#
# Why this exists: the in-container substrate-health monitor can't see
# SMART data, macOS version, or Docker VM sizing. This script
# installs a weekly host-side collector that writes those signals
# into workspace/healing/host_metrics.jsonl for the in-container
# monitor to surface.
#
# Usage:
#   ./scripts/install_host_substrate_metrics.sh install   # link + load
#   ./scripts/install_host_substrate_metrics.sh start     # kick off once
#   ./scripts/install_host_substrate_metrics.sh stop      # stop the agent
#   ./scripts/install_host_substrate_metrics.sh restart   # reload
#   ./scripts/install_host_substrate_metrics.sh uninstall # unload + remove
#   ./scripts/install_host_substrate_metrics.sh status    # is it loaded?

set -euo pipefail

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/host_substrate_metrics.plist"
PLIST_DST="$HOME/Library/LaunchAgents/org.andrus.botarmy.host-substrate-metrics.plist"
LABEL="org.andrus.botarmy.host-substrate-metrics"

cmd="${1:-install}"

case "$cmd" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DST"
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
    echo "✓ Installed and loaded $LABEL"
    echo "  Cadence: weekly (Sunday 03:00)"
    echo "  Logs: workspace/healing/.host_metrics.log"
    echo ""
    echo "OPTIONAL — for SMART data:"
    echo "  brew install smartmontools jq"
    echo "  Add a sudoers rule allowing passwordless smartctl:"
    echo "    sudo visudo -f /etc/sudoers.d/smartctl"
    echo "    $(whoami) ALL=(ALL) NOPASSWD: /usr/local/sbin/smartctl"
    echo ""
    echo "  Without sudo+smartctl the SMART fields are null but the"
    echo "  other host signals (macOS version, host disk total,"
    echo "  Docker VM size, host uptime) still flow."
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
    launchctl bootout "gui/$(id -u)/$LABEL"
    echo "✓ Stopped $LABEL"
    ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "✓ Uninstalled $LABEL"
    ;;
  status)
    if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
      echo "✓ $LABEL is loaded"
      launchctl print "gui/$(id -u)/$LABEL" | grep -E "state|last exit"
    else
      echo "✗ $LABEL is NOT loaded"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 {install|start|restart|stop|uninstall|status}"
    exit 1
    ;;
esac
