#!/usr/bin/env bash
# Install the host-side DB backup as a launchd LaunchAgent.
#
# 2026-05-16 split (Option C, docs/RESILIENCE_POSTURE.md). The gateway
# can't call `docker exec` against sibling containers — the docker-proxy
# sidecar denies /exec/.../start without an explicit EXEC: 1 flag, which
# would widen the gateway's blast radius. Instead, pg + neo4j backups
# run from the host via this LaunchAgent (which uses the host's docker
# socket directly). The gateway keeps owning chromadb (volume is
# bind-mounted, no exec needed).
#
# Schedule: daily 04:30 local time.
# Logs: workspace/backups/.db_backup_host.log
# Manifest: workspace/backups/manifest.json (shared with the gateway).
#
# Usage:
#   ./scripts/install_db_backup.sh install     # link + load
#   ./scripts/install_db_backup.sh start       # run once now (smoke test)
#   ./scripts/install_db_backup.sh restart     # reload the plist
#   ./scripts/install_db_backup.sh stop        # unload the agent
#   ./scripts/install_db_backup.sh uninstall   # unload + remove
#   ./scripts/install_db_backup.sh status      # is it loaded?

set -euo pipefail

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/db_backup_host.plist"
PLIST_DST="$HOME/Library/LaunchAgents/org.andrus.botarmy.db-backup.plist"
LABEL="org.andrus.botarmy.db-backup"

cmd="${1:-install}"

case "$cmd" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DST"
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
    echo "✓ Installed and loaded $LABEL"
    echo "  Schedule: daily 04:30 local time"
    echo "  Logs:     ~/BotArmy/crewai-team/workspace/backups/.db_backup_host.log"
    echo ""
    echo "Smoke-test once before walking away:"
    echo "  $0 start"
    ;;
  start)
    launchctl kickstart -p "gui/$(id -u)/$LABEL"
    echo "✓ Kicked off one pass via launchd. Tail the log to watch:"
    echo "  tail -F ~/BotArmy/crewai-team/workspace/backups/.db_backup_host.log"
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
