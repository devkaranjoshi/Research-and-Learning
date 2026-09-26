#!/usr/bin/env bash
# Cross-platform alternative to `make sync` for environments without rsync (e.g. Windows).
# Uses tar over ssh to push the project to the Lambda GH200 server.
#
# Usage:
#   bash scripts/sync.sh           # push
#   bash scripts/sync.sh fetch     # pull runs/ back

set -euo pipefail

SERVER="${SERVER:?set SERVER=user@host}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
# Allow Windows-style C:/... by translating to /c/... for bash
case "$KEY" in
    [A-Za-z]:/*)
        drive="${KEY%%:*}"
        rest="${KEY#*:}"
        KEY="/${drive,,}${rest}"
        ;;
esac
REMOTE="${REMOTE:-/home/ubuntu/decision-boundary-playground}"

cd "$(dirname "$0")/.."

cmd="${1:-push}"

if [[ "$cmd" == "push" ]]; then
    echo "Pushing to $SERVER:$REMOTE ..."
    ssh -i "$KEY" "$SERVER" "mkdir -p $REMOTE"
    tar --exclude='./runs' --exclude='./.venv' --exclude='./data' \
        --exclude='__pycache__' --exclude='./.git' --exclude='*.pyc' \
        --exclude='.pytest_cache' -cf - . | \
        ssh -i "$KEY" "$SERVER" "tar -xf - -C $REMOTE"
    echo "Done."
elif [[ "$cmd" == "fetch" ]]; then
    echo "Fetching runs/ from $SERVER:$REMOTE ..."
    mkdir -p runs
    ssh -i "$KEY" "$SERVER" "tar -C $REMOTE -cf - runs" | tar -xf - -C .
    echo "Done."
elif [[ "$cmd" == "setup" ]]; then
    echo "Running scripts/server_setup.sh on $SERVER ..."
    ssh -i "$KEY" -t "$SERVER" "cd $REMOTE && bash scripts/server_setup.sh"
else
    echo "usage: bash scripts/sync.sh [push|fetch|setup]" >&2
    exit 2
fi
