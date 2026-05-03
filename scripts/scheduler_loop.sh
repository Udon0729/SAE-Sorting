#!/usr/bin/env bash
# Run scheduler.py in a loop. Designed for a tmux persistent session.
set -u
cd "$(dirname "$0")/.."
while true; do
  uv run python scripts/scheduler.py 2>&1 | sed "s/^/[$(date -Is)] /"
  sleep 30
done
