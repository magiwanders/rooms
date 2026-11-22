#!/usr/bin/env bash
# Local-only static server for web/ (no caching). Reach it through an SSH tunnel
# or VS Code's Ports panel: forward 8642, then open http://localhost:8642
exec python3 "$(dirname "$0")/serve.py"
