#!/usr/bin/env bash
# Build and publish: copies the static site to /var/www/rooms, which Caddy serves
# as https://rooms.magiwanders.com (file_server, no backend).
set -euo pipefail
cd "$(dirname "$0")"
./build.sh
mkdir -p /var/www/rooms
cp web/index.html web/main.js web/shader.js web/rooms.wasm /var/www/rooms/
echo "deployed to /var/www/rooms"
