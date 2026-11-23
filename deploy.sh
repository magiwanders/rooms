#!/usr/bin/env bash
# Build and publish: copies the static site to /var/www/rooms, which Caddy serves
# as https://rooms.magiwanders.com (file_server, no backend).
set -euo pipefail
cd "$(dirname "$0")"
./build.sh
mkdir -p /var/www/rooms
cp web/*.html web/*.js web/*.wasm web/*.webmanifest web/*.png web/*.ico /var/www/rooms/
echo "deployed to /var/www/rooms"
