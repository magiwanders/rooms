#!/usr/bin/env bash
# Build the wasm module into web/. Serve web/ with any static server (serve.sh).
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/rooms.wasm web/rooms.wasm
ls -l web/rooms.wasm
