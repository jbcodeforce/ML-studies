#!/usr/bin/env bash
set -euo pipefail

export OMLX_URL="http://127.0.0.1"
export OMLX_PORT=7999
export OMLX_MODEL_DIR="${OMLX_MODEL_DIR:-$HOME/.olmx/models}"
export OMLX_API_KEY="${OMLX_API_KEY:-local-key}"

echo "Starting oMLX on ${OMLX_URL}:${OMLX_PORT}/v1 (admin: ${OMLX_URL}:${OMLX_PORT}/admin)"
echo "Model dir: ${OMLX_MODEL_DIR}"

if command -v lsof >/dev/null 2>&1 && lsof -i ":${OMLX_PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Port ${OMLX_PORT} already in use — oMLX may already be running."
  exit 0
fi

exec omlx serve --model-dir="${OMLX_MODEL_DIR}"
