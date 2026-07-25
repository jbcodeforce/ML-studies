#!/usr/bin/env bash
# Load env vars from the file pointed to by ML_ENV_FILE into the current shell.
# Usage: export ML_ENV_FILE=.env && source set_env.sh

if [[ -z "${ML_ENV_FILE:-}" ]]; then
  echo "ML_ENV_FILE is not set" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$ML_ENV_FILE" ]]; then
  echo "File not found: $ML_ENV_FILE" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -r "$ML_ENV_FILE" ]]; then
  echo "File not readable: $ML_ENV_FILE" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ML_ENV_FILE"
set +a
