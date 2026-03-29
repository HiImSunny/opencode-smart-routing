#!/bin/bash
set -e

# Load .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT:-1234}" \
  --reload
