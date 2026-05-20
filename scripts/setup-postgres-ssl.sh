#!/usr/bin/env bash
# ============================================================================
# NeuroLens — Generate self-signed SSL cert for Postgres container
# ----------------------------------------------------------------------------
# Usage:
#   ./scripts/setup-postgres-ssl.sh
#
# Generates server.crt + server.key (10-year validity), then chowns to uid 70
# (postgres user inside container). Requires sudo for chown.
#
# Idempotent: if files exist, skips generation. To regenerate, delete first.
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSL_DIR="${PROJECT_ROOT}/docker/postgres-ssl"

mkdir -p "${SSL_DIR}"

if [[ -f "${SSL_DIR}/server.crt" && -f "${SSL_DIR}/server.key" ]]; then
    echo "✓ SSL files already present at ${SSL_DIR}/"
    echo "  Delete them and re-run if you want to regenerate."
    exit 0
fi

echo "→ Generating self-signed cert (10-year validity)..."
openssl req -new -x509 -days 3650 -nodes -text \
    -out "${SSL_DIR}/server.crt" \
    -keyout "${SSL_DIR}/server.key" \
    -subj "/CN=neurolens-postgres" 2>&1 | tail -2

echo "→ Setting permissions (uid 70 = postgres user in container)..."
sudo chown 70:70 "${SSL_DIR}/server.crt" "${SSL_DIR}/server.key"
sudo chmod 600 "${SSL_DIR}/server.key"
sudo chmod 644 "${SSL_DIR}/server.crt"

ls -la "${SSL_DIR}/"

echo ""
echo "✓ SSL cert ready at ${SSL_DIR}/"
echo "  Next: cd ${PROJECT_ROOT} && docker compose -f docker/postgres-compose.yml --env-file .env up -d"
