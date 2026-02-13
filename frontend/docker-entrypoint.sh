#!/bin/sh
# Generate config.js from environment variables at container startup.
# This allows the static frontend to read runtime configuration.

cat > /usr/share/nginx/html/config.js <<EOF
window.__CONFIG__ = {
    FRONTEND_COM: "${FRONTEND_COM:-HTTP}",
    API_BASE_URL: "${API_BASE_URL:-http://localhost:8000}"
};
EOF

echo "✓ Generated config.js (FRONTEND_COM=${FRONTEND_COM:-HTTP})"

# Hand off to the original CMD (nginx)
exec "$@"
