/**
 * Frontend Configuration — Local Development Defaults
 *
 * In Docker, this file is overwritten at container startup by
 * docker-entrypoint.sh with values from environment variables.
 *
 * For local development, edit these values directly.
 */
window.__CONFIG__ = {
    FRONTEND_COM: "HTTP",            // "HTTP" or "WEBSOCKET"
    API_BASE_URL: "http://localhost:8000"
};
