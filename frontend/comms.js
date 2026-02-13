/**
 * Communication Abstraction Layer
 *
 * Provides a strategy pattern for frontend-to-backend communication.
 * Supports HTTP REST (default) and WebSocket transports, selected via
 * the FRONTEND_COM environment variable injected through config.js.
 *
 * Usage:
 *   const transport = getTransport();
 *   const data = await transport.predict(blob);
 */

// ---------------------------------------------------------------------------
// Configuration (injected by docker-entrypoint.sh into config.js)
// ---------------------------------------------------------------------------

function _getConfig() {
    const cfg = window.__CONFIG__ || {};
    return {
        mode: (cfg.FRONTEND_COM || 'HTTP').toUpperCase(),
        apiBaseUrl: cfg.API_BASE_URL || 'http://localhost:8000',
    };
}

// ---------------------------------------------------------------------------
// HTTP Transport — wraps existing fetch()-based communication
// ---------------------------------------------------------------------------

class HttpTransport {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.name = 'HTTP';
    }

    /** Run inference on a JPEG blob. */
    async predict(blob, confThreshold = 0.15) {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const response = await fetch(
            `${this.baseUrl}/predict?skip_check=false&conf_threshold=${confThreshold}`,
            { method: 'POST', body: formData }
        );

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return response.json();
    }

    /** Run inference on a File (custom image analysis). */
    async analyzeImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseUrl}/predict?skip_check=false`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) throw new Error('Analysis failed');
        return response.json();
    }

    /** Get currently loaded model info. */
    async loadCurrentModel() {
        const response = await fetch(`${this.baseUrl}/models/current`);
        if (!response.ok) throw new Error('No model loaded');
        return response.json();
    }

    /** Load a model by name/version. */
    async loadModel(modelName, version) {
        const response = await fetch(
            `${this.baseUrl}/models/load?model_name=${modelName}&version=${version}`,
            { method: 'POST' }
        );
        if (!response.ok) throw new Error('Failed to load model');
        return response.json();
    }

    /** Load weights from a specific run. */
    async loadRunWeights(runId, artifactPath) {
        const response = await fetch(
            `${this.baseUrl}/models/load_run_weights?run_id=${runId}&artifact_path=${artifactPath}`,
            { method: 'POST' }
        );
        if (!response.ok) throw new Error('Failed to load run weights');
        return response.json();
    }

    /** List all experiments. */
    async loadExperiments() {
        const response = await fetch(`${this.baseUrl}/models/experiments`);
        if (!response.ok) throw new Error('Failed to load experiments');
        return response.json();
    }

    /** Cleanup (no-op for HTTP). */
    destroy() { }

    /** Update confidence threshold (no-op for HTTP, sent per-request). */
    setConfThreshold(_threshold) { }
}

// ---------------------------------------------------------------------------
// WebSocket Transport — uses a persistent WS connection for inference,
// falls back to HTTP for all other (infrequent) operations.
// ---------------------------------------------------------------------------

class WsTransport {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.name = 'WebSocket';

        // HTTP fallback for non-inference calls
        this._http = new HttpTransport(baseUrl);

        // WebSocket state
        this._ws = null;
        this._pendingResolve = null;
        this._pendingReject = null;
        this._connected = false;
        this._connectPromise = null;
        this._confThreshold = 0.15;
        this._confThresholdSynced = false;
    }

    /** Establish the WebSocket connection (lazy, once). */
    _connect() {
        if (this._connectPromise) return this._connectPromise;

        this._connectPromise = new Promise((resolve, reject) => {
            // Derive ws:// or wss:// from the HTTP base URL
            const wsUrl = this.baseUrl
                .replace(/^http:/, 'ws:')
                .replace(/^https:/, 'wss:')
                + '/ws/predict';

            console.log(`🔌 [WS] Connecting to ${wsUrl}`);
            this._ws = new WebSocket(wsUrl);

            this._ws.onopen = () => {
                console.log('🔌 [WS] Connected');
                this._connected = true;
                resolve();
            };

            this._ws.onmessage = (event) => {
                if (this._pendingResolve) {
                    try {
                        const data = JSON.parse(event.data);
                        this._pendingResolve(data);
                    } catch (e) {
                        this._pendingReject(e);
                    }
                    this._pendingResolve = null;
                    this._pendingReject = null;
                }
            };

            this._ws.onerror = (err) => {
                console.error('❌ [WS] Error:', err);
                if (!this._connected) {
                    reject(new Error('WebSocket connection failed'));
                }
                if (this._pendingReject) {
                    this._pendingReject(new Error('WebSocket error'));
                    this._pendingResolve = null;
                    this._pendingReject = null;
                }
            };

            this._ws.onclose = () => {
                console.log('🔌 [WS] Disconnected');
                this._connected = false;
                this._connectPromise = null;
                if (this._pendingReject) {
                    this._pendingReject(new Error('WebSocket closed'));
                    this._pendingResolve = null;
                    this._pendingReject = null;
                }
            };
        });

        return this._connectPromise;
    }

    /** Run inference on a JPEG blob via WebSocket. */
    async predict(blob, confThreshold = 0.15) {
        await this._connect();

        // Sync threshold to server if changed
        if (confThreshold !== this._confThreshold || !this._confThresholdSynced) {
            this._confThreshold = confThreshold;
            this._confThresholdSynced = true;
            // Fire-and-forget threshold update
            this.setConfThreshold(confThreshold);
        }

        return new Promise(async (resolve, reject) => {
            this._pendingResolve = resolve;
            this._pendingReject = reject;

            try {
                const buffer = await blob.arrayBuffer();
                this._ws.send(buffer);
            } catch (e) {
                this._pendingResolve = null;
                this._pendingReject = null;
                reject(e);
            }
        });
    }

    // --- Non-inference calls delegate to HTTP ---

    async analyzeImage(file) { return this._http.analyzeImage(file); }
    async loadCurrentModel() { return this._http.loadCurrentModel(); }
    async loadModel(modelName, version) { return this._http.loadModel(modelName, version); }
    async loadRunWeights(runId, artifactPath) { return this._http.loadRunWeights(runId, artifactPath); }
    async loadExperiments() { return this._http.loadExperiments(); }

    /** Update the confidence threshold on the server. */
    async setConfThreshold(threshold) {
        try {
            await fetch(
                `${this.baseUrl}/config/conf_threshold?threshold=${threshold}`,
                { method: 'POST' }
            );
        } catch (e) {
            console.warn('[WS] Failed to sync conf_threshold:', e);
        }
    }

    /** Close the WebSocket connection. */
    destroy() {
        if (this._ws) {
            this._ws.close();
            this._ws = null;
        }
        this._connected = false;
        this._connectPromise = null;
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

let _transport = null;

/**
 * Get the active transport singleton.
 * Transport mode is determined by window.__CONFIG__.FRONTEND_COM.
 */
function getTransport() {
    if (_transport) return _transport;

    const { mode, apiBaseUrl } = _getConfig();

    if (mode === 'WEBSOCKET') {
        console.log(`✓ Communication mode: WebSocket (API: ${apiBaseUrl})`);
        _transport = new WsTransport(apiBaseUrl);
    } else {
        console.log(`✓ Communication mode: HTTP REST (API: ${apiBaseUrl})`);
        _transport = new HttpTransport(apiBaseUrl);
    }

    return _transport;
}

/**
 * Get the API base URL (used for non-transport operations).
 */
function getApiBaseUrl() {
    return _getConfig().apiBaseUrl;
}
