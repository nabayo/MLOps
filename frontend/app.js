/**
 * MLOps Frontend Application
 *
 * Features:
 * - Live webcam inference with finger counting
 * - Model selection and live switching
 * - Experiments dashboard with metrics
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let webcamStream = null;
let inferenceInterval = null;
let currentModelInfo = null;
let fpsCounter = { frames: 0, lastTime: Date.now() };

// ============================================================================
// Page Navigation
// ============================================================================

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageName = link.getAttribute('data-page');

            // Update active states
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            pages.forEach(p => p.classList.remove('active'));
            document.getElementById(`page-${pageName}`).classList.add('active');

            // Load page data
            if (pageName === 'models') {
                loadModels();
            } else if (pageName === 'experiments') {
                loadExperiments();
            }
        });
    });
}

// ============================================================================
// Live Inference Page
// ============================================================================

async function startWebcam() {
    try {
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');

        // Get webcam stream
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });

        video.srcObject = webcamStream;
        video.play();

        // Wait for video to be ready
        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                document.getElementById('resolution').textContent =
                    `Resolution: ${video.videoWidth}x${video.videoHeight}`;
                resolve();
            };
        });

        // Start inference loop
        startInference();

        // Update button states
        document.getElementById('startWebcam').disabled = true;
        document.getElementById('stopWebcam').disabled = false;

        showToast('Webcam started successfully', 'success');

    } catch (error) {
        console.error('Error starting webcam:', error);
        showToast('Failed to start webcam: ' + error.message, 'error');
    }
}

function stopWebcam() {
    // Stop inference
    if (inferenceInterval) {
        clearInterval(inferenceInterval);
        inferenceInterval = null;
    }

    // Stop webcam stream
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    // Clear video
    const video = document.getElementById('webcam');
    video.srcObject = null;

    // Clear canvas
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update button states
    document.getElementById('startWebcam').disabled = false;
    document.getElementById('stopWebcam').disabled = true;

    // Reset displays
    document.getElementById('fingerCount').textContent = '0';
    document.getElementById('predictionsList').innerHTML = '<p class="empty-state">No predictions yet</p>';

    showToast('Webcam stopped', 'info');
}

function startInference() {
    const video = document.getElementById('webcam');

    // Run inference every 100ms (10 FPS)
    inferenceInterval = setInterval(async () => {
        await runInference(video);
        updateFPS();
    }, 100);
}

async function runInference(video) {
    const startTime = performance.now();

    try {
        // Capture frame from video
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Convert to blob
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.95));

        // Send to API
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update displays
        updateFingerCount(data.finger_count);
        updatePredictionsList(data.predictions);
        drawPredictions(data.predictions);

        // Update latency
        const latency = performance.now() - startTime;
        document.getElementById('latency').textContent = `Latency: ${Math.round(latency)}ms`;

    } catch (error) {
        console.error('Inference error:', error);
        // Don't show toast for every error to avoid spam
    }
}

function drawPredictions(predictions) {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw each prediction
    predictions.forEach(pred => {
        const [x1, y1, x2, y2] = pred.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Draw bounding box
        ctx.strokeStyle = '#00ff41';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label background
        const label = `${pred.class_name} ${(pred.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 16px Arial';
        const textMetrics = ctx.measureText(label);
        const textHeight = 20;

        ctx.fillStyle = '#00ff41';
        ctx.fillRect(x1, y1 - textHeight - 4, textMetrics.width + 10, textHeight + 4);

        // Draw label text
        ctx.fillStyle = '#000';
        ctx.fillText(label, x1 + 5, y1 - 6);
    });
}

function updateFingerCount(count) {
    const fingerCountEl = document.getElementById('fingerCount');
    fingerCountEl.textContent = count;

    // Add animation
    fingerCountEl.classList.remove('pulse');
    setTimeout(() => fingerCountEl.classList.add('pulse'), 10);
}

function updatePredictionsList(predictions) {
    const listEl = document.getElementById('predictionsList');

    if (predictions.length === 0) {
        listEl.innerHTML = '<p class="empty-state">No predictions</p>';
        return;
    }

    listEl.innerHTML = predictions.map(pred => `
        <div class="prediction-item">
            <span class="pred-class">${pred.class_name}</span>
            <span class="pred-conf">${(pred.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}

function updateFPS() {
    fpsCounter.frames++;
    const now = Date.now();
    const elapsed = now - fpsCounter.lastTime;

    if (elapsed >= 1000) {
        const fps = Math.round((fpsCounter.frames * 1000) / elapsed);
        document.getElementById('fps').textContent = `FPS: ${fps}`;
        fpsCounter.frames = 0;
        fpsCounter.lastTime = now;
    }
}

async function loadCurrentModel() {
    try {
        const response = await fetch(`${API_BASE_URL}/models/current`);

        if (!response.ok) {
            throw new Error('No model loaded');
        }

        currentModelInfo = await response.json();
        updateModelDisplay(currentModelInfo);

    } catch (error) {
        console.error('Error loading current model:', error);
        updateModelDisplay(null);
    }
}

function updateModelDisplay(modelInfo) {
    if (!modelInfo) {
        document.getElementById('modelArch').textContent = 'None loaded';
        document.getElementById('modelVersion').textContent = '-';
        document.getElementById('modelStage').textContent = '-';
        document.getElementById('modelMap').textContent = '-';
        return;
    }

    document.getElementById('modelArch').textContent = modelInfo.architecture || 'Unknown';
    document.getElementById('modelVersion').textContent = `v${modelInfo.version || '?'}`;
    document.getElementById('modelStage').textContent = modelInfo.stage || 'Unknown';

    const map = modelInfo.metrics?.['mAP50-95'] || modelInfo.metrics?.['mAP50'];
    document.getElementById('modelMap').textContent = map ? map.toFixed(4) : 'N/A';
}

// ============================================================================
// Model Selection Page
// ============================================================================

async function loadModels() {
    const gridEl = document.getElementById('modelsGrid');
    gridEl.innerHTML = '<div class="loading">Loading models...</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/models/list`);

        if (!response.ok) {
            throw new Error('Failed to load models');
        }

        const models = await response.json();

        if (models.length === 0) {
            gridEl.innerHTML = '<p class="empty-state">No models found in registry</p>';
            return;
        }

        gridEl.innerHTML = models.map(model => `
            <div class="model-card">
                <h3 class="model-name">${model.name}</h3>
                <p class="model-desc">${model.description || 'No description'}</p>
                <div class="model-versions">
                    ${model.versions.map(version => `
                        <div class="version-item">
                            <div class="version-header">
                                <span class="version-num">v${version.version}</span>
                                <span class="badge badge-${version.stage.toLowerCase()}">${version.stage}</span>
                            </div>
                            <div class="version-metrics">
                                ${Object.entries(version.metrics || {}).slice(0, 3).map(([k, v]) => `
                                    <span class="metric">${k}: ${v.toFixed(4)}</span>
                                `).join('')}
                            </div>
                            <button class="btn btn-sm btn-primary"
                                    onclick="loadModel('${model.name}', '${version.version}')">
                                Load Model
                            </button>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading models:', error);
        gridEl.innerHTML = `<p class="error-state">Error: ${error.message}</p>`;
        showToast('Failed to load models', 'error');
    }
}

async function loadModel(modelName, version) {
    try {
        showToast('Loading model...', 'info');

        const response = await fetch(`${API_BASE_URL}/models/load?model_name=${modelName}&version=${version}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Failed to load model');
        }

        const result = await response.json();

        showToast(`Model ${modelName} v${version} loaded successfully!`, 'success');

        // Update current model info
        await loadCurrentModel();

    } catch (error) {
        console.error('Error loading model:', error);
        showToast('Failed to load model: ' + error.message, 'error');
    }
}

// ============================================================================
// Experiments Dashboard Page
// ============================================================================

async function loadExperiments() {
    const listEl = document.getElementById('experimentsList');
    listEl.innerHTML = '<div class="loading">Loading experiments...</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/models/experiments`);

        if (!response.ok) {
            throw new Error('Failed to load experiments');
        }

        const experiments = await response.json();

        if (experiments.length === 0) {
            listEl.innerHTML = '<p class="empty-state">No experiments found</p>';
            return;
        }

        listEl.innerHTML = experiments.map(exp => `
            <div class="experiment-card">
                <div class="experiment-header">
                    <h3>${exp.experiment_name}</h3>
                    <span class="run-count">${exp.run_count} runs</span>
                </div>
                <div class="runs-table-container">
                    <table class="runs-table">
                        <thead>
                            <tr>
                                <th>Run Name</th>
                                <th>Status</th>
                                <th>Architecture</th>
                                <th>mAP@50-95</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>Date</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${exp.runs.map(run => `
                                <tr>
                                    <td><span class="run-name" title="${run.run_id}">${run.run_name}</span></td>
                                    <td><span class="badge badge-${run.status.toLowerCase()}">${run.status}</span></td>
                                    <td>${run.params?.model_architecture || 'N/A'}</td>
                                    <td>${run.metrics?.['mAP50-95']?.toFixed(4) || 'N/A'}</td>
                                    <td>${run.metrics?.precision?.toFixed(4) || 'N/A'}</td>
                                    <td>${run.metrics?.recall?.toFixed(4) || 'N/A'}</td>
                                    <td>${new Date(run.start_time).toLocaleDateString()}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading experiments:', error);
        listEl.innerHTML = `<p class="error-state">Error: ${error.message}</p>`;
        showToast('Failed to load experiments', 'error');
    }
}

// ============================================================================
// Utilities
// ============================================================================

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast toast-${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize navigation
    initNavigation();

    // Load current model info
    loadCurrentModel();

    // Set up event listeners
    document.getElementById('startWebcam').addEventListener('click', startWebcam);
    document.getElementById('stopWebcam').addEventListener('click', stopWebcam);
    document.getElementById('changeModelBtn').addEventListener('click', () => {
        document.querySelectorAll('.nav-link')[1].click(); // Switch to models page
    });
    document.getElementById('refreshModels').addEventListener('click', loadModels);
    document.getElementById('refreshExperiments').addEventListener('click', loadExperiments);

    console.log('✓ MLOps Frontend initialized');
});
