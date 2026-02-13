/**
 * MLOps Frontend Application
 *
 * Features:
 * - Live webcam inference with finger counting
 * - Model selection and live switching
 * - Experiments dashboard with metrics
 */

console.log('APP.JS LOADED - VERSION 3 - WITH DETAILED LOGGING', 'background: #222; color: #bada55; font-size: 16px; font-weight: bold;');

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let webcamStream = null;
let inferenceInterval = null;
let currentModelInfo = null;
let fpsCounter = { frames: 0, lastTime: Date.now() };

// Client-side frame skipping state
let lastPredictions = [];
let skipFrameCounter = 0;
let skipFrameLimit = 6;
let isInferring = false;
let analysisInitialized = false;

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
            if (pageName === 'experiments') {
                loadExperiments();
            } else if (pageName === 'analysis' && !analysisInitialized) {
                initAnalysisPage();
            }
        });
    });
}

// ============================================================================
// Analysis Page
// ============================================================================

function initAnalysisPage() {
    console.log('initAnalysisPage called');
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');
    const btn = document.getElementById('uploadBtn');

    setupUploadListeners(zone, input, btn);
    analysisInitialized = true;
}

function setupUploadListeners(dropZone, input, btn) {
    console.log('setupUploadListeners called', { dropZone, input, btn });
    // Button trigger
    if (btn) {
        btn.onclick = (e) => {
            console.log('Upload button clicked');
            e.stopPropagation(); // Prevent bubbling causing double trigger if zone has listener
            input.click();
        };
    }

    // Zone click trigger (optional, if user clicks background)
    dropZone.onclick = () => input.click();

    input.onchange = (e) => {
        if (e.target.files.length) {
            handleAnalysisImage(e.target.files[0]);
            // Reset so re-selecting the same file still triggers change
            input.value = '';
        }
    };

    // Prevent default behaviors for all drag events to stop browser from opening file
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        // Also add to document to be safe? No, just zone is enough usually if we are precise,
        // but if user misses zone, it opens. Let's stick to zone for now, user asked "if I drag ... in the browser".
        // To be safe, we should prevent default on the whole document for drop if it's not in the zone, or just the zone.
        // Let's fix the zone first.
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight logic
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) handleAnalysisImage(files[0]);
    }, false);
}

async function handleAnalysisImage(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file', 'error');
        return;
    }

    try {
        showToast('Analyzing image...', 'info');

        // Show results container
        document.getElementById('analysisResults').style.display = 'grid';

        // Preview Original (Optional, we will draw result directly)

        // Send to API
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/predict?skip_check=false`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();

        // Display Result
        displayAnalysisResult(data, file);
        showToast('Analysis complete', 'success');

    } catch (error) {
        console.error(error);
        showToast('Analysis failed: ' + error.message, 'error');
    }
}

async function displayAnalysisResult(data, originalFile) {
    console.log('displayAnalysisResult called', data);
    const canvas = document.getElementById('analysisCanvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    // Determine image source
    let imgSrc = null;
    if (data.processed_image) {
        console.log('Using processed image from API');
        imgSrc = 'data:image/jpeg;base64,' + data.processed_image;
    } else if (originalFile) {
        console.log('Using original uploaded file');
        imgSrc = URL.createObjectURL(originalFile);
    } else {
        console.error('No image source available');
        showToast('Error: No image to display', 'error');
        return;
    }

    img.src = imgSrc;

    img.onload = () => {
        console.log('Image loaded for analysis display', img.width, img.height);

        // Resize canvas to match image aspect ratio but fit within container
        const MAX_WIDTH = 800;
        let width = img.width;
        let height = img.height;

        if (width > MAX_WIDTH) {
            const ratio = MAX_WIDTH / width;
            width = MAX_WIDTH;
            height = height * ratio;
        }

        canvas.width = width;
        canvas.height = height;

        ctx.drawImage(img, 0, 0, width, height);

        // Draw predictions
        const scale = width / img.width;
        console.log('Drawing predictions with scale', scale);

        if (data.predictions && data.predictions.length > 0) {
            data.predictions.forEach(pred => {
                const [x1, y1, x2, y2] = pred.bbox;
                // Scale
                const sx1 = x1 * scale;
                const sy1 = y1 * scale;
                const sw = (x2 - x1) * scale;
                const sh = (y2 - y1) * scale;

                ctx.strokeStyle = '#00ff41';
                ctx.lineWidth = 3;
                ctx.strokeRect(sx1, sy1, sw, sh);

                // Label
                const label = `${pred.class_name} ${(pred.confidence * 100).toFixed(1)}%`;
                ctx.font = 'bold 16px Arial';
                const tm = ctx.measureText(label);

                ctx.fillStyle = '#00ff41';
                ctx.fillRect(sx1, sy1 - 24, tm.width + 10, 24);

                ctx.fillStyle = '#000';
                ctx.fillText(label, sx1 + 5, sy1 - 6);
            });
        } else {
            console.log('No predictions to draw');
        }

        // Update Count
        document.getElementById('analysisCount').textContent = data.finger_count || 0;
    };

    img.onerror = (e) => {
        console.error('Failed to load image for display', e);
        showToast('Failed to load image', 'error');
    };
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

    // Reset client-side skip counter
    skipFrameCounter = 0;
    isInferring = false;

    // Run inference loop every 100ms (~10 FPS visual refresh)
    inferenceInterval = setInterval(() => {
        // Always redraw last known predictions on skipped frames
        // so bounding boxes persist over the live video
        drawPredictions(lastPredictions);

        skipFrameCounter++;

        if (skipFrameCounter >= skipFrameLimit && !isInferring) {
            // Time for a real prediction — guard against overlapping calls
            skipFrameCounter = 0;
            isInferring = true;
            runInference(video).finally(() => {
                isInferring = false;
            });
        }

        updateFPS();
    }, 100);
}

async function runInference(video) {
    const startTime = performance.now();
    console.log('🎬 [INFERENCE] Starting inference...');

    try {
        // Capture frame from video
        console.log('📷 [INFERENCE] Capturing frame from video...');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        console.log(`📷 [INFERENCE] Frame captured: ${canvas.width}x${canvas.height}`);

        // Convert to blob
        console.log('🔄 [INFERENCE] Converting to blob...');
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.95));
        console.log(`🔄 [INFERENCE] Blob created: ${blob.size} bytes`);

        // Send to API
        console.log('📡 [INFERENCE] Sending to API...');
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const response = await fetch(`${API_BASE_URL}/predict?skip_check=false`, {
            method: 'POST',
            body: formData
        });

        console.log(`📡 [INFERENCE] Response status: ${response.status}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('📦 [INFERENCE] Response data:', {
            finger_count: data.finger_count,
            predictions: data.predictions?.length || 0,
            preprocessing_applied: data.preprocessing_applied,
            inference_time_ms: data.inference_time_ms,
            has_processed_image: !!data.processed_image,
            processed_image_length: data.processed_image?.length || 0
        });

        // If backend returned a processed image, display it
        if (data.processed_image) {
            console.log('🖼️ [DISPLAY] Processed image found, displaying...');
            const displayCanvas = document.getElementById('canvas');
            const displayCtx = displayCanvas.getContext('2d');

            console.log(`🖼️ [DISPLAY] Canvas element: ${displayCanvas.width}x${displayCanvas.height}`);

            const img = new Image();
            img.onload = () => {
                console.log('✅ [DISPLAY] Image loaded successfully');
                // Clear canvas
                displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);

                // Draw processed (blurred) image
                displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
                console.log('✅ [DISPLAY] Blurred image drawn to canvas');

                // Draw predictions on top of processed image
                drawPredictionsOnContext(displayCtx, data.predictions);
                console.log('✅ [DISPLAY] Predictions drawn');
            };
            img.onerror = (e) => {
                console.error('❌ [DISPLAY] Failed to load image:', e);
            };
            img.src = 'data:image/jpeg;base64,' + data.processed_image;
            console.log('🖼️ [DISPLAY] Image source set, waiting for onload...');
        } else {
            console.warn('⚠️ [DISPLAY] No processed image in response, using raw video');
            // No processed image, just draw predictions on overlay canvas
            drawPredictions(data.predictions);
        }

        // Update displays
        updateFingerCount(data.finger_count);
        updatePredictionsList(data.predictions);

        // Store last predictions for overlay on skipped frames
        lastPredictions = data.predictions || [];

        // Update latency
        const latency = performance.now() - startTime;
        document.getElementById('latency').textContent = `Latency: ${Math.round(latency)}ms`;
        console.log(`⏱️ [INFERENCE] Complete in ${Math.round(latency)}ms`);

    } catch (error) {
        console.error('❌ [INFERENCE] Error:', error);
        console.error('❌ [INFERENCE] Stack:', error.stack);
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
    drawPredictionsOnContext(ctx, predictions);
}

function drawPredictionsOnContext(ctx, predictions) {
    // Draw each prediction on the given context
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
// Model Loading (from experiments page)
// ============================================================================

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

async function loadRunWeights(runId, artifactPath) {
    try {
        showToast(`Loading weights ${artifactPath}...`, 'info');

        const response = await fetch(`${API_BASE_URL}/models/load_run_weights?run_id=${runId}&artifact_path=${artifactPath}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Failed to load run weights');
        }

        const result = await response.json();

        showToast(`Loaded ${artifactPath} successfully!`, 'success');

        // Update current model info
        await loadCurrentModel();

        // Switch to analysis tab to test? Or just stay.
        // Let's stay but maybe scroll top?

    } catch (error) {
        console.error('Error loading run weights:', error);
        showToast('Failed to load weights: ' + error.message, 'error');
    }
}

// ============================================================================
// Skip Frame Configuration
// ============================================================================

function setSkipFrames(value) {
    const parsed = parseInt(value);
    if (!isNaN(parsed) && parsed >= 0) {
        skipFrameLimit = parsed;
        skipFrameCounter = 0;
        console.log(`✓ Client-side skip frame limit set to: ${skipFrameLimit}`);
        showToast(`Skip frames set to ${skipFrameLimit}`, 'success');
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
                                <th>Weights</th>
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
                                    <td>
                                        <div class="weights-actions">
                                            ${(run.available_weights || []).map(w => `
                                                <button class="btn btn-xs btn-outline"
                                                        onclick="loadRunWeights('${run.run_id}', '${w}')"
                                                        title="Load ${w}">
                                                    ${w.split('/').pop()}
                                                </button>
                                            `).join('')}
                                            ${(!run.available_weights || run.available_weights.length === 0) ? '<span class="text-muted">-</span>' : ''}
                                        </div>
                                    </td>
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
        // Navigate to experiments page to change model
        document.querySelector('.nav-link[data-page="experiments"]').click();
    });
    document.getElementById('skipFrameInput').addEventListener('change', (e) => setSkipFrames(e.target.value));
    document.getElementById('refreshExperiments').addEventListener('click', loadExperiments);

    console.log('✓ MLOps Frontend initialized');
});
