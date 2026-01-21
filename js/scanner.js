/**
 * EstacaScan - Contador de Estacas
 * Aplicación web para detectar y contar estacas usando YOLO v8 con ONNX Runtime Web
 */

import { ParticleSystem } from './particles.js';

// ===============================================
// Configuration
// ===============================================
const CONFIG = {
    // Model settings
    MODEL_PATH: './model/best.onnx',
    MODEL_INPUT_SIZE: 640,

    // Detection thresholds
    CONFIDENCE_THRESHOLD: 0.25,
    DOUBT_THRESHOLD: 0.5,  // Below this = doubt
    IOU_THRESHOLD: 0.45,

    // Colors
    COLORS: {
        confirmed: '#10b981',
        doubt: '#f59e0b',
        rejected: '#ef4444'
    },

    // Cache settings
    DB_NAME: 'EstacaScanDB',
    DB_VERSION: 1
};

// ===============================================
// ONNX Model Manager
// ===============================================
class ONNXModelManager {
    constructor() {
        this.session = null;
        this.isLoading = false;
    }

    async loadModel(onProgress) {
        if (this.session) return this.session;
        if (this.isLoading) {
            // Wait for existing load
            while (this.isLoading) {
                await new Promise(r => setTimeout(r, 100));
            }
            return this.session;
        }

        this.isLoading = true;

        try {
            onProgress?.('Verificando caché del modelo...', 10);

            // Try to load from IndexedDB cache first
            let modelBuffer = await this.getFromCache();

            if (!modelBuffer) {
                onProgress?.('Descargando modelo (36 MB)...', 20);

                // Fetch the model
                const response = await fetch(CONFIG.MODEL_PATH);
                if (!response.ok) throw new Error('Failed to fetch model');

                const contentLength = response.headers.get('content-length');
                const total = parseInt(contentLength, 10) || 36000000;
                let loaded = 0;

                // Read the stream with progress
                const reader = response.body.getReader();
                const chunks = [];

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    loaded += value.length;
                    const percent = Math.min(70, 20 + (loaded / total) * 50);
                    onProgress?.(`Descargando modelo... ${Math.round(loaded / 1024 / 1024)}MB`, percent);
                }

                // Combine chunks
                modelBuffer = new Uint8Array(loaded);
                let offset = 0;
                for (const chunk of chunks) {
                    modelBuffer.set(chunk, offset);
                    offset += chunk.length;
                }

                // Cache for future use
                onProgress?.('Guardando en caché...', 75);
                await this.saveToCache(modelBuffer);
            } else {
                onProgress?.('Modelo cargado desde caché', 70);
            }

            // Create ONNX session
            onProgress?.('Inicializando modelo...', 80);

            // Import ONNX Runtime Web
            const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/esm/ort.min.js');

            // Configure ONNX Runtime
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';

            this.session = await ort.InferenceSession.create(modelBuffer.buffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            onProgress?.('Modelo listo', 100);
            console.log('✓ ONNX model loaded successfully');

            return this.session;

        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        } finally {
            this.isLoading = false;
        }
    }

    async getFromCache() {
        return new Promise((resolve) => {
            const request = indexedDB.open(CONFIG.DB_NAME, CONFIG.DB_VERSION);

            request.onerror = () => resolve(null);

            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('models')) {
                    db.createObjectStore('models');
                }
            };

            request.onsuccess = (e) => {
                const db = e.target.result;
                try {
                    const tx = db.transaction('models', 'readonly');
                    const store = tx.objectStore('models');
                    const getReq = store.get('yolo-model');

                    getReq.onsuccess = () => {
                        resolve(getReq.result || null);
                    };
                    getReq.onerror = () => resolve(null);
                } catch {
                    resolve(null);
                }
            };
        });
    }

    async saveToCache(buffer) {
        return new Promise((resolve) => {
            const request = indexedDB.open(CONFIG.DB_NAME, CONFIG.DB_VERSION);

            request.onerror = () => resolve(false);

            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('models')) {
                    db.createObjectStore('models');
                }
            };

            request.onsuccess = (e) => {
                const db = e.target.result;
                try {
                    const tx = db.transaction('models', 'readwrite');
                    const store = tx.objectStore('models');
                    store.put(buffer, 'yolo-model');
                    tx.oncomplete = () => resolve(true);
                    tx.onerror = () => resolve(false);
                } catch {
                    resolve(false);
                }
            };
        });
    }

    async runInference(imageData) {
        if (!this.session) {
            throw new Error('Model not loaded');
        }

        const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/esm/ort.min.js');

        // Create input tensor
        const inputTensor = new ort.Tensor('float32', imageData.data, imageData.shape);

        // Run inference
        const feeds = {};
        feeds[this.session.inputNames[0]] = inputTensor;

        const results = await this.session.run(feeds);

        // Get output
        const output = results[this.session.outputNames[0]];
        return output;
    }
}

// ===============================================
// Image Processor
// ===============================================
class ImageProcessor {
    static preprocess(img, targetSize = CONFIG.MODEL_INPUT_SIZE) {
        // Create canvas for preprocessing
        const canvas = document.createElement('canvas');
        canvas.width = targetSize;
        canvas.height = targetSize;
        const ctx = canvas.getContext('2d');

        // Calculate letterbox dimensions
        const scale = Math.min(targetSize / img.width, targetSize / img.height);
        const newWidth = Math.round(img.width * scale);
        const newHeight = Math.round(img.height * scale);
        const offsetX = (targetSize - newWidth) / 2;
        const offsetY = (targetSize - newHeight) / 2;

        // Fill with gray (letterbox)
        ctx.fillStyle = '#808080';
        ctx.fillRect(0, 0, targetSize, targetSize);

        // Draw scaled image
        ctx.drawImage(img, offsetX, offsetY, newWidth, newHeight);

        // Get image data
        const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
        const pixels = imageData.data;

        // Convert to float32 and normalize (RGB, NCHW format)
        const float32Data = new Float32Array(3 * targetSize * targetSize);

        for (let i = 0; i < targetSize * targetSize; i++) {
            float32Data[i] = pixels[i * 4] / 255.0;                    // R
            float32Data[targetSize * targetSize + i] = pixels[i * 4 + 1] / 255.0;     // G
            float32Data[2 * targetSize * targetSize + i] = pixels[i * 4 + 2] / 255.0; // B
        }

        return {
            data: float32Data,
            shape: [1, 3, targetSize, targetSize],
            scale: scale,
            offsetX: offsetX,
            offsetY: offsetY,
            originalWidth: img.width,
            originalHeight: img.height
        };
    }

    static postprocess(output, preprocessInfo) {
        const { scale, offsetX, offsetY, originalWidth, originalHeight } = preprocessInfo;
        const data = output.data;
        const [batch, features, numDetections] = output.dims;

        // YOLOv8 output format: [1, 5, 8400] where 5 = [x, y, w, h, conf]
        // For detection models with classes: [1, 4+num_classes, 8400]
        const detections = [];

        for (let i = 0; i < numDetections; i++) {
            // Get values for this detection
            const x = data[i];
            const y = data[numDetections + i];
            const w = data[2 * numDetections + i];
            const h = data[3 * numDetections + i];

            // Get confidence (assuming single class or max class conf)
            let confidence = 0;
            if (features === 5) {
                confidence = data[4 * numDetections + i];
            } else {
                // Multiple classes - get max
                for (let c = 4; c < features; c++) {
                    const classConf = data[c * numDetections + i];
                    if (classConf > confidence) confidence = classConf;
                }
            }

            if (confidence < CONFIG.CONFIDENCE_THRESHOLD) continue;

            // Convert from center/size to corner coordinates
            let x1 = x - w / 2;
            let y1 = y - h / 2;
            let x2 = x + w / 2;
            let y2 = y + h / 2;

            // Remove letterbox offset and scale back to original
            x1 = (x1 - offsetX) / scale;
            y1 = (y1 - offsetY) / scale;
            x2 = (x2 - offsetX) / scale;
            y2 = (y2 - offsetY) / scale;

            // Clamp to image bounds
            x1 = Math.max(0, Math.min(originalWidth, x1));
            y1 = Math.max(0, Math.min(originalHeight, y1));
            x2 = Math.max(0, Math.min(originalWidth, x2));
            y2 = Math.max(0, Math.min(originalHeight, y2));

            // Skip invalid boxes
            if (x2 <= x1 || y2 <= y1) continue;

            detections.push({
                id: detections.length,
                x1, y1, x2, y2,
                confidence,
                classId: 0
            });
        }

        // Apply NMS
        return ImageProcessor.nms(detections, CONFIG.IOU_THRESHOLD);
    }

    static nms(detections, iouThreshold) {
        // Sort by confidence descending
        detections.sort((a, b) => b.confidence - a.confidence);

        const kept = [];
        const suppressed = new Set();

        for (let i = 0; i < detections.length; i++) {
            if (suppressed.has(i)) continue;

            kept.push(detections[i]);

            for (let j = i + 1; j < detections.length; j++) {
                if (suppressed.has(j)) continue;

                const iou = ImageProcessor.calculateIoU(detections[i], detections[j]);
                if (iou > iouThreshold) {
                    suppressed.add(j);
                }
            }
        }

        return kept;
    }

    static calculateIoU(a, b) {
        const x1 = Math.max(a.x1, b.x1);
        const y1 = Math.max(a.y1, b.y1);
        const x2 = Math.min(a.x2, b.x2);
        const y2 = Math.min(a.y2, b.y2);

        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        const union = areaA + areaB - intersection;

        return intersection / union;
    }
}

// ===============================================
// State Management
// ===============================================
class AppState {
    constructor() {
        this.reset();
    }

    reset() {
        this.image = null;
        this.imageData = null;
        this.detections = [];
        this.confirmedDetections = [];
        this.rejectedDetections = [];
        this.doubts = [];
        this.currentDoubtIndex = 0;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.addMode = false; // Mode for adding stakes manually
    }

    get totalConfirmed() {
        return this.confirmedDetections.length;
    }

    get iaBase() {
        return this.detections.filter(d => d.confidence >= CONFIG.DOUBT_THRESHOLD).length;
    }

    get manuallyAccepted() {
        return this.confirmedDetections.filter(d => d.wasDoubt).length;
    }

    get manuallyAdded() {
        return this.confirmedDetections.filter(d => d.isManual).length;
    }
}

const state = new AppState();
const modelManager = new ONNXModelManager();

// ===============================================
// DOM Elements
// ===============================================
const elements = {
    // Particles
    bgParticles: document.getElementById('bg-particles'),

    // Screens
    screenUpload: document.getElementById('screen-upload'),
    screenLoading: document.getElementById('screen-loading'),
    screenResult: document.getElementById('screen-result'),
    screenReview: document.getElementById('screen-review'),

    // Upload
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    cameraInput: document.getElementById('camera-input'),
    btnCamera: document.getElementById('btn-camera'),

    // Loading
    loadingText: document.getElementById('loading-text'),
    progressFill: document.getElementById('progress-fill'),

    // Result
    canvas: document.getElementById('canvas'),
    imageContainer: document.getElementById('image-container'),
    btnZoomIn: document.getElementById('btn-zoom-in'),
    btnZoomOut: document.getElementById('btn-zoom-out'),
    btnZoomReset: document.getElementById('btn-zoom-reset'),
    btnAddMode: document.getElementById('btn-add-mode'),
    helpTip: document.getElementById('help-tip'),

    // Review
    reviewCanvas: document.getElementById('review-canvas'),
    reviewProgressFill: document.getElementById('review-progress-fill'),
    reviewCount: document.getElementById('review-count'),
    btnReject: document.getElementById('btn-reject'),
    btnAccept: document.getElementById('btn-accept'),

    // Footer
    footer: document.getElementById('footer'),
    totalCount: document.getElementById('total-count'),
    acceptedCount: document.getElementById('accepted-count'),
    iaBaseCount: document.getElementById('ia-base-count'),
    btnStartReview: document.getElementById('btn-start-review'),
    doubtCount: document.getElementById('doubt-count'),
    btnNewAnalysis: document.getElementById('btn-new-analysis'),

    // Status
    status: document.getElementById('status'),

    // History Modal
    historyModal: document.getElementById('history-modal'),
    btnHistory: document.getElementById('btn-history'),
    btnCloseHistory: document.getElementById('btn-close-history'),
    historyList: document.getElementById('history-list'),

    // History Filters
    historyFilterHarvester: document.getElementById('history-filter-harvester'),
    harvesterSelectContainer: document.getElementById('harvester-select-container'),
    harvesterSelectTrigger: document.getElementById('harvester-select-trigger'),
    harvesterSelectOptions: document.getElementById('harvester-select-options'),
    historyFilterDate: document.getElementById('history-filter-date'),

    // Metadata Panel
    metadataPanel: document.getElementById('metadata-panel'),
    metadataToggle: document.getElementById('metadata-toggle'),
    selectHarvester: document.getElementById('select-harvester'),
    selectSector: document.getElementById('select-sector'),
    selectBatch: document.getElementById('select-batch'),
    inputNotes: document.getElementById('input-notes'),

    // Buttons
    btnSaveAnalysis: document.getElementById('btn-save-analysis'),

    // Confirmation Modal
    confirmationModal: document.getElementById('confirmation-modal'),
    btnCancelDelete: document.getElementById('btn-cancel-delete'),
    btnConfirmDelete: document.getElementById('btn-confirm-delete'),

    // Toast
    toastContainer: document.getElementById('toast-container')
};

// ===============================================
// Screen Management
// ===============================================
function showScreen(screenName) {
    const screens = ['upload', 'loading', 'result', 'review'];
    screens.forEach(name => {
        const screen = elements[`screen${name.charAt(0).toUpperCase() + name.slice(1)}`];
        if (screen) {
            screen.classList.toggle('hidden', name !== screenName);
        }
    });

    // Show/hide footer based on screen
    if (screenName === 'upload' || screenName === 'loading') {
        elements.footer.style.display = 'none';
    } else {
        elements.footer.style.display = 'flex';
    }
}

// ===============================================
// Toast Notifications
// ===============================================
function showToast(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ===============================================
// Status Updates
// ===============================================
function setStatus(text, type = 'active') {
    const statusText = elements.status.querySelector('.status-text');
    statusText.textContent = text;
    elements.status.className = `status ${type}`;
}

// ===============================================
// File Handling
// ===============================================
function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) {
        showToast('Por favor selecciona una imagen válida');
        return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
        state.imageData = e.target.result;
        await processImage(e.target.result);
    };
    reader.readAsDataURL(file);
}

// ===============================================
// Image Processing
// ===============================================
async function processImage(imageDataUrl) {
    showScreen('loading');
    setStatus('ANALIZANDO', 'loading');

    try {
        // Load model first (if not cached, this downloads it)
        await modelManager.loadModel((text, progress) => {
            elements.loadingText.textContent = text;
            elements.progressFill.style.width = `${progress}%`;
        });

        // Load image
        elements.loadingText.textContent = 'Procesando imagen...';
        const img = await loadImage(imageDataUrl);
        state.image = img;

        // Run detection
        elements.loadingText.textContent = 'Detectando estacas...';
        elements.progressFill.style.width = '85%';

        const preprocessed = ImageProcessor.preprocess(img);
        const output = await modelManager.runInference(preprocessed);
        const detections = ImageProcessor.postprocess(output, preprocessed);

        elements.progressFill.style.width = '95%';

        // Process results
        processDetections(detections);
        elements.progressFill.style.width = '100%';

        // Show results
        setTimeout(() => {
            showResults();
            setStatus('ANÁLISIS ACTIVO', 'active');
        }, 300);

    } catch (error) {
        console.error('Detection error:', error);
        showToast('Error al analizar la imagen: ' + error.message);
        showScreen('upload');
        setStatus('ERROR', 'error');
    }
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

// ===============================================
// Process Detections
// ===============================================
function processDetections(detections) {
    state.detections = detections;
    state.confirmedDetections = [];
    state.rejectedDetections = [];
    state.doubts = [];

    detections.forEach(det => {
        if (det.confidence >= CONFIG.DOUBT_THRESHOLD) {
            // High confidence - auto confirm
            state.confirmedDetections.push({ ...det, wasDoubt: false });
        } else if (det.confidence >= CONFIG.CONFIDENCE_THRESHOLD) {
            // Low confidence - mark as doubt
            state.doubts.push(det);
        }
        // Below CONFIDENCE_THRESHOLD - ignore
    });

    state.currentDoubtIndex = 0;
}

// ===============================================
// Show Results
// ===============================================
function showResults() {
    showScreen('result');
    updateStats();
    drawCanvas();

    // Show/hide review button
    if (state.doubts.length > 0) {
        elements.btnStartReview.classList.remove('hidden');
        elements.doubtCount.textContent = state.doubts.length;
    } else {
        elements.btnStartReview.classList.add('hidden');
    }

    elements.btnNewAnalysis.classList.remove('hidden');
}

// ===============================================
// Canvas Drawing
// ===============================================
function drawCanvas() {
    const ctx = elements.canvas.getContext('2d');
    const img = state.image;

    // Set canvas size
    elements.canvas.width = img.width;
    elements.canvas.height = img.height;

    // Apply zoom and pan
    ctx.save();
    ctx.scale(state.zoom, state.zoom);
    ctx.translate(state.panX, state.panY);

    // Draw image
    ctx.drawImage(img, 0, 0);

    // Draw confirmed detections (green)
    state.confirmedDetections.forEach(det => {
        drawBox(ctx, det, CONFIG.COLORS.confirmed);
    });

    // Draw doubts (yellow)
    state.doubts.forEach(det => {
        drawBox(ctx, det, CONFIG.COLORS.doubt);
    });

    // Draw rejected (red, with X)
    state.rejectedDetections.forEach(det => {
        drawBox(ctx, det, CONFIG.COLORS.rejected, true);
    });

    ctx.restore();
}

function drawBox(ctx, det, color, isRejected = false) {
    const { x1, y1, x2, y2 } = det;
    const width = x2 - x1;
    const height = y2 - y1;

    // Draw box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, width, height);

    // Draw corner markers
    const cornerSize = Math.min(15, width * 0.2, height * 0.2);
    ctx.fillStyle = color;

    // Top-left
    ctx.fillRect(x1, y1, cornerSize, 3);
    ctx.fillRect(x1, y1, 3, cornerSize);

    // Top-right
    ctx.fillRect(x2 - cornerSize, y1, cornerSize, 3);
    ctx.fillRect(x2 - 3, y1, 3, cornerSize);

    // Bottom-left
    ctx.fillRect(x1, y2 - 3, cornerSize, 3);
    ctx.fillRect(x1, y2 - cornerSize, 3, cornerSize);

    // Bottom-right
    ctx.fillRect(x2 - cornerSize, y2 - 3, cornerSize, 3);
    ctx.fillRect(x2 - 3, y2 - cornerSize, 3, cornerSize);

    // If rejected, draw X
    if (isRejected) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x1 + 5, y1 + 5);
        ctx.lineTo(x2 - 5, y2 - 5);
        ctx.moveTo(x2 - 5, y1 + 5);
        ctx.lineTo(x1 + 5, y2 - 5);
        ctx.stroke();
    }
}

// ===============================================
// Stats Update
// ===============================================
function updateStats() {
    elements.totalCount.textContent = state.totalConfirmed;
    elements.acceptedCount.textContent = `ACEPTADAS: ${state.manuallyAccepted}`;
    elements.iaBaseCount.textContent = `IA BASE: ${state.iaBase}`;

    if (state.doubts.length > 0) {
        elements.doubtCount.textContent = state.doubts.length;
        elements.btnStartReview.classList.remove('hidden');
    } else {
        elements.btnStartReview.classList.add('hidden');
    }
}

// ===============================================
// Review Mode
// ===============================================
function startReview() {
    if (state.doubts.length === 0) {
        showToast('No hay dudas por revisar');
        return;
    }

    state.currentDoubtIndex = 0;
    showScreen('review');
    showCurrentDoubt();
}

function showCurrentDoubt() {
    const doubt = state.doubts[state.currentDoubtIndex];
    if (!doubt) {
        endReview();
        return;
    }

    // Update progress
    const progress = ((state.currentDoubtIndex + 1) / state.doubts.length) * 100;
    elements.reviewProgressFill.style.width = `${progress}%`;
    elements.reviewCount.textContent = `${state.currentDoubtIndex + 1} / ${state.doubts.length}`;

    // Draw zoomed crop of the doubt
    drawDoubtCrop(doubt);
}

function drawDoubtCrop(doubt) {
    const ctx = elements.reviewCanvas.getContext('2d');
    const img = state.image;

    // Calculate crop area with padding
    const padding = 50;
    const width = doubt.x2 - doubt.x1;
    const height = doubt.y2 - doubt.y1;

    const cropX = Math.max(0, doubt.x1 - padding);
    const cropY = Math.max(0, doubt.y1 - padding);
    const cropWidth = Math.min(img.width - cropX, width + padding * 2);
    const cropHeight = Math.min(img.height - cropY, height + padding * 2);

    // Set canvas size (max 400x400)
    const maxSize = 400;
    const scale = Math.min(maxSize / cropWidth, maxSize / cropHeight);
    elements.reviewCanvas.width = cropWidth * scale;
    elements.reviewCanvas.height = cropHeight * scale;

    // Draw cropped and scaled image
    ctx.drawImage(
        img,
        cropX, cropY, cropWidth, cropHeight,
        0, 0, cropWidth * scale, cropHeight * scale
    );

    // Draw box on crop
    const boxX = (doubt.x1 - cropX) * scale;
    const boxY = (doubt.y1 - cropY) * scale;
    const boxWidth = width * scale;
    const boxHeight = height * scale;

    ctx.strokeStyle = CONFIG.COLORS.confirmed;
    ctx.lineWidth = 3;
    ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
}

function acceptDoubt() {
    const doubt = state.doubts[state.currentDoubtIndex];
    if (doubt) {
        state.confirmedDetections.push({ ...doubt, wasDoubt: true });
        state.doubts.splice(state.currentDoubtIndex, 1);
        updateStats();

        if (state.doubts.length === 0 || state.currentDoubtIndex >= state.doubts.length) {
            endReview();
        } else {
            showCurrentDoubt();
        }
    }
}

function rejectDoubt() {
    const doubt = state.doubts[state.currentDoubtIndex];
    if (doubt) {
        state.rejectedDetections.push(doubt);
        state.doubts.splice(state.currentDoubtIndex, 1);
        updateStats();

        if (state.doubts.length === 0 || state.currentDoubtIndex >= state.doubts.length) {
            endReview();
        } else {
            showCurrentDoubt();
        }
    }
}

function endReview() {
    showScreen('result');
    drawCanvas();
    updateStats();

    if (state.doubts.length === 0) {
        showToast('✓ Revisión completada');
    }
}

// ===============================================
// Zoom Controls (smoother zoom for trackpad)
// ===============================================
let zoomAccumulator = 0;
const ZOOM_SENSITIVITY = 0.08; // Lower = slower zoom

function zoomIn() {
    state.zoom = Math.min(state.zoom * 1.15, 5);
    drawCanvas();
}

function zoomOut() {
    state.zoom = Math.max(state.zoom / 1.15, 0.3);
    drawCanvas();
}

function zoomReset() {
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    drawCanvas();
}

// Smooth wheel zoom handler
function handleWheelZoom(e) {
    e.preventDefault();

    // Accumulate small deltas for smoother trackpad zoom
    zoomAccumulator += e.deltaY * ZOOM_SENSITIVITY;

    if (Math.abs(zoomAccumulator) >= 1) {
        const zoomFactor = 1 + Math.abs(zoomAccumulator) * 0.02;
        if (zoomAccumulator > 0) {
            state.zoom = Math.max(state.zoom / zoomFactor, 0.3);
        } else {
            state.zoom = Math.min(state.zoom * zoomFactor, 5);
        }
        zoomAccumulator = 0;
        drawCanvas();
    }
}

// ===============================================
// Click on Canvas to Toggle/Add Detection
// ===============================================
function handleCanvasClick(e) {
    // Ignore if we were dragging
    if (wasDragging) {
        wasDragging = false;
        return;
    }

    const rect = elements.canvas.getBoundingClientRect();
    const scaleX = elements.canvas.width / rect.width;
    const scaleY = elements.canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX / state.zoom - state.panX;
    const y = (e.clientY - rect.top) * scaleY / state.zoom - state.panY;

    // Check if clicked on a confirmed detection
    const clickedConfirmed = state.confirmedDetections.findIndex(det =>
        x >= det.x1 && x <= det.x2 && y >= det.y1 && y <= det.y2
    );

    if (clickedConfirmed !== -1) {
        const det = state.confirmedDetections.splice(clickedConfirmed, 1)[0];
        state.rejectedDetections.push(det);
        showToast('Estaca descartada');
        drawCanvas();
        updateStats();
        return;
    }

    // Check if clicked on a rejected detection (to restore it)
    const clickedRejected = state.rejectedDetections.findIndex(det =>
        x >= det.x1 && x <= det.x2 && y >= det.y1 && y <= det.y2
    );

    if (clickedRejected !== -1) {
        const det = state.rejectedDetections.splice(clickedRejected, 1)[0];
        state.confirmedDetections.push({ ...det, wasDoubt: true });
        showToast('Estaca restaurada');
        drawCanvas();
        updateStats();
        return;
    }

    // Check if clicked on a doubt
    const clickedDoubt = state.doubts.findIndex(det =>
        x >= det.x1 && x <= det.x2 && y >= det.y1 && y <= det.y2
    );

    if (clickedDoubt !== -1) {
        const det = state.doubts.splice(clickedDoubt, 1)[0];
        state.confirmedDetections.push({ ...det, wasDoubt: true });
        showToast('Duda confirmada como estaca');
        drawCanvas();
        updateStats();
        return;
    }

    // If in add mode (or Ctrl+click), add a new stake at this position
    if (state.addMode || e.ctrlKey || e.metaKey) {
        addManualStake(x, y);
    }
}

// Add a manual stake at the given position
function addManualStake(x, y) {
    // Default box size based on average detection size or fixed value
    const avgSize = calculateAverageBoxSize();
    const halfSize = avgSize / 2;

    const newDet = {
        id: Date.now(),
        x1: x - halfSize,
        y1: y - halfSize,
        x2: x + halfSize,
        y2: y + halfSize,
        confidence: 1.0,
        classId: 0,
        wasDoubt: false,
        isManual: true
    };

    // Clamp to image bounds
    newDet.x1 = Math.max(0, newDet.x1);
    newDet.y1 = Math.max(0, newDet.y1);
    newDet.x2 = Math.min(state.image.width, newDet.x2);
    newDet.y2 = Math.min(state.image.height, newDet.y2);

    state.confirmedDetections.push(newDet);
    showToast('✓ Estaca agregada manualmente');
    drawCanvas();
    updateStats();
}

function calculateAverageBoxSize() {
    const allDets = [...state.confirmedDetections, ...state.doubts];
    if (allDets.length === 0) return 40; // Default size

    let totalSize = 0;
    allDets.forEach(det => {
        const w = det.x2 - det.x1;
        const h = det.y2 - det.y1;
        totalSize += (w + h) / 2;
    });
    return totalSize / allDets.length;
}

// ===============================================
// Pan (Drag) Controls
// ===============================================
let isDragging = false;
let wasDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let lastX = 0;
let lastY = 0;
const DRAG_THRESHOLD = 5; // pixels moved to consider it a drag, not a click

function handleMouseDown(e) {
    isDragging = true;
    wasDragging = false;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    lastX = e.clientX;
    lastY = e.clientY;
    elements.canvas.style.cursor = 'grabbing';
}

function handleMouseMove(e) {
    if (!isDragging) return;

    // Check if we've moved enough to consider this a drag
    const movedX = Math.abs(e.clientX - dragStartX);
    const movedY = Math.abs(e.clientY - dragStartY);
    if (movedX > DRAG_THRESHOLD || movedY > DRAG_THRESHOLD) {
        wasDragging = true;
    }

    const deltaX = (e.clientX - lastX) / state.zoom;
    const deltaY = (e.clientY - lastY) / state.zoom;

    state.panX += deltaX;
    state.panY += deltaY;

    lastX = e.clientX;
    lastY = e.clientY;

    drawCanvas();
}

function handleMouseUp() {
    isDragging = false;
    elements.canvas.style.cursor = state.addMode ? 'crosshair' : 'grab';
}

// ===============================================
// Add Mode Toggle
// ===============================================
function toggleAddMode() {
    state.addMode = !state.addMode;

    // Update button state
    elements.btnAddMode.classList.toggle('active', state.addMode);

    // Update cursor
    elements.canvas.style.cursor = state.addMode ? 'crosshair' : 'grab';

    // Update container class for cursor override
    elements.imageContainer.classList.toggle('add-mode', state.addMode);

    // Show toast
    if (state.addMode) {
        showToast('Modo agregar: click para marcar estacas');
        showHelpTip();
    } else {
        showToast('Modo agregar desactivado');
        hideHelpTip();
    }
}

function showHelpTip() {
    if (elements.helpTip) {
        elements.helpTip.classList.add('visible');
        // Auto-hide after 5 seconds
        setTimeout(hideHelpTip, 5000);
    }
}

function hideHelpTip() {
    if (elements.helpTip) {
        elements.helpTip.classList.remove('visible');
    }
}

// ===============================================
// New Analysis
// ===============================================
function newAnalysis() {
    state.reset();
    showScreen('upload');
    setStatus('LISTO', 'active');
    elements.btnStartReview.classList.add('hidden');
    elements.btnNewAnalysis.classList.add('hidden');

    // Reset add mode button
    if (elements.btnAddMode) {
        elements.btnAddMode.classList.remove('active');
    }
    if (elements.imageContainer) {
        elements.imageContainer.classList.remove('add-mode');
    }
}

// ===============================================
// Event Listeners
// ===============================================
function initEventListeners() {
    // File input
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Camera input
    elements.cameraInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    elements.btnCamera.addEventListener('click', () => {
        elements.cameraInput.click();
    });

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('drag-over');
    });

    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('drag-over');
    });

    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // Click on upload area
    elements.uploadArea.addEventListener('click', (e) => {
        if (e.target === elements.uploadArea || e.target.classList.contains('upload-icon') || e.target.classList.contains('upload-text')) {
            elements.fileInput.click();
        }
    });

    // Zoom controls
    elements.btnZoomIn.addEventListener('click', zoomIn);
    elements.btnZoomOut.addEventListener('click', zoomOut);
    elements.btnZoomReset.addEventListener('click', zoomReset);

    // Add mode toggle
    if (elements.btnAddMode) {
        elements.btnAddMode.addEventListener('click', toggleAddMode);
    }

    // Mouse wheel zoom (smooth)
    elements.imageContainer.addEventListener('wheel', handleWheelZoom, { passive: false });

    // Canvas click (to toggle detection)
    elements.canvas.addEventListener('click', handleCanvasClick);

    // Canvas pan (drag)
    elements.canvas.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    // Touch events for mobile (pan + pinch-to-zoom)
    let touchStartX = 0;
    let touchStartY = 0;
    let lastTouchDistance = 0;
    let isTouchDragging = false;
    let wasTouchDragging = false;
    let touchDragStartX = 0;
    let touchDragStartY = 0;
    let lastDrawTime = 0;
    const TOUCH_THROTTLE_MS = 16; // ~60fps

    function getTouchDistance(touches) {
        const dx = touches[0].clientX - touches[1].clientX;
        const dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function getTouchCenter(touches) {
        return {
            x: (touches[0].clientX + touches[1].clientX) / 2,
            y: (touches[0].clientY + touches[1].clientY) / 2
        };
    }

    elements.canvas.addEventListener('touchstart', (e) => {
        e.preventDefault(); // Prevent browser gestures

        if (e.touches.length === 1) {
            // Single finger - pan
            isTouchDragging = true;
            wasTouchDragging = false;
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
            touchDragStartX = e.touches[0].clientX;
            touchDragStartY = e.touches[0].clientY;
        } else if (e.touches.length === 2) {
            // Two fingers - pinch zoom
            isTouchDragging = false;
            lastTouchDistance = getTouchDistance(e.touches);
        }
    }, { passive: false });

    elements.canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();

        // Throttle for performance
        const now = Date.now();
        if (now - lastDrawTime < TOUCH_THROTTLE_MS) return;
        lastDrawTime = now;

        if (e.touches.length === 1 && isTouchDragging) {
            // Single finger pan
            const movedX = Math.abs(e.touches[0].clientX - touchDragStartX);
            const movedY = Math.abs(e.touches[0].clientY - touchDragStartY);
            if (movedX > DRAG_THRESHOLD || movedY > DRAG_THRESHOLD) {
                wasTouchDragging = true;
            }

            const deltaX = (e.touches[0].clientX - touchStartX) / state.zoom;
            const deltaY = (e.touches[0].clientY - touchStartY) / state.zoom;

            state.panX += deltaX;
            state.panY += deltaY;

            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;

            drawCanvas();
        } else if (e.touches.length === 2) {
            // Pinch to zoom
            const currentDistance = getTouchDistance(e.touches);

            if (lastTouchDistance > 0) {
                const scale = currentDistance / lastTouchDistance;
                const newZoom = state.zoom * scale;

                // Clamp zoom
                state.zoom = Math.max(0.3, Math.min(5, newZoom));
                drawCanvas();
            }

            lastTouchDistance = currentDistance;
        }
    }, { passive: false });

    elements.canvas.addEventListener('touchend', (e) => {
        if (e.touches.length === 0) {
            // All fingers lifted
            if (!wasTouchDragging && isTouchDragging) {
                // It was a tap, not a drag - treat as click
                // Create a synthetic click event
                const touch = e.changedTouches[0];
                const clickEvent = {
                    clientX: touch.clientX,
                    clientY: touch.clientY,
                    ctrlKey: false,
                    metaKey: false
                };
                handleCanvasClick(clickEvent);
            }
            isTouchDragging = false;
            wasTouchDragging = false;
            lastTouchDistance = 0;
        } else if (e.touches.length === 1) {
            // One finger still on screen, switch to pan mode
            isTouchDragging = true;
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
            lastTouchDistance = 0;
        }
    });

    // Review buttons
    elements.btnStartReview.addEventListener('click', startReview);
    elements.btnAccept.addEventListener('click', acceptDoubt);
    elements.btnReject.addEventListener('click', rejectDoubt);

    // New analysis
    elements.btnNewAnalysis.addEventListener('click', newAnalysis);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (elements.screenReview.classList.contains('hidden')) return;

        if (e.key === 'ArrowLeft' || e.key === 'x' || e.key === 'X') {
            rejectDoubt();
        } else if (e.key === 'ArrowRight' || e.key === 'Enter' || e.key === ' ') {
            acceptDoubt();
        }
    });

    // Supabase & Metadata
    if (elements.btnSaveAnalysis) elements.btnSaveAnalysis.addEventListener('click', saveAnalysis);
    if (elements.metadataToggle) elements.metadataToggle.addEventListener('click', toggleMetadataPanel);
    if (elements.btnHistory) elements.btnHistory.addEventListener('click', openHistoryModal);
    if (elements.btnCloseHistory) elements.btnCloseHistory.addEventListener('click', closeHistoryModal);
}

// ===============================================
// Custom Dropdown Logic
// ===============================================
function initCustomDropdown() {
    const trigger = elements.harvesterSelectTrigger;
    const input = elements.historyFilterHarvester;
    const optionsList = elements.harvesterSelectOptions;
    const container = elements.harvesterSelectContainer;

    if (!trigger || !input) return;

    // Toggle dropdown on click (if not input click)
    trigger.addEventListener('click', (e) => {
        if (e.target !== input) {
            optionsList.classList.toggle('hidden');
            if (!optionsList.classList.contains('hidden')) input.focus();
        }
    });

    // Open on input focus
    input.addEventListener('focus', () => {
        optionsList.classList.remove('hidden');
    });

    // Filter options on input
    input.addEventListener('input', () => {
        input.dataset.selectedId = ''; // Reset ID on manual input
        const filter = input.value.toLowerCase();
        const options = optionsList.querySelectorAll('li');

        options.forEach(option => {
            const text = option.textContent.toLowerCase();
            if (text.includes(filter)) {
                option.style.display = 'block';
            } else {
                option.style.display = 'none';
            }
        });
    });

    // Select option
    optionsList.addEventListener('click', (e) => {
        if (e.target.tagName === 'LI') {
            const value = e.target.dataset.value;
            const text = e.target.textContent;

            // Set value
            input.value = text;
            input.dataset.selectedId = value; // Store ID

            // Highlight selected
            optionsList.querySelectorAll('li').forEach(li => li.classList.remove('selected'));
            e.target.classList.add('selected');

            // Close and trigger load
            optionsList.classList.add('hidden');
            loadHistoryList();
        }
    });

    // Close when clicking outside
    document.addEventListener('click', (e) => {
        if (!container.contains(e.target)) {
            optionsList.classList.add('hidden');
        }
    });
}

// (createParticles removed: using shared ParticleSystem module)

// ===============================================
// Supabase Functions
// ===============================================

// Load harvesters, sectors, and batches into selectors
async function loadMetadataOptions() {
    try {
        // Load harvesters
        const harvesters = await HarvesterService.getAll();
        elements.selectHarvester.innerHTML = '<option value="">Sin asignar</option>';

        // Reset custom dropdown options
        if (elements.harvesterSelectOptions) {
            elements.harvesterSelectOptions.innerHTML = '<li data-value="" class="selected">Todos los cosechadores</li>';

            harvesters.forEach(h => {
                const option = document.createElement('option');
                option.value = h.id;
                option.textContent = h.name;
                elements.selectHarvester.appendChild(option);

                // Add to custom dropdown
                const li = document.createElement('li');
                li.dataset.value = h.id;
                li.textContent = h.name;
                elements.harvesterSelectOptions.appendChild(li);
            });
        }

        // Load sectors
        const sectors = await SectorService.getAll();
        elements.selectSector.innerHTML = '<option value="">Sin asignar</option>';
        sectors.forEach(s => {
            const option = document.createElement('option');
            option.value = s.id;
            option.textContent = `${s.name} (${s.code || 'S/C'})`;
            elements.selectSector.appendChild(option);
        });

        // Load batches
        const batches = await BatchService.getAll();
        elements.selectBatch.innerHTML = '<option value="">Sin asignar</option>';
        batches.forEach(b => {
            const option = document.createElement('option');
            option.value = b.id;
            option.textContent = `${b.code} - ${b.variety}`;
            elements.selectBatch.appendChild(option);
        });

        console.log('✅ Metadata options loaded');
    } catch (error) {
        console.error('❌ Error loading metadata:', error);
        showToast('Error al cargar datos', 'error');
    }
}

// Toggle metadata panel
function toggleMetadataPanel() {
    elements.metadataPanel.classList.toggle('collapsed');
}

// Open history modal
async function openHistoryModal() {
    elements.historyModal.classList.remove('hidden');
    await loadHistoryList();
}

// Close history modal
function closeHistoryModal() {
    elements.historyModal.classList.add('hidden');
}

// Load and display history list
async function loadHistoryList() {
    try {
        elements.historyList.innerHTML = '<div class="history-loading">Cargando...</div>';

        // Get filter values
        const harvesterSelectedId = elements.historyFilterHarvester ? elements.historyFilterHarvester.dataset.selectedId : null;
        const harvesterText = elements.historyFilterHarvester ? elements.historyFilterHarvester.value.trim().toLowerCase() : '';
        const date = elements.historyFilterDate ? elements.historyFilterDate.value || null : null;

        // Fetch recent analysis
        let analyses = await AnalysisService.getRecent(50);

        // Apply filters
        // Priority to ID match if selected from dropdown
        if (harvesterSelectedId) {
            analyses = analyses.filter(a => a.harvester_id === harvesterSelectedId);
        } else if (harvesterText) {
            // Fallback to text search if typed manually
            analyses = analyses.filter(a =>
                a.harvester_name.toLowerCase().includes(harvesterText)
            );
        }
        if (date) {
            analyses = analyses.filter(a => a.created_at.startsWith(date));
        }

        // Display results
        if (analyses.length === 0) {
            elements.historyList.innerHTML = '<div class="history-loading">No hay análisis registrados</div>';
            return;
        }

        elements.historyList.innerHTML = '';
        analyses.forEach(analysis => {
            const item = createHistoryItem(analysis);
            elements.historyList.appendChild(item);
        });

    } catch (error) {
        console.error('Error loading history:', error);
        elements.historyList.innerHTML = '<div class="history-loading">Error al cargar el historial</div>';
    }
}

// Create history item HTML
function createHistoryItem(analysis) {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.dataset.id = analysis.id;

    const date = new Date(analysis.created_at);
    const dateStr = date.toLocaleString('es-AR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });

    item.innerHTML = `
        <div class="history-item-header">
            <span class="history-item-date">${dateStr}</span>
            <div class="history-item-actions">
                <button class="btn-history-action edit" data-id="${analysis.id}">Editar</button>
                <button class="btn-history-action delete" data-id="${analysis.id}">Eliminar</button>
            </div>
        </div>
        <div class="history-item-info">
            <div class="history-info-row">
                <span class="history-info-label">Estacas</span>
                <span class="history-info-value">${analysis.total_confirmed}</span>
            </div>
            <div class="history-info-row">
                <span class="history-info-label">Cosechador</span>
                <span class="history-info-value">${analysis.harvester_name || 'Sin asignar'}</span>
            </div>
            <div class="history-info-row">
                <span class="history-info-label">Sector</span>
                <span class="history-info-value">${analysis.sector_name || 'Sin asignar'}</span>
            </div>
            <div class="history-info-row">
                <span class="history-info-label">Lote</span>
                <span class="history-info-value">${analysis.batch_code || 'Sin asignar'}</span>
            </div>
        </div>
    `;

    // Add event listeners
    item.querySelector('.edit').addEventListener('click', () => editAnalysis(analysis.id));
    item.querySelector('.delete').addEventListener('click', () => deleteAnalysis(analysis.id));

    return item;
}

// Edit analysis (open edit modal - simplified for now)
async function editAnalysis(id) {
    showToast('Función de edición en desarrollo');
    // TODO: Implement edit modal
}

// Delete analysis with Custom Modal
function deleteAnalysis(id) {
    // Show custom modal
    elements.confirmationModal.classList.remove('hidden');

    // Use one-time event handlers to avoid stacking listeners
    elements.btnConfirmDelete.onclick = async () => {
        elements.confirmationModal.classList.add('hidden');
        elements.btnConfirmDelete.onclick = null; // Clean up

        try {
            await AnalysisService.delete(id);
            showToast('✅ Análisis eliminado');
            await loadHistoryList();
        } catch (error) {
            console.error('Error deleting analysis:', error);
            showToast('❌ Error al eliminar', 'error');
        }
    };

    elements.btnCancelDelete.onclick = () => {
        elements.confirmationModal.classList.add('hidden');
        elements.btnCancelDelete.onclick = null; // Clean up
    };
}

// Save current analysis to Supabase
async function saveAnalysis() {
    if (!state.image || state.totalConfirmed === 0) {
        showToast('No hay análisis para guardar');
        return;
    }

    // Validate required field: harvester
    if (!elements.selectHarvester.value) {
        showToast('⚠️ Debe seleccionar un cosechador', 'error');
        elements.selectHarvester.style.borderColor = '#ef4444';
        elements.selectHarvester.focus();

        // Remove error highlight after 3 seconds
        setTimeout(() => {
            elements.selectHarvester.style.borderColor = '';
        }, 3000);
        return;
    }


    try {
        elements.btnSaveAnalysis.classList.add('saving');
        elements.btnSaveAnalysis.textContent = 'GUARDANDO...';

        // Get global config price
        const { data: config } = await supabase.from('config').select('price_per_stake').single();
        const pricePerStake = config?.price_per_stake || 0;

        const analysisData = {
            harvesterId: elements.selectHarvester.value || null,
            sectorId: elements.selectSector.value || null,
            batchId: elements.selectBatch.value || null,
            notes: elements.inputNotes.value || null,
            totalConfirmed: state.totalConfirmed,
            iaBase: state.iaBase,
            manuallyAccepted: state.manuallyAccepted || 0,
            manuallyAdded: state.manuallyAdded || 0,
            rejected: state.rejectedDetections.length,
            doubtsReviewed: state.initialDoubtsCount || 0,
            imageWidth: state.image.width,
            imageHeight: state.image.height,
            imageHash: state.lastImageHash, // Enviar hash
            processingTime: state.processingTime || 0
        };

        // Save analysis
        const savedAnalysis = await AnalysisService.save(analysisData);

        // Update harvester stats if assigned
        if (analysisData.harvesterId) {
            await HarvesterService.updateStats(analysisData.harvesterId, pricePerStake);
        }

        // Update batch total if assigned
        if (analysisData.batchId) {
            await BatchService.updateTotal(analysisData.batchId);
        }

        elements.btnSaveAnalysis.classList.remove('saving');
        elements.btnSaveAnalysis.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg> GUARDADO';
        showToast('✅ Análisis guardado');

        // Smooth transition to new analysis
        setTimeout(() => {
            // Fade out current screen
            elements.screenResult.style.transition = 'opacity 0.5s ease';
            elements.screenResult.style.opacity = '0';

            setTimeout(() => {
                elements.screenResult.style.opacity = '1';
                elements.screenResult.style.transition = '';
                newAnalysis();
            }, 500);
        }, 800);


    } catch (error) {
        console.error('❌ Error saving analysis:', error);
        elements.btnSaveAnalysis.classList.remove('saving');
        elements.btnSaveAnalysis.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> GUARDAR';
        showToast('❌ Error al guardar: ' + error.message, 'error');
    }
}
// ===============================================
// Initialize
// ===============================================
function init() {
    // Start Particles
    if (elements.bgParticles) {
        new ParticleSystem(elements.bgParticles, 40);
    }

    showScreen('upload');
    elements.footer.style.display = 'none';
    initEventListeners();
    initCustomDropdown(); // Init custom select

    // Check Supabase connection
    import('./supabase-client.js').then(({ testConnection }) => {
        testConnection();
    });

    // Load initial data
    loadMetadataOptions();

    setStatus('LISTO', 'active');

    console.log('🌲 EstacaScan initialized');
    console.log('📦 Model will be loaded on first image upload');
}

// Start app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
