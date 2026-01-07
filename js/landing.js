/**
 * EstacaScan - Landing Page
 * Precarga el modelo en background mientras muestra la animación
 */

// ===============================================
// Configuration (same as app.js)
// ===============================================
const CONFIG = {
    MODEL_PATH: 'model/best.onnx',
    DB_NAME: 'EstacaScanDB',
    DB_VERSION: 1
};

// ===============================================
// DOM Elements
// ===============================================
const elements = {
    btnStart: document.getElementById('btn-start'),
    modelStatus: document.getElementById('model-status'),
    modelStatusText: document.getElementById('model-status-text'),
    modelProgressBar: document.getElementById('model-progress-bar'),
    bgParticles: document.getElementById('bg-particles')
};

// ===============================================
// Background Particles
// ===============================================
function createParticles() {
    const count = 20;
    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.top = `${Math.random() * 100}%`;
        particle.style.animationDelay = `${Math.random() * 15}s`;
        particle.style.animationDuration = `${10 + Math.random() * 10}s`;
        elements.bgParticles.appendChild(particle);
    }
}

// ===============================================
// Model Preloader (simplified from app.js)
// ===============================================
class ModelPreloader {
    constructor() {
        this.modelBuffer = null;
        this.isReady = false;
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

    async preload(onProgress) {
        try {
            onProgress('Verificando caché...', 10);

            // Check cache first
            let modelBuffer = await this.getFromCache();

            if (modelBuffer) {
                onProgress('Modelo en caché ✓', 100);
                this.modelBuffer = modelBuffer;
                this.isReady = true;
                return true;
            }

            // Download the model
            onProgress('Descargando modelo...', 15);

            const response = await fetch(CONFIG.MODEL_PATH);
            if (!response.ok) throw new Error('Failed to fetch model');

            const contentLength = response.headers.get('content-length');
            const total = parseInt(contentLength, 10) || 36000000;
            let loaded = 0;

            const reader = response.body.getReader();
            const chunks = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                loaded += value.length;
                const percent = Math.min(85, 15 + (loaded / total) * 70);
                const mb = Math.round(loaded / 1024 / 1024);
                onProgress(`Descargando... ${mb}MB`, percent);
            }

            // Combine chunks
            modelBuffer = new Uint8Array(loaded);
            let offset = 0;
            for (const chunk of chunks) {
                modelBuffer.set(chunk, offset);
                offset += chunk.length;
            }

            // Save to cache
            onProgress('Guardando en caché...', 90);
            await this.saveToCache(modelBuffer);

            onProgress('Modelo listo ✓', 100);
            this.modelBuffer = modelBuffer;
            this.isReady = true;
            return true;

        } catch (error) {
            console.error('Error preloading model:', error);
            onProgress('Error al cargar modelo', 0);
            return false;
        }
    }
}

// ===============================================
// Main
// ===============================================
const preloader = new ModelPreloader();

function updateProgress(text, percent) {
    elements.modelStatusText.textContent = text;
    elements.modelProgressBar.style.width = `${percent}%`;

    if (percent >= 100) {
        elements.modelStatus.classList.add('ready');
        elements.btnStart.disabled = false;
    }
}

async function init() {
    // Create background particles
    createParticles();

    // Start preloading the model
    const success = await preloader.preload(updateProgress);

    if (!success) {
        // Even if preload fails, let them try anyway
        elements.btnStart.disabled = false;
        elements.modelStatusText.textContent = 'Click para continuar';
    }
}

// Navigation to scanner
elements.btnStart.addEventListener('click', () => {
    window.location.href = 'scanner.html';
});

// Start initialization
init();
