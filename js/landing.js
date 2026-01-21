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
// ===============================================
// Background Particles System
// ===============================================
class ParticleSystem {
    constructor(container, count = 30) {
        this.container = container;
        this.count = count;
        this.particles = [];
        this.mouseX = -1000;
        this.mouseY = -1000;
        this.init();
    }

    init() {
        // Create particles
        for (let i = 0; i < this.count; i++) {
            this.createParticle();
        }

        // Mouse tracking
        document.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
        });

        document.addEventListener('mouseleave', () => {
            this.mouseX = -1000;
            this.mouseY = -1000;
        });

        // Start loop
        this.animate();
    }

    createParticle() {
        const el = document.createElement('div');
        el.className = 'particle';

        // Random properties
        const x = Math.random() * window.innerWidth;
        const y = Math.random() * window.innerHeight;
        const size = 2 + Math.random() * 4; // Variedad de tamaños

        // Style
        el.style.width = `${size}px`;
        el.style.height = `${size}px`;
        el.style.left = '0';
        el.style.top = '0';
        // Removemos la animación CSS via estilo en línea por si acaso, 
        // aunque lo ideal es quitarla del CSS
        el.style.animation = 'none';

        this.container.appendChild(el);

        this.particles.push({
            el,
            x: x,
            y: y,
            baseX: x,
            baseY: y,
            vx: (Math.random() - 0.5) * 0.5, // Velocidad flotante lenta
            vy: (Math.random() - 0.5) * 0.5,
            size: size,
            friction: 0.9,
            ease: 0.1
        });
    }

    animate() {
        // Dimensions check for responsiveness
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.particles.forEach(p => {
            // 1. Natural floating movement
            p.baseX += p.vx;
            p.baseY += p.vy;

            // Bounce off edges
            if (p.baseX < 0 || p.baseX > width) p.vx *= -1;
            if (p.baseY < 0 || p.baseY > height) p.vy *= -1;

            // 2. Magnetic Repulsion (Opposite magnets)
            const dx = this.mouseX - p.baseX;
            const dy = this.mouseY - p.baseY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const radius = 200; // Radio de influencia del mouse

            let targetX = p.baseX;
            let targetY = p.baseY;

            if (distance < radius) {
                // Calcular fuerza de repulsión (más cerca = más fuerza)
                const force = (radius - distance) / radius;
                const angle = Math.atan2(dy, dx);

                // Mover en dirección OPUESTA al mouse
                const moveX = Math.cos(angle) * force * 150; // Fuerza max 150px
                const moveY = Math.sin(angle) * force * 150;

                targetX -= moveX;
                targetY -= moveY;
            }

            // 3. Smooth transition (Linear Interpolation)
            p.x += (targetX - p.x) * p.ease;
            p.y += (targetY - p.y) * p.ease;

            // Update DOM
            p.el.style.transform = `translate(${p.x}px, ${p.y}px)`;
        });

        requestAnimationFrame(() => this.animate());
    }
}

function createParticles() {
    new ParticleSystem(elements.bgParticles, 40); // Increased density slightly
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
