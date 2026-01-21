/**
 * EstacaScan - Particle System
 * Efecto de fondo con partículas flotantes y repulsión magnética al mouse.
 */

export class ParticleSystem {
    constructor(container, count = 30) {
        this.container = container;
        this.count = count;
        this.particles = [];
        this.mouseX = -1000;
        this.mouseY = -1000;
        this.init();
    }

    init() {
        // Limpiar contenedor por si acaso
        this.container.innerHTML = '';

        // Create particles
        for (let i = 0; i < this.count; i++) {
            this.createParticle();
        }

        // Mouse tracking global
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
        const size = 2 + Math.random() * 4;

        // Style init
        el.style.width = `${size}px`;
        el.style.height = `${size}px`;
        el.style.left = '0';
        el.style.top = '0';
        el.style.transform = `translate(${x}px, ${y}px)`; // Posición inicial

        this.container.appendChild(el);

        this.particles.push({
            el,
            x: x,
            y: y,
            baseX: x,
            baseY: y,
            vx: (Math.random() - 0.5) * 0.5, // Velocidad flotante muy lenta
            vy: (Math.random() - 0.5) * 0.5,
            size: size,
            // Ajustamos 'ease' para que sea más lento ("pesado")
            // 0.1 era rápido, 0.03 es mucho más suave/lento
            ease: 0.03,
            friction: 0.95
        });
    }

    animate() {
        // Dimensions check
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.particles.forEach(p => {
            // 1. Natural floating movement
            p.baseX += p.vx;
            p.baseY += p.vy;

            // Bounce off edges (screen wrapping or bounce)
            if (p.baseX < 0 || p.baseX > width) p.vx *= -1;
            if (p.baseY < 0 || p.baseY > height) p.vy *= -1;

            // 2. Magnetic Repulsion
            const dx = this.mouseX - p.baseX;
            const dy = this.mouseY - p.baseY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const radius = 250; // Aumenté un poco el radio para que empiece a moverse antes

            let targetX = p.baseX;
            let targetY = p.baseY;

            if (distance < radius) {
                const force = (radius - distance) / radius;
                const angle = Math.atan2(dy, dx);

                // Repulsión suave
                const moveX = Math.cos(angle) * force * 100;
                const moveY = Math.sin(angle) * force * 100;

                targetX -= moveX;
                targetY -= moveY;
            }

            // 3. Smooth transition (Interpolation)
            // Aquí es donde el 'ease' 0.03 hace que el movimiento sea lento y fluido
            p.x += (targetX - p.x) * p.ease;
            p.y += (targetY - p.y) * p.ease;

            // Update DOM
            p.el.style.transform = `translate(${p.x}px, ${p.y}px)`;
        });

        requestAnimationFrame(() => this.animate());
    }
}
