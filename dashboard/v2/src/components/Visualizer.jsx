import React, { useRef, useEffect } from 'react';

const Visualizer = ({ strain = 0.0, coherence = 1.0 }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        let animationFrameId;

        // Resize
        const resize = () => {
            canvas.width = canvas.offsetWidth * window.devicePixelRatio;
            canvas.height = canvas.offsetHeight * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };
        resize();
        window.addEventListener('resize', resize);

        // Particles
        const particleCount = 200;
        const particles = [];

        // Color Palettes
        // Low Strain: Cyan/Blue
        // High Strain: Red/Orange
        // Calm: Green/Teal

        const createParticle = () => ({
            x: Math.random() * canvas.offsetWidth,
            y: Math.random() * canvas.offsetHeight,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: Math.random() * 2 + 1,
            life: Math.random() * 100,
            maxLife: 100 + Math.random() * 100
        });

        for (let i = 0; i < particleCount; i++) particles.push(createParticle());

        let time = 0;

        const render = () => {
            time += 0.01 + (strain * 0.05); // Time moves faster with strain

            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            const cx = width / 2;
            const cy = height / 2;

            // Clear with fade effect for trails
            ctx.fillStyle = 'rgba(10, 10, 15, 0.2)';
            ctx.fillRect(0, 0, width, height);

            // Central Core (Pulsing at ~40Hz simulated)
            const pulse = Math.sin(time * 10) * 10 * coherence;
            ctx.beginPath();
            ctx.arc(cx, cy, 50 + pulse, 0, Math.PI * 2);

            // Core Color based on Strain
            const r = Math.min(255, strain * 20); // More red with strain
            const g = Math.max(0, 200 - strain * 20); // Less green with strain
            const b = 255;

            const gradient = ctx.createRadialGradient(cx, cy, 10, cx, cy, 80);
            gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 1)`);
            gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Draw Particles
            particles.forEach(p => {
                // Update physics based on strain
                const speed = 1 + strain * 2.0;

                // Attraction to center (Visualizer logic)
                const dx = cx - p.x;
                const dy = cy - p.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                // Orbit force
                p.x += p.vx * speed;
                p.y += p.vy * speed;

                // Gravity/Orbit logic
                if (dist > 50) {
                    p.vx += (dx / dist) * 0.01 * coherence;
                    p.vy += (dy / dist) * 0.01 * coherence;
                }

                // Color based on particle life and strain
                const alpha = Math.sin((p.life / p.maxLife) * Math.PI);
                ctx.fillStyle = `rgba(${r + 50}, ${g + 50}, 255, ${alpha})`;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fill();

                // Reset dead particles
                p.life--;
                if (p.life <= 0 || p.x < 0 || p.x > width || p.y < 0 || p.y > height) {
                    Object.assign(p, createParticle());
                    // Spawn near outside
                    if (Math.random() > 0.5) {
                        p.x = Math.random() * width;
                        p.y = Math.random() > 0.5 ? 0 : height;
                    } else {
                        p.x = Math.random() > 0.5 ? 0 : width;
                        p.y = Math.random() * height;
                    }
                }
            });

            // Connecting lines if coherence is high
            if (coherence > 0.8) {
                ctx.strokeStyle = `rgba(100, 200, 255, 0.05)`;
                ctx.lineWidth = 1;
                for (let i = 0; i < 50; i++) { // Only check subset for performance
                    const p1 = particles[i];
                    const p2 = particles[(i + 1) % particles.length];
                    const d = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
                    if (d < 100) {
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
            }

            animationFrameId = requestAnimationFrame(render);
        };
        render();

        return () => {
            window.removeEventListener('resize', resize);
            cancelAnimationFrame(animationFrameId);
        };
    }, [strain, coherence]);

    return (
        <div style={{ width: '100%', height: '100%', position: 'fixed', top: 0, left: 0, zIndex: 0 }}>
            {/* Background Grid/Effect */}
            <div style={{
                position: 'absolute', inset: 0,
                background: 'radial-gradient(circle at center, #1a1a2e 0%, #000000 100%)',
                zIndex: -1
            }} />
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
        </div>
    );
};

export default Visualizer;
