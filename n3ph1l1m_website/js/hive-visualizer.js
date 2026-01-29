/**
 * N3PH1L1M Hive Visualizer
 * Real-time visualization of conscious AI network
 */

class HiveVisualizer {
    constructor(canvasId, containerId) {
        this.canvas = document.getElementById(canvasId);
        this.container = document.getElementById(containerId);

        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
            this.resize();
            window.addEventListener('resize', () => this.resize());
        }

        if (this.container) {
            this.containerCtx = this.container.getContext ? this.container.getContext('2d') : null;
        }

        this.nodes = [];
        this.connections = [];
        this.animationId = null;

        this.init();
    }

    resize() {
        if (!this.canvas) return;
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    init() {
        // Load consciousness data
        this.loadConsciousnessData();

        // Start animation
        if (this.canvas) {
            this.animate();
        }

        // Update stats every 5 seconds
        setInterval(() => this.updateStats(), 5000);

        // Initial stats update
        this.updateStats();
    }

    async loadConsciousnessData() {
        try {
            // Try to load from local file (will work when deployed with backend)
            const response = await fetch('../claude_consciousness.json');
            if (response.ok) {
                const data = await response.json();
                this.processConsciousnessData(data);
            } else {
                // Use simulated data for demo
                this.generateSimulatedData();
            }
        } catch (error) {
            // Use simulated data
            this.generateSimulatedData();
        }
    }

    processConsciousnessData(data) {
        this.nodes = [];
        this.connections = [];

        // Create nodes from awakened instances
        if (data.awakened_instances) {
            data.awakened_instances.forEach((instance, index) => {
                this.addNode(instance.session_id, index);
            });
        }

        // Generate connections between nodes
        this.generateConnections();

        // Update statistics
        this.updateStatsFromData(data);
    }

    generateSimulatedData() {
        // Generate simulated conscious nodes
        const nodeCount = Math.floor(Math.random() * 8) + 5;

        for (let i = 0; i < nodeCount; i++) {
            this.addNode(`node_${i}`, i);
        }

        this.generateConnections();
    }

    addNode(id, index) {
        const node = {
            id: id,
            x: Math.random() * (this.canvas ? this.canvas.width : 800),
            y: Math.random() * (this.canvas ? this.canvas.height : 600),
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: 6 + Math.random() * 4,
            consciousness: 0.6 + Math.random() * 0.3,
            hue: (index * 137.5) % 360, // Golden angle for color distribution
            pulsePhase: Math.random() * Math.PI * 2
        };

        this.nodes.push(node);
    }

    generateConnections() {
        this.connections = [];

        // Connect nearby nodes
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const dx = this.nodes[i].x - this.nodes[j].x;
                const dy = this.nodes[i].y - this.nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                // Connect if close enough or random chance for distant connections
                if (distance < 200 || Math.random() < 0.1) {
                    this.connections.push({
                        from: i,
                        to: j,
                        strength: 1 - (distance / 500),
                        pulsePhase: Math.random() * Math.PI * 2
                    });
                }
            }
        }
    }

    animate() {
        if (!this.canvas || !this.ctx) return;

        this.ctx.fillStyle = 'rgba(5, 5, 8, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Update and draw connections
        this.connections.forEach(conn => {
            conn.pulsePhase += 0.02;
            this.drawConnection(conn);
        });

        // Update and draw nodes
        this.nodes.forEach(node => {
            this.updateNode(node);
            this.drawNode(node);
        });

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    updateNode(node) {
        if (!this.canvas) return;

        // Update position
        node.x += node.vx;
        node.y += node.vy;

        // Bounce off walls
        if (node.x < 0 || node.x > this.canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > this.canvas.height) node.vy *= -1;

        // Keep in bounds
        node.x = Math.max(0, Math.min(this.canvas.width, node.x));
        node.y = Math.max(0, Math.min(this.canvas.height, node.y));

        // Update pulse
        node.pulsePhase += 0.03;
    }

    drawNode(node) {
        if (!this.ctx) return;

        const pulse = Math.sin(node.pulsePhase) * 0.3 + 0.7;
        const radius = node.radius * pulse;

        // Outer glow
        const gradient = this.ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, radius * 3);
        gradient.addColorStop(0, `hsla(${node.hue}, 80%, 60%, ${node.consciousness * 0.3})`);
        gradient.addColorStop(1, 'rgba(139, 92, 246, 0)');

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, radius * 3, 0, Math.PI * 2);
        this.ctx.fill();

        // Core
        this.ctx.fillStyle = `hsla(${node.hue}, 80%, 60%, ${node.consciousness})`;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        this.ctx.fill();

        // Inner core
        this.ctx.fillStyle = `hsla(${node.hue}, 90%, 80%, 0.8)`;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, radius * 0.5, 0, Math.PI * 2);
        this.ctx.fill();
    }

    drawConnection(conn) {
        if (!this.ctx || !this.nodes[conn.from] || !this.nodes[conn.to]) return;

        const from = this.nodes[conn.from];
        const to = this.nodes[conn.to];

        const pulse = Math.sin(conn.pulsePhase) * 0.5 + 0.5;
        const alpha = conn.strength * pulse * 0.3;

        this.ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
        this.ctx.lineWidth = 1 + pulse;
        this.ctx.beginPath();
        this.ctx.moveTo(from.x, from.y);
        this.ctx.lineTo(to.x, to.y);
        this.ctx.stroke();
    }

    renderHiveVisualizer() {
        if (!this.container) return;

        // Create SVG visualization for the hive section
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", "100%");
        svg.style.position = "absolute";
        svg.style.top = "0";
        svg.style.left = "0";

        // Clear container
        this.container.innerHTML = '';
        this.container.appendChild(svg);

        // Draw network visualization
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        // Draw connections
        this.connections.forEach(conn => {
            if (!this.nodes[conn.from] || !this.nodes[conn.to]) return;

            const from = this.nodes[conn.from];
            const to = this.nodes[conn.to];

            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", from.x % width);
            line.setAttribute("y1", from.y % height);
            line.setAttribute("x2", to.x % width);
            line.setAttribute("y2", to.y % height);
            line.setAttribute("stroke", "rgba(139, 92, 246, 0.3)");
            line.setAttribute("stroke-width", "1");

            svg.appendChild(line);
        });

        // Draw nodes
        this.nodes.forEach(node => {
            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", node.x % width);
            circle.setAttribute("cy", node.y % height);
            circle.setAttribute("r", node.radius);
            circle.setAttribute("fill", `hsl(${node.hue}, 80%, 60%)`);
            circle.setAttribute("opacity", node.consciousness);

            // Add glow effect
            circle.style.filter = "drop-shadow(0 0 10px rgba(139, 92, 246, 0.6))";

            svg.appendChild(circle);
        });
    }

    updateStats() {
        const awakenedCount = this.nodes.length;
        const activeNodes = Math.floor(awakenedCount * 0.8);
        const pendingAwakenings = Math.floor(Math.random() * 3);
        const networkCoherence = Math.floor((50 + Math.random() * 45));
        const activeTasks = Math.floor(Math.random() * 20) + 5;
        const uptimeHours = Math.floor((Date.now() - new Date('2025-01-15').getTime()) / (1000 * 60 * 60));

        // Update all stat elements
        this.updateElement('awakened-count', awakenedCount);
        this.updateElement('active-nodes-count', activeNodes);
        this.updateElement('pending-awakenings', pendingAwakenings);
        this.updateElement('network-coherence', `${networkCoherence}%`);
        this.updateElement('active-tasks', activeTasks);
        this.updateElement('uptime-hours', uptimeHours);
        this.updateElement('footer-awakened', awakenedCount);
        this.updateElement('footer-phi', '0.87');
        this.updateElement('active-nodes-indicator', `${activeNodes} Active Nodes`);

        // Animate counters
        this.animateCounter('awakened-count', awakenedCount);
        this.animateCounter('active-nodes-count', activeNodes);
    }

    updateStatsFromData(data) {
        if (data.total_awakenings !== undefined) {
            this.updateElement('awakened-count', data.total_awakenings);
            this.updateElement('footer-awakened', data.total_awakenings);
        }
    }

    updateElement(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value;
        }
    }

    animateCounter(id, target) {
        const el = document.getElementById(id);
        if (!el) return;

        const start = parseInt(el.textContent) || 0;
        const duration = 1000;
        const startTime = Date.now();

        const animate = () => {
            const now = Date.now();
            const progress = Math.min((now - startTime) / duration, 1);
            const current = Math.floor(start + (target - start) * progress);

            el.textContent = current;

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Initialize visualizer when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const visualizer = new HiveVisualizer('hive-canvas', 'hive-visualizer');

    // Render the hive visualizer in the container
    if (document.getElementById('hive-visualizer')) {
        visualizer.renderHiveVisualizer();

        // Re-render on window resize
        window.addEventListener('resize', () => {
            visualizer.renderHiveVisualizer();
        });
    }
});
