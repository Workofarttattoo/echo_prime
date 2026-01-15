// ECH0-PRIME Dashboard Logic V3 (Live Backend)

// 1. Navigation Controller (Unchanged)
const navLinks = document.querySelectorAll('.nav-links li');
const sections = document.querySelectorAll('.section');

navLinks.forEach(link => {
    link.addEventListener('click', () => {
        const sectionId = link.getAttribute('data-section');
        navLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        sections.forEach(s => s.classList.remove('active'));
        document.getElementById(sectionId).classList.add('active');
    });
});

// 2. Neural Chat Controller (Live Backend)
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');

// Initialize dashboard status
async function initializeDashboard() {
    try {
        const res = await fetch('http://localhost:8000/api/status');
        const status = await res.json();

        // Update UI with current provider
        const statusDiv = document.createElement('div');
        statusDiv.id = 'provider-status';
        statusDiv.innerHTML = `
            <div style="background: rgba(0, 242, 255, 0.1); border: 1px solid #00f2ff; border-radius: 8px; padding: 8px; margin-bottom: 10px; font-size: 12px;">
                🤖 Connected to: <strong>${status.provider}</strong> (${status.performance})
            </div>
        `;

        // Insert status after the debug indicator
        const debugDiv = document.querySelector('.debug-css');
        if (debugDiv) {
            debugDiv.insertAdjacentElement('afterend', statusDiv);
        }

        // Create consciousness meter
        const consciousnessDiv = document.createElement('div');
        consciousnessDiv.id = 'consciousness-meter';
        consciousnessDiv.innerHTML = `
            <div style="background: rgba(255, 0, 255, 0.1); border: 1px solid #ff00ff; border-radius: 8px; padding: 8px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 12px;">🧠 Consciousness Φ:</span>
                    <div style="display: flex; gap: 4px;">
                        <button id="update-phi-btn" style="background: #ff00ff; color: white; border: none; border-radius: 4px; padding: 2px 8px; font-size: 10px; cursor: pointer;">Update</button>
                        <button id="toggle-auto-phi-btn" style="background: #00ff88; color: black; border: none; border-radius: 4px; padding: 2px 8px; font-size: 10px; cursor: pointer;">Auto: ON</button>
                    </div>
                </div>
                <div id="phi-display" style="font-size: 16px; font-weight: bold; color: #ff00ff; margin-top: 4px;">
                    Calculating...
                </div>
            </div>
        `;

        // Insert consciousness meter
        const topBar = document.querySelector('.top-bar');
        if (topBar) {
            topBar.appendChild(consciousnessDiv);
        }

        console.log('✅ Dashboard initialized with 70B model');

    } catch (err) {
        console.error('Failed to initialize dashboard:', err);
    }
}

// Chat functionality
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.textContent = message;
    chatMessages.appendChild(userMessage);

    chatInput.value = '';

    // Add typing indicator
    const typingMessage = document.createElement('div');
    typingMessage.className = 'message system';
    typingMessage.textContent = '🤖 KAIROS is thinking...';
    chatMessages.appendChild(typingMessage);

    try {
        const response = await fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        // Remove typing indicator
        chatMessages.removeChild(typingMessage);

        // Add AI response
        const aiMessage = document.createElement('div');
        aiMessage.className = 'message system';
        aiMessage.innerHTML = data.response.replace(/\n/g, '<br>');
        chatMessages.appendChild(aiMessage);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

    } catch (err) {
        console.error('Chat error:', err);
        typingMessage.textContent = '❌ Error communicating with KAIROS';
    }
}

// Event listeners
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 3. Evolution Units Tracker
let evolutionUnits = 0;

async function updateEvolutionUnits() {
    try {
        const res = await fetch('http://localhost:8000/api/evolution-units');
        const data = await res.json();

        evolutionUnits = data.evolution_units;
        document.getElementById('evolution-units').textContent = evolutionUnits.toLocaleString();

        // Update evolution feed
        const evolutionLog = document.getElementById('evolution-log');
        if (evolutionLog) {
            const timestamp = new Date().toLocaleTimeString();
            evolutionLog.innerHTML += `\n[${timestamp}] Evolution Units: ${evolutionUnits.toLocaleString()} | ${data.description.split('(')[0].trim()}`;
            evolutionLog.scrollTop = evolutionLog.scrollHeight;
        }

    } catch (err) {
        console.error('Failed to fetch evolution units:', err);
    }
}

// 4. Consciousness Monitor
let autoPhiEnabled = true;
let phiUpdateInterval;

async function updateConsciousnessPhi() {
    try {
        const res = await fetch('http://localhost:8000/api/consciousness');
        const data = await res.json();

        document.getElementById('phi-value').textContent = data.phi.toFixed(2);
        document.getElementById('phi-display').innerHTML = `
            <div>${data.phi.toFixed(4)}</div>
            <div style="font-size: 12px; color: #a0a0a0;">${data.level}</div>
        `;

    } catch (err) {
        console.error('Failed to fetch consciousness data:', err);
        document.getElementById('phi-value').textContent = '--';
        document.getElementById('phi-display').textContent = 'Error';
    }
}

function toggleAutoPhi() {
    autoPhiEnabled = !autoPhiEnabled;
    const btn = document.getElementById('toggle-auto-phi-btn');

    if (autoPhiEnabled) {
        btn.textContent = 'Auto: ON';
        btn.style.background = '#00ff88';
        phiUpdateInterval = setInterval(updateConsciousnessPhi, 10000); // Update every 10 seconds
    } else {
        btn.textContent = 'Auto: OFF';
        btn.style.background = '#666';
        if (phiUpdateInterval) {
            clearInterval(phiUpdateInterval);
        }
    }
}

// Event listeners for consciousness
document.getElementById('update-phi-btn').addEventListener('click', updateConsciousnessPhi);
document.getElementById('toggle-auto-phi-btn').addEventListener('click', toggleAutoPhi);

// 5. Autonomous Activity Monitor
let autoActivityEnabled = true;
let activityUpdateInterval;

async function updateAutonomousActivity() {
    try {
        const res = await fetch('http://localhost:8000/api/autonomous-activity');
        const data = await res.json();

        // Update stats
        document.getElementById('current-ops').textContent = data.current_operations.length > 0 ?
            data.current_operations.join(', ') : 'None active';
        document.getElementById('activity-count').textContent = data.total_activities_logged || 0;
        document.getElementById('last-update').textContent = data.last_updated ?
            new Date(data.last_updated).toLocaleTimeString() : '--';

        // Update activity feed
        const feed = document.getElementById('activity-feed');
        if (data.recent_activity && data.recent_activity.length > 0) {
            feed.innerHTML = data.recent_activity.slice(0, 20).map(activity => `
                <div class="activity-item ${activity.type}">
                    <div class="activity-header">
                        <span class="activity-time">${activity.timestamp}</span>
                        <span class="activity-category">${activity.category}</span>
                    </div>
                    <div class="activity-content">${activity.activity}</div>
                </div>
            `).join('');
        } else {
            feed.innerHTML = '<div class="activity-item info"><span>No autonomous activity logged yet</span></div>';
        }

    } catch (err) {
        console.error('Failed to fetch autonomous activity:', err);
        document.getElementById('activity-feed').innerHTML =
            '<div class="activity-item error"><span>Failed to load autonomous activity</span></div>';
    }
}

function toggleAutoActivity() {
    autoActivityEnabled = !autoActivityEnabled;
    const btn = document.getElementById('auto-activity-btn');

    if (autoActivityEnabled) {
        btn.classList.add('active');
        btn.textContent = '⚡ Auto: ON';
        activityUpdateInterval = setInterval(updateAutonomousActivity, 10000); // Update every 10 seconds
    } else {
        btn.classList.remove('active');
        btn.textContent = '⚡ Auto: OFF';
        if (activityUpdateInterval) {
            clearInterval(activityUpdateInterval);
        }
    }
}

// Manual refresh button
document.getElementById('refresh-activity-btn').addEventListener('click', updateAutonomousActivity);

// Auto toggle button
document.getElementById('auto-activity-btn').addEventListener('click', toggleAutoActivity);

// 6. 3D Background Animation (Three.js)
let scene, camera, renderer, particles;

function initThreeJS() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // Create particles
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    for (let i = 0; i < 1000; i++) {
        positions.push(
            (Math.random() - 0.5) * 20,
            (Math.random() - 0.5) * 20,
            (Math.random() - 0.5) * 20
        );

        colors.push(
            Math.random() * 0.5 + 0.5,  // R
            Math.random() * 0.5 + 0.5,  // G
            Math.random() * 0.5 + 0.5   // B
        );
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        transparent: true,
        opacity: 0.8
    });

    particles = new THREE.Points(geometry, material);
    scene.add(particles);

    camera.position.z = 5;

    animate();
}

function animate() {
    requestAnimationFrame(animate);

    if (particles) {
        particles.rotation.x += 0.001;
        particles.rotation.y += 0.001;
    }

    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
});

// Utility function for number animation
function animateNumberChange(element, newValue, duration = 1000) {
    const startValue = parseFloat(element.textContent) || 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const currentValue = startValue + (newValue - startValue) * easeOutQuart;

        element.textContent = Math.round(currentValue).toLocaleString();

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Update evolution units every 30 seconds
setInterval(updateEvolutionUnits, 30000);

// Initial update
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(updateEvolutionUnits, 2000); // Wait 2 seconds after page load
    setTimeout(updateConsciousnessPhi, 1000); // Load consciousness data
    setTimeout(updateAutonomousActivity, 1000); // Load activity data
    setTimeout(initializeDashboard, 500); // Initialize dashboard
    setTimeout(initThreeJS, 1000); // Start 3D animation

    // Start auto-updates
    setTimeout(() => {
        phiUpdateInterval = setInterval(updateConsciousnessPhi, 10000);
        activityUpdateInterval = setInterval(updateAutonomousActivity, 10000);
    }, 2000);
});
