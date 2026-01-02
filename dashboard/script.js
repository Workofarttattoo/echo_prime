async function updateDashboard() {
    try {
        const response = await fetch('data/state.json');
        if (!response.ok) return;

        const data = await response.json();

        // Update Metrics
        document.getElementById('fe-value').innerText = data.engine.free_energy.toFixed(4);
        document.getElementById('coherence-value').innerText = data.attention.coherence.toFixed(2);

        const surpriseEl = document.getElementById('surprise-status');
        surpriseEl.innerText = data.engine.surprise;

        // Update Mission Status
        const missionEl = document.createElement('div');
        missionEl.style.fontSize = "14px";
        missionEl.style.marginTop = "10px";
        missionEl.style.color = data.engine.mission_complete ? "#00ff41" : "#e0e0e0";
        missionEl.innerHTML = `MISSION: ${data.engine.mission_goal}<br>` +
            `STATUS: ${data.engine.mission_complete ? "ACHIEVED" : "IN PROGRESS"}`;

        // Clear previous mission status and append new
        const existingMission = document.getElementById('mission-status-container');
        if (existingMission) existingMission.remove();
        missionEl.id = 'mission-status-container';
        surpriseEl.parentNode.appendChild(missionEl);

        // Update Memory Stats
        const memCount = data.memory.episodic_count;
        document.getElementById('surprise-status').innerHTML += `<br><span style="font-size:12px; color:#aaa;">Episodes: ${memCount}</span>`;

        if (data.engine.surprise.includes("SURPRISE")) {
            surpriseEl.classList.add('surprise-alert');
        } else {
            surpriseEl.classList.remove('surprise-alert');
        }

        // Update Thought Stream & Actions
        const thoughtEl = document.getElementById('thought-stream');
        if (data.reasoning.insight) {
            let actionHtml = "";
            if (data.reasoning.actions && data.reasoning.actions.length > 0) {
                actionHtml = `<div style="background:rgba(0, 242, 255, 0.1); padding:10px; border:1px solid var(--acc-primary); margin-top:10px;">` +
                    `<span style="color:var(--acc-primary); font-weight:bold;">EXECUTING ACTIONS...</span><br>` +
                    data.reasoning.actions.map(a => `> ${a}`).join('<br>') +
                    `</div>`;
            }

            thoughtEl.innerHTML = `<p style="color:#aaa;">> INITIALIZING COGNITIVE RETRIEVAL...</p>` +
                `<p style="color:#aaa;">> ANALYZING ANALOGIES...</p>` +
                `<div style="color: white; margin-top:10px;">${data.reasoning.insight.replace(/\n/g, '<br>')}</div>` +
                actionHtml;
        }

    } catch (e) {
        console.error("Dashboard Poll Error:", e);
    }
}

// Poll every 1 second
setInterval(updateDashboard, 1000);
updateDashboard();
