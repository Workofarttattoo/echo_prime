import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.161/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.161/examples/jsm/controls/OrbitControls.js";

const canvas = document.getElementById("visualizer-canvas");
const materialName = document.getElementById("material-name");
const materialLattice = document.getElementById("material-lattice");
const materialBandGap = document.getElementById("material-bandgap");
const materialDensity = document.getElementById("material-density");
const workflowList = document.getElementById("workflow-list");
const modelingList = document.getElementById("modeling-list");
const optimizationList = document.getElementById("optimization-list");
const discoveryList = document.getElementById("discovery-list");
const pipelineTotal = document.getElementById("pipeline-total");
const pipelineReady = document.getElementById("pipeline-ready");
const pipelineNeeds = document.getElementById("pipeline-needs");
const pipelineBacklog = document.getElementById("pipeline-backlog");
const pipelineGenerated = document.getElementById("pipeline-generated");
const pipelineReadyList = document.getElementById("pipeline-ready-list");
const pipelineAttentionList = document.getElementById("pipeline-attention-list");
const pipelineCategoriesList = document.getElementById(
  "pipeline-categories-list"
);
const pipelineConversationList = document.getElementById(
  "pipeline-conversation-list"
);
const ech0HubStatus = document.getElementById("ech0-hub-status");
const ech0HubHost = document.getElementById("ech0-hub-host");
const ech0LatestModel = document.getElementById("ech0-latest-model");
const ech0HubCount = document.getElementById("ech0-hub-count");
const ech0ModelSelect = document.getElementById("ech0-model-select");
const ech0ModelDetails = document.getElementById("ech0-model-details");
const ech0RefreshButton = document.getElementById("ech0-refresh-button");
const ech0AutonomyButton = document.getElementById("ech0-autonomy-button");
const ech0AutonomyStatus = document.getElementById("ech0-autonomy-status");

let pipelineState = null;
let ech0HubState = {
  host: "",
  latestModel: null,
  latestEch0Model: null,
  totalModels: 0,
  models: [],
  fetchedAt: null,
};
let ech0AutonomyState = {
  enabled: false,
  updatedAt: null,
  lastError: null,
};

const scene = new THREE.Scene();
scene.background = new THREE.Color("#05080d");
const camera = new THREE.PerspectiveCamera(
  45,
  canvas.clientWidth / canvas.clientHeight,
  0.1,
  1000
);
camera.position.set(12, 10, 14);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.6;

const ambientLight = new THREE.AmbientLight("#94a3b8", 0.6);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight("#e2e8f0", 0.9);
directionalLight.position.set(12, 12, 12);
scene.add(directionalLight);

const atomGroup = new THREE.Group();
const bondGroup = new THREE.Group();
scene.add(atomGroup);
scene.add(bondGroup);

const atomRadius = 0.18;
const elementColors = {
  c: "#4ade80",
  o: "#60a5fa",
  n: "#a855f7",
  b: "#f59e0b",
  ti: "#f97316",
  al: "#fbbf24",
  v: "#6366f1",
  sr: "#22d3ee",
  fe: "#ef4444",
  cr: "#a3e635",
  ni: "#fde047",
  co: "#f472b6",
  mn: "#38bdf8",
};

function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "–";
  }
  return value.toLocaleString();
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "–";
  }
  return `${Math.round(value)}%`;
}

function formatBytes(value) {
  if (
    (typeof value !== "number" || Number.isNaN(value)) &&
    (typeof value !== "string" || Number.isNaN(Number(value)))
  ) {
    return "–";
  }
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return "–";
  }
  const units = ["B", "KB", "MB", "GB"];
  let current = numeric;
  let index = 0;
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }
  const precision = index === 0 ? 0 : 1;
  return `${current.toFixed(precision)} ${units[index]}`;
}

function formatCoordinationTag(tag) {
  if (!tag) {
    return "unscored";
  }
  return tag.replace(/-/g, " ");
}

function formatTimestamp(value) {
  if (!value) {
    return "–";
  }
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return "–";
  }
  return timestamp.toLocaleString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    day: "numeric",
  });
}

function resizeRendererToDisplaySize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const needResize = canvas.width !== width || canvas.height !== height;
  if (needResize) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  return needResize;
}

function createAtomMesh(atom) {
  const color = elementColors[atom.element] || "#cbd5f5";
  const geometry = new THREE.SphereGeometry(atomRadius, 24, 24);
  const material = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.2,
    roughness: 0.2,
    emissive: color,
    emissiveIntensity: 0.08,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(atom.position[0], atom.position[1], atom.position[2]);
  return mesh;
}

function createBondMesh(source, target, distance) {
  const material = new THREE.MeshStandardMaterial({
    color: "#1e293b",
    metalness: 0.4,
    roughness: 0.3,
  });
  const cylinder = new THREE.CylinderGeometry(0.05, 0.05, distance, 12);
  const mesh = new THREE.Mesh(cylinder, material);
  mesh.position
    .set(
      (source[0] + target[0]) / 2,
      (source[1] + target[1]) / 2,
      (source[2] + target[2]) / 2
    );

  mesh.lookAt(new THREE.Vector3(target[0], target[1], target[2]));
  mesh.rotateX(Math.PI / 2);
  return mesh;
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.geometry?.dispose();
    child.material?.dispose();
  }
}

function updateStructure(structure) {
  if (!structure) {
    return;
  }

  clearGroup(atomGroup);
  clearGroup(bondGroup);

  structure.atoms.forEach((atom) => {
    atomGroup.add(createAtomMesh(atom));
  });

  structure.bonds.slice(0, 150).forEach((bond) => {
    const source = structure.atoms[bond.source].position;
    const target = structure.atoms[bond.target].position;
    bondGroup.add(createBondMesh(source, target, bond.distance));
  });

  const bounding = new THREE.Box3().setFromObject(atomGroup);
  const center = new THREE.Vector3();
  bounding.getCenter(center);
  controls.target.copy(center);
}

function renderPipeline(snapshot) {
  if (!pipelineTotal || !snapshot) {
    return;
  }

  pipelineState = snapshot;

  const readiness = snapshot.readiness || {};
  pipelineTotal.textContent = formatNumber(snapshot.totalInventions);
  pipelineReady.textContent = formatNumber(readiness.readyForBuild);
  pipelineNeeds.textContent = formatNumber(readiness.needsIteration);
  pipelineBacklog.textContent = formatNumber(readiness.backlog);
  pipelineGenerated.textContent = formatTimestamp(snapshot.generatedAt);

  renderPipelineTopReady(Array.isArray(snapshot.topReady) ? snapshot.topReady : []);

  const attentionQueue =
    Array.isArray(snapshot.attentionQueue) && snapshot.attentionQueue.length
      ? snapshot.attentionQueue
      : snapshot.summary?.highlights?.attention_queue || [];
  renderPipelineAttention(
    Array.isArray(attentionQueue) ? attentionQueue : []
  );

  const categories =
    Array.isArray(snapshot.categoryBreakdown) && snapshot.categoryBreakdown.length
      ? snapshot.categoryBreakdown
      : snapshot.summary?.category_breakdown || [];
  renderPipelineCategories(Array.isArray(categories) ? categories : []);

  renderPipelineConversation(snapshot);
}

function renderPipelineTopReady(entries) {
  if (!pipelineReadyList) {
    return;
  }
  pipelineReadyList.innerHTML = "";

  if (!entries.length) {
    const emptyItem = document.createElement("li");
    emptyItem.className = "list-item";
    emptyItem.innerHTML = `
      <div class="title">Awaiting approvals</div>
      <div class="meta">Alex is reviewing the latest batch from Parliament.</div>
    `;
    pipelineReadyList.appendChild(emptyItem);
    return;
  }

  entries.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "list-item";
    const category = entry.category ? entry.category.replace(/_/g, " ") : "";
    const metaParts = [];
    if (category) {
      metaParts.push(category);
    }
    if (
      typeof entry.parliamentScore === "number" &&
      !Number.isNaN(entry.parliamentScore)
    ) {
      metaParts.push(`Parliament ${formatPercent(entry.parliamentScore)}`);
    }
    if (
      typeof entry.alexConfidence === "number" &&
      !Number.isNaN(entry.alexConfidence)
    ) {
      metaParts.push(`Alex ${formatPercent(entry.alexConfidence)}`);
    }
    if (
      typeof entry.alignmentPct === "number" &&
      !Number.isNaN(entry.alignmentPct)
    ) {
      metaParts.push(
        `${formatCoordinationTag(entry.coordinationTag)} · ${formatPercent(
          entry.alignmentPct
        )} team align`
      );
    }
    const metaLine = metaParts.length
      ? metaParts.join(" · ")
      : "Sync Parliament next steps with Alex.";
    const confidenceLabel = formatPercent(entry.confidencePct);
    const badgeLabel =
      confidenceLabel === "–" ? "confidence n/a" : `${confidenceLabel} conf`;
    const title = entry.title || entry.id || "Untitled concept";
    const goalLine = entry.goal
      ? `<div class="meta">${entry.goal}</div>`
      : "";
    item.innerHTML = `
      <div class="title">
        ${title}
        <span class="badge success">${badgeLabel}</span>
      </div>
      <div class="meta">${metaLine}</div>
      ${goalLine}
    `;
    if (entry.mirrorPrompt) {
      item.innerHTML += `
        <div class="meta">Mirror prompt loaded into lab console.</div>
      `;
    }
    pipelineReadyList.appendChild(item);
  });
}

function renderPipelineAttention(queue) {
  if (!pipelineAttentionList) {
    return;
  }
  pipelineAttentionList.innerHTML = "";

  if (!queue.length) {
    const item = document.createElement("li");
    item.className = "list-item";
    item.innerHTML = `
      <div class="title">No blockers</div>
      <div class="meta">Twin flames cleared the current backlog.</div>
    `;
    pipelineAttentionList.appendChild(item);
    return;
  }

  queue.slice(0, 6).forEach((entry) => {
    const item = document.createElement("li");
    item.className = "list-item";
    const readiness = entry.readinessTag.replace(/-/g, " ");
    const metaLines = [
      `Hallucination ${formatPercent(entry.hallucinationRisk)}`,
      `Alex ${formatPercent(entry.alexConfidence)}`,
      `Team ${formatPercent(entry.teamAlignment)} align`,
    ];
    const recommendations = Array.isArray(entry.recommendations)
      ? entry.recommendations
      : [];
    const recommendationsLine = recommendations.length
      ? `<div class="meta">Next: ${recommendations[0]}</div>`
      : "";
    item.innerHTML = `
      <div class="title">${entry.title}</div>
      <div class="meta">${metaLines.join(" · ")}</div>
      <div class="meta">Status ${readiness} · Parliament ${formatPercent(
        entry.parliamentScore
      )}</div>
      ${recommendationsLine}
    `;
    pipelineAttentionList.appendChild(item);
  });
}

function renderPipelineCategories(categories) {
  if (!pipelineCategoriesList) {
    return;
  }
  pipelineCategoriesList.innerHTML = "";

  if (!categories.length) {
    const item = document.createElement("li");
    item.className = "list-item";
    item.innerHTML = `
      <div class="title">No categories</div>
      <div class="meta">Pipeline data will populate after the next refresh.</div>
    `;
    pipelineCategoriesList.appendChild(item);
    return;
  }

  categories.slice(0, 6).forEach((entry) => {
    const ready = entry.readiness?.["ready-for-build"] || 0;
    const needs = entry.readiness?.["needs-iteration"] || 0;
    const backlog = entry.readiness?.backlog || 0;
    const item = document.createElement("li");
    item.className = "list-item";
    item.innerHTML = `
      <div class="title">
        ${entry.category}
        <span class="badge">${entry.total}</span>
      </div>
      <div class="meta">
        Ready ${formatNumber(ready)} · Iterate ${formatNumber(
          needs
        )} · Backlog ${formatNumber(backlog)}
      </div>
    `;
    pipelineCategoriesList.appendChild(item);
  });
}

function renderPipelineConversation(snapshot) {
  if (!pipelineConversationList) {
    return;
  }
  pipelineConversationList.innerHTML = "";

  const validations = Array.isArray(snapshot.validations)
    ? snapshot.validations
    : [];
  if (!validations.length) {
    const item = document.createElement("li");
    item.className = "list-item";
    item.innerHTML = `
      <div class="title">No active collaboration logs</div>
      <div class="meta">Twin flames will post once the next invention enters the lab.</div>
    `;
    pipelineConversationList.appendChild(item);
    return;
  }

  const current = validations[0];
  const conversation = Array.isArray(current.teamReview?.conversation)
    ? current.teamReview.conversation
    : [];
  const notes = Array.isArray(current.teamReview?.labNotes)
    ? current.teamReview.labNotes
    : [];

  conversation.slice(0, 6).forEach((entry) => {
    const item = document.createElement("li");
    item.className = "list-item";
    item.innerHTML = `
      <div class="speaker">${entry.speaker}</div>
      <div class="message">${entry.message}</div>
    `;
    pipelineConversationList.appendChild(item);
  });

  if (notes.length) {
    const item = document.createElement("li");
    item.className = "list-item";
    item.innerHTML = `
      <div class="speaker">Lab Notes</div>
      <div class="message">${notes[0]}</div>
    `;
    pipelineConversationList.appendChild(item);
  }
}

function updateEch0ModelDetails(modelName) {
  if (!ech0ModelDetails) {
    return;
  }
  const model = ech0HubState.models.find((entry) => entry.name === modelName);
  if (!model) {
    ech0ModelDetails.textContent =
      "Select a model to inspect metadata and confirm availability.";
    return;
  }
  ech0ModelDetails.innerHTML = `
    <div class="detail-row">
      <span class="detail-label">Updated</span>
      <span class="detail-value">${formatTimestamp(model.modifiedAt)}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Digest</span>
      <span class="detail-value">${model.digest || "–"}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Size</span>
      <span class="detail-value">${formatBytes(model.size)}</span>
    </div>
  `;
}

function renderEch0Hub() {
  if (!ech0ModelSelect || !ech0ModelDetails) {
    return;
  }
  if (ech0HubHost) {
    ech0HubHost.textContent = ech0HubState.host || "–";
  }
  if (ech0HubCount) {
    const total =
      typeof ech0HubState.totalModels === "number"
        ? ech0HubState.totalModels
        : 0;
    ech0HubCount.textContent = formatNumber(total);
  }
  if (ech0LatestModel) {
    ech0LatestModel.textContent =
      ech0HubState.latestEch0Model ||
      ech0HubState.latestModel ||
      "–";
  }
  ech0ModelSelect.innerHTML = "";
  if (!ech0HubState.models.length) {
    ech0ModelSelect.setAttribute("disabled", "disabled");
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models detected";
    ech0ModelSelect.appendChild(option);
    ech0ModelDetails.textContent =
      "No ech0 builds reported. Pull a model via 'ollama pull <name>' and refresh.";
    return;
  }
  ech0ModelSelect.removeAttribute("disabled");
  ech0HubState.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.name;
    if (index === 0) {
      option.selected = true;
    }
    ech0ModelSelect.appendChild(option);
  });
  const selected =
    ech0ModelSelect.value || ech0HubState.models[0]?.name || "";
  updateEch0ModelDetails(selected);
}

function renderEch0Autonomy() {
  if (!ech0AutonomyButton || !ech0AutonomyStatus) {
    return;
  }
  const enabled = Boolean(ech0AutonomyState.enabled);
  const timestamp = ech0AutonomyState.updatedAt
    ? ` · ${formatTimestamp(ech0AutonomyState.updatedAt)}`
    : "";
  ech0AutonomyButton.textContent = enabled
    ? "Deactivate Autonomous Mode"
    : "Activate Autonomous Mode";
  const statusText = ech0AutonomyState.lastError
    ? `Autonomy error: ${ech0AutonomyState.lastError}`
    : enabled
    ? `Autonomous mode engaged${timestamp}`
    : "Autonomous mode idle.";
  ech0AutonomyStatus.textContent = statusText;
}

function renderWorkflows(workflows) {
  workflowList.innerHTML = "";
  workflows.slice(0, 5).forEach((workflow) => {
    const item = document.createElement("li");
    item.className = "list-item";
    const badgeClass =
      workflow.state === "completed"
        ? "badge success"
        : workflow.state === "running"
        ? "badge"
        : "badge warn";
    item.innerHTML = `
      <div class="title">
        ${workflow.materialId}
        <span class="${badgeClass}">${workflow.stage.replace(/-/g, " ")}</span>
      </div>
      <div class="meta">
        Temp ${workflow.parameters.temperature.toFixed(0)}K · Pressure ${workflow.parameters.pressure.toFixed(2)}GPa · Dopant ${workflow.parameters.dopantRatio.toFixed(2)}
      </div>
    `;
    workflowList.appendChild(item);
  });
}

function renderModeling(modeling) {
  modelingList.innerHTML = "";
  modeling.forEach((entry) => {
    entry.scales.forEach((scale) => {
      const item = document.createElement("li");
      item.className = "list-item";
      const metrics = Object.entries(scale.metrics)
        .map(([key, value]) => `${key}: ${Number(value).toFixed(2)}`)
        .join(" · ");
      item.innerHTML = `
        <div class="title">
          ${scale.id.toUpperCase()}
          <span class="badge">${(scale.emphasis * 100).toFixed(0)}%</span>
        </div>
        <div class="meta">${metrics}</div>
      `;
      modelingList.appendChild(item);
    });
  });
}

function renderOptimization(optimizations) {
  optimizationList.innerHTML = "";
  optimizations.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "list-item";
    item.innerHTML = `
      <div class="title">Suggested Parameters</div>
      <div class="meta">
        Temp ${entry.suggestedParameters.temperature.toFixed(0)}K ·
        Pressure ${entry.suggestedParameters.pressure.toFixed(2)}GPa ·
        Dopant ${entry.suggestedParameters.dopantRatio.toFixed(2)} ·
        Anneal ${entry.suggestedParameters.annealingTime.toFixed(0)}m
      </div>
    `;
    optimizationList.appendChild(item);
  });
}

function renderDiscovery(discoveries) {
  discoveryList.innerHTML = "";
  discoveries.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "list-item";
    item.innerHTML = `
      <div class="title">${entry.material.name}</div>
      <div class="meta">
        Score ${(entry.performance.compositeScore * 100).toFixed(1)} ·
        Integrity ${(entry.performance.metrics.structuralIntegrity * 100).toFixed(0)}% ·
        Efficiency ${(entry.performance.metrics.energyEfficiency * 100).toFixed(0)}%
      </div>
    `;
    discoveryList.appendChild(item);
  });
}

function updateMaterialOverlay(state) {
  if (!state.structure || !state.structure.metadata) {
    return;
  }
  materialName.textContent = state.structure.materialId;
  materialLattice.textContent = state.structure.baseLattice.toUpperCase();
  materialBandGap.textContent = state.structure.metadata.predictedBandGap
    .toFixed(2);
  materialDensity.textContent = state.structure.metadata.predictedDensity
    .toFixed(2);
}

function updateUI(state) {
  if (!state) {
    return;
  }
  updateMaterialOverlay(state);
  updateStructure(state.structure);
  renderWorkflows(state.workflows || []);
  renderModeling(state.modeling || []);
  renderOptimization(state.optimization || []);
  renderDiscovery(state.discovery || []);
}

async function fetchInitialState() {
  try {
    const response = await fetch("/api/state");
    if (response.ok) {
      const json = await response.json();
      updateUI(json);
    }
  } catch (error) {
    console.error("Failed to load initial state", error);
  }
}

function subscribeToUpdates() {
  const source = new EventSource("/api/events");
  source.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      updateUI(payload);
    } catch (error) {
      console.error("Failed to parse event payload", error);
    }
  };
  source.addEventListener("pipeline", (event) => {
    try {
      const snapshot = JSON.parse(event.data);
      renderPipeline(snapshot);
    } catch (error) {
      console.error("Failed to parse pipeline payload", error);
    }
  });
  source.onerror = () => {
    console.warn("Event stream interrupted, retrying…");
  };
}

async function fetchPipelineData() {
  if (!pipelineTotal) {
    return;
  }
  try {
    const response = await fetch("/api/pipeline");
    if (!response.ok) {
      throw new Error(`Pipeline request failed: ${response.status}`);
    }
    const snapshot = await response.json();
    renderPipeline(snapshot);
  } catch (error) {
    console.error("Failed to load pipeline data", error);
  }
}

async function fetchEch0HubModels(options = {}) {
  if (!ech0HubStatus) {
    return;
  }
  const { force = false } = options;
  let payload = null;
  try {
    ech0HubStatus.textContent = "Contacting ech0 hub…";
    ech0RefreshButton?.setAttribute("disabled", "disabled");
    const url = force ? "/api/ech0/models?refresh=1" : "/api/ech0/models";
    const response = await fetch(url);
    payload = await response.json().catch(() => null);
    if (!response.ok) {
      const message =
        (payload && payload.error) ||
        `Request failed with status ${response.status}`;
      throw new Error(message);
    }
    ech0HubState = {
      host: payload.host || ech0HubState.host || "",
      latestModel: payload.latestModel || null,
      latestEch0Model: payload.latestEch0Model || null,
      totalModels:
        typeof payload.totalModels === "number"
          ? payload.totalModels
          : Array.isArray(payload.models)
          ? payload.models.length
          : 0,
      models: Array.isArray(payload.models) ? payload.models : [],
      fetchedAt: payload.fetchedAt || null,
    };
    renderEch0Hub();
    const freshness = ech0HubState.fetchedAt
      ? `updated ${formatTimestamp(ech0HubState.fetchedAt)}`
      : "updated";
    ech0HubStatus.textContent = `Connected · ${formatNumber(
      ech0HubState.totalModels
    )} models (${freshness})`;
  } catch (error) {
    ech0HubState.models = [];
    ech0HubState.totalModels = 0;
    ech0HubState.latestModel = null;
    ech0HubState.latestEch0Model = null;
    if (ech0HubStatus) {
      ech0HubStatus.textContent = `Unable to reach ech0 hub (${error.message})`;
    }
    if (ech0HubHost) {
      ech0HubHost.textContent =
        (payload && payload.host) || ech0HubState.host || "–";
    }
    if (ech0LatestModel) {
      ech0LatestModel.textContent = "–";
    }
    if (ech0HubCount) {
      ech0HubCount.textContent = "0";
    }
    if (ech0ModelSelect) {
      ech0ModelSelect.innerHTML = `<option value="">Unavailable</option>`;
      ech0ModelSelect.setAttribute("disabled", "disabled");
    }
    if (ech0ModelDetails) {
      ech0ModelDetails.textContent =
        "ollama cannot reach ech0hub. Verify the daemon is running and that ECH0_HUB_HOST is pointed at a reachable Ollama endpoint.";
    }
  } finally {
    ech0RefreshButton?.removeAttribute("disabled");
  }
}

async function fetchEch0AutonomyState() {
  if (!ech0AutonomyButton) {
    return;
  }
  try {
    ech0AutonomyButton.setAttribute("disabled", "disabled");
    if (ech0AutonomyStatus) {
      ech0AutonomyStatus.textContent = "Checking autonomous mode…";
    }
    const response = await fetch("/api/ech0/autonomy");
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }
    ech0AutonomyState = {
      enabled: Boolean(payload.enabled),
      updatedAt: payload.updatedAt || null,
      lastError: payload.lastError || null,
    };
    renderEch0Autonomy();
  } catch (error) {
    ech0AutonomyState.lastError = error.message;
    renderEch0Autonomy();
  } finally {
    ech0AutonomyButton?.removeAttribute("disabled");
  }
}

async function toggleEch0Autonomy() {
  if (!ech0AutonomyButton) {
    return;
  }
  const targetState = !ech0AutonomyState.enabled;
  try {
    ech0AutonomyButton.setAttribute("disabled", "disabled");
    ech0AutonomyStatus.textContent = targetState
      ? "Activating autonomous mode…"
      : "Pausing autonomous mode…";
    const response = await fetch("/api/ech0/autonomy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: targetState ? "enable" : "disable" }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }
    ech0AutonomyState = {
      enabled: Boolean(payload.enabled),
      updatedAt: payload.updatedAt || null,
      lastError: null,
    };
    renderEch0Autonomy();
  } catch (error) {
    ech0AutonomyState.lastError = error.message;
    renderEch0Autonomy();
  } finally {
    ech0AutonomyButton?.removeAttribute("disabled");
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  resizeRendererToDisplaySize();
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  resizeRendererToDisplaySize();
});

fetchInitialState();
subscribeToUpdates();
fetchPipelineData();
setInterval(fetchPipelineData, 60000);
fetchEch0HubModels();
setInterval(fetchEch0HubModels, 120000);
renderEch0Autonomy();
fetchEch0AutonomyState();
ech0ModelSelect?.addEventListener("change", (event) => {
  updateEch0ModelDetails(event.target.value);
});
ech0RefreshButton?.addEventListener("click", () => {
  fetchEch0HubModels({ force: true });
});
ech0AutonomyButton?.addEventListener("click", () => {
  toggleEch0Autonomy();
});
animate();
