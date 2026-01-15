import { createServer } from "node:http";
import { promises as fs, watch } from "node:fs";
import { extname, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { createVisualizationSystem } from "../src/index.js";
import { defaultConfig } from "../config/default.js";
import { runDockerSmokeTest } from "./docker-smoke.js";
import { pipelineConfig } from "../config/pipeline.js";
import { ech0HubConfig } from "../config/ech0Hub.js";

const filePath = fileURLToPath(import.meta.url);
const baseDir = dirname(filePath);
const projectRoot = join(baseDir, "..");
const publicDir = join(projectRoot, "public");
const pipelineCache = {
  data: null,
  loadedAt: 0,
  errorLogged: false,
};
let pipelineWatcher;
let pipelineReloadTimer = null;
let pipelineRefreshTimer;
const pipelineRefreshInterval = Math.max(
  5000,
  pipelineConfig.refreshIntervalMs || 30000
);
const ech0HubCache = {
  payload: null,
  fetchedAt: 0,
  host: ech0HubConfig.host,
  errorLogged: false,
};
const ech0AutonomyState = {
  enabled: false,
  updatedAt: null,
  hostResponse: null,
  lastError: null,
};

const orchestrator = createVisualizationSystem(defaultConfig);
const clients = new Set();

orchestrator.onState((state) => {
  const payload = `data: ${JSON.stringify(state)}\n\n`;
  clients.forEach((response) => response.write(payload));
});

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml; charset=utf-8",
  ".ico": "image/x-icon",
};

async function serveStatic(request, response) {
  const url = request.url === "/" ? "/index.html" : request.url;
  const targetPath = join(publicDir, decodeURIComponent(url.split("?")[0]));
  try {
    const data = await fs.readFile(targetPath);
    const contentType =
      mimeTypes[extname(targetPath)] || "application/octet-stream";
    response.writeHead(200, { "Content-Type": contentType });
    response.end(data);
  } catch (error) {
    if (error.code === "ENOENT") {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Not Found");
      return;
    }
    response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Internal Server Error");
  }
}

async function loadPipelineSnapshot() {
  try {
    const stats = await fs.stat(pipelineConfig.dataPath);
    if (!pipelineCache.data || pipelineCache.loadedAt < stats.mtimeMs) {
      const raw = await fs.readFile(pipelineConfig.dataPath, "utf-8");
      const parsed = JSON.parse(raw);
      const summary = parsed.summary || {};
      const readiness = summary.readiness || {};
      const topReady = Array.isArray(parsed.top_ready_for_build)
        ? parsed.top_ready_for_build.slice(
            0,
            pipelineConfig.readySampleSize
          )
        : [];
      const validations = Array.isArray(parsed.validations)
        ? parsed.validations
        : [];
      const highlights = summary.highlights || {};
      pipelineCache.data = {
        generatedAt: parsed.generated_at,
        source: parsed.source || null,
        totalInventions:
          parsed.total_inventions || validations.length || topReady.length,
        readiness: {
          readyForBuild: readiness["ready-for-build"] || 0,
          needsIteration: readiness["needs-iteration"] || 0,
          backlog: readiness.backlog || 0,
        },
        topReady,
        summary,
        validations,
        attentionQueue: Array.isArray(highlights.attention_queue)
          ? highlights.attention_queue
          : [],
        categoryBreakdown: Array.isArray(summary.category_breakdown)
          ? summary.category_breakdown
          : [],
      };
      pipelineCache.loadedAt = stats.mtimeMs;
      pipelineCache.errorLogged = false;
    }
    return pipelineCache.data;
  } catch (error) {
    if (!pipelineCache.errorLogged) {
      console.error("[error] Failed to load pipeline data", error);
      pipelineCache.errorLogged = true;
    }
    pipelineCache.data = {
      generatedAt: null,
      source: null,
      totalInventions: 0,
      readiness: {
        readyForBuild: 0,
        needsIteration: 0,
        backlog: 0,
      },
      topReady: [],
      summary: {
        readiness: {
          "ready-for-build": 0,
          "needs-iteration": 0,
          backlog: 0,
        },
        parliament: { APPROVED: 0, NEEDS_REFINEMENT: 0, ON_HOLD: 0 },
        alex: { APPROVED: 0, NEEDS_REVISION: 0, REJECTED: 0 },
        metrics: {},
        category_breakdown: [],
        highlights: { attention_queue: [] },
      },
      validations: [],
      attentionQueue: [],
      categoryBreakdown: [],
    };
    pipelineCache.loadedAt = Date.now();
    return pipelineCache.data;
  }
}

function emitPipelineSnapshot(snapshot, target) {
  if (!snapshot) {
    return;
  }
  const payload = `event: pipeline\ndata: ${JSON.stringify(snapshot)}\n\n`;
  if (target) {
    target.write(payload);
    return;
  }
  clients.forEach((response) => response.write(payload));
}

async function refreshPipelineSnapshot(options = {}) {
  const { broadcast = false, target = null, force = false } = options;
  const previousStamp = pipelineCache.loadedAt;
  const snapshot = await loadPipelineSnapshot();
  if (!snapshot) {
    return null;
  }
  const changed = pipelineCache.loadedAt !== previousStamp;
  if (target || force || (broadcast && changed)) {
    emitPipelineSnapshot(snapshot, target || undefined);
  }
  return snapshot;
}

function normalizeModelEntry(entry) {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const name = entry.name || entry.model || entry.tag;
  if (!name) {
    return null;
  }
  const modifiedRaw =
    entry.modified_at ||
    entry.modifiedAt ||
    entry.updated_at ||
    entry.updatedAt ||
    entry.created_at ||
    entry.createdAt ||
    null;
  const modifiedMs = modifiedRaw ? Date.parse(modifiedRaw) : null;
  return {
    name,
    digest: entry.digest || entry.sha256 || null,
    size: typeof entry.size === "number" ? entry.size : null,
    modifiedAt: modifiedMs ? new Date(modifiedMs).toISOString() : null,
    modifiedMs: modifiedMs || 0,
  };
}

function resolveLatestModel(models, namespace) {
  if (!Array.isArray(models) || !models.length) {
    return null;
  }
  if (!namespace) {
    return models[0]?.name || null;
  }
  const prefix = namespace.toLowerCase();
  const match = models.find((model) =>
    model.name.toLowerCase().startsWith(prefix)
  );
  return match?.name || null;
}

async function refreshEch0HubModels(options = {}) {
  const { force = false } = options;
  const now = Date.now();
  const isStale =
    force ||
    !ech0HubCache.payload ||
    ech0HubCache.host !== ech0HubConfig.host ||
    now - ech0HubCache.fetchedAt > ech0HubConfig.cacheMs;
  if (!isStale) {
    return ech0HubCache.payload;
  }

  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    ech0HubConfig.timeoutMs
  );
  try {
    const response = await fetch(`${ech0HubConfig.host}/api/tags`, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`ech0 hub responded with status ${response.status}`);
    }
    const payload = await response.json();
    const normalized = Array.isArray(payload?.models)
      ? payload.models
          .map((entry) => normalizeModelEntry(entry))
          .filter(Boolean)
          .sort((a, b) => (b.modifiedMs || 0) - (a.modifiedMs || 0))
          .map((entry) => {
            const { modifiedMs, ...rest } = entry;
            return rest;
          })
      : [];
    const formatted = {
      host: ech0HubConfig.host,
      fetchedAt: new Date().toISOString(),
      latestModel: normalized[0]?.name || null,
      latestEch0Model: resolveLatestModel(normalized, ech0HubConfig.namespace),
      totalModels: normalized.length,
      models: normalized,
    };
    ech0HubCache.payload = formatted;
    ech0HubCache.fetchedAt = now;
    ech0HubCache.host = ech0HubConfig.host;
    ech0HubCache.errorLogged = false;
    return formatted;
  } catch (error) {
    if (!ech0HubCache.errorLogged) {
      console.error("[error] ech0 hub lookup failed", error);
      ech0HubCache.errorLogged = true;
    }
    throw new Error(
      `Unable to reach ech0 hub at ${ech0HubConfig.host}: ${error.message}`
    );
  } finally {
    clearTimeout(timeout);
  }
}

function schedulePipelineReload() {
  if (pipelineReloadTimer) {
    return;
  }
  pipelineReloadTimer = setTimeout(() => {
    pipelineReloadTimer = null;
    refreshPipelineSnapshot({ broadcast: true }).catch((error) =>
      console.error("[error] Pipeline refresh failed", error)
    );
  }, 250);
}

function respondJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
  });
  response.end(JSON.stringify(payload));
}

async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  if (!chunks.length) {
    return {};
  }
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  if (!raw.length) {
    return {};
  }
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error("Invalid JSON payload");
  }
}

function normalizeAutonomyAction(options = {}) {
  const { enable, action } = options;
  if (typeof enable === "boolean") {
    return enable ? "enable" : "disable";
  }
  if (typeof action === "string") {
    const lowered = action.toLowerCase();
    if (["enable", "disable"].includes(lowered)) {
      return lowered;
    }
    if (lowered === "toggle") {
      return ech0AutonomyState.enabled ? "disable" : "enable";
    }
  }
  return ech0AutonomyState.enabled ? "disable" : "enable";
}

async function sendAutonomyCommand(action) {
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    ech0HubConfig.timeoutMs
  );
  try {
    const response = await fetch(`${ech0HubConfig.host}/api/autonomy`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ action }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(
        `ech0 hub responded with ${response.status}: ${text || "error"}`
      );
    }
    const payload = await response.json().catch(() => ({}));
    return payload;
  } catch (error) {
    throw new Error(
      `Unable to send autonomy command to ech0 hub at ${ech0HubConfig.host}: ${error.message}`
    );
  } finally {
    clearTimeout(timeout);
  }
}

async function startPipelineWatcher() {
  try {
    await fs.access(pipelineConfig.dataPath);
  } catch (error) {
    setTimeout(startPipelineWatcher, pipelineRefreshInterval);
    return;
  }

  try {
    pipelineWatcher?.close?.();
    pipelineWatcher = watch(pipelineConfig.dataPath, (eventType) => {
      if (eventType === "change" || eventType === "rename") {
        schedulePipelineReload();
      }
    });
    pipelineWatcher.on("error", (error) => {
      console.error("[error] Pipeline watcher error", error);
      pipelineWatcher?.close?.();
      setTimeout(startPipelineWatcher, pipelineRefreshInterval);
    });
  } catch (error) {
    console.error("[error] Unable to start pipeline watcher", error);
    setTimeout(startPipelineWatcher, pipelineRefreshInterval);
  }
}

const server = createServer(async (request, response) => {
  if (request.url === "/api/state") {
    response.writeHead(200, {
      "Content-Type": "application/json; charset=utf-8",
    });
    response.end(JSON.stringify(orchestrator.state.snapshot()));
    return;
  }

  if (request.url === "/api/events") {
    response.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      Connection: "keep-alive",
      "Cache-Control": "no-cache",
    });
    response.write("event: heartbeat\ndata: connected\n\n");
    clients.add(response);
    request.on("close", () => {
      clients.delete(response);
    });
    try {
      await refreshPipelineSnapshot({ target: response, force: true });
    } catch (error) {
      console.error("[error] Unable to send initial pipeline snapshot", error);
    }
    return;
  }

  if (request.url && request.url.startsWith("/api/ech0/models")) {
    const parsedUrl = new URL(request.url, "http://localhost");
    const shouldForce = ["1", "true"].includes(
      (parsedUrl.searchParams.get("refresh") || "").toLowerCase()
    );
    try {
      const snapshot = await refreshEch0HubModels({ force: shouldForce });
      response.writeHead(200, {
        "Content-Type": "application/json; charset=utf-8",
      });
      response.end(JSON.stringify(snapshot));
    } catch (error) {
      response.writeHead(502, {
        "Content-Type": "application/json; charset=utf-8",
      });
      response.end(
        JSON.stringify({
          error: error.message,
          host: ech0HubConfig.host,
        })
      );
    }
    return;
  }

  if (request.url && request.url.startsWith("/api/ech0/autonomy")) {
    if (request.method === "GET") {
      respondJson(response, 200, {
        enabled: ech0AutonomyState.enabled,
        updatedAt: ech0AutonomyState.updatedAt,
        hostResponse: ech0AutonomyState.hostResponse,
        host: ech0HubConfig.host,
        lastError: ech0AutonomyState.lastError,
      });
      return;
    }
    if (request.method !== "POST") {
      respondJson(response, 405, { error: "Method not allowed" });
      return;
    }
    try {
      const body = await readJsonBody(request);
      const action = normalizeAutonomyAction(body || {});
      const enable = action === "enable";
      const hostResponse = await sendAutonomyCommand(action);
      ech0AutonomyState.enabled = enable;
      ech0AutonomyState.updatedAt = new Date().toISOString();
      ech0AutonomyState.hostResponse = hostResponse;
      ech0AutonomyState.lastError = null;
      respondJson(response, 200, {
        enabled: ech0AutonomyState.enabled,
        updatedAt: ech0AutonomyState.updatedAt,
        hostResponse: ech0AutonomyState.hostResponse,
        host: ech0HubConfig.host,
      });
    } catch (error) {
      ech0AutonomyState.lastError = error.message;
      respondJson(response, 502, {
        error: error.message,
        host: ech0HubConfig.host,
        enabled: ech0AutonomyState.enabled,
      });
    }
    return;
  }

  if (request.url === "/api/pipeline") {
    try {
      const snapshot = await refreshPipelineSnapshot();
      response.writeHead(200, {
        "Content-Type": "application/json; charset=utf-8",
      });
      response.end(JSON.stringify(snapshot));
    } catch (error) {
      response.writeHead(503, {
        "Content-Type": "application/json; charset=utf-8",
      });
      response.end(
        JSON.stringify({
          error: "Pipeline data unavailable",
        })
      );
    }
    return;
  }

  await serveStatic(request, response);
});

server.listen(defaultConfig.port, () => {
  console.log(
    `[info] 3D visualization server listening on port ${defaultConfig.port}`
  );
});

refreshPipelineSnapshot()
  .then(() => emitPipelineSnapshot(pipelineCache.data))
  .catch((error) =>
    console.error("[error] Initial pipeline load failed", error)
  );
startPipelineWatcher();
pipelineRefreshTimer = setInterval(() => {
  refreshPipelineSnapshot({ broadcast: true }).catch((error) =>
    console.error("[error] Scheduled pipeline refresh failed", error)
  );
}, pipelineRefreshInterval);

const enableSmoke = process.env.ENABLE_DOCKER_SMOKE !== "off";
if (enableSmoke) {
  runDockerSmokeTest().catch((error) => {
    console.error("[error] Smoke test failed", error);
  });
}

function shutdown() {
  orchestrator.stop();
  clients.forEach((response) => response.end());
  pipelineWatcher?.close?.();
  if (pipelineRefreshTimer) {
    clearInterval(pipelineRefreshTimer);
    pipelineRefreshTimer = undefined;
  }
  if (pipelineReloadTimer) {
    clearTimeout(pipelineReloadTimer);
    pipelineReloadTimer = null;
  }
  server.close(() => {
    console.log("[info] Server closed");
    process.exit(0);
  });
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
