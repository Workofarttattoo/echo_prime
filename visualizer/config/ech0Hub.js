const DEFAULT_HOST =
  process.env.ECH0_HUB_HOST ||
  process.env.OLLAMA_HOST ||
  "http://localhost:11434";

function normalizeHost(value) {
  if (!value) {
    return "http://localhost:11434";
  }
  let normalized = value.trim();
  if (!normalized) {
    return "http://localhost:11434";
  }
  if (!/^https?:\/\//i.test(normalized)) {
    normalized = `http://${normalized}`;
  }
  return normalized.replace(/\/+$/, "");
}

function parseNumber(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return parsed;
}

export const ech0HubConfig = {
  host: normalizeHost(DEFAULT_HOST),
  namespace: (process.env.ECH0_HUB_NAMESPACE || "ech0").trim(),
  timeoutMs: Math.max(
    500,
    parseNumber(process.env.ECH0_HUB_TIMEOUT_MS || "15000", 15000)
  ),
  cacheMs: Math.max(
    1000,
    parseNumber(process.env.ECH0_HUB_CACHE_MS || "30000", 30000)
  ),
};
