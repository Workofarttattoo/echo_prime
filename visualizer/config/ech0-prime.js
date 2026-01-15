const asNumber = (value, fallback) => {
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
};

const asBoolean = (value, fallback = false) => {
  if (value === undefined) {
    return fallback;
  }
  return value === "on" || value === "true" || value === "1";
};

export const ech0PrimeConfig = {
  enabled: asBoolean(process.env.ECH0_PRIME_ENABLED, false),
  autonomyLevel: asNumber(process.env.ECH0_PRIME_AUTONOMY_LEVEL, 3),
  maxParallel: asNumber(process.env.ECH0_PRIME_MAX_PARALLEL, 1),
  queueMaxSize: asNumber(process.env.ECH0_PRIME_QUEUE_MAX, 50),
  telemetryEnabled: asBoolean(process.env.ECH0_PRIME_TELEMETRY, true),
  checkpointDir:
    process.env.ECH0_PRIME_CHECKPOINT_DIR?.trim() ||
    "logs/ech0-prime-checkpoints",
  sandbox: {
    enabled: asBoolean(process.env.ECH0_PRIME_SANDBOX_ENABLED, false),
    image:
      process.env.ECH0_PRIME_SANDBOX_IMAGE?.trim() || "node:20-alpine",
    timeoutMs: asNumber(process.env.ECH0_PRIME_SANDBOX_TIMEOUT_MS, 30000),
    cpuLimit: process.env.ECH0_PRIME_SANDBOX_CPU?.trim() || "1",
    memoryLimit: process.env.ECH0_PRIME_SANDBOX_MEMORY?.trim() || "512m",
    networkMode: process.env.ECH0_PRIME_SANDBOX_NETWORK?.trim() || "none",
    workdir: process.env.ECH0_PRIME_SANDBOX_WORKDIR?.trim() || "/workspace",
    dryRun: asBoolean(process.env.ECH0_PRIME_SANDBOX_DRY_RUN, true),
  },
};
