import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const configPath = fileURLToPath(import.meta.url);
const configDir = dirname(configPath);
const projectRoot = dirname(configDir);

export const pipelineConfig = {
  dataPath:
    process.env.PIPELINE_DATA_PATH ||
    join(projectRoot, "..", "consciousness", "ech0_invention_pipeline_validations.json"),
  readySampleSize: Number.parseInt(
    process.env.PIPELINE_READY_SAMPLE || "12",
    10
  ),
  refreshIntervalMs: Number.parseInt(
    process.env.PIPELINE_REFRESH_MS || "30000",
    10
  ),
};
