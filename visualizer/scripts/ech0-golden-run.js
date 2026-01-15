#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import { dirname, resolve, join } from "node:path";
import { fileURLToPath } from "node:url";

import { SopEngine } from "./ech0-sop-engine.js";

const baseDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(baseDir, "..");
const qulabScript = resolve(
  projectRoot,
  "qulab-infinite",
  "scripts",
  "golden-run-smoke.py"
);
const defaultOutDir = resolve(projectRoot, "logs", "ech0-golden-runs");
const defaultSeed = Number.parseInt(process.env.ECH0_GOLDEN_SEED || "1337", 10);
const pythonBin = process.env.ECH0_GOLDEN_PYTHON || "python3";

const sortObject = (value) => {
  if (Array.isArray(value)) {
    return value.map((entry) => sortObject(entry));
  }
  if (value && typeof value === "object") {
    return Object.keys(value)
      .sort()
      .reduce((accumulator, key) => {
        accumulator[key] = sortObject(value[key]);
        return accumulator;
      }, {});
  }
  return value;
};

const hashPayload = (payload) => {
  const serialized = JSON.stringify(sortObject(payload));
  return createHash("sha256").update(serialized).digest("hex");
};

const runCommand = (command, args, options = {}) =>
  new Promise((resolvePromise) => {
    const child = spawn(command, args, options);
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (code) => {
      resolvePromise({ code, stdout, stderr });
    });
  });

const parseArgs = (rawArgs) => {
  const options = {
    seed: Number.isNaN(defaultSeed) ? 1337 : defaultSeed,
    out: null,
    help: false,
  };
  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    switch (arg) {
      case "--seed":
        options.seed = Number.parseInt(rawArgs[i + 1] || "", 10) || options.seed;
        i += 1;
        break;
      case "--out":
        options.out = rawArgs[i + 1] || options.out;
        i += 1;
        break;
      case "-h":
      case "--help":
        options.help = true;
        break;
      default:
        break;
    }
  }
  return options;
};

const printHelp = () => {
  console.log("ech0 golden run orchestrator");
  console.log("");
  console.log("Usage: node scripts/ech0-golden-run.js [options]");
  console.log("");
  console.log("Options:");
  console.log("  --seed <number>     Deterministic seed (default 1337)");
  console.log("  --out <path>        Output report path relative to visualizer/");
  console.log("  -h, --help          Show this help message");
};

const getCommitHash = async () => {
  const result = await runCommand("git", ["rev-parse", "HEAD"], {
    cwd: projectRoot,
  });
  if (result.code !== 0) {
    return "UNKNOWN";
  }
  return result.stdout.trim() || "UNKNOWN";
};

const runQulabGoldenRun = async (seed) => {
  const result = await runCommand(
    pythonBin,
    [qulabScript, "--seed", String(seed), "--json"],
    { cwd: projectRoot }
  );
  if (result.code !== 0) {
    throw new Error(result.stderr || "Qulab golden run failed");
  }
  try {
    return JSON.parse(result.stdout.trim());
  } catch (error) {
    throw new Error("Failed to parse Qulab JSON output");
  }
};

const buildReport = async ({ seed, qulabSummary }) => {
  const sopEngine = new SopEngine();
  const intakePrompt = sopEngine.buildPrompt(
    "researcher",
    "Define the golden run objective and success criteria.",
    {
      seed,
      qulabScript: "qulab-infinite/scripts/golden-run-smoke.py",
      target: "Deterministic pipeline proof",
    }
  );
  const reviewPrompt = sopEngine.buildPrompt(
    "reviewer",
    "Assess the deterministic output and highlight risks.",
    {
      seed,
      topCandidate: qulabSummary.top_candidate?.name || "none",
      artifactPath: qulabSummary.artifact_path,
      payloadSha256: qulabSummary.payload_sha256,
    }
  );
  const reviewSummary = [
    `Deterministic run completed with seed ${seed}.`,
    `Top candidate: ${qulabSummary.top_candidate?.name || "none"}.`,
    `Artifact hash: ${qulabSummary.payload_sha256}.`,
    "Outputs are simulated and non-clinical.",
  ].join(" ");
  const payload = {
    seed,
    ech0: {
      intakePrompt,
      reviewPrompt,
      reviewSummary,
    },
    qulab: qulabSummary,
    disclaimer: "SIMULATION ONLY. NOT FOR CLINICAL OR PRODUCTION USE.",
  };
  return payload;
};

const main = async () => {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  const seed = options.seed;
  const qulabSummary = await runQulabGoldenRun(seed);
  const payload = await buildReport({ seed, qulabSummary });
  const payloadSha256 = hashPayload(payload);
  const timestamp = new Date().toISOString();
  const report = {
    timestamp,
    payloadSha256,
    provenance: {
      repoCommit: await getCommitHash(),
      node: process.version,
      platform: process.platform,
    },
    payload,
  };
  const runStamp = timestamp.replace(/[:.]/g, "-");
  const outName = options.out || `ech0_golden_run_${runStamp}_seed${seed}.json`;
  const outPath = resolve(defaultOutDir, outName);
  await fs.mkdir(dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(report, null, 2));
  const outputSha256 = createHash("sha256")
    .update(await fs.readFile(outPath))
    .digest("hex");
  console.log("[info] ech0 golden run complete");
  console.log(`[info] Report: ${outPath}`);
  console.log(`[info] Payload SHA256: ${payloadSha256}`);
  console.log(`[info] Output SHA256: ${outputSha256}`);
};

main().catch((error) => {
  console.error("[error] ech0 golden run failed", error.message);
  process.exit(1);
});
