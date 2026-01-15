#!/usr/bin/env node

import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const baseDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(baseDir, "..");
const reportDir = resolve(projectRoot, "logs", "ech0-golden-runs");

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

const sha256 = (data) => createHash("sha256").update(data).digest("hex");

const hashJsonPayload = (payload) => {
  const serialized = JSON.stringify(sortObject(payload));
  return sha256(serialized);
};

const parseArgs = (rawArgs) => {
  const options = {
    reportPath: null,
    help: false,
  };
  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    switch (arg) {
      case "--report":
        options.reportPath = rawArgs[i + 1] || options.reportPath;
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
  console.log("ech0 golden run verifier");
  console.log("");
  console.log("Usage: node scripts/ech0-golden-run-verify.js [options]");
  console.log("");
  console.log("Options:");
  console.log("  --report <path>   Report path (defaults to newest report)");
  console.log("  -h, --help        Show this help message");
};

const getLatestReport = async () => {
  const entries = await fs.readdir(reportDir);
  const reports = entries.filter((entry) => entry.endsWith(".json"));
  if (!reports.length) {
    throw new Error("No golden run reports found");
  }
  const stats = await Promise.all(
    reports.map(async (entry) => {
      const fullPath = resolve(reportDir, entry);
      const stat = await fs.stat(fullPath);
      return { entry, fullPath, mtime: stat.mtimeMs };
    })
  );
  stats.sort((a, b) => b.mtime - a.mtime);
  return stats[0].fullPath;
};

const ensureFields = (label, obj, fields) => {
  const missing = fields.filter((field) => !obj || obj[field] === undefined);
  if (missing.length) {
    throw new Error(`${label} missing fields: ${missing.join(", ")}`);
  }
};

const verify = async (reportPath) => {
  const reportRaw = await fs.readFile(reportPath, "utf8");
  const report = JSON.parse(reportRaw);

  ensureFields("report", report, ["timestamp", "payloadSha256", "provenance", "payload"]);
  ensureFields("provenance", report.provenance, ["repoCommit", "node", "platform"]);
  ensureFields("payload", report.payload, ["seed", "ech0", "qulab", "disclaimer"]);
  ensureFields("qulab summary", report.payload.qulab, [
    "artifact_path",
    "payload_sha256",
    "output_sha256",
    "seed",
  ]);

  const computedPayloadHash = hashJsonPayload(report.payload);
  if (computedPayloadHash !== report.payloadSha256) {
    throw new Error("payload hash mismatch in ech0 report");
  }

  const qulabArtifactPath = resolve(projectRoot, report.payload.qulab.artifact_path);
  const qulabRaw = await fs.readFile(qulabArtifactPath, "utf8");
  const qulabArtifact = JSON.parse(qulabRaw);
  ensureFields("qulab artifact", qulabArtifact, ["payload", "provenance"]);

  const qulabPayloadHash = hashJsonPayload(qulabArtifact.payload);
  const qulabOutputHash = sha256(qulabRaw);

  if (qulabPayloadHash !== report.payload.qulab.payload_sha256) {
    throw new Error("Qulab payload hash mismatch");
  }
  if (qulabOutputHash !== report.payload.qulab.output_sha256) {
    throw new Error("Qulab output hash mismatch");
  }

  return {
    reportPath,
    payloadSha256: computedPayloadHash,
    qulabArtifact: report.payload.qulab.artifact_path,
    qulabPayloadSha256: qulabPayloadHash,
    qulabOutputSha256: qulabOutputHash,
    seed: report.payload.seed,
  };
};

const main = async () => {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  const reportPath = options.reportPath || (await getLatestReport());
  const summary = await verify(reportPath);
  console.log("[info] Golden run verification passed");
  console.log(`[info] Report: ${summary.reportPath}`);
  console.log(`[info] Seed: ${summary.seed}`);
  console.log(`[info] Report payload SHA256: ${summary.payloadSha256}`);
  console.log(`[info] Qulab artifact: ${summary.qulabArtifact}`);
  console.log(`[info] Qulab payload SHA256: ${summary.qulabPayloadSha256}`);
  console.log(`[info] Qulab output SHA256: ${summary.qulabOutputSha256}`);
};

main().catch((error) => {
  console.error("[error] Golden run verification failed", error.message);
  process.exit(1);
});
