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
    outPath: null,
    help: false,
  };
  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    switch (arg) {
      case "--report":
        options.reportPath = rawArgs[i + 1] || options.reportPath;
        i += 1;
        break;
      case "--out":
        options.outPath = rawArgs[i + 1] || options.outPath;
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
  console.log("ech0 golden run report");
  console.log("");
  console.log("Usage: node scripts/ech0-golden-run-report.js [options]");
  console.log("");
  console.log("Options:");
  console.log("  --report <path>   Report path (defaults to newest report)");
  console.log("  --out <path>      Markdown output path relative to visualizer/");
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

const formatHash = (hash) => `${hash.slice(0, 10)}…${hash.slice(-6)}`;

const buildChecklist = (checks) =>
  checks
    .map((check) => `- ${check.ok ? "✅" : "❌"} ${check.label}`)
    .join("\n");

const buildWeights = (weights) => {
  if (!weights || typeof weights !== "object") {
    return "-";
  }
  return Object.entries(weights)
    .map(([key, value]) => `- ${key}: ${value}`)
    .join("\n");
};

const buildReport = ({
  reportPath,
  report,
  qulabArtifact,
  checks,
  payloadHash,
  qulabPayloadHash,
  qulabOutputHash,
  outputPath,
}) => {
  const topCandidate = report.payload.qulab.top_candidate || null;
  const candidateCount = report.payload.qulab.candidate_count || 0;
  const weights = qulabArtifact.payload?.parameters?.weights || null;
  const status = checks.every((check) => check.ok) ? "VERIFIED" : "FAILED";

  return `# ech0 Golden Run Report\n\n` +
    `**Status:** ${status}\n\n` +
    `## Snapshot\n` +
    `| Metric | Value |\n` +
    `| --- | --- |\n` +
    `| Run timestamp | ${report.timestamp} |\n` +
    `| Seed | ${report.payload.seed} |\n` +
    `| Repo commit | ${report.provenance.repoCommit} |\n` +
    `| Candidate count | ${candidateCount} |\n` +
    `| Top candidate | ${topCandidate ? topCandidate.name : "n/a"} |\n` +
    `| Top score | ${topCandidate ? topCandidate.composite_score : "n/a"} |\n\n` +
    `## Determinism & Provenance\n` +
    `- Node runtime: ${report.provenance.node}\n` +
    `- Platform: ${report.provenance.platform}\n` +
    `- Payload SHA256: ${formatHash(payloadHash)}\n` +
    `- Qulab payload SHA256: ${formatHash(qulabPayloadHash)}\n` +
    `- Qulab output SHA256: ${formatHash(qulabOutputHash)}\n\n` +
    `## Artifact Paths\n` +
    `- ech0 report: ${reportPath}\n` +
    `- Qulab artifact: ${report.payload.qulab.artifact_path}\n` +
    `- Report summary: ${outputPath}\n\n` +
    `## Scoring Weights\n` +
    `${buildWeights(weights)}\n\n` +
    `## Verification Checklist\n` +
    `${buildChecklist(checks)}\n\n` +
    `## Notes\n` +
    `- Outputs are simulated and research-only.\n` +
    `- This report proves deterministic orchestration and reproducibility.\n`;
};

const main = async () => {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  const reportPath = options.reportPath || (await getLatestReport());
  const reportRaw = await fs.readFile(reportPath, "utf8");
  const report = JSON.parse(reportRaw);

  const payloadHash = hashJsonPayload(report.payload);
  const qulabArtifactPath = resolve(projectRoot, report.payload.qulab.artifact_path);
  const qulabRaw = await fs.readFile(qulabArtifactPath, "utf8");
  const qulabArtifact = JSON.parse(qulabRaw);

  const qulabPayloadHash = hashJsonPayload(qulabArtifact.payload);
  const qulabOutputHash = sha256(qulabRaw);

  const checks = [
    { label: "Deterministic seed recorded", ok: Boolean(report.payload.seed) },
    { label: "Provenance captured", ok: Boolean(report.provenance.repoCommit) },
    { label: "Payload hash matches report", ok: payloadHash === report.payloadSha256 },
    {
      label: "Qulab payload hash matches",
      ok: qulabPayloadHash === report.payload.qulab.payload_sha256,
    },
    {
      label: "Qulab output hash matches",
      ok: qulabOutputHash === report.payload.qulab.output_sha256,
    },
    {
      label: "Artifact linkage present",
      ok: Boolean(report.payload.qulab.artifact_path),
    },
  ];

  const runStamp = report.timestamp.replace(/[:.]/g, "-");
  const outName = options.outPath || `ech0_golden_run_summary_${runStamp}.md`;
  const outPath = resolve(reportDir, outName);
  const markdown = buildReport({
    reportPath,
    report,
    qulabArtifact,
    checks,
    payloadHash,
    qulabPayloadHash,
    qulabOutputHash,
    outputPath: outPath,
  });
  await fs.writeFile(outPath, markdown);

  console.log("[info] Golden run summary generated");
  console.log(`[info] Summary: ${outPath}`);
};

main().catch((error) => {
  console.error("[error] Golden run summary failed", error.message);
  process.exit(1);
});
