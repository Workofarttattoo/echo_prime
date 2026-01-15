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
    title: "ech0 Golden Run One-Pager",
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
      case "--title":
        options.title = rawArgs[i + 1] || options.title;
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
  console.log("ech0 golden run one-pager");
  console.log("");
  console.log("Usage: node scripts/ech0-golden-run-onepager.js [options]");
  console.log("");
  console.log("Options:");
  console.log("  --report <path>   Report path (defaults to newest report)");
  console.log("  --out <path>      HTML output path relative to visualizer/");
  console.log("  --title <title>   Override the report title");
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

const renderChecks = (checks) =>
  checks
    .map(
      (check) =>
        `<li class="${check.ok ? "pass" : "fail"}">` +
        `${check.ok ? "✓" : "✕"} ${check.label}</li>`
    )
    .join("");

const renderMetricRow = (label, value) =>
  `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;

const renderWeightList = (weights) => {
  if (!weights || typeof weights !== "object") {
    return "<div class=\"empty\">n/a</div>";
  }
  return `<ul>${Object.entries(weights)
    .map(([key, value]) => `<li>${key}: ${value}</li>`)
    .join("")}</ul>`;
};

const buildHtml = ({
  title,
  report,
  reportPath,
  qulabArtifact,
  checks,
  payloadHash,
  qulabPayloadHash,
  qulabOutputHash,
}) => {
  const status = checks.every((check) => check.ok) ? "VERIFIED" : "FAILED";
  const topCandidate = report.payload.qulab.top_candidate || {};
  const weights = qulabArtifact.payload?.parameters?.weights || null;
  const candidateCount = report.payload.qulab.candidate_count || 0;
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>${title}</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f6f7fb;
        --card: #ffffff;
        --ink: #101828;
        --muted: #667085;
        --accent: #3b82f6;
        --success: #12b76a;
        --danger: #f04438;
        --border: #eaecf0;
      }
      body {
        margin: 0;
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        background: var(--bg);
        color: var(--ink);
      }
      .page {
        max-width: 900px;
        margin: 32px auto;
        padding: 24px;
      }
      header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
      }
      .badge {
        background: ${status === "VERIFIED" ? "var(--success)" : "var(--danger)"};
        color: white;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.04em;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
      }
      .card {
        background: var(--card);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      }
      h1 {
        font-size: 24px;
        margin: 0 0 4px;
      }
      h2 {
        font-size: 16px;
        margin: 0 0 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .metric {
        display: flex;
        justify-content: space-between;
        font-size: 14px;
        padding: 6px 0;
        border-bottom: 1px solid var(--border);
      }
      .metric:last-child {
        border-bottom: none;
      }
      .hash {
        font-family: "JetBrains Mono", "SFMono-Regular", monospace;
        font-size: 12px;
      }
      .muted {
        color: var(--muted);
        font-size: 13px;
      }
      ul {
        padding-left: 18px;
        margin: 0;
      }
      li {
        margin-bottom: 6px;
        font-size: 13px;
      }
      li.pass {
        color: var(--success);
      }
      li.fail {
        color: var(--danger);
      }
      footer {
        margin-top: 20px;
        font-size: 12px;
        color: var(--muted);
      }
      .empty {
        color: var(--muted);
        font-size: 13px;
      }
    </style>
  </head>
  <body>
    <div class="page">
      <header>
        <div>
          <h1>${title}</h1>
          <div class="muted">${report.timestamp} · Seed ${report.payload.seed}</div>
        </div>
        <span class="badge">${status}</span>
      </header>

      <div class="grid">
        <section class="card">
          <h2>Snapshot</h2>
          ${renderMetricRow("Repo commit", report.provenance.repoCommit)}
          ${renderMetricRow("Candidates", candidateCount)}
          ${renderMetricRow("Top candidate", topCandidate.name || "n/a")}
          ${renderMetricRow("Top score", topCandidate.composite_score ?? "n/a")}
          ${renderMetricRow("Platform", report.provenance.platform)}
        </section>

        <section class="card">
          <h2>Hashes</h2>
          <div class="metric"><span>ech0 payload</span><strong class="hash">${formatHash(
            payloadHash
          )}</strong></div>
          <div class="metric"><span>Qulab payload</span><strong class="hash">${formatHash(
            qulabPayloadHash
          )}</strong></div>
          <div class="metric"><span>Qulab output</span><strong class="hash">${formatHash(
            qulabOutputHash
          )}</strong></div>
          <div class="metric"><span>Node runtime</span><strong>${report.provenance.node}</strong></div>
        </section>

        <section class="card">
          <h2>Top Candidate</h2>
          ${renderMetricRow("Name", topCandidate.name || "n/a")}
          ${renderMetricRow("Stability", topCandidate.metrics?.stability ?? "n/a")}
          ${renderMetricRow("Yield", topCandidate.metrics?.yield ?? "n/a")}
          ${renderMetricRow("Novelty", topCandidate.metrics?.novelty ?? "n/a")}
          ${renderMetricRow("Risk", topCandidate.metrics?.risk ?? "n/a")}
        </section>

        <section class="card">
          <h2>Weights</h2>
          ${renderWeightList(weights)}
        </section>
      </div>

      <section class="card" style="margin-top: 16px;">
        <h2>Verification Checklist</h2>
        <ul>${renderChecks(checks)}</ul>
      </section>

      <section class="card" style="margin-top: 16px;">
        <h2>Artifacts</h2>
        <div class="muted">ech0 report: ${reportPath}</div>
        <div class="muted">Qulab artifact: ${report.payload.qulab.artifact_path}</div>
      </section>

      <footer>
        Outputs are simulated and research-only. This one-pager proves deterministic
        orchestration and reproducibility, not real-world lab production.
      </footer>
    </div>
  </body>
</html>`;
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
  const outName = options.outPath || `ech0_golden_run_onepager_${runStamp}.html`;
  const outPath = resolve(reportDir, outName);
  const html = buildHtml({
    title: options.title,
    report,
    reportPath,
    qulabArtifact,
    checks,
    payloadHash,
    qulabPayloadHash,
    qulabOutputHash,
  });
  await fs.writeFile(outPath, html);

  console.log("[info] Golden run one-pager generated");
  console.log(`[info] One-pager: ${outPath}`);
};

main().catch((error) => {
  console.error("[error] Golden run one-pager failed", error.message);
  process.exit(1);
});
