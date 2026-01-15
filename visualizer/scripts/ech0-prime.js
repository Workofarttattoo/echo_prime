#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { ech0PrimeConfig } from "../config/ech0-prime.js";
import { SopEngine } from "./ech0-sop-engine.js";
import { WorkflowGraph } from "./ech0-workflow-graph.js";
import { TaskQueue } from "./ech0-task-queue.js";
import { TelemetryTracker } from "./ech0-telemetry.js";
import { SandboxRunner } from "./ech0-sandbox.js";
import { CrewOrchestrator } from "./ech0-crew-orchestrator.js";

const printHelp = () => {
  console.log("ech0 prime orchestrator");
  console.log("");
  console.log("Usage: node scripts/ech0-prime.js [options]");
  console.log("");
  console.log("Options:");
  console.log("  --task <text>          Queue a task description");
  console.log("  --role <role>          Apply role to following tasks");
  console.log("  --priority <number>    Apply priority to following tasks");
  console.log("  --file <path>          Load tasks from a JSON array");
  console.log("  -h, --help             Show this help message");
  console.log("");
  console.log("Set ECH0_PRIME_ENABLED=on to run tasks.");
};

const parseArgs = (rawArgs) => {
  const options = {
    tasks: [],
    defaultRole: null,
    defaultPriority: 0,
    file: null,
    help: false,
  };
  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    switch (arg) {
      case "--task":
        options.tasks.push({
          description: rawArgs[i + 1],
          role: options.defaultRole,
          priority: options.defaultPriority,
        });
        i += 1;
        break;
      case "--role":
        options.defaultRole = rawArgs[i + 1] || options.defaultRole;
        i += 1;
        break;
      case "--priority":
        options.defaultPriority = Number(rawArgs[i + 1]) || 0;
        i += 1;
        break;
      case "--file":
        options.file = rawArgs[i + 1] || options.file;
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

const loadTasks = async (filePath) => {
  if (!filePath) {
    return [];
  }
  const resolved = resolve(filePath);
  const raw = await readFile(resolved, "utf8");
  const payload = JSON.parse(raw);
  if (!Array.isArray(payload)) {
    throw new Error("Task file must be a JSON array");
  }
  return payload.map((task) => ({
    description: task.description || task.summary || "",
    role: task.role || null,
    priority: Number.isFinite(task.priority) ? task.priority : 0,
    command: task.command || null,
    payload: task.payload || {},
  }));
};

const buildOrchestrator = (config) => {
  const sopEngine = new SopEngine();
  const workflowGraph = new WorkflowGraph();
  sopEngine.listRoles().forEach((roleId) => {
    workflowGraph.addNode(roleId);
  });
  workflowGraph.setEntry(sopEngine.defaultRole);
  const taskQueue = new TaskQueue({ maxSize: config.queueMaxSize });
  const telemetry = new TelemetryTracker({ enabled: config.telemetryEnabled });
  const sandboxRunner = new SandboxRunner(config.sandbox);
  const orchestrator = new CrewOrchestrator({
    sopEngine,
    workflowGraph,
    taskQueue,
    telemetry,
    sandboxRunner,
  });
  orchestrator.registerRoleHandler("researcher", async ({ task, prompt }) => {
    return {
      content: `${prompt}\n\nFocus: ${task.description}`,
    };
  });
  orchestrator.registerRoleHandler(
    "engineer",
    async ({ task, prompt, sandboxRunner: runner }) => {
      let sandboxResult = null;
      if (task.command && runner) {
        sandboxResult = await runner.run(task.command, { dryRun: !runner.enabled });
      }
      return {
        content: `${prompt}\n\nImplementation focus: ${task.description}`,
        metadata: sandboxResult ? { sandbox: sandboxResult } : null,
      };
    }
  );
  orchestrator.registerRoleHandler("reviewer", async ({ task, prompt }) => {
    return {
      content: `${prompt}\n\nReview focus: ${task.description}`,
    };
  });
  return orchestrator;
};

const run = async () => {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  if (!ech0PrimeConfig.enabled) {
    console.log("[warn] ech0 prime is disabled. Set ECH0_PRIME_ENABLED=on.");
    return;
  }
  const fileTasks = await loadTasks(options.file).catch((error) => {
    console.log(`[error] ${error.message}`);
    return [];
  });
  const tasks = [...fileTasks, ...options.tasks].filter((task) => task.description);
  if (!tasks.length) {
    console.log("[warn] No tasks provided.");
    printHelp();
    return;
  }
  const orchestrator = buildOrchestrator(ech0PrimeConfig);
  tasks.forEach((task) => {
    orchestrator.queueTask(task, {
      role: task.role || options.defaultRole,
      priority: task.priority,
    });
  });
  console.log(`[info] Running ${tasks.length} ech0 prime task(s).`);
  const results = await orchestrator.runQueue();
  console.log(`[info] Completed ${results.length} task(s).`);
  const summary = orchestrator.telemetry.summary();
  console.log("[info] Telemetry summary:");
  console.log(JSON.stringify(summary, null, 2));
};

run();
