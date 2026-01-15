import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  SandboxRuntime,
  TaskDecomposer,
  createMissionProfile,
  loadRoleTemplate,
} from "../echo-cairo/index.js";

const filePath = fileURLToPath(import.meta.url);
const baseDir = dirname(filePath);
const templatePath = join(
  baseDir,
  "..",
  "echo-cairo",
  "agents",
  "templates",
  "sample-roles.json"
);

const template = await loadRoleTemplate(templatePath);
const mission = createMissionProfile(template);

const model = {
  async call({ mode, goal, tasks, results }) {
    if (mode === "analyze") {
      return { tasks: [`Clarify scope: ${goal}`, "Draft outline"] };
    }
    if (mode === "create") {
      return { tasks: ["Refine draft", "Publish summary"] };
    }
    if (mode === "summarize") {
      return {
        summary: `Summary for ${goal}: ${results.length} task(s) done.`,
      };
    }
    return { tasks: tasks || [] };
  },
};

const decomposer = new TaskDecomposer({ model, maxTasks: 4 });
const run = await decomposer.run(
  "Build an onboarding outline",
  async (task) => `Completed: ${task}`
);

const sandbox = new SandboxRuntime();
const sandboxResult = await sandbox.execute({ command: ["pwd"] });

console.log("Mission:", JSON.stringify(mission, null, 2));
console.log("Task run:", JSON.stringify(run, null, 2));
console.log("Sandbox:", JSON.stringify(sandboxResult, null, 2));
