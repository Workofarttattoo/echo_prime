import { SopEngine } from "./ech0-sop-engine.js";
import { WorkflowGraph } from "./ech0-workflow-graph.js";
import { TaskQueue } from "./ech0-task-queue.js";
import { TelemetryTracker } from "./ech0-telemetry.js";

export class CrewOrchestrator {
  constructor(options = {}) {
    this.sopEngine = options.sopEngine || new SopEngine();
    this.workflowGraph = options.workflowGraph || new WorkflowGraph();
    this.taskQueue = options.taskQueue || new TaskQueue();
    this.telemetry = options.telemetry || new TelemetryTracker({ enabled: false });
    this.sandboxRunner = options.sandboxRunner || null;
    this.roleHandlers = new Map();
    this.conversation = [];
    this.defaultRole = options.defaultRole || this.sopEngine.defaultRole;
  }

  registerRoleHandler(roleId, handler) {
    if (!roleId || typeof handler !== "function") {
      return false;
    }
    this.roleHandlers.set(roleId, handler);
    if (!this.workflowGraph.hasNode(roleId)) {
      this.workflowGraph.addNode(roleId);
    }
    return true;
  }

  queueTask(task, options = {}) {
    return this.taskQueue.add(task, options);
  }

  async runQueue() {
    const results = [];
    while (this.taskQueue.hasPending()) {
      const task = this.taskQueue.next();
      if (!task) {
        break;
      }
      const result = await this.runTask(task);
      this.taskQueue.markComplete(task.id, result);
      results.push(result);
    }
    return results;
  }

  async runTask(task) {
    const roleId = task.role || this.defaultRole;
    const role = this.sopEngine.getRole(roleId);
    const prompt = this.sopEngine.buildPrompt(roleId, task.description, task.payload);
    const handler = this.roleHandlers.get(roleId) || this.defaultHandler;
    const span = this.telemetry.startSpan("role-turn", {
      role: roleId,
      taskId: task.id,
    });
    try {
      const response = await handler({
        task,
        role,
        prompt,
        conversation: [...this.conversation],
        sandboxRunner: this.sandboxRunner,
      });
      const output = typeof response === "string" ? response : response?.content;
      this.conversation.push({
        role: roleId,
        content: output || "",
        taskId: task.id,
        recordedAt: new Date().toISOString(),
      });
      this.workflowGraph.updateContext({
        lastTaskId: task.id,
        lastRole: roleId,
      });
      if (this.workflowGraph.hasNode(roleId)) {
        this.workflowGraph.moveTo(roleId, { taskId: task.id, role: roleId });
      }
      this.telemetry.recordOutcome(true);
      span.end({ status: "ok" });
      return {
        ok: true,
        taskId: task.id,
        role: roleId,
        output: output || "",
        workflow: this.workflowGraph.getState(),
        metadata: response?.metadata || null,
      };
    } catch (error) {
      this.telemetry.recordOutcome(false);
      span.end({ status: "error", message: error.message });
      return {
        ok: false,
        taskId: task.id,
        role: roleId,
        error: error.message,
      };
    }
  }

  defaultHandler({ prompt }) {
    return { content: prompt };
  }

  getTranscript() {
    return this.conversation.map((entry) => ({ ...entry }));
  }
}
