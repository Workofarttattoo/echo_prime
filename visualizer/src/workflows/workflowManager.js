import { ExperimentWorkflow } from "./experimentWorkflow.js";

export class WorkflowManager {
  constructor(config = {}) {
    this.maxConcurrent = config.maxConcurrent || 3;
    this.stageDurations = config.stageDurations;
    this.stages = config.stages;
    this.workflows = new Map();
  }

  scheduleWorkflow(definition) {
    const workflow = new ExperimentWorkflow(definition, {
      stageDurations: this.stageDurations,
      stages: this.stages,
    });
    this.workflows.set(workflow.id, workflow);
    return workflow;
  }

  updateWorkflows(currentTime, payloadById = {}) {
    const active = Array.from(this.workflows.values()).filter(
      (workflow) => workflow.state !== "completed" && workflow.state !== "failed"
    );

    const running = active.filter((workflow) => workflow.state === "running");
    const availableSlots = Math.max(0, this.maxConcurrent - running.length);
    if (availableSlots > 0) {
      const scheduled = active.filter(
        (workflow) => workflow.state === "scheduled"
      );
      scheduled
        .slice(0, availableSlots)
        .forEach((workflow) => workflow.update(currentTime));
    }

    this.workflows.forEach((workflow, id) => {
      workflow.update(currentTime, payloadById[id]);
    });

    return this.getSnapshot();
  }

  getSnapshot() {
    return Array.from(this.workflows.values()).map((workflow) =>
      workflow.toJSON()
    );
  }

  ensureCapacity() {
    const completed = Array.from(this.workflows.values()).filter(
      (workflow) => workflow.state === "completed"
    );
    if (completed.length > 50) {
      completed.slice(0, completed.length - 50).forEach((workflow) => {
        this.workflows.delete(workflow.id);
      });
    }
  }
}
