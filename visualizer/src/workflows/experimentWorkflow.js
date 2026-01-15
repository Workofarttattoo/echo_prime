const defaultStageDurations = {
  "ingesting-data": 1500,
  "predicting-crystal-structure": 2000,
  "running-simulations": 2500,
  "optimizing-experiment": 1800,
  "awaiting-validation": 1200,
};

export class ExperimentWorkflow {
  constructor(definition, options = {}) {
    this.id = definition.id;
    this.material = definition.material;
    this.parameters = definition.parameters;
    this.createdAt = Date.now();
    this.stageIndex = 0;
    this.state = "scheduled";
    this.stageDurations = options.stageDurations || defaultStageDurations;
    this.stages = options.stages || [
      "ingesting-data",
      "predicting-crystal-structure",
      "running-simulations",
      "optimizing-experiment",
      "awaiting-validation",
    ];
    this.logs = [];
    this.metrics = {};
    this.nextStageTime = this.createdAt + this.stageDurations[this.stages[0]];
  }

  toJSON() {
    return {
      id: this.id,
      materialId: this.material.id,
      parameters: this.parameters,
      stage: this.stages[this.stageIndex],
      state: this.state,
      createdAt: this.createdAt,
      logs: this.logs.slice(-12),
      metrics: this.metrics,
    };
  }

  update(currentTime, payload = {}) {
    if (this.state === "completed" || this.state === "failed") {
      return this.state;
    }

    if (this.state === "scheduled") {
      this.state = "running";
      this.appendLog(`Workflow started at stage ${this.stages[0]}`);
    }

    if (currentTime >= this.nextStageTime) {
      this.stageIndex += 1;
      if (this.stageIndex >= this.stages.length) {
        this.state = "completed";
        this.metrics = payload.metrics || this.metrics;
        this.appendLog("Workflow completed successfully");
        return this.state;
      }
      const stage = this.stages[this.stageIndex];
      this.nextStageTime =
        currentTime + (this.stageDurations[stage] || 1500);
      this.appendLog(`Advanced to stage ${stage}`);
    }

    if (payload.metrics) {
      this.metrics = { ...this.metrics, ...payload.metrics };
    }

    if (payload.log) {
      this.appendLog(payload.log);
    }

    return this.state;
  }

  appendLog(entry) {
    this.logs.push({
      entry,
      timestamp: Date.now(),
    });
    if (this.logs.length > 200) {
      this.logs.shift();
    }
  }
}
