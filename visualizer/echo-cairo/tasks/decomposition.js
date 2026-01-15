function splitGoal(goal) {
  if (!goal) {
    return [];
  }
  const chunks = goal
    .split(/\n|\.|;/)
    .map((part) => part.trim())
    .filter(Boolean);
  return chunks.length ? chunks : [goal];
}

export class TaskDecomposer {
  constructor({ model, maxTasks = 5 } = {}) {
    if (!model || typeof model.call !== "function") {
      throw new Error("TaskDecomposer requires a model with call().");
    }
    this.model = model;
    this.maxTasks = maxTasks;
  }

  async _callModel(mode, payload) {
    const response = await this.model.call({ mode, ...payload });
    return response || {};
  }

  async analyze(goal, context = {}) {
    const response = await this._callModel("analyze", { goal, context });
    if (Array.isArray(response.tasks)) {
      return response.tasks.slice(0, this.maxTasks);
    }
    return splitGoal(goal).slice(0, this.maxTasks);
  }

  async execute(tasks, executor, context = {}) {
    if (typeof executor !== "function") {
      return tasks.map((task) => ({ task, result: null }));
    }
    const results = [];
    for (const task of tasks) {
      const result = await executor(task, context);
      results.push({ task, result });
    }
    return results;
  }

  async createNextTasks(goal, tasks, results, context = {}) {
    const response = await this._callModel("create", {
      goal,
      tasks,
      results,
      context,
    });
    if (Array.isArray(response.tasks)) {
      return response.tasks.slice(0, this.maxTasks);
    }
    return [];
  }

  async summarize(goal, results, context = {}) {
    const response = await this._callModel("summarize", {
      goal,
      results,
      context,
    });
    if (response.summary) {
      return response.summary;
    }
    return results.map((entry) => `${entry.task}: ${entry.result ?? ""}`).join("\n");
  }

  async run(goal, executor, context = {}) {
    const tasks = await this.analyze(goal, context);
    const results = await this.execute(tasks, executor, context);
    const nextTasks = await this.createNextTasks(goal, tasks, results, context);
    const summary = await this.summarize(goal, results, context);
    return {
      goal,
      tasks,
      results,
      nextTasks,
      summary,
    };
  }
}
