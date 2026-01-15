import { randomUUID } from "node:crypto";

export class TaskQueue {
  constructor(options = {}) {
    this.maxSize = options.maxSize || 50;
    this.items = [];
    this.completed = [];
  }

  add(task, options = {}) {
    if (this.items.length >= this.maxSize) {
      return null;
    }
    const entry = {
      id: options.id || randomUUID(),
      description: task?.description || task?.summary || String(task || ""),
      role: options.role || task?.role || null,
      priority: Number.isFinite(options.priority)
        ? options.priority
        : Number.isFinite(task?.priority)
        ? task.priority
        : 0,
      payload: { ...(task?.payload || {}) },
      command: task?.command || null,
      status: "pending",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    this.items.push(entry);
    return entry;
  }

  size() {
    return this.items.length;
  }

  hasPending() {
    return this.items.length > 0;
  }

  list() {
    return this.items.map((item) => ({ ...item }));
  }

  peek() {
    const sorted = this.sortedItems();
    return sorted[0] ? { ...sorted[0] } : null;
  }

  next() {
    if (!this.items.length) {
      return null;
    }
    const sorted = this.sortedItems();
    const nextItem = sorted[0];
    const index = this.items.findIndex((item) => item.id === nextItem.id);
    if (index >= 0) {
      const [removed] = this.items.splice(index, 1);
      removed.status = "in-progress";
      removed.updatedAt = new Date().toISOString();
      return removed;
    }
    return null;
  }

  reprioritize(id, priority) {
    const item = this.items.find((entry) => entry.id === id);
    if (!item) {
      return false;
    }
    item.priority = priority;
    item.updatedAt = new Date().toISOString();
    return true;
  }

  markComplete(id, result = {}) {
    const entryIndex = this.completed.findIndex((item) => item.id === id);
    const completedAt = new Date().toISOString();
    if (entryIndex >= 0) {
      this.completed[entryIndex] = {
        ...this.completed[entryIndex],
        status: "completed",
        result,
        completedAt,
      };
      return this.completed[entryIndex];
    }
    const existing = this.items.find((item) => item.id === id);
    const completed = {
      ...(existing || { id }),
      status: "completed",
      result,
      completedAt,
      updatedAt: completedAt,
    };
    if (existing) {
      this.items = this.items.filter((item) => item.id !== id);
    }
    this.completed.push(completed);
    return completed;
  }

  sortedItems() {
    return [...this.items].sort((a, b) => {
      if (a.priority === b.priority) {
        return a.createdAt.localeCompare(b.createdAt);
      }
      return b.priority - a.priority;
    });
  }
}
