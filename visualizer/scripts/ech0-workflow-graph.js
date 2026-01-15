import { randomUUID } from "node:crypto";

export class WorkflowGraph {
  constructor(options = {}) {
    this.nodes = new Map();
    this.edges = new Map();
    this.currentNode = options.entry || null;
    this.context = { ...(options.context || {}) };
    this.history = [];
    this.checkpoints = [];
  }

  addNode(id, metadata = {}) {
    if (!id) {
      return this;
    }
    this.nodes.set(id, { id, ...metadata });
    if (!this.currentNode) {
      this.currentNode = id;
    }
    return this;
  }

  hasNode(id) {
    return this.nodes.has(id);
  }

  addEdge(from, to, condition = null) {
    if (!this.nodes.has(from) || !this.nodes.has(to)) {
      return this;
    }
    const edges = this.edges.get(from) || [];
    edges.push({ to, condition });
    this.edges.set(from, edges);
    return this;
  }

  setEntry(id) {
    if (this.nodes.has(id)) {
      this.currentNode = id;
    }
    return this;
  }

  updateContext(patch = {}) {
    this.context = { ...this.context, ...patch };
    return this.context;
  }

  getState() {
    return {
      node: this.currentNode,
      context: { ...this.context },
    };
  }

  transition(payload = {}) {
    if (!this.currentNode) {
      return { transitioned: false, ...this.getState() };
    }
    const edges = this.edges.get(this.currentNode) || [];
    if (payload.nextNode) {
      const direct = edges.find((edge) => edge.to === payload.nextNode);
      if (direct && this.evaluateEdge(direct, payload)) {
        return this.moveTo(payload.nextNode, payload);
      }
    }
    for (const edge of edges) {
      if (this.evaluateEdge(edge, payload)) {
        return this.moveTo(edge.to, payload);
      }
    }
    return { transitioned: false, ...this.getState() };
  }

  moveTo(nextNode, payload = {}) {
    if (!this.nodes.has(nextNode)) {
      return { transitioned: false, ...this.getState() };
    }
    const previous = this.currentNode;
    this.currentNode = nextNode;
    this.history.push({
      id: randomUUID(),
      from: previous,
      to: nextNode,
      at: new Date().toISOString(),
      payload,
    });
    return { transitioned: true, ...this.getState() };
  }

  checkpoint(label = "checkpoint") {
    const snapshot = {
      id: randomUUID(),
      label,
      node: this.currentNode,
      context: { ...this.context },
      createdAt: new Date().toISOString(),
    };
    this.checkpoints.push(snapshot);
    return snapshot;
  }

  restore(snapshot) {
    if (!snapshot) {
      return this.getState();
    }
    this.currentNode = snapshot.node || this.currentNode;
    this.context = { ...(snapshot.context || {}) };
    return this.getState();
  }

  evaluateEdge(edge, payload) {
    if (!edge.condition) {
      return true;
    }
    return edge.condition({
      context: this.context,
      payload,
      node: this.currentNode,
    });
  }
}
