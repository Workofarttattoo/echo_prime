import { Command, isCommand } from "./command.js";

export const START = "__start__";
export const END = "__end__";

function resolveNextNode(edges, currentNode, override) {
  if (override) {
    return Array.isArray(override) ? override[0] : override;
  }
  const next = edges.get(currentNode) || [];
  if (next.length === 0) {
    return null;
  }
  if (next.length > 1) {
    throw new Error(`Multiple edges found for ${currentNode}. Provide Command.goto.`);
  }
  return next[0];
}

function mergeState(state, update, reducers) {
  if (!update || typeof update !== "object") {
    return state;
  }
  const nextState = { ...state };
  for (const [key, value] of Object.entries(update)) {
    if (reducers[key]) {
      nextState[key] = reducers[key](nextState[key], value);
    } else {
      nextState[key] = value;
    }
  }
  return nextState;
}

export class StateGraph {
  constructor({ reducers = {} } = {}) {
    this.nodes = new Map();
    this.edges = new Map();
    this.entryPoint = null;
    this.finishPoint = null;
    this.reducers = reducers;
  }

  addNode(name, handler) {
    if (!name || typeof handler !== "function") {
      throw new Error("Node requires a name and handler.");
    }
    this.nodes.set(name, handler);
  }

  addEdge(from, to) {
    if (!this.edges.has(from)) {
      this.edges.set(from, []);
    }
    this.edges.get(from).push(to);
  }

  setEntryPoint(name) {
    this.entryPoint = name;
  }

  setFinishPoint(name) {
    this.finishPoint = name;
  }

  compile() {
    if (!this.entryPoint) {
      throw new Error("Entry point not set.");
    }
    if (!this.finishPoint) {
      throw new Error("Finish point not set.");
    }
    return new CompiledGraph({
      nodes: this.nodes,
      edges: this.edges,
      entryPoint: this.entryPoint,
      finishPoint: this.finishPoint,
      reducers: this.reducers,
    });
  }
}

class CompiledGraph {
  constructor({ nodes, edges, entryPoint, finishPoint, reducers }) {
    this.nodes = nodes;
    this.edges = edges;
    this.entryPoint = entryPoint;
    this.finishPoint = finishPoint;
    this.reducers = reducers;
  }

  async invoke(initialState = {}, context = {}) {
    return this._run({
      state: initialState,
      startNode: this.entryPoint,
      context,
    });
  }

  async resume(snapshot, resumeValue, context = {}) {
    if (!snapshot || snapshot.status !== "interrupt") {
      throw new Error("Snapshot required for resume.");
    }
    return this._run({
      state: snapshot.state,
      startNode: snapshot.nextNode,
      context: { ...context, resume: resumeValue, interrupt: snapshot.interrupt },
    });
  }

  async *stream(initialState = {}, context = {}) {
    const iterator = this._runStream({
      state: initialState,
      startNode: this.entryPoint,
      context,
    });
    for await (const item of iterator) {
      yield item;
    }
  }

  async _run({ state, startNode, context }) {
    let currentState = { ...state };
    let currentNode = startNode;
    while (currentNode && currentNode !== END) {
      const executedNode = currentNode;
      const handler = this.nodes.get(executedNode);
      if (!handler) {
        throw new Error(`Node not found: ${executedNode}`);
      }
      const result = await handler(currentState, context);
      let nextNode = null;
      if (isCommand(result)) {
        currentState = mergeState(currentState, result.update, this.reducers);
        if (result.interrupt !== null && result.interrupt !== undefined) {
          nextNode = resolveNextNode(this.edges, executedNode, result.goto);
          return {
            status: "interrupt",
            state: currentState,
            interrupt: result.interrupt,
            nextNode,
          };
        }
        nextNode = resolveNextNode(this.edges, executedNode, result.goto);
      } else {
        currentState = mergeState(currentState, result, this.reducers);
        nextNode = resolveNextNode(this.edges, executedNode, null);
      }
      if (executedNode === this.finishPoint) {
        return {
          status: "complete",
          state: currentState,
          node: executedNode,
        };
      }
      currentNode = nextNode;
    }
    return { status: "complete", state: currentState, node: currentNode };
  }

  async *_runStream({ state, startNode, context }) {
    let currentState = { ...state };
    let currentNode = startNode;
    while (currentNode && currentNode !== END) {
      const executedNode = currentNode;
      const handler = this.nodes.get(executedNode);
      if (!handler) {
        throw new Error(`Node not found: ${executedNode}`);
      }
      yield { type: "node_start", node: executedNode, state: currentState };
      const result = await handler(currentState, context);
      let nextNode = null;
      if (isCommand(result)) {
        currentState = mergeState(currentState, result.update, this.reducers);
        if (result.interrupt !== null && result.interrupt !== undefined) {
          nextNode = resolveNextNode(this.edges, executedNode, result.goto);
          yield {
            type: "interrupt",
            node: executedNode,
            state: currentState,
            interrupt: result.interrupt,
            nextNode,
          };
          return;
        }
        nextNode = resolveNextNode(this.edges, executedNode, result.goto);
      } else {
        currentState = mergeState(currentState, result, this.reducers);
        nextNode = resolveNextNode(this.edges, executedNode, null);
      }
      yield { type: "node_end", node: executedNode, state: currentState };
      if (executedNode === this.finishPoint) {
        return;
      }
      currentNode = nextNode;
    }
  }
}

export { CompiledGraph };
