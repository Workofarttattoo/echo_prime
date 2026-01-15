import { ToolResultStatus, createToolResult, isToolResult } from "./result.js";

export class ToolRegistry {
  constructor() {
    this.tools = new Map();
    this.allowlist = null;
    this.denylist = new Set();
  }

  register(tool) {
    if (!tool || !tool.name) {
      throw new Error("Tool must include a name.");
    }
    this.tools.set(tool.name, tool);
  }

  setAllowlist(names = []) {
    this.allowlist = new Set(names);
  }

  setDenylist(names = []) {
    this.denylist = new Set(names);
  }

  isAllowed(name) {
    if (this.allowlist && !this.allowlist.has(name)) {
      return false;
    }
    if (this.denylist.has(name)) {
      return false;
    }
    return true;
  }

  get(name) {
    return this.tools.get(name) || null;
  }

  list() {
    return Array.from(this.tools.values());
  }

  exportSchemas() {
    return this.list().map((tool) => tool.schema);
  }

  async execute(name, args, context) {
    if (!this.isAllowed(name)) {
      return createToolResult({
        status: ToolResultStatus.error,
        error: `Tool not allowed: ${name}`,
      });
    }
    const tool = this.get(name);
    if (!tool) {
      return createToolResult({
        status: ToolResultStatus.error,
        error: `Tool not found: ${name}`,
      });
    }
    const result = await tool.execute(args, context);
    if (isToolResult(result)) {
      return result;
    }
    return createToolResult({
      status: ToolResultStatus.success,
      output: result,
    });
  }
}
