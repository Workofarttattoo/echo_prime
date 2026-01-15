import { createToolResult, ToolResultStatus } from "./result.js";
import { createToolSchema, validateArgs } from "./schema.js";

export class BaseTool {
  constructor({ name, description, parameters, strict = false, handler }) {
    this.schema = createToolSchema({
      name,
      description,
      parameters,
      strict,
    });
    this.name = this.schema.name;
    this.description = this.schema.description;
    this.handler = handler;
  }

  async run(args, context) {
    if (typeof this.handler === "function") {
      return this.handler(args, context);
    }
    throw new Error(`Tool ${this.name} does not implement run().`);
  }

  async execute(args, context) {
    const errors = validateArgs(this.schema, args);
    if (errors.length) {
      return createToolResult({
        status: ToolResultStatus.error,
        error: errors.join(" | "),
      });
    }
    try {
      const output = await this.run(args, context);
      return createToolResult({
        status: ToolResultStatus.success,
        output,
      });
    } catch (error) {
      return createToolResult({
        status: ToolResultStatus.error,
        error: error?.message || String(error),
      });
    }
  }
}
