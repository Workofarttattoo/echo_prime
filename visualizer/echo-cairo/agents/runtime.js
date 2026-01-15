import {
  createAssistantMessage,
  createToolCallMessage,
  createToolResultMessage,
  createHandoffMessage,
} from "./messages.js";

export class AgentRuntime {
  constructor({ model, toolRegistry, maxIterations = 3 } = {}) {
    if (!model || typeof model.call !== "function") {
      throw new Error("AgentRuntime requires a model with call().");
    }
    this.model = model;
    this.toolRegistry = toolRegistry;
    this.maxIterations = maxIterations;
  }

  async run(messages, context = {}) {
    const workingMessages = Array.isArray(messages) ? [...messages] : [];
    const toolSchemas = this.toolRegistry?.exportSchemas() || [];
    let iterations = 0;

    while (iterations < this.maxIterations) {
      iterations += 1;
      const response = await this.model.call({
        messages: workingMessages,
        tools: toolSchemas,
        context,
      });

      if (!response) {
        return {
          status: "error",
          error: "Model returned no response.",
          messages: workingMessages,
        };
      }

      if (response.type === "tool_call") {
        const toolName = response.toolName;
        const args = response.args || {};
        workingMessages.push(createToolCallMessage(toolName, args));
        const toolResult = await this.toolRegistry.execute(
          toolName,
          args,
          context
        );
        workingMessages.push(createToolResultMessage(toolName, toolResult));
        if (toolResult.status === "error") {
          return {
            status: "error",
            error: toolResult.error,
            messages: workingMessages,
          };
        }
        continue;
      }

      if (response.type === "handoff") {
        workingMessages.push(createHandoffMessage(response.target, response.content));
        return {
          status: "handoff",
          target: response.target,
          content: response.content,
          messages: workingMessages,
        };
      }

      const content = response.content ?? response.message ?? "";
      workingMessages.push(createAssistantMessage(content));
      return {
        status: "complete",
        output: content,
        messages: workingMessages,
      };
    }

    return {
      status: "error",
      error: "Max iterations reached without completion.",
      messages: workingMessages,
    };
  }
}
