export const MessageRole = {
  system: "system",
  user: "user",
  assistant: "assistant",
  tool: "tool",
};

export function createMessage(role, content, meta = {}) {
  return {
    role,
    content,
    ...meta,
  };
}

export function createSystemMessage(content, meta = {}) {
  return createMessage(MessageRole.system, content, meta);
}

export function createUserMessage(content, meta = {}) {
  return createMessage(MessageRole.user, content, meta);
}

export function createAssistantMessage(content, meta = {}) {
  return createMessage(MessageRole.assistant, content, meta);
}

export function createToolCallMessage(toolName, args) {
  return createMessage(MessageRole.assistant, null, {
    type: "tool_call",
    toolName,
    args,
  });
}

export function createToolResultMessage(toolName, result) {
  return createMessage(MessageRole.tool, null, {
    type: "tool_result",
    toolName,
    result,
  });
}

export function createHandoffMessage(target, content) {
  return createMessage(MessageRole.assistant, content, {
    type: "handoff",
    target,
  });
}
