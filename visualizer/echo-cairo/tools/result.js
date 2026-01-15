export const ToolResultStatus = {
  success: "success",
  error: "error",
};

export function createToolResult({
  status = ToolResultStatus.success,
  output = null,
  error = null,
  meta = {},
} = {}) {
  return {
    status,
    output,
    error,
    meta,
  };
}

export function isToolResult(value) {
  return Boolean(value && typeof value === "object" && value.status);
}
