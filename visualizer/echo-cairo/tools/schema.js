const typeChecks = {
  string: (value) => typeof value === "string",
  number: (value) => typeof value === "number" && Number.isFinite(value),
  integer: (value) => Number.isInteger(value),
  boolean: (value) => typeof value === "boolean",
  object: (value) => value !== null && typeof value === "object" && !Array.isArray(value),
  array: (value) => Array.isArray(value),
};

export function createToolSchema({
  name,
  description = "",
  parameters = {},
  strict = false,
} = {}) {
  if (!name) {
    throw new Error("Tool schema requires a name.");
  }
  const normalized = normalizeParameters(parameters);
  if (strict && normalized.additionalProperties !== false) {
    normalized.additionalProperties = false;
  }
  return {
    name,
    description,
    parameters: normalized,
    strict,
  };
}

export function normalizeParameters(parameters = {}) {
  const normalized = {
    type: parameters.type || "object",
    properties: parameters.properties || {},
    required: Array.isArray(parameters.required) ? parameters.required : [],
    additionalProperties:
      parameters.additionalProperties === undefined
        ? true
        : parameters.additionalProperties,
  };
  return normalized;
}

export function validateArgs(schema, args) {
  const errors = [];
  const params = schema?.parameters || normalizeParameters({});
  const required = params.required || [];
  const properties = params.properties || {};

  for (const key of required) {
    if (args?.[key] === undefined) {
      errors.push(`Missing required argument: ${key}`);
    }
  }

  if (!args || typeof args !== "object") {
    if (required.length) {
      errors.push("Arguments must be an object.");
    }
    return errors;
  }

  for (const [key, value] of Object.entries(args)) {
    const definition = properties[key];
    if (!definition) {
      if (params.additionalProperties === false || schema.strict) {
        errors.push(`Unexpected argument: ${key}`);
      }
      continue;
    }
    if (definition.enum && !definition.enum.includes(value)) {
      errors.push(`Invalid value for ${key}: ${value}`);
    }
    if (definition.type) {
      const checker = typeChecks[definition.type];
      if (checker && !checker(value)) {
        errors.push(`Invalid type for ${key}: expected ${definition.type}`);
      }
    }
  }

  return errors;
}
