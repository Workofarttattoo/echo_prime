import { promises as fs } from "node:fs";
import { extname } from "node:path";

function stripComments(line) {
  const hashIndex = line.indexOf("#");
  if (hashIndex === -1) {
    return line;
  }
  return line.slice(0, hashIndex);
}

function parseScalar(rawValue) {
  const trimmed = rawValue.trim();
  if (trimmed === "") {
    return "";
  }
  if (
    (trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  if (trimmed === "true") {
    return true;
  }
  if (trimmed === "false") {
    return false;
  }
  if (!Number.isNaN(Number(trimmed))) {
    return Number(trimmed);
  }
  return trimmed;
}

function nextNonEmptyLine(lines, startIndex) {
  for (let index = startIndex + 1; index < lines.length; index += 1) {
    const cleaned = stripComments(lines[index]).trim();
    if (cleaned) {
      return { line: cleaned, indent: lines[index].match(/^\s*/)[0].length };
    }
  }
  return null;
}

function parseSimpleYaml(text) {
  const lines = text.split(/\r?\n/);
  const root = {};
  const stack = [{ indent: -1, container: root }];

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index];
    const stripped = stripComments(rawLine);
    if (!stripped.trim()) {
      continue;
    }
    const indent = rawLine.match(/^\s*/)[0].length;
    let trimmed = stripped.trim();

    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) {
      stack.pop();
    }

    const current = stack[stack.length - 1].container;

    if (trimmed.startsWith("- ")) {
      const itemText = trimmed.slice(2).trim();
      if (!Array.isArray(current)) {
        throw new Error("YAML format error: list item without list container.");
      }
      if (!itemText) {
        const newItem = {};
        current.push(newItem);
        stack.push({ indent, container: newItem });
        continue;
      }
      const separatorIndex = itemText.indexOf(":");
      if (separatorIndex !== -1) {
        const key = itemText.slice(0, separatorIndex).trim();
        const valueText = itemText.slice(separatorIndex + 1).trim();
        const newItem = { [key]: parseScalar(valueText) };
        current.push(newItem);
        const nextLine = nextNonEmptyLine(lines, index);
        if (nextLine && nextLine.indent > indent) {
          stack.push({ indent, container: newItem });
        }
      } else {
        current.push(parseScalar(itemText));
      }
      continue;
    }

    const separatorIndex = trimmed.indexOf(":");
    if (separatorIndex === -1) {
      continue;
    }
    const key = trimmed.slice(0, separatorIndex).trim();
    const valueText = trimmed.slice(separatorIndex + 1).trim();
    if (valueText) {
      current[key] = parseScalar(valueText);
      continue;
    }
    const nextLine = nextNonEmptyLine(lines, index);
    const shouldBeArray = Boolean(nextLine && nextLine.line.startsWith("- "));
    const container = shouldBeArray ? [] : {};
    current[key] = container;
    stack.push({ indent, container });
  }

  return root;
}

export function parseRoleTemplate(content, sourceName = "") {
  const trimmed = content.trim();
  if (!trimmed) {
    return {};
  }
  if (sourceName.endsWith(".json") || trimmed.startsWith("{") || trimmed.startsWith("[")) {
    return JSON.parse(trimmed);
  }
  return parseSimpleYaml(trimmed);
}

export async function loadRoleTemplate(filePath) {
  const raw = await fs.readFile(filePath, "utf-8");
  return parseRoleTemplate(raw, extname(filePath));
}

export function validateRoleTemplate(template) {
  const errors = [];
  if (!template || typeof template !== "object") {
    errors.push("Template must be an object.");
    return errors;
  }
  const roles = Array.isArray(template.roles) ? template.roles : [];
  if (!roles.length) {
    errors.push("Template requires at least one role.");
  }
  roles.forEach((role, index) => {
    if (!role?.name) {
      errors.push(`Role ${index + 1} missing name.`);
    }
    if (!role?.goal) {
      errors.push(`Role ${index + 1} missing goal.`);
    }
  });
  return errors;
}

export function createMissionProfile(template) {
  const roles = Array.isArray(template.roles) ? template.roles : [];
  return {
    mission: template.mission || { name: "Untitled mission" },
    roles,
    metadata: template.metadata || {},
  };
}
