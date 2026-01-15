const defaultRoles = [
  {
    id: "researcher",
    name: "Researcher",
    mission: "Clarify objectives, gather evidence, and summarize findings.",
    steps: [
      "Confirm the task scope and desired outcome.",
      "Collect primary sources and verify provenance.",
      "Summarize insights with confidence levels.",
      "List open questions and recommended follow-ups.",
    ],
    output: "Evidence-backed brief with gaps and next steps.",
  },
  {
    id: "engineer",
    name: "Engineer",
    mission: "Design and implement safe, testable solutions.",
    steps: [
      "Break the task into actionable implementation steps.",
      "Identify dependencies, risks, and rollback paths.",
      "Implement with safety checks and minimal scope.",
      "Summarize validation steps and remaining risks.",
    ],
    output: "Implementation plan with verification checklist.",
  },
  {
    id: "reviewer",
    name: "Reviewer",
    mission: "Assess quality, safety, and alignment with goals.",
    steps: [
      "Review outputs against acceptance criteria.",
      "Spot gaps, inconsistencies, or unsafe assumptions.",
      "Recommend adjustments or approvals.",
      "Confirm readiness for handoff or release.",
    ],
    output: "Risk-focused review summary with approvals.",
  },
];

const formatValue = (value) => {
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
};

const formatContext = (context) => {
  if (!context || typeof context !== "object") {
    return [];
  }
  return Object.entries(context)
    .map(([key, value]) => {
      const formatted = formatValue(value);
      if (!formatted) {
        return null;
      }
      return `${key}: ${formatted}`;
    })
    .filter(Boolean);
};

export class SopEngine {
  constructor(options = {}) {
    this.roles = new Map();
    this.defaultRole = options.defaultRole || "researcher";
    const roles = options.roles || defaultRoles;
    roles.forEach((role) => this.addRole(role));
  }

  addRole(role) {
    if (!role || !role.id) {
      return false;
    }
    this.roles.set(role.id, { ...role });
    return true;
  }

  getRole(roleId) {
    return this.roles.get(roleId) || this.roles.get(this.defaultRole) || null;
  }

  listRoles() {
    return Array.from(this.roles.keys());
  }

  buildPrompt(roleId, task, context = {}) {
    const role = this.getRole(roleId);
    if (!role) {
      return task ? `Task: ${task}` : "";
    }
    const sections = [
      `Role: ${role.name}`,
      `Mission: ${role.mission}`,
      "SOP:",
      ...role.steps.map((step, index) => `${index + 1}. ${step}`),
    ];
    if (task) {
      sections.push(`Task: ${task}`);
    }
    const contextLines = formatContext(context);
    if (contextLines.length) {
      sections.push("Context:");
      sections.push(...contextLines);
    }
    if (role.output) {
      sections.push(`Output: ${role.output}`);
    }
    return sections.join("\n");
  }
}
