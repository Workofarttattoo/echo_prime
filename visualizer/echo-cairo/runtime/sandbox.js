import { spawn } from "node:child_process";
import { createToolResult, ToolResultStatus } from "../tools/result.js";

const DEFAULT_ALLOWLIST = ["ls", "pwd", "cat", "rg", "echo"];
const WRITE_COMMANDS = new Set([
  "rm",
  "mv",
  "cp",
  "touch",
  "mkdir",
  "chmod",
  "chown",
  "tee",
  "dd",
  "sudo",
  "git",
  "npm",
  "node",
  "python",
  "pip",
]);

function hasShellMeta(args) {
  return args.some((arg) => /[|&;><]/.test(arg));
}

export class SandboxRuntime {
  constructor({ allowlist = DEFAULT_ALLOWLIST } = {}) {
    this.allowlist = new Set(allowlist);
  }

  isAllowed(command) {
    return this.allowlist.has(command);
  }

  isWriteCommand(command, args) {
    if (WRITE_COMMANDS.has(command)) {
      return true;
    }
    return hasShellMeta(args);
  }

  async execute({ command, args = [], cwd = process.cwd(), allowWrite = false } = {}) {
    if (!Array.isArray(command) && typeof command !== "string") {
      throw new Error("SandboxRuntime requires a command string or array.");
    }
    const parts = Array.isArray(command) ? command : [command, ...args];
    const [cmd, ...rest] = parts;
    if (!cmd) {
      return createToolResult({
        status: ToolResultStatus.error,
        error: "Missing command.",
      });
    }
    if (!this.isAllowed(cmd)) {
      return createToolResult({
        status: ToolResultStatus.error,
        error: `Command not allowlisted: ${cmd}`,
      });
    }
    if (!allowWrite && this.isWriteCommand(cmd, rest)) {
      return createToolResult({
        status: ToolResultStatus.error,
        error: `Write command blocked in read-only mode: ${cmd}`,
      });
    }

    return new Promise((resolve) => {
      const child = spawn(cmd, rest, { cwd, shell: false });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.on("close", (code) => {
        const status = code === 0 ? ToolResultStatus.success : ToolResultStatus.error;
        resolve(
          createToolResult({
            status,
            output: stdout.trim(),
            error: stderr.trim() || (code === 0 ? null : `Exit code ${code}`),
            meta: { exitCode: code },
          })
        );
      });
      child.on("error", (error) => {
        resolve(
          createToolResult({
            status: ToolResultStatus.error,
            error: error.message,
          })
        );
      });
    });
  }
}
