import { spawn } from "node:child_process";

const runProcess = (command, args, timeoutMs) =>
  new Promise((resolve) => {
    const startedAt = Date.now();
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let finished = false;

    const finish = (result) => {
      if (finished) {
        return;
      }
      finished = true;
      resolve({
        ...result,
        durationMs: Date.now() - startedAt,
      });
    };

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish({ ok: false, exitCode: null, stdout, stderr, timedOut: true });
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      clearTimeout(timer);
      finish({ ok: false, exitCode: null, stdout, stderr: error.message });
    });

    child.on("close", (exitCode) => {
      clearTimeout(timer);
      finish({ ok: exitCode === 0, exitCode, stdout, stderr, timedOut: false });
    });
  });

export class SandboxRunner {
  constructor(options = {}) {
    this.enabled = options.enabled ?? false;
    this.image = options.image || "node:20-alpine";
    this.timeoutMs = options.timeoutMs || 30000;
    this.cpuLimit = options.cpuLimit || "1";
    this.memoryLimit = options.memoryLimit || "512m";
    this.networkMode = options.networkMode || "none";
    this.workdir = options.workdir || "/workspace";
    this.dryRun = options.dryRun ?? true;
  }

  buildInvocation(command) {
    const commandText = Array.isArray(command)
      ? command.join(" ")
      : String(command || "");
    const args = [
      "run",
      "--rm",
      "-i",
      "--network",
      this.networkMode,
      "--cpus",
      this.cpuLimit,
      "--memory",
      this.memoryLimit,
      "-w",
      this.workdir,
      this.image,
      "/bin/sh",
      "-lc",
      commandText,
    ];
    return { command: "docker", args };
  }

  async run(command, options = {}) {
    if (!this.enabled) {
      return {
        ok: false,
        error: "Sandbox disabled",
        command: String(command || ""),
      };
    }
    const invocation = this.buildInvocation(command);
    const dryRun = options.dryRun ?? this.dryRun;
    if (dryRun) {
      return {
        ok: true,
        dryRun: true,
        command: `${invocation.command} ${invocation.args.join(" ")}`,
      };
    }
    const result = await runProcess(
      invocation.command,
      invocation.args,
      options.timeoutMs || this.timeoutMs
    );
    return { ...result, command: invocation.command, args: invocation.args };
  }
}
