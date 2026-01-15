#!/usr/bin/env node

/**
 * Conversational CLI for talking with the local ech0 v4 model via Ollama.
 * Mimics a ChatGPT-like loop and keeps multi-turn context in memory.
 */

import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { writeFile, readFile } from "node:fs/promises";

const DEFAULT_MODEL = process.env.ECH0_MODEL?.trim() || "ech0-v4";
const DEFAULT_HOST = process.env.OLLAMA_HOST?.trim() || "http://localhost:11434";
const DEFAULT_TEMP =
  process.env.ECH0_TEMPERATURE !== undefined
    ? Number(process.env.ECH0_TEMPERATURE)
    : 0.7;
const DEFAULT_SYSTEM =
  process.env.ECH0_SYSTEM_PROMPT?.trim() ||
  "You are ech0 v4, a grounded local AI. Be concise, truthful, and reference Josh's instructions when relevant.";

function printHelp() {
  console.log("ech0 chat CLI");
  console.log("");
  console.log("Usage: node scripts/ech0-chat-cli.js [options]");
  console.log("");
  console.log("Options:");
  console.log("  -m, --model <name>        Ollama model to chat with (default ech0-v4)");
  console.log("  -H, --host <url>          Base Ollama host URL (default http://localhost:11434)");
  console.log("  -t, --temp <value>        Sampling temperature (default 0.7)");
  console.log("  -s, --system <prompt>     Override the system prompt");
  console.log("  --no-system               Start without a system prompt");
  console.log("  -h, --help                Show this help");
  console.log("");
  console.log("Interactive commands:");
  console.log("  /help                     Show runtime commands");
  console.log("  /system [text]            Show or set the system prompt");
  console.log("  /reset                    Clear conversation (keeps system prompt)");
  console.log("  /save <file>              Persist conversation JSON");
  console.log("  /load <file>              Load conversation JSON");
  console.log("  /exit                     Quit the CLI");
}

function parseArgs(rawArgs) {
  const options = {
    model: DEFAULT_MODEL,
    host: DEFAULT_HOST,
    temperature: Number.isNaN(DEFAULT_TEMP) ? 0.7 : DEFAULT_TEMP,
    systemPrompt: DEFAULT_SYSTEM,
    help: false,
  };

  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    switch (arg) {
      case "-m":
      case "--model":
        options.model = rawArgs[i + 1] || options.model;
        i += 1;
        break;
      case "-H":
      case "--host":
        options.host = (rawArgs[i + 1] || options.host).replace(/\/+$/, "");
        i += 1;
        break;
      case "-t":
      case "--temp":
        options.temperature = Number(rawArgs[i + 1]) || options.temperature;
        i += 1;
        break;
      case "-s":
      case "--system":
        options.systemPrompt = rawArgs[i + 1] || options.systemPrompt;
        i += 1;
        break;
      case "--no-system":
        options.systemPrompt = "";
        break;
      case "-h":
      case "--help":
        options.help = true;
        break;
      default:
        break;
    }
  }

  return options;
}

async function ensureOllama(host, model) {
  try {
    const response = await fetch(`${host}/api/tags`);
    if (!response.ok) {
      throw new Error(`ollama responded with ${response.status}`);
    }
    const payload = await response.json();
    const modelFound = payload?.models?.some((entry) => entry.name === model);
    if (!modelFound) {
      console.log(
        `[warn] Model ${model} not found via ${host}. Run 'ollama pull ${model}' if needed.`
      );
    }
  } catch (error) {
    throw new Error(
      `Cannot reach ollama at ${host}. Start it with 'ollama serve'. (${error.message})`
    );
  }
}

async function callEch0(host, model, temperature, messages) {
  const response = await fetch(`${host}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: {
        temperature,
      },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`ollama error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  return data?.message?.content?.trim() || "";
}

function conversationTemplate(systemPrompt) {
  if (!systemPrompt) {
    return [];
  }
  return [{ role: "system", content: systemPrompt }];
}

async function handleCommand(command, state) {
  const [keyword, ...rest] = command.slice(1).trim().split(/\s+/);
  const argument = command.slice(keyword.length + 2).trim();

  switch (keyword) {
    case "help":
      console.log("Slash commands:");
      console.log("  /help               Show this menu");
      console.log("  /system [text]      Show or update the system prompt");
      console.log("  /reset              Clear the conversation");
      console.log("  /save <file>        Save conversation to disk");
      console.log("  /load <file>        Load conversation from disk");
      console.log("  /exit               Exit the CLI");
      return { exit: false };
    case "system":
      if (!argument) {
        console.log(`[info] Current system prompt: ${state.systemPrompt || "(none)"}`);
        return { exit: false };
      }
      state.systemPrompt = argument;
      state.messages = conversationTemplate(state.systemPrompt);
      console.log("[info] Updated system prompt and reset conversation.");
      return { exit: false };
    case "reset":
      state.messages = conversationTemplate(state.systemPrompt);
      console.log("[info] Conversation cleared.");
      return { exit: false };
    case "save":
      if (!argument) {
        console.log("[warn] Please provide a file path. Example: /save logs/ech0-chat.json");
        return { exit: false };
      }
      await writeFile(
        argument,
        JSON.stringify(
          {
            model: state.model,
            host: state.host,
            systemPrompt: state.systemPrompt,
            messages: state.messages,
            savedAt: new Date().toISOString(),
          },
          null,
          2
        )
      );
      console.log(`[info] Conversation saved to ${argument}`);
      return { exit: false };
    case "load":
      if (!argument) {
        console.log("[warn] Provide a path to a previously saved conversation JSON.");
        return { exit: false };
      }
      try {
        const fileContents = await readFile(argument, "utf8");
        const payload = JSON.parse(fileContents);
        if (!Array.isArray(payload.messages)) {
          throw new Error("Missing messages array");
        }
        state.systemPrompt = payload.systemPrompt || state.systemPrompt;
        state.messages = payload.messages;
        console.log(`[info] Loaded ${payload.messages.length} messages from ${argument}`);
      } catch (error) {
        console.log(`[error] Could not load conversation: ${error.message}`);
      }
      return { exit: false };
    case "exit":
      return { exit: true };
    default:
      console.log(`[warn] Unknown command: /${keyword}`);
      return { exit: false };
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  if (options.help) {
    printHelp();
    process.exit(0);
  }

  const state = {
    model: options.model,
    host: options.host,
    temperature: options.temperature,
    systemPrompt: options.systemPrompt,
    messages: conversationTemplate(options.systemPrompt),
  };

  console.log(`[info] Starting ech0 chat CLI → model=${state.model} host=${state.host}`);
  try {
    await ensureOllama(state.host, state.model);
    console.log("[info] Connected to ollama.");
  } catch (error) {
    console.log(`[error] ${error.message}`);
    process.exit(1);
  }

  const rl = createInterface({ input, output });
  let active = true;

  const exitGracefully = () => {
    if (!active) {
      return;
    }
    active = false;
    console.log("\n[info] Exiting ech0 chat CLI.");
    rl.close();
    process.exit(0);
  };

  process.on("SIGINT", exitGracefully);

  while (active) {
    const line = await rl.question("you> ");
    const trimmed = line.trim();

    if (!trimmed) {
      continue;
    }

    if (trimmed.startsWith("/")) {
      const { exit } = await handleCommand(trimmed, state);
      if (exit) {
        exitGracefully();
      }
      continue;
    }

    const userMessage = { role: "user", content: trimmed };
    state.messages.push(userMessage);
    console.log("[info] ech0 is thinking...");
    try {
      const reply = await callEch0(
        state.host,
        state.model,
        state.temperature,
        state.messages
      );
      state.messages.push({ role: "assistant", content: reply });
      console.log(`ech0> ${reply}`);
    } catch (error) {
      state.messages.pop();
      console.log(`[error] ${error.message}`);
    }
  }
}

main().catch((error) => {
  console.log(`[error] Unexpected failure: ${error.message}`);
  process.exit(1);
});
