import {
  AgentRuntime,
  BaseTool,
  Command,
  StateGraph,
  ToolRegistry,
  createUserMessage,
} from "../echo-cairo/index.js";

const tools = new ToolRegistry();

tools.register(
  new BaseTool({
    name: "add",
    description: "Add two numbers.",
    parameters: {
      type: "object",
      properties: {
        a: { type: "number" },
        b: { type: "number" },
      },
      required: ["a", "b"],
      additionalProperties: false,
    },
    handler: ({ a, b }) => a + b,
  })
);

tools.register(
  new BaseTool({
    name: "uppercase",
    description: "Uppercase a string.",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string" },
      },
      required: ["text"],
      additionalProperties: false,
    },
    handler: ({ text }) => text.toUpperCase(),
  })
);

const scriptedModel = {
  async call({ messages }) {
    const lastToolResult = [...messages]
      .reverse()
      .find((message) => message.type === "tool_result");
    if (!lastToolResult) {
      return {
        type: "tool_call",
        toolName: "add",
        args: { a: 2, b: 3 },
      };
    }
    if (lastToolResult.toolName === "add") {
      return {
        type: "tool_call",
        toolName: "uppercase",
        args: { text: `sum:${lastToolResult.result.output}` },
      };
    }
    return {
      type: "message",
      content: `Done: ${lastToolResult.result.output}`,
    };
  },
};

const agent = new AgentRuntime({
  model: scriptedModel,
  toolRegistry: tools,
  maxIterations: 4,
});

const graph = new StateGraph({
  reducers: {
    events: (prev = [], next) => [
      ...prev,
      ...(Array.isArray(next) ? next : [next]),
    ],
  },
});

graph.addNode("start", () =>
  Command.interrupt(
    { prompt: "Confirm agent run." },
    { update: { events: ["awaiting-confirm"] }, goto: "agent" }
  )
);

graph.addNode("agent", async (state, context) => {
  const result = await agent.run([createUserMessage("add 2 and 3")], context);
  return {
    events: ["agent-complete"],
    confirmed: Boolean(context.resume?.approved),
    output: result.output,
  };
});

graph.setEntryPoint("start");
graph.setFinishPoint("agent");

const compiled = graph.compile();
const firstRun = await compiled.invoke({ events: [] });

if (firstRun.status === "interrupt") {
  const resumed = await compiled.resume(firstRun, { approved: true });
  console.log("Final state:", JSON.stringify(resumed.state, null, 2));
} else {
  console.log("Final state:", JSON.stringify(firstRun.state, null, 2));
}
