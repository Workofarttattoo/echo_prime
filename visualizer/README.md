# Advanced 3D Visualization System

This module delivers a self-contained experimentation environment that couples automated material discovery workflows with a real-time 3D visualizer. It exposes simulated crystal structure predictions, multi-scale modeling analytics, and experiment optimization feedback through a browser-based scene powered by Three.js.

## Capabilities

- Automated experiment workflows orchestrate multi-stage pipelines (ingest → structure prediction → simulation → optimization → validation) with live status monitoring.
- Crystal structure prediction engine synthesizes lattice geometries, atom-level data, and bond graphs from configurable material seeds under varying thermodynamic conditions.
- Material discovery AI agent balances exploration and exploitation to propose high-potential candidates, logs performance scores, and adapts to optimization feedback.
- Multi-scale modeling framework aggregates atomic, mesoscopic, and continuum metrics and produces coupled stability profiles for the visualizer.
- Experiment optimization engine applies iterative heuristics to refine processing parameters (temperature, pressure, dopant ratio, annealing time) using recent outcomes.

## Getting Started

```bash
cd visualizer
npm install
npm run start
```

Then open `http://localhost:4100` (or the port you configure) to inspect the visualization dashboard.

### Docker smoke stream

The previous Docker telemetry smoke test now runs alongside the visualization system. Disable it with `ENABLE_DOCKER_SMOKE=off` when Docker is unavailable.

## ech0 Chat CLI

Talk to the local `ech0 v4` Ollama model with a conversational shell:

```bash
npm run ech0:chat
```

Runtime commands:

- `/system <prompt>` – show or update the system prompt (conversation resets when updated).
- `/reset` – wipe the current transcript while keeping the system instructions.
- `/save <file>` and `/load <file>` – persist or restore chat history JSON.
- `/exit` – leave the CLI. You can also use `Ctrl+C`.

Environment variables `ECH0_MODEL`, `ECH0_TEMPERATURE`, and `OLLAMA_HOST` override the defaults. Ensure `ollama serve` is running locally and that `ollama pull ech0-v4` has been executed before launching the CLI.

## ech0 Hub Monitor

The dashboard now includes an **ech0 Hub** card that pings the Ollama daemon, lists every available model, and highlights the most recent `ech0-*` build. Use the dropdown to inspect model metadata (last updated timestamp, digest, and binary size) and click **Refresh** to force a new lookup.

If Ollama cannot reach the configured hub you will see guidance in the card, mirroring the CLI warning (`ollama cannot reach ech0hub`) so operators know to restart `ollama serve` or adjust the host.

### Autonomous Mode Control

Click **Activate Autonomous Mode** in the ech0 Hub card to push an `"enable"` command to the configured ech0 hub (proxied through `/api/ech0/autonomy`). The button toggles between activation/deactivation, shows the last response timestamp, and reports transport errors inline so you know when a crash occurred during training—restart `ech0_hub_enhanced.py` or point `ECH0_HUB_HOST` at a healthy instance before re-engaging autonomy.

### Hub environment variables

| Variable | Purpose |
| --- | --- |
| `ECH0_HUB_HOST` | Base URL for the Ollama daemon (falls back to `OLLAMA_HOST` or `http://localhost:11434`). |
| `ECH0_HUB_NAMESPACE` | Optional prefix for highlighting `ech0` models in the UI (default `ech0`). |
| `ECH0_HUB_TIMEOUT_MS` | How long to wait for the Ollama `/api/tags` response before reporting an error (default `15000`). |
| `ECH0_HUB_CACHE_MS` | Cache duration for the server-side model list to avoid hammering Ollama (default `30000`). |

## ech0 Prime Orchestrator

The ech0 Prime scaffold layers SOP-driven prompts, workflow state, crew orchestration, task queues, sandboxed execution, and telemetry into a single entry point. It is disabled by default so you can stage it safely.

```bash
ECH0_PRIME_ENABLED=on npm run ech0:prime -- --task "Outline the SOP upgrades" --role researcher
```

Load tasks from a JSON array with `--file` and include optional fields like `role`, `priority`, `command`, or `payload` to drive sandbox runs and workflow context.

### ech0 Prime environment variables

| Variable | Purpose |
| --- | --- |
| `ECH0_PRIME_ENABLED` | Enable the orchestrator (`on` to run). |
| `ECH0_PRIME_AUTONOMY_LEVEL` | Autonomy setting for downstream policy checks (default `3`). |
| `ECH0_PRIME_MAX_PARALLEL` | Planned parallel task limit (default `1`). |
| `ECH0_PRIME_QUEUE_MAX` | Queue capacity before rejecting tasks (default `50`). |
| `ECH0_PRIME_TELEMETRY` | Toggle telemetry capture (`on` by default). |
| `ECH0_PRIME_CHECKPOINT_DIR` | Directory for workflow checkpoints (default `logs/ech0-prime-checkpoints`). |
| `ECH0_PRIME_SANDBOX_ENABLED` | Toggle Docker sandbox runs (`off` by default). |
| `ECH0_PRIME_SANDBOX_IMAGE` | Docker image for sandbox execution (default `node:20-alpine`). |
| `ECH0_PRIME_SANDBOX_TIMEOUT_MS` | Sandbox timeout in milliseconds (default `30000`). |
| `ECH0_PRIME_SANDBOX_CPU` | CPU quota for the sandbox container (default `1`). |
| `ECH0_PRIME_SANDBOX_MEMORY` | Memory quota for the sandbox container (default `512m`). |
| `ECH0_PRIME_SANDBOX_NETWORK` | Docker network mode (default `none`). |
| `ECH0_PRIME_SANDBOX_WORKDIR` | Working directory inside the container (default `/workspace`). |
| `ECH0_PRIME_SANDBOX_DRY_RUN` | Emit Docker commands without executing them (default `on`). |

## ech0 Golden Run

Run the deterministic golden run to demonstrate the ech0 → Qulab → ech0 handoff with reproducible artifacts. The output is simulated for demo purposes and is not a real-world lab result.

```bash
npm run ech0:golden-run -- --seed 1337
```

Verify the latest run against the acceptance criteria:

```bash
npm run ech0:golden-verify
```

Generate an investor-friendly summary:

```bash
npm run ech0:golden-report
```

Generate a one-page HTML brief for PDF export:

```bash
npm run ech0:golden-onepager
```

Artifacts:

- `logs/qulab-golden-runs/` – deterministic Qulab simulation output with hashes.
- `logs/ech0-golden-runs/` – end-to-end report tying ech0 prompts to Qulab output.
- `docs/golden-run.md` – acceptance criteria and verification checklist.

## Configuration

All tunables accept environment overrides and default to the values in `config/default.js`.

| Variable | Purpose |
| --- | --- |
| `PORT` | HTTP port for the visualization server (default `4100`). |
| `TICK_INTERVAL_MS` | Update cadence for experiment ticks. |
| `MAX_CONCURRENT_WORKFLOWS` | Concurrent automated workflows allowed. |
| `DISCOVERY_EXPLORATION_BIAS`, `DISCOVERY_EXPLOIT_BIAS`, `DISCOVERY_NOVELTY_THRESHOLD` | Control the discovery agent's exploration/exploitation balance. |
| `PREDICTOR_LATTICE_SPACING`, `PREDICTOR_EXPANSION_COEFF` | Adjust lattice spacing and thermal expansion sensitivity. |
| `MODELING_COUPLING_STRENGTH` | Scales multi-scale coupling intensity. |
| `OPT_MAX_ITERATIONS`, `OPT_LEARNING_RATE`, `OPT_EXPLORATION_WEIGHT`, `OPT_EXPLOITATION_WEIGHT` | Shape the experiment optimization search strategy. |
| `ENABLE_DOCKER_SMOKE` | Toggle the Docker log/stat stream (set to `off` to skip). |

## Project Layout

- `scripts/start.js` – Boots the SSE-enabled HTTP server, streams orchestrator updates, serves `public/`, and optionally runs the Docker smoke test.
- `scripts/docker-smoke.js` – Isolated Docker telemetry smoke workflow reused by the main entry point.
- `config/default.js` – Central configuration defaults with environment overrides.
- `src/index.js` – Visualization orchestrator coordinating discovery, prediction, modeling, optimization, and workflow management.
- `src/ai/materialDiscoveryAgent.js` – Exploration/exploitation agent for material proposals.
- `src/prediction/crystalStructurePredictor.js` – Generates lattice vectors, atom placements, and bond networks.
- `src/modeling/multiScaleModeler.js` – Aggregates metrics across atomic, mesoscopic, and continuum scales.
- `src/optimization/experimentOptimizer.js` – Iterative refinement of experiment parameters.
- `src/workflows` – Workflow definitions and scheduling logic.
- `public/` – Three.js front-end with SSE subscription for live state updates.

## Validation Checklist

- `npm run start` – Launches the visualization server and optional Docker smoke stream.
- Browser dashboard – Confirm atoms render, workflows update, and metrics refresh every tick.
- Environment overrides – Spot-check by exporting `PORT`, `TICK_INTERVAL_MS`, or `ENABLE_DOCKER_SMOKE=off` before starting.

Capture any manual findings (e.g., Docker socket availability, HTTPS deployments) in PR notes following the repository guidelines. For Materials Project validation of extended databases, see `qulab-infinite/README.md`.
