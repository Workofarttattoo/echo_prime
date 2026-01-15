# Repository Guidelines

## Project Structure & Module Organization
Everything lives inside `visualizer/`. Runtime entry points sit in `scripts/` (e.g., `scripts/start.js` and `scripts/docker-smoke.js`), front-end code belongs in `src/`, and static Three.js assets stay in `public/`. Configuration defaults are tracked in `config/`, shared datasets in `data/`, and runtime artifacts or manual audit logs go under `logs/`. Keep new helpers beside their consumers—UI utilities in `src/`, CLI helpers next to the corresponding script, and future suites under `tests/` so tooling grows with the runtime.

## Build, Test, and Development Commands
Install dependencies first: `npm install`. Use `npm run start` for the local SSE server plus Docker telemetry smoke test (opt out with `ENABLE_DOCKER_SMOKE=off`). `node scripts/start.js` is the fastest dev loop when the npm wrapper is unnecessary. Run `npm run start:prod` to mirror HTTPS deployments—you must export `HTTPS_KEY`, `HTTPS_CERT`, and optional `HTTPS_CA`. Validate reproducible bundles with `npm run build`, and point smoke targets via `DOCKER_SOCKET=/var/run/docker.sock` or `DOCKER_CONTAINER_ID=<id>`.

## Coding Style & Naming Conventions
Author ES modules with explicit `import`/`export`, two-space indentation, double quotes, and trailing semicolons. Favor `camelCase` for identifiers, reserve `PascalCase` for React-like components, and create files in kebab-case (for example `stream-utils.js`). Emit telemetry with `[info]`, `[warn]`, or `[error]` prefixes, and colocate environment-aware knobs inside `config/`. When in doubt, follow the existing patterns in `src/index.js` and `config/default.js`.

## Testing Guidelines
No automated harness exists yet, so add exploratory specs beneath `tests/` (e.g., `tests/scripts/start.test.js`) and document coverage gaps in pull requests. Manual smoke steps: confirm Docker socket access, run `npm run start` to watch workflow ticks, execute `npm run start:prod` for HTTPS, and inspect `logs/` or `LOG_FILE_PATH` output. Avoid committing sensitive logs; reference them in PR descriptions instead.

## Commit & Pull Request Guidelines
Use Conventional Commits in the present tense (`feat: add audit mux`, `fix: guard missing cert`). Each PR should summarize intent, list validation commands that were run, attach relevant Docker or audit logs/screenshots, and link tracking issues. Call out any feature flags or environment variables you introduced so reviewers can reproduce the smoke run.

## Security & Configuration Tips
Ensure Docker Desktop or the daemon exposes `/var/run/docker.sock` with read access before launching smoke tests. Never check in certificates or secrets—pass them via env vars like `HTTPS_KEY` or `OLLAMA_HOST`. During failure drills, redirect output with `LOG_FILE_PATH=/tmp/gavel.log` or temporarily disable auditing by exporting `AUDIT_ENABLED=off` to keep the default logs clean.
