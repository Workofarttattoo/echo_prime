# Environment Setup Playbook

Use this checklist the first time you bring the Qulab Infinite Python toolchain online.

## 1. Python Tooling

```bash
# Install Poetry (preferred) or UV on macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Add Poetry to PATH for the current shell
export PATH="$HOME/.local/bin:$PATH"

cd visualizer/qulab-infinite
poetry install
```

If your environment blocks outbound HTTP you can vendor the wheels from an
approved mirror and point Poetry at it via `POETRY_HTTP_BASIC_*` variables.

## 2. Postgres

1. Install Postgres (e.g., `brew install postgresql@16`).
2. Start the service: `pg_ctl -D /usr/local/var/postgresql@16 start`.
3. Create a database:

```bash
createdb qulab_infinite
```

4. Export the connection string so Prefect can see it:

```bash
export DATABASE_URL="postgresql+psycopg://$(whoami)@localhost:5432/qulab_infinite"
```

5. Bootstrap schema + DuckDB mirror (optional):

```bash
poetry run python ../scripts/bootstrap-measurements.py \
  --postgres-url "${DATABASE_URL}" \
  --duckdb-path ../../artifacts/duckdb/measurements.duckdb
```

## 3. Prefect ETL Verification

```bash
poetry run python ../scripts/run-ghz-benchmark.py \
  --num-qubits 3 \
  --iterations 5 \
  --postgres-url "${DATABASE_URL}"
```

On success you should see `[info] ETL flow inserted ...`.

## 4. Data Versioning

Install DVC (or LakeFS CLI) and initialise remotes:

```bash
pipx install dvc
cd visualizer
dvc init
dvc remote add -d qulab-local-storage ../../artifacts/dvc-storage
dvc add data/raw/quantum/ghz
dvc push
```

If you use LakeFS, mirror the same structure by mounting `visualizer/data/raw`
into a repository branch and storing the commit hash alongside the Prefect flow
run metadata.

## 5. Calibration Notebook Integration

After the ETL run, launch Jupyter using Poetry:

```bash
poetry run jupyter lab
```

Open `notebooks/ghz_calibration.py` (compatible with Jupytext) and execute the
cells to pull measurements from Postgres, run the PyMC posterior update, and
plot 5–95% predictive intervals for the GHZ fidelity benchmark.
