# Qulab Infinite Data Backbone

This package bootstraps the Python toolchain that supports data ingest, calibration, and benchmarking for Qulab Infinite.

## Environments

The project is managed with Poetry. Install dependencies with:

```bash
poetry install
```

Activate the shell to run Prefect flows or calibration notebooks:

```bash
poetry shell
```

## Key Capabilities

- Enforced unit handling via `pint` and dataset-level vocabularies.
- Deterministic ETL flows orchestrated with Prefect and validated with Pydantic.
- Posterior inference pipelines using PyMC + ArviZ with artifact storage in DVC.
- Quantum benchmarks (e.g., GHZ fidelity) implemented with Qiskit and PennyLane.

## Golden Run Smoke Test

Generate a deterministic golden run artifact for demo validation:

```bash
cd visualizer/qulab-infinite
python scripts/golden-run-smoke.py --seed 1337
```

Artifacts are written to `visualizer/logs/qulab-golden-runs/` and include a payload hash plus provenance metadata.

## Repository Integration

Shared configuration (units, taxonomies, benchmark registry) lives under `visualizer/config/`. Prefect flows and utilities are exposed via `visualizer/scripts/` entry points for CLI integration.

Setup instructions live in `docs/environment-setup.md`.

### ETL Flow

- `prefect` flow entry point: `qulab_infinite.etl.run_measurement_etl`.
- CLI wrapper: `visualizer/scripts/bootstrap-measurements.py`.
- Raw datasets live under `visualizer/data/raw/`.

### GHZ Benchmark Prototype

- Generate samples: `visualizer/scripts/run-ghz-benchmark.py --num-qubits 3 --iterations 5`.
- The script writes raw JSON payloads and can optionally invoke the ETL flow using `--postgres-url`.

## Extended Materials Validation

Use the Material Project validator to vet large curated datasets before importing
them into the experimentation stack.

1. Drop the database files (any `*.db.json` or `.jsonl`) into `qulab-infinite/data/materials/`.
2. Install dependencies via Poetry if you have not already: `cd visualizer/qulab-infinite && poetry install`.
3. Run the validator from the qulab-infinite root:

   ```bash
   cd visualizer/qulab-infinite
   MP_API_KEY=your_key_here poetry run python scripts/validate-extended-materials.py
   ```

The CLI scans `data/materials/` by default, extracts every Materials Project ID,
and fetches authoritative reference values (density, band gap, formation energy,
etc.). Results are written to `logs/materials/mp-validation-report.json`, and a
non-zero exit code is returned when any property check fails. To validate an
external dataset without copying it into the repo, pass the path explicitly:

```bash
MP_API_KEY=... poetry run python scripts/validate-extended-materials.py \
  ../QuLabInfinite/materials_lab/data/extended_materials_db.json
```

Large JSON dumps are streamed via `ijson`, so the validator no longer loads
multi-gigabyte payloads into memory. If you need to spot-check or throttle API
usage, combine `--max-materials` with repeated runs, or lower the default rate
limit via `--rate-limit`.

### Bulk Materials Project Download

Pull a fresh copy of the Materials Project summary dataset as compressed JSONL:

```bash
cd visualizer/qulab-infinite
MP_API_KEY=your_key_here poetry run python scripts/download-mp-dataset.py \
  --output data/materials/mp-summary.jsonl.gz \
  --chunk-size 1000
```

Important flags:

- `--chunk-size` – number of records per API call (default 1000, MP caps at 2500).
- `--start-chunk` – resume from a later chunk if a prior run was interrupted.
- `--max-chunks` – stop after N chunks (useful for sampling or incremental syncs).
- `--rate-limit-hz` – throttle outbound requests (default 5 req/s).
- `--extra-property` – request additional MP fields (repeat the flag as needed).

Each run appends to the specified JSONL/JSONL.GZ file and writes a companion
manifest (`<output>.meta.json`) summarizing total chunks and materials captured.
Keep `MP_API_KEY` exported just like the validator before launching the download.
