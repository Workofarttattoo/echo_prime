#!/usr/bin/env python3
"""Run a GHZ fidelity benchmark and push results through the ETL flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "qulab-infinite" / "src"
sys.path.append(str(PYTHON_SRC))

from qulab_infinite.benchmarks.ghz import build_measurement_record, run_ghz_benchmark  # noqa: E402
from qulab_infinite.etl.flows import run_measurement_etl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=3, help="Number of qubits in the GHZ circuit.")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of repeated simulations to estimate sigma."
    )
    parser.add_argument(
        "--dataset-id",
        default="ghz-fidelity-bench-v1",
        help="Identifier for the generated dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "quantum" / "ghz" / "ghz_fidelity_generated.json",
        help="Where to write the raw JSON dataset.",
    )
    parser.add_argument("--postgres-url", help="If supplied, the ETL flow will persist results to Postgres.")
    return parser.parse_args()


def write_dataset(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    else:
        existing = []
    existing.append(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)


def main() -> None:
    args = parse_args()
    result = run_ghz_benchmark(num_qubits=args.num_qubits, iterations=args.iterations)
    record = build_measurement_record(result, dataset_id=args.dataset_id, datum_index=0)

    print(f"[info] GHZ fidelity mean={result.fidelity:.6f} sigma={result.sigma:.6f}")
    write_dataset(args.output, record.raw_payload())
    print(f"[info] Wrote raw dataset entry to {args.output}")

    if args.postgres_url:
        inserted = run_measurement_etl(raw_path=str(args.output), postgres_url=args.postgres_url)
        print(f"[info] ETL flow inserted {inserted} new rows.")
    else:
        print("[warn] Postgres URL not supplied; skipping persistence.")


if __name__ == "__main__":
    main()
