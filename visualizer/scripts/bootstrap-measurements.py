#!/usr/bin/env python3
"""Initialise the measurements schema and optional DuckDB mirror."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "qulab-infinite" / "src"
sys.path.append(str(PYTHON_SRC))

from qulab_infinite.database.engines import create_postgres_engine  # type: ignore  # noqa: E402
from qulab_infinite.database.schema import ensure_schema, sync_duckdb_mirror  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postgres-url",
        required=True,
        help="SQLAlchemy-compatible Postgres URL (e.g. postgresql+psycopg://user:pass@host:5432/db)",
    )
    parser.add_argument(
        "--duckdb-path",
        help="Optional DuckDB file path for materialising analytical mirrors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = create_postgres_engine(args.postgres_url)
    print("[info] Ensuring measurement schema exists...")
    ensure_schema(engine)

    if args.duckdb_path:
        duck_path = Path(args.duckdb_path)
        print(f"[info] Writing DuckDB mirror to {duck_path} ...")
        sync_duckdb_mirror(engine, duck_path)
    print("[info] Bootstrap complete.")


if __name__ == "__main__":
    main()
